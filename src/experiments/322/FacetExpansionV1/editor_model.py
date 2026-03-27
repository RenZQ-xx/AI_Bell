from __future__ import annotations

import math

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_types: int, dropout: float = 0.0):
        super().__init__()
        self.edge_embedding = nn.Embedding(edge_types, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(hidden_dim, 1)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = hidden.shape
        source, target = edge_index
        source_hidden = hidden[:, source]
        target_hidden = hidden[:, target]
        edge_hidden = self.edge_embedding(edge_type).unsqueeze(0).expand(batch_size, -1, -1)

        query = self.query(target_hidden)
        key = self.key(source_hidden + edge_hidden)
        value = self.value(source_hidden + edge_hidden)
        logits = (query * key).sum(dim=-1) / math.sqrt(hidden_dim)
        logits = logits + self.edge_bias(edge_hidden).squeeze(-1)

        weights = torch.zeros_like(logits)
        for node_id in range(num_nodes):
            mask = target == node_id
            if mask.any():
                weights[:, mask] = torch.softmax(logits[:, mask], dim=-1)

        messages = value * weights.unsqueeze(-1)
        aggregated = torch.zeros_like(hidden)
        scatter_index = target.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
        aggregated.scatter_add_(1, scatter_index, messages)
        updated = self.update(torch.cat([hidden, aggregated], dim=-1))
        return self.norm(hidden + self.dropout(updated))


class ConditionalEditModel(nn.Module):
    def __init__(
        self,
        point_dim: int,
        edge_types: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.0,
        history_dim: int | None = None,
        history_layers: int = 2,
        max_history: int = 6,
    ):
        super().__init__()
        history_dim = hidden_dim if history_dim is None else history_dim
        self.max_history = max_history
        self.history_dim = history_dim
        if history_dim > 0:
            self.history_step_encoder = nn.Sequential(
                nn.Linear(4, history_dim),
                nn.SiLU(),
                nn.Linear(history_dim, history_dim),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=history_dim,
                nhead=4,
                dim_feedforward=max(history_dim * 2, 4),
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=history_layers)
            self.history_positional = nn.Embedding(max_history + 1, history_dim)
        else:
            self.history_step_encoder = None
            self.history_encoder = None
            self.history_positional = None
        self.node_encoder = nn.Sequential(
            nn.Linear(point_dim + 10 + history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim=hidden_dim, edge_types=edge_types, dropout=dropout) for _ in range(layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.novelty_head = nn.Sequential(
            nn.Linear(hidden_dim + history_dim + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().clamp_min(0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = weights / denom
        return torch.einsum("bn,bnh->bh", normalized, hidden)

    def encode_backbone(
        self,
        point_features: torch.Tensor,
        seed_mask: torch.Tensor,
        archive_context: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        trajectory_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes = seed_mask.shape
        points = point_features.unsqueeze(0).expand(batch_size, -1, -1)
        seed_centered = seed_mask - 0.5
        seed_density = seed_mask.mean(dim=-1, keepdim=True).expand(-1, num_nodes)
        archive_centered = archive_context - 0.5
        archive_density = archive_context.mean(dim=-1, keepdim=True).expand(-1, num_nodes)
        novelty_hint = torch.abs(seed_mask - archive_context)
        node_inputs = torch.stack(
            [
                seed_mask,
                1.0 - seed_mask,
                seed_centered,
                seed_density,
                seed_mask * seed_density,
                seed_centered * seed_density,
                archive_context,
                archive_centered,
                archive_density,
                novelty_hint,
            ],
            dim=-1,
        )
        history_embedding = self.encode_trajectory(seed_mask, archive_context, trajectory_masks)
        if self.history_dim > 0:
            history_context = history_embedding.unsqueeze(1).expand(-1, num_nodes, -1)
        else:
            history_context = node_inputs.new_zeros(batch_size, num_nodes, 0)
        hidden = self.node_encoder(torch.cat([points, node_inputs, history_context], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, edge_index=edge_index, edge_type=edge_type)
        return hidden, history_embedding

    def extract_candidate_embeddings(
        self,
        point_features: torch.Tensor,
        seed_mask: torch.Tensor,
        archive_context: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        candidate_mask: torch.Tensor,
        trajectory_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden, history_embedding = self.encode_backbone(
            point_features=point_features,
            seed_mask=seed_mask,
            archive_context=archive_context,
            edge_index=edge_index,
            edge_type=edge_type,
            trajectory_masks=trajectory_masks,
        )
        global_summary = hidden.mean(dim=1)
        candidate_summary = self.masked_pool(hidden, candidate_mask)
        seed_summary = self.masked_pool(hidden, seed_mask)
        archive_summary = self.masked_pool(hidden, archive_context)
        return {
            "history_embedding": history_embedding,
            "global_summary": global_summary,
            "candidate_summary": candidate_summary,
            "seed_summary": seed_summary,
            "archive_summary": archive_summary,
        }

    def forward(
        self,
        point_features: torch.Tensor,
        seed_mask: torch.Tensor,
        archive_context: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        trajectory_masks: torch.Tensor | None = None,
        novelty_candidate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, num_nodes = seed_mask.shape
        hidden, history_embedding = self.encode_backbone(
            point_features=point_features,
            seed_mask=seed_mask,
            archive_context=archive_context,
            edge_index=edge_index,
            edge_type=edge_type,
            trajectory_masks=trajectory_masks,
        )
        global_hidden = hidden.mean(dim=1, keepdim=True).expand(-1, num_nodes, -1)
        logits = self.readout(torch.cat([hidden, global_hidden], dim=-1)).squeeze(-1)
        edit_prob = torch.sigmoid(logits)
        novelty_mask = edit_prob if novelty_candidate_mask is None else novelty_candidate_mask
        edit_summary = torch.stack(
            [
                novelty_mask.mean(dim=-1),
                novelty_mask.std(dim=-1),
                torch.abs(novelty_mask - seed_mask).mean(dim=-1),
                torch.abs(novelty_mask - archive_context).mean(dim=-1),
                (novelty_mask * seed_mask).mean(dim=-1),
                (novelty_mask * archive_context).mean(dim=-1),
                (novelty_mask - archive_context).abs().max(dim=-1).values,
                (novelty_mask - seed_mask).abs().max(dim=-1).values,
            ],
            dim=-1,
        )
        global_summary = hidden.mean(dim=1)
        novelty_inputs = torch.cat([history_embedding, global_summary, edit_summary], dim=-1)
        novelty_logits = self.novelty_head(novelty_inputs).squeeze(-1)
        return logits, novelty_logits

    def encode_trajectory(
        self,
        seed_mask: torch.Tensor,
        archive_context: torch.Tensor,
        trajectory_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_nodes = seed_mask.shape
        if trajectory_masks is None:
            trajectory_masks = seed_mask.unsqueeze(1)

        if self.history_dim <= 0:
            return seed_mask.new_zeros(batch_size, 0)

        history_len = trajectory_masks.shape[1]
        if history_len > self.max_history:
            trajectory_masks = trajectory_masks[:, -self.max_history :]
            history_len = trajectory_masks.shape[1]

        history_centered = trajectory_masks - 0.5
        history_density = trajectory_masks.mean(dim=-1, keepdim=True).expand(-1, -1, num_nodes)
        similarity_to_seed = 1.0 - torch.abs(trajectory_masks - seed_mask.unsqueeze(1))
        step_features = torch.stack(
            [
                trajectory_masks.mean(dim=-1),
                history_centered.mean(dim=-1),
                history_density[..., 0],
                similarity_to_seed.mean(dim=-1),
            ],
            dim=-1,
        )
        position_ids = torch.arange(history_len, device=seed_mask.device).unsqueeze(0).expand(batch_size, -1)
        encoded_steps = self.history_step_encoder(step_features) + self.history_positional(position_ids)
        encoded_steps = self.history_encoder(encoded_steps)

        archive_summary = archive_context.mean(dim=-1, keepdim=True)
        recency_weights = torch.linspace(0.5, 1.0, history_len, device=seed_mask.device).view(1, history_len, 1)
        pooled = (encoded_steps * recency_weights).sum(dim=1) / recency_weights.sum(dim=1).clamp_min(1e-6)
        return pooled + archive_summary.expand(-1, self.history_dim)
