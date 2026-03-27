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


class GraphEscapeScorer(nn.Module):
    def __init__(
        self,
        point_dim: int,
        edge_types: int,
        hidden_dim: int = 192,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(point_dim + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim=hidden_dim, edge_types=edge_types, dropout=dropout) for _ in range(layers)]
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().clamp_min(0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = weights / denom
        return torch.einsum("bn,bnh->bh", normalized, hidden)

    def forward(
        self,
        point_features: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        energy_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_nodes = source_mask.shape
        points = point_features.unsqueeze(0).expand(batch_size, -1, -1)
        diff = candidate_mask - source_mask
        absdiff = diff.abs()
        source_density = source_mask.mean(dim=-1, keepdim=True).expand(-1, num_nodes)
        candidate_density = candidate_mask.mean(dim=-1, keepdim=True).expand(-1, num_nodes)
        node_features = torch.stack(
            [
                source_mask,
                candidate_mask,
                1.0 - source_mask,
                1.0 - candidate_mask,
                diff,
                absdiff,
                source_density,
                candidate_density,
            ],
            dim=-1,
        )
        hidden = self.node_encoder(torch.cat([points, node_features], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, edge_index=edge_index, edge_type=edge_type)

        source_summary = self.masked_pool(hidden, source_mask)
        candidate_summary = self.masked_pool(hidden, candidate_mask)
        diff_summary = candidate_summary - source_summary
        global_summary = hidden.mean(dim=1)
        stats = torch.stack(
            [
                absdiff.mean(dim=-1),
                candidate_mask.sum(dim=-1),
                source_mask.sum(dim=-1),
                energy_value if energy_value is not None else source_mask.new_zeros(batch_size),
            ],
            dim=-1,
        )
        score_input = torch.cat([global_summary, source_summary, candidate_summary, diff_summary, stats], dim=-1)
        return self.score_head(score_input).squeeze(-1)


class SetEscapeScorer(nn.Module):
    def __init__(
        self,
        point_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = point_dim + 8
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_dim
        self.node_encoder = nn.Sequential(*blocks)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().clamp_min(0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = weights / denom
        return torch.einsum("bn,bnh->bh", normalized, hidden)

    def forward(
        self,
        point_features: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        energy_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_nodes = source_mask.shape
        points = point_features.unsqueeze(0).expand(batch_size, -1, -1)
        diff = candidate_mask - source_mask
        absdiff = diff.abs()
        source_density = source_mask.mean(dim=-1, keepdim=True).expand(-1, num_nodes)
        candidate_density = candidate_mask.mean(dim=-1, keepdim=True).expand(-1, num_nodes)
        node_features = torch.stack(
            [
                source_mask,
                candidate_mask,
                1.0 - source_mask,
                1.0 - candidate_mask,
                diff,
                absdiff,
                source_density,
                candidate_density,
            ],
            dim=-1,
        )
        hidden = self.node_encoder(torch.cat([points, node_features], dim=-1))
        source_summary = self.masked_pool(hidden, source_mask)
        candidate_summary = self.masked_pool(hidden, candidate_mask)
        diff_summary = candidate_summary - source_summary
        global_summary = hidden.mean(dim=1)
        stats = torch.stack(
            [
                absdiff.mean(dim=-1),
                candidate_mask.sum(dim=-1),
                source_mask.sum(dim=-1),
                energy_value if energy_value is not None else source_mask.new_zeros(batch_size),
            ],
            dim=-1,
        )
        score_input = torch.cat([global_summary, source_summary, candidate_summary, diff_summary, stats], dim=-1)
        return self.score_head(score_input).squeeze(-1)
