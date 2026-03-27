from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time_index: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        freq = torch.exp(
            torch.arange(half, device=time_index.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        phase = time_index.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if emb.shape[-1] < self.hidden_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return self.proj(emb)


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_types: int, dropout: float = 0.0):
        super().__init__()
        self.edge_embedding = nn.Embedding(edge_types, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(hidden_dim, 1)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, time_emb: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = hidden.shape
        source, target = edge_index
        source_hidden = hidden[:, source]
        target_hidden = hidden[:, target]
        source_time = time_emb[:, source]
        target_time = time_emb[:, target]
        edge_hidden = self.edge_embedding(edge_type).unsqueeze(0).expand(batch_size, -1, -1)

        query = self.query(target_hidden + target_time)
        key = self.key(source_hidden + source_time + edge_hidden)
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
        updated = self.update(torch.cat([hidden, aggregated, time_emb], dim=-1))
        return self.norm(hidden + self.dropout(updated))


class GeometricDiffUCOModel(nn.Module):
    def __init__(
        self,
        point_dim: int,
        edge_types: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(point_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(hidden_dim=hidden_dim, edge_types=edge_types, dropout=dropout) for _ in range(layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        point_features: torch.Tensor,
        state: torch.Tensor,
        time_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes = state.shape
        points = point_features.unsqueeze(0).expand(batch_size, -1, -1)
        state_features = torch.stack([state, 1.0 - state, state - 0.5], dim=-1)
        hidden = self.node_encoder(torch.cat([points, state_features], dim=-1))
        global_time = self.time_embedding(time_index)
        time_emb = global_time.unsqueeze(1).expand(-1, num_nodes, -1)

        for layer in self.layers:
            hidden = layer(hidden, time_emb, edge_index, edge_type)

        logits = self.readout(torch.cat([hidden, time_emb], dim=-1)).squeeze(-1)
        return logits
