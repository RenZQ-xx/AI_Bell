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
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(point_dim + 6, hidden_dim),
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

    def forward(
        self,
        point_features: torch.Tensor,
        seed_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes = seed_mask.shape
        points = point_features.unsqueeze(0).expand(batch_size, -1, -1)
        seed_centered = seed_mask - 0.5
        seed_density = seed_mask.mean(dim=-1, keepdim=True).expand(-1, num_nodes)
        node_inputs = torch.stack(
            [
                seed_mask,
                1.0 - seed_mask,
                seed_centered,
                seed_density,
                seed_mask * seed_density,
                seed_centered * seed_density,
            ],
            dim=-1,
        )
        hidden = self.node_encoder(torch.cat([points, node_inputs], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, edge_index=edge_index, edge_type=edge_type)

        global_hidden = hidden.mean(dim=1, keepdim=True).expand(-1, num_nodes, -1)
        logits = self.readout(torch.cat([hidden, global_hidden], dim=-1)).squeeze(-1)
        return logits
