from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class TransitionSample:
    source_index: int
    source_class_id: int
    source_example_id: int
    target_class_id: int
    target_index: int
    target_example_id: int
    source_mask: list[int]
    target_mask: list[int]
    hamming_to_target: int


class ClassConditionedTransitionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mask_dim: int,
        class_embed_dim: int = 32,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.source_class_embedding = nn.Embedding(num_classes + 1, class_embed_dim)
        self.target_class_embedding = nn.Embedding(num_classes + 1, class_embed_dim)
        input_dim = mask_dim + class_embed_dim * 2 + 2
        blocks: list[nn.Module] = []
        current = input_dim
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
        self.backbone = nn.Sequential(*blocks)
        self.mask_head = nn.Linear(current, mask_dim)
        self.cardinality_head = nn.Linear(current, 1)

    def forward(
        self,
        source_mask: torch.Tensor,
        source_class_id: torch.Tensor,
        target_class_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_class_embed = self.source_class_embedding(source_class_id)
        target_class_embed = self.target_class_embedding(target_class_id)
        stats = torch.stack(
            [
                source_mask.mean(dim=-1),
                source_mask.sum(dim=-1),
            ],
            dim=-1,
        )
        hidden = self.backbone(torch.cat([source_mask, source_class_embed, target_class_embed, stats], dim=-1))
        return self.mask_head(hidden), self.cardinality_head(hidden).squeeze(-1)


def build_class_index_map(class_ids: Iterable[int]) -> dict[int, int]:
    sorted_ids = sorted(set(int(class_id) for class_id in class_ids))
    return {class_id: idx + 1 for idx, class_id in enumerate(sorted_ids)}
