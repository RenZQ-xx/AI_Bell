from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class PlaneSetCoverageConfig:
    novelty_scale: float = 1.0
    plane_threshold: float = 0.08
    boundary_threshold: float = 6.0
    cardinality_margin: float = 1.0
    gate_sharpness: float = 25.0
    neighbor_count: int = 1


class PlaneSetCoverageObjective:
    def __init__(self, config: PlaneSetCoverageConfig):
        self.config = config

    def _plane_embedding(self, offset: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        vector = torch.cat([normal, offset.unsqueeze(-1)], dim=-1)
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _quality_gate(
        self,
        plane: torch.Tensor,
        boundary: torch.Tensor,
        cardinality: torch.Tensor,
        min_cardinality: int,
    ) -> torch.Tensor:
        sharpness = max(float(self.config.gate_sharpness), 1e-6)
        plane_gate = torch.sigmoid((self.config.plane_threshold - plane) * sharpness)
        boundary_gate = torch.sigmoid((self.config.boundary_threshold - boundary) * sharpness)
        cardinality_target = float(min_cardinality) + float(self.config.cardinality_margin)
        cardinality_gate = torch.sigmoid((cardinality - cardinality_target) * sharpness)
        return plane_gate * boundary_gate * cardinality_gate

    def coverage_bonus(
        self,
        offset: torch.Tensor,
        normal: torch.Tensor,
        plane: torch.Tensor,
        boundary: torch.Tensor,
        cardinality: torch.Tensor,
        min_cardinality: int,
    ) -> Dict[str, torch.Tensor]:
        embeddings = self._plane_embedding(offset=offset, normal=normal)
        gate = self._quality_gate(
            plane=plane,
            boundary=boundary,
            cardinality=cardinality,
            min_cardinality=min_cardinality,
        )
        batch_size = embeddings.shape[0]
        if batch_size <= 1:
            zeros = torch.zeros(batch_size, device=embeddings.device, dtype=embeddings.dtype)
            return {
                "bonus": zeros,
                "gate": gate,
                "max_similarity": zeros,
            }

        similarity = torch.abs(embeddings @ embeddings.transpose(0, 1)).clamp(0.0, 1.0)
        similarity.fill_diagonal_(0.0)
        gated_similarity = similarity * gate.unsqueeze(0)
        neighbor_count = max(1, min(int(self.config.neighbor_count), batch_size - 1))
        top_similarities, _ = torch.topk(gated_similarity, k=neighbor_count, dim=-1)
        max_similarity = top_similarities.mean(dim=-1)
        novelty = (1.0 - max_similarity).clamp_min(0.0)
        bonus = self.config.novelty_scale * gate * novelty
        return {
            "bonus": bonus,
            "gate": gate,
            "max_similarity": max_similarity,
        }
