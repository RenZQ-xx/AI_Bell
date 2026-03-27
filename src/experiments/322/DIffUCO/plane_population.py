from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class PlanePopulationConfig:
    elite_count: int = 4
    diversity_scale: float = 1.0
    plane_threshold: float = 0.08
    boundary_threshold: float = 6.0
    cardinality_margin: float = 1.0
    gate_sharpness: float = 25.0
    gate_penalty_scale: float = 50.0


class PlanePopulationObjective:
    def __init__(self, config: PlanePopulationConfig):
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

    def objective_terms(
        self,
        offset: torch.Tensor,
        normal: torch.Tensor,
        plane: torch.Tensor,
        boundary: torch.Tensor,
        cardinality: torch.Tensor,
        total_energy: torch.Tensor,
        min_cardinality: int,
    ) -> Dict[str, torch.Tensor]:
        batch_size = total_energy.shape[0]
        gate = self._quality_gate(
            plane=plane,
            boundary=boundary,
            cardinality=cardinality,
            min_cardinality=min_cardinality,
        )
        if batch_size == 0:
            zero = torch.zeros((), device=total_energy.device, dtype=total_energy.dtype)
            return {
                "objective": zero,
                "elite_energy": zero,
                "elite_diversity": zero,
                "gate": gate,
                "elite_mask": torch.zeros_like(total_energy),
            }

        effective_energy = total_energy + self.config.gate_penalty_scale * (1.0 - gate)
        elite_count = max(1, min(int(self.config.elite_count), batch_size))
        elite_indices = torch.topk(-effective_energy, k=elite_count, dim=0).indices
        elite_mask = torch.zeros_like(total_energy)
        elite_mask.scatter_(0, elite_indices, 1.0)

        elite_energy = total_energy.gather(0, elite_indices).mean()

        if elite_count <= 1:
            elite_diversity = torch.zeros((), device=total_energy.device, dtype=total_energy.dtype)
        else:
            embeddings = self._plane_embedding(offset=offset, normal=normal)
            elite_embeddings = embeddings.index_select(0, elite_indices)
            similarity = torch.abs(elite_embeddings @ elite_embeddings.transpose(0, 1)).clamp(0.0, 1.0)
            identity = torch.eye(elite_count, device=similarity.device, dtype=similarity.dtype)
            pairwise_distance = (1.0 - similarity) * (1.0 - identity)
            elite_diversity = pairwise_distance.sum() / max(elite_count * (elite_count - 1), 1)

        objective = elite_energy - self.config.diversity_scale * elite_diversity
        return {
            "objective": objective,
            "elite_energy": elite_energy,
            "elite_diversity": elite_diversity,
            "gate": gate,
            "elite_mask": elite_mask,
        }
