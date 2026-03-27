from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class PlaneDensityConfig:
    capacity: int = 128
    dedup_similarity_threshold: float = 0.995
    sigma: float = 0.12
    novelty_scale: float = 1.0
    plane_threshold: float = 0.08
    boundary_threshold: float = 6.0
    cardinality_margin: float = 1.0
    gate_sharpness: float = 25.0


class PlaneDensityBank:
    def __init__(self, config: PlaneDensityConfig):
        self.config = config
        self._vectors: List[torch.Tensor] = []

    def _unit_plane_vector(self, offset: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        vector = torch.cat([offset.unsqueeze(-1), normal], dim=-1)
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _density(self, vectors: torch.Tensor) -> torch.Tensor:
        if not self._vectors:
            return torch.zeros(vectors.shape[0], device=vectors.device, dtype=vectors.dtype)
        memory = torch.stack(self._vectors, dim=0).to(vectors.device)
        similarities = torch.matmul(vectors, memory.transpose(0, 1)).abs()
        distances = 1.0 - similarities
        sigma_sq = max(self.config.sigma * self.config.sigma, 1e-8)
        kernels = torch.exp(-(distances.pow(2)) / sigma_sq)
        return kernels.max(dim=-1).values

    def quality_gate(
        self,
        plane: torch.Tensor,
        boundary: torch.Tensor,
        cardinality: torch.Tensor,
        min_cardinality: int,
    ) -> torch.Tensor:
        sharpness = self.config.gate_sharpness
        plane_gate = torch.sigmoid(sharpness * (self.config.plane_threshold - plane))
        boundary_gate = torch.sigmoid(sharpness * (self.config.boundary_threshold - boundary))
        cardinality_target = float(min_cardinality) + self.config.cardinality_margin
        cardinality_gate = torch.sigmoid(sharpness * 0.25 * (cardinality - cardinality_target))
        return plane_gate * boundary_gate * cardinality_gate

    def novelty_bonus(
        self,
        offset: torch.Tensor,
        normal: torch.Tensor,
        plane: torch.Tensor,
        boundary: torch.Tensor,
        cardinality: torch.Tensor,
        min_cardinality: int,
    ) -> Dict[str, torch.Tensor]:
        vectors = self._unit_plane_vector(offset, normal)
        density = self._density(vectors)
        gate = self.quality_gate(plane=plane, boundary=boundary, cardinality=cardinality, min_cardinality=min_cardinality)
        bonus = self.config.novelty_scale * gate * (1.0 - density)
        return {
            "bonus": bonus,
            "density": density,
            "gate": gate,
        }

    def update(self, analyzed_items: Sequence[Dict[str, object] | None]) -> None:
        for item in analyzed_items:
            if item is None:
                continue
            normal = item.get("normal")
            offset = item.get("offset")
            if normal is None or offset is None:
                continue
            normal_tensor = torch.tensor(normal, dtype=torch.float32)
            offset_tensor = torch.tensor([float(offset)], dtype=torch.float32)
            vector = torch.cat([offset_tensor, normal_tensor], dim=0)
            vector = vector / vector.norm().clamp_min(1e-6)

            should_add = True
            if self._vectors:
                memory = torch.stack(self._vectors, dim=0)
                similarity = torch.matmul(memory, vector).abs().max().item()
                if similarity >= self.config.dedup_similarity_threshold:
                    should_add = False
            if not should_add:
                continue
            if len(self._vectors) >= self.config.capacity:
                self._vectors.pop(0)
            self._vectors.append(vector)

    def size(self) -> int:
        return len(self._vectors)
