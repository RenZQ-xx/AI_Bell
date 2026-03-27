from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch

if __package__ in (None, ""):
    from facet_reference import normalize_row, orbit_members
else:
    from .facet_reference import normalize_row, orbit_members


@dataclass
class PlaneOrbitMemoryConfig:
    activation_count_threshold: int = 5
    max_active_classes: int = 4
    gamma: float = 0.5
    sigma: float = 0.12
    tau_orbit: float = 0.03
    vector_eps: float = 1e-6


@dataclass
class PlaneOrbitClassMemory:
    count: int = 0
    orbit_vectors: torch.Tensor | None = None


class PlaneOrbitMemoryBank:
    def __init__(self, config: PlaneOrbitMemoryConfig):
        self.config = config
        self.classes: Dict[str, PlaneOrbitClassMemory] = {}

    def _row_to_unit_vector(self, row: Sequence[int]) -> torch.Tensor:
        vector = torch.tensor(row, dtype=torch.float32)
        return vector / vector.norm().clamp_min(self.config.vector_eps)

    def _orbit_vectors(self, canonical_row: Sequence[int]) -> torch.Tensor:
        normalized = normalize_row(tuple(int(value) for value in canonical_row))
        members = orbit_members(normalized)
        vectors = torch.stack([self._row_to_unit_vector(member) for member in members], dim=0)
        return vectors

    def update(self, analyzed_items: Sequence[Dict[str, object] | None]) -> None:
        for item in analyzed_items:
            if item is None or item.get("tier") != "exact_match":
                continue
            canonical_key = item.get("canonical_key")
            canonical_row = item.get("canonical_integer_row")
            if canonical_key is None or canonical_row is None:
                continue
            key = str(canonical_key)
            entry = self.classes.get(key)
            if entry is None:
                entry = PlaneOrbitClassMemory(
                    count=0,
                    orbit_vectors=self._orbit_vectors(canonical_row),
                )
                self.classes[key] = entry
            entry.count += 1

    def active_classes(self) -> List[tuple[str, PlaneOrbitClassMemory]]:
        active = [
            (key, entry)
            for key, entry in self.classes.items()
            if entry.count >= self.config.activation_count_threshold and entry.orbit_vectors is not None
        ]
        active.sort(key=lambda item: item[1].count, reverse=True)
        return active[: self.config.max_active_classes]

    def _soft_plane_vector(self, offset: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        vector = torch.cat([offset.unsqueeze(-1), normal], dim=-1)
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(self.config.vector_eps)

    def repulsion(self, offset: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        active = self.active_classes()
        if not active:
            return torch.zeros(offset.shape[0], device=offset.device, dtype=offset.dtype)

        plane_vectors = self._soft_plane_vector(offset, normal)
        sigma_sq = max(self.config.sigma * self.config.sigma, 1e-8)
        repulsion = torch.zeros(offset.shape[0], device=offset.device, dtype=offset.dtype)
        for _, entry in active:
            orbit_vectors = entry.orbit_vectors.to(offset.device)
            similarities = torch.matmul(plane_vectors, orbit_vectors.transpose(0, 1)).abs()
            best_similarity = self.config.tau_orbit * torch.logsumexp(similarities / self.config.tau_orbit, dim=-1)
            class_distance = (1.0 - best_similarity).clamp_min(0.0)
            weight = self.config.gamma * float(torch.log1p(torch.tensor(float(entry.count))).item())
            repulsion = repulsion + weight * torch.exp(-(class_distance.pow(2)) / sigma_sq)
        return repulsion
