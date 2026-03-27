from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch

if __package__ in (None, ""):
    from symmetry_graph import build_state_group_permutations
else:
    from .symmetry_graph import build_state_group_permutations


@dataclass
class OrbitMemoryConfig:
    per_class_capacity: int = 8
    activation_count_threshold: int = 5
    max_active_classes: int = 4
    similarity_dedup_threshold: float = 0.92
    sigma: float = 0.18
    gamma: float = 0.5
    tau_orbit: float = 0.05
    tau_prototype: float = 0.05
    jaccard_eps: float = 1e-6


@dataclass
class OrbitClassMemory:
    count: int = 0
    prototypes: List[torch.Tensor] = field(default_factory=list)


class OrbitMemoryBank:
    def __init__(self, config: OrbitMemoryConfig):
        self.config = config
        self.classes: Dict[str, OrbitClassMemory] = {}
        permutations = build_state_group_permutations()
        self.state_permutations = torch.tensor(permutations, dtype=torch.long)

    def _soft_jaccard(self, probabilities: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        intersection = (probabilities.unsqueeze(1) * masks.unsqueeze(0)).sum(dim=-1)
        union = probabilities.sum(dim=-1, keepdim=True) + masks.sum(dim=-1).unsqueeze(0) - intersection
        similarity = intersection / union.clamp_min(self.config.jaccard_eps)
        return similarity

    def _prototype_similarity(self, candidate: torch.Tensor, prototype: torch.Tensor) -> float:
        similarity = self._soft_jaccard(
            candidate.unsqueeze(0),
            prototype.unsqueeze(0),
        )
        return float(similarity.squeeze().item())

    def update(self, analyzed_items: Sequence[Dict[str, object] | None], masks: torch.Tensor) -> None:
        detached_masks = masks.detach().to(dtype=torch.float32, device="cpu")
        for item, mask in zip(analyzed_items, detached_masks):
            if item is None or item.get("tier") != "exact_match":
                continue
            canonical_key = item.get("canonical_key")
            if canonical_key is None:
                continue
            key = str(canonical_key)
            entry = self.classes.setdefault(key, OrbitClassMemory())
            entry.count += 1

            should_add = True
            for existing in entry.prototypes:
                if self._prototype_similarity(mask, existing) >= self.config.similarity_dedup_threshold:
                    should_add = False
                    break
            if not should_add:
                continue

            if len(entry.prototypes) >= self.config.per_class_capacity:
                entry.prototypes.pop(0)
            entry.prototypes.append(mask.clone())

    def active_classes(self) -> List[tuple[str, OrbitClassMemory]]:
        active = [
            (key, entry)
            for key, entry in self.classes.items()
            if entry.count >= self.config.activation_count_threshold and entry.prototypes
        ]
        active.sort(key=lambda item: item[1].count, reverse=True)
        return active[: self.config.max_active_classes]

    def _orbit_distance(self, probabilities: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        transformed = prototypes[:, self.state_permutations]
        batch_size = probabilities.shape[0]
        num_prototypes, num_permutations, num_nodes = transformed.shape
        flat_transformed = transformed.reshape(num_prototypes * num_permutations, num_nodes).to(probabilities.device)
        similarities = self._soft_jaccard(probabilities, flat_transformed)
        distances = 1.0 - similarities
        distances = distances.view(batch_size, num_prototypes, num_permutations)
        return -self.config.tau_orbit * torch.logsumexp(-distances / self.config.tau_orbit, dim=-1)

    def repulsion(self, probabilities: torch.Tensor) -> torch.Tensor:
        active = self.active_classes()
        if not active:
            return torch.zeros(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype)

        repulsion = torch.zeros(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype)
        sigma_sq = max(self.config.sigma * self.config.sigma, 1e-8)
        for _, entry in active:
            prototypes = torch.stack(entry.prototypes, dim=0).to(probabilities.device)
            orbit_distances = self._orbit_distance(probabilities, prototypes)
            class_distance = -self.config.tau_prototype * torch.logsumexp(
                -orbit_distances / self.config.tau_prototype,
                dim=-1,
            )
            weight = self.config.gamma * float(torch.log1p(torch.tensor(float(entry.count))).item())
            repulsion = repulsion + weight * torch.exp(-(class_distance.pow(2)) / sigma_sq)
        return repulsion
