from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class PlaneSetDPPConfig:
    reward_scale: float = 1.0
    plane_threshold: float = 0.08
    boundary_threshold: float = 6.0
    cardinality_margin: float = 1.0
    gate_sharpness: float = 25.0
    similarity_temperature: float = 8.0
    diagonal_jitter: float = 1e-3


class PlaneSetDPPObjective:
    def __init__(self, config: PlaneSetDPPConfig):
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

    def dpp_bonus(
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
        zeros = torch.zeros(batch_size, device=embeddings.device, dtype=embeddings.dtype)
        if batch_size <= 1:
            return {
                "bonus": zeros,
                "gate": gate,
                "mean_similarity": zeros,
                "logdet": torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype).squeeze(0),
            }

        similarity = torch.abs(embeddings @ embeddings.transpose(0, 1)).clamp(0.0, 1.0)
        temperature = max(float(self.config.similarity_temperature), 1e-6)
        base_kernel = torch.exp(-temperature * (1.0 - similarity))
        quality = gate.sqrt().unsqueeze(-1)
        kernel = quality * base_kernel * quality.transpose(0, 1)
        jitter = max(float(self.config.diagonal_jitter), 1e-8)
        kernel = kernel + torch.eye(batch_size, device=kernel.device, dtype=kernel.dtype) * jitter

        sign, logabsdet = torch.linalg.slogdet(kernel)
        safe_logdet = torch.where(sign > 0.0, logabsdet, torch.zeros_like(logabsdet))
        baseline_logdet = float(batch_size) * torch.log(
            torch.tensor(jitter, device=kernel.device, dtype=kernel.dtype)
        )
        diversity_gain = (safe_logdet - baseline_logdet).clamp_min(0.0)

        weighted_similarity = similarity * gate.unsqueeze(0) * gate.unsqueeze(1)
        mean_similarity = weighted_similarity.sum(dim=-1) / gate.sum().clamp_min(1e-6)
        normalized_bonus = self.config.reward_scale * diversity_gain / float(batch_size)
        bonus = gate * normalized_bonus
        return {
            "bonus": bonus,
            "gate": gate,
            "mean_similarity": mean_similarity,
            "logdet": safe_logdet,
            "diversity_gain": diversity_gain,
        }
