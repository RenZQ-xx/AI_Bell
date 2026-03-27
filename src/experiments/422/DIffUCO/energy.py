from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

if __package__ in (None, ""):
    from geometry import validate_subset, weighted_plane_statistics
else:
    from .geometry import validate_subset, weighted_plane_statistics


@dataclass
class EnergyConfig:
    min_cardinality: int = 26
    card_penalty: float = 25.0
    plane_weight: float = 1.0
    facet_dim_weight: float = 8.0
    rank_ratio_eps: float = 1e-6
    boundary_weight: float = 4.0
    support_weight: float = 1.0
    active_weight: float = 1.0
    inactive_weight: float = 0.35
    support_margin: float = 0.05
    active_probability_power: float = 1.0
    active_loss_type: str = "quadratic"
    active_huber_delta: float = 0.1
    plane_topk: int = 0
    active_topk: int = 0
    hard_plane_eps: float = 1e-6
    hard_support_tol: float = 1e-6


class GeometricHyperplaneEnergy(nn.Module):
    def __init__(self, points: torch.Tensor, config: EnergyConfig):
        super().__init__()
        self.register_buffer("points", points)
        self.config = config

    def _topk_filter(self, weights: torch.Tensor, topk: int) -> torch.Tensor:
        if topk <= 0 or topk >= weights.shape[-1]:
            return weights
        selected = torch.topk(weights, k=topk, dim=-1).indices
        filtered = torch.zeros_like(weights)
        filtered.scatter_(1, selected, weights.gather(1, selected))
        return filtered

    def _active_residual(self, signed_batch: torch.Tensor) -> torch.Tensor:
        squared = signed_batch.pow(2)
        if self.config.active_loss_type == "quadratic":
            return squared
        if self.config.active_loss_type == "huber":
            delta = max(float(self.config.active_huber_delta), 1e-6)
            absolute = signed_batch.abs()
            return torch.where(
                absolute <= delta,
                0.5 * squared,
                delta * (absolute - 0.5 * delta),
            )
        raise ValueError(f"Unsupported active_loss_type: {self.config.active_loss_type}")

    def _orientation_terms(
        self,
        probabilities: torch.Tensor,
        normal: torch.Tensor,
        orientation_sign: float,
    ) -> Dict[str, torch.Tensor]:
        oriented_normal = orientation_sign * normal
        projections = self.points @ oriented_normal.transpose(0, 1)
        offset = -projections.max(dim=0).values
        signed = projections + offset.unsqueeze(0)

        active_weights = probabilities.pow(self.config.active_probability_power)
        active_weights = self._topk_filter(active_weights, self.config.active_topk)
        active_norm = active_weights.sum(dim=-1).clamp_min(1e-6)
        inactive_weights = 1.0 - probabilities
        inactive_norm = inactive_weights.sum(dim=-1).clamp_min(1e-6)

        signed_batch = signed.transpose(0, 1)
        active_residual = self._active_residual(signed_batch)
        active_loss = (active_weights * active_residual).sum(dim=-1) / active_norm
        inactive_slack = torch.relu(self.config.support_margin + signed_batch)
        inactive_loss = (inactive_weights * inactive_slack.pow(2)).sum(dim=-1) / inactive_norm
        support_loss = torch.relu(signed.max(dim=0).values).pow(2)
        boundary_loss = (
            self.config.support_weight * support_loss
            + self.config.active_weight * active_loss
            + self.config.inactive_weight * inactive_loss
        )
        return {
            "boundary": boundary_loss,
            "support": support_loss,
            "active": active_loss,
            "inactive": inactive_loss,
            "normal": oriented_normal,
            "offset": offset,
            "signed": signed_batch,
        }

    def forward(self, probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        probabilities = probabilities.clamp(1e-5, 1.0 - 1e-5)
        plane_weights = self._topk_filter(probabilities, self.config.plane_topk)
        plane = weighted_plane_statistics(self.points, plane_weights)
        plane_term = plane.eigenvalues[:, 0]
        second_eigenvalue = plane.eigenvalues[:, 1]
        facet_dim_term = plane_term / (second_eigenvalue + self.config.rank_ratio_eps)

        positive_orientation = self._orientation_terms(probabilities, plane.normal, orientation_sign=1.0)
        negative_orientation = self._orientation_terms(probabilities, plane.normal, orientation_sign=-1.0)
        choose_positive = positive_orientation["boundary"] <= negative_orientation["boundary"]

        def pick(key: str) -> torch.Tensor:
            positive_value = positive_orientation[key]
            negative_value = negative_orientation[key]
            selector = choose_positive
            while selector.ndim < positive_value.ndim:
                selector = selector.unsqueeze(-1)
            return torch.where(selector, positive_value, negative_value)

        cardinality = probabilities.sum(dim=-1)
        deficit = torch.relu(self.config.min_cardinality - cardinality)
        # Keep the minimum-cardinality constraint, but avoid rewarding larger
        # supporting sets beyond the threshold. This makes the energy more
        # neutral between narrower and wider valid facets.
        card_term = self.config.card_penalty * deficit.pow(2)

        boundary_term = pick("boundary")
        total = (
            self.config.plane_weight * plane_term
            + self.config.facet_dim_weight * facet_dim_term
            + self.config.boundary_weight * boundary_term
            + card_term
        )
        return {
            "total": total,
            "plane": plane_term,
            "facet_dim": facet_dim_term,
            "second_eigenvalue": second_eigenvalue,
            "boundary": boundary_term,
            "support": pick("support"),
            "active": pick("active"),
            "inactive": pick("inactive"),
            "cardinality": cardinality,
            "card_loss": card_term,
            "normal": pick("normal"),
            "offset": pick("offset"),
        }


    def log_p_0(self, probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
        safe_temperature = max(float(temperature), 1e-6)
        return -self(probabilities)["total"] / safe_temperature

    def validate_hard_mask(self, mask: torch.Tensor) -> Dict[str, object]:
        indices = mask.nonzero(as_tuple=False).flatten().tolist()
        return validate_subset(
            self.points.detach().cpu().numpy(),
            indices,
            min_cardinality=self.config.min_cardinality,
            plane_eps=self.config.hard_plane_eps,
            support_tol=self.config.hard_support_tol,
        )

    def energy_of_hard_mask(self, mask: torch.Tensor) -> float:
        probabilities = mask.float().unsqueeze(0).to(self.points.device)
        return float(self(probabilities)["total"].item())
