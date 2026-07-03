from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryConfig:
    """Numerical thresholds for the clean CandidateFamily demo line."""

    rank_tol: float = 1e-5
    support_tol: float = 1e-6
    supportability_start_rank: int = 0
    supportability_direction_samples: int = 512
    supportability_seed: int = 20260430
    supportability_direction_bank: str = "key_dependent"
    supportability_constraint_verifier_enabled: bool = False
    supportability_constraint_verifier_min_rank: int = 18
    flat_capacity_method: str = "child_rank"
    min_terminal_cardinality: int = 26
    plane_eps: float = 1e-6
    facet_rank_eps: float = 1e-5
