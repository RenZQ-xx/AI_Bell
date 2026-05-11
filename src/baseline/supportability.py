from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from .bell322 import generate_bell322_points


@dataclass(frozen=True)
class SupportabilityMetrics:
    """Low-rank test for whether a prefix can lie in a supporting hyperplane."""

    closer_side: float
    positive: float
    negative: float
    supporting_shift: float
    null_dim: float

    @property
    def is_supportable(self) -> bool:
        return self.supporting_shift <= 1e-12 and self.closer_side == 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "closer_side": self.closer_side,
            "positive": self.positive,
            "negative": self.negative,
            "supporting_shift": self.supporting_shift,
            "null_dim": self.null_dim,
        }


class SupportabilityAnalyzer:
    """Sample normals containing a selected affine hull.

    This is the clean baseline version of the old
    `supportability_boundary_metrics`.  At rank below 25, a selected prefix does
    not determine a unique hyperplane normal.  Instead of fitting one arbitrary
    SVD normal, we search directions in the normal space of the prefix and ask
    whether any such direction can support the full 64-vertex point cloud.
    """

    def __init__(
        self,
        *,
        points: np.ndarray | None = None,
        rank_tol: float = 1e-5,
        support_tol: float = 1e-6,
    ) -> None:
        self.points = generate_bell322_points() if points is None else np.asarray(points)
        self.rank_tol = float(rank_tol)
        self.support_tol = float(support_tol)
        self._intrinsic_points: np.ndarray | None = None
        self._intrinsic_rank: int | None = None
        self._metrics_cache: dict[tuple[tuple[int, ...], int, int, str, Hashable | None], SupportabilityMetrics] = {}
        self._direction_bank_cache: dict[tuple[int, int, int], np.ndarray] = {}

    @property
    def intrinsic_rank(self) -> int:
        self.intrinsic_coordinates()
        assert self._intrinsic_rank is not None
        return self._intrinsic_rank

    def intrinsic_coordinates(self) -> np.ndarray:
        """Project the 64 vertices to their intrinsic affine span."""
        if self._intrinsic_points is None:
            centered = self.points - self.points.mean(axis=0, keepdims=True)
            _, singulars, vh = np.linalg.svd(centered, full_matrices=False)
            rank = int(np.sum(singulars > self.rank_tol))
            self._intrinsic_rank = rank
            self._intrinsic_points = centered @ vh[:rank].T
        return self._intrinsic_points

    def metrics(
        self,
        selected_indices: Sequence[int],
        *,
        direction_samples: int = 512,
        seed: int = 20260430,
        direction_bank: str = "global_cached",
        cache_key: Hashable | None = None,
    ) -> SupportabilityMetrics:
        """Return supportability metrics for selected vertex indices."""
        indices = tuple(sorted(set(int(index) for index in selected_indices)))
        sample_count = max(1, int(direction_samples))
        bank_mode = str(direction_bank)
        key = (indices, sample_count, int(seed), bank_mode, cache_key)
        if key in self._metrics_cache:
            return self._metrics_cache[key]

        coords = self.intrinsic_coordinates()
        if len(indices) <= 1:
            out = SupportabilityMetrics(
                closer_side=32.0,
                positive=32.0,
                negative=32.0,
                supporting_shift=1e6,
                null_dim=float(self._intrinsic_rank or coords.shape[1]),
            )
            self._metrics_cache[key] = out
            return out

        selected = coords[list(indices)]
        centroid = selected.mean(axis=0)
        centered = selected - centroid
        _, singulars, vh = np.linalg.svd(centered, full_matrices=True)
        rank = int(np.sum(singulars > self.rank_tol))
        null_basis = vh[rank:].T
        null_dim = int(null_basis.shape[1])
        if null_dim <= 0:
            out = SupportabilityMetrics(
                closer_side=32.0,
                positive=32.0,
                negative=32.0,
                supporting_shift=1e6,
                null_dim=0.0,
            )
            self._metrics_cache[key] = out
            return out

        projected = (coords - centroid) @ null_basis
        directions = self._candidate_directions(
            projected=projected,
            null_dim=null_dim,
            sample_count=sample_count,
            seed=int(seed),
            direction_bank=bank_mode,
            cache_key=cache_key if cache_key is not None else indices,
        )
        signed = projected @ directions.T
        positive_shift = np.maximum(np.max(signed, axis=0), 0.0)
        negative_shift = np.maximum(np.max(-signed, axis=0), 0.0)
        shifts = np.minimum(positive_shift, negative_shift) ** 2
        best_index = int(np.argmin(shifts))
        best_signed = signed[:, best_index]
        positive = int(np.sum(best_signed > self.support_tol))
        negative = int(np.sum(best_signed < -self.support_tol))
        out = SupportabilityMetrics(
            closer_side=float(min(positive, negative)),
            positive=float(positive),
            negative=float(negative),
            supporting_shift=float(shifts[best_index]),
            null_dim=float(null_dim),
        )
        self._metrics_cache[key] = out
        return out

    def _candidate_directions(
        self,
        *,
        projected: np.ndarray,
        null_dim: int,
        sample_count: int,
        seed: int,
        direction_bank: str,
        cache_key: Hashable,
    ) -> np.ndarray:
        if null_dim == 1:
            return np.asarray([[1.0], [-1.0]], dtype=float)

        random_count = max(sample_count, 64)
        if direction_bank == "global_cached":
            bank_key = (random_count, null_dim, seed)
            if bank_key not in self._direction_bank_cache:
                rng = np.random.default_rng(seed + 9176 * null_dim)
                self._direction_bank_cache[bank_key] = rng.normal(size=(random_count, null_dim))
            directions = self._direction_bank_cache[bank_key]
        else:
            if isinstance(cache_key, tuple):
                if all(int(value) in (0, 1) for value in cache_key):
                    stable_code = sum(
                        (index + 17) * (int(value) + 1)
                        for index, value in enumerate(cache_key)
                        if int(value) == 1
                    )
                else:
                    stable_code = sum((index + 17) * (int(value) + 1) for index, value in enumerate(cache_key))
            else:
                stable_code = sum((index + 17) * (int(value) + 1) for index, value in enumerate(indices))
            rng = np.random.default_rng(seed + 1009 * stable_code + 9176 * null_dim)
            directions = rng.normal(size=(random_count, null_dim))

        basis = np.eye(null_dim, dtype=float)
        nonzero_projected = projected[np.linalg.norm(projected, axis=1) > self.rank_tol]
        if len(nonzero_projected) > 0:
            point_dirs = nonzero_projected / np.maximum(
                np.linalg.norm(nonzero_projected, axis=1, keepdims=True),
                1e-12,
            )
            directions = np.vstack([directions, basis, -basis, point_dirs, -point_dirs])
        else:
            directions = np.vstack([directions, basis, -basis])
        return directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
