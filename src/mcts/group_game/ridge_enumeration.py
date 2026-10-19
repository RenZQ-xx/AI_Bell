from __future__ import annotations

"""Bounded, resumable ridge enumeration using only a facet's vertices."""

from itertools import combinations
from math import comb

import numpy as np
from scipy.linalg import null_space

from .model import block_word


class RidgeEnumerator:
    def __init__(self, points: np.ndarray, indices: list[int], *, max_candidates: int):
        self.indices = indices
        self.examined = 0
        self.seen: set[int] = set()
        self.rejected = 0
        self.duplicates = 0
        retained_size = points.shape[1] - 1
        self.total = comb(len(indices), retained_size) if len(indices) >= retained_size else 0
        self.eligible = 0 < self.total <= max_candidates
        self.finished = not self.eligible
        if self.eligible:
            tight = points[indices]
            affine = np.column_stack((np.ones(len(indices)), tight))
            self.dependencies = null_space(affine.T, rcond=1e-10)
            self.candidates = combinations(range(len(indices)), len(indices) - retained_size)

    def advance(self, batch_size: int) -> int | None:
        if self.finished:
            return None
        for _ in range(batch_size):
            excluded = next(self.candidates)
            self.examined += 1
            self.finished = self.examined == self.total
            # A supporting slack vector must be orthogonal to every affine
            # dependency. A one-dimensional, one-signed kernel exposes a ridge.
            # Gale rows can vanish at pyramid apices. Relative-only tolerance
            # mistakes roundoff in an all-zero submatrix for a genuine rank.
            _u, singular, vt = np.linalg.svd(self.dependencies[list(excluded)].T, full_matrices=True)
            kernel = vt[np.count_nonzero(singular > 1e-9):].T
            retained = None
            if kernel.shape[1] == 1:
                slack = kernel[:, 0]
                if slack.sum() < 0:
                    slack = -slack
                if np.min(slack) >= -1e-8 and np.max(slack) > 1e-8:
                    removed = {excluded[j] for j in np.flatnonzero(slack > 1e-8)}
                    retained = block_word(tuple(v for j, v in enumerate(self.indices) if j not in removed))
            if retained is None:
                self.rejected += 1
            elif retained in self.seen:
                self.duplicates += 1
            else:
                self.seen.add(retained)
                return retained
            if self.finished:
                break
        return None

    def to_dict(self) -> dict[str, object]:
        return {"eligible": self.eligible, "total_candidates": self.total,
                "examined": self.examined, "finished": self.finished,
                "distinct_ridges": len(self.seen), "rejected": self.rejected,
                "duplicates": self.duplicates}
