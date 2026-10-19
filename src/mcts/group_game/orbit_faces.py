from __future__ import annotations

"""Stream orbit-union deletions from an observed facet, without class labels."""

from collections import Counter, deque
from itertools import combinations
from math import comb
import hashlib

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from baseline.geometry import affine_rank
from .model import block_word


class OrbitFaceEnumerator:
    def __init__(self, points, source: int, patterns, *, min_rank: int,
                 max_candidates: int, max_removed_orbits: int = 3, seed: int = 0):
        if max_candidates < 0 or max_removed_orbits <= 0:
            raise ValueError("invalid orbit face budget")
        self.points = points
        self.source = source
        self.indices = [i for i in range(64) if source >> i & 1]
        self.min_rank = min_rank
        self.max_candidates = max_candidates
        self.max_removed_orbits = max_removed_orbits
        self.examined = 0
        self.total = 0
        self.stats = Counter()
        self.streams = deque()
        self.rank_cache = {}
        self.emitted = []
        self.dependencies = null_space(
            np.column_stack((np.ones(len(self.indices)), points[self.indices])).T,
            rcond=1e-10,
        )
        self.blocks = {}
        for pattern in patterns:
            inside = tuple(block_word(b) for b in pattern.blocks if source & block_word(b))
            if any(b & ~source for b in inside):
                raise ValueError("orbit partition does not preserve the source facet")
            digest = hashlib.sha256(repr((seed, source, pattern.pattern_id)).encode("ascii")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
            inside = tuple(inside[i] for i in rng.permutation(len(inside)))
            self.blocks[pattern.pattern_id] = inside
            for count in range(1, min(max_removed_orbits, len(inside)) + 1):
                self.total += comb(len(inside), count)
                self.streams.append((pattern, combinations(inside, count)))

    @property
    def finished(self):
        return self.examined >= min(self.total, self.max_candidates)

    def advance(self, batch_size: int):
        for _ in range(batch_size):
            if self.finished:
                return None
            while self.streams:
                pattern, stream = self.streams.popleft()
                try:
                    removed_blocks = next(stream)
                except StopIteration:
                    continue
                self.streams.append((pattern, stream))
                break
            else:
                raise RuntimeError("orbit streams exhausted before their candidate count")
            self.examined += 1
            removed = sum(removed_blocks)
            retained = self.source & ~removed
            if retained not in self.rank_cache:
                selected = [i for i in self.indices if retained >> i & 1]
                self.rank_cache[retained] = affine_rank(self.points[selected]) if selected else -1
            rank = self.rank_cache[retained]
            if rank < self.min_rank or rank > 24:
                self.stats["rank_low" if rank < self.min_rank else "rank_high"] += 1
                continue
            # A face has a slack vector in the affine column space, zero on R
            # and strictly positive on every removed orbit. Gale dependencies
            # test this in at most max_removed_orbits variables.
            incidence = np.array([[(word >> i) & 1 for word in removed_blocks]
                                  for i in self.indices], dtype=float)
            constraints = self.dependencies.T @ incidence
            _u, singular, vt = np.linalg.svd(constraints, full_matrices=True)
            constraint_rank = np.count_nonzero(singular > 1e-9)
            kernel = vt[constraint_rank:].T
            valid = False
            if kernel.shape[1] == 1:
                slack = kernel[:, 0]
                valid = bool(np.all(slack > 1e-8) or np.all(slack < -1e-8))
            elif kernel.shape[1] > 1:
                self.stats["support_lp_attempts"] += 1
                check = linprog(np.zeros(len(removed_blocks)),
                                A_eq=vt[:constraint_rank], b_eq=np.zeros(constraint_rank),
                                bounds=[(1, None)] * len(removed_blocks), method="highs")
                valid = bool(check.success)
            if not valid:
                self.stats["not_supporting"] += 1
                continue
            self.stats["accepted"] += 1
            self.emitted.append({"retained_word_hex": f"0x{retained:016x}",
                                 "pattern_id": pattern.pattern_id, "rank": rank,
                                 "removed_orbit_count": len(removed_blocks)})
            return retained, pattern, rank
        return None

    def to_dict(self):
        return {"source_word_hex": f"0x{self.source:016x}",
                "max_candidates": self.max_candidates, "total_candidates": self.total,
                "max_removed_orbits": self.max_removed_orbits, "examined": self.examined,
                "finished": self.finished, "exhaustive": self.examined == self.total,
                "statistics": dict(self.stats), "emitted": self.emitted,
                "pattern_blocks": {p: [f"0x{b:016x}" for b in blocks]
                                   for p, blocks in self.blocks.items()}}
