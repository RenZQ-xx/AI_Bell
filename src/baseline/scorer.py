from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .bell322 import generate_bell322_points
from .facet_validator import FacetLabel, FacetValidator
from .geometry import affine_rank
from .orbit_blocks import BlockKey, add_block, unselected_blocks
from .support_masks import block_key_to_vertex_indices, block_key_to_vertex_mask
from .supportability import SupportabilityAnalyzer, SupportabilityMetrics


@dataclass(frozen=True)
class ScorerConfig:
    """Small set of baseline scoring weights."""

    rank_tol: float = 1e-5
    support_tol: float = 1e-6
    phase_b_start_rank: int = 22
    supportability_start_rank: int = 0
    supportability_direction_samples: int = 512
    supportability_seed: int = 20260430
    supportability_direction_bank: str = "key_dependent"
    supportability_gate_weight: float = 6.0
    supportability_gate_closer_weight: float = 1.0
    nonpositive_rank_gain_penalty: float = 8.0
    rank_gain_weight_scale: float = 1.0
    flat_penalty_weight_scale: float = 1.0
    flat_capacity_method: str = "child_rank"
    rank24_entrance_weight: float = 1.0
    rank24_entrance_exists_weight: float = 4.0
    rank24_class44_entrance_weight: float = 1.0
    rank24_invalid_entrance_weight: float = 1.0
    rare_terminal_score: float = 100.0
    target_terminal_score: float = 10.0
    valid_terminal_score: float = 1.0
    invalid_terminal_score: float = -10.0
    invalid_rank_terminal_score: float = -100.0
    class44_terminal_score: float = -5.0
    boundary_invalid_base_score: float = -20.0
    boundary_invalid_closer_weight: float = 2.0
    terminal_scoring_mode: str = "static"
    dynamic_new_class_score: float = 100.0
    dynamic_known_class_score: float = 10.0
    dynamic_frequent_class_score: float = -5.0
    dynamic_frequent_class_threshold: int = 16
    cache_heavy_max_entries: int = 2000
    cache_light_max_entries: int = 12000


@dataclass(frozen=True)
class ExpansionScore:
    """Score payload for one add-block action."""

    action: int
    key: BlockKey
    score: float
    phase: str
    old_rank: int
    new_rank: int
    rank_gain: int
    flat_capacity: int
    supportability: SupportabilityMetrics | None
    terminal: FacetLabel | None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "key": list(self.key),
            "score": self.score,
            "phase": self.phase,
            "old_rank": self.old_rank,
            "new_rank": self.new_rank,
            "rank_gain": self.rank_gain,
            "flat_capacity": self.flat_capacity,
            "supportability": None if self.supportability is None else self.supportability.to_dict(),
            "terminal": None if self.terminal is None else self.terminal.to_dict(),
        }


class ExpansionScorer:
    """Baseline scorer for zero-start add-orbit-block search."""

    def __init__(
        self,
        *,
        blocks: Sequence[Sequence[int]],
        points: np.ndarray | None = None,
        validator: FacetValidator | None = None,
        config: ScorerConfig | None = None,
        target_classes: set[int] | None = None,
        rare_target_classes: set[int] | None = None,
    ) -> None:
        self.blocks = [tuple(int(vertex) for vertex in block) for block in blocks]
        self.points = generate_bell322_points() if points is None else np.asarray(points)
        self.config = ScorerConfig() if config is None else config
        self.validator = validator or FacetValidator(
            points=self.points,
            support_tol=self.config.support_tol,
            facet_rank_eps=self.config.rank_tol,
        )
        self.supportability = SupportabilityAnalyzer(
            points=self.points,
            rank_tol=self.config.rank_tol,
            support_tol=self.config.support_tol,
        )
        self.target_classes = set(range(1, 47)) if target_classes is None else set(target_classes)
        self.rare_target_classes = set() if rare_target_classes is None else set(rare_target_classes)
        self.discovered_label_counts: Counter[str] = Counter()
        self.rank_cache: dict[BlockKey, int] = {}
        self.vertices_cache: dict[BlockKey, list[int]] = {}
        self.mask_cache: dict[BlockKey, np.ndarray] = {}
        self.flat_cache: dict[BlockKey, int] = {}
        self.child_flat_p50_cache: dict[BlockKey, float] = {}
        self.boundary_cache: dict[BlockKey, dict[str, float]] = {}
        self.terminal_cache: dict[BlockKey, FacetLabel] = {}
        self.rank24_entrance_cache: dict[BlockKey, dict[str, int]] = {}

    def _cache_put(self, cache: dict, key: BlockKey, value: object, *, heavy: bool) -> None:
        """Insert into cache with a bounded size to avoid unbounded memory growth."""
        cache[key] = value
        max_entries = self.config.cache_heavy_max_entries if heavy else self.config.cache_light_max_entries
        if max_entries > 0 and len(cache) > max_entries:
            cache.pop(next(iter(cache)))

    def vertex_indices(self, key: BlockKey) -> list[int]:
        """Expand selected blocks into deterministic vertex ids."""
        if key not in self.vertices_cache:
            self._cache_put(
                self.vertices_cache,
                key,
                block_key_to_vertex_indices(key, self.blocks),
                heavy=True,
            )
        return self.vertices_cache[key]

    def vertex_mask(self, key: BlockKey) -> np.ndarray:
        """Return the 64-bit hard support mask for a block key."""
        if key not in self.mask_cache:
            self._cache_put(
                self.mask_cache,
                key,
                block_key_to_vertex_mask(key, self.blocks, vertex_count=len(self.points)),
                heavy=True,
            )
        return self.mask_cache[key]

    def affine_rank(self, key: BlockKey) -> int:
        """Affine rank of the selected support."""
        if key not in self.rank_cache:
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                rank = 0
            else:
                rank = affine_rank(self.points[indices], rank_eps=self.config.rank_tol)
            self._cache_put(self.rank_cache, key, rank, heavy=False)
        return self.rank_cache[key]

    def fit_boundary_metrics(self, key: BlockKey) -> dict[str, float]:
        """Fast SVD fitted-boundary metrics used by the legacy terminal score."""
        if key not in self.boundary_cache:
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                self._cache_put(
                    self.boundary_cache,
                    key,
                    {
                        "closer_side": 32.0,
                        "positive": 32.0,
                        "negative": 32.0,
                        "supporting_shift": 1e6,
                    },
                    heavy=False,
                )
                return self.boundary_cache[key]
            selected = self.points[indices]
            centroid = selected.mean(axis=0)
            centered = selected - centroid
            _u, _singulars, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
            offset = -float(np.dot(normal, centroid))
            signed = self.points @ normal + offset
            positive = int(np.sum(signed > self.config.support_tol))
            negative = int(np.sum(signed < -self.config.support_tol))
            supporting_shift = min(max(float(signed.max()), 0.0) ** 2, max(float(-signed.min()), 0.0) ** 2)
            self._cache_put(
                self.boundary_cache,
                key,
                {
                "closer_side": float(min(positive, negative)),
                "positive": float(positive),
                "negative": float(negative),
                "supporting_shift": float(supporting_shift),
                },
                heavy=False,
            )
        return self.boundary_cache[key]

    def terminal_label(self, key: BlockKey) -> FacetLabel:
        """Terminal validation for a candidate that already reached rank >= 25."""
        if key not in self.terminal_cache:
            self._cache_put(self.terminal_cache, key, self.validator.validate_mask(self.vertex_mask(key)), heavy=False)
        return self.terminal_cache[key]

    def terminal_score(self, key: BlockKey) -> float:
        """Reward exact target terminal facets using the restored static rule."""
        config = self.config
        boundary = self.fit_boundary_metrics(key)
        if boundary["closer_side"] > 0:
            return config.boundary_invalid_base_score - config.boundary_invalid_closer_weight * boundary["closer_side"]
        terminal = self.terminal_label(key)
        if not terminal.is_exact:
            return config.invalid_terminal_score
        class_id = exact_class_id(terminal.label)
        if config.terminal_scoring_mode == "dynamic":
            count = self.discovered_label_counts[terminal.label]
            if count <= 0:
                return config.dynamic_new_class_score
            if count >= config.dynamic_frequent_class_threshold:
                return config.dynamic_frequent_class_score
            return config.dynamic_known_class_score
        if class_id in self.rare_target_classes:
            return config.rare_terminal_score
        if class_id == 44:
            return config.class44_terminal_score
        if class_id in self.target_classes:
            return config.target_terminal_score
        return config.valid_terminal_score

    def supportability_metrics(self, key: BlockKey) -> SupportabilityMetrics:
        """Low-rank supportability existence test."""
        return self.supportability.metrics(
            self.vertex_indices(key),
            direction_samples=self.config.supportability_direction_samples,
            seed=self.config.supportability_seed,
            direction_bank=self.config.supportability_direction_bank,
            cache_key=key,
        )

    def flat_capacity(self, key: BlockKey) -> int:
        """Count unselected blocks that do not increase the affine rank."""
        if key not in self.flat_cache:
            if self.config.flat_capacity_method == "child_rank":
                rank = self.affine_rank(key)
                self._cache_put(
                    self.flat_cache,
                    key,
                    sum(1 for action in unselected_blocks(key) if self.affine_rank(add_block(key, action)) == rank),
                    heavy=False,
                )
                return self.flat_cache[key]

            rank = self.affine_rank(key)
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                self._cache_put(
                    self.flat_cache,
                    key,
                    sum(1 for action in unselected_blocks(key) if self.affine_rank(add_block(key, action)) == rank),
                    heavy=False,
                )
                return self.flat_cache[key]

            selected = self.points[indices]
            anchor = selected[0]
            centered = selected - anchor
            _, singulars, vh = np.linalg.svd(centered, full_matrices=False)
            hull_rank = int(np.sum(singulars > self.config.rank_tol))
            basis = np.zeros((self.points.shape[1], 0), dtype=float) if hull_rank <= 0 else vh[:hull_rank].T

            count = 0
            for action in unselected_blocks(key):
                block_points = self.points[list(self.blocks[action])]
                delta = block_points - anchor
                residual = delta - (delta @ basis) @ basis.T if basis.shape[1] > 0 else delta
                if float(np.max(np.linalg.norm(residual, axis=1))) <= 2.0 * self.config.rank_tol:
                    count += 1
            self._cache_put(self.flat_cache, key, count, heavy=False)
        return self.flat_cache[key]

    def child_flat_p50(self, key: BlockKey) -> float:
        """Median flat capacity among one-step children that increase rank."""
        if key not in self.child_flat_p50_cache:
            rank = self.affine_rank(key)
            values: list[float] = []
            for action in unselected_blocks(key):
                child = add_block(key, action)
                child_rank = self.affine_rank(child)
                if child_rank <= rank:
                    continue
                values.append(0.0 if child_rank >= 25 else float(self.flat_capacity(child)))
            self._cache_put(
                self.child_flat_p50_cache,
                key,
                0.0 if not values else float(np.percentile(np.asarray(values), 50)),
                heavy=False,
            )
        return self.child_flat_p50_cache[key]

    def rank24_entrance_metrics(self, key: BlockKey) -> dict[str, int]:
        """Count one-step terminal exits from a rank-24 prefix."""
        if key not in self.rank24_entrance_cache:
            rare = 0
            class44 = 0
            invalid = 0
            other_valid = 0
            for action in unselected_blocks(key):
                candidate = add_block(key, action)
                if self.affine_rank(candidate) < 25:
                    continue
                label = self.terminal_label(candidate)
                if not label.is_exact:
                    invalid += 1
                    continue
                class_id = exact_class_id(label.label)
                if class_id in self.rare_target_classes:
                    rare += 1
                elif class_id == 44:
                    class44 += 1
                else:
                    other_valid += 1
            self._cache_put(
                self.rank24_entrance_cache,
                key,
                {
                    "rare": rare,
                    "class44": class44,
                    "invalid": invalid,
                    "other_valid": other_valid,
                },
                heavy=False,
            )
        return self.rank24_entrance_cache[key]

    def phase_weights(self, rank: int) -> tuple[float, float, float]:
        """Weights for rank gain, supportability, and flatness by search phase."""
        if rank < self.config.phase_b_start_rank:
            return 2.0, 0.2, 0.3
        if rank <= 23:
            return 1.5, 0.5, 1.0
        return 1.0, 1.5, 1.2

    def score_action(self, key: BlockKey, action: int) -> ExpansionScore:
        """Score the candidate produced by adding one block."""
        candidate = add_block(key, action)
        old_rank = self.affine_rank(key)
        new_rank = self.affine_rank(candidate)
        rank_gain = new_rank - old_rank

        terminal: FacetLabel | None = None
        support: SupportabilityMetrics | None = None
        flat = 0 if new_rank >= 25 else self.flat_capacity(candidate)

        if new_rank >= 25:
            score = self.terminal_score(candidate)
            terminal = self.terminal_label(candidate)
            phase = "C"
        else:
            w_rank, _w_support, w_flat = self.phase_weights(new_rank)
            score = (
                self.config.rank_gain_weight_scale * w_rank * float(rank_gain)
                - self.config.flat_penalty_weight_scale * w_flat * math.log1p(float(flat))
            )
            if rank_gain <= 0:
                score -= self.config.nonpositive_rank_gain_penalty
            if new_rank >= self.config.supportability_start_rank:
                support = self.supportability_metrics(candidate)
                if support.supporting_shift > self.config.support_tol**2 or support.closer_side > 0.0:
                    gate_penalty = math.log1p(support.supporting_shift) + (
                        self.config.supportability_gate_closer_weight * support.closer_side
                    )
                    score -= self.config.supportability_gate_weight * gate_penalty
            if new_rank == 24 and self.config.rank24_entrance_weight != 0.0:
                entrance = self.rank24_entrance_metrics(candidate)
                score += self.config.rank24_entrance_weight * (
                    self.config.rank24_entrance_exists_weight * (1.0 if entrance["rare"] > 0 else 0.0)
                    + math.log1p(float(entrance["rare"]))
                    - self.config.rank24_class44_entrance_weight * math.log1p(float(entrance["class44"]))
                    - self.config.rank24_invalid_entrance_weight * math.log1p(float(entrance["invalid"]))
                )
            phase = "A" if new_rank < self.config.phase_b_start_rank else ("B1" if new_rank <= 23 else "B2")

        return ExpansionScore(
            action=int(action),
            key=candidate,
            score=float(score),
            phase=phase,
            old_rank=int(old_rank),
            new_rank=int(new_rank),
            rank_gain=int(rank_gain),
            flat_capacity=int(flat),
            supportability=support,
            terminal=terminal,
        )


def exact_class_id(label: str) -> int | None:
    if label.startswith("exact:class"):
        return int(label[len("exact:class"):])
    if label.startswith("exact:"):
        return int(label.split(":", 1)[1])
    return None
