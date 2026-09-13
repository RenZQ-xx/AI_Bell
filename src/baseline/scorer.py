from __future__ import annotations

import math
import weakref
from collections import Counter
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Mapping, Sequence

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
    flat_capacity_method: str = "affine_hull"
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


@dataclass(frozen=True)
class ExpansionStructure:
    """Reward-independent geometry for one add-block action."""

    action: int
    key: BlockKey
    phase: str
    old_rank: int
    new_rank: int
    rank_gain: int
    flat_capacity: int
    supportability: SupportabilityMetrics | None
    terminal: FacetLabel | None
    base_score: float | None


@dataclass
class SharedTerminalValidationCache:
    """Facet labels shared across scorers by their 64-bit support mask."""

    labels: dict[int, FacetLabel] = field(default_factory=dict)
    max_entries: int = 25000
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def get(self, support_key: int) -> FacetLabel | None:
        with self._lock:
            label = self.labels.get(int(support_key))
            if label is None:
                self.misses += 1
            else:
                self.hits += 1
            return label

    def put(self, support_key: int, label: FacetLabel) -> None:
        with self._lock:
            self.labels[int(support_key)] = label
            if self.max_entries > 0 and len(self.labels) > self.max_entries:
                self.labels.pop(next(iter(self.labels)))
                self.evictions += 1


@dataclass
class ScorerStructureCache:
    """Reward-independent scorer caches for one ordered block partition."""

    rank: dict[BlockKey, int] = field(default_factory=dict)
    vertices: dict[BlockKey, list[int]] = field(default_factory=dict)
    masks: dict[BlockKey, np.ndarray] = field(default_factory=dict)
    flat: dict[BlockKey, int] = field(default_factory=dict)
    child_flat_p50: dict[BlockKey, float] = field(default_factory=dict)
    boundary: dict[BlockKey, dict[str, float]] = field(default_factory=dict)
    terminal: dict[BlockKey, FacetLabel] = field(default_factory=dict)
    support_keys: dict[BlockKey, int] = field(default_factory=dict)
    actions: dict[tuple[BlockKey, int], ExpansionStructure] = field(default_factory=dict)
    rank24_entrances: dict[BlockKey, tuple[dict[int, int], int]] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)


@dataclass
class SharedScorerStructureCache:
    """Share geometry for identical partitions without sharing MCTS values."""

    partitions: dict[
        tuple[tuple[tuple[int, ...], ...], ScorerConfig],
        ScorerStructureCache,
    ] = field(default_factory=dict)
    max_partitions: int = 4
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    _active_partitions: weakref.WeakValueDictionary = field(
        default_factory=weakref.WeakValueDictionary,
        repr=False,
    )
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def for_partition(
        self,
        blocks: Sequence[Sequence[int]],
        config: ScorerConfig,
    ) -> ScorerStructureCache:
        ordered_partition = tuple(tuple(int(vertex) for vertex in block) for block in blocks)
        key = (ordered_partition, config)
        with self._lock:
            shared = self._active_partitions.get(key)
            if shared is None:
                self.misses += 1
                shared = ScorerStructureCache()
                self._active_partitions[key] = shared
            else:
                self.hits += 1
            self.partitions.pop(key, None)
            self.partitions[key] = shared
            if self.max_partitions > 0 and len(self.partitions) > self.max_partitions:
                self.partitions.pop(next(iter(self.partitions)))
                self.evictions += 1
            return shared


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
        terminal_validation_cache: SharedTerminalValidationCache | None = None,
        structure_cache: SharedScorerStructureCache | None = None,
    ) -> None:
        self.blocks = [tuple(int(vertex) for vertex in block) for block in blocks]
        self.block_support_words = tuple(
            self._block_support_word(block)
            for block in self.blocks
        )
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
        self.shared_terminal_validation_cache = (
            SharedTerminalValidationCache()
            if terminal_validation_cache is None
            else terminal_validation_cache
        )
        self.discovered_label_counts: Counter[str] = Counter()
        shared_structure = (
            ScorerStructureCache()
            if structure_cache is None
            else structure_cache.for_partition(self.blocks, self.config)
        )
        self._shared_structure = shared_structure
        self._structure_lock = shared_structure._lock
        self.rank_cache = shared_structure.rank
        self.vertices_cache = shared_structure.vertices
        self.mask_cache = shared_structure.masks
        self.flat_cache = shared_structure.flat
        self.child_flat_p50_cache = shared_structure.child_flat_p50
        self.boundary_cache = shared_structure.boundary
        self.terminal_cache = shared_structure.terminal
        self.support_key_cache = shared_structure.support_keys
        self.action_structure_cache = shared_structure.actions
        self.rank24_entrance_cache = shared_structure.rank24_entrances

    @staticmethod
    def _block_support_word(block: Sequence[int]) -> int:
        word = 0
        for vertex in block:
            word |= 1 << int(vertex)
        return word

    _CACHE_MISSING = object()

    def _cache_get(self, cache: dict, key: object) -> object:
        lock = getattr(self, "_structure_lock", None)
        if lock is None:
            return cache.get(key, self._CACHE_MISSING)
        with lock:
            return cache.get(key, self._CACHE_MISSING)

    def _cache_put(
        self,
        cache: dict,
        key: object,
        value: object,
        *,
        heavy: bool,
    ) -> object:
        """Insert into cache with a bounded size to avoid unbounded memory growth."""
        lock = getattr(self, "_structure_lock", None)

        def put() -> object:
            cached = cache.get(key, self._CACHE_MISSING)
            if cached is not self._CACHE_MISSING:
                return cached
            cache[key] = value
            max_entries = (
                self.config.cache_heavy_max_entries
                if heavy
                else self.config.cache_light_max_entries
            )
            if max_entries > 0 and len(cache) > max_entries:
                oldest = next(iter(cache), None)
                if oldest is not None:
                    cache.pop(oldest, None)
            return value

        if lock is None:
            return put()
        with lock:
            return put()

    def vertex_indices(self, key: BlockKey) -> list[int]:
        """Expand selected blocks into deterministic vertex ids."""
        cached = self._cache_get(self.vertices_cache, key)
        if cached is not self._CACHE_MISSING:
            return cached  # type: ignore[return-value]
        return self._cache_put(
            self.vertices_cache,
            key,
            block_key_to_vertex_indices(key, self.blocks),
            heavy=True,
        )  # type: ignore[return-value]

    def vertex_mask(self, key: BlockKey) -> np.ndarray:
        """Return the 64-bit hard support mask for a block key."""
        cached = self._cache_get(self.mask_cache, key)
        if cached is not self._CACHE_MISSING:
            return cached  # type: ignore[return-value]
        return self._cache_put(
            self.mask_cache,
            key,
            block_key_to_vertex_mask(
                key,
                self.blocks,
                vertex_count=len(self.points),
            ),
            heavy=True,
        )  # type: ignore[return-value]

    def affine_rank(self, key: BlockKey) -> int:
        """Affine rank of the selected support."""
        cached = self._cache_get(self.rank_cache, key)
        if cached is not self._CACHE_MISSING:
            return int(cached)
        indices = self.vertex_indices(key)
        rank = (
            0
            if len(indices) <= 1
            else affine_rank(self.points[indices], rank_eps=self.config.rank_tol)
        )
        return int(self._cache_put(self.rank_cache, key, rank, heavy=False))

    def fit_boundary_metrics(self, key: BlockKey) -> dict[str, float]:
        """Fast SVD fitted-boundary metrics used by the legacy terminal score."""
        cached = self._cache_get(self.boundary_cache, key)
        if cached is not self._CACHE_MISSING:
            return cached  # type: ignore[return-value]
        indices = self.vertex_indices(key)
        if len(indices) <= 1:
            value = {
                "closer_side": 32.0,
                "positive": 32.0,
                "negative": 32.0,
                "supporting_shift": 1e6,
            }
        else:
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
            supporting_shift = min(
                max(float(signed.max()), 0.0) ** 2,
                max(float(-signed.min()), 0.0) ** 2,
            )
            value = {
                "closer_side": float(min(positive, negative)),
                "positive": float(positive),
                "negative": float(negative),
                "supporting_shift": float(supporting_shift),
            }
        return self._cache_put(
                self.boundary_cache,
                key,
                value,
                heavy=False,
            )  # type: ignore[return-value]

    def terminal_label(self, key: BlockKey) -> FacetLabel:
        """Terminal validation for a candidate that already reached rank >= 25."""
        cached = self._cache_get(self.terminal_cache, key)
        if cached is not self._CACHE_MISSING:
            return cached  # type: ignore[return-value]
        support_key = self.support_key(key)
        label = self.shared_terminal_validation_cache.get(support_key)
        if label is None:
            label = self.validator.validate_mask(self.vertex_mask(key))
            self.shared_terminal_validation_cache.put(support_key, label)
        return self._cache_put(
            self.terminal_cache,
            key,
            label,
            heavy=False,
        )  # type: ignore[return-value]

    def support_key(self, key: BlockKey) -> int:
        cached = self._cache_get(self.support_key_cache, key)
        if cached is not self._CACHE_MISSING:
            return int(cached)
        support_key = 0
        for block_index, selected in enumerate(key):
            if int(selected):
                support_key |= self.block_support_words[block_index]
        return int(
            self._cache_put(
                self.support_key_cache,
                key,
                support_key,
                heavy=False,
            )
        )

    def terminal_score(
        self,
        key: BlockKey,
        *,
        rare_target_classes: set[int] | None = None,
        discovered_label_counts: Mapping[str, int] | None = None,
    ) -> float:
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
            counts = self.discovered_label_counts if discovered_label_counts is None else discovered_label_counts
            count = int(counts.get(terminal.label, 0))
            if count <= 0:
                return config.dynamic_new_class_score
            if count >= config.dynamic_frequent_class_threshold:
                return config.dynamic_frequent_class_score
            return config.dynamic_known_class_score
        active_rare_targets = self.rare_target_classes if rare_target_classes is None else rare_target_classes
        if class_id in active_rare_targets:
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
        cached = self._cache_get(self.flat_cache, key)
        if cached is not self._CACHE_MISSING:
            return int(cached)
        rank = self.affine_rank(key)
        indices = self.vertex_indices(key)
        if self.config.flat_capacity_method == "child_rank" or len(indices) <= 1:
            count = sum(
                1
                for action in unselected_blocks(key)
                if self.affine_rank(add_block(key, action)) == rank
            )
        else:
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
        return int(self._cache_put(self.flat_cache, key, count, heavy=False))

    def child_flat_p50(self, key: BlockKey) -> float:
        """Median flat capacity among one-step children that increase rank."""
        cached = self._cache_get(self.child_flat_p50_cache, key)
        if cached is not self._CACHE_MISSING:
            return float(cached)
        rank = self.affine_rank(key)
        values: list[float] = []
        for action in unselected_blocks(key):
            child = add_block(key, action)
            child_rank = self.affine_rank(child)
            if child_rank <= rank:
                continue
            values.append(
                0.0 if child_rank >= 25 else float(self.flat_capacity(child))
            )
        value = 0.0 if not values else float(np.percentile(np.asarray(values), 50))
        return float(
            self._cache_put(
                self.child_flat_p50_cache,
                key,
                value,
                heavy=False,
            )
        )

    def rank24_entrance_metrics(self, key: BlockKey) -> dict[str, int]:
        """Count one-step terminal exits from a rank-24 prefix."""
        exact_class_counts, invalid = self.rank24_entrance_class_counts(key)
        rare = sum(
            count
            for class_id, count in exact_class_counts.items()
            if class_id in self.rare_target_classes
        )
        class44 = 0 if 44 in self.rare_target_classes else int(exact_class_counts.get(44, 0))
        return {
            "rare": int(rare),
            "class44": class44,
            "invalid": int(invalid),
            "other_valid": int(sum(exact_class_counts.values()) - rare - class44),
        }

    def rank24_entrance_class_counts(
        self,
        key: BlockKey,
    ) -> tuple[dict[int, int], int]:
        """Return validated class counts for every one-step rank-24 exit."""

        cached = self._cache_get(self.rank24_entrance_cache, key)
        if cached is self._CACHE_MISSING:
            exact_class_counts: Counter[int] = Counter()
            invalid = 0
            for action in unselected_blocks(key):
                candidate = add_block(key, action)
                candidate_rank = self.affine_rank(candidate)
                if candidate_rank < 25:
                    continue
                if candidate_rank > 25:
                    invalid += 1
                    continue
                label = self.terminal_label(candidate)
                if not label.is_exact:
                    invalid += 1
                    continue
                class_id = exact_class_id(label.label)
                if class_id is not None:
                    exact_class_counts[class_id] += 1
            cached = self._cache_put(
                self.rank24_entrance_cache,
                key,
                (dict(exact_class_counts), invalid),
                heavy=False,
            )
        exact_class_counts, invalid = cached  # type: ignore[misc]
        return dict(exact_class_counts), int(invalid)

    def phase_weights(self, rank: int) -> tuple[float, float, float]:
        """Weights for rank gain, supportability, and flatness by search phase."""
        if rank < self.config.phase_b_start_rank:
            return 2.0, 0.2, 0.3
        if rank <= 23:
            return 1.5, 0.5, 1.0
        return 1.0, 1.5, 1.2

    def score_action(
        self,
        key: BlockKey,
        action: int,
        *,
        terminal_score_fn: Callable[[BlockKey, FacetLabel], float] | None = None,
    ) -> ExpansionScore:
        """Score the candidate produced by adding one block."""
        structure = self.action_structure(key, action)
        if structure.terminal is not None:
            score = (
                self.terminal_score(structure.key)
                if terminal_score_fn is None
                else float(terminal_score_fn(structure.key, structure.terminal))
            )
        else:
            score = float(structure.base_score if structure.base_score is not None else 0.0)
            if structure.new_rank == 24 and self.config.rank24_entrance_weight != 0.0:
                entrance = self.rank24_entrance_metrics(structure.key)
                score += self.config.rank24_entrance_weight * (
                    self.config.rank24_entrance_exists_weight * (1.0 if entrance["rare"] > 0 else 0.0)
                    + math.log1p(float(entrance["rare"]))
                    - self.config.rank24_class44_entrance_weight * math.log1p(float(entrance["class44"]))
                    - self.config.rank24_invalid_entrance_weight * math.log1p(float(entrance["invalid"]))
                )
        return ExpansionScore(
            action=structure.action,
            key=structure.key,
            score=float(score),
            phase=structure.phase,
            old_rank=structure.old_rank,
            new_rank=structure.new_rank,
            rank_gain=structure.rank_gain,
            flat_capacity=structure.flat_capacity,
            supportability=structure.supportability,
            terminal=structure.terminal,
        )

    def action_structure(self, key: BlockKey, action: int) -> ExpansionStructure:
        cache_key = (key, int(action))
        cached = self._cache_get(self.action_structure_cache, cache_key)
        if cached is not self._CACHE_MISSING:
            return cached  # type: ignore[return-value]

        candidate = add_block(key, action)
        old_rank = self.affine_rank(key)
        new_rank = self.affine_rank(candidate)
        rank_gain = new_rank - old_rank

        terminal: FacetLabel | None = None
        support: SupportabilityMetrics | None = None
        flat = 0 if new_rank >= 25 else self.flat_capacity(candidate)

        base_score: float | None = None
        if new_rank >= 25:
            terminal = self.terminal_label(candidate)
            phase = "C"
        else:
            w_rank, _w_support, w_flat = self.phase_weights(new_rank)
            base_score = (
                self.config.rank_gain_weight_scale * w_rank * float(rank_gain)
                - self.config.flat_penalty_weight_scale * w_flat * math.log1p(float(flat))
            )
            if rank_gain <= 0:
                base_score -= self.config.nonpositive_rank_gain_penalty
            if new_rank >= self.config.supportability_start_rank:
                support = self.supportability_metrics(candidate)
                if support.supporting_shift > self.config.support_tol**2 or support.closer_side > 0.0:
                    gate_penalty = math.log1p(support.supporting_shift) + (
                        self.config.supportability_gate_closer_weight * support.closer_side
                    )
                    base_score -= self.config.supportability_gate_weight * gate_penalty
            phase = "A" if new_rank < self.config.phase_b_start_rank else ("B1" if new_rank <= 23 else "B2")

        structure = ExpansionStructure(
            action=int(action),
            key=candidate,
            phase=phase,
            old_rank=int(old_rank),
            new_rank=int(new_rank),
            rank_gain=int(rank_gain),
            flat_capacity=int(flat),
            supportability=support,
            terminal=terminal,
            base_score=None if base_score is None else float(base_score),
        )
        return self._cache_put(
            self.action_structure_cache,
            cache_key,
            structure,
            heavy=False,
        )  # type: ignore[return-value]


def exact_class_id(label: str) -> int | None:
    if label.startswith("exact:class"):
        return int(label[len("exact:class"):])
    if label.startswith("exact:"):
        return int(label.split(":", 1)[1])
    return None
