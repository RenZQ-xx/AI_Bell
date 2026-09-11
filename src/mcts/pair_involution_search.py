from __future__ import annotations

"""Pair-involution MCTS specialized for 26-vertex Bell 3-2-2 facets.

The Bell symmetry group contains 261 fixed-point-free involutions on the 64
deterministic vertices.  Conjugation reduces them to 18 pair partitions.  This
module gives every partition an independent MCTS tree and searches for exactly
13 selected pairs.  Only prefixes with the maximum possible affine rank are
kept; rank-21 and rank-23 prefixes are completed exhaustively.

The Corrector is deliberately local: it removes one or two pairs from an exact
terminal support and sends the retained rank-23/rank-21 prefix through the same
cached exhaustive tail.  Trees have independent values and RNG streams, while
they share global class discovery and terminal-support classification.

No known class support is projected onto a pair partition during search.
Rank-19 lookahead scores actions only by completing them to rank-25 terminals;
the PairTargetBank name is retained solely for a post-search representability
report and that object is never passed to a search tree.
"""

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from baseline.bell322 import generate_bell322_points
from baseline.orbit_blocks import Permutation, build_state_group_permutations
from baseline.reference_classes import (
    DEFAULT_EXAMPLES_PATH,
    build_support_class_index,
)


DEFAULT_I200_BASELINE = Path(
    "src/mcts/runs/interrupt_search_class1_i200_shadow_r20_r21.json"
)
DEFAULT_I500_BASELINE = Path(
    "src/mcts/runs/interrupt_search_class1_i500_shadow_r20_r21.json"
)


@dataclass(frozen=True)
class PairPattern:
    """One conjugacy-inequivalent fixed-point-free involution."""

    pattern_id: int
    involution: Permutation
    conjugacy_size: int
    blocks: tuple[tuple[int, int], ...]


def _inverse_permutation(permutation: Sequence[int]) -> Permutation:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[int(target)] = int(source)
    return tuple(inverse)


def _pair_blocks(involution: Sequence[int]) -> tuple[tuple[int, int], ...]:
    blocks = tuple(
        sorted(
            (int(vertex), int(image))
            for vertex, image in enumerate(involution)
            if int(vertex) < int(image)
        )
    )
    if len(blocks) != 32:
        raise ValueError(f"expected 32 pairs, got {len(blocks)}")
    covered = {vertex for block in blocks for vertex in block}
    if covered != set(range(64)):
        raise ValueError("pair partition does not cover all 64 vertices")
    return blocks


@lru_cache(maxsize=1)
def build_pair_involution_patterns() -> tuple[PairPattern, ...]:
    """Return the 18 conjugacy classes of fixed-point-free involutions."""
    group = build_state_group_permutations()
    involutions = tuple(
        permutation
        for permutation in group
        if all(
            permutation[permutation[index]] == index
            and permutation[index] != index
            for index in range(64)
        )
    )
    involution_set = set(involutions)
    inverses = tuple(_inverse_permutation(permutation) for permutation in group)
    unseen = set(involutions)
    classes: list[tuple[Permutation, int]] = []
    while unseen:
        representative = min(unseen)
        conjugates = {
            tuple(
                conjugator[representative[inverse[index]]]
                for index in range(64)
            )
            for conjugator, inverse in zip(group, inverses)
        }
        conjugates.intersection_update(involution_set)
        unseen.difference_update(conjugates)
        classes.append((min(conjugates), len(conjugates)))

    classes.sort(key=lambda item: item[0])
    patterns = tuple(
        PairPattern(
            pattern_id=index,
            involution=representative,
            conjugacy_size=conjugacy_size,
            blocks=_pair_blocks(representative),
        )
        for index, (representative, conjugacy_size) in enumerate(classes, start=1)
    )
    if len(involutions) != 261 or len(patterns) != 18:
        raise RuntimeError(
            "unexpected Bell symmetry involutions: "
            f"fixed_point_free={len(involutions)}, conjugacy_classes={len(patterns)}"
        )
    return patterns


class PairTargetBank:
    """Post-search report of known supports representable by one pair pattern.

    Search trees never receive this object.  It is intentionally constructed
    only after every scheduled iteration has finished.
    """

    def __init__(
        self,
        pattern: PairPattern,
        support_class_index: Mapping[int, int],
        target_classes: Sequence[int],
    ) -> None:
        self.pattern = pattern
        target_set = {int(value) for value in target_classes}
        block_words = tuple((1 << first) | (1 << second) for first, second in pattern.blocks)
        masks: dict[int, set[int]] = {}
        for support_word, raw_class_id in support_class_index.items():
            class_id = int(raw_class_id)
            word = int(support_word)
            if class_id not in target_set or word.bit_count() != 26:
                continue
            block_mask = 0
            representable = True
            for block_index, block_word in enumerate(block_words):
                intersection = word & block_word
                if intersection == block_word:
                    block_mask |= 1 << block_index
                elif intersection != 0:
                    representable = False
                    break
            if representable and block_mask.bit_count() == 13:
                masks.setdefault(class_id, set()).add(block_mask)
        self.masks_by_class = {
            class_id: tuple(sorted(class_masks))
            for class_id, class_masks in sorted(masks.items())
        }


@dataclass(frozen=True)
class PairSearchConfig:
    initial_class_id: int = 1
    iterations: int = 200
    target_classes: tuple[int, ...] = tuple(range(1, 47))
    examples_path: Path = DEFAULT_EXAMPLES_PATH
    seed: int = 20260502
    exploration_constant: float = 1.4
    discount: float = 0.97
    progressive_k0: int = 1
    progressive_alpha: float = 1.0
    progressive_beta: float = 0.5
    epoch_value_decay: float = 0.25
    new_class_reward: float = 100.0
    known_class_reward: float = 10.0
    invalid_reward: float = -10.0
    dead_end_reward: float = -25.0
    full_rank_tol: float = 1e-9
    terminal_cache_max_entries: int = 300_000
    rank19_lookahead_candidate_pool: int = 8
    corrector_enabled: bool = True
    corrector_interval: int = 8
    corrector_max_events_per_tree: int = 24
    corrector_source_pool: int = 8
    checkpoint_interval: int = 50
    pattern_ids: tuple[int, ...] | None = None
    baseline_path: Path | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.progressive_k0 <= 0:
            raise ValueError("progressive_k0 must be positive")
        if self.progressive_alpha < 0.0 or not 0.0 < self.progressive_beta <= 1.0:
            raise ValueError("invalid progressive widening parameters")
        if not 0.0 <= self.epoch_value_decay <= 1.0:
            raise ValueError("epoch_value_decay must be in [0, 1]")
        if self.rank19_lookahead_candidate_pool < 0:
            raise ValueError("rank19_lookahead_candidate_pool must be nonnegative")
        if self.corrector_interval <= 0:
            raise ValueError("corrector_interval must be positive")
        if self.corrector_max_events_per_tree < 0:
            raise ValueError("corrector_max_events_per_tree must be nonnegative")
        if self.corrector_source_pool <= 0:
            raise ValueError("corrector_source_pool must be positive")


@dataclass
class SharedReferenceTerminalCache:
    """Cache class IDs by 64-bit support; rewards stay dynamic and uncached."""

    support_class_index: Mapping[int, int]
    max_entries: int = 300_000
    results: dict[int, int | None] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    exact_results: int = 0
    nonreference_results: int = 0

    def classify(self, support_word: int) -> int | None:
        word = int(support_word)
        if word in self.results:
            self.hits += 1
            return self.results[word]
        self.misses += 1
        class_id = self.support_class_index.get(word)
        result = None if class_id is None else int(class_id)
        self.results[word] = result
        if result is None:
            self.nonreference_results += 1
        else:
            self.exact_results += 1
        if self.max_entries > 0 and len(self.results) > self.max_entries:
            self.results.pop(next(iter(self.results)))
            self.evictions += 1
        return result

    def to_dict(self) -> dict[str, int]:
        return {
            "entries": len(self.results),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "exact_results": self.exact_results,
            "nonreference_results": self.nonreference_results,
        }


@dataclass
class GlobalDiscoveryState:
    initial_class_id: int
    target_classes: set[int]
    new_class_reward: float
    known_class_reward: float
    started_at: float
    discovered_classes: set[int] = field(default_factory=set)
    class_hit_counts: Counter[int] = field(default_factory=Counter)
    timeline: list[dict[str, object]] = field(default_factory=list)
    epoch: int = 0

    @classmethod
    def from_config(
        cls,
        config: PairSearchConfig,
        *,
        started_at: float,
    ) -> "GlobalDiscoveryState":
        state = cls(
            initial_class_id=int(config.initial_class_id),
            target_classes={int(value) for value in config.target_classes},
            new_class_reward=float(config.new_class_reward),
            known_class_reward=float(config.known_class_reward),
            started_at=float(started_at),
        )
        state.discovered_classes.add(int(config.initial_class_id))
        state.class_hit_counts[int(config.initial_class_id)] += 1
        state.timeline.append(
            {
                "event_index": 0,
                "class_id": int(config.initial_class_id),
                "source": "initial_seed",
                "pattern_id": None,
                "local_iteration": 0,
                "global_iteration": 0,
                "round": 0,
                "wall_seconds": 0.0,
                "selected_blocks": [],
                "support_word_hex": None,
            }
        )
        return state

    def observe(
        self,
        class_id: int,
        *,
        source: str,
        pattern_id: int,
        local_iteration: int,
        global_iteration: int,
        round_index: int,
        selected_mask: int,
        support_word: int,
    ) -> tuple[float, bool]:
        normalized = int(class_id)
        self.class_hit_counts[normalized] += 1
        is_new = normalized not in self.discovered_classes
        if is_new:
            self.discovered_classes.add(normalized)
            self.epoch += 1
            self.timeline.append(
                {
                    "event_index": len(self.timeline),
                    "class_id": normalized,
                    "source": str(source),
                    "pattern_id": int(pattern_id),
                    "local_iteration": int(local_iteration),
                    "global_iteration": int(global_iteration),
                    "round": int(round_index),
                    "wall_seconds": time.perf_counter() - self.started_at,
                    "selected_blocks": _selected_block_indices(selected_mask),
                    "support_word_hex": f"0x{int(support_word):016x}",
                }
            )
        reward = self.new_class_reward if is_new else self.known_class_reward
        return float(reward), bool(is_new)


def _selected_block_indices(mask: int) -> list[int]:
    return [index for index in range(32) if int(mask) & (1 << index)]


def tail_completion_masks(
    selected_mask: int,
    *,
    target_block_count: int = 13,
    total_blocks: int = 32,
) -> tuple[int, ...]:
    """Enumerate all fixed-cardinality completions of a rank-tail prefix."""
    selected = int(selected_mask)
    needed = int(target_block_count) - selected.bit_count()
    if needed < 0:
        return ()
    remaining = [index for index in range(total_blocks) if selected & (1 << index) == 0]
    return tuple(
        selected | sum(1 << index for index in addition)
        for addition in itertools.combinations(remaining, needed)
    )


def corrected_prefix_masks(
    terminal_mask: int,
    delete_count: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Enumerate retained prefixes obtained by deleting selected pairs."""
    selected = _selected_block_indices(terminal_mask)
    return tuple(
        (
            int(terminal_mask) & ~sum(1 << index for index in removed),
            tuple(int(index) for index in removed),
        )
        for removed in itertools.combinations(selected, int(delete_count))
    )


def _extend_pair_basis(
    basis: np.ndarray,
    augmented_points: np.ndarray,
    block: tuple[int, int],
    tolerance: float,
) -> tuple[np.ndarray, int]:
    """Increment an orthonormal row basis by the two vertices in one block."""
    added_rows: list[np.ndarray] = []
    for vertex in block:
        residual = np.asarray(augmented_points[int(vertex)], dtype=np.float64).copy()
        for _pass in range(2):
            if basis.shape[0] > 0:
                residual -= (residual @ basis.T) @ basis
            for row in added_rows:
                residual -= float(np.dot(residual, row)) * row
        norm = float(np.linalg.norm(residual))
        if norm > float(tolerance):
            added_rows.append(residual / norm)
    if not added_rows:
        return basis, 0
    return np.concatenate([basis, np.vstack(added_rows)], axis=0), len(added_rows)


def _basis_for_mask(
    selected_mask: int,
    blocks: Sequence[tuple[int, int]],
    augmented_points: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, bool]:
    basis = np.zeros((0, augmented_points.shape[1]), dtype=np.float64)
    for block_index in _selected_block_indices(selected_mask):
        basis, added = _extend_pair_basis(
            basis,
            augmented_points,
            blocks[block_index],
            tolerance,
        )
        if added != 2:
            return basis, False
    return basis, True


@dataclass
class PairNode:
    selected_mask: int
    support_word: int
    depth: int
    affine_rank: int
    basis: np.ndarray = field(repr=False)
    visits: int = 0
    value_sum: float = 0.0
    epoch: int = 0
    children: dict[int, "PairNode"] = field(default_factory=dict)
    viable_actions: tuple[int, ...] | None = None

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits <= 0 else self.value_sum / float(self.visits)


@dataclass(frozen=True)
class TailCacheEntry:
    exact_class_ids: tuple[int, ...]
    full_rank_terminal_count: int
    invalid_terminal_count: int
    logical_candidate_count: int


@dataclass(frozen=True)
class TailOutcome:
    reward: float
    entry: TailCacheEntry
    new_class_ids: tuple[int, ...]
    cache_hit: bool


@dataclass
class CorrectorEvent:
    event_index: int
    pattern_id: int
    local_iteration: int
    global_iteration: int
    delete_count: int
    source_class_id: int
    source_blocks: list[int]
    deleted_blocks: list[int]
    corrected_blocks: list[int]
    corrected_rank: int
    compatible_missing_class_ids: list[int]
    tail_cache_hit: bool
    logical_terminal_candidates: int
    full_rank_terminal_candidates: int
    exact_class_ids: list[int]
    new_class_ids: list[int]
    reward: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return dict(vars(self))


class PairPatternTree:
    """One MCTS tree with a fixed pair partition."""

    def __init__(
        self,
        *,
        pattern: PairPattern,
        augmented_points: np.ndarray,
        terminal_cache: SharedReferenceTerminalCache,
        global_discovery: GlobalDiscoveryState,
        config: PairSearchConfig,
    ) -> None:
        self.pattern = pattern
        self.augmented_points = augmented_points
        self.terminal_cache = terminal_cache
        self.global_discovery = global_discovery
        self.config = config
        self.rng = random.Random(config.seed + 104_729 * pattern.pattern_id)
        root_basis = np.zeros((0, augmented_points.shape[1]), dtype=np.float64)
        self.root = PairNode(
            selected_mask=0,
            support_word=0,
            depth=0,
            affine_rank=0,
            basis=root_basis,
        )
        self.nodes: dict[int, PairNode] = {0: self.root}
        self.tail_rank21_cache: dict[int, TailCacheEntry] = {}
        self.tail_rank23_cache: dict[int, TailCacheEntry] = {}
        self.corrector_seen_prefixes: dict[int, set[int]] = {1: set(), 2: set()}
        self.corrector_events: list[CorrectorEvent] = []
        self.exact_terminals: dict[int, int] = {}
        self.exact_terminal_order: list[int] = []
        self.stats: Counter[str] = Counter()
        self.rank_prune_counts: Counter[int] = Counter()
        self.policy_bucket_counts: Counter[str] = Counter()
        self.exact_hit_counts: Counter[int] = Counter()
        self.local_new_classes: list[int] = []
        self.elapsed_seconds = 0.0

    def _refresh_epoch(self, node: PairNode) -> None:
        if node.epoch >= self.global_discovery.epoch:
            return
        delta = self.global_discovery.epoch - node.epoch
        node.value_sum *= self.config.epoch_value_decay**delta
        node.epoch = self.global_discovery.epoch

    def _support_word_after(self, support_word: int, action: int) -> int:
        first, second = self.pattern.blocks[int(action)]
        return int(support_word) | (1 << first) | (1 << second)

    def _extensions(
        self,
        selected_mask: int,
        basis: np.ndarray,
        *,
        actions: Sequence[int] | None = None,
        stage: str,
    ) -> list[tuple[int, np.ndarray]]:
        candidates = (
            [index for index in range(32) if selected_mask & (1 << index) == 0]
            if actions is None
            else [int(index) for index in actions]
        )
        accepted: list[tuple[int, np.ndarray]] = []
        for action in candidates:
            self.stats[f"{stage}_action_checks"] += 1
            child_basis, added = _extend_pair_basis(
                basis,
                self.augmented_points,
                self.pattern.blocks[action],
                self.config.full_rank_tol,
            )
            if added == 2:
                accepted.append((action, child_basis))
                self.stats[f"{stage}_full_rank_actions"] += 1
            else:
                observed_affine_rank = max(0, int(child_basis.shape[0]) - 1)
                self.rank_prune_counts[observed_affine_rank] += 1
                self.stats[f"{stage}_rank_pruned_actions"] += 1
        return accepted

    def _policy_choice(
        self,
        candidates: Sequence[tuple[int, np.ndarray]],
    ) -> tuple[int, np.ndarray]:
        if not candidates:
            raise ValueError("policy choice requires at least one candidate")
        self.policy_bucket_counts["uniform_full_rank"] += 1
        return self.rng.choice(list(candidates))

    def _node_viable_actions(self, node: PairNode) -> tuple[int, ...]:
        if node.viable_actions is None:
            extensions = self._extensions(
                node.selected_mask,
                node.basis,
                stage="tree",
            )
            node.viable_actions = tuple(action for action, _basis in extensions)
        return node.viable_actions

    def _progressive_width(self, visits: int, available: int) -> int:
        raw = (
            float(self.config.progressive_k0)
            + self.config.progressive_alpha * float(max(0, visits)) ** self.config.progressive_beta
        )
        return min(int(available), max(1, int(math.floor(raw))))

    def _select_tree_path(self) -> tuple[list[PairNode], PairNode | None]:
        node = self.root
        path = [node]
        while node.depth < 11:
            self._refresh_epoch(node)
            viable = self._node_viable_actions(node)
            unexpanded = [action for action in viable if action not in node.children]
            width = self._progressive_width(node.visits, len(viable))
            if unexpanded and len(node.children) < width:
                extensions = self._extensions(
                    node.selected_mask,
                    node.basis,
                    actions=unexpanded,
                    stage="expansion",
                )
                action, child_basis = self._policy_choice(
                    extensions,
                )
                child_mask = node.selected_mask | (1 << action)
                child = self.nodes.get(child_mask)
                if child is None:
                    child = PairNode(
                        selected_mask=child_mask,
                        support_word=self._support_word_after(node.support_word, action),
                        depth=node.depth + 1,
                        affine_rank=2 * (node.depth + 1) - 1,
                        basis=child_basis,
                        epoch=self.global_discovery.epoch,
                    )
                    self.nodes[child_mask] = child
                node.children[action] = child
                path.append(child)
                self.stats["expanded_nodes"] += 1
                return path, child
            if not node.children:
                self.stats["tree_dead_ends"] += 1
                return path, None

            log_parent = math.log(max(2, node.visits + 1))
            scored_children: list[tuple[float, int, PairNode]] = []
            for action, child in node.children.items():
                self._refresh_epoch(child)
                exploration = self.config.exploration_constant * math.sqrt(
                    log_parent / float(max(1, child.visits))
                )
                scored_children.append(
                    (child.mean_value + exploration, action, child)
                )
            best_score = max(value for value, _action, _child in scored_children)
            tied = [
                (action, child)
                for value, action, child in scored_children
                if math.isclose(value, best_score, rel_tol=1e-12, abs_tol=1e-12)
            ]
            _action, node = self.rng.choice(tied)
            path.append(node)
        return path, node

    def _rank19_lookahead_choice(
        self,
        candidates: Sequence[tuple[int, np.ndarray]],
        *,
        selected_mask: int,
        support_word: int,
        local_iteration: int,
        global_iteration: int,
    ) -> tuple[int, np.ndarray, float]:
        """Score rank-21 children only by their fully enumerated terminals."""
        pool_size = min(
            len(candidates),
            max(1, int(self.config.rank19_lookahead_candidate_pool)),
        )
        pool = (
            list(candidates)
            if len(candidates) <= pool_size
            else self.rng.sample(list(candidates), pool_size)
        )
        evaluated: list[tuple[tuple[float, int, int], int, np.ndarray]] = []
        for action, child_basis in pool:
            child_mask = selected_mask | (1 << action)
            child_support = self._support_word_after(support_word, action)
            outcome = self._exhaust_rank21(
                selected_mask=child_mask,
                support_word=child_support,
                basis=child_basis,
                local_iteration=local_iteration,
                global_iteration=global_iteration,
                source="rank19_terminal_lookahead",
            )
            self.stats["rank19_lookahead_candidates"] += 1
            self.stats["rank19_lookahead_full_rank_terminals"] += (
                outcome.entry.full_rank_terminal_count
            )
            evaluated.append(
                (
                    (
                        float(outcome.reward),
                        len(outcome.entry.exact_class_ids),
                        int(outcome.entry.full_rank_terminal_count),
                    ),
                    action,
                    child_basis,
                )
            )
        best_key = max(item[0] for item in evaluated)
        tied = [item for item in evaluated if item[0] == best_key]
        score, action, child_basis = self.rng.choice(tied)
        self.policy_bucket_counts["rank19_terminal_lookahead"] += 1
        return action, child_basis, float(score[0])

    def _rollout_to_rank21(
        self,
        node: PairNode,
        *,
        local_iteration: int,
        global_iteration: int,
    ) -> tuple[int, int, np.ndarray, float | None] | None:
        selected_mask = int(node.selected_mask)
        support_word = int(node.support_word)
        basis = node.basis
        lookahead_reward: float | None = None
        while selected_mask.bit_count() < 11:
            extensions = self._extensions(
                selected_mask,
                basis,
                stage="rollout",
            )
            if not extensions:
                self.stats["rollout_dead_ends"] += 1
                return None
            if (
                selected_mask.bit_count() == 10
                and self.config.rank19_lookahead_candidate_pool > 0
            ):
                action, basis, lookahead_reward = self._rank19_lookahead_choice(
                    extensions,
                    selected_mask=selected_mask,
                    support_word=support_word,
                    local_iteration=local_iteration,
                    global_iteration=global_iteration,
                )
            else:
                action, basis = self._policy_choice(extensions)
            selected_mask |= 1 << action
            support_word = self._support_word_after(support_word, action)
        return selected_mask, support_word, basis, lookahead_reward

    def _record_exact_terminal(self, selected_mask: int, class_id: int) -> None:
        if selected_mask not in self.exact_terminals:
            self.exact_terminals[selected_mask] = int(class_id)
            self.exact_terminal_order.append(selected_mask)

    def _evaluate_terminal(
        self,
        *,
        selected_mask: int,
        support_word: int,
        local_iteration: int,
        global_iteration: int,
        source: str,
    ) -> tuple[float, int | None, bool]:
        self.stats["terminal_evaluations"] += 1
        class_id = self.terminal_cache.classify(support_word)
        if class_id is None:
            self.stats["nonreference_terminals"] += 1
            return float(self.config.invalid_reward), None, False
        self.stats["exact_terminal_hits"] += 1
        self.exact_hit_counts[class_id] += 1
        self._record_exact_terminal(selected_mask, class_id)
        reward, is_new = self.global_discovery.observe(
            class_id,
            source=source,
            pattern_id=self.pattern.pattern_id,
            local_iteration=local_iteration,
            global_iteration=global_iteration,
            round_index=local_iteration,
            selected_mask=selected_mask,
            support_word=support_word,
        )
        if is_new:
            self.local_new_classes.append(class_id)
            self.stats["new_global_classes"] += 1
        return reward, class_id, is_new

    def _cached_tail_outcome(self, entry: TailCacheEntry) -> TailOutcome:
        reward = (
            self.config.known_class_reward
            if entry.exact_class_ids
            else self.config.invalid_reward
        )
        return TailOutcome(
            reward=float(reward),
            entry=entry,
            new_class_ids=(),
            cache_hit=True,
        )

    def _exhaust_rank23(
        self,
        *,
        selected_mask: int,
        support_word: int,
        basis: np.ndarray,
        local_iteration: int,
        global_iteration: int,
        source: str,
    ) -> TailOutcome:
        cached = self.tail_rank23_cache.get(selected_mask)
        if cached is not None:
            self.stats["rank23_tail_cache_hits"] += 1
            return self._cached_tail_outcome(cached)
        self.stats["rank23_tail_prefixes"] += 1
        before = set(self.global_discovery.discovered_classes)
        exact_classes: set[int] = set()
        full_rank_count = 0
        invalid_count = 0
        logical_count = 32 - selected_mask.bit_count()
        best_reward = -math.inf
        extensions = self._extensions(
            selected_mask,
            basis,
            stage="rank23_tail",
        )
        for action, child_basis in extensions:
            if child_basis.shape[0] != 26:
                continue
            terminal_mask = selected_mask | (1 << action)
            terminal_support = self._support_word_after(support_word, action)
            reward, class_id, _is_new = self._evaluate_terminal(
                selected_mask=terminal_mask,
                support_word=terminal_support,
                local_iteration=local_iteration,
                global_iteration=global_iteration,
                source=f"{source}:rank23_tail",
            )
            full_rank_count += 1
            best_reward = max(best_reward, reward)
            if class_id is None:
                invalid_count += 1
            else:
                exact_classes.add(class_id)
        if best_reward == -math.inf:
            best_reward = float(self.config.dead_end_reward)
        entry = TailCacheEntry(
            exact_class_ids=tuple(sorted(exact_classes)),
            full_rank_terminal_count=full_rank_count,
            invalid_terminal_count=invalid_count,
            logical_candidate_count=logical_count,
        )
        self.tail_rank23_cache[selected_mask] = entry
        new_classes = tuple(sorted(self.global_discovery.discovered_classes - before))
        return TailOutcome(
            reward=float(best_reward),
            entry=entry,
            new_class_ids=new_classes,
            cache_hit=False,
        )

    def _exhaust_rank21(
        self,
        *,
        selected_mask: int,
        support_word: int,
        basis: np.ndarray,
        local_iteration: int,
        global_iteration: int,
        source: str,
    ) -> TailOutcome:
        cached = self.tail_rank21_cache.get(selected_mask)
        if cached is not None:
            self.stats["rank21_tail_cache_hits"] += 1
            return self._cached_tail_outcome(cached)
        self.stats["rank21_tail_prefixes"] += 1
        before = set(self.global_discovery.discovered_classes)
        exact_classes: set[int] = set()
        full_rank_count = 0
        invalid_count = 0
        logical_count = 0
        best_reward = -math.inf
        rank23_extensions = self._extensions(
            selected_mask,
            basis,
            stage="rank21_tail",
        )
        for action, child_basis in rank23_extensions:
            child_mask = selected_mask | (1 << action)
            child_support = self._support_word_after(support_word, action)
            outcome = self._exhaust_rank23(
                selected_mask=child_mask,
                support_word=child_support,
                basis=child_basis,
                local_iteration=local_iteration,
                global_iteration=global_iteration,
                source=source,
            )
            exact_classes.update(outcome.entry.exact_class_ids)
            full_rank_count += outcome.entry.full_rank_terminal_count
            invalid_count += outcome.entry.invalid_terminal_count
            logical_count += outcome.entry.logical_candidate_count
            best_reward = max(best_reward, outcome.reward)
        if best_reward == -math.inf:
            best_reward = float(self.config.dead_end_reward)
        entry = TailCacheEntry(
            exact_class_ids=tuple(sorted(exact_classes)),
            full_rank_terminal_count=full_rank_count,
            invalid_terminal_count=invalid_count,
            logical_candidate_count=logical_count,
        )
        self.tail_rank21_cache[selected_mask] = entry
        new_classes = tuple(sorted(self.global_discovery.discovered_classes - before))
        return TailOutcome(
            reward=float(best_reward),
            entry=entry,
            new_class_ids=new_classes,
            cache_hit=False,
        )

    def _corrector_proposal(
        self,
        delete_count: int,
    ) -> tuple[int, int, int, tuple[int, ...]] | None:
        sources = self.exact_terminal_order[-self.config.corrector_source_pool :]
        if not sources:
            return None
        proposals: list[tuple[int, int, int, tuple[int, ...]]] = []
        for source_mask in reversed(sources):
            source_class_id = self.exact_terminals[source_mask]
            for corrected_mask, removed in corrected_prefix_masks(
                source_mask,
                delete_count,
            ):
                self.stats["corrector_proposal_checks"] += 1
                if corrected_mask in self.corrector_seen_prefixes[delete_count]:
                    continue
                proposals.append(
                    (
                        source_mask,
                        source_class_id,
                        corrected_mask,
                        removed,
                    )
                )
        if not proposals:
            return None
        chosen = self.rng.choice(proposals)
        self.corrector_seen_prefixes[delete_count].add(chosen[2])
        return chosen

    def _run_corrector(
        self,
        *,
        local_iteration: int,
        global_iteration: int,
    ) -> float | None:
        if not self.config.corrector_enabled:
            return None
        if local_iteration % self.config.corrector_interval != 0:
            return None
        if len(self.corrector_events) >= self.config.corrector_max_events_per_tree:
            return None
        best_reward: float | None = None
        for delete_count in (1, 2):
            if len(self.corrector_events) >= self.config.corrector_max_events_per_tree:
                break
            proposal = self._corrector_proposal(delete_count)
            if proposal is None:
                continue
            source_mask, source_class_id, corrected_mask, removed = proposal
            event_started = time.perf_counter()
            basis, full_rank = _basis_for_mask(
                corrected_mask,
                self.pattern.blocks,
                self.augmented_points,
                self.config.full_rank_tol,
            )
            if not full_rank:
                self.stats["corrector_rank_pruned_prefixes"] += 1
                continue
            support_word = 0
            for action in _selected_block_indices(corrected_mask):
                support_word = self._support_word_after(support_word, action)
            if delete_count == 1:
                outcome = self._exhaust_rank23(
                    selected_mask=corrected_mask,
                    support_word=support_word,
                    basis=basis,
                    local_iteration=local_iteration,
                    global_iteration=global_iteration,
                    source="corrector_remove1",
                )
            else:
                outcome = self._exhaust_rank21(
                    selected_mask=corrected_mask,
                    support_word=support_word,
                    basis=basis,
                    local_iteration=local_iteration,
                    global_iteration=global_iteration,
                    source="corrector_remove2",
                )
            self.corrector_events.append(
                CorrectorEvent(
                    event_index=len(self.corrector_events) + 1,
                    pattern_id=self.pattern.pattern_id,
                    local_iteration=local_iteration,
                    global_iteration=global_iteration,
                    delete_count=delete_count,
                    source_class_id=source_class_id,
                    source_blocks=_selected_block_indices(source_mask),
                    deleted_blocks=list(removed),
                    corrected_blocks=_selected_block_indices(corrected_mask),
                    corrected_rank=23 if delete_count == 1 else 21,
                    compatible_missing_class_ids=[],
                    tail_cache_hit=outcome.cache_hit,
                    logical_terminal_candidates=outcome.entry.logical_candidate_count,
                    full_rank_terminal_candidates=outcome.entry.full_rank_terminal_count,
                    exact_class_ids=list(outcome.entry.exact_class_ids),
                    new_class_ids=list(outcome.new_class_ids),
                    reward=float(outcome.reward),
                    elapsed_seconds=time.perf_counter() - event_started,
                )
            )
            self.stats[f"corrector_remove{delete_count}_events"] += 1
            best_reward = (
                float(outcome.reward)
                if best_reward is None
                else max(best_reward, float(outcome.reward))
            )
        return best_reward

    def run_iteration(self, local_iteration: int, global_iteration: int) -> None:
        started = time.perf_counter()
        path, leaf = self._select_tree_path()
        if leaf is None:
            reward = float(self.config.dead_end_reward)
        else:
            rollout = self._rollout_to_rank21(
                leaf,
                local_iteration=local_iteration,
                global_iteration=global_iteration,
            )
            if rollout is None:
                reward = float(self.config.dead_end_reward)
            else:
                selected_mask, support_word, basis, lookahead_reward = rollout
                outcome = self._exhaust_rank21(
                    selected_mask=selected_mask,
                    support_word=support_word,
                    basis=basis,
                    local_iteration=local_iteration,
                    global_iteration=global_iteration,
                    source="mcts_filler",
                )
                reward = float(outcome.reward)
                if lookahead_reward is not None:
                    reward = max(reward, float(lookahead_reward))

        corrector_reward = self._run_corrector(
            local_iteration=local_iteration,
            global_iteration=global_iteration,
        )
        if corrector_reward is not None:
            reward = max(reward, corrector_reward)

        backed_up = float(reward)
        for node in reversed(path):
            self._refresh_epoch(node)
            node.visits += 1
            node.value_sum += backed_up
            backed_up *= self.config.discount
        self.stats["iterations"] += 1
        self.elapsed_seconds += time.perf_counter() - started

    def summary(
        self,
        representability_bank: PairTargetBank | None = None,
    ) -> dict[str, object]:
        max_depth = max(node.depth for node in self.nodes.values())
        masks_by_class = (
            {}
            if representability_bank is None
            else representability_bank.masks_by_class
        )
        return {
            "pattern_id": self.pattern.pattern_id,
            "seed": self.config.seed + 104_729 * self.pattern.pattern_id,
            "conjugacy_size": self.pattern.conjugacy_size,
            "involution": list(self.pattern.involution),
            "blocks": [list(block) for block in self.pattern.blocks],
            "iterations": self.stats["iterations"],
            "elapsed_seconds": self.elapsed_seconds,
            "node_count": len(self.nodes),
            "max_tree_depth": max_depth,
            "root_visits": self.root.visits,
            "root_child_count": len(self.root.children),
            "representability_analysis_phase": (
                "pending" if representability_bank is None else "post_search"
            ),
            "representable_target_mask_counts": {
                str(class_id): len(masks)
                for class_id, masks in masks_by_class.items()
            },
            "representable_class_ids": sorted(masks_by_class),
            "new_global_class_ids": sorted(set(self.local_new_classes)),
            "exact_hit_counts": {
                str(class_id): count
                for class_id, count in sorted(self.exact_hit_counts.items())
            },
            "exact_terminal_support_count": len(self.exact_terminals),
            "tail_cache": {
                "rank21_entries": len(self.tail_rank21_cache),
                "rank23_entries": len(self.tail_rank23_cache),
                "rank21_hits": self.stats["rank21_tail_cache_hits"],
                "rank23_hits": self.stats["rank23_tail_cache_hits"],
            },
            "rank_prune_counts": {
                str(rank): count for rank, count in sorted(self.rank_prune_counts.items())
            },
            "policy_bucket_counts": dict(sorted(self.policy_bucket_counts.items())),
            "stats": dict(sorted(self.stats.items())),
            "corrector_events": [event.to_dict() for event in self.corrector_events],
        }


def _coverage_from_baseline(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        values = summary.get("encountered_rare_target_classes")
        if isinstance(values, list):
            return sorted(int(value) for value in values)
    values = payload.get("exact_class_counts", {})
    if isinstance(values, dict):
        return sorted(int(value) for value in values)
    return []


def _automatic_baseline(config: PairSearchConfig) -> Path | None:
    if config.baseline_path is not None:
        return config.baseline_path
    if config.initial_class_id != 1:
        return None
    if config.iterations == 200:
        return DEFAULT_I200_BASELINE
    if config.iterations == 500:
        return DEFAULT_I500_BASELINE
    return None


def _baseline_comparison(
    config: PairSearchConfig,
    pair_coverage: Sequence[int],
) -> dict[str, object] | None:
    baseline_path = _automatic_baseline(config)
    if baseline_path is None or not baseline_path.exists():
        return None
    baseline = _coverage_from_baseline(baseline_path)
    baseline_set = set(baseline)
    pair_set = {int(value) for value in pair_coverage}
    combined = sorted(baseline_set | pair_set)
    targets = {int(value) for value in config.target_classes}
    return {
        "baseline_path": str(baseline_path),
        "baseline_class_ids": baseline,
        "baseline_coverage_count": len(baseline_set & targets),
        "pair_class_ids": sorted(pair_set),
        "pair_coverage_count": len(pair_set & targets),
        "pair_only_added_class_ids": sorted(pair_set - baseline_set),
        "baseline_only_class_ids": sorted(baseline_set - pair_set),
        "combined_class_ids": combined,
        "combined_coverage_count": len(set(combined) & targets),
        "combined_missing_class_ids": sorted(targets - set(combined)),
    }


def _config_dict(config: PairSearchConfig) -> dict[str, object]:
    return {
        "initial_class_id": config.initial_class_id,
        "iterations_per_tree": config.iterations,
        "target_classes": list(config.target_classes),
        "examples_path": str(config.examples_path),
        "seed": config.seed,
        "exploration_constant": config.exploration_constant,
        "discount": config.discount,
        "progressive_k0": config.progressive_k0,
        "progressive_alpha": config.progressive_alpha,
        "progressive_beta": config.progressive_beta,
        "epoch_value_decay": config.epoch_value_decay,
        "new_class_reward": config.new_class_reward,
        "known_class_reward": config.known_class_reward,
        "invalid_reward": config.invalid_reward,
        "dead_end_reward": config.dead_end_reward,
        "full_rank_tol": config.full_rank_tol,
        "terminal_cache_max_entries": config.terminal_cache_max_entries,
        "target_guidance_enabled": False,
        "target_bank_usage": "post_search_diagnostics_only",
        "rank19_lookahead_candidate_pool": config.rank19_lookahead_candidate_pool,
        "corrector_enabled": config.corrector_enabled,
        "corrector_interval": config.corrector_interval,
        "corrector_max_events_per_tree": config.corrector_max_events_per_tree,
        "corrector_source_pool": config.corrector_source_pool,
        "checkpoint_interval": config.checkpoint_interval,
        "pattern_ids": None if config.pattern_ids is None else list(config.pattern_ids),
    }


def _aggregate_tree_stats(trees: Sequence[PairPatternTree]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for tree in trees:
        total.update(tree.stats)
    return dict(sorted(total.items()))


def _build_payload(
    *,
    config: PairSearchConfig,
    trees: Sequence[PairPatternTree],
    global_discovery: GlobalDiscoveryState,
    terminal_cache: SharedReferenceTerminalCache,
    setup_seconds: float,
    elapsed_seconds: float,
    rounds_completed: int,
    status: str,
    representability_banks: Mapping[int, PairTargetBank] | None = None,
    post_search_diagnostics_seconds: float = 0.0,
    search_elapsed_seconds: float | None = None,
) -> dict[str, object]:
    targets = {int(value) for value in config.target_classes}
    coverage = sorted(global_discovery.discovered_classes & targets)
    search_discoveries = sorted(
        class_id
        for class_id in coverage
        if class_id != int(config.initial_class_id)
    )
    banks = {} if representability_banks is None else representability_banks
    runs = [
        tree.summary(banks.get(tree.pattern.pattern_id))
        for tree in trees
    ]
    corrector_events = [
        event.to_dict()
        for tree in trees
        for event in tree.corrector_events
    ]
    payload: dict[str, object] = {
        "meta": {
            "script": "src/mcts/pair_involution_search.py",
            "algorithm": "18_tree_pair_involution_mcts_terminal_lookahead_corrector",
            "status": status,
            "pair_pattern_count": len(trees),
            "fixed_point_free_involution_count": 261,
            "selected_block_constraint": 13,
            "selected_vertex_constraint": 26,
            "full_rank_prefix_rule": "rank(2k vertices) == 2k - 1",
            "tail_exhaustion_ranks": [21, 23],
            "terminal_lookahead_ranks": [19, 21, 23],
            "terminal_classifier": "shared_exact_support_index",
            "class_support_usage_during_search": "complete_terminal_validation_only",
            "pair_target_bank_constructed_after_search": status == "complete",
            "setup_seconds": setup_seconds,
            "search_elapsed_seconds": (
                elapsed_seconds
                if search_elapsed_seconds is None
                else search_elapsed_seconds
            ),
            "post_search_diagnostics_seconds": post_search_diagnostics_seconds,
            "elapsed_seconds": elapsed_seconds,
            **_config_dict(config),
        },
        "summary": {
            "rounds_completed": rounds_completed,
            "scheduled_tree_iterations": rounds_completed * len(trees),
            "coverage_class_ids": coverage,
            "coverage_count": len(coverage),
            "search_discovered_class_ids": search_discoveries,
            "search_discovery_count": len(search_discoveries),
            "missing_class_ids": sorted(targets - set(coverage)),
            "class18_found_by_pair_search": 18 in search_discoveries,
            "class_hit_counts": {
                str(class_id): count
                for class_id, count in sorted(global_discovery.class_hit_counts.items())
            },
            "representable_class_ids": sorted(
                {
                    class_id
                    for bank in banks.values()
                    for class_id in bank.masks_by_class
                }
            ),
            "aggregate_tree_stats": _aggregate_tree_stats(trees),
        },
        "discovery_timeline": list(global_discovery.timeline),
        "terminal_cache": terminal_cache.to_dict(),
        "corrector": {
            "event_count": len(corrector_events),
            "remove1_event_count": sum(
                1 for event in corrector_events if event["delete_count"] == 1
            ),
            "remove2_event_count": sum(
                1 for event in corrector_events if event["delete_count"] == 2
            ),
            "new_class_ids": sorted(
                {
                    int(class_id)
                    for event in corrector_events
                    for class_id in event["new_class_ids"]
                }
            ),
            "events": corrector_events,
        },
        "runs": runs,
    }
    if status == "complete":
        comparison = _baseline_comparison(config, coverage)
        if comparison is not None:
            payload["baseline_comparison"] = comparison
    return payload


def _write_outputs(output_path: Path, payload: dict[str, object]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    runs_path = output_path.with_name(f"{output_path.stem}.runs.json")
    runs_path.write_text(
        json.dumps(payload["runs"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return runs_path


def run_pair_involution_search(
    config: PairSearchConfig,
    *,
    output_path: Path,
) -> dict[str, object]:
    """Run the fixed 18-tree ensemble and write summary plus per-tree records."""
    started_at = time.perf_counter()
    setup_started = time.perf_counter()
    patterns = build_pair_involution_patterns()
    if config.pattern_ids is not None:
        selected_ids = {int(value) for value in config.pattern_ids}
        patterns = tuple(pattern for pattern in patterns if pattern.pattern_id in selected_ids)
        missing_ids = selected_ids - {pattern.pattern_id for pattern in patterns}
        if missing_ids:
            raise ValueError(f"unknown pair pattern ids: {sorted(missing_ids)}")
    support_class_index = build_support_class_index(config.examples_path)
    points = np.asarray(generate_bell322_points(), dtype=np.float64)
    augmented_points = np.concatenate(
        [np.ones((len(points), 1), dtype=np.float64), points],
        axis=1,
    )
    terminal_cache = SharedReferenceTerminalCache(
        support_class_index=support_class_index,
        max_entries=config.terminal_cache_max_entries,
    )
    global_discovery = GlobalDiscoveryState.from_config(
        config,
        started_at=started_at,
    )
    trees = [
        PairPatternTree(
            pattern=pattern,
            augmented_points=augmented_points,
            terminal_cache=terminal_cache,
            global_discovery=global_discovery,
            config=config,
        )
        for pattern in patterns
    ]
    setup_seconds = time.perf_counter() - setup_started
    global_iteration = 0
    rounds_completed = 0
    for round_index in range(1, config.iterations + 1):
        for tree in trees:
            global_iteration += 1
            tree.run_iteration(round_index, global_iteration)
        rounds_completed = round_index
        if (
            config.checkpoint_interval > 0
            and round_index % config.checkpoint_interval == 0
            and round_index < config.iterations
        ):
            elapsed = time.perf_counter() - started_at
            checkpoint = _build_payload(
                config=config,
                trees=trees,
                global_discovery=global_discovery,
                terminal_cache=terminal_cache,
                setup_seconds=setup_seconds,
                elapsed_seconds=elapsed,
                rounds_completed=rounds_completed,
                status="running",
            )
            _write_outputs(output_path, checkpoint)
            print(
                f"round={round_index}/{config.iterations} "
                f"coverage={len(global_discovery.discovered_classes & set(config.target_classes))} "
                f"class18={'yes' if 18 in global_discovery.discovered_classes else 'no'} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    diagnostics_started = time.perf_counter()
    search_elapsed_seconds = diagnostics_started - started_at
    representability_banks = {
        pattern.pattern_id: PairTargetBank(
            pattern,
            support_class_index,
            config.target_classes,
        )
        for pattern in patterns
    }
    post_search_diagnostics_seconds = time.perf_counter() - diagnostics_started
    elapsed_seconds = time.perf_counter() - started_at
    payload = _build_payload(
        config=config,
        trees=trees,
        global_discovery=global_discovery,
        terminal_cache=terminal_cache,
        setup_seconds=setup_seconds,
        elapsed_seconds=elapsed_seconds,
        rounds_completed=rounds_completed,
        status="complete",
        representability_banks=representability_banks,
        post_search_diagnostics_seconds=post_search_diagnostics_seconds,
        search_elapsed_seconds=search_elapsed_seconds,
    )
    runs_path = _write_outputs(output_path, payload)
    payload["runs_path"] = str(runs_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search 13-pair, full-rank Bell 3-2-2 supports with 18 involution trees."
        )
    )
    parser.add_argument("--initial-class-id", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--exploration-constant", type=float, default=1.4)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--progressive-k0", type=int, default=1)
    parser.add_argument("--progressive-alpha", type=float, default=1.0)
    parser.add_argument("--progressive-beta", type=float, default=0.5)
    parser.add_argument("--epoch-value-decay", type=float, default=0.25)
    parser.add_argument("--new-class-reward", type=float, default=100.0)
    parser.add_argument("--known-class-reward", type=float, default=10.0)
    parser.add_argument("--invalid-reward", type=float, default=-10.0)
    parser.add_argument("--dead-end-reward", type=float, default=-25.0)
    parser.add_argument("--full-rank-tol", type=float, default=1e-9)
    parser.add_argument("--terminal-cache-max-entries", type=int, default=300_000)
    parser.add_argument("--rank19-lookahead-candidate-pool", type=int, default=8)
    parser.add_argument("--corrector-interval", type=int, default=8)
    parser.add_argument("--corrector-max-events-per-tree", type=int, default=24)
    parser.add_argument("--corrector-source-pool", type=int, default=8)
    parser.add_argument("--disable-corrector", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--pattern-ids", type=int, nargs="*")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> PairSearchConfig:
    return PairSearchConfig(
        initial_class_id=int(args.initial_class_id),
        iterations=int(args.iterations),
        target_classes=tuple(int(value) for value in args.target_classes),
        examples_path=args.examples,
        seed=int(args.seed),
        exploration_constant=float(args.exploration_constant),
        discount=float(args.discount),
        progressive_k0=int(args.progressive_k0),
        progressive_alpha=float(args.progressive_alpha),
        progressive_beta=float(args.progressive_beta),
        epoch_value_decay=float(args.epoch_value_decay),
        new_class_reward=float(args.new_class_reward),
        known_class_reward=float(args.known_class_reward),
        invalid_reward=float(args.invalid_reward),
        dead_end_reward=float(args.dead_end_reward),
        full_rank_tol=float(args.full_rank_tol),
        terminal_cache_max_entries=int(args.terminal_cache_max_entries),
        rank19_lookahead_candidate_pool=int(args.rank19_lookahead_candidate_pool),
        corrector_enabled=not bool(args.disable_corrector),
        corrector_interval=int(args.corrector_interval),
        corrector_max_events_per_tree=int(args.corrector_max_events_per_tree),
        corrector_source_pool=int(args.corrector_source_pool),
        checkpoint_interval=int(args.checkpoint_interval),
        pattern_ids=(
            None
            if args.pattern_ids is None
            else tuple(int(value) for value in args.pattern_ids)
        ),
        baseline_path=args.baseline,
    )


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    output_path = args.output or Path(
        "src/mcts/runs/"
        f"pair_involution_search_class{config.initial_class_id}_i{config.iterations}.json"
    )
    payload = run_pair_involution_search(config, output_path=output_path)
    summary = payload["summary"]
    result = {
        "elapsed_seconds": payload["meta"]["elapsed_seconds"],
        "coverage_count": summary["coverage_count"],
        "coverage_class_ids": summary["coverage_class_ids"],
        "search_discovered_class_ids": summary["search_discovered_class_ids"],
        "class18_found_by_pair_search": summary["class18_found_by_pair_search"],
        "combined_coverage_count": (
            payload.get("baseline_comparison", {}).get("combined_coverage_count")
            if isinstance(payload.get("baseline_comparison"), dict)
            else None
        ),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print(output_path, flush=True)
    print(output_path.with_name(f"{output_path.stem}.runs.json"), flush=True)


if __name__ == "__main__":
    main()
