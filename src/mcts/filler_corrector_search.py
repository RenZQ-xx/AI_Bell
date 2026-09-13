from __future__ import annotations

"""Paper-inspired block Filler-Corrector search.

The Filler is the existing add-only interruptible MCTS. The Corrector maps the
paper's retained-row action to orbit blocks: it protects a leading high-rank
core, removes one or more non-core blocks from an exact terminal state, and
lets an independently seeded MCTS window repack the corrected prefix.

Paper: https://arxiv.org/abs/2511.13391

This module deliberately uses an online contextual-bandit Corrector rather
than the paper's trained PPO policy. Candidate deletions are evaluated against
an explicit no-op arm, and the shared policy learns only from observed search
outcomes instead of undiscovered-class examples.
"""

import argparse
import copy
import itertools
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Sequence

import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from baseline.orbit_blocks import (
    BlockKey,
    add_block,
    empty_key,
    selected_blocks,
)
from baseline.reference_classes import DEFAULT_EXAMPLES_PATH
from baseline.scorer import ExpansionScorer, exact_class_id
from mcts.interrupt_search import (
    InterruptGlobalDiscoveryState,
    InterruptSearchConfig,
    InterruptSearchReport,
    InterruptSearchState,
    InterruptSearchTask,
    _default_task_factory,
    _global_discovery_epoch,
    _productive_iteration_limit,
    _run_one_iteration,
    _should_stop_as_terminal_sink,
    _should_stop_for_exact_frequency,
    _should_stop_productive_for_no_novelty,
    run_interruptible_search,
)
from mcts.search import (
    ExactClassDiscovery,
    MCTSNode,
    TerminalHit,
    _get_or_create_node,
)


DEFAULT_BASELINE_PATH = Path(
    "src/mcts/runs/interrupt_search_probe_i200_productive_weighted_rerun.json"
)


@dataclass(frozen=True)
class CorrectorConfig:
    """Controls the remove-and-repack Corrector policy."""

    protected_rank: int = 8
    min_corrected_rank: int = 23
    max_corrected_rank: int = 24
    terminal_rank: int = 25
    max_remove_blocks: int = 2
    candidate_pool: int = 12
    start_iteration: int = 200
    stagnation_iterations: int = 80
    window_iterations: int = 48
    cooldown_iterations: int = 80
    max_corrections_per_task: int = 3
    max_corrections_per_sink_task: int = 1
    new_class_reward: float = 100.0
    distinct_exact_reward: float = 2.0
    size_delta_weight: float = 1.0
    wall_time_penalty: float = 0.01
    tournament_candidates: int = 4
    tournament_pilot_iterations: int = 4
    tournament_finalists: int = 2
    tournament_finalist_iterations: int = 8
    ucb_exploration: float = 1.0
    ucb_ridge: float = 1.0
    discovery_epoch_decay: float = 0.5
    reward_clip: float = 20.0
    new_basin_reward: float = 0.0
    basin_novelty_weight: float = 0.0
    repeated_basin_penalty: float = 0.0
    invalid_terminal_penalty: float = 0.0
    basin_trigger_min_exact_hits: int = 0
    basin_stagnation_iterations: int = 0
    basin_trigger_duplicate_ratio: float = 1.0
    terminal_basin_reservoir_size: int = 0
    source_basin_round_robin: bool = False
    persist_corrected_root: bool = False
    lightweight_tournament_arms: bool = False
    basin_signature_mode: str = "support"
    rank_stratified_candidates: bool = False
    lookahead_new_class_reward: float = 0.0
    lookahead_entropy_weight: float = 0.0
    lookahead_dominant_class_penalty: float = 0.0
    commit_requires_global_novelty: bool = False
    global_iteration_budget: int = 0
    max_concurrent_tournaments: int = 0

    def __post_init__(self) -> None:
        if self.protected_rank < 0:
            raise ValueError("protected_rank must be nonnegative")
        if self.min_corrected_rank > self.max_corrected_rank:
            raise ValueError("min_corrected_rank cannot exceed max_corrected_rank")
        if self.max_corrected_rank >= self.terminal_rank:
            raise ValueError("corrected states must remain below terminal_rank")
        if self.max_remove_blocks <= 0:
            raise ValueError("max_remove_blocks must be positive")
        if self.candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        if self.window_iterations <= 0:
            raise ValueError("window_iterations must be positive")
        if self.tournament_candidates <= 0:
            raise ValueError("tournament_candidates must be positive")
        if self.tournament_pilot_iterations <= 0:
            raise ValueError("tournament_pilot_iterations must be positive")
        if not 1 <= self.tournament_finalists <= self.tournament_candidates + 1:
            raise ValueError("tournament_finalists must include between one and all tournament arms")
        if self.tournament_finalist_iterations < 0:
            raise ValueError("tournament_finalist_iterations must be nonnegative")
        if self.ucb_exploration < 0.0:
            raise ValueError("ucb_exploration must be nonnegative")
        if self.ucb_ridge <= 0.0:
            raise ValueError("ucb_ridge must be positive")
        if not 0.0 <= self.discovery_epoch_decay <= 1.0:
            raise ValueError("discovery_epoch_decay must be in [0, 1]")
        if self.wall_time_penalty < 0.0:
            raise ValueError("wall_time_penalty must be nonnegative")
        if self.reward_clip <= 0.0:
            raise ValueError("reward_clip must be positive")
        if self.new_basin_reward < 0.0:
            raise ValueError("new_basin_reward must be nonnegative")
        if self.basin_novelty_weight < 0.0:
            raise ValueError("basin_novelty_weight must be nonnegative")
        if self.repeated_basin_penalty < 0.0:
            raise ValueError("repeated_basin_penalty must be nonnegative")
        if self.invalid_terminal_penalty < 0.0:
            raise ValueError("invalid_terminal_penalty must be nonnegative")
        if self.basin_trigger_min_exact_hits < 0:
            raise ValueError("basin_trigger_min_exact_hits must be nonnegative")
        if self.basin_stagnation_iterations < 0:
            raise ValueError("basin_stagnation_iterations must be nonnegative")
        if not 0.0 <= self.basin_trigger_duplicate_ratio <= 1.0:
            raise ValueError("basin_trigger_duplicate_ratio must be in [0, 1]")
        if self.terminal_basin_reservoir_size < 0:
            raise ValueError("terminal_basin_reservoir_size must be nonnegative")
        if self.basin_signature_mode not in {"support", "class_transition"}:
            raise ValueError(
                "basin_signature_mode must be 'support' or 'class_transition'"
            )
        if self.lookahead_new_class_reward < 0.0:
            raise ValueError("lookahead_new_class_reward must be nonnegative")
        if self.lookahead_entropy_weight < 0.0:
            raise ValueError("lookahead_entropy_weight must be nonnegative")
        if self.lookahead_dominant_class_penalty < 0.0:
            raise ValueError("lookahead_dominant_class_penalty must be nonnegative")
        if self.global_iteration_budget < 0:
            raise ValueError("global_iteration_budget must be nonnegative")
        if self.max_concurrent_tournaments < 0:
            raise ValueError("max_concurrent_tournaments must be nonnegative")
        if (
            self.lightweight_tournament_arms
            and not self.persist_corrected_root
            and not self.commit_requires_global_novelty
        ):
            raise ValueError(
                "lightweight_tournament_arms requires a persistent or novelty-gated root"
            )


@dataclass(frozen=True)
class CorrectionProposal:
    """One retained-block action selected by the Corrector."""

    source_label: str
    source_key: BlockKey
    source_rank: int
    protected_blocks: tuple[int, ...]
    removed_blocks: tuple[int, ...]
    corrected_key: BlockKey
    corrected_rank: int
    compatible_undiscovered_classes: tuple[int, ...]
    entrance_rare_count: int
    entrance_valid_count: int
    entrance_invalid_count: int
    entrance_class_entropy: float
    entrance_dominant_class_ratio: float
    policy_score: float
    context_features: tuple[float, ...] = field(default_factory=tuple)
    policy_mean: float = 0.0
    policy_uncertainty: float = 0.0


@dataclass
class CorrectorEvent:
    """Logged remove-and-repack transition."""

    event_index: int
    search_index: int
    start_class_id: int
    correction_index: int
    trigger_iteration: int
    filler_start_iteration: int
    source_label: str
    source_rank: int
    source_blocks: list[int]
    source_vertex_count: int
    protected_blocks: list[int]
    removed_blocks: list[int]
    corrected_blocks: list[int]
    corrected_rank: int
    corrected_vertex_count: int
    compatible_undiscovered_classes: list[int]
    entrance_rare_count: int
    entrance_valid_count: int
    entrance_invalid_count: int
    entrance_class_entropy: float
    entrance_dominant_class_ratio: float
    policy_score: float
    filler_seed: int
    planned_filler_iterations: int
    finished_iteration: int | None = None
    completed_filler_iterations: int = 0
    repacked_label: str | None = None
    repacked_rank: int | None = None
    repacked_blocks: list[int] = field(default_factory=list)
    repacked_vertex_count: int | None = None
    exact_hit_count: int = 0
    new_global_class_ids: list[int] = field(default_factory=list)
    size_delta_reward: float | None = None
    coverage_augmented_reward: float | None = None
    counterfactual_advantage: float | None = None
    selected_arm: str | None = None
    committed_arm: str | None = None
    proxy_winner_rejected: bool = False
    arm_evaluations: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "search_index": self.search_index,
            "start_class_id": self.start_class_id,
            "correction_index": self.correction_index,
            "trigger_iteration": self.trigger_iteration,
            "filler_start_iteration": self.filler_start_iteration,
            "source_label": self.source_label,
            "source_rank": self.source_rank,
            "source_blocks": list(self.source_blocks),
            "source_vertex_count": self.source_vertex_count,
            "protected_blocks": list(self.protected_blocks),
            "removed_blocks": list(self.removed_blocks),
            "corrected_blocks": list(self.corrected_blocks),
            "corrected_rank": self.corrected_rank,
            "corrected_vertex_count": self.corrected_vertex_count,
            "compatible_undiscovered_classes": list(self.compatible_undiscovered_classes),
            "entrance_rare_count": self.entrance_rare_count,
            "entrance_valid_count": self.entrance_valid_count,
            "entrance_invalid_count": self.entrance_invalid_count,
            "entrance_class_entropy": self.entrance_class_entropy,
            "entrance_dominant_class_ratio": self.entrance_dominant_class_ratio,
            "policy_score": self.policy_score,
            "filler_seed": self.filler_seed,
            "planned_filler_iterations": self.planned_filler_iterations,
            "finished_iteration": self.finished_iteration,
            "completed_filler_iterations": self.completed_filler_iterations,
            "repacked_label": self.repacked_label,
            "repacked_rank": self.repacked_rank,
            "repacked_blocks": list(self.repacked_blocks),
            "repacked_vertex_count": self.repacked_vertex_count,
            "exact_hit_count": self.exact_hit_count,
            "new_global_class_ids": list(self.new_global_class_ids),
            "size_delta_reward": self.size_delta_reward,
            "coverage_augmented_reward": self.coverage_augmented_reward,
            "counterfactual_advantage": self.counterfactual_advantage,
            "selected_arm": self.selected_arm,
            "committed_arm": self.committed_arm,
            "proxy_winner_rejected": self.proxy_winner_rejected,
            "arm_evaluations": list(self.arm_evaluations),
        }


@dataclass(frozen=True)
class _CorrectionCandidate:
    source: TerminalHit
    protected_blocks: tuple[int, ...]
    removed_blocks: tuple[int, ...]
    corrected_key: BlockKey
    corrected_rank: int
    compatible_undiscovered_classes: tuple[int, ...]
    cheap_score: float
    source_frequency: float
    source_vertex_count: int
    removed_vertex_count: int


@dataclass
class _CorrectionArm:
    name: str
    proposal: CorrectionProposal | None
    nodes: dict[BlockKey, MCTSNode]
    root: MCTSNode
    resume_root: MCTSNode
    rng: random.Random
    reference_vertex_count: int
    seen_signatures: Counter[tuple[int, ...]] = field(default_factory=Counter)
    synchronized_discovery_epoch: int = -1
    rollouts_started: int = 0
    completed_iterations: int = 0
    elapsed_seconds: float = 0.0
    exact_labels: set[str] = field(default_factory=set)
    new_global_class_ids: list[int] = field(default_factory=list)
    exact_hit_count: int = 0
    best_hit: TerminalHit | None = None
    basin_counts: Counter[int] = field(default_factory=Counter)
    new_basin_count: int = 0
    repeated_basin_count: int = 0
    basin_novelty_sum: float = 0.0
    invalid_terminal_count: int = 0

    def utility_rate(
        self,
        scorer: ExpansionScorer,
        config: CorrectorConfig,
    ) -> float:
        repacked_vertex_count = self.reference_vertex_count
        if self.best_hit is not None:
            repacked_vertex_count = _vertex_count(scorer, self.best_hit.key)
        size_delta = (
            0
            if self.proposal is None
            else repacked_vertex_count - self.reference_vertex_count
        )
        raw = (
            float(config.new_class_reward) * len(self.new_global_class_ids)
            + float(config.distinct_exact_reward) * len(self.exact_labels)
            + float(config.size_delta_weight) * float(size_delta)
            + float(config.new_basin_reward) * float(self.new_basin_count)
            + float(config.basin_novelty_weight) * float(self.basin_novelty_sum)
            - float(config.repeated_basin_penalty)
            * float(self.repeated_basin_count)
            - float(config.invalid_terminal_penalty)
            * float(self.invalid_terminal_count)
        )
        iterations = max(1, int(self.completed_iterations))
        return (
            raw / float(iterations)
            - float(config.wall_time_penalty)
            * float(self.elapsed_seconds)
            / float(iterations)
        )

    def to_dict(
        self,
        scorer: ExpansionScorer,
        config: CorrectorConfig,
    ) -> dict[str, object]:
        repacked_vertex_count = self.reference_vertex_count
        repacked_label: str | None = None
        if self.best_hit is not None:
            repacked_vertex_count = _vertex_count(scorer, self.best_hit.key)
            repacked_label = self.best_hit.label
        return {
            "name": self.name,
            "removed_blocks": (
                [] if self.proposal is None else list(self.proposal.removed_blocks)
            ),
            "corrected_rank": (
                None if self.proposal is None else int(self.proposal.corrected_rank)
            ),
            "completed_iterations": int(self.completed_iterations),
            "elapsed_seconds": float(self.elapsed_seconds),
            "exact_hit_count": int(self.exact_hit_count),
            "exact_labels": sorted(self.exact_labels),
            "new_global_class_ids": sorted(self.new_global_class_ids),
            "new_basin_count": int(self.new_basin_count),
            "repeated_basin_count": int(self.repeated_basin_count),
            "basin_novelty_sum": float(self.basin_novelty_sum),
            "invalid_terminal_count": int(self.invalid_terminal_count),
            "distinct_basin_count": len(self.basin_counts),
            "reference_vertex_count": int(self.reference_vertex_count),
            "repacked_vertex_count": int(repacked_vertex_count),
            "repacked_label": repacked_label,
            "utility_rate": float(self.utility_rate(scorer, config)),
            "policy_score": (
                0.0 if self.proposal is None else float(self.proposal.policy_score)
            ),
            "policy_mean": (
                0.0 if self.proposal is None else float(self.proposal.policy_mean)
            ),
            "policy_uncertainty": (
                0.0
                if self.proposal is None
                else float(self.proposal.policy_uncertainty)
            ),
        }


def remove_blocks(key: BlockKey, block_indices: Sequence[int]) -> BlockKey:
    """Return a key with the requested selected blocks removed."""
    values = [int(value) for value in key]
    for raw_index in block_indices:
        index = int(raw_index)
        if index < 0 or index >= len(values):
            raise IndexError(f"block_index {index} out of range for {len(values)} blocks")
        values[index] = 0
    return tuple(values)


def _vertex_count(scorer: ExpansionScorer, key: BlockKey) -> int:
    return sum(len(scorer.blocks[index]) for index in selected_blocks(key))


def _leading_rank_core(
    scorer: ExpansionScorer,
    source: TerminalHit,
    target_rank: int,
) -> tuple[int, ...]:
    """Protect the leading rank-building blocks from the source trajectory."""
    if target_rank <= 0:
        return ()
    selected = set(selected_blocks(source.key))
    ordered = [int(action) for action in source.path if int(action) in selected]
    ordered.extend(sorted(selected - set(ordered)))
    core_key = empty_key(len(source.key))
    core: list[int] = []
    current_rank = scorer.affine_rank(core_key)
    for action in ordered:
        candidate = add_block(core_key, action)
        new_rank = scorer.affine_rank(candidate)
        if new_rank <= current_rank:
            continue
        core_key = candidate
        core.append(action)
        current_rank = new_rank
        if current_rank >= target_rank:
            break
    return tuple(sorted(core))


def _clone_search_graph(
    nodes: dict[BlockKey, MCTSNode],
    root: MCTSNode,
) -> tuple[dict[BlockKey, MCTSNode], MCTSNode]:
    """Clone one task-local MCTS graph while preserving its internal links."""
    cloned_nodes = copy.deepcopy(nodes)
    cloned_root = cloned_nodes.get(root.key)
    if cloned_root is None:
        raise RuntimeError("search root is missing from its node table")
    return cloned_nodes, cloned_root


CORRECTOR_FEATURE_NAMES = (
    "bias",
    "corrected_rank",
    "rank_loss",
    "remove_fraction",
    "removed_vertex_fraction",
    "source_vertex_fraction",
    "protected_fraction",
    "remaining_block_fraction",
    "source_frequency",
    "valid_entrance_fraction",
    "invalid_entrance_fraction",
    "unseen_entrance_fraction",
    "entrance_class_entropy",
)


class SharedCorrectorPolicy:
    """Thread-safe contextual UCB policy shared by every Corrector task."""

    def __init__(self, config: CorrectorConfig) -> None:
        self.config = config
        dimension = len(CORRECTOR_FEATURE_NAMES)
        self._matrix = np.eye(dimension, dtype=float) * float(config.ucb_ridge)
        self._target = np.zeros(dimension, dtype=float)
        self._lock = RLock()
        self._epoch = -1
        self.observations = 0
        self.reward_sum = 0.0
        self._basin_counts: Counter[int] = Counter()
        self._active_tournaments = 0
        self._active_tournament_peak = 0
        self._reserved_iterations = 0
        self._completed_corrector_iterations = 0
        self._denied_tournaments = 0

    def _synchronize_epoch(self, discovery_epoch: int) -> None:
        normalized = int(discovery_epoch)
        if self._epoch < 0:
            self._epoch = normalized
            return
        if normalized <= self._epoch:
            return
        decay = float(self.config.discovery_epoch_decay) ** (normalized - self._epoch)
        ridge = float(self.config.ucb_ridge)
        identity = np.eye(len(CORRECTOR_FEATURE_NAMES), dtype=float) * ridge
        self._matrix = identity + decay * (self._matrix - identity)
        self._target *= decay
        self._epoch = normalized

    def evaluate(
        self,
        features: Sequence[float],
        *,
        discovery_epoch: int,
    ) -> tuple[float, float, float]:
        vector = np.asarray(tuple(float(value) for value in features), dtype=float)
        if vector.shape != (len(CORRECTOR_FEATURE_NAMES),):
            raise ValueError("invalid Corrector feature vector")
        with self._lock:
            self._synchronize_epoch(discovery_epoch)
            solved = np.linalg.solve(self._matrix, vector)
            theta = np.linalg.solve(self._matrix, self._target)
            mean = float(vector @ theta)
            uncertainty = math.sqrt(max(0.0, float(vector @ solved)))
            score = mean + float(self.config.ucb_exploration) * uncertainty
            return mean, uncertainty, score

    def observe(
        self,
        features: Sequence[float],
        reward: float,
        *,
        discovery_epoch: int,
    ) -> None:
        vector = np.asarray(tuple(float(value) for value in features), dtype=float)
        clipped = max(
            -float(self.config.reward_clip),
            min(float(self.config.reward_clip), float(reward)),
        )
        with self._lock:
            self._synchronize_epoch(discovery_epoch)
            self._matrix += np.outer(vector, vector)
            self._target += clipped * vector
            self.observations += 1
            self.reward_sum += clipped

    def observe_basin(self, signature: int) -> tuple[bool, float]:
        """Record one group-normalized basin in the shared online archive."""

        normalized = int(signature)
        with self._lock:
            previous = int(self._basin_counts[normalized])
            self._basin_counts[normalized] += 1
        return previous == 0, 1.0 / math.sqrt(1.0 + float(previous))

    def try_begin_tournament(self, planned_iterations: int) -> bool:
        """Reserve bounded shared Corrector capacity for one task."""

        planned = max(0, int(planned_iterations))
        with self._lock:
            concurrency_limit = int(self.config.max_concurrent_tournaments)
            budget_limit = int(self.config.global_iteration_budget)
            over_concurrency = (
                concurrency_limit > 0
                and self._active_tournaments >= concurrency_limit
            )
            over_budget = (
                budget_limit > 0
                and self._completed_corrector_iterations
                + self._reserved_iterations
                + planned
                > budget_limit
            )
            if over_concurrency or over_budget:
                self._denied_tournaments += 1
                return False
            self._active_tournaments += 1
            self._active_tournament_peak = max(
                self._active_tournament_peak,
                self._active_tournaments,
            )
            self._reserved_iterations += planned
            return True

    def finish_tournament(self, planned_iterations: int, actual_iterations: int) -> None:
        planned = max(0, int(planned_iterations))
        actual = max(0, int(actual_iterations))
        with self._lock:
            self._active_tournaments = max(0, self._active_tournaments - 1)
            self._reserved_iterations = max(0, self._reserved_iterations - planned)
            self._completed_corrector_iterations += actual

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "feature_names": list(CORRECTOR_FEATURE_NAMES),
                "observations": int(self.observations),
                "reward_sum": float(self.reward_sum),
                "mean_reward": (
                    0.0
                    if self.observations <= 0
                    else float(self.reward_sum) / float(self.observations)
                ),
                "discovery_epoch": int(self._epoch),
                "shared_basin_count": len(self._basin_counts),
                "shared_basin_observations": int(sum(self._basin_counts.values())),
                "active_tournaments": int(self._active_tournaments),
                "active_tournament_peak": int(self._active_tournament_peak),
                "completed_corrector_iterations": int(
                    self._completed_corrector_iterations
                ),
                "reserved_corrector_iterations": int(self._reserved_iterations),
                "denied_tournaments": int(self._denied_tournaments),
            }


class BlockCorrector:
    """Prior-clean structural proposal policy with shared contextual feedback."""

    def __init__(
        self,
        config: CorrectorConfig,
        shared_policy: SharedCorrectorPolicy | None = None,
    ) -> None:
        self.config = config
        self.shared_policy = shared_policy or SharedCorrectorPolicy(config)
        self.tried_corrected_keys: set[BlockKey] = set()
        self._source_cursor = 0

    def observe(
        self,
        context_features: Sequence[float],
        reward: float,
        *,
        discovery_epoch: int = 0,
    ) -> None:
        self.shared_policy.observe(
            context_features,
            reward,
            discovery_epoch=discovery_epoch,
        )

    def _features(
        self,
        state: InterruptSearchState,
        candidate: _CorrectionCandidate,
        entrance: dict[str, float],
    ) -> tuple[float, ...]:
        source_blocks = selected_blocks(candidate.source.key)
        corrected_blocks = selected_blocks(candidate.corrected_key)
        valid_count = int(
            entrance["rare"] + entrance["class44"] + entrance["other_valid"]
        )
        entrance_total = max(1, valid_count + int(entrance["invalid"]))
        frequency_scale = math.log1p(candidate.source_frequency)
        return (
            1.0,
            float(candidate.corrected_rank) / float(max(1, self.config.terminal_rank)),
            float(candidate.source.rank - candidate.corrected_rank)
            / float(max(1, self.config.terminal_rank)),
            float(len(candidate.removed_blocks))
            / float(max(1, self.config.max_remove_blocks)),
            float(candidate.removed_vertex_count) / 64.0,
            float(candidate.source_vertex_count) / 64.0,
            float(len(candidate.protected_blocks)) / float(max(1, len(source_blocks))),
            float(len(corrected_blocks)) / float(max(1, len(state.scorer.blocks))),
            frequency_scale / (1.0 + frequency_scale),
            float(valid_count) / float(entrance_total),
            float(entrance["invalid"]) / float(entrance_total),
            float(entrance.get("unseen", 0.0)) / float(entrance_total),
            float(entrance.get("class_entropy", 0.0)),
        )

    def propose(
        self,
        state: InterruptSearchState,
        *,
        rng: random.Random,
    ) -> CorrectionProposal | None:
        proposals = self.propose_candidates(state, rng=rng, limit=1)
        return None if not proposals else proposals[0]

    def propose_candidates(
        self,
        state: InterruptSearchState,
        *,
        rng: random.Random,
        limit: int,
    ) -> list[CorrectionProposal]:
        scorer = state.scorer
        exact_sources = [
            hit
            for label, hit in sorted(state.terminal_bests.items())
            if exact_class_id(label) is not None
            and int(hit.rank) >= self.config.terminal_rank
        ]
        if self.config.terminal_basin_reservoir_size > 0:
            exact_sources.extend(
                hit
                for hit in getattr(state, "corrector_terminal_sources", ())
                if exact_class_id(hit.label) is not None
                and int(hit.rank) >= self.config.terminal_rank
            )
            exact_sources = list(
                {
                    (hit.label, hit.key): hit
                    for hit in exact_sources
                }.values()
            )
        candidates: list[_CorrectionCandidate] = []
        for source in exact_sources:
            protected = _leading_rank_core(scorer, source, self.config.protected_rank)
            protected_set = set(protected)
            deletable = [
                block
                for block in selected_blocks(source.key)
                if block not in protected_set
            ]
            source_frequency = float(
                state.global_discovery.discovered_label_counts.get(source.label, 0)
            )
            source_vertex_count = _vertex_count(scorer, source.key)
            for remove_count in range(
                1,
                min(self.config.max_remove_blocks, len(deletable)) + 1,
            ):
                for removed in itertools.combinations(deletable, remove_count):
                    corrected_key = remove_blocks(source.key, removed)
                    if corrected_key in self.tried_corrected_keys:
                        continue
                    corrected_rank = int(scorer.affine_rank(corrected_key))
                    if not (
                        self.config.min_corrected_rank
                        <= corrected_rank
                        <= self.config.max_corrected_rank
                    ):
                        continue
                    removed_vertices = sum(
                        len(scorer.blocks[index]) for index in removed
                    )
                    cheap_score = (
                        float(corrected_rank) / float(self.config.terminal_rank)
                        + 0.05 * math.log1p(source_frequency)
                        - 0.10 * float(removed_vertices) / 64.0
                    )
                    candidates.append(
                        _CorrectionCandidate(
                            source=source,
                            protected_blocks=protected,
                            removed_blocks=tuple(int(value) for value in removed),
                            corrected_key=corrected_key,
                            corrected_rank=corrected_rank,
                            compatible_undiscovered_classes=(),
                            cheap_score=float(cheap_score),
                            source_frequency=source_frequency,
                            source_vertex_count=source_vertex_count,
                            removed_vertex_count=removed_vertices,
                        )
                    )
        if not candidates:
            return []

        by_corrected_key: dict[BlockKey, _CorrectionCandidate] = {}
        for candidate in candidates:
            previous = by_corrected_key.get(candidate.corrected_key)
            if previous is None or (
                candidate.cheap_score,
                -len(candidate.removed_blocks),
                candidate.removed_blocks,
            ) > (
                previous.cheap_score,
                -len(previous.removed_blocks),
                previous.removed_blocks,
            ):
                by_corrected_key[candidate.corrected_key] = candidate
        candidates = list(by_corrected_key.values())

        by_source: dict[object, list[_CorrectionCandidate]] = {}
        for candidate in candidates:
            source_key: object = candidate.source.label
            if self.config.source_basin_round_robin:
                source_key = (candidate.source.label, candidate.source.key)
            by_source.setdefault(source_key, []).append(candidate)
        for source_candidates in by_source.values():
            source_candidates.sort(
                key=lambda item: (
                    item.cheap_score,
                    item.corrected_rank,
                    -len(item.removed_blocks),
                    item.removed_blocks,
                ),
                reverse=True,
            )
        source_labels = sorted(by_source, key=str)
        cursor = self._source_cursor % len(source_labels)
        rotated_labels = source_labels[cursor:] + source_labels[:cursor]
        shortlist: list[_CorrectionCandidate] = []
        shortlisted_keys: set[BlockKey] = set()
        if self.config.rank_stratified_candidates:
            for rank in range(
                int(self.config.max_corrected_rank),
                int(self.config.min_corrected_rank) - 1,
                -1,
            ):
                rank_candidates = [
                    candidate
                    for candidate in candidates
                    if int(candidate.corrected_rank) == rank
                ]
                if not rank_candidates:
                    continue
                chosen = max(
                    rank_candidates,
                    key=lambda item: (
                        item.cheap_score,
                        -len(item.removed_blocks),
                        item.removed_blocks,
                    ),
                )
                shortlist.append(chosen)
                shortlisted_keys.add(chosen.corrected_key)
                if len(shortlist) >= self.config.candidate_pool:
                    break
        round_index = 0
        max_source_candidates = max(len(values) for values in by_source.values())
        while len(shortlist) < self.config.candidate_pool:
            added = False
            for label in rotated_labels:
                source_candidates = by_source[label]
                if round_index >= len(source_candidates):
                    continue
                candidate = source_candidates[round_index]
                if candidate.corrected_key in shortlisted_keys:
                    continue
                shortlist.append(candidate)
                shortlisted_keys.add(candidate.corrected_key)
                added = True
                if len(shortlist) >= self.config.candidate_pool:
                    break
            round_index += 1
            if not added and round_index >= max_source_candidates:
                break
        self._source_cursor = (cursor + max(1, len(shortlist))) % len(source_labels)

        evaluated: list[
            tuple[
                float,
                _CorrectionCandidate,
                dict[str, float],
                tuple[float, ...],
                float,
                float,
                tuple[int, ...],
            ]
        ] = []
        discovery_epoch = _global_discovery_epoch(state.global_discovery)
        discovered_snapshot = getattr(
            state.global_discovery,
            "discovered_classes_snapshot",
            None,
        )
        discovered_classes = (
            set(int(value) for value in discovered_snapshot())
            if callable(discovered_snapshot)
            else set(
                int(value)
                for value in getattr(
                    state.global_discovery,
                    "discovered_exact_classes",
                    (),
                )
            )
        )
        for candidate in shortlist:
            entrance: dict[str, float] = {
                "rare": 0.0,
                "class44": 0.0,
                "other_valid": 0.0,
                "invalid": 0.0,
                "unseen": 0.0,
                "class_entropy": 0.0,
                "dominant_ratio": 0.0,
            }
            unseen_classes: tuple[int, ...] = ()
            if candidate.corrected_rank == self.config.max_corrected_rank:
                entrance.update(scorer.rank24_entrance_metrics(candidate.corrected_key))
                class_count_fn = getattr(scorer, "rank24_entrance_class_counts", None)
                class_counts = (
                    class_count_fn(candidate.corrected_key)[0]
                    if callable(class_count_fn)
                    else {}
                )
                unseen_classes = tuple(
                    sorted(
                        int(class_id)
                        for class_id in class_counts
                        if int(class_id) not in discovered_classes
                    )
                )
                entrance["unseen"] = float(
                    sum(class_counts[class_id] for class_id in unseen_classes)
                )
                total_exits = int(sum(class_counts.values()))
                if total_exits > 0:
                    probabilities = [
                        float(count) / float(total_exits)
                        for count in class_counts.values()
                    ]
                    entropy = -sum(
                        probability * math.log(probability)
                        for probability in probabilities
                        if probability > 0.0
                    )
                    if len(probabilities) > 1:
                        entropy /= math.log(float(len(probabilities)))
                    entrance["class_entropy"] = float(entropy)
                    entrance["dominant_ratio"] = max(probabilities)
            valid_count = int(
                entrance["rare"] + entrance["class44"] + entrance["other_valid"]
            )
            features = self._features(state, candidate, entrance)
            policy_mean, policy_uncertainty, learned_score = self.shared_policy.evaluate(
                features,
                discovery_epoch=discovery_epoch,
            )
            policy_score = (
                candidate.cheap_score
                + 0.50 * float(valid_count)
                - 0.10 * float(entrance["invalid"])
                + float(self.config.lookahead_new_class_reward)
                * float(len(unseen_classes))
                + float(self.config.lookahead_entropy_weight)
                * float(entrance["class_entropy"])
                - float(self.config.lookahead_dominant_class_penalty)
                * float(entrance["dominant_ratio"])
                + learned_score
            )
            evaluated.append(
                (
                    float(policy_score),
                    candidate,
                    entrance,
                    features,
                    policy_mean,
                    policy_uncertainty,
                    unseen_classes,
                )
            )

        rng.shuffle(evaluated)
        evaluated.sort(
            key=lambda item: (
                item[0],
                item[1].corrected_rank,
                -len(item[1].removed_blocks),
            ),
            reverse=True,
        )
        selected_evaluated = evaluated[: max(1, int(limit))]
        if self.config.rank_stratified_candidates:
            selected_evaluated = []
            selected_ranks: set[int] = set()
            for item in evaluated:
                rank = int(item[1].corrected_rank)
                if rank in selected_ranks:
                    continue
                selected_evaluated.append(item)
                selected_ranks.add(rank)
                if len(selected_evaluated) >= max(1, int(limit)):
                    break
            if len(selected_evaluated) < max(1, int(limit)):
                selected_keys = {
                    item[1].corrected_key for item in selected_evaluated
                }
                for item in evaluated:
                    if item[1].corrected_key in selected_keys:
                        continue
                    selected_evaluated.append(item)
                    selected_keys.add(item[1].corrected_key)
                    if len(selected_evaluated) >= max(1, int(limit)):
                        break

        proposals: list[CorrectionProposal] = []
        for (
            policy_score,
            chosen,
            entrance,
            features,
            policy_mean,
            uncertainty,
            unseen_classes,
        ) in selected_evaluated:
            self.tried_corrected_keys.add(chosen.corrected_key)
            proposals.append(
                CorrectionProposal(
                    source_label=chosen.source.label,
                    source_key=chosen.source.key,
                    source_rank=int(chosen.source.rank),
                    protected_blocks=chosen.protected_blocks,
                    removed_blocks=chosen.removed_blocks,
                    corrected_key=chosen.corrected_key,
                    corrected_rank=chosen.corrected_rank,
                    compatible_undiscovered_classes=unseen_classes,
                    entrance_rare_count=int(entrance["rare"]),
                    entrance_valid_count=int(
                        entrance["rare"]
                        + entrance["class44"]
                        + entrance["other_valid"]
                    ),
                    entrance_invalid_count=int(entrance["invalid"]),
                    entrance_class_entropy=float(entrance["class_entropy"]),
                    entrance_dominant_class_ratio=float(
                        entrance["dominant_ratio"]
                    ),
                    policy_score=float(policy_score),
                    context_features=features,
                    policy_mean=float(policy_mean),
                    policy_uncertainty=float(uncertainty),
                )
            )
        return proposals


class FillerCorrectorTask(InterruptSearchTask):
    """Interrupt task with a shared-policy, counterfactual Corrector tournament."""

    def __init__(
        self,
        base_task: InterruptSearchTask,
        *,
        corrector_config: CorrectorConfig,
        event_sink: list[CorrectorEvent],
        shared_policy: SharedCorrectorPolicy | None = None,
    ) -> None:
        for name, value in vars(base_task).items():
            setattr(self, name, value)
        self.corrector_config = corrector_config
        self.corrector = BlockCorrector(corrector_config, shared_policy)
        self.event_sink = event_sink
        self._active_event: CorrectorEvent | None = None
        self._arms: list[_CorrectionArm] = []
        self._active_arm: _CorrectionArm | None = None
        self._phase_queue: list[tuple[_CorrectionArm, int]] = []
        self._phase_name = ""
        self._active_remaining = 0
        self._planned_iterations = 0
        self._reserved_tournament_iterations = 0
        self._tournament_rng: random.Random | None = None
        self._corrections_started = 0
        self._next_corrector_iteration = 0
        self._basin_counts: Counter[int] = Counter()
        self._last_new_basin_iteration = 0
        self._terminal_basin_reservoir: dict[int, TerminalHit] = {}
        self.state.corrector_terminal_sources = []  # type: ignore[attr-defined]

    @property
    def requires_resident_service(self) -> bool:
        """Keep an in-progress counterfactual tournament on the scheduler."""

        return self._active_event is not None

    @property
    def structural_novelty_count(self) -> int:
        """Expose observed terminal-basis diversity to an optional scheduler."""

        return len(self._basin_counts)

    @property
    def structural_terminal_hit_count(self) -> int:
        return int(sum(self._basin_counts.values()))

    def release_search_resources(self) -> None:
        """Release every tournament graph when an external scheduler parks us."""

        if self._reserved_tournament_iterations > 0:
            completed = sum(arm.completed_iterations for arm in self._arms)
            self.corrector.shared_policy.finish_tournament(
                self._reserved_tournament_iterations,
                completed,
            )
            self._reserved_tournament_iterations = 0
        graphs: dict[int, dict[BlockKey, MCTSNode]] = {
            id(arm.nodes): arm.nodes for arm in self._arms
        }
        graphs[id(self.state.nodes)] = self.state.nodes
        for nodes in graphs.values():
            for node in nodes.values():
                node.parent = None
                node.children.clear()
                node.unexpanded_actions.clear()
                node.action_priors.clear()
                node.action_scores.clear()
            nodes.clear()
        self.state.release_tree()
        self._active_event = None
        self._arms = []
        self._active_arm = None
        self._phase_queue = []
        self._phase_name = ""
        self._active_remaining = 0
        self._planned_iterations = 0
        self._tournament_rng = None

    def _basin_tracking_enabled(self) -> bool:
        config = self.corrector_config
        return bool(
            config.new_basin_reward
            or config.basin_novelty_weight
            or config.repeated_basin_penalty
            or config.basin_trigger_min_exact_hits
            or config.terminal_basin_reservoir_size
        )

    def _basin_signature(self, hit: TerminalHit) -> int:
        """Identify an observed terminal basis without consulting class examples."""

        if self.corrector_config.basin_signature_mode == "class_transition":
            class_id = exact_class_id(hit.label)
            if class_id is not None:
                return 1000 * int(self.class_id) + int(class_id)
        return int(self.state.scorer.support_key(hit.key))

    def _record_basin_hit(
        self,
        counts: Counter[int],
        hit: TerminalHit,
    ) -> tuple[bool, float]:
        signature = self._basin_signature(hit)
        previous = int(counts[signature])
        counts[signature] += 1
        self._remember_terminal_basin(signature, hit)
        if self.corrector_config.basin_signature_mode == "class_transition":
            return self.corrector.shared_policy.observe_basin(signature)
        return previous == 0, 1.0 / math.sqrt(1.0 + float(previous))

    def _remember_terminal_basin(self, signature: int, hit: TerminalHit) -> None:
        limit = int(self.corrector_config.terminal_basin_reservoir_size)
        if limit <= 0:
            return
        existing = self._terminal_basin_reservoir.get(int(signature))
        if existing is None and len(self._terminal_basin_reservoir) >= limit:
            victim = max(
                self._terminal_basin_reservoir,
                key=lambda value: (
                    int(self._basin_counts.get(value, 0)),
                    -int(value),
                ),
            )
            self._terminal_basin_reservoir.pop(victim, None)
        if existing is None or float(hit.score) > float(existing.score):
            self._terminal_basin_reservoir[int(signature)] = hit
        self.state.corrector_terminal_sources = list(  # type: ignore[attr-defined]
            self._terminal_basin_reservoir.values()
        )

    def _basin_trigger_ready(self, iteration: int) -> bool:
        minimum_hits = int(self.corrector_config.basin_trigger_min_exact_hits)
        stagnation = int(self.corrector_config.basin_stagnation_iterations)
        if minimum_hits <= 0 or stagnation <= 0:
            return False
        total_hits = int(sum(self._basin_counts.values()))
        if total_hits < minimum_hits:
            return False
        duplicate_ratio = 1.0 - float(len(self._basin_counts)) / float(total_hits)
        if duplicate_ratio < float(
            self.corrector_config.basin_trigger_duplicate_ratio
        ):
            return False
        return int(iteration) - int(self._last_new_basin_iteration) >= stagnation

    def _correction_limit(self) -> int:
        if self.state.productive:
            return max(0, int(self.corrector_config.max_corrections_per_task))
        return max(0, int(self.corrector_config.max_corrections_per_sink_task))

    def _eligible_for_correction(self) -> bool:
        if self._active_event is not None:
            return False
        if self._corrections_started >= self._correction_limit():
            return False
        iteration = int(self.state.iterations_completed)
        if iteration < self._next_corrector_iteration:
            return False
        if iteration >= int(self.iteration_limit):
            return False
        if not any(
            exact_class_id(label) is not None for label in self.state.terminal_bests
        ):
            return False
        if not self.state.productive:
            return _should_stop_for_exact_frequency(self) or _should_stop_as_terminal_sink(self)
        if self._basin_trigger_ready(iteration):
            return True
        if iteration < int(self.corrector_config.start_iteration):
            return False
        novelty_anchor = int(self.last_new_class_iteration or 0)
        return (
            iteration - novelty_anchor
            >= int(self.corrector_config.stagnation_iterations)
        )

    def _arm_score(self, arm: _CorrectionArm) -> tuple[float, float, str]:
        prior = 0.0 if arm.proposal is None else float(arm.proposal.policy_score)
        if arm.completed_iterations <= 0:
            return (-math.inf, prior, arm.name)
        return (
            arm.utility_rate(self.state.scorer, self.corrector_config),
            prior,
            arm.name,
        )

    def _round_robin_queue(
        self,
        arms: Sequence[_CorrectionArm],
        iterations_per_arm: int,
    ) -> list[tuple[_CorrectionArm, int]]:
        queue: list[tuple[_CorrectionArm, int]] = []
        scheduler_rng = self._tournament_rng
        for _ in range(max(0, int(iterations_per_arm))):
            round_arms = list(arms)
            if scheduler_rng is not None:
                scheduler_rng.shuffle(round_arms)
            queue.extend((arm, 1) for arm in round_arms)
        return queue

    def _store_active_arm_state(self) -> None:
        arm = self._active_arm
        if arm is None:
            return
        arm.nodes = self.state.nodes
        if self.state.root is not None:
            arm.root = self.state.root
        arm.seen_signatures = self.state.seen_signatures
        arm.synchronized_discovery_epoch = int(
            self.state.synchronized_discovery_epoch
        )
        arm.rollouts_started = int(self.state.rollouts_started)
        active_rng = getattr(self.state, "_rng", None)
        if active_rng is not None:
            arm.rng = active_rng

    def _switch_to_arm(self, arm: _CorrectionArm, iterations: int) -> bool:
        remaining = self._planned_iterations - sum(
            item.completed_iterations for item in self._arms
        )
        budget = min(max(0, int(iterations)), max(0, int(remaining)))
        if budget <= 0:
            return False
        self._store_active_arm_state()
        self._active_arm = arm
        self._active_remaining = budget
        self.state.nodes = arm.nodes
        self.state.root = arm.root
        self.state.seen_signatures = arm.seen_signatures
        self.state.synchronized_discovery_epoch = int(
            arm.synchronized_discovery_epoch
        )
        self.state.rollouts_started = int(arm.rollouts_started)
        self.state._rng = arm.rng  # type: ignore[attr-defined]
        return True

    def _advance_tournament(self) -> None:
        if self._active_event is None:
            return
        if (
            self.state.iterations_completed >= self.iteration_limit
            or sum(arm.completed_iterations for arm in self._arms)
            >= self._planned_iterations
        ):
            self._finish_correction()
            return

        while self._phase_queue:
            arm, budget = self._phase_queue.pop(0)
            if self._switch_to_arm(arm, budget):
                return

        if self._phase_name == "pilot":
            finalists = sorted(self._arms, key=self._arm_score, reverse=True)[
                : min(int(self.corrector_config.tournament_finalists), len(self._arms))
            ]
            self._phase_name = "finalist"
            self._phase_queue = self._round_robin_queue(
                finalists,
                int(self.corrector_config.tournament_finalist_iterations),
            )
            self._advance_tournament()
            return

        if self._phase_name == "finalist":
            winner = max(self._arms, key=self._arm_score)
            remaining = self._planned_iterations - sum(
                arm.completed_iterations for arm in self._arms
            )
            self._phase_name = "winner"
            if remaining > 0 and self._switch_to_arm(winner, remaining):
                return

        self._finish_correction()

    def _start_correction(self) -> bool:
        if not self._eligible_for_correction():
            return False
        remaining_budget = int(self.iteration_limit) - int(
            self.state.iterations_completed
        )
        planned_iterations = min(
            int(self.corrector_config.window_iterations),
            remaining_budget,
        )
        if planned_iterations <= 0:
            return False
        if not self.corrector.shared_policy.try_begin_tournament(
            planned_iterations
        ):
            self._next_corrector_iteration = (
                int(self.state.iterations_completed)
                + max(1, int(self.corrector_config.cooldown_iterations))
            )
            return False
        self._reserved_tournament_iterations = planned_iterations

        proposal_seed = (
            int(self.state.config.seed)
            + 104729 * (self._corrections_started + 1)
            + 1009 * int(self.search_index)
        )
        try:
            proposals = self.corrector.propose_candidates(
                self.state,
                rng=random.Random(proposal_seed),
                limit=int(self.corrector_config.tournament_candidates),
            )
        except Exception:
            self.corrector.shared_policy.finish_tournament(
                planned_iterations,
                0,
            )
            self._reserved_tournament_iterations = 0
            raise
        if not proposals:
            self.corrector.shared_policy.finish_tournament(
                planned_iterations,
                0,
            )
            self._reserved_tournament_iterations = 0
            self._next_corrector_iteration = (
                int(self.state.iterations_completed)
                + max(1, int(self.corrector_config.cooldown_iterations))
            )
            return False

        self._corrections_started += 1
        filler_seed = proposal_seed + 1_000_003
        base_root = self.state.root
        if base_root is None:
            raise RuntimeError("Corrector cannot start without a Filler root")
        base_nodes = self.state.nodes
        base_seen_signatures = Counter(self.state.seen_signatures)
        base_synchronized_epoch = int(self.state.synchronized_discovery_epoch)
        base_rollouts_started = int(self.state.rollouts_started)
        base_rng = getattr(self.state, "_rng", None)
        if base_rng is None:
            base_rng = random.Random(filler_seed)
        base_rng_state = base_rng.getstate()
        source_vertex_count = _vertex_count(
            self.state.scorer,
            proposals[0].source_key,
        )

        def paired_rng() -> random.Random:
            rng = random.Random()
            rng.setstate(base_rng_state)
            return rng

        if self.corrector_config.lightweight_tournament_arms:
            no_op_nodes, no_op_root = base_nodes, base_root
        else:
            no_op_nodes, no_op_root = _clone_search_graph(base_nodes, base_root)
        self._arms = [
            _CorrectionArm(
                name="no_op",
                proposal=None,
                nodes=no_op_nodes,
                root=no_op_root,
                resume_root=no_op_root,
                rng=paired_rng(),
                reference_vertex_count=source_vertex_count,
                seen_signatures=Counter(base_seen_signatures),
                synchronized_discovery_epoch=base_synchronized_epoch,
                rollouts_started=base_rollouts_started,
                basin_counts=Counter(self._basin_counts),
            )
        ]
        for index, proposal in enumerate(proposals, start=1):
            if self.corrector_config.lightweight_tournament_arms:
                arm_nodes: dict[BlockKey, MCTSNode] = {}
                resume_root = _get_or_create_node(
                    arm_nodes,
                    self.state.scorer,
                    proposal.corrected_key,
                    path=selected_blocks(proposal.corrected_key),
                )
            else:
                arm_nodes, resume_root = _clone_search_graph(base_nodes, base_root)
            root = _get_or_create_node(
                arm_nodes,
                self.state.scorer,
                proposal.corrected_key,
                path=selected_blocks(proposal.corrected_key),
            )
            self._arms.append(
                _CorrectionArm(
                    name=f"delete_{index}",
                    proposal=proposal,
                    nodes=arm_nodes,
                    root=root,
                    resume_root=resume_root,
                    rng=paired_rng(),
                    reference_vertex_count=_vertex_count(
                        self.state.scorer,
                        proposal.source_key,
                    ),
                    seen_signatures=Counter(base_seen_signatures),
                    synchronized_discovery_epoch=base_synchronized_epoch,
                    rollouts_started=base_rollouts_started,
                    basin_counts=Counter(self._basin_counts),
                )
            )
        self._tournament_rng = random.Random(proposal_seed + 2_000_003)

        first = proposals[0]
        event = CorrectorEvent(
            event_index=len(self.event_sink) + 1,
            search_index=int(self.search_index),
            start_class_id=int(self.class_id),
            correction_index=self._corrections_started,
            trigger_iteration=int(self.state.iterations_completed),
            filler_start_iteration=int(self.state.iterations_completed) + 1,
            source_label=first.source_label,
            source_rank=first.source_rank,
            source_blocks=selected_blocks(first.source_key),
            source_vertex_count=_vertex_count(self.state.scorer, first.source_key),
            protected_blocks=list(first.protected_blocks),
            removed_blocks=list(first.removed_blocks),
            corrected_blocks=selected_blocks(first.corrected_key),
            corrected_rank=first.corrected_rank,
            corrected_vertex_count=_vertex_count(
                self.state.scorer,
                first.corrected_key,
            ),
            compatible_undiscovered_classes=list(
                first.compatible_undiscovered_classes
            ),
            entrance_rare_count=first.entrance_rare_count,
            entrance_valid_count=first.entrance_valid_count,
            entrance_invalid_count=first.entrance_invalid_count,
            entrance_class_entropy=first.entrance_class_entropy,
            entrance_dominant_class_ratio=first.entrance_dominant_class_ratio,
            policy_score=first.policy_score,
            filler_seed=filler_seed,
            planned_filler_iterations=planned_iterations,
        )
        self._active_event = event
        self._planned_iterations = planned_iterations
        self._phase_name = "pilot"
        self._phase_queue = self._round_robin_queue(
            self._arms,
            int(self.corrector_config.tournament_pilot_iterations),
        )
        self.event_sink.append(event)
        self.state.stop_reason = None
        self._advance_tournament()
        return self._active_event is not None

    def _observe_active_arm(
        self,
        discoveries: Sequence[ExactClassDiscovery],
        *,
        previous_exact_serial: int,
        previous_invalid_terminal_count: int,
        elapsed_seconds: float,
    ) -> None:
        arm = self._active_arm
        if arm is None:
            return
        arm.completed_iterations += 1
        arm.elapsed_seconds += float(elapsed_seconds)
        self._store_active_arm_state()
        for discovery in discoveries:
            class_id = int(discovery.class_id)
            if class_id not in arm.new_global_class_ids:
                arm.new_global_class_ids.append(class_id)
        exact_delta = max(0, int(self.state.exact_hit_serial) - previous_exact_serial)
        arm.exact_hit_count += exact_delta
        current_invalid_count = sum(
            int(count)
            for label, count in self.state.encountered.items()
            if str(label).startswith("invalid:")
        )
        arm.invalid_terminal_count += max(
            0,
            current_invalid_count - int(previous_invalid_terminal_count),
        )
        if exact_delta <= 0:
            return
        hit = self.state.last_exact_hit
        if hit is None:
            return
        if self._basin_tracking_enabled():
            is_new_basin, basin_novelty = self._record_basin_hit(
                arm.basin_counts,
                hit,
            )
            arm.basin_novelty_sum += float(basin_novelty)
            if is_new_basin:
                arm.new_basin_count += 1
            else:
                arm.repeated_basin_count += 1
        if exact_class_id(hit.label) is not None:
            arm.exact_labels.add(hit.label)
        if arm.best_hit is None:
            arm.best_hit = hit
            return
        current_size = _vertex_count(self.state.scorer, hit.key)
        best_size = _vertex_count(self.state.scorer, arm.best_hit.key)
        if (current_size, hit.score) > (best_size, arm.best_hit.score):
            arm.best_hit = hit

    def _finish_correction(self) -> None:
        event = self._active_event
        if event is None or not self._arms:
            return
        self._store_active_arm_state()
        winner = max(self._arms, key=self._arm_score)
        no_op = next(arm for arm in self._arms if arm.proposal is None)
        committed = winner
        if self.corrector_config.commit_requires_global_novelty:
            novel_arms = [arm for arm in self._arms if arm.new_global_class_ids]
            committed = (
                max(novel_arms, key=self._arm_score) if novel_arms else no_op
            )
        no_op_rate = no_op.utility_rate(self.state.scorer, self.corrector_config)
        winner_rate = winner.utility_rate(self.state.scorer, self.corrector_config)
        has_counterfactual = no_op.completed_iterations > 0
        event.finished_iteration = int(self.state.iterations_completed)
        event.completed_filler_iterations = sum(
            arm.completed_iterations for arm in self._arms
        )
        if self._reserved_tournament_iterations > 0:
            self.corrector.shared_policy.finish_tournament(
                self._reserved_tournament_iterations,
                event.completed_filler_iterations,
            )
            self._reserved_tournament_iterations = 0
        event.exact_hit_count = sum(arm.exact_hit_count for arm in self._arms)
        event.new_global_class_ids = sorted(
            {
                int(class_id)
                for arm in self._arms
                for class_id in arm.new_global_class_ids
            }
        )
        event.selected_arm = winner.name
        event.committed_arm = committed.name
        event.proxy_winner_rejected = committed is not winner
        event.counterfactual_advantage = (
            float(winner_rate - no_op_rate)
            if has_counterfactual
            else None
        )
        event.arm_evaluations = [
            arm.to_dict(self.state.scorer, self.corrector_config)
            for arm in self._arms
        ]

        discovery_epoch = _global_discovery_epoch(self.state.global_discovery)
        for arm in self._arms:
            proposal = arm.proposal
            if (
                proposal is None
                or arm.completed_iterations <= 0
                or not has_counterfactual
            ):
                continue
            advantage = (
                arm.utility_rate(self.state.scorer, self.corrector_config)
                - no_op_rate
            )
            self.corrector.observe(
                proposal.context_features,
                advantage,
                discovery_epoch=discovery_epoch,
            )

        proposal = winner.proposal
        if proposal is None:
            event.removed_blocks = []
            event.corrected_blocks = selected_blocks(winner.root.key)
            event.corrected_rank = int(winner.root.rank)
            event.corrected_vertex_count = _vertex_count(
                self.state.scorer,
                winner.root.key,
            )
            event.policy_score = 0.0
            if winner.best_hit is None:
                event.repacked_blocks = selected_blocks(winner.root.key)
                event.repacked_rank = int(winner.root.rank)
                event.repacked_vertex_count = _vertex_count(
                    self.state.scorer,
                    winner.root.key,
                )
            else:
                event.repacked_label = winner.best_hit.label
                event.repacked_rank = int(winner.best_hit.rank)
                event.repacked_blocks = selected_blocks(winner.best_hit.key)
                event.repacked_vertex_count = _vertex_count(
                    self.state.scorer,
                    winner.best_hit.key,
                )
        else:
            event.source_label = proposal.source_label
            event.source_rank = proposal.source_rank
            event.source_blocks = selected_blocks(proposal.source_key)
            event.source_vertex_count = _vertex_count(
                self.state.scorer,
                proposal.source_key,
            )
            event.protected_blocks = list(proposal.protected_blocks)
            event.removed_blocks = list(proposal.removed_blocks)
            event.corrected_blocks = selected_blocks(proposal.corrected_key)
            event.corrected_rank = proposal.corrected_rank
            event.corrected_vertex_count = _vertex_count(
                self.state.scorer,
                proposal.corrected_key,
            )
            event.entrance_rare_count = proposal.entrance_rare_count
            event.entrance_valid_count = proposal.entrance_valid_count
            event.entrance_invalid_count = proposal.entrance_invalid_count
            event.entrance_class_entropy = proposal.entrance_class_entropy
            event.entrance_dominant_class_ratio = (
                proposal.entrance_dominant_class_ratio
            )
            event.compatible_undiscovered_classes = list(
                proposal.compatible_undiscovered_classes
            )
            event.policy_score = proposal.policy_score
            if winner.best_hit is None:
                repacked_key = proposal.corrected_key
                event.repacked_rank = proposal.corrected_rank
            else:
                repacked_key = winner.best_hit.key
                event.repacked_label = winner.best_hit.label
                event.repacked_rank = int(winner.best_hit.rank)
            event.repacked_blocks = selected_blocks(repacked_key)
            event.repacked_vertex_count = _vertex_count(
                self.state.scorer,
                repacked_key,
            )

        event.size_delta_reward = (
            0.0
            if proposal is None
            else float(
                int(event.repacked_vertex_count or event.source_vertex_count)
                - int(event.source_vertex_count)
            )
        )
        event.coverage_augmented_reward = float(
            self.corrector_config.size_delta_weight * event.size_delta_reward
            + self.corrector_config.new_class_reward
            * len(event.new_global_class_ids)
            + self.corrector_config.new_basin_reward
            * int(winner.new_basin_count)
            + self.corrector_config.basin_novelty_weight
            * float(winner.basin_novelty_sum)
            - self.corrector_config.repeated_basin_penalty
            * int(winner.repeated_basin_count)
            - self.corrector_config.invalid_terminal_penalty
            * int(winner.invalid_terminal_count)
        )

        self.state.nodes = committed.nodes
        self.state.root = (
            committed.root
            if self.corrector_config.persist_corrected_root
            and committed.proposal is not None
            else committed.resume_root
        )
        self.state.seen_signatures = committed.seen_signatures
        self.state.synchronized_discovery_epoch = int(
            committed.synchronized_discovery_epoch
        )
        self.state.rollouts_started = int(committed.rollouts_started)
        self.state._rng = committed.rng  # type: ignore[attr-defined]
        self._basin_counts = Counter(committed.basin_counts)
        if committed.new_basin_count > 0:
            self._last_new_basin_iteration = int(self.state.iterations_completed)
        for arm in self._arms:
            if arm is committed:
                continue
            for node in arm.nodes.values():
                node.parent = None
                node.children.clear()
        self._next_corrector_iteration = (
            int(self.state.iterations_completed)
            + max(0, int(self.corrector_config.cooldown_iterations))
        )
        self._active_event = None
        self._arms = []
        self._active_arm = None
        self._phase_queue = []
        self._phase_name = ""
        self._active_remaining = 0
        self._planned_iterations = 0
        self._tournament_rng = None

    def step(self) -> list[ExactClassDiscovery]:
        if self.finished:
            return []
        if self.state.iterations_completed >= self.iteration_limit:
            if self._active_event is not None:
                self._finish_correction()
            self.state.stop_reason = "iterations_exhausted"
            return []

        iteration_index = self.state.iterations_completed + 1
        previous_local_discoveries = set(
            self.state.local_discovered_exact_classes
        )
        previous_exact_serial = int(self.state.exact_hit_serial)
        previous_invalid_terminal_count = sum(
            int(count)
            for label, count in self.state.encountered.items()
            if str(label).startswith("invalid:")
        )
        started = time.perf_counter()
        discoveries = _run_one_iteration(self.state, iteration_index)
        iteration_elapsed = time.perf_counter() - started
        self.state.iterations_completed = iteration_index
        new_local_discoveries = (
            self.state.local_discovered_exact_classes - previous_local_discoveries
        )
        if new_local_discoveries:
            self.last_new_class_iteration = iteration_index
        if discoveries:
            self.last_global_new_iteration = iteration_index
        if any(
            int(class_id) != int(self.class_id)
            for class_id in new_local_discoveries
        ):
            self.state.productive = True
            self.iteration_limit = max(
                self.iteration_limit,
                _productive_iteration_limit(self),
            )

        if self._active_event is not None:
            self._observe_active_arm(
                discoveries,
                previous_exact_serial=previous_exact_serial,
                previous_invalid_terminal_count=previous_invalid_terminal_count,
                elapsed_seconds=iteration_elapsed,
            )
            self._active_remaining -= 1
            if (
                self._active_remaining <= 0
                or self.state.iterations_completed >= self.iteration_limit
            ):
                self._advance_tournament()
        elif (
            self._basin_tracking_enabled()
            and int(self.state.exact_hit_serial) > previous_exact_serial
            and self.state.last_exact_hit is not None
        ):
            is_new_basin, _novelty = self._record_basin_hit(
                self._basin_counts,
                self.state.last_exact_hit,
            )
            if is_new_basin:
                self._last_new_basin_iteration = iteration_index

        if self._active_event is not None:
            return discoveries
        if self._start_correction():
            return discoveries
        if self.state.iterations_completed < min(
            int(self.iteration_limit),
            max(0, int(self.minimum_service_iterations)),
        ):
            return discoveries

        if self.state.productive:
            if _should_stop_productive_for_no_novelty(self):
                self.state.stop_reason = "productive_novelty_patience"
            elif self.state.iterations_completed >= self.iteration_limit:
                self.state.stop_reason = "iterations_exhausted"
        elif _should_stop_for_exact_frequency(self):
            self.state.stop_reason = "exact_frequency_threshold"
        elif _should_stop_as_terminal_sink(self):
            self.state.stop_reason = "self_or_invalid_sink"
        elif self.state.iterations_completed >= self.iteration_limit:
            self.state.stop_reason = "iterations_exhausted"
        return discoveries


def _corrector_summary(
    events: Sequence[CorrectorEvent],
    *,
    shared_policy: SharedCorrectorPolicy | None = None,
) -> dict[str, object]:
    completed = [event for event in events if event.finished_iteration is not None]
    new_classes = sorted(
        {
            int(class_id)
            for event in completed
            for class_id in event.new_global_class_ids
        }
    )
    size_rewards = [
        float(event.size_delta_reward)
        for event in completed
        if event.size_delta_reward is not None
    ]
    advantages = [
        float(event.counterfactual_advantage)
        for event in completed
        if event.counterfactual_advantage is not None
    ]
    selected_arms = Counter(
        str(event.selected_arm)
        for event in completed
        if event.selected_arm is not None
    )
    committed_arms = Counter(
        str(event.committed_arm)
        for event in completed
        if event.committed_arm is not None
    )
    return {
        "transition_count": len(events),
        "completed_transition_count": len(completed),
        "tasks_corrected": sorted({event.start_class_id for event in events}),
        "new_global_class_ids": new_classes,
        "new_global_class_count": len(new_classes),
        "positive_size_delta_count": sum(1 for reward in size_rewards if reward > 0.0),
        "mean_size_delta_reward": (
            0.0 if not size_rewards else sum(size_rewards) / float(len(size_rewards))
        ),
        "selected_arm_counts": dict(sorted(selected_arms.items())),
        "no_op_win_count": int(selected_arms.get("no_op", 0)),
        "committed_arm_counts": dict(sorted(committed_arms.items())),
        "proxy_winner_rejected_count": sum(
            1 for event in completed if event.proxy_winner_rejected
        ),
        "positive_counterfactual_count": sum(
            1 for advantage in advantages if advantage > 0.0
        ),
        "mean_counterfactual_advantage": (
            0.0 if not advantages else sum(advantages) / float(len(advantages))
        ),
        "shared_policy": (
            None if shared_policy is None else shared_policy.summary()
        ),
        "events": [event.to_dict() for event in events],
    }


def _coverage_classes(payload: dict[str, object]) -> list[int]:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return []
    values = summary.get("encountered_rare_target_classes", [])
    if not isinstance(values, list):
        return []
    return sorted(int(value) for value in values)


def _baseline_comparison(
    current_payload: dict[str, object],
    baseline_path: Path | None,
) -> dict[str, object] | None:
    if baseline_path is None or not baseline_path.exists():
        return None
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_classes = _coverage_classes(baseline_payload)
    current_classes = _coverage_classes(current_payload)
    baseline_set = set(baseline_classes)
    current_set = set(current_classes)
    return {
        "baseline_path": str(baseline_path),
        "baseline_coverage_count": len(baseline_classes),
        "current_coverage_count": len(current_classes),
        "coverage_delta": len(current_classes) - len(baseline_classes),
        "added_class_ids": sorted(current_set - baseline_set),
        "lost_class_ids": sorted(baseline_set - current_set),
        "baseline_class_ids": baseline_classes,
        "current_class_ids": current_classes,
    }


def run_filler_corrector_search(
    search_config: InterruptSearchConfig,
    corrector_config: CorrectorConfig,
    *,
    output_path: Path,
    baseline_path: Path | None = DEFAULT_BASELINE_PATH,
) -> tuple[InterruptSearchReport, dict[str, object]]:
    """Run Filler-Corrector with the same stack and pattern policy as baseline."""
    global_discovery = InterruptGlobalDiscoveryState.from_config(search_config)
    base_factory = _default_task_factory(search_config, global_discovery)
    corrector_events: list[CorrectorEvent] = []
    shared_corrector_policy = SharedCorrectorPolicy(corrector_config)

    def task_factory(
        class_id: int,
        search_index: int,
        parent_search_index: int | None,
    ) -> FillerCorrectorTask:
        return FillerCorrectorTask(
            base_factory(class_id, search_index, parent_search_index),
            corrector_config=corrector_config,
            event_sink=corrector_events,
            shared_policy=shared_corrector_policy,
        )

    started = time.perf_counter()
    report = run_interruptible_search(
        search_config,
        task_factory=task_factory,
        global_discovery_state=global_discovery,
        deduplicate_patterns=True,
        output_path=output_path,
    )
    elapsed_seconds = time.perf_counter() - started
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    meta = payload.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["script"] = "src/mcts/filler_corrector_search.py"
        meta["algorithm"] = "counterfactual_contextual_block_filler_corrector"
        meta["paper"] = "https://arxiv.org/abs/2511.13391"
        meta["elapsed_seconds"] = elapsed_seconds
        meta["corrector"] = {
            name: value
            for name, value in vars(corrector_config).items()
        }
    payload["corrector"] = _corrector_summary(
        corrector_events,
        shared_policy=shared_corrector_policy,
    )
    comparison = _baseline_comparison(payload, baseline_path)
    if comparison is not None:
        payload["baseline_comparison"] = comparison
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-inspired block Filler-Corrector MCTS exploration."
    )
    parser.add_argument("--initial-class-id", type=int, default=7)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--rare-target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=26)
    parser.add_argument("--exploration-constant", type=float, default=1.4)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-temperature", type=float, default=0.85)
    parser.add_argument("--expansion-candidate-pool", type=int, default=16)
    parser.add_argument("--rollout-candidate-pool", type=int, default=4)
    parser.add_argument("--widening-score-batch", type=int, default=4)
    parser.add_argument("--widening-score-batch-max", type=int, default=16)
    parser.add_argument("--widening-score-batch-scale", type=float, default=1.0)
    parser.add_argument("--widening-score-batch-beta", type=float, default=0.5)
    parser.add_argument("--rollout-score-batch", type=int, default=8)
    parser.add_argument("--productive-rollout-score-batch", type=int, default=16)
    parser.add_argument("--broad-rollout-score-batch", type=int, default=24)
    parser.add_argument("--broad-rollout-interval", type=int, default=8)
    parser.add_argument("--progressive-k0", type=int, default=1)
    parser.add_argument("--progressive-alpha", type=float, default=1.0)
    parser.add_argument("--progressive-beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--rank24-entrance-exists-weight", type=float, default=6.0)
    parser.add_argument("--terminal-scoring-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--dynamic-new-class-score", type=float, default=100.0)
    parser.add_argument("--dynamic-known-class-score", type=float, default=10.0)
    parser.add_argument("--dynamic-frequent-class-score", type=float, default=-5.0)
    parser.add_argument("--dynamic-frequent-class-threshold", type=int, default=16)
    parser.add_argument("--exact-stop-ratio", type=float, default=2.0 / 5.0)
    parser.add_argument("--adaptive-min-iterations", type=int, default=50)
    parser.add_argument("--adaptive-min-terminal-hits", type=int, default=20)
    parser.add_argument("--adaptive-sink-ratio", type=float, default=0.9)
    parser.add_argument("--adaptive-extra-iterations", type=int, default=100)
    parser.add_argument("--adaptive-max-multiplier", type=float, default=3.0)
    parser.add_argument("--productive-patience-iterations", type=int, default=200)
    parser.add_argument("--corrector-protected-rank", type=int, default=8)
    parser.add_argument("--corrector-min-rank", type=int, default=23)
    parser.add_argument("--corrector-max-rank", type=int, default=24)
    parser.add_argument("--corrector-max-remove-blocks", type=int, default=2)
    parser.add_argument("--corrector-candidate-pool", type=int, default=12)
    parser.add_argument("--corrector-start-iteration", type=int, default=200)
    parser.add_argument("--corrector-stagnation-iterations", type=int, default=80)
    parser.add_argument("--corrector-window-iterations", type=int, default=48)
    parser.add_argument("--corrector-cooldown-iterations", type=int, default=80)
    parser.add_argument("--corrector-max-per-task", type=int, default=3)
    parser.add_argument("--corrector-max-per-sink-task", type=int, default=1)
    parser.add_argument("--corrector-distinct-exact-reward", type=float, default=2.0)
    parser.add_argument("--corrector-size-delta-weight", type=float, default=1.0)
    parser.add_argument("--corrector-wall-time-penalty", type=float, default=0.01)
    parser.add_argument("--corrector-tournament-candidates", type=int, default=4)
    parser.add_argument("--corrector-pilot-iterations", type=int, default=4)
    parser.add_argument("--corrector-finalists", type=int, default=2)
    parser.add_argument("--corrector-finalist-iterations", type=int, default=8)
    parser.add_argument("--corrector-ucb-exploration", type=float, default=1.0)
    parser.add_argument("--corrector-ucb-ridge", type=float, default=1.0)
    parser.add_argument("--corrector-epoch-decay", type=float, default=0.5)
    parser.add_argument("--corrector-reward-clip", type=float, default=20.0)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/mcts/runs/interrupt_search_filler_corrector_i200.json"),
    )
    return parser.parse_args()


def _search_config_from_args(args: argparse.Namespace) -> InterruptSearchConfig:
    return InterruptSearchConfig(
        initial_class_id=int(args.initial_class_id),
        rep_index=int(args.rep_index),
        pattern_index=int(args.pattern_index),
        target_classes=tuple(int(value) for value in args.target_classes),
        rare_target_classes=tuple(int(value) for value in args.rare_target_classes),
        examples_path=args.examples,
        iterations=int(args.iterations),
        max_depth=int(args.max_depth),
        exploration_constant=float(args.exploration_constant),
        discount=float(args.discount),
        prior_temperature=float(args.prior_temperature),
        rollout_temperature=float(args.rollout_temperature),
        expansion_candidate_pool=int(args.expansion_candidate_pool),
        rollout_candidate_pool=int(args.rollout_candidate_pool),
        widening_score_batch=int(args.widening_score_batch),
        widening_score_batch_max=int(args.widening_score_batch_max),
        widening_score_batch_scale=float(args.widening_score_batch_scale),
        widening_score_batch_beta=float(args.widening_score_batch_beta),
        rollout_score_batch=int(args.rollout_score_batch),
        productive_rollout_score_batch=int(args.productive_rollout_score_batch),
        broad_rollout_score_batch=int(args.broad_rollout_score_batch),
        broad_rollout_interval=int(args.broad_rollout_interval),
        progressive_k0=int(args.progressive_k0),
        progressive_alpha=float(args.progressive_alpha),
        progressive_beta=float(args.progressive_beta),
        seed=int(args.seed),
        rank24_entrance_exists_weight=float(args.rank24_entrance_exists_weight),
        terminal_scoring_mode=str(args.terminal_scoring_mode),
        dynamic_new_class_score=float(args.dynamic_new_class_score),
        dynamic_known_class_score=float(args.dynamic_known_class_score),
        dynamic_frequent_class_score=float(args.dynamic_frequent_class_score),
        dynamic_frequent_class_threshold=int(args.dynamic_frequent_class_threshold),
        exact_stop_ratio=float(args.exact_stop_ratio),
        adaptive_min_iterations=int(args.adaptive_min_iterations),
        adaptive_min_terminal_hits=int(args.adaptive_min_terminal_hits),
        adaptive_sink_ratio=float(args.adaptive_sink_ratio),
        adaptive_extra_iterations=int(args.adaptive_extra_iterations),
        adaptive_max_multiplier=float(args.adaptive_max_multiplier),
        productive_patience_iterations=int(args.productive_patience_iterations),
    )


def _corrector_config_from_args(args: argparse.Namespace) -> CorrectorConfig:
    return CorrectorConfig(
        protected_rank=int(args.corrector_protected_rank),
        min_corrected_rank=int(args.corrector_min_rank),
        max_corrected_rank=int(args.corrector_max_rank),
        max_remove_blocks=int(args.corrector_max_remove_blocks),
        candidate_pool=int(args.corrector_candidate_pool),
        start_iteration=int(args.corrector_start_iteration),
        stagnation_iterations=int(args.corrector_stagnation_iterations),
        window_iterations=int(args.corrector_window_iterations),
        cooldown_iterations=int(args.corrector_cooldown_iterations),
        max_corrections_per_task=int(args.corrector_max_per_task),
        max_corrections_per_sink_task=int(args.corrector_max_per_sink_task),
        new_class_reward=float(args.dynamic_new_class_score),
        distinct_exact_reward=float(args.corrector_distinct_exact_reward),
        size_delta_weight=float(args.corrector_size_delta_weight),
        wall_time_penalty=float(args.corrector_wall_time_penalty),
        tournament_candidates=int(args.corrector_tournament_candidates),
        tournament_pilot_iterations=int(args.corrector_pilot_iterations),
        tournament_finalists=int(args.corrector_finalists),
        tournament_finalist_iterations=int(args.corrector_finalist_iterations),
        ucb_exploration=float(args.corrector_ucb_exploration),
        ucb_ridge=float(args.corrector_ucb_ridge),
        discovery_epoch_decay=float(args.corrector_epoch_decay),
        reward_clip=float(args.corrector_reward_clip),
    )


def main() -> None:
    args = parse_args()
    _report, payload = run_filler_corrector_search(
        _search_config_from_args(args),
        _corrector_config_from_args(args),
        output_path=args.output,
        baseline_path=args.baseline,
    )
    result = {
        "stop_reason": payload.get("stop_reason"),
        "coverage": _coverage_classes(payload),
        "corrector_new_classes": payload.get("corrector", {}).get(
            "new_global_class_ids", []
        ),
        "baseline_comparison": payload.get("baseline_comparison"),
    }
    print(json.dumps(result, ensure_ascii=False))
    print(args.output)
    print(args.output.with_name(f"{args.output.stem}.runs.json"))


if __name__ == "__main__":
    main()
