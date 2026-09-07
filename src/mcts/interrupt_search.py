from __future__ import annotations

"""Stack-based MCTS supervisor.

This runner pauses the current search whenever a new exact class is found,
pushes a child search rooted at that class, and then resumes the parent search
after the child search finishes. It also stops a search early when any exact
class count exceeds `2/5 * iterations`.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from baseline.orbit_blocks import (
    BlockKey,
    PartitionKey,
    add_block,
    canonical_partition_key,
    canonical_support_key,
    empty_key,
    selected_blocks,
    unselected_blocks,
)
from baseline.reference_classes import (
    DEFAULT_EXAMPLES_PATH,
    parse_example_rows,
    support_mask_from_row,
)
from baseline.scorer import (
    ExpansionScorer,
    SharedScorerStructureCache,
    SharedTerminalValidationCache,
    exact_class_id,
)
from mcts.queue_search import build_class_scorer, summarize
from mcts.search import (
    ClassCompatibilityBank,
    ExactClassDiscovery,
    MCTSConfig,
    MCTSNode,
    MCTSResult,
    MCTSValueComponents,
    TerminalHit,
    _backpropagate_components,
    _estimate_state_value,
    _get_or_create_node,
    _prepare_actions,
    _progressive_child_limit,
    _register_hit,
    _refresh_discovery_dependent_action_scores,
    _refresh_terminal_action_scores,
    _rollout,
    _select_child,
    _select_progressive_action,
)
from mcts.decision_trace import emit as _trace_emit, set_trace_iteration


@dataclass(frozen=True)
class InterruptSearchConfig:
    """Inputs for stack-based interruptible MCTS exploration."""

    initial_class_id: int = 1
    rep_index: int = 1
    pattern_index: int = 0
    target_classes: tuple[int, ...] = field(default_factory=lambda: tuple(range(1, 47)))
    rare_target_classes: tuple[int, ...] = field(default_factory=tuple)
    examples_path: Path = DEFAULT_EXAMPLES_PATH
    iterations: int = 2000
    max_depth: int = 60
    exploration_constant: float = 1.4
    discount: float = 0.97
    prior_temperature: float = 1.0
    rollout_temperature: float = 0.85
    expansion_candidate_pool: int = 16
    rollout_candidate_pool: int = 4
    widening_score_batch: int = 4
    widening_score_batch_max: int = 16
    widening_score_batch_scale: float = 1.0
    widening_score_batch_beta: float = 0.5
    rollout_score_batch: int = 8
    productive_rollout_score_batch: int = 16
    broad_rollout_score_batch: int = 24
    broad_rollout_interval: int = 8
    progressive_k0: int = 1
    progressive_alpha: float = 1.0
    progressive_beta: float = 0.5
    progressive_bucket_quota: int = 1
    discovery_epoch_value_decay: float = 0.25
    seed: int = 20260502
    rank24_entrance_exists_weight: float = 6.0
    terminal_scoring_mode: str = "static"
    dynamic_new_class_score: float = 100.0
    dynamic_known_class_score: float = 10.0
    dynamic_frequent_class_score: float = -5.0
    dynamic_frequent_class_threshold: int = 16
    exact_stop_ratio: float = 2.0 / 5.0
    adaptive_min_iterations: int = 50
    adaptive_min_terminal_hits: int = 20
    adaptive_sink_ratio: float = 0.9
    adaptive_extra_iterations: int = 100
    adaptive_max_multiplier: float = 3.0
    productive_patience_iterations: int = 200
    compatibility_frontier_extension_iterations: int = 50
    compatibility_frontier_min_rank: int = 22
    compatibility_frontier_shadow_ranks: tuple[int, ...] = (20, 21)


@dataclass(frozen=True)
class InterruptDiscoveryEvent:
    """An exact-class hit that caused a child search to be pushed."""

    event_index: int
    global_iteration: int
    search_index: int
    parent_search_index: int | None
    start_class_id: int
    class_id: int
    iteration: int
    depth: int
    score: float
    rank: int
    label: str
    path: list[int]
    chosen_blocks: list[int]
    pattern_status: str | None = None
    pattern_signature: list[list[int]] | None = None
    basin_key: str | None = None
    rare_removed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "global_iteration": self.global_iteration,
            "search_index": self.search_index,
            "parent_search_index": self.parent_search_index,
            "start_class_id": self.start_class_id,
            "class_id": self.class_id,
            "iteration": self.iteration,
            "depth": self.depth,
            "score": self.score,
            "rank": self.rank,
            "label": self.label,
            "path": list(self.path),
            "chosen_blocks": list(self.chosen_blocks),
            "pattern_status": self.pattern_status,
            "pattern_signature": [list(block) for block in self.pattern_signature] if self.pattern_signature is not None else None,
            "basin_key": self.basin_key,
            "rare_removed": bool(self.rare_removed),
        }


@dataclass(frozen=True)
class SearchTaskIdentity:
    """A structural partition plus one symmetry-inequivalent facet basin."""

    partition_key: PartitionKey
    basin_key: int


@dataclass(frozen=True)
class CompatibilityFrontierDiagnosticExample:
    """One representative node or action outcome from a frontier scan."""

    outcome: str
    node_rank: int
    selected_word: int
    action: int | None = None
    compatible_class_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "node_rank": self.node_rank,
            "selected_word": self.selected_word,
            "selected_word_hex": f"0x{self.selected_word:x}",
            "action": self.action,
            "compatible_class_ids": list(self.compatible_class_ids),
        }


_FRONTIER_DIAGNOSTIC_EXAMPLE_LIMIT = 8


@dataclass(frozen=True)
class CompatibilityFrontierSnapshot:
    """Actionable near-terminal edges that still retain missing classes."""

    class_ids: tuple[int, ...]
    max_rank: int
    edge_fingerprints: frozenset[tuple[int, int, int, int]]
    rank_min: int = 22
    rank_max_exclusive: int = 25
    missing_class_ids: tuple[int, ...] = ()
    scanned_node_count: int = 0
    candidate_action_count: int = 0
    outcome_counts: tuple[tuple[str, int], ...] = ()
    missing_class_compatible_action_counts: tuple[tuple[int, int], ...] = ()
    examples: tuple[CompatibilityFrontierDiagnosticExample, ...] = ()

    @property
    def edge_count(self) -> int:
        return len(self.edge_fingerprints)

    def diagnostic_dict(
        self,
        *,
        iteration: int,
        discovery_epoch: int,
    ) -> dict[str, object]:
        return {
            "iteration": int(iteration),
            "discovery_epoch": int(discovery_epoch),
            "rank_min": int(self.rank_min),
            "rank_max_exclusive": int(self.rank_max_exclusive),
            "missing_class_ids": list(self.missing_class_ids),
            "scanned_node_count": int(self.scanned_node_count),
            "candidate_action_count": int(self.candidate_action_count),
            "retained_class_ids": list(self.class_ids),
            "retained_edge_count": int(self.edge_count),
            "max_rank": int(self.max_rank),
            "outcome_counts": {
                str(outcome): int(count)
                for outcome, count in self.outcome_counts
            },
            "missing_class_compatible_action_counts": {
                str(class_id): int(count)
                for class_id, count in self.missing_class_compatible_action_counts
            },
            "example_limit_per_outcome": _FRONTIER_DIAGNOSTIC_EXAMPLE_LIMIT,
            "examples": [example.to_dict() for example in self.examples],
        }


@dataclass(frozen=True)
class InterruptSearchRun:
    """One exact-class-rooted search frame."""

    search_index: int
    parent_search_index: int | None
    start_class_id: int
    seed: int
    iteration_limit: int
    stop_reason: str | None
    productive: bool
    cross_class_count: int
    last_new_class_iteration: int | None
    last_global_new_iteration: int | None
    compatibility_frontier_extensions: int
    compatibility_frontier_extension_deadline: int
    compatibility_frontier_observations: int
    last_compatibility_frontier_class_ids: list[int]
    last_compatibility_frontier_max_rank: int
    last_compatibility_frontier_edge_count: int
    last_compatibility_frontier_new_edge_count: int
    compatibility_frontier_outcome_counts: dict[str, int]
    compatibility_frontier_observation_details: list[dict[str, object]]
    compatibility_frontier_shadow_outcome_counts: dict[str, dict[str, int]]
    compatibility_frontier_shadow_observation_details: list[dict[str, object]]
    result: dict[str, object]
    interrupted_by_class_ids: list[int]
    resumed_after_class_ids: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "search_index": self.search_index,
            "parent_search_index": self.parent_search_index,
            "start_class_id": self.start_class_id,
            "seed": self.seed,
            "iteration_limit": self.iteration_limit,
            "stop_reason": self.stop_reason,
            "productive": self.productive,
            "cross_class_count": self.cross_class_count,
            "last_new_class_iteration": self.last_new_class_iteration,
            "last_global_new_iteration": self.last_global_new_iteration,
            "compatibility_frontier_extensions": self.compatibility_frontier_extensions,
            "compatibility_frontier_extension_deadline": self.compatibility_frontier_extension_deadline,
            "compatibility_frontier_observations": self.compatibility_frontier_observations,
            "last_compatibility_frontier_class_ids": list(
                self.last_compatibility_frontier_class_ids
            ),
            "last_compatibility_frontier_max_rank": self.last_compatibility_frontier_max_rank,
            "last_compatibility_frontier_edge_count": self.last_compatibility_frontier_edge_count,
            "last_compatibility_frontier_new_edge_count": self.last_compatibility_frontier_new_edge_count,
            "compatibility_frontier_outcome_counts": dict(
                self.compatibility_frontier_outcome_counts
            ),
            "compatibility_frontier_observation_details": list(
                self.compatibility_frontier_observation_details
            ),
            "compatibility_frontier_shadow_outcome_counts": {
                str(rank): dict(counts)
                for rank, counts in self.compatibility_frontier_shadow_outcome_counts.items()
            },
            "compatibility_frontier_shadow_observation_details": list(
                self.compatibility_frontier_shadow_observation_details
            ),
            "result": self.result,
            "interrupted_by_class_ids": list(self.interrupted_by_class_ids),
            "resumed_after_class_ids": list(self.resumed_after_class_ids),
        }


@dataclass(frozen=True)
class InterruptSearchReport:
    """Aggregate output of the interruptible supervisor."""

    meta: dict[str, object]
    started_class_ids: list[int]
    finished_class_ids: list[int]
    interrupt_events: list[InterruptDiscoveryEvent]
    runs: list[InterruptSearchRun]
    summary: dict[str, object]
    stop_reason: str
    exact_class_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "meta": self.meta,
            "started_class_ids": list(self.started_class_ids),
            "finished_class_ids": list(self.finished_class_ids),
            "interrupt_events": [event.to_dict() for event in self.interrupt_events],
            "runs": [run.to_dict() for run in self.runs],
            "summary": self.summary,
            "stop_reason": self.stop_reason,
            "exact_class_counts": dict(sorted(self.exact_class_counts.items())),
        }


def _start_json_array(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[\n", encoding="utf-8")


def _append_json_array_record(path: Path, payload: dict[str, object], *, first_record: bool) -> None:
    with path.open("a", encoding="utf-8") as handle:
        if not first_record:
            handle.write(",\n")
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))


def _close_json_array(path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n]\n")


def _write_json_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


@dataclass
class InterruptGlobalDiscoveryState:
    """Discovery knowledge shared by every task in one interrupt search."""

    initial_rare_target_classes: set[int] = field(default_factory=set)
    remaining_rare_target_classes: set[int] = field(default_factory=set)
    discovered_exact_classes: set[int] = field(default_factory=set)
    discovered_label_counts: Counter[str] = field(default_factory=Counter)
    discovery_epoch: int = 0
    terminal_validation_cache: SharedTerminalValidationCache = field(
        default_factory=SharedTerminalValidationCache
    )
    structure_cache: SharedScorerStructureCache = field(
        default_factory=SharedScorerStructureCache
    )

    @classmethod
    def from_config(cls, config: InterruptSearchConfig) -> InterruptGlobalDiscoveryState:
        rare_targets = set(int(value) for value in config.rare_target_classes)
        return cls(
            initial_rare_target_classes=set(rare_targets),
            remaining_rare_target_classes=set(rare_targets),
        )

    def mark_discovered(self, class_id: int) -> bool:
        normalized = int(class_id)
        if normalized in self.discovered_exact_classes:
            return False
        self.discovered_exact_classes.add(normalized)
        self.remaining_rare_target_classes.discard(normalized)
        self.discovery_epoch += 1
        return True

    def active_rare_target_classes(self) -> set[int]:
        return self.remaining_rare_target_classes

@dataclass
class InterruptSearchState:
    scorer: ExpansionScorer
    config: MCTSConfig
    global_discovery: InterruptGlobalDiscoveryState = field(default_factory=InterruptGlobalDiscoveryState)
    compatibility_bank: ClassCompatibilityBank | None = None
    nodes: dict[BlockKey, MCTSNode] = field(default_factory=dict)
    root: MCTSNode | None = None
    terminal_bests: dict[str, TerminalHit] = field(default_factory=dict)
    exact_discoveries: list[ExactClassDiscovery] = field(default_factory=list)
    local_discovered_exact_classes: set[int] = field(default_factory=set)
    seen_signatures: Counter[tuple[int, ...]] = field(default_factory=Counter)
    encountered: Counter[str] = field(default_factory=Counter)
    best: TerminalHit | None = None
    iterations_completed: int = 0
    rollouts_started: int = 0
    stop_reason: str | None = None
    productive: bool = False
    last_exact_hit: TerminalHit | None = None
    exact_hit_serial: int = 0
    last_exact_observation: tuple[int, str, BlockKey] | None = None
    synchronized_discovery_epoch: int = -1

    def release_tree(self) -> None:
        """Release a completed task's cyclic MCTS graph immediately."""
        for node in self.nodes.values():
            node.parent = None
            node.children.clear()
            node.unexpanded_actions.clear()
            node.action_priors.clear()
            node.action_scores.clear()
        self.nodes.clear()
        self.root = None
        if self.compatibility_bank is not None:
            self.compatibility_bank.clear_cache()
        self.seen_signatures.clear()
        if hasattr(self, "_rng"):
            delattr(self, "_rng")


@dataclass
class InterruptSearchTask:
    """One runnable search frame."""

    class_id: int
    search_index: int
    parent_search_index: int | None
    state: InterruptSearchState
    threshold: float
    base_iteration_limit: int
    max_iteration_limit: int
    adaptive_min_iterations: int
    adaptive_min_terminal_hits: int
    adaptive_sink_ratio: float
    adaptive_extra_iterations: int
    productive_patience_iterations: int
    iteration_limit: int
    compatibility_frontier_extension_iterations: int = 50
    compatibility_frontier_min_rank: int = 22
    compatibility_frontier_shadow_ranks: tuple[int, ...] = (20, 21)
    compatibility_frontier_extension_deadline: int = 0
    compatibility_frontier_extensions: int = 0
    compatibility_frontier_observations: int = 0
    compatibility_frontier_epoch_observations: int = 0
    compatibility_frontier_discovery_epoch: int = -1
    compatibility_frontier_seen_edges: set[tuple[int, int, int, int]] = field(
        default_factory=set,
        repr=False,
    )
    last_compatibility_frontier_class_ids: tuple[int, ...] = field(default_factory=tuple)
    last_compatibility_frontier_max_rank: int = 0
    last_compatibility_frontier_edge_count: int = 0
    last_compatibility_frontier_new_edge_count: int = 0
    compatibility_frontier_outcome_counts: Counter[str] = field(default_factory=Counter)
    compatibility_frontier_observation_details: list[dict[str, object]] = field(
        default_factory=list
    )
    compatibility_frontier_shadow_outcome_counts: dict[int, Counter[str]] = field(
        default_factory=dict
    )
    compatibility_frontier_shadow_observation_details: list[dict[str, object]] = field(
        default_factory=list
    )
    last_new_class_iteration: int | None = None
    last_global_new_iteration: int | None = None
    started_interrupts: list[int] = field(default_factory=list)
    resumed_interrupts: list[int] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.state.stop_reason is not None

    def step(self) -> list[ExactClassDiscovery]:
        if self.finished:
            return []
        if self.state.iterations_completed >= self.iteration_limit:
            self.state.stop_reason = "iterations_exhausted"
            return []

        iteration_index = self.state.iterations_completed + 1
        previous_local_discoveries = set(self.state.local_discovered_exact_classes)
        discoveries = _run_one_iteration(self.state, iteration_index)
        self.state.iterations_completed = iteration_index
        new_local_discoveries = self.state.local_discovered_exact_classes - previous_local_discoveries
        if new_local_discoveries:
            self.last_new_class_iteration = iteration_index
        if any(int(discovery.class_id) != int(self.class_id) for discovery in discoveries):
            self.last_global_new_iteration = iteration_index
        if any(int(class_id) != int(self.class_id) for class_id in new_local_discoveries):
            self.state.productive = True
            self.iteration_limit = max(
                self.iteration_limit,
                _productive_iteration_limit(self),
            )

        if self.state.productive:
            if _should_stop_productive_for_no_novelty(self):
                self.state.stop_reason = "productive_global_novelty_patience"
            elif self.state.iterations_completed >= self.iteration_limit:
                self.state.stop_reason = "iterations_exhausted"
        elif _should_stop_for_exact_frequency(self):
            self.state.stop_reason = "exact_frequency_threshold"
        elif _should_stop_as_terminal_sink(self):
            self.state.stop_reason = "self_or_invalid_sink"
        elif self.state.iterations_completed >= self.iteration_limit:
            self.state.stop_reason = "iterations_exhausted"
        return discoveries

    def result(self) -> MCTSResult:
        root_visits = 0 if self.state.root is None else self.state.root.visits
        return MCTSResult(
            best=self.state.best,
            terminal_bests=self.state.terminal_bests,
            exact_discoveries=self.state.exact_discoveries,
            encountered_label_counts=self.state.encountered,
            iterations_completed=self.state.iterations_completed,
            nodes_created=len(self.state.nodes),
            root_visits=root_visits,
        )


def _productive_iteration_limit(task: InterruptSearchTask) -> int:
    distinct_cross_classes = sum(
        1
        for class_id in task.state.local_discovered_exact_classes
        if int(class_id) != int(task.class_id)
    )
    weighted_limit = task.base_iteration_limit + (
        max(0, int(task.adaptive_extra_iterations)) * distinct_cross_classes
    )
    return min(task.max_iteration_limit, max(task.base_iteration_limit, weighted_limit))


def _build_state(
    config: InterruptSearchConfig,
    class_id: int,
    *,
    search_index: int,
    global_discovery: InterruptGlobalDiscoveryState | None = None,
) -> tuple[InterruptSearchState, tuple[tuple[int, ...], ...]]:
    shared_discovery = global_discovery or InterruptGlobalDiscoveryState.from_config(config)
    scorer, pattern_signature = build_class_scorer(
        _queue_like_config(config),
        class_id,
        rare_target_classes=shared_discovery.active_rare_target_classes(),
        terminal_validation_cache=shared_discovery.terminal_validation_cache,
        structure_cache=shared_discovery.structure_cache,
    )
    mcts_config = MCTSConfig(
        iterations=int(config.iterations),
        max_depth=int(config.max_depth),
        exploration_constant=float(config.exploration_constant),
        discount=float(config.discount),
        prior_temperature=float(config.prior_temperature),
        rollout_temperature=float(config.rollout_temperature),
        expansion_candidate_pool=int(config.expansion_candidate_pool),
        rollout_candidate_pool=int(config.rollout_candidate_pool),
        widening_score_batch=int(config.widening_score_batch),
        widening_score_batch_max=int(config.widening_score_batch_max),
        widening_score_batch_scale=float(config.widening_score_batch_scale),
        widening_score_batch_beta=float(config.widening_score_batch_beta),
        rollout_score_batch=int(config.rollout_score_batch),
        productive_rollout_score_batch=int(config.productive_rollout_score_batch),
        broad_rollout_score_batch=int(config.broad_rollout_score_batch),
        broad_rollout_interval=int(config.broad_rollout_interval),
        progressive_k0=int(config.progressive_k0),
        progressive_alpha=float(config.progressive_alpha),
        progressive_beta=float(config.progressive_beta),
        progressive_bucket_quota=int(config.progressive_bucket_quota),
        discovery_epoch_value_decay=float(config.discovery_epoch_value_decay),
        seed=int(config.seed) + 1009 * (search_index - 1),
        selection_survival_weight=1.0,
        selection_novelty_weight=1.0,
        compatibility_examples_path=config.examples_path,
    )
    state = InterruptSearchState(
        scorer=scorer,
        config=mcts_config,
        global_discovery=shared_discovery,
        compatibility_bank=ClassCompatibilityBank.from_scorer(scorer, config.examples_path),
    )
    root_key = empty_key(len(scorer.blocks))
    state.root = _get_or_create_node(state.nodes, scorer, root_key, path=[])
    return state, pattern_signature


def _known_exact_terminal_score(
    state: InterruptSearchState,
    label: str,
    *,
    hit_count: int | None = None,
) -> float:
    class_id = exact_class_id(label)
    if class_id is None:
        raise ValueError(f"expected an exact terminal label, got {label!r}")
    config = state.scorer.config
    if config.terminal_scoring_mode == "dynamic":
        count = max(
            1,
            int(
                state.global_discovery.discovered_label_counts.get(label, 0)
                if hit_count is None
                else hit_count
            ),
        )
        if count >= config.dynamic_frequent_class_threshold:
            return float(config.dynamic_frequent_class_score)
        return float(config.dynamic_known_class_score)
    if class_id == 44:
        return float(config.class44_terminal_score)
    if class_id in state.scorer.target_classes:
        return float(config.target_terminal_score)
    return float(config.valid_terminal_score)


def _synchronize_discovery_epoch(
    state: InterruptSearchState,
    *,
    terminal_score_fn: Callable[..., float],
) -> None:
    """Refresh mutable action values when another task discovers a new class."""
    active_rare_targets = state.global_discovery.active_rare_target_classes()
    if state.scorer.rare_target_classes != active_rare_targets:
        state.scorer.rare_target_classes = set(active_rare_targets)

    current_epoch = int(state.global_discovery.discovery_epoch)
    if state.synchronized_discovery_epoch == current_epoch:
        return

    if state.synchronized_discovery_epoch >= 0:
        decay = min(1.0, max(0.0, float(state.config.discovery_epoch_value_decay)))
        for node in state.nodes.values():
            node.value_visits *= decay
            node.value_sum *= decay
            node.escape_sum *= decay
            node.survival_sum *= decay
            node.novelty_sum *= decay

    for node in state.nodes.values():
        _refresh_discovery_dependent_action_scores(
            node,
            state.scorer,
            state.config,
            terminal_score_fn=terminal_score_fn,
        )
    state.synchronized_discovery_epoch = current_epoch


def _run_one_iteration(state: InterruptSearchState, iteration_index: int) -> list[ExactClassDiscovery]:
    set_trace_iteration(iteration_index)
    rng = getattr(state, "_rng", None)
    if rng is None:
        rng = random.Random(state.config.seed + 1009 * iteration_index)
        state._rng = rng  # type: ignore[attr-defined]

    root = state.root
    if root is None:
        raise RuntimeError("search state is missing a root node")

    path_nodes: list[MCTSNode] = [root]
    node = root
    new_discoveries: list[ExactClassDiscovery] = []
    _trace_emit(
        "iteration_start",
        root_visits=root.visits,
        node_count=len(state.nodes),
        discovery_epoch=state.global_discovery.discovery_epoch,
        discovered_class_ids=sorted(state.global_discovery.discovered_exact_classes),
    )

    def terminal_score(key: BlockKey, terminal: object) -> float:
        terminal_label = str(getattr(terminal, "label", ""))
        class_id = exact_class_id(terminal_label)
        if class_id is not None and class_id in state.global_discovery.discovered_exact_classes:
            return _known_exact_terminal_score(state, terminal_label)
        return float(
            state.scorer.terminal_score(
                key,
                rare_target_classes=state.global_discovery.active_rare_target_classes(),
                discovered_label_counts=state.global_discovery.discovered_label_counts,
            )
        )

    _synchronize_discovery_epoch(state, terminal_score_fn=terminal_score)

    def node_components(key: BlockKey, *, survival: float = 0.0) -> MCTSValueComponents:
        bank = state.compatibility_bank
        discovered_exact_classes = state.global_discovery.discovered_exact_classes
        if bank is None or not discovered_exact_classes:
            return MCTSValueComponents(escape=0.0, survival=float(survival), novelty=0.0)
        active_classes = sorted(discovered_exact_classes)
        current_total = bank.total_compat_count(key, active_classes)
        escape = 0.0 if current_total <= 0 else 1.0 / (1.0 + float(current_total))
        signature = bank.active_signature(key, active_classes)
        novelty = 1.0 / (1.0 + float(state.seen_signatures[signature]))
        return MCTSValueComponents(escape=float(escape), survival=float(survival), novelty=float(novelty))

    def record_signature(key: BlockKey) -> None:
        bank = state.compatibility_bank
        if bank is None:
            return
        signature = bank.active_signature(key, state.global_discovery.discovered_exact_classes)
        state.seen_signatures[signature] += 1

    def record_exact_discovery(
        terminal: object,
        *,
        score: float,
        depth: int,
        key: BlockKey,
        path: Sequence[int],
        rank: int,
    ) -> None:
        terminal_label = getattr(terminal, "label", None)
        if terminal_label is None:
            return
        class_id = exact_class_id(str(terminal_label))
        if class_id is None:
            return
        observation = (int(iteration_index), str(terminal_label), key)
        if observation != state.last_exact_observation:
            state.last_exact_hit = TerminalHit(
                label=str(terminal_label),
                key=key,
                path=list(path),
                score=float(score),
                rank=int(rank),
            )
            state.exact_hit_serial += 1
            state.last_exact_observation = observation
        is_local_discovery = class_id not in state.local_discovered_exact_classes
        is_global_discovery = state.global_discovery.mark_discovered(class_id)
        state.local_discovered_exact_classes.add(class_id)
        state.scorer.rare_target_classes.discard(class_id)
        if not is_local_discovery:
            return
        discovery = ExactClassDiscovery(
            class_id=class_id,
            label=str(terminal_label),
            iteration=iteration_index,
            depth=depth,
            score=float(score),
            rank=int(rank),
            path=list(path),
            chosen_blocks=selected_blocks(key),
        )
        state.exact_discoveries.append(discovery)
        if is_global_discovery:
            new_discoveries.append(discovery)
        record_signature(key)

    while True:
        _trace_emit(
            "node_visit",
            path=list(node.path),
            selected_blocks=selected_blocks(node.key),
            rank=node.rank,
            visits=node.visits,
            value_visits=node.value_visits,
            escape_q=node.escape_q_value,
            survival_q=node.survival_q_value,
            novelty_q=node.novelty_q_value,
            child_count=len(node.children),
            unexpanded_count=len(node.unexpanded_actions),
        )
        if node.is_terminal:
            terminal_score_value = terminal_score(node.key, node.terminal)
            _trace_emit(
                "terminal_observation",
                source="tree_node",
                path=list(node.path),
                rank=node.rank,
                label=None if node.terminal is None else node.terminal.label,
                score=terminal_score_value,
            )
            if node.terminal is not None and node.terminal.is_exact:
                record_exact_discovery(
                    node.terminal,
                    score=terminal_score_value,
                    depth=len(node.path),
                    key=node.key,
                    path=node.path,
                    rank=node.rank,
                )
            components = node_components(node.key, survival=terminal_score_value)
            state.best = _register_hit(
                node,
                terminal_score_value,
                terminal_bests=state.terminal_bests,
                encountered=state.encountered,
                scorer=state.scorer,
                best=state.best,
                global_discovered_label_counts=state.global_discovery.discovered_label_counts,
            )
            _backpropagate_components(
                path_nodes,
                components,
                survival_weight=state.config.selection_survival_weight,
                novelty_weight=state.config.selection_novelty_weight,
            )
            break

        if node.rank >= 25:
            terminal = state.scorer.terminal_label(node.key)
            terminal_score_value = terminal_score(node.key, terminal)
            _trace_emit(
                "terminal_observation",
                source="tree_rank25",
                path=list(node.path),
                rank=node.rank,
                label=terminal.label,
                score=terminal_score_value,
            )
            if terminal.is_exact:
                record_exact_discovery(
                    terminal,
                    score=terminal_score_value,
                    depth=len(node.path),
                    key=node.key,
                    path=node.path,
                    rank=node.rank,
                )
            components = node_components(node.key, survival=terminal_score_value)
            state.best = _register_hit(
                node,
                terminal_score_value,
                terminal=terminal,
                terminal_bests=state.terminal_bests,
                encountered=state.encountered,
                scorer=state.scorer,
                best=state.best,
                global_discovered_label_counts=state.global_discovery.discovered_label_counts,
            )
            _backpropagate_components(
                path_nodes,
                components,
                survival_weight=state.config.selection_survival_weight,
                novelty_weight=state.config.selection_novelty_weight,
            )
            break

        if not node.actions_initialized:
            _prepare_actions(
                node,
                state.scorer,
                state.config,
                terminal_score_fn=terminal_score,
            )

        if len(node.children) < _progressive_child_limit(node, state.config) and node.unexpanded_actions:
            action = _select_progressive_action(
                node,
                rng,
                scorer=state.scorer,
                cfg=state.config,
                compatibility_bank=state.compatibility_bank,
                discovered_exact_classes=state.global_discovery.discovered_exact_classes,
                seen_signatures=state.seen_signatures,
                terminal_score_fn=terminal_score,
                adaptive_score_batch=state.productive,
            )
            child_key = add_block(node.key, action)
            _trace_emit("expand_action", parent_path=list(node.path), action=action)
            child_path = [*node.path, int(action)]
            child = _get_or_create_node(state.nodes, state.scorer, child_key, path=child_path, parent=node, action=action)
            child.prior = node.action_priors.get(action, child.prior)
            node.children[action] = child
            node.unexpanded_actions = [item for item in node.unexpanded_actions if item != action]
            path_nodes.append(child)

            step = node.action_scores[action]
            if child.is_terminal or child.rank >= 25:
                terminal = child.terminal if child.terminal is not None else state.scorer.terminal_label(child.key)
                terminal_score_value = terminal_score(child.key, terminal)
                _trace_emit(
                    "terminal_observation",
                    source="expanded_child",
                    path=list(child.path),
                    rank=child.rank,
                    label=terminal.label,
                    score=terminal_score_value,
                )
                if terminal.is_exact:
                    record_exact_discovery(
                        terminal,
                        score=terminal_score_value,
                        depth=len(child.path),
                        key=child.key,
                        path=child.path,
                        rank=child.rank,
                    )
                components = node_components(child.key, survival=terminal_score_value)
                state.best = _register_hit(
                    child,
                    terminal_score_value,
                    terminal=terminal,
                    terminal_bests=state.terminal_bests,
                    encountered=state.encountered,
                    scorer=state.scorer,
                    best=state.best,
                    global_discovered_label_counts=state.global_discovery.discovered_label_counts,
                )
                _backpropagate_components(
                    path_nodes,
                    components,
                    survival_weight=state.config.selection_survival_weight,
                    novelty_weight=state.config.selection_novelty_weight,
                )
            else:
                def _record_for_rollout(*args, **kwargs):
                    # _rollout may pass through an `iteration_index` keyword; strip it
                    # because the local record_exact_discovery does not accept it.
                    kwargs.pop("iteration_index", None)
                    return record_exact_discovery(*args, **kwargs)

                state.rollouts_started += 1
                rollout_value, rollout_components, state.best = _rollout(
                    child,
                    state.scorer,
                    state.config,
                    rng,
                    state.terminal_bests,
                    state.encountered,
                    state.best,
                    _record_for_rollout,
                    state.compatibility_bank,
                    state.global_discovery.discovered_exact_classes,
                    state.seen_signatures,
                    iteration_index,
                    terminal_score_fn=terminal_score,
                    global_discovered_label_counts=state.global_discovery.discovered_label_counts,
                    rollout_score_batch=_rollout_score_batch_for_rollout(
                        state,
                        state.rollouts_started,
                    ),
                )
                components = MCTSValueComponents(
                    escape=rollout_components.escape,
                    survival=float(step.score) + rollout_components.survival,
                    novelty=rollout_components.novelty,
                )
                record_signature(child.key)
                _backpropagate_components(
                    path_nodes,
                    components,
                    survival_weight=state.config.selection_survival_weight,
                    novelty_weight=state.config.selection_novelty_weight,
                )
            break

        _refresh_terminal_action_scores(
            node,
            state.scorer,
            state.config,
            terminal_score_fn=terminal_score,
        )
        next_node = _select_child(
            node,
            state.config.exploration_constant,
            survival_weight=state.config.selection_survival_weight,
            novelty_weight=state.config.selection_novelty_weight,
        )
        if next_node is None:
            value = _estimate_state_value(
                node.key,
                state.scorer,
                state.config,
                compatibility_bank=state.compatibility_bank,
                discovered_exact_classes=state.global_discovery.discovered_exact_classes,
                seen_signatures=state.seen_signatures,
                terminal_score_fn=terminal_score,
            )
            _backpropagate_components(
                path_nodes,
                value,
                survival_weight=state.config.selection_survival_weight,
                novelty_weight=state.config.selection_novelty_weight,
            )
            break
        node = next_node
        path_nodes.append(node)

    return new_discoveries


def _rollout_score_batch_for_rollout(
    state: InterruptSearchState,
    rollout_index: int,
) -> int:
    config = state.config
    if not state.productive:
        return max(1, int(config.rollout_score_batch))
    batch_size = max(
        1,
        int(config.rollout_score_batch),
        int(config.productive_rollout_score_batch),
    )
    interval = int(config.broad_rollout_interval)
    if interval > 0 and int(rollout_index) % interval == 0:
        batch_size = max(batch_size, int(config.broad_rollout_score_batch))
    return batch_size


def _should_stop_for_exact_frequency(task: InterruptSearchTask) -> bool:
    """Stop only when the undiscovered-cross root class is strictly dominant."""
    state = task.state
    if state.productive:
        return False
    if any(class_id != task.class_id for class_id in state.local_discovered_exact_classes):
        return False

    self_label = f"exact:class{int(task.class_id)}"
    self_count = int(state.scorer.discovered_label_counts.get(self_label, 0))
    if self_count <= float(task.threshold):
        return False
    other_counts = [
        int(count)
        for label, count in state.scorer.discovered_label_counts.items()
        if str(label) != self_label
    ]
    return not other_counts or self_count > max(other_counts)


def _should_stop_productive_for_no_novelty(task: InterruptSearchTask) -> bool:
    if not task.state.productive:
        return False
    patience = int(task.productive_patience_iterations)
    if patience <= 0:
        return False
    if task.state.iterations_completed < task.base_iteration_limit:
        return False
    last_global_new = 0 if task.last_global_new_iteration is None else int(task.last_global_new_iteration)
    if task.state.iterations_completed - last_global_new < patience:
        return False

    current_iteration = int(task.state.iterations_completed)
    extension = int(
        getattr(task, "compatibility_frontier_extension_iterations", 0)
    )
    if extension <= 0:
        return True

    discovery_epoch = int(task.state.global_discovery.discovery_epoch)
    if int(getattr(task, "compatibility_frontier_discovery_epoch", -1)) != discovery_epoch:
        _reset_compatibility_frontier_epoch(task, discovery_epoch)

    extension_deadline = int(
        getattr(task, "compatibility_frontier_extension_deadline", 0)
    )
    if current_iteration < extension_deadline:
        return False

    snapshot = _compatibility_frontier_snapshot(task)
    _record_compatibility_frontier_diagnostics(
        task,
        snapshot,
        iteration=current_iteration,
        discovery_epoch=discovery_epoch,
    )
    _record_compatibility_frontier_shadow_diagnostics(
        task,
        iteration=current_iteration,
        discovery_epoch=discovery_epoch,
    )
    previous_class_ids = set(
        getattr(task, "last_compatibility_frontier_class_ids", ())
    )
    previous_max_rank = int(
        getattr(task, "last_compatibility_frontier_max_rank", 0)
    )
    seen_edges = getattr(task, "compatibility_frontier_seen_edges", None)
    if seen_edges is None:
        seen_edges = set()
        task.compatibility_frontier_seen_edges = seen_edges
    new_edges = snapshot.edge_fingerprints - seen_edges
    first_epoch_observation = int(
        getattr(task, "compatibility_frontier_epoch_observations", 0)
    ) == 0
    has_progress = (
        first_epoch_observation
        or snapshot.max_rank > previous_max_rank
        or bool(set(snapshot.class_ids) - previous_class_ids)
        or bool(new_edges)
    )

    seen_edges.update(snapshot.edge_fingerprints)
    task.compatibility_frontier_observations = int(
        getattr(task, "compatibility_frontier_observations", 0)
    ) + 1
    task.compatibility_frontier_epoch_observations = int(
        getattr(task, "compatibility_frontier_epoch_observations", 0)
    ) + 1
    task.last_compatibility_frontier_class_ids = snapshot.class_ids
    task.last_compatibility_frontier_max_rank = snapshot.max_rank
    task.last_compatibility_frontier_edge_count = snapshot.edge_count
    task.last_compatibility_frontier_new_edge_count = len(new_edges)

    if current_iteration >= int(task.iteration_limit):
        return True
    if not snapshot.class_ids or not has_progress:
        return True

    next_deadline = min(int(task.iteration_limit), current_iteration + extension)
    if next_deadline <= current_iteration:
        return True
    task.compatibility_frontier_extension_deadline = next_deadline
    task.compatibility_frontier_extensions = int(
        getattr(task, "compatibility_frontier_extensions", 0)
    ) + 1
    return False


def _compatibility_frontier_class_ids(task: InterruptSearchTask) -> tuple[int, ...]:
    """Return missing classes retained by the actionable near-terminal frontier."""
    return _compatibility_frontier_snapshot(task).class_ids


def _compatibility_frontier_snapshot(
    task: InterruptSearchTask,
    *,
    min_rank: int | None = None,
    max_rank_exclusive: int = 25,
    classify_out_of_band: bool = True,
) -> CompatibilityFrontierSnapshot:
    state = task.state
    configured_min_rank = (
        getattr(task, "compatibility_frontier_min_rank", 22)
        if min_rank is None
        else min_rank
    )
    rank_min = min(24, max(0, int(configured_min_rank)))
    rank_max_exclusive = min(
        25,
        max(rank_min + 1, int(max_rank_exclusive)),
    )
    bank = getattr(state, "compatibility_bank", None)
    target_classes = {
        int(class_id)
        for class_id in getattr(state.scorer, "target_classes", ())
    }
    discovered_classes = {
        int(class_id)
        for class_id in state.global_discovery.discovered_exact_classes
    }
    missing_classes = tuple(sorted(target_classes - discovered_classes))
    if bank is None:
        return CompatibilityFrontierSnapshot(
            class_ids=(),
            max_rank=0,
            edge_fingerprints=frozenset(),
            rank_min=rank_min,
            rank_max_exclusive=rank_max_exclusive,
            missing_class_ids=missing_classes,
            outcome_counts=(("snapshot_no_compatibility_bank", 1),),
        )
    if not target_classes:
        return CompatibilityFrontierSnapshot(
            class_ids=(),
            max_rank=0,
            edge_fingerprints=frozenset(),
            rank_min=rank_min,
            rank_max_exclusive=rank_max_exclusive,
            outcome_counts=(("snapshot_no_target_classes", 1),),
        )
    if not missing_classes:
        return CompatibilityFrontierSnapshot(
            class_ids=(),
            max_rank=0,
            edge_fingerprints=frozenset(),
            rank_min=rank_min,
            rank_max_exclusive=rank_max_exclusive,
            outcome_counts=(("snapshot_no_missing_classes", 1),),
        )

    target_class_ids = tuple(sorted(target_classes))
    missing_class_set = set(missing_classes)
    retained_classes: set[int] = set()
    edge_fingerprints: set[tuple[int, int, int, int]] = set()
    outcome_counts: Counter[str] = Counter()
    missing_class_compatible_action_counts = Counter(
        {int(class_id): 0 for class_id in missing_classes}
    )
    example_buckets: dict[str, list[CompatibilityFrontierDiagnosticExample]] = {}
    scanned_node_count = 0
    candidate_action_count = 0
    max_rank = 0

    def record_outcome(
        outcome: str,
        *,
        node_rank: int,
        selected_word: int,
        action: int | None = None,
        compatible_class_ids: Sequence[int] = (),
    ) -> None:
        outcome_counts[outcome] += 1
        examples = example_buckets.setdefault(outcome, [])
        if len(examples) >= _FRONTIER_DIAGNOSTIC_EXAMPLE_LIMIT:
            return
        examples.append(
            CompatibilityFrontierDiagnosticExample(
                outcome=outcome,
                node_rank=int(node_rank),
                selected_word=int(selected_word),
                action=None if action is None else int(action),
                compatible_class_ids=tuple(
                    sorted(int(class_id) for class_id in compatible_class_ids)
                ),
            )
        )

    for node in state.nodes.values():
        node_rank = int(node.rank)
        if not classify_out_of_band and not (
            rank_min <= node_rank < rank_max_exclusive
        ):
            continue
        scanned_node_count += 1
        selected_word = sum(
            1 << index
            for index, selected in enumerate(node.key)
            if int(selected)
        )
        if node.is_terminal:
            record_outcome(
                "node_terminal",
                node_rank=node_rank,
                selected_word=selected_word,
            )
            continue
        if node_rank < rank_min:
            record_outcome(
                "node_rank_below_min",
                node_rank=node_rank,
                selected_word=selected_word,
            )
            continue
        if node_rank >= rank_max_exclusive:
            record_outcome(
                (
                    "node_rank_at_or_above_facet"
                    if rank_max_exclusive == 25
                    else "node_rank_at_or_above_max"
                ),
                node_rank=node_rank,
                selected_word=selected_word,
            )
            continue
        if node.actions_initialized:
            actions = list(node.unexpanded_actions)
        else:
            actions = [
                action
                for action in unselected_blocks(node.key)
                if action not in node.children
            ]
        for action in node.children:
            record_outcome(
                "action_already_expanded",
                node_rank=node_rank,
                selected_word=selected_word,
                action=int(action),
            )
        if not actions:
            record_outcome(
                "node_no_unexpanded_actions",
                node_rank=node_rank,
                selected_word=selected_word,
            )
            continue
        record_outcome(
            "node_actionable",
            node_rank=node_rank,
            selected_word=selected_word,
        )
        for action in actions:
            candidate_action_count += 1
            child_key = add_block(node.key, action)
            target_compatible_classes = tuple(
                int(class_id)
                for class_id in bank.active_signature(child_key, target_class_ids)
            )
            compatible_classes = tuple(
                class_id
                for class_id in target_compatible_classes
                if class_id in missing_class_set
            )
            if compatible_classes:
                record_outcome(
                    "action_retained_missing_compatibility",
                    node_rank=node_rank,
                    selected_word=selected_word,
                    action=int(action),
                    compatible_class_ids=compatible_classes,
                )
                for class_id in compatible_classes:
                    missing_class_compatible_action_counts[int(class_id)] += 1
            elif target_compatible_classes:
                record_outcome(
                    "action_only_discovered_compatibility",
                    node_rank=node_rank,
                    selected_word=selected_word,
                    action=int(action),
                    compatible_class_ids=target_compatible_classes,
                )
                continue
            else:
                record_outcome(
                    "action_no_target_compatibility",
                    node_rank=node_rank,
                    selected_word=selected_word,
                    action=int(action),
                )
                continue
            max_rank = max(max_rank, node_rank)
            retained_classes.update(compatible_classes)
            edge_fingerprints.update(
                (node_rank, selected_word, int(action), int(class_id))
                for class_id in compatible_classes
            )
    return CompatibilityFrontierSnapshot(
        class_ids=tuple(sorted(retained_classes)),
        max_rank=max_rank,
        edge_fingerprints=frozenset(edge_fingerprints),
        rank_min=rank_min,
        rank_max_exclusive=rank_max_exclusive,
        missing_class_ids=missing_classes,
        scanned_node_count=scanned_node_count,
        candidate_action_count=candidate_action_count,
        outcome_counts=tuple(sorted(outcome_counts.items())),
        missing_class_compatible_action_counts=tuple(
            sorted(missing_class_compatible_action_counts.items())
        ),
        examples=tuple(
            example
            for outcome in sorted(example_buckets)
            for example in example_buckets[outcome]
        ),
    )


def _record_compatibility_frontier_diagnostics(
    task: InterruptSearchTask,
    snapshot: CompatibilityFrontierSnapshot,
    *,
    iteration: int,
    discovery_epoch: int,
) -> None:
    aggregate = Counter(
        getattr(task, "compatibility_frontier_outcome_counts", {})
    )
    aggregate.update(dict(snapshot.outcome_counts))
    task.compatibility_frontier_outcome_counts = aggregate

    observations = getattr(
        task,
        "compatibility_frontier_observation_details",
        None,
    )
    if observations is None:
        observations = []
        task.compatibility_frontier_observation_details = observations
    observations.append(
        snapshot.diagnostic_dict(
            iteration=iteration,
            discovery_epoch=discovery_epoch,
        )
    )


def _record_compatibility_frontier_shadow_diagnostics(
    task: InterruptSearchTask,
    *,
    iteration: int,
    discovery_epoch: int,
) -> None:
    """Observe lower-rank compatibility without changing the budget frontier."""
    budget_min_rank = min(
        24,
        max(0, int(getattr(task, "compatibility_frontier_min_rank", 22))),
    )
    shadow_ranks = tuple(
        sorted(
            {
                int(rank)
                for rank in getattr(
                    task,
                    "compatibility_frontier_shadow_ranks",
                    (20, 21),
                )
                if 0 <= int(rank) < budget_min_rank
            }
        )
    )
    aggregates = getattr(
        task,
        "compatibility_frontier_shadow_outcome_counts",
        None,
    )
    if aggregates is None:
        aggregates = {}
        task.compatibility_frontier_shadow_outcome_counts = aggregates
    observations = getattr(
        task,
        "compatibility_frontier_shadow_observation_details",
        None,
    )
    if observations is None:
        observations = []
        task.compatibility_frontier_shadow_observation_details = observations

    for rank in shadow_ranks:
        snapshot = _compatibility_frontier_snapshot(
            task,
            min_rank=rank,
            max_rank_exclusive=rank + 1,
            classify_out_of_band=False,
        )
        aggregate = Counter(aggregates.get(rank, {}))
        aggregate.update(dict(snapshot.outcome_counts))
        aggregates[rank] = aggregate
        detail = snapshot.diagnostic_dict(
            iteration=iteration,
            discovery_epoch=discovery_epoch,
        )
        detail["shadow_rank"] = int(rank)
        observations.append(detail)


def _reset_compatibility_frontier_epoch(
    task: InterruptSearchTask,
    discovery_epoch: int,
) -> None:
    task.compatibility_frontier_discovery_epoch = int(discovery_epoch)
    task.compatibility_frontier_extension_deadline = 0
    task.compatibility_frontier_epoch_observations = 0
    task.compatibility_frontier_seen_edges = set()
    task.last_compatibility_frontier_class_ids = ()
    task.last_compatibility_frontier_max_rank = 0
    task.last_compatibility_frontier_edge_count = 0
    task.last_compatibility_frontier_new_edge_count = 0


def _should_stop_as_terminal_sink(task: InterruptSearchTask) -> bool:
    """Stop roots that are empirically only producing themselves or invalids."""
    state = task.state
    if state.productive:
        return False
    if state.iterations_completed < task.adaptive_min_iterations:
        return False
    if any(class_id != task.class_id for class_id in state.local_discovered_exact_classes):
        return False

    terminal_hits = sum(int(count) for count in state.encountered.values())
    if terminal_hits < task.adaptive_min_terminal_hits:
        return False

    self_label = f"exact:class{int(task.class_id)}"
    self_hits = int(state.encountered.get(self_label, 0))
    invalid_hits = sum(
        int(count)
        for label, count in state.encountered.items()
        if str(label).startswith("invalid:")
    )
    sink_ratio = float(self_hits + invalid_hits) / float(max(1, terminal_hits))
    return sink_ratio >= float(task.adaptive_sink_ratio)


def run_interruptible_search(
    config: InterruptSearchConfig,
    *,
    task_factory: Callable[[int, int, int | None], InterruptSearchTask] | None = None,
    global_discovery_state: InterruptGlobalDiscoveryState | None = None,
    deduplicate_patterns: bool | None = None,
    output_path: Path | None = None,
) -> InterruptSearchReport:
    global_discovery = (
        InterruptGlobalDiscoveryState.from_config(config)
        if global_discovery_state is None
        else global_discovery_state
    )
    current_rare_target_classes = global_discovery.remaining_rare_target_classes
    started_class_ids: set[int] = set()
    finished_class_ids: set[int] = set()
    stack: list[InterruptSearchTask] = []
    interrupt_events: list[InterruptDiscoveryEvent] = []
    runs: list[InterruptSearchRun] = []
    exact_class_counts: Counter[int] = Counter()
    search_index = 0
    supervisor_iterations = 0
    discovery_event_index = 0
    using_default_task_factory = task_factory is None
    task_factory = task_factory or _default_task_factory(config, global_discovery)
    run_log_path = None if output_path is None else output_path.with_name(f"{output_path.stem}.runs.json")
    first_run_record = True

    if run_log_path is not None:
        _start_json_array(run_log_path)

    enable_pattern_dedup = (
        using_default_task_factory
        if deduplicate_patterns is None
        else bool(deduplicate_patterns)
    )

    # A partition may contain several symmetry-inequivalent facet basins. Only
    # an identical (partition, basin) pair is deduplicated; different basins get
    # independent trees even when they use the same block partition.
    claimed_partitions: set[PartitionKey] = set()
    claimed_task_identities: set[SearchTaskIdentity] = set()

    # register initial task and its pattern signature
    _scorer, initial_pattern = build_class_scorer(
        _queue_like_config(config),
        int(config.initial_class_id),
        rare_target_classes=current_rare_target_classes,
        terminal_validation_cache=global_discovery.terminal_validation_cache,
        structure_cache=global_discovery.structure_cache,
    )
    initial_pattern = _canonical_pattern_signature(initial_pattern)
    initial_identity = _search_task_identity(config, config.initial_class_id, initial_pattern)
    claimed_partitions.add(initial_identity.partition_key)
    claimed_task_identities.add(initial_identity)
    initial_task = task_factory(int(config.initial_class_id), 1, None)
    stack.append(initial_task)
    started_class_ids.add(int(config.initial_class_id))

    while stack:
        task = stack[-1]
        supervisor_iterations += 1
        discoveries = task.step()
        pushed_child = False

        for discovery in discoveries:
            global_discovery.mark_discovered(discovery.class_id)
            exact_class_counts[discovery.class_id] += 1
            rare_removed = (
                discovery.class_id in global_discovery.initial_rare_target_classes
                and exact_class_counts[discovery.class_id] == 1
            )
            if discovery.class_id in started_class_ids:
                continue

            _, candidate_pattern_signature = build_class_scorer(
                _queue_like_config(config),
                int(discovery.class_id),
                rare_target_classes=current_rare_target_classes,
                terminal_validation_cache=global_discovery.terminal_validation_cache,
                structure_cache=global_discovery.structure_cache,
            )
            candidate_pattern_signature = _canonical_pattern_signature(
                candidate_pattern_signature
            )
            candidate_identity = _search_task_identity(
                config,
                discovery.class_id,
                candidate_pattern_signature,
            )
            same_partition = candidate_identity.partition_key in claimed_partitions
            same_task_identity = (
                candidate_identity in claimed_task_identities
                if enable_pattern_dedup
                else False
            )

            if not same_task_identity:
                claimed_partitions.add(candidate_identity.partition_key)
                claimed_task_identities.add(candidate_identity)
                search_index += 1
                child_task = task_factory(
                    discovery.class_id,
                    search_index + 1,
                    task.search_index,
                )
                stack.append(child_task)
                started_class_ids.add(discovery.class_id)
                task.started_interrupts.append(discovery.class_id)
                pattern_status = (
                    "new_basin_same_partition"
                    if same_partition
                    else "unique_partition"
                )
                pushed_child = True
            else:
                pattern_status = "same_task_identity"

            discovery_event_index += 1
            interrupt_events.append(
                InterruptDiscoveryEvent(
                    event_index=discovery_event_index,
                    global_iteration=supervisor_iterations,
                    search_index=task.search_index,
                    parent_search_index=task.parent_search_index,
                    start_class_id=task.class_id,
                    class_id=discovery.class_id,
                    iteration=discovery.iteration,
                    depth=discovery.depth,
                    score=discovery.score,
                    rank=discovery.rank,
                    label=discovery.label,
                    path=list(discovery.path),
                    chosen_blocks=list(discovery.chosen_blocks),
                    pattern_status=pattern_status,
                    pattern_signature=[
                        list(block) for block in candidate_pattern_signature
                    ],
                    basin_key=f"0x{candidate_identity.basin_key:016x}",
                    rare_removed=rare_removed,
                )
            )
            if output_path is not None:
                _write_json_snapshot(
                    output_path,
                    _build_snapshot_payload(
                        config=config,
                        started_class_ids=started_class_ids,
                        finished_class_ids=finished_class_ids,
                        interrupt_events=interrupt_events,
                        runs=runs,
                        exact_class_counts=exact_class_counts,
                        stop_reason="running",
                        run_log_path=run_log_path,
                    ),
                )
            break

        if pushed_child:
            continue

        if task.finished:
            finished_class_ids.add(task.class_id)
            task_result = task.result()
            run_record = InterruptSearchRun(
                search_index=task.search_index,
                parent_search_index=task.parent_search_index,
                start_class_id=task.class_id,
                seed=task.state.config.seed,
                iteration_limit=task.iteration_limit,
                stop_reason=task.state.stop_reason,
                productive=bool(getattr(task.state, "productive", False)),
                cross_class_count=sum(
                    1
                    for class_id in getattr(
                        task.state,
                        "local_discovered_exact_classes",
                        set(),
                    )
                    if int(class_id) != int(task.class_id)
                ),
                last_new_class_iteration=getattr(
                    task,
                    "last_new_class_iteration",
                    None,
                ),
                last_global_new_iteration=getattr(
                    task,
                    "last_global_new_iteration",
                    None,
                ),
                compatibility_frontier_extensions=int(
                    getattr(task, "compatibility_frontier_extensions", 0)
                ),
                compatibility_frontier_extension_deadline=int(
                    getattr(task, "compatibility_frontier_extension_deadline", 0)
                ),
                compatibility_frontier_observations=int(
                    getattr(task, "compatibility_frontier_observations", 0)
                ),
                last_compatibility_frontier_class_ids=list(
                    getattr(task, "last_compatibility_frontier_class_ids", ())
                ),
                last_compatibility_frontier_max_rank=int(
                    getattr(task, "last_compatibility_frontier_max_rank", 0)
                ),
                last_compatibility_frontier_edge_count=int(
                    getattr(task, "last_compatibility_frontier_edge_count", 0)
                ),
                last_compatibility_frontier_new_edge_count=int(
                    getattr(task, "last_compatibility_frontier_new_edge_count", 0)
                ),
                compatibility_frontier_outcome_counts={
                    str(outcome): int(count)
                    for outcome, count in Counter(
                        getattr(task, "compatibility_frontier_outcome_counts", {})
                    ).items()
                },
                compatibility_frontier_observation_details=list(
                    getattr(task, "compatibility_frontier_observation_details", ())
                ),
                compatibility_frontier_shadow_outcome_counts={
                    str(rank): {
                        str(outcome): int(count)
                        for outcome, count in Counter(counts).items()
                    }
                    for rank, counts in getattr(
                        task,
                        "compatibility_frontier_shadow_outcome_counts",
                        {},
                    ).items()
                },
                compatibility_frontier_shadow_observation_details=list(
                    getattr(
                        task,
                        "compatibility_frontier_shadow_observation_details",
                        (),
                    )
                ),
                result=task_result.to_dict(),
                interrupted_by_class_ids=list(task.started_interrupts),
                resumed_after_class_ids=list(task.resumed_interrupts),
            )
            runs.append(run_record)
            if run_log_path is not None:
                _append_json_array_record(
                    run_log_path,
                    run_record.to_dict(),
                    first_record=first_run_record,
                )
                first_run_record = False
            release_tree = getattr(task.state, "release_tree", None)
            if callable(release_tree):
                release_tree()
            stack.pop()
            if stack:
                stack[-1].resumed_interrupts.append(task.class_id)

    stop_reason = "stack_empty"
    report = _build_report(
        config=config,
        started_class_ids=started_class_ids,
        finished_class_ids=finished_class_ids,
        interrupt_events=interrupt_events,
        runs=runs,
        exact_class_counts=exact_class_counts,
        stop_reason=stop_reason,
    )
    if output_path is not None:
        _write_json_snapshot(output_path, _snapshot_payload_from_report(report, run_log_path=run_log_path))
    if run_log_path is not None:
        _close_json_array(run_log_path)
    return report


def _snapshot_payload_from_report(
    report: InterruptSearchReport,
    *,
    run_log_path: Path | None,
) -> dict[str, object]:
    return {
        "meta": report.meta,
        "started_class_ids": report.started_class_ids,
        "finished_class_ids": report.finished_class_ids,
        "interrupt_events": [event.to_dict() for event in report.interrupt_events],
        "discovery_timeline": _discovery_timeline(report.interrupt_events),
        "summary": report.summary,
        "stop_reason": report.stop_reason,
        "exact_class_counts": report.exact_class_counts,
        "run_log_path": None if run_log_path is None else str(run_log_path),
    }


def _build_snapshot_payload(
    *,
    config: InterruptSearchConfig,
    started_class_ids: set[int],
    finished_class_ids: set[int],
    interrupt_events: list[InterruptDiscoveryEvent],
    runs: list[InterruptSearchRun],
    exact_class_counts: Counter[int],
    stop_reason: str,
    run_log_path: Path | None,
) -> dict[str, object]:
    report = _build_report(
        config=config,
        started_class_ids=started_class_ids,
        finished_class_ids=finished_class_ids,
        interrupt_events=interrupt_events,
        runs=runs,
        exact_class_counts=exact_class_counts,
        stop_reason=stop_reason,
    )
    return _snapshot_payload_from_report(report, run_log_path=run_log_path)


def _discovery_timeline(events: Sequence[InterruptDiscoveryEvent]) -> list[dict[str, object]]:
    return [
        {
            "event_index": event.event_index,
            "global_iteration": event.global_iteration,
            "search_index": event.search_index,
            "start_class_id": event.start_class_id,
            "class_id": event.class_id,
            "local_iteration": event.iteration,
            "depth": event.depth,
            "pattern_status": event.pattern_status,
            "rare_removed": event.rare_removed,
        }
        for event in events
    ]


def _build_report(
    *,
    config: InterruptSearchConfig,
    started_class_ids: set[int],
    finished_class_ids: set[int],
    interrupt_events: list[InterruptDiscoveryEvent],
    runs: list[InterruptSearchRun],
    exact_class_counts: Counter[int],
    stop_reason: str,
) -> InterruptSearchReport:
    # Build summary from run payloads to avoid holding a second copy of results.
    label_counts: Counter[str] = Counter()
    encountered: Counter[str] = Counter()
    exact_counts: Counter[int] = Counter()
    for run in runs:
        result = run.result
        best = result.get("best") if isinstance(result, dict) else getattr(result, "best", None)
        if best is not None:
            label = best.get("label") if isinstance(best, dict) else getattr(best, "label", None)
            if label is not None:
                class_id = exact_class_id(str(label))
                compact = f"exact:class{class_id}" if class_id is not None else label
                label_counts[compact] += 1
        encountered_counts = result.get("encountered_label_counts", {}) if isinstance(result, dict) else getattr(result, "encountered_label_counts", {}) or {}
        for label, count in encountered_counts.items():
            compact = f"exact:class{exact_class_id(str(label))}" if exact_class_id(str(label)) is not None else label
            encountered[compact] += int(count)
        exact_ids = result.get("exact_class_ids", []) if isinstance(result, dict) else getattr(result, "exact_class_ids", []) or []
        for cid in exact_ids:
            exact_counts[int(cid)] += 1
    opened_rare = sorted(
        class_id
        for label in label_counts
        if (class_id := exact_class_id(label)) in set(int(value) for value in config.rare_target_classes)
    )
    encountered_rare = sorted(
        class_id
        for label, count in encountered.items()
        if count > 0 and (class_id := exact_class_id(label)) in set(int(value) for value in config.rare_target_classes)
    )
    summary = {
        "label_counts": dict(sorted(label_counts.items())),
        "exact_class_counts": {str(key): value for key, value in sorted(exact_counts.items())},
        "opened_rare_target_classes": opened_rare,
        "rare_target_coverage_count": len(opened_rare),
        "encountered_label_counts": dict(sorted(encountered.items())),
        "encountered_rare_target_classes": encountered_rare,
        "encountered_rare_target_coverage_count": len(encountered_rare),
    }
    meta = {
        "script": "src/mcts/interrupt_search.py",
        "initial_class_id": int(config.initial_class_id),
        "rep_index": int(config.rep_index),
        "pattern_index": int(config.pattern_index),
        "examples_path": str(config.examples_path),
        "target_classes": [int(value) for value in sorted(config.target_classes)],
        "rare_target_classes": [int(value) for value in sorted(config.rare_target_classes)],
        "iterations": int(config.iterations),
        "max_depth": int(config.max_depth),
        "exploration_constant": float(config.exploration_constant),
        "discount": float(config.discount),
        "prior_temperature": float(config.prior_temperature),
        "rollout_temperature": float(config.rollout_temperature),
        "expansion_candidate_pool": int(config.expansion_candidate_pool),
        "rollout_candidate_pool": int(config.rollout_candidate_pool),
        "widening_score_batch": int(config.widening_score_batch),
        "widening_score_batch_max": int(config.widening_score_batch_max),
        "widening_score_batch_scale": float(config.widening_score_batch_scale),
        "widening_score_batch_beta": float(config.widening_score_batch_beta),
        "rollout_score_batch": int(config.rollout_score_batch),
        "productive_rollout_score_batch": int(config.productive_rollout_score_batch),
        "broad_rollout_score_batch": int(config.broad_rollout_score_batch),
        "broad_rollout_interval": int(config.broad_rollout_interval),
        "progressive_k0": int(config.progressive_k0),
        "progressive_alpha": float(config.progressive_alpha),
        "progressive_beta": float(config.progressive_beta),
        "progressive_bucket_quota": int(config.progressive_bucket_quota),
        "discovery_epoch_value_decay": float(config.discovery_epoch_value_decay),
        "seed": int(config.seed),
        "rank24_entrance_exists_weight": float(config.rank24_entrance_exists_weight),
        "terminal_scoring_mode": str(config.terminal_scoring_mode),
        "dynamic_new_class_score": float(config.dynamic_new_class_score),
        "dynamic_known_class_score": float(config.dynamic_known_class_score),
        "dynamic_frequent_class_score": float(config.dynamic_frequent_class_score),
        "dynamic_frequent_class_threshold": int(config.dynamic_frequent_class_threshold),
        "exact_stop_ratio": float(config.exact_stop_ratio),
        "adaptive_min_iterations": int(config.adaptive_min_iterations),
        "adaptive_min_terminal_hits": int(config.adaptive_min_terminal_hits),
        "adaptive_sink_ratio": float(config.adaptive_sink_ratio),
        "adaptive_extra_iterations": int(config.adaptive_extra_iterations),
        "adaptive_max_multiplier": float(config.adaptive_max_multiplier),
        "productive_patience_iterations": int(config.productive_patience_iterations),
        "compatibility_frontier_extension_iterations": int(
            config.compatibility_frontier_extension_iterations
        ),
        "compatibility_frontier_min_rank": int(config.compatibility_frontier_min_rank),
        "compatibility_frontier_shadow_ranks": [
            int(rank) for rank in config.compatibility_frontier_shadow_ranks
        ],
        "task_identity": "group_canonical_partition_and_basin",
        "performance": {
            "flat_capacity_method": "affine_hull",
            "known_facet_classifier": "tight_support_index",
            "structure_cache_hot_partitions": 4,
        },
    }
    return InterruptSearchReport(
        meta=meta,
        started_class_ids=sorted(started_class_ids),
        finished_class_ids=sorted(finished_class_ids),
        interrupt_events=interrupt_events,
        runs=runs,
        summary=summary,
        stop_reason=stop_reason,
        exact_class_counts={str(key): value for key, value in sorted(exact_class_counts.items())},
    )


def _default_task_factory(
    config: InterruptSearchConfig,
    global_discovery: InterruptGlobalDiscoveryState,
) -> Callable[[int, int, int | None], InterruptSearchTask]:
    def factory(class_id: int, search_index: int, parent_search_index: int | None) -> InterruptSearchTask:
        scorer, _pattern_signature = build_class_scorer(
            _queue_like_config(config),
            class_id,
            rare_target_classes=global_discovery.active_rare_target_classes(),
            terminal_validation_cache=global_discovery.terminal_validation_cache,
            structure_cache=global_discovery.structure_cache,
        )
        mcts_config = MCTSConfig(
            iterations=int(config.iterations),
            max_depth=int(config.max_depth),
            exploration_constant=float(config.exploration_constant),
            discount=float(config.discount),
            prior_temperature=float(config.prior_temperature),
            rollout_temperature=float(config.rollout_temperature),
            expansion_candidate_pool=int(config.expansion_candidate_pool),
            rollout_candidate_pool=int(config.rollout_candidate_pool),
            widening_score_batch=int(config.widening_score_batch),
            widening_score_batch_max=int(config.widening_score_batch_max),
            widening_score_batch_scale=float(config.widening_score_batch_scale),
            widening_score_batch_beta=float(config.widening_score_batch_beta),
            rollout_score_batch=int(config.rollout_score_batch),
            productive_rollout_score_batch=int(config.productive_rollout_score_batch),
            broad_rollout_score_batch=int(config.broad_rollout_score_batch),
            broad_rollout_interval=int(config.broad_rollout_interval),
            progressive_k0=int(config.progressive_k0),
            progressive_alpha=float(config.progressive_alpha),
            progressive_beta=float(config.progressive_beta),
            progressive_bucket_quota=int(config.progressive_bucket_quota),
            discovery_epoch_value_decay=float(config.discovery_epoch_value_decay),
            seed=int(config.seed) + 1009 * (search_index - 1),
            selection_survival_weight=1.0,
            selection_novelty_weight=1.0,
            compatibility_examples_path=config.examples_path,
        )
        state = InterruptSearchState(
            scorer=scorer,
            config=mcts_config,
            global_discovery=global_discovery,
            compatibility_bank=ClassCompatibilityBank.from_scorer(scorer, config.examples_path),
        )
        root_key = empty_key(len(scorer.blocks))
        state.root = _get_or_create_node(state.nodes, scorer, root_key, path=[])
        base_limit = int(config.iterations)
        max_limit = max(base_limit, int(round(base_limit * float(config.adaptive_max_multiplier))))
        threshold = float(config.exact_stop_ratio) * float(max_limit)
        compatibility_frontier_min_rank = min(
            24,
            max(0, int(config.compatibility_frontier_min_rank)),
        )
        return InterruptSearchTask(
            class_id=class_id,
            search_index=search_index,
            parent_search_index=parent_search_index,
            state=state,
            threshold=threshold,
            base_iteration_limit=base_limit,
            max_iteration_limit=max_limit,
            adaptive_min_iterations=min(base_limit, max(1, int(config.adaptive_min_iterations))),
            adaptive_min_terminal_hits=max(1, int(config.adaptive_min_terminal_hits)),
            adaptive_sink_ratio=float(config.adaptive_sink_ratio),
            adaptive_extra_iterations=max(1, int(config.adaptive_extra_iterations)),
            productive_patience_iterations=max(0, int(config.productive_patience_iterations)),
            iteration_limit=base_limit,
            compatibility_frontier_extension_iterations=max(
                0,
                int(config.compatibility_frontier_extension_iterations),
            ),
            compatibility_frontier_min_rank=compatibility_frontier_min_rank,
            compatibility_frontier_shadow_ranks=tuple(
                sorted(
                    {
                        int(rank)
                        for rank in config.compatibility_frontier_shadow_ranks
                        if 0 <= int(rank) < compatibility_frontier_min_rank
                    }
                )
            ),
        )

    return factory


def _queue_like_config(config: InterruptSearchConfig) -> object:
    return type(
        "QueueLikeConfig",
        (),
        {
            "examples_path": config.examples_path,
            "rep_index": config.rep_index,
            "pattern_index": config.pattern_index,
            "rare_target_classes": config.rare_target_classes,
            "target_classes": config.target_classes,
            "rank24_entrance_exists_weight": config.rank24_entrance_exists_weight,
            "terminal_scoring_mode": config.terminal_scoring_mode,
            "dynamic_new_class_score": config.dynamic_new_class_score,
            "dynamic_known_class_score": config.dynamic_known_class_score,
            "dynamic_frequent_class_score": config.dynamic_frequent_class_score,
            "dynamic_frequent_class_threshold": config.dynamic_frequent_class_threshold,
        },
    )()


def _canonical_pattern_signature(signature: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Normalize a pattern signature so semantically identical block sets compare equal."""
    return tuple(sorted(tuple(sorted(block)) for block in signature))


def _search_task_identity(
    config: InterruptSearchConfig,
    class_id: int,
    pattern_signature: tuple[tuple[int, ...], ...],
) -> SearchTaskIdentity:
    """Build the two-level identity used for task deduplication."""
    rows = parse_example_rows(config.examples_path)
    class_rows = rows.get(int(class_id))
    if class_rows is None:
        raise KeyError(f"unknown exact class {class_id}")
    row = class_rows.get(int(config.rep_index))
    if row is None:
        available = ", ".join(str(key) for key in sorted(class_rows))
        raise KeyError(
            f"class {class_id} has no rep_index {config.rep_index}; available: {available}"
        )
    return SearchTaskIdentity(
        partition_key=canonical_partition_key(pattern_signature),
        basin_key=canonical_support_key(support_mask_from_row(row)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stack-based interruptible MCTS exploration.")
    parser.add_argument("--initial-class-id", type=int, default=1)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--rare-target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=60)
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
    parser.add_argument("--progressive-bucket-quota", type=int, default=1)
    parser.add_argument("--discovery-epoch-value-decay", type=float, default=0.25)
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
    parser.add_argument("--compatibility-frontier-extension-iterations", type=int, default=50)
    parser.add_argument("--compatibility-frontier-min-rank", type=int, default=22)
    parser.add_argument(
        "--compatibility-frontier-shadow-ranks",
        type=int,
        nargs="*",
        default=[20, 21],
    )
    parser.add_argument("--output", type=Path, default=Path("src/mcts/runs/interrupt_search_probe.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InterruptSearchConfig(
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
        progressive_bucket_quota=int(args.progressive_bucket_quota),
        discovery_epoch_value_decay=float(args.discovery_epoch_value_decay),
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
        compatibility_frontier_extension_iterations=int(
            args.compatibility_frontier_extension_iterations
        ),
        compatibility_frontier_min_rank=int(args.compatibility_frontier_min_rank),
        compatibility_frontier_shadow_ranks=tuple(
            int(rank) for rank in args.compatibility_frontier_shadow_ranks
        ),
    )
    started = time.perf_counter()
    report = run_interruptible_search(config, output_path=args.output)
    elapsed_seconds = time.perf_counter() - started
    payload = report.to_dict()
    payload["meta"]["elapsed_seconds"] = elapsed_seconds
    _write_json_snapshot(args.output, payload)
    print(json.dumps({"stop_reason": payload["stop_reason"], "finished_class_ids": payload["finished_class_ids"]}, ensure_ascii=False))
    print(f"elapsed_seconds={elapsed_seconds:.3f}")
    print(args.output)
    print(args.output.with_name(f"{args.output.stem}.runs.json"))


if __name__ == "__main__":
    main()
