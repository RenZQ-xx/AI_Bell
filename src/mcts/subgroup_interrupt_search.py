from __future__ import annotations

"""Unified class-interrupt and subgroup-pattern MCTS for Bell 322.

The search has two coupled frontiers:

* class tasks retain the LIFO interrupt behavior of ``interrupt_search``;
* subgroup tasks are scheduled fairly from the facet-free minimal-overgroup
  atlas and widened to level two only after productive observations.

Every subgroup inferred for a class comes from the terminal support that was
actually reached by search.  Reference examples are used for terminal class
validation and for the explicitly supplied initial class only.
"""

import argparse
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from baseline.orbit_blocks import (
    BlockKey,
    PartitionKey,
    build_orbit_patterns_from_support,
    canonical_partition_key,
)
from baseline.reference_classes import (
    DEFAULT_EXAMPLES_PATH,
    parse_example_rows,
    support_mask_from_row,
    support_word_from_indices,
)
from baseline.scorer import ExpansionScorer, ScorerConfig
from mcts.interrupt_search import (
    InterruptGlobalDiscoveryState,
    InterruptSearchConfig,
    InterruptSearchTask,
    _write_json_snapshot,
    build_interrupt_task_from_scorer,
)
from mcts.subgroup_patterns import (
    SubgroupGrowthEdge,
    SubgroupPattern,
    SubgroupPatternAtlas,
    SupportPatternAnalysis,
    load_subgroup_pattern_atlas,
    support_word_from_mask,
)


DEFAULT_ATOMIC_TAIL_CANDIDATES = 512


@dataclass(frozen=True)
class SubgroupInterruptSearchConfig:
    interrupt: InterruptSearchConfig = field(
        default_factory=lambda: InterruptSearchConfig(iterations=200)
    )
    atlas_path: Path | None = None
    bootstrap_minimal_roots: bool = True
    bootstrap_root_limit: int = 24
    root_iterations: int | None = None
    level2_iterations: int | None = None
    class_step_burst: int = 8
    max_pattern_tasks: int = 80
    growth_children_per_discovery: int = 2
    observed_level2_per_discovery: int = 2
    include_level2_support_analysis: bool = True
    max_level2_pair_checks: int = 1024
    max_level2_matches: int = 64
    subgroup_warmup_iterations: int = 50
    identity_tree_count: int = 4

    def __post_init__(self) -> None:
        if self.bootstrap_root_limit < 0:
            raise ValueError("bootstrap_root_limit must be nonnegative")
        if self.root_iterations is not None and self.root_iterations <= 0:
            raise ValueError("root_iterations must be positive")
        if self.level2_iterations is not None and self.level2_iterations <= 0:
            raise ValueError("level2_iterations must be positive")
        if self.class_step_burst <= 0:
            raise ValueError("class_step_burst must be positive")
        if self.max_pattern_tasks < 0:
            raise ValueError("max_pattern_tasks must be nonnegative")
        if self.growth_children_per_discovery < 0:
            raise ValueError("growth_children_per_discovery must be nonnegative")
        if self.observed_level2_per_discovery < 0:
            raise ValueError("observed_level2_per_discovery must be nonnegative")
        if self.max_level2_pair_checks < 0:
            raise ValueError("max_level2_pair_checks must be nonnegative")
        if self.max_level2_matches < 0:
            raise ValueError("max_level2_matches must be nonnegative")
        if self.subgroup_warmup_iterations <= 0:
            raise ValueError("subgroup_warmup_iterations must be positive")
        if self.identity_tree_count <= 0:
            raise ValueError("identity_tree_count must be positive")


@dataclass(frozen=True)
class PatternJob:
    pattern_id: str
    anchor_class_id: int
    source_class_id: int
    reason: str
    parent_pattern_id: str | None = None
    growth_relation: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern_id": self.pattern_id,
            "anchor_class_id": self.anchor_class_id,
            "source_class_id": self.source_class_id,
            "reason": self.reason,
            "parent_pattern_id": self.parent_pattern_id,
            "growth_relation": self.growth_relation,
        }


@dataclass
class SearchFrame:
    frame_id: int
    kind: str
    task: InterruptSearchTask
    anchor_class_id: int
    source_class_id: int
    support_word: int | None
    pattern_id: str | None = None
    reason: str = ""
    parent_pattern_id: str | None = None
    growth_relation: str | None = None
    ensemble_index: int | None = None
    racing_warmup_limit: int | None = None
    racing_promoted: bool = False
    started_at: float = field(default_factory=time.perf_counter)
    discovery_class_ids: list[int] = field(default_factory=list)
    step_calls: int = 0
    suspended_step_calls: int = 0
    step_seconds: float = 0.0
    max_step_seconds: float = 0.0
    max_step_iteration: int = 0
    steps_over_one_second: int = 0
    steps_over_five_seconds: int = 0

    def to_run_dict(self) -> dict[str, object]:
        result = self.task.result()
        return {
            "frame_id": self.frame_id,
            "kind": self.kind,
            "anchor_class_id": self.anchor_class_id,
            "source_class_id": self.source_class_id,
            "support_word_hex": (
                None
                if self.support_word is None
                else f"0x{self.support_word:016x}"
            ),
            "pattern_id": self.pattern_id,
            "reason": self.reason,
            "parent_pattern_id": self.parent_pattern_id,
            "growth_relation": self.growth_relation,
            "ensemble_index": self.ensemble_index,
            "racing_warmup_limit": self.racing_warmup_limit,
            "racing_promoted": self.racing_promoted,
            "seed": self.task.state.config.seed,
            "iteration_limit": self.task.iteration_limit,
            "iterations_completed": self.task.state.iterations_completed,
            "stop_reason": self.task.state.stop_reason,
            "productive": self.task.state.productive,
            "discovery_class_ids": list(self.discovery_class_ids),
            "elapsed_seconds": time.perf_counter() - self.started_at,
            "step_timing": {
                "calls": self.step_calls,
                "suspended_calls": self.suspended_step_calls,
                "seconds": self.step_seconds,
                "average_seconds": (
                    self.step_seconds / self.step_calls
                    if self.step_calls > 0
                    else 0.0
                ),
                "max_seconds": self.max_step_seconds,
                "max_local_iteration": self.max_step_iteration,
                "steps_over_one_second": self.steps_over_one_second,
                "steps_over_five_seconds": self.steps_over_five_seconds,
            },
            "rank23_tail": {
                "prefixes": len(self.task.state.rank23_tail_prefixes),
                "cache_hits": self.task.state.rank23_tail_cache_hits,
                "logical_candidates": self.task.state.rank23_tail_logical_candidates,
                "pair_candidates_processed": (
                    self.task.state.rank23_tail_pair_candidates_processed
                ),
                "full_rank_terminals": self.task.state.rank23_tail_full_rank_terminals,
                "exact_supports": self.task.state.rank23_tail_exact_supports,
                "completed_prefixes": self.task.state.rank23_tail_completed_prefixes,
                "shared_completed_prefixes": (
                    self.task.state.rank23_tail_shared_completed_prefixes
                ),
                "partial_prefixes": len(self.task.state.rank23_tail_progress),
                "pair_candidate_capacity": (
                    self.task.state.rank23_tail_pair_candidate_capacity
                ),
                "pending_prefixes": len(self.task.state.rank23_tail_pending_set),
                "pending_peak": self.task.state.rank23_tail_pending_peak,
                "batches": self.task.state.rank23_tail_batches,
                "active_service_batches": (
                    self.task.state.rank23_tail_active_service_batches
                ),
                "batch_seconds": self.task.state.rank23_tail_batch_seconds,
                "active_service_seconds": (
                    self.task.state.rank23_tail_active_service_seconds
                ),
                "max_batch_seconds": self.task.state.rank23_tail_max_batch_seconds,
                "max_batch_candidates": (
                    self.task.state.rank23_tail_max_batch_candidates
                ),
                "discovery_batches": self.task.state.rank23_tail_discovery_batches,
                "deferred_iterations": (
                    self.task.state.rank23_tail_deferred_iterations
                ),
            },
            "result": result.to_dict(),
        }


@dataclass(frozen=True)
class UnifiedDiscoveryEvent:
    event_index: int
    class_id: int
    wall_seconds: float
    scheduler_iteration: int
    local_iteration: int
    source_frame_id: int
    source_kind: str
    source_class_id: int
    source_pattern_id: str | None
    support_word: int
    chosen_blocks: tuple[int, ...]
    score: float
    rank: int
    pattern_analysis: SupportPatternAnalysis

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "class_id": self.class_id,
            "wall_seconds": self.wall_seconds,
            "scheduler_iteration": self.scheduler_iteration,
            "local_iteration": self.local_iteration,
            "source_frame_id": self.source_frame_id,
            "source_kind": self.source_kind,
            "source_class_id": self.source_class_id,
            "source_pattern_id": self.source_pattern_id,
            "support_word_hex": f"0x{self.support_word:016x}",
            "chosen_blocks": list(self.chosen_blocks),
            "score": self.score,
            "rank": self.rank,
            "pattern_analysis": self.pattern_analysis.to_dict(),
        }


class PatternFrontier:
    """Deduplicated pending subgroup jobs keyed by canonical partitions."""

    def __init__(self, atlas: SubgroupPatternAtlas, max_tasks: int) -> None:
        self.atlas = atlas
        self.max_tasks = max(0, int(max_tasks))
        self.pending: deque[PatternJob] = deque()
        self.claimed_pattern_ids: set[str] = set()
        self.claimed_partitions: set[PartitionKey] = set()
        self.partition_by_pattern_id: dict[str, PartitionKey] = {}
        self.skipped: list[dict[str, object]] = []

    def _partition_key(self, pattern: SubgroupPattern) -> PartitionKey:
        cached = self.partition_by_pattern_id.get(pattern.pattern_id)
        if cached is None:
            cached = (
                canonical_partition_key(pattern.blocks)
                if pattern.canonical_partition is None
                else pattern.canonical_partition
            )
            self.partition_by_pattern_id[pattern.pattern_id] = cached
        return cached

    def enqueue(self, job: PatternJob) -> bool:
        pattern = self.atlas.by_id[job.pattern_id]
        if job.pattern_id in self.claimed_pattern_ids:
            self.promote((job.pattern_id,))
            return False
        if len(self.claimed_pattern_ids) >= self.max_tasks:
            self.skipped.append({**job.to_dict(), "skip_reason": "pattern_task_cap"})
            return False
        partition = self._partition_key(pattern)
        if partition in self.claimed_partitions:
            self.skipped.append(
                {**job.to_dict(), "skip_reason": "canonical_partition_duplicate"}
            )
            return False
        self.claimed_pattern_ids.add(job.pattern_id)
        self.claimed_partitions.add(partition)
        self.pending.append(job)
        return True

    def promote(self, pattern_ids: Sequence[str]) -> None:
        priorities = {str(value) for value in pattern_ids}
        if not priorities or not self.pending:
            return
        promoted = [job for job in self.pending if job.pattern_id in priorities]
        retained = [job for job in self.pending if job.pattern_id not in priorities]
        self.pending = deque((*promoted, *retained))

    def pop(self) -> PatternJob | None:
        return None if not self.pending else self.pending.popleft()


def _scorer_config(config: InterruptSearchConfig) -> ScorerConfig:
    return ScorerConfig(
        rank24_entrance_exists_weight=float(config.rank24_entrance_exists_weight),
        terminal_scoring_mode=str(config.terminal_scoring_mode),
        dynamic_new_class_score=float(config.dynamic_new_class_score),
        dynamic_known_class_score=float(config.dynamic_known_class_score),
        dynamic_frequent_class_score=float(config.dynamic_frequent_class_score),
        dynamic_frequent_class_threshold=int(config.dynamic_frequent_class_threshold),
    )


def _scorer_for_blocks(
    config: InterruptSearchConfig,
    global_discovery: InterruptGlobalDiscoveryState,
    blocks: Sequence[Sequence[int]],
) -> ExpansionScorer:
    return ExpansionScorer(
        blocks=blocks,
        rare_target_classes=global_discovery.active_rare_target_classes(),
        target_classes=set(int(value) for value in config.target_classes),
        terminal_validation_cache=global_discovery.terminal_validation_cache,
        structure_cache=global_discovery.structure_cache,
        config=_scorer_config(config),
    )


def _initial_support_word(config: InterruptSearchConfig) -> int:
    examples = parse_example_rows(config.examples_path)
    rows = examples.get(int(config.initial_class_id))
    if rows is None:
        raise KeyError(f"unknown initial class {config.initial_class_id}")
    row = rows.get(int(config.rep_index))
    if row is None:
        raise KeyError(
            f"class {config.initial_class_id} has no rep_index {config.rep_index}"
        )
    return support_word_from_mask(support_mask_from_row(row))


def _blocks_from_observed_support(
    support_word: int,
    *,
    class_id: int,
) -> tuple[tuple[int, ...], ...]:
    support = tuple(
        1 if int(support_word) & (1 << index) else 0
        for index in range(64)
    )
    patterns = build_orbit_patterns_from_support(
        support,
        class_id=int(class_id),
        max_patterns=1,
    )
    if not patterns:
        raise RuntimeError("observed support produced no stabilizer pattern")
    return patterns[0].orbits


def _support_word_for_discovery(frame: SearchFrame, chosen_blocks: Sequence[int]) -> int:
    key_values = [0] * len(frame.task.state.scorer.blocks)
    for block_index in chosen_blocks:
        key_values[int(block_index)] = 1
    key: BlockKey = tuple(key_values)
    scorer = frame.task.state.scorer
    selected_word = int(scorer.support_key(key))
    terminal = scorer.terminal_label(key)
    validation = terminal.validation
    if validation.normal is None or validation.offset is None:
        return selected_word
    signed = scorer.points @ validation.normal + float(validation.offset)
    tight = np.flatnonzero(np.abs(signed) <= float(scorer.config.support_tol))
    return support_word_from_indices(tight.tolist())


def _pattern_task_config(
    config: SubgroupInterruptSearchConfig,
    pattern: SubgroupPattern,
) -> InterruptSearchConfig:
    if pattern.level == 1:
        iterations = config.root_iterations
    else:
        iterations = config.level2_iterations
    budget = int(config.interrupt.iterations if iterations is None else iterations)
    return replace(
        config.interrupt,
        iterations=budget,
        adaptive_max_multiplier=1.0,
        adaptive_extra_iterations=1,
        compatibility_frontier_extension_iterations=0,
    )


def _growth_priority(
    item: tuple[SubgroupPattern, SubgroupGrowthEdge],
    *,
    relation_counts: Counter[str],
    scheduled_fine_types: set[str],
) -> tuple[object, ...]:
    pattern, edge = item
    scale_order = {"tiny": 0, "small": 1, "medium": 2}
    return (
        int(pattern.fine_type in scheduled_fine_types),
        relation_counts[edge.relation],
        scale_order.get(pattern.reduction_scale, 3),
        -len(pattern.blocks),
        -int(pattern.source_count),
        pattern.pattern_id,
    )


def _config_dict(config: SubgroupInterruptSearchConfig) -> dict[str, object]:
    base = config.interrupt
    return {
        "initial_class_id": base.initial_class_id,
        "class_iterations": base.iterations,
        "target_classes": list(base.target_classes),
        "rare_target_classes": list(base.rare_target_classes),
        "examples_path": str(base.examples_path),
        "seed": base.seed,
        "bootstrap_minimal_roots": config.bootstrap_minimal_roots,
        "bootstrap_root_limit": config.bootstrap_root_limit,
        "root_iterations": (
            base.iterations if config.root_iterations is None else config.root_iterations
        ),
        "level2_iterations": (
            base.iterations if config.level2_iterations is None else config.level2_iterations
        ),
        "class_step_burst": config.class_step_burst,
        "max_pattern_tasks": config.max_pattern_tasks,
        "growth_children_per_discovery": config.growth_children_per_discovery,
        "observed_level2_per_discovery": config.observed_level2_per_discovery,
        "include_level2_support_analysis": config.include_level2_support_analysis,
        "max_level2_pair_checks": config.max_level2_pair_checks,
        "max_level2_matches": config.max_level2_matches,
        "subgroup_warmup_iterations": config.subgroup_warmup_iterations,
        "identity_tree_count": config.identity_tree_count,
        "rank23_tail_enabled": base.rank23_tail_enabled,
        "rank23_tail_max_prefixes": base.rank23_tail_max_prefixes,
        "rank23_tail_candidates_per_step": base.rank23_tail_candidates_per_step,
        "rank23_tail_active_service": base.rank23_tail_active_service,
    }


def run_subgroup_interrupt_search(
    config: SubgroupInterruptSearchConfig,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    atlas = load_subgroup_pattern_atlas(config.atlas_path)
    base = config.interrupt
    global_discovery = InterruptGlobalDiscoveryState.from_config(base)
    global_discovery.mark_discovered(int(base.initial_class_id))

    frontier = PatternFrontier(atlas, config.max_pattern_tasks)
    active_pattern_frames: deque[SearchFrame] = deque()
    racing_pattern_frame: SearchFrame | None = None
    continuation_frame: SearchFrame | None = None
    identity_frames: deque[SearchFrame] = deque()
    class_stack: list[SearchFrame] = []
    started_class_ids: set[int] = set()
    finished_class_ids: set[int] = set()
    completed_runs: list[dict[str, object]] = []
    discovery_events: list[UnifiedDiscoveryEvent] = []
    initial_analyses: list[dict[str, object]] = []
    relation_counts: Counter[str] = Counter()
    scheduled_fine_types: set[str] = set()
    frame_serial = 0
    scheduler_iterations = 0
    scheduler_step_seconds = 0.0
    scheduler_max_step_seconds = 0.0
    scheduler_max_step_frame_id: int | None = None
    scheduler_max_step_local_iteration = 0
    scheduler_steps_over_one_second = 0
    scheduler_steps_over_five_seconds = 0
    class_steps_since_pattern = 0
    prefer_identity = True
    identity_finished_count = 0

    def next_frame_id() -> int:
        nonlocal frame_serial
        frame_serial += 1
        return frame_serial

    def build_class_frame(
        class_id: int,
        support_word: int,
        *,
        source_class_id: int,
        reason: str,
        parent_frame_id: int | None,
        kind: str = "class",
        ensemble_index: int | None = None,
        blocks: Sequence[Sequence[int]] | None = None,
    ) -> SearchFrame:
        resolved_blocks = (
            _blocks_from_observed_support(support_word, class_id=class_id)
            if blocks is None
            else tuple(tuple(int(vertex) for vertex in block) for block in blocks)
        )
        scorer = _scorer_for_blocks(base, global_discovery, resolved_blocks)
        frame_id = next_frame_id()
        task = build_interrupt_task_from_scorer(
            base,
            global_discovery,
            scorer,
            class_id=class_id,
            search_index=frame_id,
            parent_search_index=parent_frame_id,
        )
        return SearchFrame(
            frame_id=frame_id,
            kind=str(kind),
            task=task,
            anchor_class_id=int(class_id),
            source_class_id=int(source_class_id),
            support_word=int(support_word),
            reason=reason,
            ensemble_index=ensemble_index,
        )

    def build_pattern_frame(job: PatternJob) -> SearchFrame:
        pattern = atlas.by_id[job.pattern_id]
        task_config = _pattern_task_config(config, pattern)
        scorer = _scorer_for_blocks(task_config, global_discovery, pattern.blocks)
        frame_id = next_frame_id()
        task = build_interrupt_task_from_scorer(
            task_config,
            global_discovery,
            scorer,
            class_id=job.anchor_class_id,
            search_index=frame_id,
            parent_search_index=None,
        )
        return SearchFrame(
            frame_id=frame_id,
            kind="subgroup",
            task=task,
            anchor_class_id=job.anchor_class_id,
            source_class_id=job.source_class_id,
            support_word=None,
            pattern_id=job.pattern_id,
            reason=job.reason,
            parent_pattern_id=job.parent_pattern_id,
            growth_relation=job.growth_relation,
            racing_warmup_limit=min(
                int(task.iteration_limit),
                int(config.subgroup_warmup_iterations),
            ),
        )

    def enqueue_pattern(
        pattern_id: str,
        *,
        anchor_class_id: int,
        source_class_id: int,
        reason: str,
        parent_pattern_id: str | None = None,
        growth_relation: str | None = None,
    ) -> bool:
        added = frontier.enqueue(
            PatternJob(
                pattern_id=pattern_id,
                anchor_class_id=int(anchor_class_id),
                source_class_id=int(source_class_id),
                reason=reason,
                parent_pattern_id=parent_pattern_id,
                growth_relation=growth_relation,
            )
        )
        if added:
            pattern = atlas.by_id[pattern_id]
            if pattern.fine_type:
                scheduled_fine_types.add(pattern.fine_type)
            if growth_relation:
                relation_counts[growth_relation] += 1
        return added

    def enqueue_growth(
        root_id: str,
        *,
        anchor_class_id: int,
        source_class_id: int,
    ) -> None:
        candidates = sorted(
            atlas.children(root_id),
            key=lambda item: _growth_priority(
                item,
                relation_counts=relation_counts,
                scheduled_fine_types=scheduled_fine_types,
            ),
        )
        added = 0
        for child, edge in candidates:
            if enqueue_pattern(
                child.pattern_id,
                anchor_class_id=anchor_class_id,
                source_class_id=source_class_id,
                reason="productive_root_growth",
                parent_pattern_id=root_id,
                growth_relation=edge.relation,
            ):
                added += 1
            if added >= int(config.growth_children_per_discovery):
                break

    def promote_active(pattern_ids: Sequence[str]) -> None:
        priorities = {str(value) for value in pattern_ids}
        if not priorities or not active_pattern_frames:
            return
        promoted = [
            frame for frame in active_pattern_frames
            if frame.pattern_id in priorities
        ]
        retained = [
            frame for frame in active_pattern_frames
            if frame.pattern_id not in priorities
        ]
        active_pattern_frames.clear()
        active_pattern_frames.extend((*promoted, *retained))

    initial_word = _initial_support_word(base)
    initial_blocks = _blocks_from_observed_support(
        initial_word,
        class_id=int(base.initial_class_id),
    )
    for ensemble_index in range(1, int(config.identity_tree_count) + 1):
        identity_frames.append(
            build_class_frame(
                int(base.initial_class_id),
                initial_word,
                source_class_id=int(base.initial_class_id),
                reason="identity_ensemble",
                parent_frame_id=None,
                kind="identity",
                ensemble_index=ensemble_index,
                blocks=initial_blocks,
            )
        )
    started_class_ids.add(int(base.initial_class_id))

    initial_analysis = atlas.analyze_support(
        initial_word,
        include_level2=config.include_level2_support_analysis,
        max_level2_pair_checks=config.max_level2_pair_checks,
        max_level2_matches=config.max_level2_matches,
    )
    initial_analyses.append(
        {
            "class_id": int(base.initial_class_id),
            "source": "initial_seed",
            **initial_analysis.to_dict(),
        }
    )

    if config.bootstrap_minimal_roots:
        for pattern in atlas.roots[: int(config.bootstrap_root_limit)]:
            enqueue_pattern(
                pattern.pattern_id,
                anchor_class_id=int(base.initial_class_id),
                source_class_id=int(base.initial_class_id),
                reason="identity_minimal_overgroup",
            )
    for root_id in initial_analysis.root_pattern_ids:
        enqueue_pattern(
            root_id,
            anchor_class_id=int(base.initial_class_id),
            source_class_id=int(base.initial_class_id),
            reason="initial_support_stabilizer_root",
        )
    for pattern_id in initial_analysis.level2_pattern_ids[
        : int(config.observed_level2_per_discovery)
    ]:
        enqueue_pattern(
            pattern_id,
            anchor_class_id=int(base.initial_class_id),
            source_class_id=int(base.initial_class_id),
            reason="initial_support_level2_subgroup",
        )

    def snapshot(stop_reason: str) -> dict[str, object]:
        discovered = sorted(global_discovery.discovered_classes_snapshot())
        targets = {int(value) for value in base.target_classes}
        referenced_pattern_ids = set(frontier.claimed_pattern_ids)
        for analysis_record in initial_analyses:
            referenced_pattern_ids.update(analysis_record.get("root_pattern_ids", []))
            referenced_pattern_ids.update(analysis_record.get("level2_pattern_ids", []))
        for event in discovery_events:
            referenced_pattern_ids.update(event.pattern_analysis.root_pattern_ids)
            referenced_pattern_ids.update(event.pattern_analysis.level2_pattern_ids)
        return {
            "meta": {
                "script": "src/mcts/subgroup_interrupt_search.py",
                "algorithm": (
                    "subgroup_racing_identity_ensemble_atomic_rank23_tail"
                    if base.rank23_tail_active_service
                    else "subgroup_racing_identity_ensemble_rank23_tail"
                ),
                "status": stop_reason,
                "atlas_path": str(atlas.source_path),
                "atlas_source": dict(atlas.source_metadata),
                "atlas_counts": {
                    "identity": 1,
                    "minimal_roots": len(atlas.roots),
                    "pair_roots": len(atlas.pair_roots),
                    "level2_nodes": len(atlas.level2),
                    "growth_edges": len(atlas.edges),
                },
                "prior_policy": (
                    "group_atlas_only; class patterns are inferred from observed terminal supports"
                ),
                "config": _config_dict(config),
                "elapsed_seconds": time.perf_counter() - started_at,
            },
            "summary": {
                "coverage_class_ids": discovered,
                "coverage_count": len(set(discovered) & targets),
                "missing_class_ids": sorted(targets - set(discovered)),
                "started_class_ids": sorted(started_class_ids),
                "finished_class_ids": sorted(finished_class_ids),
                "pattern_tasks_claimed": len(frontier.claimed_pattern_ids),
                "pattern_tasks_pending": len(frontier.pending),
                "pattern_frames_active": len(active_pattern_frames),
                "pattern_racing_active": racing_pattern_frame is not None,
                "rank23_continuation_active": continuation_frame is not None,
                "identity_frames_active": len(identity_frames),
                "identity_trees_finished": identity_finished_count,
                "rank23_tail_cache_entries": len(
                    global_discovery.rank23_tail_cache
                ),
                "completed_run_count": len(completed_runs),
                "scheduler_iterations": scheduler_iterations,
                "scheduler_step_seconds": scheduler_step_seconds,
                "scheduler_max_step_seconds": scheduler_max_step_seconds,
                "scheduler_max_step_frame_id": scheduler_max_step_frame_id,
                "scheduler_max_step_local_iteration": (
                    scheduler_max_step_local_iteration
                ),
                "scheduler_steps_over_one_second": (
                    scheduler_steps_over_one_second
                ),
                "scheduler_steps_over_five_seconds": (
                    scheduler_steps_over_five_seconds
                ),
            },
            "initial_pattern_analyses": initial_analyses,
            "pattern_catalog": {
                pattern_id: atlas.by_id[pattern_id].to_dict()
                for pattern_id in sorted(referenced_pattern_ids)
            },
            "discovery_timeline": [event.to_dict() for event in discovery_events],
            "runs": completed_runs,
            "pending_pattern_jobs": [job.to_dict() for job in frontier.pending],
            "pattern_jobs_skipped": list(frontier.skipped),
        }

    stop_reason = "frontiers_empty"
    targets = {int(value) for value in base.target_classes}
    while (
        continuation_frame is not None
        or class_stack
        or identity_frames
        or racing_pattern_frame is not None
        or active_pattern_frames
        or frontier.pending
    ):
        if targets.issubset(global_discovery.discovered_classes_snapshot()):
            stop_reason = "target_coverage_complete"
            break

        if continuation_frame is not None:
            frame = continuation_frame
            continuation_frame = None
        else:
            has_class_work = bool(class_stack or identity_frames)
            has_pattern_work = bool(
                racing_pattern_frame is not None
                or active_pattern_frames
                or frontier.pending
            )
            should_run_pattern = has_pattern_work and (
                racing_pattern_frame is not None
                or not has_class_work
                or class_steps_since_pattern >= int(config.class_step_burst)
            )
            if should_run_pattern:
                if racing_pattern_frame is not None:
                    frame = racing_pattern_frame
                elif frontier.pending:
                    job = frontier.pop()
                    if job is None:
                        continue
                    frame = build_pattern_frame(job)
                    racing_pattern_frame = frame
                else:
                    frame = active_pattern_frames.popleft()
                class_steps_since_pattern = 0
            else:
                if identity_frames and class_stack:
                    if prefer_identity:
                        frame = identity_frames.popleft()
                    else:
                        frame = class_stack[-1]
                    prefer_identity = not prefer_identity
                elif identity_frames:
                    frame = identity_frames.popleft()
                else:
                    frame = class_stack[-1]
                class_steps_since_pattern += 1

        scheduler_iterations += 1
        step_started_at = time.perf_counter()
        discoveries = frame.task.step()
        step_seconds = time.perf_counter() - step_started_at
        frame.step_calls += 1
        frame.step_seconds += step_seconds
        if step_seconds > frame.max_step_seconds:
            frame.max_step_seconds = step_seconds
            frame.max_step_iteration = int(frame.task.state.iterations_completed)
        if step_seconds >= 1.0:
            frame.steps_over_one_second += 1
            scheduler_steps_over_one_second += 1
        if step_seconds >= 5.0:
            frame.steps_over_five_seconds += 1
            scheduler_steps_over_five_seconds += 1
        scheduler_step_seconds += step_seconds
        if step_seconds > scheduler_max_step_seconds:
            scheduler_max_step_seconds = step_seconds
            scheduler_max_step_frame_id = frame.frame_id
            scheduler_max_step_local_iteration = int(
                frame.task.state.iterations_completed
            )
        if getattr(frame.task.state, "rank23_pending_iteration", None) is not None:
            frame.suspended_step_calls += 1
            if discoveries:
                raise RuntimeError(
                    "a suspended rank-23 iteration exposed discoveries early"
                )
            continuation_frame = frame
            if output_path is not None and scheduler_iterations % 250 == 0:
                _write_json_snapshot(output_path, snapshot("running"))
            continue
        for discovery in discoveries:
            class_id = int(discovery.class_id)
            support_word = _support_word_for_discovery(
                frame,
                discovery.chosen_blocks,
            )
            analysis = atlas.analyze_support(
                support_word,
                include_level2=config.include_level2_support_analysis,
                max_level2_pair_checks=config.max_level2_pair_checks,
                max_level2_matches=config.max_level2_matches,
            )
            frame.discovery_class_ids.append(class_id)
            event = UnifiedDiscoveryEvent(
                event_index=len(discovery_events) + 1,
                class_id=class_id,
                wall_seconds=time.perf_counter() - started_at,
                scheduler_iteration=scheduler_iterations,
                local_iteration=int(discovery.iteration),
                source_frame_id=frame.frame_id,
                source_kind=frame.kind,
                source_class_id=frame.anchor_class_id,
                source_pattern_id=frame.pattern_id,
                support_word=support_word,
                chosen_blocks=tuple(int(value) for value in discovery.chosen_blocks),
                score=float(discovery.score),
                rank=int(discovery.rank),
                pattern_analysis=analysis,
            )
            discovery_events.append(event)

            if class_id not in started_class_ids:
                started_class_ids.add(class_id)
                class_stack.append(
                    build_class_frame(
                        class_id,
                        support_word,
                        source_class_id=frame.anchor_class_id,
                        reason="observed_terminal_interrupt",
                        parent_frame_id=frame.frame_id,
                    )
                )

            for root_id in analysis.root_pattern_ids:
                enqueue_pattern(
                    root_id,
                    anchor_class_id=class_id,
                    source_class_id=class_id,
                    reason="observed_support_stabilizer_root",
                )
            frontier.promote(analysis.root_pattern_ids)
            promote_active(analysis.root_pattern_ids)

            for pattern_id in analysis.level2_pattern_ids[
                : int(config.observed_level2_per_discovery)
            ]:
                enqueue_pattern(
                    pattern_id,
                    anchor_class_id=class_id,
                    source_class_id=class_id,
                    reason="observed_support_level2_subgroup",
                )

            if frame.kind == "subgroup" and frame.pattern_id is not None:
                source_pattern = atlas.by_id[frame.pattern_id]
                if source_pattern.level == 1:
                    enqueue_growth(
                        source_pattern.pattern_id,
                        anchor_class_id=class_id,
                        source_class_id=class_id,
                    )

            if output_path is not None:
                _write_json_snapshot(output_path, snapshot("running"))

        if frame.kind == "subgroup" and not frame.racing_promoted:
            warmup_limit = int(frame.racing_warmup_limit or 0)
            if frame.task.state.iterations_completed >= warmup_limit:
                if frame.task.state.productive:
                    frame.racing_promoted = True
                elif frame.task.iteration_limit > warmup_limit:
                    frame.task.state.stop_reason = "subgroup_racing_pruned"
                if racing_pattern_frame is frame:
                    racing_pattern_frame = None

        if frame.task.finished:
            if frame.kind == "class":
                if not class_stack:
                    raise RuntimeError("class interrupt stack is unexpectedly empty")
                if class_stack[-1] is not frame:
                    # A discovery pushed a child in this same step.  Keep the
                    # finished parent frame below it until normal LIFO resume.
                    continue
                class_stack.pop()
                finished_class_ids.add(frame.anchor_class_id)
            elif frame.kind == "identity":
                identity_finished_count += 1
                if identity_finished_count >= int(config.identity_tree_count):
                    finished_class_ids.add(frame.anchor_class_id)
            elif racing_pattern_frame is frame:
                racing_pattern_frame = None
            completed_runs.append(frame.to_run_dict())
            release_tree = getattr(frame.task.state, "release_tree", None)
            if callable(release_tree):
                release_tree()
        elif frame.kind == "subgroup":
            if frame.racing_promoted:
                active_pattern_frames.append(frame)
            elif racing_pattern_frame is None:
                racing_pattern_frame = frame
        elif frame.kind == "identity":
            identity_frames.append(frame)

        if output_path is not None and scheduler_iterations % 250 == 0:
            _write_json_snapshot(output_path, snapshot("running"))

    remaining_frames = [*class_stack, *identity_frames, *active_pattern_frames]
    if racing_pattern_frame is not None:
        remaining_frames.append(racing_pattern_frame)
    for frame in remaining_frames:
        release_tree = getattr(frame.task.state, "release_tree", None)
        if callable(release_tree):
            release_tree()

    payload = snapshot(stop_reason)
    if output_path is not None:
        _write_json_snapshot(output_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified Bell 322 subgroup-pattern and interrupt MCTS."
    )
    parser.add_argument("--initial-class-id", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--root-iterations", type=int)
    parser.add_argument("--level2-iterations", type=int)
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--rare-target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--class-step-burst", type=int, default=8)
    parser.add_argument("--subgroup-warmup-iterations", type=int, default=50)
    parser.add_argument("--identity-tree-count", type=int, default=4)
    parser.add_argument("--max-pattern-tasks", type=int, default=80)
    parser.add_argument("--bootstrap-root-limit", type=int, default=24)
    parser.add_argument("--no-bootstrap-roots", action="store_true")
    parser.add_argument("--growth-children-per-discovery", type=int, default=2)
    parser.add_argument("--observed-level2-per-discovery", type=int, default=2)
    parser.add_argument("--max-level2-pair-checks", type=int, default=1024)
    parser.add_argument("--no-level2-support-analysis", action="store_true")
    parser.add_argument("--no-rank23-tail", action="store_true")
    parser.add_argument("--rank23-tail-max-prefixes", type=int, default=24)
    parser.add_argument(
        "--rank23-tail-candidates-per-step",
        type=int,
        default=DEFAULT_ATOMIC_TAIL_CANDIDATES,
    )
    parser.add_argument("--no-rank23-tail-active-service", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    interrupt = InterruptSearchConfig(
        initial_class_id=int(args.initial_class_id),
        iterations=int(args.iterations),
        target_classes=tuple(int(value) for value in args.target_classes),
        rare_target_classes=tuple(int(value) for value in args.rare_target_classes),
        examples_path=args.examples,
        seed=int(args.seed),
        rank23_tail_enabled=not bool(args.no_rank23_tail),
        rank23_tail_max_prefixes=int(args.rank23_tail_max_prefixes),
        rank23_tail_candidates_per_step=max(
            0,
            int(args.rank23_tail_candidates_per_step),
        ),
        rank23_tail_active_service=(
            not bool(args.no_rank23_tail)
            and not bool(args.no_rank23_tail_active_service)
        ),
    )
    config = SubgroupInterruptSearchConfig(
        interrupt=interrupt,
        atlas_path=args.atlas,
        bootstrap_minimal_roots=not bool(args.no_bootstrap_roots),
        bootstrap_root_limit=int(args.bootstrap_root_limit),
        root_iterations=args.root_iterations,
        level2_iterations=args.level2_iterations,
        class_step_burst=int(args.class_step_burst),
        subgroup_warmup_iterations=int(args.subgroup_warmup_iterations),
        identity_tree_count=int(args.identity_tree_count),
        max_pattern_tasks=int(args.max_pattern_tasks),
        growth_children_per_discovery=int(args.growth_children_per_discovery),
        observed_level2_per_discovery=int(args.observed_level2_per_discovery),
        include_level2_support_analysis=not bool(args.no_level2_support_analysis),
        max_level2_pair_checks=int(args.max_level2_pair_checks),
    )
    output = args.output or Path(
        "src/mcts/runs/"
        f"subgroup_interrupt_class{interrupt.initial_class_id}_i{interrupt.iterations}.json"
    )
    payload = run_subgroup_interrupt_search(config, output_path=output)
    summary = payload["summary"]
    print(f"output={output}")
    print(
        "coverage="
        f"{summary['coverage_count']}/{len(interrupt.target_classes)} "
        f"classes={summary['coverage_class_ids']}"
    )
    print(f"missing={summary['missing_class_ids']}")
    print(f"elapsed_seconds={payload['meta']['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
