from __future__ import annotations

"""Paper-inspired block Filler-Corrector search.

The Filler is the existing add-only interruptible MCTS. The Corrector maps the
paper's retained-row action to orbit blocks: it protects a leading high-rank
core, removes one or more non-core blocks from an exact terminal state, and
lets an independently seeded MCTS window repack the corrected prefix.

Paper: https://arxiv.org/abs/2511.13391

This module deliberately uses a transparent heuristic Corrector rather than
the paper's trained PPO policy. There is no Corrector training corpus in this
repository, while the structural scorer, compatibility bank, and global class
reward already provide useful local policy signals.
"""

import argparse
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
    _productive_iteration_limit,
    _run_one_iteration,
    _should_stop_as_terminal_sink,
    _should_stop_for_exact_frequency,
    _should_stop_productive_for_no_novelty,
    run_interruptible_search,
)
from mcts.search import (
    ExactClassDiscovery,
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
    max_remove_blocks: int = 1
    candidate_pool: int = 12
    start_iteration: int = 200
    stagnation_iterations: int = 80
    window_iterations: int = 48
    cooldown_iterations: int = 80
    max_corrections_per_task: int = 3
    max_corrections_per_sink_task: int = 1
    new_class_reward: float = 100.0

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
    policy_score: float


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


class BlockCorrector:
    """Heuristic retained-block policy with lightweight online feedback."""

    def __init__(self, config: CorrectorConfig) -> None:
        self.config = config
        self.tried_corrected_keys: set[BlockKey] = set()
        self._removed_reward_sum: dict[int, float] = {}
        self._removed_reward_count: dict[int, int] = {}
        self._source_cursor = 0

    def observe(self, removed_blocks: Sequence[int], reward: float) -> None:
        for block in removed_blocks:
            index = int(block)
            self._removed_reward_sum[index] = self._removed_reward_sum.get(index, 0.0) + float(reward)
            self._removed_reward_count[index] = self._removed_reward_count.get(index, 0) + 1

    def _learned_block_value(self, removed_blocks: Sequence[int]) -> float:
        values = []
        for block in removed_blocks:
            count = self._removed_reward_count.get(int(block), 0)
            if count > 0:
                values.append(self._removed_reward_sum[int(block)] / float(count))
        return 0.0 if not values else sum(values) / float(len(values))

    def propose(
        self,
        state: InterruptSearchState,
        *,
        rng: random.Random,
    ) -> CorrectionProposal | None:
        scorer = state.scorer
        remaining_targets = sorted(
            int(class_id)
            for class_id in scorer.target_classes
            if int(class_id) not in state.global_discovery.discovered_exact_classes
        )
        exact_sources = [
            hit
            for label, hit in sorted(state.terminal_bests.items())
            if exact_class_id(label) is not None and int(hit.rank) >= self.config.terminal_rank
        ]
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
            for remove_count in range(1, min(self.config.max_remove_blocks, len(deletable)) + 1):
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
                    compatible: tuple[int, ...] = ()
                    if state.compatibility_bank is not None and remaining_targets:
                        compatible = state.compatibility_bank.active_signature(
                            corrected_key,
                            remaining_targets,
                        )
                    removed_vertices = sum(len(scorer.blocks[index]) for index in removed)
                    learned_value = self._learned_block_value(removed)
                    cheap_score = (
                        8.0 * float(corrected_rank)
                        + 0.75 * float(len(compatible))
                        + math.log1p(source_frequency)
                        - 0.25 * float(removed_vertices)
                        + 0.05 * learned_value
                    )
                    candidates.append(
                        _CorrectionCandidate(
                            source=source,
                            protected_blocks=protected,
                            removed_blocks=tuple(int(value) for value in removed),
                            corrected_key=corrected_key,
                            corrected_rank=corrected_rank,
                            compatible_undiscovered_classes=compatible,
                            cheap_score=float(cheap_score),
                        )
                    )
        if not candidates:
            return None

        by_source: dict[str, list[_CorrectionCandidate]] = {}
        for candidate in candidates:
            by_source.setdefault(candidate.source.label, []).append(candidate)
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
        source_labels = sorted(by_source)
        cursor = self._source_cursor % len(source_labels)
        rotated_labels = source_labels[cursor:] + source_labels[:cursor]
        shortlist: list[_CorrectionCandidate] = []
        round_index = 0
        while len(shortlist) < self.config.candidate_pool:
            added = False
            for label in rotated_labels:
                source_candidates = by_source[label]
                if round_index >= len(source_candidates):
                    continue
                shortlist.append(source_candidates[round_index])
                added = True
                if len(shortlist) >= self.config.candidate_pool:
                    break
            if not added:
                break
            round_index += 1
        self._source_cursor = (cursor + max(1, len(shortlist))) % len(source_labels)
        evaluated: list[tuple[float, _CorrectionCandidate, dict[str, int]]] = []
        for candidate in shortlist:
            entrance = {"rare": 0, "class44": 0, "other_valid": 0, "invalid": 0}
            if candidate.corrected_rank == self.config.max_corrected_rank:
                entrance = scorer.rank24_entrance_metrics(candidate.corrected_key)
            valid_count = int(entrance["rare"] + entrance["class44"] + entrance["other_valid"])
            policy_score = (
                candidate.cheap_score
                + 100.0 * float(entrance["rare"])
                + 3.0 * float(valid_count - entrance["rare"])
                - 0.5 * float(entrance["invalid"])
            )
            evaluated.append((float(policy_score), candidate, entrance))

        best_score = max(item[0] for item in evaluated)
        tied = [item for item in evaluated if math.isclose(item[0], best_score, abs_tol=1e-12)]
        policy_score, chosen, entrance = rng.choice(tied)
        self.tried_corrected_keys.add(chosen.corrected_key)
        return CorrectionProposal(
            source_label=chosen.source.label,
            source_key=chosen.source.key,
            source_rank=int(chosen.source.rank),
            protected_blocks=chosen.protected_blocks,
            removed_blocks=chosen.removed_blocks,
            corrected_key=chosen.corrected_key,
            corrected_rank=chosen.corrected_rank,
            compatible_undiscovered_classes=chosen.compatible_undiscovered_classes,
            entrance_rare_count=int(entrance["rare"]),
            entrance_valid_count=int(
                entrance["rare"] + entrance["class44"] + entrance["other_valid"]
            ),
            entrance_invalid_count=int(entrance["invalid"]),
            policy_score=float(policy_score),
        )


class FillerCorrectorTask(InterruptSearchTask):
    """Interrupt task that alternates the base tree with corrected side trees."""

    def __init__(
        self,
        base_task: InterruptSearchTask,
        *,
        corrector_config: CorrectorConfig,
        event_sink: list[CorrectorEvent],
    ) -> None:
        for name, value in vars(base_task).items():
            setattr(self, name, value)
        self.corrector_config = corrector_config
        self.corrector = BlockCorrector(corrector_config)
        self.event_sink = event_sink
        self._active_event: CorrectorEvent | None = None
        self._active_proposal: CorrectionProposal | None = None
        self._active_remaining = 0
        self._saved_root = None
        self._saved_rng: random.Random | None = None
        self._phase_start_exact_serial = 0
        self._phase_seen_exact_serial = 0
        self._phase_best_hit: TerminalHit | None = None
        self._corrections_started = 0
        self._next_corrector_iteration = 0

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
        if not any(exact_class_id(label) is not None for label in self.state.terminal_bests):
            return False
        if not self.state.productive:
            return _should_stop_for_exact_frequency(self) or _should_stop_as_terminal_sink(self)
        if iteration < int(self.corrector_config.start_iteration):
            return False
        novelty_anchor = int(self.last_new_class_iteration or 0)
        return iteration - novelty_anchor >= int(self.corrector_config.stagnation_iterations)

    def _start_correction(self) -> bool:
        if not self._eligible_for_correction():
            return False
        proposal_seed = (
            int(self.state.config.seed)
            + 104729 * (self._corrections_started + 1)
            + 1009 * int(self.search_index)
        )
        proposal = self.corrector.propose(
            self.state,
            rng=random.Random(proposal_seed),
        )
        if proposal is None:
            return False
        remaining_budget = int(self.iteration_limit) - int(self.state.iterations_completed)
        planned_iterations = min(int(self.corrector_config.window_iterations), remaining_budget)
        if planned_iterations <= 0:
            return False

        self._corrections_started += 1
        filler_seed = proposal_seed + 1_000_003
        self._saved_root = self.state.root
        self._saved_rng = getattr(self.state, "_rng", None)
        self.state.root = _get_or_create_node(
            self.state.nodes,
            self.state.scorer,
            proposal.corrected_key,
            path=selected_blocks(proposal.corrected_key),
        )
        self.state._rng = random.Random(filler_seed)  # type: ignore[attr-defined]
        self._active_remaining = planned_iterations
        self._phase_start_exact_serial = int(self.state.exact_hit_serial)
        self._phase_seen_exact_serial = int(self.state.exact_hit_serial)
        self._phase_best_hit = None
        event = CorrectorEvent(
            event_index=len(self.event_sink) + 1,
            search_index=int(self.search_index),
            start_class_id=int(self.class_id),
            correction_index=self._corrections_started,
            trigger_iteration=int(self.state.iterations_completed),
            filler_start_iteration=int(self.state.iterations_completed) + 1,
            source_label=proposal.source_label,
            source_rank=proposal.source_rank,
            source_blocks=selected_blocks(proposal.source_key),
            source_vertex_count=_vertex_count(self.state.scorer, proposal.source_key),
            protected_blocks=list(proposal.protected_blocks),
            removed_blocks=list(proposal.removed_blocks),
            corrected_blocks=selected_blocks(proposal.corrected_key),
            corrected_rank=proposal.corrected_rank,
            corrected_vertex_count=_vertex_count(self.state.scorer, proposal.corrected_key),
            compatible_undiscovered_classes=list(proposal.compatible_undiscovered_classes),
            entrance_rare_count=proposal.entrance_rare_count,
            entrance_valid_count=proposal.entrance_valid_count,
            entrance_invalid_count=proposal.entrance_invalid_count,
            policy_score=proposal.policy_score,
            filler_seed=filler_seed,
            planned_filler_iterations=planned_iterations,
        )
        self._active_event = event
        self._active_proposal = proposal
        self.event_sink.append(event)
        self.state.stop_reason = None
        return True

    def _observe_corrected_iteration(
        self,
        discoveries: Sequence[ExactClassDiscovery],
    ) -> None:
        event = self._active_event
        if event is None:
            return
        for discovery in discoveries:
            class_id = int(discovery.class_id)
            if class_id not in event.new_global_class_ids:
                event.new_global_class_ids.append(class_id)
        if int(self.state.exact_hit_serial) == self._phase_seen_exact_serial:
            return
        self._phase_seen_exact_serial = int(self.state.exact_hit_serial)
        hit = self.state.last_exact_hit
        if hit is None:
            return
        if self._phase_best_hit is None:
            self._phase_best_hit = hit
            return
        current_size = _vertex_count(self.state.scorer, hit.key)
        best_size = _vertex_count(self.state.scorer, self._phase_best_hit.key)
        if (current_size, hit.score) > (best_size, self._phase_best_hit.score):
            self._phase_best_hit = hit

    def _finish_correction(self) -> None:
        event = self._active_event
        proposal = self._active_proposal
        if event is None or proposal is None:
            return
        event.finished_iteration = int(self.state.iterations_completed)
        event.completed_filler_iterations = (
            event.finished_iteration - event.filler_start_iteration + 1
        )
        event.exact_hit_count = max(
            0,
            int(self.state.exact_hit_serial) - self._phase_start_exact_serial,
        )
        if self._phase_best_hit is None:
            repacked_key = proposal.corrected_key
            event.repacked_rank = proposal.corrected_rank
        else:
            repacked_key = self._phase_best_hit.key
            event.repacked_label = self._phase_best_hit.label
            event.repacked_rank = int(self._phase_best_hit.rank)
        event.repacked_blocks = selected_blocks(repacked_key)
        event.repacked_vertex_count = _vertex_count(self.state.scorer, repacked_key)
        event.size_delta_reward = float(event.repacked_vertex_count - event.source_vertex_count)
        event.coverage_augmented_reward = float(
            event.size_delta_reward
            + self.corrector_config.new_class_reward * len(event.new_global_class_ids)
        )
        self.corrector.observe(
            event.removed_blocks,
            event.coverage_augmented_reward,
        )

        self.state.root = self._saved_root
        if self._saved_rng is None:
            if hasattr(self.state, "_rng"):
                delattr(self.state, "_rng")
        else:
            self.state._rng = self._saved_rng  # type: ignore[attr-defined]
        self._next_corrector_iteration = (
            int(self.state.iterations_completed)
            + max(0, int(self.corrector_config.cooldown_iterations))
        )
        self._active_event = None
        self._active_proposal = None
        self._active_remaining = 0
        self._saved_root = None
        self._saved_rng = None
        self._phase_best_hit = None

    def step(self) -> list[ExactClassDiscovery]:
        if self.finished:
            return []
        if self.state.iterations_completed >= self.iteration_limit:
            if self._active_event is not None:
                self._finish_correction()
            self.state.stop_reason = "iterations_exhausted"
            return []

        iteration_index = self.state.iterations_completed + 1
        previous_local_discoveries = set(self.state.local_discovered_exact_classes)
        discoveries = _run_one_iteration(self.state, iteration_index)
        self.state.iterations_completed = iteration_index
        new_local_discoveries = self.state.local_discovered_exact_classes - previous_local_discoveries
        if new_local_discoveries:
            self.last_new_class_iteration = iteration_index
        if any(int(class_id) != int(self.class_id) for class_id in new_local_discoveries):
            self.state.productive = True
            self.iteration_limit = max(self.iteration_limit, _productive_iteration_limit(self))

        if self._active_event is not None:
            self._observe_corrected_iteration(discoveries)
            self._active_remaining -= 1
            if self._active_remaining <= 0 or self.state.iterations_completed >= self.iteration_limit:
                self._finish_correction()

        if self._active_event is not None:
            return discoveries
        if self._start_correction():
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


def _corrector_summary(events: Sequence[CorrectorEvent]) -> dict[str, object]:
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

    def task_factory(
        class_id: int,
        search_index: int,
        parent_search_index: int | None,
    ) -> FillerCorrectorTask:
        return FillerCorrectorTask(
            base_factory(class_id, search_index, parent_search_index),
            corrector_config=corrector_config,
            event_sink=corrector_events,
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
        meta["algorithm"] = "heuristic_block_filler_corrector"
        meta["paper"] = "https://arxiv.org/abs/2511.13391"
        meta["elapsed_seconds"] = elapsed_seconds
        meta["corrector"] = {
            name: value
            for name, value in vars(corrector_config).items()
        }
    payload["corrector"] = _corrector_summary(corrector_events)
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
    parser.add_argument("--corrector-max-remove-blocks", type=int, default=1)
    parser.add_argument("--corrector-candidate-pool", type=int, default=12)
    parser.add_argument("--corrector-start-iteration", type=int, default=200)
    parser.add_argument("--corrector-stagnation-iterations", type=int, default=80)
    parser.add_argument("--corrector-window-iterations", type=int, default=48)
    parser.add_argument("--corrector-cooldown-iterations", type=int, default=80)
    parser.add_argument("--corrector-max-per-task", type=int, default=3)
    parser.add_argument("--corrector-max-per-sink-task", type=int, default=1)
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
