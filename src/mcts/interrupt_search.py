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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from baseline.orbit_blocks import BlockKey, add_block, empty_key, selected_blocks, unselected_blocks
from baseline.reference_classes import DEFAULT_EXAMPLES_PATH, parse_example_rows, support_mask_from_row
from baseline.scorer import ExpansionScorer, ScorerConfig, exact_class_id
from mcts.queue_search import build_class_scorer, summarize
from mcts.search import (
    ExactClassDiscovery,
    ExpansionScore,
    MCTSConfig,
    MCTSNode,
    MCTSResult,
    TerminalHit,
    _backpropagate,
    _choose_unexpanded_action,
    _estimate_state_value,
    _get_or_create_node,
    _prepare_actions,
    _register_hit,
    _rollout,
    _select_child,
)


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
    seed: int = 20260502
    rank24_entrance_exists_weight: float = 6.0
    terminal_scoring_mode: str = "static"
    dynamic_new_class_score: float = 100.0
    dynamic_known_class_score: float = 10.0
    dynamic_frequent_class_score: float = -5.0
    dynamic_frequent_class_threshold: int = 16
    exact_stop_ratio: float = 2.0 / 5.0


@dataclass(frozen=True)
class InterruptDiscoveryEvent:
    """An exact-class hit that caused a child search to be pushed."""

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
    rare_removed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
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
            "rare_removed": bool(self.rare_removed),
        }


@dataclass(frozen=True)
class InterruptSearchRun:
    """One exact-class-rooted search frame."""

    search_index: int
    parent_search_index: int | None
    start_class_id: int
    seed: int
    result: dict[str, object]
    interrupted_by_class_ids: list[int]
    resumed_after_class_ids: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "search_index": self.search_index,
            "parent_search_index": self.parent_search_index,
            "start_class_id": self.start_class_id,
            "seed": self.seed,
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
    tmp_path.replace(path)


@dataclass
class InterruptSearchState:
    scorer: ExpansionScorer
    config: MCTSConfig
    nodes: dict[BlockKey, MCTSNode] = field(default_factory=dict)
    root: MCTSNode | None = None
    terminal_bests: dict[str, TerminalHit] = field(default_factory=dict)
    exact_discoveries: list[ExactClassDiscovery] = field(default_factory=list)
    discovered_exact_classes: set[int] = field(default_factory=set)
    encountered: Counter[str] = field(default_factory=Counter)
    best: TerminalHit | None = None
    iterations_completed: int = 0
    stop_reason: str | None = None


@dataclass
class InterruptSearchTask:
    """One runnable search frame."""

    class_id: int
    search_index: int
    parent_search_index: int | None
    state: InterruptSearchState
    threshold: float
    started_interrupts: list[int] = field(default_factory=list)
    resumed_interrupts: list[int] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.state.stop_reason is not None

    def step(self) -> list[ExactClassDiscovery]:
        if self.finished:
            return []
        if self.state.iterations_completed >= self.state.config.iterations:
            self.state.stop_reason = "iterations_exhausted"
            return []

        iteration_index = self.state.iterations_completed + 1
        discoveries = _run_one_iteration(self.state, iteration_index)
        self.state.iterations_completed = iteration_index
        if _should_stop_for_exact_frequency(self.state, self.threshold):
            self.state.stop_reason = "exact_frequency_threshold"
        elif self.state.iterations_completed >= self.state.config.iterations:
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


def _build_state(config: InterruptSearchConfig, class_id: int, *, search_index: int) -> tuple[InterruptSearchState, tuple[tuple[int, ...], ...]]:
    scorer, pattern_signature = build_class_scorer(
        _queue_like_config(config),
        class_id,
        rare_target_classes=set(int(value) for value in config.rare_target_classes),
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
        seed=int(config.seed) + 1009 * (search_index - 1),
    )
    state = InterruptSearchState(scorer=scorer, config=mcts_config)
    root_key = empty_key(len(scorer.blocks))
    state.root = _get_or_create_node(state.nodes, scorer, root_key, path=[])
    return state, pattern_signature


def _run_one_iteration(state: InterruptSearchState, iteration_index: int) -> list[ExactClassDiscovery]:
    rng = random.Random(state.config.seed + 1009 * iteration_index)
    # Preserve the same RNG across iterations by mutating the seeded generator.
    if iteration_index == 1:
        state._rng = rng  # type: ignore[attr-defined]
    else:
        rng = getattr(state, "_rng", rng)
    state._rng = rng  # type: ignore[attr-defined]

    root = state.root
    if root is None:
        raise RuntimeError("search state is missing a root node")

    path_nodes: list[MCTSNode] = [root]
    node = root
    new_discoveries: list[ExactClassDiscovery] = []

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
        if class_id is None or class_id in state.discovered_exact_classes:
            return
        state.discovered_exact_classes.add(class_id)
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
        new_discoveries.append(discovery)

    while True:
        if node.is_terminal:
            terminal_score = float(state.scorer.terminal_score(node.key))
            if node.terminal is not None and node.terminal.is_exact:
                record_exact_discovery(
                    node.terminal,
                    score=terminal_score,
                    depth=len(node.path),
                    key=node.key,
                    path=node.path,
                    rank=node.rank,
                )
            state.best = _register_hit(
                node,
                terminal_score,
                terminal_bests=state.terminal_bests,
                encountered=state.encountered,
                scorer=state.scorer,
                best=state.best,
            )
            _backpropagate(path_nodes, terminal_score)
            break

        if node.rank >= 25:
            terminal = state.scorer.terminal_label(node.key)
            terminal_score = float(state.scorer.terminal_score(node.key))
            if terminal.is_exact:
                record_exact_discovery(
                    terminal,
                    score=terminal_score,
                    depth=len(node.path),
                    key=node.key,
                    path=node.path,
                    rank=node.rank,
                )
            state.best = _register_hit(
                node,
                terminal_score,
                terminal=terminal,
                terminal_bests=state.terminal_bests,
                encountered=state.encountered,
                scorer=state.scorer,
                best=state.best,
            )
            _backpropagate(path_nodes, terminal_score)
            break

        if not node.unexpanded_actions:
            _prepare_actions(node, state.scorer, state.config)

        if node.unexpanded_actions:
            action = _choose_unexpanded_action(node, rng)
            child_key = add_block(node.key, action)
            child_path = [*node.path, int(action)]
            child = _get_or_create_node(state.nodes, state.scorer, child_key, path=child_path, parent=node, action=action)
            child.prior = node.action_priors.get(action, child.prior)
            node.children[action] = child
            node.unexpanded_actions = [item for item in node.unexpanded_actions if item != action]
            path_nodes.append(child)

            step = node.action_scores[action]
            if child.is_terminal or child.rank >= 25:
                terminal = child.terminal if child.terminal is not None else state.scorer.terminal_label(child.key)
                terminal_score = float(step.score)
                if terminal.is_exact:
                    record_exact_discovery(
                        terminal,
                        score=terminal_score,
                        depth=len(child.path),
                        key=child.key,
                        path=child.path,
                        rank=child.rank,
                    )
                state.best = _register_hit(
                    child,
                    terminal_score,
                    terminal=terminal,
                    terminal_bests=state.terminal_bests,
                    encountered=state.encountered,
                    scorer=state.scorer,
                    best=state.best,
                )
                _backpropagate(path_nodes, terminal_score)
            else:
                def _record_for_rollout(*args, **kwargs):
                    # _rollout may pass through an `iteration_index` keyword; strip it
                    # because the local record_exact_discovery does not accept it.
                    kwargs.pop("iteration_index", None)
                    return record_exact_discovery(*args, **kwargs)

                rollout_value, state.best = _rollout(
                    child,
                    state.scorer,
                    state.config,
                    rng,
                    state.terminal_bests,
                    state.encountered,
                    state.best,
                    _record_for_rollout,
                    iteration_index,
                )
                total_value = float(step.score) + rollout_value
                _backpropagate(path_nodes, total_value)
            break

        next_node = _select_child(node, state.config.exploration_constant)
        if next_node is None:
            value = _estimate_state_value(node.key, state.scorer, state.config)
            _backpropagate(path_nodes, value)
            break
        node = next_node
        path_nodes.append(node)

    return new_discoveries


def _should_stop_for_exact_frequency(state: InterruptSearchState, threshold: float) -> bool:
    return any(count > threshold for count in state.scorer.discovered_label_counts.values())


def run_interruptible_search(
    config: InterruptSearchConfig,
    *,
    task_factory: Callable[[int, int, int | None], InterruptSearchTask] | None = None,
    output_path: Path | None = None,
) -> InterruptSearchReport:
    current_rare_target_classes = set(int(value) for value in config.rare_target_classes)
    started_class_ids: set[int] = set()
    finished_class_ids: set[int] = set()
    stack: list[InterruptSearchTask] = []
    interrupt_events: list[InterruptDiscoveryEvent] = []
    runs: list[InterruptSearchRun] = []
    # store lightweight result dicts to minimize memory
    exact_class_counts: Counter[int] = Counter()
    search_index = 0
    using_default_task_factory = task_factory is None
    task_factory = task_factory or _default_task_factory(config)
    run_log_path = None if output_path is None else output_path.with_name(f"{output_path.stem}.runs.json")
    first_run_record = True

    if run_log_path is not None:
        _start_json_array(run_log_path)

    enable_pattern_dedup = using_default_task_factory

    # track pattern shapes globally so only the first discovered class per
    # abstract block grouping is searched. The key is the multiset of block
    # sizes, which treats subset variants such as 14/17/21 as one pattern.
    claimed_pattern_shapes: set[tuple[int, ...]] = set()

    # register initial task and its pattern signature
    _scorer, initial_pattern = build_class_scorer(
        _queue_like_config(config),
        int(config.initial_class_id),
        rare_target_classes=current_rare_target_classes,
    )
    initial_pattern = _canonical_pattern_signature(initial_pattern)
    initial_shape = _pattern_shape_key(initial_pattern)
    claimed_pattern_shapes.add(initial_shape)
    initial_task = task_factory(int(config.initial_class_id), 1, None)
    stack.append(initial_task)
    started_class_ids.add(int(config.initial_class_id))

    while stack:
        task = stack[-1]
        discoveries = task.step()

        pushed_child = False
        for discovery in discoveries:
            exact_class_counts[discovery.class_id] += 1
            if discovery.class_id in started_class_ids:
                continue

            # determine pattern signature and whether to skip starting a new search
            _, candidate_pattern_signature = build_class_scorer(
                _queue_like_config(config),
                int(discovery.class_id),
                rare_target_classes=current_rare_target_classes,
            )
            candidate_pattern_signature = _canonical_pattern_signature(candidate_pattern_signature)
            candidate_shape = _pattern_shape_key(candidate_pattern_signature)
            same_pattern = candidate_shape in claimed_pattern_shapes if enable_pattern_dedup else False
            rare_removed = False
            if discovery.class_id in current_rare_target_classes:
                current_rare_target_classes.remove(discovery.class_id)
                rare_removed = True

            if not same_pattern:
                claimed_pattern_shapes.add(candidate_shape)
                search_index += 1
                child_task = task_factory(discovery.class_id, search_index + 1, task.search_index)
                stack.append(child_task)
                started_class_ids.add(discovery.class_id)
                task.started_interrupts.append(discovery.class_id)
                pattern_status = "unique"
                pushed_child = True
            else:
                pattern_status = "same_as_started"

            interrupt_events.append(
                InterruptDiscoveryEvent(
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
                    pattern_signature=[list(block) for block in candidate_pattern_signature],
                    rare_removed=rare_removed,
                )
            )
            break

        if pushed_child:
            continue

        if task.finished:
            finished_class_ids.add(task.class_id)
            task_result = task.result()
            task_result_dict = task_result.to_dict()
            run_record = InterruptSearchRun(
                search_index=task.search_index,
                parent_search_index=task.parent_search_index,
                start_class_id=task.class_id,
                seed=task.state.config.seed,
                result=task_result_dict,
                interrupted_by_class_ids=list(task.started_interrupts),
                resumed_after_class_ids=list(task.resumed_interrupts),
            )
            runs.append(run_record)
            if run_log_path is not None:
                _append_json_array_record(run_log_path, run_record.to_dict(), first_record=first_run_record)
                first_run_record = False
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
        _write_json_snapshot(
            output_path,
            {
                "meta": report.meta,
                "started_class_ids": report.started_class_ids,
                "finished_class_ids": report.finished_class_ids,
                "summary": report.summary,
                "stop_reason": report.stop_reason,
                "exact_class_counts": report.exact_class_counts,
                "run_log_path": None if run_log_path is None else str(run_log_path),
            },
        )
    if run_log_path is not None:
        _close_json_array(run_log_path)
    return report


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
        "seed": int(config.seed),
        "rank24_entrance_exists_weight": float(config.rank24_entrance_exists_weight),
        "terminal_scoring_mode": str(config.terminal_scoring_mode),
        "dynamic_new_class_score": float(config.dynamic_new_class_score),
        "dynamic_known_class_score": float(config.dynamic_known_class_score),
        "dynamic_frequent_class_score": float(config.dynamic_frequent_class_score),
        "dynamic_frequent_class_threshold": int(config.dynamic_frequent_class_threshold),
        "exact_stop_ratio": float(config.exact_stop_ratio),
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


def _default_task_factory(config: InterruptSearchConfig) -> Callable[[int, int, int | None], InterruptSearchTask]:
    def factory(class_id: int, search_index: int, parent_search_index: int | None) -> InterruptSearchTask:
        scorer, _pattern_signature = build_class_scorer(
            _queue_like_config(config),
            class_id,
            rare_target_classes=set(int(value) for value in config.rare_target_classes),
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
            seed=int(config.seed) + 1009 * (search_index - 1),
        )
        state = InterruptSearchState(scorer=scorer, config=mcts_config)
        root_key = empty_key(len(scorer.blocks))
        state.root = _get_or_create_node(state.nodes, scorer, root_key, path=[])
        threshold = (2.0 / 5.0) * float(config.iterations)
        return InterruptSearchTask(
            class_id=class_id,
            search_index=search_index,
            parent_search_index=parent_search_index,
            state=state,
            threshold=threshold,
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


def _pattern_shape_key(signature: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return an abstract, order-independent key for a pattern.

    The user-defined pattern identity is the block grouping shape, not the raw
    vertex set. Two classes belong to the same pattern if their blocks have the
    same size multiset, even when the exact chosen subset of blocks differs.
    """
    return tuple(sorted(len(block) for block in signature))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stack-based interruptible MCTS exploration.")
    parser.add_argument("--initial-class-id", type=int, default=1)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--rare-target-classes", type=int, nargs="*", default=[])
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--exploration-constant", type=float, default=1.4)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-temperature", type=float, default=0.85)
    parser.add_argument("--expansion-candidate-pool", type=int, default=16)
    parser.add_argument("--rollout-candidate-pool", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--rank24-entrance-exists-weight", type=float, default=6.0)
    parser.add_argument("--terminal-scoring-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--dynamic-new-class-score", type=float, default=100.0)
    parser.add_argument("--dynamic-known-class-score", type=float, default=10.0)
    parser.add_argument("--dynamic-frequent-class-score", type=float, default=-5.0)
    parser.add_argument("--dynamic-frequent-class-threshold", type=int, default=16)
    parser.add_argument("--exact-stop-ratio", type=float, default=2.0 / 5.0)
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
        seed=int(args.seed),
        rank24_entrance_exists_weight=float(args.rank24_entrance_exists_weight),
        terminal_scoring_mode=str(args.terminal_scoring_mode),
        dynamic_new_class_score=float(args.dynamic_new_class_score),
        dynamic_known_class_score=float(args.dynamic_known_class_score),
        dynamic_frequent_class_score=float(args.dynamic_frequent_class_score),
        dynamic_frequent_class_threshold=int(args.dynamic_frequent_class_threshold),
        exact_stop_ratio=float(args.exact_stop_ratio),
    )
    report = run_interruptible_search(config, output_path=args.output)
    payload = report.to_dict()
    print(json.dumps({"stop_reason": payload["stop_reason"], "finished_class_ids": payload["finished_class_ids"]}, ensure_ascii=False))
    print(args.output)
    print(args.output.with_name(f"{args.output.stem}.runs.json"))


if __name__ == "__main__":
    main()