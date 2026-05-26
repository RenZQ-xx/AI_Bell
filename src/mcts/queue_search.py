from __future__ import annotations

"""Queue-driven MCTS supervisor.

This module runs one MCTS search at a time. Each search can discover new exact
classes; those classes are queued as new search roots, while already-started
classes are never used again as roots.
"""

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import sys

# Make this module runnable as a script from the repository root. Ensure the
# package `mcts` and sibling packages like `baseline` are importable by
# inserting the `src` directory into `sys.path`.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from baseline.orbit_blocks import build_orbit_patterns_from_support
from baseline.reference_classes import DEFAULT_EXAMPLES_PATH, parse_example_rows, support_mask_from_row
from baseline.scorer import ExpansionScorer, ScorerConfig, exact_class_id

from mcts.search import ExactClassDiscovery, MCTSConfig, MCTSResult, run_mcts_search


@dataclass(frozen=True)
class QueueSearchConfig:
    """Inputs for queue-driven MCTS exploration."""

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


@dataclass(frozen=True)
class QueueDiscoveryEvent:
    """A first-time exact-class discovery recorded by the supervisor."""

    search_index: int
    start_class_id: int
    class_id: int
    iteration: int
    depth: int
    score: float
    rank: int
    label: str
    pattern_status: str
    pattern_signature: list[list[int]]
    rare_removed: bool
    path: list[int]
    chosen_blocks: list[int]

    @property
    def summary(self) -> str:
        return f"第{self.search_index}次搜索中第{self.iteration}次iteration第{self.depth}深度时找到 class {self.class_id}"

    def to_dict(self) -> dict[str, object]:
        return {
            "search_index": self.search_index,
            "start_class_id": self.start_class_id,
            "class_id": self.class_id,
            "iteration": self.iteration,
            "depth": self.depth,
            "score": self.score,
            "rank": self.rank,
            "label": self.label,
            "pattern_status": self.pattern_status,
            "pattern_signature": [list(block) for block in self.pattern_signature],
            "rare_removed": self.rare_removed,
            "path": list(self.path),
            "chosen_blocks": list(self.chosen_blocks),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class QueueSearchRun:
    """One MCTS run launched from one queued class root."""

    search_index: int
    start_class_id: int
    seed: int
    pattern_index: int
    result: dict[str, object]
    discovered_class_ids: list[int]
    newly_queued_class_ids: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "search_index": self.search_index,
            "start_class_id": self.start_class_id,
            "seed": self.seed,
            "pattern_index": self.pattern_index,
            "result": self.result,
            "discovered_class_ids": list(self.discovered_class_ids),
            "newly_queued_class_ids": list(self.newly_queued_class_ids),
        }


@dataclass(frozen=True)
class QueueSearchReport:
    """Aggregate output of the queue-driven search supervisor."""

    meta: dict[str, object]
    started_class_ids: list[int]
    discovered_class_ids: list[int]
    remaining_queue: list[int]
    remaining_rare_target_classes: list[int]
    stop_reason: str
    discovery_events: list[QueueDiscoveryEvent]
    runs: list[QueueSearchRun]
    summary: dict[str, object]
    label_counts: dict[str, int]
    exact_class_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "meta": self.meta,
            "started_class_ids": list(self.started_class_ids),
            "discovered_class_ids": list(self.discovered_class_ids),
            "remaining_queue": list(self.remaining_queue),
            "remaining_rare_target_classes": list(self.remaining_rare_target_classes),
            "stop_reason": self.stop_reason,
            "discovery_events": [event.to_dict() for event in self.discovery_events],
            "runs": [run.to_dict() for run in self.runs],
            "summary": self.summary,
            "label_counts": dict(sorted(self.label_counts.items())),
            "exact_class_counts": dict(sorted(self.exact_class_counts.items())),
        }


def compact_label(label: str) -> str:
    class_id = exact_class_id(label)
    if class_id is not None:
        return f"exact:class{class_id}"
    return label


def summarize(results: Sequence[MCTSResult], rare_target_classes: set[int]) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    encountered: Counter[str] = Counter()
    exact_class_counts: Counter[int] = Counter()
    for result in results:
        if result.best is not None:
            label_counts[compact_label(result.best.label)] += 1
        for label, count in result.encountered_label_counts.items():
            compact = compact_label(label)
            encountered[compact] += int(count)
            class_id = exact_class_id(label)
            if class_id is not None:
                exact_class_counts[class_id] += 1
    opened_rare = sorted(
        class_id
        for label in label_counts
        if (class_id := exact_class_id(label)) in rare_target_classes
    )
    encountered_rare = sorted(
        class_id
        for label, count in encountered.items()
        if count > 0 and (class_id := exact_class_id(label)) in rare_target_classes
    )
    return {
        "label_counts": dict(sorted(label_counts.items())),
        "exact_class_counts": {str(key): value for key, value in sorted(exact_class_counts.items())},
        "opened_rare_target_classes": opened_rare,
        "rare_target_coverage_count": len(opened_rare),
        "encountered_label_counts": dict(sorted(encountered.items())),
        "encountered_rare_target_classes": encountered_rare,
        "encountered_rare_target_coverage_count": len(encountered_rare),
    }


def build_class_scorer(
    config: QueueSearchConfig,
    class_id: int,
    *,
    rare_target_classes: set[int] | None = None,
) -> tuple[ExpansionScorer, tuple[tuple[int, ...], ...]]:
    """Build the scorer for one exact class root using its representative support."""
    example_rows = parse_example_rows(config.examples_path)
    class_rows = example_rows.get(int(class_id))
    if class_rows is None:
        raise KeyError(f"unknown exact class {class_id}")
    row = class_rows.get(int(config.rep_index))
    if row is None:
        available = ", ".join(str(key) for key in sorted(class_rows))
        raise KeyError(f"class {class_id} has no rep_index {config.rep_index}; available: {available}")

    support = support_mask_from_row(row)
    patterns = build_orbit_patterns_from_support(
        support,
        class_id=int(class_id),
        rep_index=int(config.rep_index),
        max_patterns=int(config.pattern_index) + 1,
    )
    if int(config.pattern_index) >= len(patterns):
        raise IndexError(
            f"pattern_index {config.pattern_index} is out of range for class {class_id}; got {len(patterns)} patterns"
        )
    pattern = patterns[int(config.pattern_index)]
    active_rare_target_classes = set(int(value) for value in config.rare_target_classes) if rare_target_classes is None else set(int(value) for value in rare_target_classes)
    scorer = ExpansionScorer(
        blocks=pattern.orbits,
        rare_target_classes=active_rare_target_classes,
        target_classes=set(int(value) for value in config.target_classes),
        config=ScorerConfig(
            rank24_entrance_exists_weight=float(config.rank24_entrance_exists_weight),
            terminal_scoring_mode=str(config.terminal_scoring_mode),
            dynamic_new_class_score=float(config.dynamic_new_class_score),
            dynamic_known_class_score=float(config.dynamic_known_class_score),
            dynamic_frequent_class_score=float(config.dynamic_frequent_class_score),
            dynamic_frequent_class_threshold=int(config.dynamic_frequent_class_threshold),
        ),
    )
    return scorer, tuple(tuple(int(v) for v in block) for block in pattern.orbits)


def _pattern_shape_key(signature: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return an abstract, order-independent key for a pattern.

    The key is the multiset of block sizes, which matches the user's notion of
    pattern identity for subset variants like 14/17/21.
    """
    return tuple(sorted(len(block) for block in signature))


def run_queue_supervisor(
    config: QueueSearchConfig,
    *,
    search_runner: Callable[[int, int], MCTSResult] | None = None,
    output_path: Path | None = None,
) -> QueueSearchReport:
    """Run queue-driven searches until the queue is empty or all targets are found."""
    current_rare_target_classes = set(int(value) for value in config.rare_target_classes)
    started_class_ids: set[int] = set()
    discovered_class_ids: set[int] = set()
    queue: deque[int] = deque([int(config.initial_class_id)])
    discovery_events: list[QueueDiscoveryEvent] = []
    runs: list[QueueSearchRun] = []
    results: list[MCTSResult] = []
    label_counts: Counter[str] = Counter()
    exact_class_counts: Counter[int] = Counter()

    target_classes = set(int(value) for value in config.target_classes)
    enable_pattern_dedup = search_runner is None
    search_runner = search_runner or _default_search_runner(config, current_rare_target_classes)
    search_index = 0
    queued_pattern_shapes: set[tuple[int, ...]] = set()
    started_pattern_shapes: set[tuple[int, ...]] = set()

    while queue and discovered_class_ids != target_classes:
        start_class_id = int(queue.popleft())
        if start_class_id in started_class_ids:
            continue

        _start_scorer, current_pattern_signature = build_class_scorer(
            config,
            start_class_id,
            rare_target_classes=current_rare_target_classes,
        )
        started_class_ids.add(start_class_id)
        # claim the abstract block shape for the starting pattern so subset
        # variants of the same grouping are considered identical.
        current_shape = _pattern_shape_key(current_pattern_signature)
        started_pattern_shapes.add(current_shape)
        search_index += 1
        result = search_runner(start_class_id, int(config.seed) + 1009 * (search_index - 1))
        results.append(result)

        run_discovered: list[int] = []
        queued_this_run: list[int] = []
        for discovery in result.exact_discoveries:
            label_counts[discovery.label] += 1
            class_id = int(discovery.class_id)
            exact_class_counts[class_id] += 1
            _, candidate_pattern_signature = build_class_scorer(
                config,
                class_id,
                rare_target_classes=current_rare_target_classes,
            )
            candidate_shape = _pattern_shape_key(candidate_pattern_signature)
            same_pattern = enable_pattern_dedup and (
                candidate_shape == current_shape
                or candidate_shape in queued_pattern_shapes
                or candidate_shape in started_pattern_shapes
            )
            rare_removed = False
            if class_id in current_rare_target_classes:
                current_rare_target_classes.remove(class_id)
                rare_removed = True
            if class_id in target_classes and class_id not in discovered_class_ids:
                discovered_class_ids.add(class_id)
                discovery_events.append(
                    QueueDiscoveryEvent(
                        search_index=search_index,
                        start_class_id=start_class_id,
                        class_id=class_id,
                        iteration=int(discovery.iteration),
                        depth=int(discovery.depth),
                        score=float(discovery.score),
                        rank=int(discovery.rank),
                        label=str(discovery.label),
                        pattern_status=(
                            "same_as_current"
                            if same_pattern and candidate_pattern_signature == current_pattern_signature
                            else "same_as_queued"
                            if same_pattern
                            else "unique"
                        ),
                        pattern_signature=[list(block) for block in candidate_pattern_signature],
                        rare_removed=rare_removed,
                        path=list(discovery.path),
                        chosen_blocks=list(discovery.chosen_blocks),
                    )
                )
                run_discovered.append(class_id)
            if (not same_pattern) and class_id not in started_class_ids and class_id not in queue:
                queue.append(class_id)
                queued_this_run.append(class_id)
                queued_pattern_shapes.add(candidate_shape)

        runs.append(
            QueueSearchRun(
                search_index=search_index,
                start_class_id=start_class_id,
                seed=int(config.seed) + 1009 * (search_index - 1),
                pattern_index=int(config.pattern_index),
                result=result.to_dict(),
                discovered_class_ids=run_discovered,
                newly_queued_class_ids=queued_this_run,
            )
        )

        if output_path is not None:
            snapshot = _build_report(
                config=config,
                started_class_ids=started_class_ids,
                discovered_class_ids=discovered_class_ids,
                queue=queue,
                remaining_rare_target_classes=current_rare_target_classes,
                discovery_events=discovery_events,
                runs=runs,
                results=results,
                label_counts=label_counts,
                exact_class_counts=exact_class_counts,
                stop_reason="all_target_classes_found" if discovered_class_ids == target_classes else "queue_empty",
            )
            _write_json_snapshot(output_path, snapshot.to_dict())

    stop_reason = "all_target_classes_found" if discovered_class_ids == target_classes else "queue_empty"
    return _build_report(
        config=config,
        started_class_ids=started_class_ids,
        discovered_class_ids=discovered_class_ids,
        queue=queue,
        remaining_rare_target_classes=current_rare_target_classes,
        discovery_events=discovery_events,
        runs=runs,
        results=results,
        label_counts=label_counts,
        exact_class_counts=exact_class_counts,
        stop_reason=stop_reason,
    )


def _build_report(
    *,
    config: QueueSearchConfig,
    started_class_ids: set[int],
    discovered_class_ids: set[int],
    queue: deque[int],
    remaining_rare_target_classes: set[int],
    discovery_events: list[QueueDiscoveryEvent],
    runs: list[QueueSearchRun],
    results: Sequence[MCTSResult],
    label_counts: Counter[str],
    exact_class_counts: Counter[int],
    stop_reason: str,
) -> QueueSearchReport:
    meta = {
        "script": "src/mcts/queue_search.py",
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
    }
    summary = summarize(results, set(int(value) for value in config.rare_target_classes))
    return QueueSearchReport(
        meta=meta,
        started_class_ids=sorted(started_class_ids),
        discovered_class_ids=sorted(discovered_class_ids),
        remaining_queue=sorted(set(queue)),
        remaining_rare_target_classes=sorted(remaining_rare_target_classes),
        stop_reason=stop_reason,
        discovery_events=discovery_events,
        runs=runs,
        summary=summary,
        label_counts=dict(sorted(label_counts.items())),
        exact_class_counts={str(key): value for key, value in sorted(exact_class_counts.items())},
    )


def _write_json_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _default_search_runner(config: QueueSearchConfig, current_rare_target_classes: set[int]) -> Callable[[int, int], MCTSResult]:
    def runner(start_class_id: int, seed: int) -> MCTSResult:
        scorer, _pattern_signature = build_class_scorer(
            config,
            start_class_id,
            rare_target_classes=current_rare_target_classes,
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
            seed=int(seed),
        )
        return run_mcts_search(scorer, config=mcts_config)

    return runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queue-driven MCTS exploration over exact classes.")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/mcts/runs/s7_mcts_probe_i500.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = QueueSearchConfig(
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
    )
    report = run_queue_supervisor(config, output_path=args.output)
    payload = report.to_dict()
    print(json.dumps({"stop_reason": payload["stop_reason"], "discovered_class_ids": payload["discovered_class_ids"]}, ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
