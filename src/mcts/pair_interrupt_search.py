from __future__ import annotations

"""Hybrid pair-involution discovery and interruptible MCTS search.

The pair phase searches all selected fixed-point-free involution patterns
without target masks.  Every exact class found at a complete 13-pair terminal
is then injected as a discovery event at the root of the interrupt supervisor.
The supervisor opens a normal class-rooted interrupt tree for each injected
class before resuming the original root task.

Only observed terminal supports cross the bridge.  Pair masks, representable
target masks, and post-search PairTargetBank diagnostics never enter an
interrupt scorer.  Interrupt tasks share one global discovery state, terminal
validation cache, and structure cache.
"""

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from baseline.facet_validator import FacetValidator
from baseline.reference_classes import (
    DEFAULT_EXAMPLES_PATH,
    build_support_class_index,
)
from baseline.scorer import exact_class_id
from mcts.interrupt_search import (
    InterruptGlobalDiscoveryState,
    InterruptSearchConfig,
    InterruptSearchReport,
    InterruptSearchTask,
    _default_task_factory,
    run_interruptible_search,
)
from mcts.pair_involution_search import (
    PairSearchConfig,
    run_pair_involution_search,
)
from mcts.search import ExactClassDiscovery


@dataclass(frozen=True)
class PairInterruptSearchConfig:
    """Shared top-level configuration for both search phases."""

    initial_class_id: int = 1
    iterations: int = 200
    target_classes: tuple[int, ...] = tuple(range(1, 47))
    rare_target_classes: tuple[int, ...] = tuple(range(1, 47))
    examples_path: Path = DEFAULT_EXAMPLES_PATH
    rep_index: int = 1
    pattern_index: int = 0
    seed: int = 20260502
    pair_pattern_ids: tuple[int, ...] | None = None
    rank19_lookahead_candidate_pool: int = 8
    corrector_enabled: bool = True
    corrector_interval: int = 8
    corrector_max_events_per_tree: int = 24
    corrector_source_pool: int = 8
    pair_checkpoint_interval: int = 50
    reuse_complete_pair_output: bool = False

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.rank19_lookahead_candidate_pool < 0:
            raise ValueError("rank19_lookahead_candidate_pool must be nonnegative")
        if self.corrector_interval <= 0:
            raise ValueError("corrector_interval must be positive")
        if self.corrector_max_events_per_tree < 0:
            raise ValueError("corrector_max_events_per_tree must be nonnegative")
        if self.corrector_source_pool <= 0:
            raise ValueError("corrector_source_pool must be positive")


@dataclass(frozen=True)
class PairBridgeSeed:
    """One exact pair terminal promoted to an interrupt-stack seed."""

    class_id: int
    pattern_id: int
    pair_round: int
    pair_local_iteration: int
    pair_global_iteration: int
    pair_wall_seconds: float
    selected_blocks: tuple[int, ...]
    support_word: int | None
    source: str

    def as_exact_discovery(self, *, score: float) -> ExactClassDiscovery:
        return ExactClassDiscovery(
            class_id=int(self.class_id),
            label=f"exact:class{int(self.class_id)}",
            iteration=0,
            depth=len(self.selected_blocks),
            score=float(score),
            rank=25,
            path=list(self.selected_blocks),
            chosen_blocks=list(self.selected_blocks),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": int(self.class_id),
            "source": self.source,
            "pattern_id": int(self.pattern_id),
            "pair_round": int(self.pair_round),
            "pair_local_iteration": int(self.pair_local_iteration),
            "pair_global_iteration": int(self.pair_global_iteration),
            "pair_wall_seconds": float(self.pair_wall_seconds),
            "selected_blocks": list(self.selected_blocks),
            "support_word_hex": (
                None
                if self.support_word is None
                else f"0x{int(self.support_word):016x}"
            ),
        }


@dataclass
class TimedInterruptGlobalDiscoveryState(InterruptGlobalDiscoveryState):
    """Interrupt global state with wall-clock records for first discoveries."""

    hybrid_started_at: float = 0.0
    record_timeline: bool = False
    timed_discoveries: list[dict[str, object]] = field(default_factory=list)

    def mark_discovered(self, class_id: int) -> bool:
        is_new = super().mark_discovered(class_id)
        if is_new and self.record_timeline:
            self.timed_discoveries.append(
                {
                    "class_id": int(class_id),
                    "source": "interrupt_search",
                    "wall_seconds": time.perf_counter() - self.hybrid_started_at,
                    "discovery_epoch": int(self.discovery_epoch),
                }
            )
        return is_new


@dataclass
class _InterruptProgress:
    started_at: float
    global_discovery: TimedInterruptGlobalDiscoveryState
    completed_tasks: int = 0
    emitted_bridge_classes: list[int] = field(default_factory=list)


@dataclass
class _HybridInterruptTask:
    """Duck-typed task adapter that emits bridge seeds before normal steps."""

    delegate: InterruptSearchTask
    pending_seeds: deque[PairBridgeSeed]
    progress: _InterruptProgress
    bridge_score: float
    completion_reported: bool = False

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    def step(self) -> list[ExactClassDiscovery]:
        if self.pending_seeds:
            seed = self.pending_seeds.popleft()
            self.progress.emitted_bridge_classes.append(int(seed.class_id))
            print(
                "bridge_seed "
                f"class={seed.class_id} pattern={seed.pattern_id} "
                f"pair_round={seed.pair_round}",
                flush=True,
            )
            return [seed.as_exact_discovery(score=self.bridge_score)]

        discoveries = self.delegate.step()
        if discoveries:
            print(
                "interrupt_discovery "
                f"from_class={self.delegate.class_id} "
                f"classes={','.join(str(item.class_id) for item in discoveries)} "
                f"iteration={self.delegate.state.iterations_completed}",
                flush=True,
            )
        if self.delegate.finished and not self.completion_reported:
            self.completion_reported = True
            self.progress.completed_tasks += 1
            elapsed = time.perf_counter() - self.progress.started_at
            print(
                "interrupt_task_finished "
                f"class={self.delegate.class_id} "
                f"iterations={self.delegate.state.iterations_completed} "
                f"stop={self.delegate.state.stop_reason} "
                f"completed_tasks={self.progress.completed_tasks} "
                f"coverage={len(self.progress.global_discovery.discovered_exact_classes)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        return discoveries


class PairBridgeTaskFactory:
    """Wrap the initial interrupt task with terminal discoveries from pair MCTS."""

    def __init__(
        self,
        base_factory: Callable[[int, int, int | None], InterruptSearchTask],
        *,
        initial_class_id: int,
        seeds: Sequence[PairBridgeSeed],
        progress: _InterruptProgress,
        bridge_score: float,
    ) -> None:
        self.base_factory = base_factory
        self.initial_class_id = int(initial_class_id)
        self.seeds = tuple(seeds)
        self.progress = progress
        self.bridge_score = float(bridge_score)
        self._initial_wrapped = False

    def __call__(
        self,
        class_id: int,
        search_index: int,
        parent_search_index: int | None,
    ) -> _HybridInterruptTask:
        delegate = self.base_factory(class_id, search_index, parent_search_index)
        is_initial = (
            not self._initial_wrapped
            and parent_search_index is None
            and int(class_id) == self.initial_class_id
        )
        if is_initial:
            self._initial_wrapped = True
        pending = deque(self.seeds if is_initial else ())
        return _HybridInterruptTask(
            delegate=delegate,
            pending_seeds=pending,
            progress=self.progress,
            bridge_score=self.bridge_score,
        )


def _bridge_seeds_from_pair_payload(
    payload: dict[str, object],
    *,
    initial_class_id: int,
) -> tuple[PairBridgeSeed, ...]:
    raw_timeline = payload.get("discovery_timeline", [])
    if not isinstance(raw_timeline, list):
        return ()
    seen = {int(initial_class_id)}
    seeds: list[PairBridgeSeed] = []
    for raw_event in raw_timeline:
        if not isinstance(raw_event, dict) or "class_id" not in raw_event:
            continue
        class_id = int(raw_event["class_id"])
        if class_id in seen:
            continue
        seen.add(class_id)
        raw_word = raw_event.get("support_word_hex")
        support_word = (
            None
            if raw_word in (None, "")
            else int(str(raw_word), 16)
        )
        seeds.append(
            PairBridgeSeed(
                class_id=class_id,
                pattern_id=int(raw_event.get("pattern_id") or 0),
                pair_round=int(raw_event.get("round") or 0),
                pair_local_iteration=int(raw_event.get("local_iteration") or 0),
                pair_global_iteration=int(raw_event.get("global_iteration") or 0),
                pair_wall_seconds=float(raw_event.get("wall_seconds") or 0.0),
                selected_blocks=tuple(
                    int(value) for value in raw_event.get("selected_blocks", [])
                ),
                support_word=support_word,
                source=str(raw_event.get("source") or "pair_terminal"),
            )
        )
    return tuple(seeds)


def _component_paths(output_path: Path) -> tuple[Path, Path]:
    pair_path = output_path.with_name(f"{output_path.stem}.pair.json")
    interrupt_path = output_path.with_name(f"{output_path.stem}.interrupt.json")
    return pair_path, interrupt_path


def _pair_config(
    config: PairInterruptSearchConfig,
    *,
    output_path: Path,
) -> PairSearchConfig:
    return PairSearchConfig(
        initial_class_id=int(config.initial_class_id),
        iterations=int(config.iterations),
        target_classes=tuple(int(value) for value in config.target_classes),
        examples_path=config.examples_path,
        seed=int(config.seed),
        rank19_lookahead_candidate_pool=int(
            config.rank19_lookahead_candidate_pool
        ),
        corrector_enabled=bool(config.corrector_enabled),
        corrector_interval=int(config.corrector_interval),
        corrector_max_events_per_tree=int(
            config.corrector_max_events_per_tree
        ),
        corrector_source_pool=int(config.corrector_source_pool),
        checkpoint_interval=int(config.pair_checkpoint_interval),
        pattern_ids=config.pair_pattern_ids,
        baseline_path=output_path.with_name(
            f"{output_path.stem}.historical_baseline_disabled.json"
        ),
    )


def _interrupt_config(config: PairInterruptSearchConfig) -> InterruptSearchConfig:
    return InterruptSearchConfig(
        initial_class_id=int(config.initial_class_id),
        rep_index=int(config.rep_index),
        pattern_index=int(config.pattern_index),
        target_classes=tuple(int(value) for value in config.target_classes),
        rare_target_classes=tuple(
            int(value) for value in config.rare_target_classes
        ),
        examples_path=config.examples_path,
        iterations=int(config.iterations),
        seed=int(config.seed),
    )


def _seed_interrupt_global_state(
    config: PairInterruptSearchConfig,
    pair_payload: dict[str, object],
    seeds: Sequence[PairBridgeSeed],
    *,
    hybrid_started_at: float,
) -> TimedInterruptGlobalDiscoveryState:
    interrupt_config = _interrupt_config(config)
    state = TimedInterruptGlobalDiscoveryState.from_config(interrupt_config)
    state.hybrid_started_at = float(hybrid_started_at)
    state.mark_discovered(int(config.initial_class_id))
    for seed in seeds:
        state.mark_discovered(int(seed.class_id))

    summary = pair_payload.get("summary", {})
    raw_counts = summary.get("class_hit_counts", {}) if isinstance(summary, dict) else {}
    if isinstance(raw_counts, dict):
        for raw_class_id, raw_count in raw_counts.items():
            state.discovered_label_counts[
                f"exact:class{int(raw_class_id)}"
            ] += int(raw_count)
    state.record_timeline = True
    return state


def _prewarm_interrupt_terminal_cache(
    state: TimedInterruptGlobalDiscoveryState,
    seeds: Sequence[PairBridgeSeed],
    *,
    examples_path: Path,
) -> int:
    support_index = build_support_class_index(examples_path)
    validator = FacetValidator(support_class_index=support_index)
    seeded = 0
    for seed in seeds:
        if seed.support_word is None:
            continue
        label = validator.validate_indices(
            [
                vertex
                for vertex in range(64)
                if int(seed.support_word) & (1 << vertex)
            ]
        )
        observed_class = exact_class_id(label.label)
        if observed_class != int(seed.class_id):
            raise RuntimeError(
                "pair bridge terminal changed during validation: "
                f"expected class {seed.class_id}, got {label.label}"
            )
        state.terminal_validation_cache.put(int(seed.support_word), label)
        seeded += 1
    return seeded


def _load_reusable_pair_payload(
    path: Path,
    config: PairInterruptSearchConfig,
) -> dict[str, object] | None:
    if not config.reuse_complete_pair_output or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pair output is not a JSON object: {path}")
    meta = payload.get("meta", {})
    if not isinstance(meta, dict) or meta.get("status") != "complete":
        raise ValueError(f"pair output is not complete: {path}")

    expected_pattern_ids = (
        None
        if config.pair_pattern_ids is None
        else list(config.pair_pattern_ids)
    )
    expected = {
        "initial_class_id": int(config.initial_class_id),
        "iterations_per_tree": int(config.iterations),
        "target_classes": [int(value) for value in config.target_classes],
        "seed": int(config.seed),
        "rank19_lookahead_candidate_pool": int(
            config.rank19_lookahead_candidate_pool
        ),
        "corrector_enabled": bool(config.corrector_enabled),
        "corrector_interval": int(config.corrector_interval),
        "corrector_max_events_per_tree": int(
            config.corrector_max_events_per_tree
        ),
        "corrector_source_pool": int(config.corrector_source_pool),
        "pattern_ids": expected_pattern_ids,
    }
    mismatches = {
        key: {"expected": value, "observed": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "pair output configuration does not match requested run: "
            f"{mismatches}"
        )
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _compact_interrupt_phase(
    report: InterruptSearchReport,
    *,
    output_path: Path,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "output_path": str(output_path),
        "runs_path": str(
            output_path.with_name(f"{output_path.stem}.runs.json")
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "started_class_ids": list(report.started_class_ids),
        "finished_class_ids": list(report.finished_class_ids),
        "interrupt_events": [event.to_dict() for event in report.interrupt_events],
        "summary": report.summary,
        "stop_reason": report.stop_reason,
        "exact_class_counts": report.exact_class_counts,
    }


def _combined_timeline(
    pair_payload: dict[str, object],
    state: TimedInterruptGlobalDiscoveryState,
    report: InterruptSearchReport,
) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    raw_pair_timeline = pair_payload.get("discovery_timeline", [])
    if isinstance(raw_pair_timeline, list):
        for raw_event in raw_pair_timeline:
            if isinstance(raw_event, dict):
                timeline.append({"phase": "pair", **raw_event})

    first_interrupt_event_by_class = {}
    for event in report.interrupt_events:
        first_interrupt_event_by_class.setdefault(int(event.class_id), event)
    for timed_event in state.timed_discoveries:
        class_id = int(timed_event["class_id"])
        event = first_interrupt_event_by_class.get(class_id)
        enriched = {"phase": "interrupt", **timed_event}
        if event is not None:
            enriched.update(
                {
                    "global_iteration": int(event.global_iteration),
                    "search_index": int(event.search_index),
                    "start_class_id": int(event.start_class_id),
                    "local_iteration": int(event.iteration),
                    "depth": int(event.depth),
                }
            )
        timeline.append(enriched)
    timeline.sort(key=lambda item: float(item.get("wall_seconds", 0.0)))
    for event_index, event in enumerate(timeline):
        event["hybrid_event_index"] = event_index
    return timeline


def run_pair_interrupt_search(
    config: PairInterruptSearchConfig,
    *,
    output_path: Path,
    pair_runner: Callable[..., dict[str, object]] = run_pair_involution_search,
    interrupt_runner: Callable[..., InterruptSearchReport] = run_interruptible_search,
) -> dict[str, object]:
    """Run pair discovery, bridge exact terminals, then run interrupt MCTS."""
    invocation_started_at = time.perf_counter()
    pair_output_path, interrupt_output_path = _component_paths(output_path)

    pair_payload = _load_reusable_pair_payload(pair_output_path, config)
    pair_output_reused = pair_payload is not None
    if pair_payload is None:
        pair_started_at = time.perf_counter()
        pair_payload = pair_runner(
            _pair_config(config, output_path=output_path),
            output_path=pair_output_path,
        )
        pair_elapsed_seconds = time.perf_counter() - pair_started_at
    else:
        pair_meta = pair_payload.get("meta", {})
        pair_elapsed_seconds = float(
            pair_meta.get("elapsed_seconds", 0.0)
            if isinstance(pair_meta, dict)
            else 0.0
        )
        print(
            "pair_phase_reused "
            f"path={pair_output_path} elapsed={pair_elapsed_seconds:.1f}s",
            flush=True,
        )
    pair_completed_at = time.perf_counter()
    hybrid_started_at = (
        invocation_started_at
        if not pair_output_reused
        else pair_completed_at - pair_elapsed_seconds
    )
    seeds = _bridge_seeds_from_pair_payload(
        pair_payload,
        initial_class_id=config.initial_class_id,
    )

    global_discovery = _seed_interrupt_global_state(
        config,
        pair_payload,
        seeds,
        hybrid_started_at=hybrid_started_at,
    )
    terminal_cache_seed_count = _prewarm_interrupt_terminal_cache(
        global_discovery,
        seeds,
        examples_path=config.examples_path,
    )
    interrupt_config = _interrupt_config(config)
    progress = _InterruptProgress(
        started_at=time.perf_counter(),
        global_discovery=global_discovery,
    )
    task_factory = PairBridgeTaskFactory(
        _default_task_factory(interrupt_config, global_discovery),
        initial_class_id=config.initial_class_id,
        seeds=seeds,
        progress=progress,
        bridge_score=100.0,
    )
    bridge_elapsed_seconds = time.perf_counter() - pair_completed_at

    pair_summary = pair_payload.get("summary", {})
    _write_json(
        output_path,
        {
            "meta": {
                "script": "src/mcts/pair_interrupt_search.py",
                "algorithm": "pair_terminal_bridge_interrupt_mcts",
                "status": "interrupt_running",
                "iterations_per_pair_tree_and_interrupt_task": int(
                    config.iterations
                ),
                "target_guidance_enabled": False,
                "pair_output_reused": pair_output_reused,
            },
            "pair_phase": {
                "output_path": str(pair_output_path),
                "elapsed_seconds": pair_elapsed_seconds,
                "summary": pair_summary,
            },
            "bridge": {
                "seed_class_ids": [seed.class_id for seed in seeds],
                "terminal_cache_seed_count": terminal_cache_seed_count,
                "events": [seed.to_dict() for seed in seeds],
            },
        },
    )

    interrupt_started_at = time.perf_counter()
    interrupt_report = interrupt_runner(
        interrupt_config,
        task_factory=task_factory,
        global_discovery_state=global_discovery,
        deduplicate_patterns=True,
        output_path=interrupt_output_path,
    )
    interrupt_elapsed_seconds = time.perf_counter() - interrupt_started_at
    invocation_elapsed_seconds = time.perf_counter() - invocation_started_at
    elapsed_seconds = (
        pair_elapsed_seconds
        + bridge_elapsed_seconds
        + interrupt_elapsed_seconds
    )

    targets = {int(value) for value in config.target_classes}
    coverage = sorted(global_discovery.discovered_exact_classes & targets)
    interrupt_new_classes = sorted(
        int(event["class_id"]) for event in global_discovery.timed_discoveries
    )
    timeline = _combined_timeline(
        pair_payload,
        global_discovery,
        interrupt_report,
    )
    payload: dict[str, object] = {
        "meta": {
            "script": "src/mcts/pair_interrupt_search.py",
            "algorithm": "pair_terminal_bridge_interrupt_mcts",
            "status": "complete",
            "initial_class_id": int(config.initial_class_id),
            "iterations_per_pair_tree_and_interrupt_task": int(config.iterations),
            "target_classes": sorted(targets),
            "rare_target_classes": sorted(
                int(value) for value in config.rare_target_classes
            ),
            "examples_path": str(config.examples_path),
            "seed": int(config.seed),
            "pair_pattern_ids": (
                None
                if config.pair_pattern_ids is None
                else list(config.pair_pattern_ids)
            ),
            "target_guidance_enabled": False,
            "pair_target_bank_usage": "post_search_diagnostics_only",
            "bridge_payload": "observed_complete_terminal_supports_only",
            "interrupt_global_discovery_shared": True,
            "interrupt_terminal_validation_cache_shared": True,
            "interrupt_structure_cache_shared": True,
            "pair_terminal_cache_bridge": "exact_discovered_supports_only",
            "pair_output_reused": pair_output_reused,
            "pair_elapsed_seconds": pair_elapsed_seconds,
            "bridge_elapsed_seconds": bridge_elapsed_seconds,
            "interrupt_elapsed_seconds": interrupt_elapsed_seconds,
            "elapsed_seconds": elapsed_seconds,
            "wall_seconds_this_invocation": invocation_elapsed_seconds,
        },
        "summary": {
            "coverage_class_ids": coverage,
            "coverage_count": len(coverage),
            "missing_class_ids": sorted(targets - set(coverage)),
            "pair_discovered_class_ids": [seed.class_id for seed in seeds],
            "bridge_seeded_class_ids": list(progress.emitted_bridge_classes),
            "interrupt_new_class_ids": interrupt_new_classes,
            "interrupt_new_class_count": len(interrupt_new_classes),
            "class18_found_by_pair_phase": any(
                seed.class_id == 18 for seed in seeds
            ),
            "class18_started_by_interrupt_phase": (
                18 in interrupt_report.started_class_ids
            ),
            "interrupt_started_class_count": len(
                interrupt_report.started_class_ids
            ),
            "interrupt_finished_class_count": len(
                interrupt_report.finished_class_ids
            ),
        },
        "discovery_timeline": timeline,
        "pair_phase": {
            "output_path": str(pair_output_path),
            "runs_path": str(
                pair_output_path.with_name(f"{pair_output_path.stem}.runs.json")
            ),
            "elapsed_seconds": pair_elapsed_seconds,
            "summary": pair_summary,
            "terminal_cache": pair_payload.get("terminal_cache", {}),
        },
        "bridge": {
            "seed_class_ids": [seed.class_id for seed in seeds],
            "terminal_cache_seed_count": terminal_cache_seed_count,
            "events": [seed.to_dict() for seed in seeds],
        },
        "interrupt_phase": _compact_interrupt_phase(
            interrupt_report,
            output_path=interrupt_output_path,
            elapsed_seconds=interrupt_elapsed_seconds,
        ),
        "interrupt_shared_caches": {
            "terminal_validation_entries": len(
                global_discovery.terminal_validation_cache.labels
            ),
            "terminal_validation_hits": int(
                global_discovery.terminal_validation_cache.hits
            ),
            "terminal_validation_misses": int(
                global_discovery.terminal_validation_cache.misses
            ),
            "terminal_validation_evictions": int(
                global_discovery.terminal_validation_cache.evictions
            ),
            "structure_cache_hits": int(global_discovery.structure_cache.hits),
            "structure_cache_misses": int(global_discovery.structure_cache.misses),
            "structure_cache_evictions": int(
                global_discovery.structure_cache.evictions
            ),
        },
    }
    _write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run unbiased pair-terminal discovery and bridge discoveries into "
            "stack-based interrupt MCTS."
        )
    )
    parser.add_argument("--initial-class-id", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--target-classes",
        type=int,
        nargs="*",
        default=list(range(1, 47)),
    )
    parser.add_argument(
        "--rare-target-classes",
        type=int,
        nargs="*",
        default=list(range(1, 47)),
    )
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES_PATH)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--pair-pattern-ids", type=int, nargs="*")
    parser.add_argument(
        "--rank19-lookahead-candidate-pool",
        type=int,
        default=8,
    )
    parser.add_argument("--disable-corrector", action="store_true")
    parser.add_argument("--corrector-interval", type=int, default=8)
    parser.add_argument("--corrector-max-events-per-tree", type=int, default=24)
    parser.add_argument("--corrector-source-pool", type=int, default=8)
    parser.add_argument("--pair-checkpoint-interval", type=int, default=50)
    parser.add_argument("--reuse-complete-pair-output", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> PairInterruptSearchConfig:
    return PairInterruptSearchConfig(
        initial_class_id=int(args.initial_class_id),
        iterations=int(args.iterations),
        target_classes=tuple(int(value) for value in args.target_classes),
        rare_target_classes=tuple(
            int(value) for value in args.rare_target_classes
        ),
        examples_path=args.examples,
        rep_index=int(args.rep_index),
        pattern_index=int(args.pattern_index),
        seed=int(args.seed),
        pair_pattern_ids=(
            None
            if args.pair_pattern_ids is None
            else tuple(int(value) for value in args.pair_pattern_ids)
        ),
        rank19_lookahead_candidate_pool=int(
            args.rank19_lookahead_candidate_pool
        ),
        corrector_enabled=not bool(args.disable_corrector),
        corrector_interval=int(args.corrector_interval),
        corrector_max_events_per_tree=int(
            args.corrector_max_events_per_tree
        ),
        corrector_source_pool=int(args.corrector_source_pool),
        pair_checkpoint_interval=int(args.pair_checkpoint_interval),
        reuse_complete_pair_output=bool(args.reuse_complete_pair_output),
    )


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    output_path = args.output or Path(
        "src/mcts/runs/"
        f"pair_interrupt_search_class{config.initial_class_id}_i{config.iterations}.json"
    )
    payload = run_pair_interrupt_search(config, output_path=output_path)
    print(
        json.dumps(
            {
                "elapsed_seconds": payload["meta"]["elapsed_seconds"],
                "coverage_count": payload["summary"]["coverage_count"],
                "coverage_class_ids": payload["summary"]["coverage_class_ids"],
                "pair_discovered_class_ids": payload["summary"][
                    "pair_discovered_class_ids"
                ],
                "interrupt_new_class_ids": payload["summary"][
                    "interrupt_new_class_ids"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(output_path, flush=True)


if __name__ == "__main__":
    main()
