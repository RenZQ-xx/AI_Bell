from __future__ import annotations

"""Globally scheduled intrinsic subgroup searches for Bell (4,2,2).

Each subgroup owns a persistent reduced-coordinate MCTS tree.  A facet is
lifted and classified by one shared full-space oracle, while a shared novelty
registry makes the +10 class reward global across every lane.  Scheduling is
facet-blind: known node facet rows and target counts are never loaded.
"""

import argparse
import json
import math
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT, IDENTITY_PATTERN_ID, HandoffPatternAtlas
from .geometry import FacetOracle
from .reduced_subgroup_search import (
    ReducedSearchConfig,
    ReducedSubgroupMCTS,
    ReducedSubgroupProblem,
    SharedDiscoveryState,
)


DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs" / "global_reduced"
RACE_CYCLE = (
    "productive",
    "productive",
    "productive",
    "productive",
    "frontier",
    "productive",
    "productive",
    "frontier",
    "productive",
    "productive",
    "productive",
    "frontier",
    "productive",
    "fair",
    "structure",
    "revive",
)
STRUCTURE_CYCLE = ("C2 x C2", "C4", "C6", "S3")


@dataclass(frozen=True)
class GlobalReducedConfig:
    iterations: int = 10_000
    seed: int = 4_222_026
    warmup_iterations: int = 1
    race_chunk: int = 4
    lane_exploration: float = 1.25
    dormant_after_iterations: int = 160
    progress_interval: int = 250
    max_wall_seconds: float | None = None
    stop_class_count: int | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.warmup_iterations <= 0 or self.race_chunk <= 0:
            raise ValueError("warm-up and race chunk sizes must be positive")
        if self.lane_exploration < 0:
            raise ValueError("lane exploration must be nonnegative")
        if self.dormant_after_iterations <= 0:
            raise ValueError("dormant_after_iterations must be positive")
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")
        if self.stop_class_count is not None and self.stop_class_count <= 0:
            raise ValueError("stop_class_count must be positive")


@dataclass
class LaneState:
    node_id: str
    structure: str
    fair_index: int
    engine: ReducedSubgroupMCTS | None = field(default=None, repr=False)
    iterations: int = 0
    selections: int = 0
    elapsed_seconds: float = 0.0
    global_new_classes: int = 0
    local_new_classes: int = 0
    reduced_facets: int = 0
    successful_lifts: int = 0
    recent_utility: float = 0.0
    recent_frontier: float = 0.0
    last_global_discovery: int = 0
    first_selected_at: int | None = None
    last_selected_at: int | None = None
    global_class_ids: list[str] = field(default_factory=list)
    dormant_revivals: int = 0

    @property
    def activated(self) -> bool:
        return self.engine is not None


class GlobalReducedSearch:
    def __init__(
        self,
        config: GlobalReducedConfig,
        *,
        handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
        node_ids: Sequence[str] = (),
        lane_config: ReducedSearchConfig | None = None,
    ) -> None:
        self.config = config
        self.handoff_root = Path(handoff_root).resolve()
        requested = tuple(dict.fromkeys(map(str, node_ids)))
        atlas = HandoffPatternAtlas(
            self.handoff_root,
            allowed_pattern_ids=(requested or None),
        )
        available = {
            pattern.pattern_id: pattern
            for pattern in atlas.patterns
            if pattern.pattern_id != IDENTITY_PATTERN_ID
        }
        missing = [node_id for node_id in requested if node_id not in available]
        if missing:
            raise ValueError(f"unknown complete subgroup nodes: {missing}")
        fair_order = tuple(
            node_id
            for node_id in atlas.fair_order(config.seed)
            if node_id != IDENTITY_PATTERN_ID
        )
        if not fair_order:
            raise ValueError("no subgroup lanes selected")
        self.lanes = {
            node_id: LaneState(
                node_id=node_id,
                structure=available[node_id].structure,
                fair_index=index,
            )
            for index, node_id in enumerate(fair_order)
        }
        self.fair_order = fair_order
        self.lane_config = lane_config or ReducedSearchConfig(
            iterations=config.iterations,
            seed=config.seed,
            progress_interval=0,
        )
        self.shared_discovery = SharedDiscoveryState()
        self.full_oracle = FacetOracle(
            handoff_root=self.handoff_root,
            use_catalog=True,
            canonicalize=True,
        )
        self.global_iteration = 0
        self.scheduler_round = 0
        self.warmup_cursor = 0
        self.structure_cursor = 0
        self.started_at = 0.0
        self.termination_reason = "not_started"
        self.stats: Counter[str] = Counter()

    def _activate(self, lane: LaneState) -> ReducedSubgroupMCTS:
        if lane.engine is not None:
            return lane.engine
        problem = ReducedSubgroupProblem.load(self.handoff_root, lane.node_id)
        seed = self.config.seed ^ zlib.crc32(lane.node_id.encode("ascii"))
        lane_config = replace(
            self.lane_config,
            iterations=self.config.iterations,
            seed=seed,
            progress_interval=0,
            max_wall_seconds=None,
        )
        lane.engine = ReducedSubgroupMCTS(
            problem,
            lane_config,
            handoff_root=self.handoff_root,
            shared_discovery=self.shared_discovery,
            full_oracle=self.full_oracle,
        )
        lane.engine.started_at = time.perf_counter()
        lane.engine.termination_reason = "global_scheduler"
        self.stats["lanes_activated"] += 1
        return lane.engine

    def _warmup_choice(self) -> LaneState | None:
        lane_count = len(self.fair_order)
        for offset in range(lane_count):
            index = (self.warmup_cursor + offset) % lane_count
            lane = self.lanes[self.fair_order[index]]
            if lane.iterations < self.config.warmup_iterations:
                self.warmup_cursor = (index + 1) % lane_count
                return lane
        return None

    def _exploration_bonus(self, lane: LaneState) -> float:
        return self.config.lane_exploration * math.sqrt(
            math.log(self.scheduler_round + 2.0) / (lane.selections + 1.0)
        )

    def _productive_score(self, lane: LaneState) -> tuple[float, int, int]:
        score = (
            lane.recent_utility
            + 0.08 * lane.recent_frontier
            + self._exploration_bonus(lane)
        )
        return score, -lane.iterations, -lane.fair_index

    def _frontier_score(self, lane: LaneState) -> tuple[float, int, int]:
        score = lane.recent_frontier + 0.55 * self._exploration_bonus(lane)
        return score, -lane.iterations, -lane.fair_index

    def _is_dormant(self, lane: LaneState) -> bool:
        return bool(
            lane.global_new_classes == 0
            and lane.iterations >= self.config.dormant_after_iterations
        )

    def _revival_score(self, lane: LaneState) -> tuple[float, int, int]:
        lift_rate = lane.successful_lifts / max(1, lane.iterations)
        local_rate = lane.local_new_classes / max(1, lane.iterations)
        score = lane.recent_frontier + 0.50 * lift_rate + 0.25 * local_rate
        return score, -lane.iterations, -lane.fair_index

    def _race_choice(self) -> tuple[LaneState, str]:
        policy = RACE_CYCLE[self.scheduler_round % len(RACE_CYCLE)]
        candidates = list(self.lanes.values())
        active = [item for item in candidates if not self._is_dormant(item)]
        dormant = [item for item in candidates if self._is_dormant(item)]
        active = active or candidates
        if policy == "fair":
            lane = min(
                active,
                key=lambda item: (item.iterations, item.selections, item.fair_index),
            )
        elif policy == "frontier":
            lane = max(active, key=self._frontier_score)
        elif policy == "structure":
            structure = STRUCTURE_CYCLE[
                self.structure_cursor % len(STRUCTURE_CYCLE)
            ]
            self.structure_cursor += 1
            structural = [item for item in active if item.structure == structure]
            lane = max(structural or active, key=self._productive_score)
        elif policy == "revive":
            if dormant:
                lane = max(dormant, key=self._revival_score)
            else:
                policy = "productive"
                productive = [item for item in active if item.global_new_classes]
                lane = max(productive or active, key=self._productive_score)
        else:
            productive = [item for item in active if item.global_new_classes]
            lane = max(productive or active, key=self._productive_score)
        self.scheduler_round += 1
        return lane, policy

    def _wall_limit_reached(self) -> bool:
        return bool(
            self.config.max_wall_seconds is not None
            and time.perf_counter() - self.started_at >= self.config.max_wall_seconds
        )

    def _run_lane(self, lane: LaneState, count: int, policy: str) -> int:
        engine = self._activate(lane)
        before_global = len(self.shared_discovery.discovery_events)
        before_local = len(engine.discovered_classes)
        before_reduced = len(engine.discovered_reduced_facets)
        before_lifts = int(engine.geometry.stats["successful_full_lifts"])
        started = time.perf_counter()
        completed = 0
        for _ in range(count):
            if self.global_iteration >= self.config.iterations or self._wall_limit_reached():
                break
            self.global_iteration += 1
            lane.iterations += 1
            engine.run_iteration(
                lane.iterations,
                global_iteration=self.global_iteration,
            )
            completed += 1
            if (
                self.config.stop_class_count is not None
                and len(self.shared_discovery.discovered_classes)
                >= self.config.stop_class_count
            ):
                break
        elapsed = time.perf_counter() - started
        lane.elapsed_seconds += elapsed
        lane.selections += int(completed > 0)
        if policy == "revive" and completed:
            lane.dormant_revivals += 1
        lane.first_selected_at = lane.first_selected_at or self.global_iteration - completed + 1
        lane.last_selected_at = self.global_iteration
        new_events = self.shared_discovery.discovery_events[before_global:]
        global_gain = len(new_events)
        local_gain = len(engine.discovered_classes) - before_local
        reduced_gain = len(engine.discovered_reduced_facets) - before_reduced
        lift_gain = int(engine.geometry.stats["successful_full_lifts"]) - before_lifts
        lane.global_new_classes += global_gain
        lane.local_new_classes += local_gain
        lane.reduced_facets = len(engine.discovered_reduced_facets)
        lane.successful_lifts += lift_gain
        lane.global_class_ids.extend(str(event["class_id"]) for event in new_events)
        if global_gain:
            lane.last_global_discovery = self.global_iteration
        scale = max(1, completed)
        utility = (
            10.0 * global_gain
            + 0.10 * local_gain
            + 0.01 * reduced_gain
            + 0.05 * lift_gain
        ) / scale
        frontier = (0.15 * reduced_gain + lift_gain) / scale
        smoothing = 0.35
        if lane.selections == 1:
            lane.recent_utility = utility
            lane.recent_frontier = frontier
        else:
            lane.recent_utility = (
                (1.0 - smoothing) * lane.recent_utility + smoothing * utility
            )
            lane.recent_frontier = (
                (1.0 - smoothing) * lane.recent_frontier + smoothing * frontier
            )
        self.stats[f"scheduler_{policy}_selections"] += int(completed > 0)
        self.stats[f"scheduler_{policy}_iterations"] += completed
        self.stats["lane_elapsed_milliseconds"] += round(elapsed * 1000)
        return completed

    def _progress(self) -> None:
        active = sum(lane.activated for lane in self.lanes.values())
        print(
            f"[422-global-reduced] iteration={self.global_iteration} "
            f"classes={len(self.shared_discovery.discovered_classes)} "
            f"active_lanes={active}/{len(self.lanes)} "
            f"elapsed={time.perf_counter() - self.started_at:.1f}s",
            flush=True,
        )

    def run(self) -> dict[str, object]:
        self.started_at = time.perf_counter()
        self.shared_discovery.started_at = self.started_at
        self.termination_reason = "iteration_budget"
        next_progress = self.config.progress_interval
        while self.global_iteration < self.config.iterations:
            if self._wall_limit_reached():
                self.termination_reason = "wall_time_limit"
                break
            if (
                self.config.stop_class_count is not None
                and len(self.shared_discovery.discovered_classes)
                >= self.config.stop_class_count
            ):
                self.termination_reason = "class_count_target"
                break
            lane = self._warmup_choice()
            if lane is not None:
                policy = "warmup"
                count = 1
            else:
                lane, policy = self._race_choice()
                count = self.config.race_chunk
            if not self._run_lane(lane, count, policy):
                if self._wall_limit_reached():
                    self.termination_reason = "wall_time_limit"
                break
            if self.config.progress_interval and self.global_iteration >= next_progress:
                self._progress()
                while next_progress <= self.global_iteration:
                    next_progress += self.config.progress_interval
        return self.result()

    def result(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self.started_at if self.started_at else 0.0
        lane_summaries = []
        for node_id in self.fair_order:
            lane = self.lanes[node_id]
            engine = lane.engine
            lane_summaries.append(
                {
                    "node_id": lane.node_id,
                    "structure": lane.structure,
                    "activated": lane.activated,
                    "iterations": lane.iterations,
                    "selections": lane.selections,
                    "elapsed_seconds": lane.elapsed_seconds,
                    "global_new_class_count": lane.global_new_classes,
                    "global_new_class_ids": lane.global_class_ids,
                    "local_class_count": (
                        len(engine.discovered_classes) if engine is not None else 0
                    ),
                    "reduced_facet_count": lane.reduced_facets,
                    "successful_lift_hits": lane.successful_lifts,
                    "recent_utility": lane.recent_utility,
                    "recent_frontier": lane.recent_frontier,
                    "first_selected_at": lane.first_selected_at,
                    "last_selected_at": lane.last_selected_at,
                    "last_global_discovery": lane.last_global_discovery,
                    "dormant": self._is_dormant(lane),
                    "dormant_revivals": lane.dormant_revivals,
                    "tree_node_count": len(engine.nodes) if engine is not None else 0,
                    "tree_edge_count": (
                        sum(len(node.edges) for node in engine.nodes.values())
                        if engine is not None
                        else 0
                    ),
                    "root_basin_count": (
                        len(engine.root_basin_terminal_hits)
                        if engine is not None
                        else 0
                    ),
                    "productive_root_basin_count": (
                        len(engine.root_basin_discovery_hits)
                        if engine is not None
                        else 0
                    ),
                    "retained_face_count": (
                        len(engine.retained_face_exits) if engine is not None else 0
                    ),
                    "retained_face_exit_count": (
                        sum(
                            len(exits)
                            for exits in engine.retained_face_exits.values()
                        )
                        if engine is not None
                        else 0
                    ),
                    "search_stats": dict(engine.stats) if engine is not None else {},
                }
            )
        return {
            "algorithm": "bell422_global_intrinsic_subgroup_racing_mcts_v2",
            "prior_boundary": {
                "search_reads": [
                    "complete_nodes.csv: subgroup structure metadata only",
                    "nodes/*/reduced_vertices.csv",
                    "nodes/*/orbit_blocks.csv",
                ],
                "terminal_only_reads": ["global_facet_classes.csv"],
                "never_used": [
                    "nodes/*/facets.csv",
                    "complete_nodes.csv: facet_count",
                    "known reduced facet supports or H rows",
                    "per-node target class lists",
                ],
                "target_labels_affect_search": False,
            },
            "features": {
                "lazy_subgroup_lanes": True,
                "persistent_tree_per_lane": True,
                "facet_blind_lane_racing": True,
                "global_discovered_registry": True,
                "global_new_class_reward": True,
                "global_discovery_epoch": True,
                "shared_full_space_oracle": True,
                "canonical_lift_deduplication": True,
                "fair_structure_rotation": True,
                "stage_aware_lane_dormancy": True,
                "low_frequency_dormant_revival": True,
                "lift_conditioned_frontier_score": True,
            },
            "config": {
                "global": asdict(self.config),
                "lane": asdict(self.lane_config),
                "race_cycle": list(RACE_CYCLE),
                "structure_cycle": list(STRUCTURE_CYCLE),
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "iterations_completed": self.global_iteration,
                "planned_iterations": self.config.iterations,
                "termination_reason": self.termination_reason,
                "truncated": self.termination_reason == "wall_time_limit",
            },
            "coverage": {
                "discovered_class_count": len(
                    self.shared_discovery.discovered_classes
                ),
                "discovered_class_ids": list(
                    self.shared_discovery.discovered_classes
                ),
                "discoveries": self.shared_discovery.discovery_events,
                "class_frequencies": dict(self.shared_discovery.class_frequencies),
                "global_epoch": self.shared_discovery.epoch,
            },
            "scheduler": {
                "lane_count": len(self.lanes),
                "activated_lane_count": sum(
                    lane.activated for lane in self.lanes.values()
                ),
                "warmup_complete": all(
                    lane.iterations >= self.config.warmup_iterations
                    for lane in self.lanes.values()
                ),
                "rounds": self.scheduler_round,
                "statistics": dict(self.stats),
                "lanes": lane_summaries,
            },
            "shared_full_geometry": self.full_oracle.to_dict(),
        }

    @staticmethod
    def write_result(result: dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bell 4-2-2 global intrinsic subgroup racing MCTS"
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--node-id", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=4_222_026)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--race-chunk", type=int, default=4)
    parser.add_argument("--lane-exploration", type=float, default=1.25)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--max-wall-hours", type=float)
    parser.add_argument("--stop-class-count", type=int)
    parser.add_argument("--lane-max-depth", type=int, default=48)
    parser.add_argument("--output", type=Path)
    return parser


def run_search(
    *,
    config: GlobalReducedConfig,
    handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    node_ids: Sequence[str] = (),
    lane_config: ReducedSearchConfig | None = None,
    output: Path | None = None,
) -> dict[str, object]:
    engine = GlobalReducedSearch(
        config,
        handoff_root=handoff_root,
        node_ids=node_ids,
        lane_config=lane_config,
    )
    result = engine.run()
    if output is not None:
        engine.write_result(result, output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_wall_hours is not None and args.max_wall_hours <= 0:
        raise SystemExit("--max-wall-hours must be positive")
    config = GlobalReducedConfig(
        iterations=args.iterations,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        race_chunk=args.race_chunk,
        lane_exploration=args.lane_exploration,
        progress_interval=args.progress_interval,
        max_wall_seconds=(
            3600.0 * args.max_wall_hours
            if args.max_wall_hours is not None
            else None
        ),
        stop_class_count=args.stop_class_count,
    )
    lane_config = ReducedSearchConfig(
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.lane_max_depth,
        progress_interval=0,
    )
    output = args.output or (
        DEFAULT_RUNS_DIR
        / f"global_i{args.iterations}_s{args.seed}_reduced_racing_v2.json"
    )
    result = run_search(
        config=config,
        handoff_root=args.handoff_root,
        node_ids=args.node_id,
        lane_config=lane_config,
        output=output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "runtime": result["runtime"],
                "discovered_class_count": result["coverage"][
                    "discovered_class_count"
                ],
                "activated_lane_count": result["scheduler"][
                    "activated_lane_count"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
