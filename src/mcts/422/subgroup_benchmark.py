from __future__ import annotations

"""Full-space fixed-partition control for the Bell 4-2-2 v25 search.

This legacy control still uses the original 256 vertices and affine rank 79.
Use ``reduced_subgroup_search`` for the actual intrinsic reduced-polytope
subproblem. Facet labels remain evaluation-only in both experiments.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT, HandoffPatternAtlas
from .model import GameNode, MacroAction, SearchState
from .subset_probe import PROFILE_OVERRIDES, build_config
from .v25_search import Bell422V25Search, V25SearchConfig


DEFAULT_NODE_ID = "BFS422-H-0000"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs" / "subgroup_benchmarks"


def load_target_class_ids(
    handoff_root: Path | str,
    node_id: str,
) -> tuple[str, ...]:
    """Load labels for evaluation without exposing facet geometry to search."""
    path = (
        Path(handoff_root).resolve()
        / "02_complete_nodes"
        / "nodes"
        / node_id
        / "facets.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing subgroup facet checklist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        class_ids = {
            row["facet_class_id"]
            for row in csv.DictReader(handle)
            if row.get("facet_class_id")
        }
    return tuple(sorted(class_ids))


class FixedSubgroupSearch(Bell422V25Search):
    """Run the v25 tree with every geometric action tied to one partition."""

    def __init__(
        self,
        config: V25SearchConfig,
        *,
        fixed_pattern_id: str,
        handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    ) -> None:
        atlas = HandoffPatternAtlas(
            handoff_root,
            allowed_pattern_ids={fixed_pattern_id},
        )
        if fixed_pattern_id not in atlas.by_id:
            raise ValueError(f"unknown subgroup node: {fixed_pattern_id}")
        super().__init__(config, handoff_root=handoff_root, atlas=atlas)
        self.fixed_pattern_id = fixed_pattern_id
        self.pattern_order = (fixed_pattern_id,)
        root_state = SearchState(0, "filler", 0, fixed_pattern_id)
        self.root = GameNode(root_state, self.rank_oracle.rank(0))
        self.nodes = {root_state.key: self.root}

    def _candidate_pattern_ids(self, node: GameNode) -> tuple[str, ...]:
        del node
        return (self.fixed_pattern_id,)

    def _rollout_pattern_ids(self, node: GameNode, policy: str) -> list[str]:
        del node, policy
        return [self.fixed_pattern_id]

    def _actions_for_pattern(
        self,
        node: GameNode,
        pattern_id: str,
        lane: str,
    ) -> list[MacroAction]:
        if pattern_id != self.fixed_pattern_id:
            return []
        return [
            action
            for action in super()._actions_for_pattern(node, pattern_id, lane)
            if action.pattern_id == self.fixed_pattern_id
        ]


def run_benchmark(
    *,
    handoff_root: Path,
    node_id: str,
    config: V25SearchConfig,
    output: Path,
) -> dict[str, object]:
    target_ids = load_target_class_ids(handoff_root, node_id)
    engine = FixedSubgroupSearch(
        config,
        fixed_pattern_id=node_id,
        handoff_root=handoff_root,
    )
    target_set = set(target_ids)
    target_events: list[dict[str, object]] = []
    covered: set[str] = set()
    event_cursor = 0
    engine.started_at = time.perf_counter()
    engine.termination_reason = "iteration_budget"

    for iteration in range(1, config.iterations + 1):
        engine.run_iteration(iteration)
        for event in engine.discovery_events[event_cursor:]:
            class_id = str(event["class_id"])
            if class_id in target_set and class_id not in covered:
                covered.add(class_id)
                target_events.append(
                    {
                        "class_id": class_id,
                        "iteration": int(event["iteration"]),
                        "depth": int(event["depth"]),
                        "source": event["source"],
                    }
                )
        event_cursor = len(engine.discovery_events)
        if config.progress_interval and iteration % config.progress_interval == 0:
            print(
                f"[422-subgroup] node={node_id} iteration={iteration} "
                f"target={len(covered)}/{len(target_ids)} "
                f"all_classes={len(engine.discovered)} "
                f"nodes={len(engine.nodes)} "
                f"elapsed={time.perf_counter() - engine.started_at:.1f}s",
                flush=True,
            )
        if len(covered) == len(target_ids):
            engine.termination_reason = "target_coverage"
            break
        if (
            config.max_wall_seconds is not None
            and time.perf_counter() - engine.started_at >= config.max_wall_seconds
        ):
            engine.termination_reason = "wall_time_limit"
            break

    result = engine.result()
    result["algorithm"] = "bell422_full_space_fixed_partition_control_v25"
    result["prior_boundary"] = {
        "search_reads": [
            "complete_nodes.csv: selected node metadata",
            f"nodes/{node_id}/orbit_blocks.csv",
        ],
        "terminal_only_reads": ["global_facet_classes.csv"],
        "evaluation_only_reads": [
            f"nodes/{node_id}/facets.csv: facet_class_id column only"
        ],
        "never_used_by_search": [
            f"nodes/{node_id}/facets.csv: h_row",
            "target tight supports",
            "target block selections",
        ],
        "initial_discovered_classes": [],
        "unseen_class_examples_used": False,
    }
    result["subgroup_benchmark"] = {
        "node_id": node_id,
        "structure": engine.atlas.pattern(node_id).structure,
        "orbit_block_count": engine.atlas.pattern(node_id).block_count,
        "target_class_count": len(target_ids),
        "target_class_ids": list(target_ids),
        "covered_target_count": len(covered),
        "covered_target_ids": [
            class_id for class_id in target_ids if class_id in covered
        ],
        "missing_target_ids": [
            class_id for class_id in target_ids if class_id not in covered
        ],
        "target_discoveries": target_events,
        "non_target_discovered_ids": [
            class_id for class_id in engine.discovered if class_id not in target_set
        ],
        "coverage_fraction": len(covered) / len(target_ids) if target_ids else 1.0,
        "fixed_partition_enforced": True,
        "target_labels_affect_search": False,
    }
    engine._write_result(result, output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-space fixed-partition Bell 4-2-2 v25 control"
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--node-id", default=DEFAULT_NODE_ID)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4_222_025)
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_OVERRIDES), default="default"
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--max-wall-minutes", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config(
        iterations=args.iterations,
        seed=args.seed,
        profile=args.profile,
        progress_interval=args.progress_interval,
    )
    if args.max_wall_minutes is not None:
        if args.max_wall_minutes <= 0:
            raise SystemExit("--max-wall-minutes must be positive")
        config = replace(config, max_wall_seconds=60.0 * args.max_wall_minutes)
    output = args.output or (
        DEFAULT_RUNS_DIR
        / f"{args.node_id.lower()}_i{args.iterations}_{args.profile}_s{args.seed}.json"
    )
    result = run_benchmark(
        handoff_root=args.handoff_root,
        node_id=args.node_id,
        config=config,
        output=output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "node_id": args.node_id,
                "profile": args.profile,
                "effective_config": asdict(config),
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "iterations_completed": result["runtime"]["iterations_completed"],
                "all_discovered_class_ids": result["coverage"][
                    "discovered_class_ids"
                ],
                "covered_target_ids": result["subgroup_benchmark"][
                    "covered_target_ids"
                ],
                "missing_target_ids": result["subgroup_benchmark"][
                    "missing_target_ids"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
