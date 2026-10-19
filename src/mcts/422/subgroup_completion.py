from __future__ import annotations

"""Independent completion experiments for reduced Bell (4,2,2) subgroups.

Each selected complete node receives its own MCTS, discovery registry, reward
state, and full-space facet oracle.  Facet labels are read only to select an
evaluation scope and to measure post-search coverage; facet supports and H rows
are never exposed to the search.
"""

import argparse
import csv
import json
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from .atlas import DEFAULT_HANDOFF_ROOT
from .reduced_subgroup_search import (
    DEFAULT_RUNS_DIR as REDUCED_RUNS_DIR,
    ReducedSearchConfig,
    ReducedSubgroupMCTS,
    ReducedSubgroupProblem,
)


DEFAULT_RUNS_DIR = REDUCED_RUNS_DIR / "completion"
TYPICAL_NODE_IDS = (
    "BFS422-H-0000",  # baseline: 16 classes, one missing at i1000 in v1
    "BFS422-H-1171",  # largest checklist: 1729 classes
    "BFS422-H-0271",  # 1537 classes at intrinsic dimension 21
    "BFS422-H-0208",  # difficult C4 representative
    "BFS422-H-1193",  # large C6 representative
    "BFS422-H-1195",  # large S3 representative
    "BFS422-H-0605",  # dimension 24 with 215 classes
    "BFS422-H-0617",  # dimension 26 with 73 classes
    "BFS422-H-0078",  # maximum dimension 29 control
)


@dataclass(frozen=True)
class CompletionNode:
    node_id: str
    structure: str
    intrinsic_dimension: int
    reduced_vertex_count: int
    irredundant_vertex_count: int
    facet_row_count: int
    class_ids: frozenset[str]
    unique_class_ids: frozenset[str]

    @property
    def class_count(self) -> int:
        return len(self.class_ids)

    @property
    def unique_class_count(self) -> int:
        return len(self.unique_class_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "structure": self.structure,
            "intrinsic_dimension": self.intrinsic_dimension,
            "reduced_vertex_count": self.reduced_vertex_count,
            "irredundant_vertex_count": self.irredundant_vertex_count,
            "facet_row_count": self.facet_row_count,
            "class_count": self.class_count,
            "unique_class_count": self.unique_class_count,
        }


@dataclass(frozen=True)
class CompletionInventory:
    nodes: tuple[CompletionNode, ...]
    class_count: int
    facet_row_count: int

    @property
    def by_id(self) -> dict[str, CompletionNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def nonempty_nodes(self) -> tuple[CompletionNode, ...]:
        return tuple(node for node in self.nodes if node.class_ids)

    @property
    def required_nodes(self) -> tuple[CompletionNode, ...]:
        return tuple(node for node in self.nodes if node.unique_class_ids)

    def to_dict(self) -> dict[str, object]:
        structures: dict[str, int] = defaultdict(int)
        nonempty_structures: dict[str, int] = defaultdict(int)
        for node in self.nodes:
            structures[node.structure] += 1
            if node.class_ids:
                nonempty_structures[node.structure] += 1
        required_union = set().union(
            *(node.class_ids for node in self.required_nodes)
        )
        return {
            "node_count": len(self.nodes),
            "nonempty_node_count": len(self.nonempty_nodes),
            "zero_facet_node_count": len(self.nodes) - len(self.nonempty_nodes),
            "required_cover_node_count": len(self.required_nodes),
            "redundant_nonempty_node_count": (
                len(self.nonempty_nodes) - len(self.required_nodes)
            ),
            "facet_row_count": self.facet_row_count,
            "class_count": self.class_count,
            "required_cover_class_count": len(required_union),
            "nodes_by_structure": dict(structures),
            "nonempty_nodes_by_structure": dict(nonempty_structures),
        }


def _intrinsic_dimension(record_path: Path) -> int:
    with (record_path / "reduced_vertices.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        return 0
    return len(json.loads(row["intrinsic_reduced_vertex"]))


def load_inventory(
    handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
) -> CompletionInventory:
    root = Path(handoff_root).resolve() / "02_complete_nodes"
    with (root / "complete_nodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        metadata = list(csv.DictReader(handle))

    class_ids_by_node: dict[str, frozenset[str]] = {}
    facet_rows_by_node: dict[str, int] = {}
    class_owners: dict[str, set[str]] = defaultdict(set)
    dimensions: dict[str, int] = {}
    for row in metadata:
        node_id = row["node_id"]
        record_path = root / row["record_path"]
        with (record_path / "facets.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            facets = list(csv.DictReader(handle))
        class_ids = frozenset(
            item["facet_class_id"]
            for item in facets
            if item.get("facet_class_id")
        )
        class_ids_by_node[node_id] = class_ids
        facet_rows_by_node[node_id] = len(facets)
        dimensions[node_id] = _intrinsic_dimension(record_path)
        for class_id in class_ids:
            class_owners[class_id].add(node_id)

    nodes = tuple(
        CompletionNode(
            node_id=row["node_id"],
            structure=row["structure"],
            intrinsic_dimension=dimensions[row["node_id"]],
            reduced_vertex_count=int(row["reduced_vertex_count"]),
            irredundant_vertex_count=int(row["irredundant_vertex_count"]),
            facet_row_count=facet_rows_by_node[row["node_id"]],
            class_ids=class_ids_by_node[row["node_id"]],
            unique_class_ids=frozenset(
                class_id
                for class_id in class_ids_by_node[row["node_id"]]
                if len(class_owners[class_id]) == 1
            ),
        )
        for row in metadata
    )
    return CompletionInventory(
        nodes=nodes,
        class_count=len(class_owners),
        facet_row_count=sum(facet_rows_by_node.values()),
    )


def select_nodes(
    inventory: CompletionInventory,
    *,
    scope: str,
    node_ids: Iterable[str] = (),
) -> tuple[CompletionNode, ...]:
    by_id = inventory.by_id
    requested = tuple(dict.fromkeys(map(str, node_ids)))
    if requested:
        missing = [node_id for node_id in requested if node_id not in by_id]
        if missing:
            raise ValueError(f"unknown complete subgroup nodes: {missing}")
        return tuple(by_id[node_id] for node_id in requested)
    if scope == "typical":
        return tuple(by_id[node_id] for node_id in TYPICAL_NODE_IDS)
    if scope == "required":
        return inventory.required_nodes
    if scope == "nonempty":
        return inventory.nonempty_nodes
    if scope == "all":
        return inventory.nodes
    raise ValueError(f"unknown completion scope: {scope}")


def _result_path(
    output_dir: Path,
    node_id: str,
    iterations: int,
    seed: int,
) -> Path:
    return output_dir / (
        f"{node_id.lower()}_i{iterations}_s{seed}_independent_completion_v1.json"
    )


def _complete_saved_result(path: Path, node: CompletionNode) -> dict[str, object] | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    problem = result.get("problem", {})
    coverage = result.get("coverage", {})
    if problem.get("node_id") != node.node_id:
        return None
    if int(coverage.get("target_class_count", -1)) != node.class_count:
        return None
    return result


def run_completion_suite(
    *,
    handoff_root: Path | str,
    selected_nodes: Sequence[CompletionNode],
    config: ReducedSearchConfig,
    output_dir: Path,
    summary_output: Path,
    resume: bool = False,
    max_wall_seconds: float | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs: list[dict[str, object]] = []
    union_discovered: set[str] = set()
    union_targets: set[str] = set()
    termination_reason = "node_scope_complete"

    for index, node in enumerate(selected_nodes, start=1):
        if (
            max_wall_seconds is not None
            and time.perf_counter() - started >= max_wall_seconds
        ):
            termination_reason = "suite_wall_time_limit"
            break
        union_targets.update(node.class_ids)
        node_seed = config.seed ^ zlib.crc32(node.node_id.encode("ascii"))
        output = _result_path(
            output_dir,
            node.node_id,
            config.iterations,
            node_seed,
        )
        result = _complete_saved_result(output, node) if resume else None
        resumed = result is not None
        if result is None:
            problem = ReducedSubgroupProblem.load(handoff_root, node.node_id)
            node_config = replace(
                config,
                seed=node_seed,
                max_wall_seconds=None,
            )
            engine = ReducedSubgroupMCTS(
                problem,
                node_config,
                handoff_root=handoff_root,
            )
            result = engine.run(target_ids=tuple(sorted(node.class_ids)))
            engine.write_result(result, output)

        discovered = set(result["coverage"]["discovered_class_ids"])
        covered = discovered & node.class_ids
        unique_covered = discovered & node.unique_class_ids
        union_discovered.update(discovered)
        run = {
            **node.to_dict(),
            "index": index,
            "output": str(output),
            "resumed": resumed,
            "iterations_completed": result["runtime"]["iterations_completed"],
            "termination_reason": result["runtime"]["termination_reason"],
            "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            "discovered_class_count": len(discovered),
            "covered_target_count": len(covered),
            "missing_target_count": node.class_count - len(covered),
            "covered_unique_class_count": len(unique_covered),
            "missing_unique_class_count": (
                node.unique_class_count - len(unique_covered)
            ),
            "coverage_fraction": (
                len(covered) / node.class_count if node.class_count else 1.0
            ),
        }
        runs.append(run)
        summary = {
            "algorithm": "bell422_independent_reduced_subgroup_completion_v1",
            "prior_boundary": {
                "search_reads": [
                    "complete_nodes.csv: selected node metadata",
                    "selected nodes/*/reduced_vertices.csv",
                    "selected nodes/*/orbit_blocks.csv",
                ],
                "evaluation_only_reads": [
                    "selected nodes/*/facets.csv: facet_class_id column only"
                ],
                "never_used_by_search": [
                    "nodes/*/facets.csv: h_row",
                    "known reduced facet supports",
                    "target vertex selections",
                ],
                "cross_node_discovery_sharing": False,
                "cross_node_reward_sharing": False,
                "cross_node_tree_sharing": False,
            },
            "config": {
                "lane": asdict(config),
                "selected_node_count": len(selected_nodes),
                "resume": resume,
                "max_wall_seconds": max_wall_seconds,
            },
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "completed_node_count": len(runs),
                "termination_reason": termination_reason,
            },
            "coverage": {
                "selected_target_union_count": len(union_targets),
                "discovered_union_count": len(union_discovered),
                "covered_selected_union_count": len(
                    union_targets & union_discovered
                ),
                "missing_selected_union_count": len(
                    union_targets - union_discovered
                ),
                "fully_completed_node_count": sum(
                    run["missing_target_count"] == 0 for run in runs
                ),
            },
            "runs": runs,
        }
        ReducedSubgroupMCTS.write_result(summary, summary_output)
        print(
            f"[422-completion] {index}/{len(selected_nodes)} "
            f"node={node.node_id} target={len(covered)}/{node.class_count} "
            f"unique={len(unique_covered)}/{node.unique_class_count} "
            f"elapsed={run['elapsed_seconds']:.1f}s",
            flush=True,
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent reduced-subgroup completion experiments"
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument(
        "--scope",
        choices=("typical", "required", "nonempty", "all"),
        default="typical",
    )
    parser.add_argument("--node-id", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=4_222_027)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-wall-hours", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_nodes is not None and args.max_nodes <= 0:
        raise SystemExit("--max-nodes must be positive")
    if args.max_wall_hours is not None and args.max_wall_hours <= 0:
        raise SystemExit("--max-wall-hours must be positive")
    inventory = load_inventory(args.handoff_root)
    if args.inventory_only:
        print(json.dumps(inventory.to_dict(), indent=2))
        return 0
    selected = select_nodes(
        inventory,
        scope=args.scope,
        node_ids=args.node_id,
    )
    if args.max_nodes is not None:
        selected = selected[: args.max_nodes]
    config = ReducedSearchConfig(
        iterations=args.iterations,
        seed=args.seed,
        progress_interval=args.progress_interval,
    )
    summary_output = args.summary_output or (
        args.output_dir
        / f"{args.scope}_i{args.iterations}_s{args.seed}_completion_v1.json"
    )
    summary = run_completion_suite(
        handoff_root=args.handoff_root,
        selected_nodes=selected,
        config=config,
        output_dir=args.output_dir / f"{args.scope}_i{args.iterations}",
        summary_output=summary_output,
        resume=args.resume,
        max_wall_seconds=(
            3600.0 * args.max_wall_hours
            if args.max_wall_hours is not None
            else None
        ),
    )
    print(json.dumps({
        "summary_output": str(summary_output),
        "runtime": summary["runtime"],
        "coverage": summary["coverage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
