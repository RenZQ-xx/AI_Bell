from __future__ import annotations

"""Isolated, fixed-scope experiments for difficult reduced Bell nodes."""

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT
from .catalog_reproduction import audit_node, node_seed, save_json
from .geometry import FacetOracle
from .reduced_adaptive_exploration_search import (
    ReducedAdaptiveExplorationConfig, ReducedAdaptiveExplorationMCTS,
    ReducedAdaptiveProbeOnlyConfig, ReducedDiverseEntryOnlyConfig,
)
from .reduced_balanced_continuation_search import (
    ReducedBalancedContinuationConfig, ReducedBalancedContinuationMCTS,
)
from .reduced_continuation_search import (
    ReducedContinuationConfig, ReducedContinuationMCTS, audit_local_paths,
)
from .reduced_credit_search import ReducedCreditConfig, ReducedCreditMCTS
from .reduced_frontier_search import ReducedFrontierConfig, ReducedFrontierMCTS
from .reduced_frontier_schedule_search import ReducedFrontierScheduleConfig, ReducedFrontierScheduleMCTS
from .reduced_fresh_geometry_search import (
    ReducedFreshGeometryConfig, ReducedFreshGeometryMCTS, ReducedFreshProbeOnlyConfig, ReducedSparseQuotaOnlyConfig,
)
from .reduced_multi_exit_search import ReducedMultiExitConfig, ReducedMultiExitMCTS, audit_lower_face_exits
from .reduced_novelty_search import (
    CHANGED_REWARD_FIELDS, ReducedNoveltyConfig, ReducedNoveltyMCTS, audit_reduced_witnesses,
)
from .reduced_ridge_exit_search import ReducedRidgeExitConfig, ReducedRidgeExitMCTS, audit_ridge_exits
from .reduced_subgroup_search import (
    ReducedSearchConfig,
    ReducedSubgroupMCTS,
    ReducedSubgroupProblem,
)
from .subgroup_completion import load_inventory


DIFFICULT_NODES = {
    "BFS422-H-1171": "large_checklist_low_coverage",
    "BFS422-H-0208": "difficult_c4_large_checklist",
    "BFS422-H-1195": "difficult_s3_large_checklist",
    "BFS422-H-0587": "repeated_class_trap",
    "BFS422-H-0390": "near_completion_tail",
    "BFS422-H-0596": "no_successful_lift",
}
RUNS_DIR = Path(__file__).resolve().parent / "runs"
BASELINE_MANIFEST = RUNS_DIR / "catalog_reproduction" / "v1_s4222027" / "manifest.json"
VARIANTS = {
    "baseline": (ReducedSearchConfig, ReducedSubgroupMCTS),
    "credit": (ReducedCreditConfig, ReducedCreditMCTS),
    "continuation": (ReducedContinuationConfig, ReducedContinuationMCTS),
    "balanced_continuation": (ReducedBalancedContinuationConfig, ReducedBalancedContinuationMCTS),
    "ridge_exit": (ReducedRidgeExitConfig, ReducedRidgeExitMCTS),
    "reduced_novelty": (ReducedNoveltyConfig, ReducedNoveltyMCTS),
    "shared_frontier": (ReducedFrontierConfig, ReducedFrontierMCTS),
    "multi_exit": (ReducedMultiExitConfig, ReducedMultiExitMCTS),
    "frontier_schedule": (ReducedFrontierScheduleConfig, ReducedFrontierScheduleMCTS),
    "fresh_geometry": (ReducedFreshGeometryConfig, ReducedFreshGeometryMCTS),
    "fresh_geometry_probe_only": (ReducedFreshProbeOnlyConfig, ReducedFreshGeometryMCTS),
    "fresh_geometry_quota_only": (ReducedSparseQuotaOnlyConfig, ReducedFreshGeometryMCTS),
    "adaptive_geometry": (ReducedAdaptiveExplorationConfig, ReducedAdaptiveExplorationMCTS),
    "adaptive_probe_only": (ReducedAdaptiveProbeOnlyConfig, ReducedAdaptiveExplorationMCTS),
    "diverse_entry_only": (ReducedDiverseEntryOnlyConfig, ReducedAdaptiveExplorationMCTS),
}


def discovery_metrics(result: dict, window: int = 100) -> dict:
    events = result["coverage"]["discoveries"]
    iterations = int(result["runtime"]["iterations_completed"])
    removals = Counter(
        sum(action["kind"] == "remove" for action in event["path"])
        for event in events
    )
    hits = sum(result["coverage"]["class_frequencies"].values())
    geometry = result["geometry"]["statistics"]
    successful = geometry.get("successful_full_lifts", 0)
    failed = geometry.get("failed_full_lifts", 0)
    return {
        "growth_windows": [
            {
                "start_iteration_exclusive": start,
                "end_iteration_inclusive": min(start + window, iterations),
                "new_classes": sum(
                    start < event["iteration"] <= min(start + window, iterations)
                    for event in events
                ),
            }
            for start in range(0, iterations, window)
        ],
        "last_discovery_iteration": max(
            (event["iteration"] for event in events), default=0
        ),
        "iterations_since_last_discovery": iterations - max(
            (event["iteration"] for event in events), default=0
        ),
        "valid_class_hits": hits,
        "repeat_class_hit_fraction": (hits - len(events)) / hits if hits else None,
        "validation_miss_lift_success_fraction": (
            successful / (successful + failed) if successful + failed else None
        ),
        "direct_filler_discoveries": removals[0],
        "corrector_assisted_discoveries": len(events) - removals[0],
        "remove_count_histogram": dict(sorted(removals.items())),
        "path_action_count_histogram": dict(sorted(Counter(
            len(event["path"]) for event in events
        ).items())),
        "episode_discoveries": dict(Counter(
            event.get("episode_kind", "empty_root") for event in events
        )),
        "local_remove_count_histogram": dict(sorted(Counter(
            sum(action["kind"] == "remove" for action in event.get("local_path", event["path"]))
            for event in events
        ).items())),
        "failed_lift_origin_discoveries": sum(
            bool(event.get("continuation_origin") and not event["continuation_origin"]["lifted"])
            for event in events
        ),
        "terminal_completion_sources": dict(Counter(
            event["path"][-1].get("source", "unknown") if event["path"] else "unknown"
            for event in events
        )),
    }


def baseline_snapshot(manifest_path: Path, iterations: int) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lanes = {lane["node_id"]: lane for lane in manifest["nodes"]}
    snapshots = []
    for node_id, reason in DIFFICULT_NODES.items():
        lane = lanes[node_id]
        result = json.loads(
            (manifest_path.parent / lane["output"]).read_text(encoding="utf-8")
        )
        prefix = [
            event for event in result["coverage"]["discoveries"]
            if event["iteration"] <= iterations
        ]
        snapshots.append({
            "node_id": node_id,
            "selection_reason": reason,
            "structure": lane["structure"],
            "intrinsic_dimension": lane["intrinsic_dimension"],
            "irredundant_vertex_count": lane["irredundant_vertex_count"],
            "target_count": lane["class_count"],
            "saved_iterations": lane["iterations"],
            "saved_covered_count": lane["covered_target_count"],
            "saved_active_seconds": lane["active_seconds"],
            "same_budget_prefix_covered_count": len(prefix),
            "same_budget_prefix_class_ids": [event["class_id"] for event in prefix],
            "result": str(manifest_path.parent / lane["output"]),
        })
    return {
        "manifest": str(manifest_path),
        "source_total_iterations": manifest["runtime"]["iterations_completed"],
        "source_seed": manifest["config"]["seed"],
        "prefix_iteration_limit": iterations,
        "nodes": snapshots,
    }


def reduced_facet_metrics(result: dict, window: int = 100) -> dict:
    events = result["reduced_facets"]["discovery_events"]
    iterations = result["runtime"]["iterations_completed"]
    lifted = sum(event["lifted"] for event in events)
    return {
        "unique_count": len(events), "lifted_closure_count": lifted,
        "lifted_closure_fraction": lifted / len(events) if events else None,
        "full_classes_per_reduced_closure": len(result["coverage"]["discoveries"]) / len(events) if events else None,
        "growth_windows": [
            {"end_iteration_inclusive": min(start + window, iterations),
             "new_reduced_facets": sum(start < event["iteration"] <= min(start + window, iterations) for event in events)}
            for start in range(0, iterations, window)
        ],
    }


def run_benchmark(
    *,
    handoff_root: Path,
    output_dir: Path,
    config: ReducedSearchConfig,
    baseline_manifest: Path,
    variant: str = "baseline",
    reference_summary: Path | None = None,
    additional_reference_summaries: Sequence[Path] = (),
) -> dict:
    if variant not in VARIANTS:
        raise ValueError("unknown benchmark variant")
    config_class, engine_class = VARIANTS[variant]
    if type(config) is not config_class:
        raise ValueError("configuration does not match benchmark variant")
    reference = load_matched_reference(reference_summary, config) if reference_summary else None
    additional_references = [
        (path, load_matched_reference(path, config)) for path in additional_reference_summaries
    ]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = baseline_snapshot(baseline_manifest, config.iterations)
    inventory = load_inventory(handoff_root).by_id
    summary = {
        "algorithm": f"bell422_difficult_node_reduced_{variant}_benchmark_v1",
        "variant": variant,
        "search_core": engine_class.__name__,
        "config": asdict(config),
        "selected_node_ids": list(DIFFICULT_NODES),
        "selection_baseline": baseline,
        "matched_reference_summary": str(reference_summary) if reference_summary else None,
        "additional_reference_summaries": [str(path) for path in additional_reference_summaries],
        "prior_boundary": {
            "node_selection_uses_previous_performance": True,
            "unseen_witnesses_used_by_actions": False,
            "catalog_labels_used_for_stopping_and_evaluation_only": True,
            "independent_node_tree_reward_and_discovery": True,
            "search_initialization": "fresh_empty_root_and_discovery",
        },
        "state": "RUNNING",
        "current_node": None,
        "runs": [],
    }
    summary_path = output_dir / "summary.json"
    save_json(summary, summary_path)
    audit_oracle = FacetOracle(handoff_root=handoff_root)
    for node_id, reason in DIFFICULT_NODES.items():
        summary["current_node"] = node_id
        save_json(summary, summary_path)
        node = inventory[node_id]
        problem = ReducedSubgroupProblem.load(handoff_root, node_id)
        engine = engine_class(
            problem,
            replace(config, seed=node_seed(config.seed, node_id)),
            handoff_root=handoff_root,
        )
        result = engine.run(target_ids=tuple(sorted(node.class_ids)))
        # Record normals from actual search terminals, never from catalog witnesses.
        for event in result["coverage"]["discoveries"]:
            word = int(event["reduced_facet_word_hex"], 16)
            terminal = (engine.geometry.lift_cache[word] if isinstance(config, ReducedNoveltyConfig)
                        else engine.geometry.validate(word))
            event["canonical_h_row"] = list(terminal.full_result.canonical_h_row)
        result_path = output_dir / f"{node_id.lower()}.json"
        save_json(result, result_path)
        audit_started = time.perf_counter()
        audit = audit_node(
            problem, result["coverage"]["discoveries"], handoff_root, audit_oracle
        )
        if isinstance(config, ReducedContinuationConfig) and config.enable_local_continuation:
            local_audit = audit_local_paths(
                problem, result["coverage"]["discoveries"], config, handoff_root, audit_oracle
            )
            audit["local_path_audit"] = local_audit
            audit["failure_count"] += local_audit["failure_count"]
        if isinstance(config, ReducedRidgeExitConfig) and config.enable_deterministic_ridge_exits:
            ridge_events = (result["reduced_facets"]["witnesses"]
                            if isinstance(config, ReducedNoveltyConfig) else result["coverage"]["discoveries"])
            ridge_audit = audit_ridge_exits(
                problem, ridge_events, handoff_root, audit_oracle
            )
            audit["ridge_exit_audit"] = ridge_audit
            audit["failure_count"] += ridge_audit["failure_count"]
        if isinstance(config, ReducedNoveltyConfig):
            reduced_audit = audit_reduced_witnesses(
                problem, result["reduced_facets"]["witnesses"], config, handoff_root
            )
            audit["reduced_witness_audit"] = reduced_audit
            audit["failure_count"] += reduced_audit["failure_count"]
        if isinstance(config, ReducedMultiExitConfig) and config.enable_lower_face_exits:
            lower_audit = audit_lower_face_exits(
                problem, result["reduced_facets"]["witnesses"], handoff_root
            )
            audit["lower_face_exit_audit"] = lower_audit
            audit["failure_count"] += lower_audit["failure_count"]
        audit["elapsed_seconds"] = time.perf_counter() - audit_started
        audit_path = output_dir / f"{node_id.lower()}.audit.json"
        save_json(audit, audit_path)
        old_seed = next(item for item in baseline["nodes"] if item["node_id"] == node_id)
        run = {
            **node.to_dict(),
            "selection_reason": reason,
            "output": str(result_path),
            "audit_output": str(audit_path),
            "runtime": result["runtime"],
            "covered_target_count": result["coverage"]["covered_target_count"],
            "missing_target_count": len(result["coverage"]["missing_target_ids"]),
            "old_seed_same_budget_covered_count": old_seed["same_budget_prefix_covered_count"],
            "metrics": discovery_metrics(result),
            "audit": audit,
        }
        if "continuation" in result:
            run["continuation"] = result["continuation"]
            run["continuation_stats"] = {
                key: value for key, value in result["search_stats"].items()
                if "continuation" in key or "bridge" in key or "not_replayed" in key
            }
        if "entry_rollout" in result:
            run["entry_rollout"] = result["entry_rollout"]
            run["entry_diagnostics"] = result["entry_diagnostics"]
        if "ridge_exits" in result:
            run["ridge_exits"] = result["ridge_exits"]
        if "reduced_facets" in result:
            run["reduced_facet_metrics"] = reduced_facet_metrics(result)
        if "reduced_novelty" in result:
            run["reduced_novelty"] = result["reduced_novelty"]
        if "face_frontier" in result:
            run["face_frontier"] = result["face_frontier"]
        if "lower_face_exits" in result:
            run["lower_face_exits"] = result["lower_face_exits"]
        if "frontier_schedule" in result:
            run["frontier_schedule"] = {key: value for key, value in result["frontier_schedule"].items()
                                        if key not in ("sources", "cached_faces", "bank_history")}
        if "fresh_geometry" in result:
            run["fresh_geometry"] = {key: value for key, value in result["fresh_geometry"].items()
                                     if key != "probe_events"}
        if "adaptive_exploration" in result:
            run["adaptive_exploration"] = {key: value for key, value in result["adaptive_exploration"].items()
                                           if key not in ("sources", "entrance_events")}
        if reference is not None:
            run["comparison"] = canonical_comparison(reference, node_id, result)
        if additional_references:
            run["additional_comparisons"] = [
                {"reference_summary": str(path), **canonical_comparison(other, node_id, result)}
                for path, other in additional_references
            ]
        summary["runs"].append(run)
        summary["elapsed_seconds"] = time.perf_counter() - started
        print(
            f"[difficult-node] {node_id} "
            f"coverage={run['covered_target_count']}/{node.class_count} "
            f"iterations={result['runtime']['iterations_completed']} "
            f"search={result['runtime']['elapsed_seconds']:.1f}s "
            f"audit_failures={audit['failure_count']}",
            flush=True,
        )
        if audit["failure_count"]:
            summary["state"] = "AUDIT_FAILED"
            save_json(summary, summary_path)
            return summary
        save_json(summary, summary_path)
    summary["state"] = "COMPLETED"
    summary["current_node"] = None
    summary["elapsed_seconds"] = time.perf_counter() - started
    save_json(summary, summary_path)
    return summary


def load_matched_reference(path: Path, config: ReducedSearchConfig) -> dict:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference["state"] != "COMPLETED" or reference["selected_node_ids"] != list(DIFFICULT_NODES):
        raise ValueError("reference is not a complete benchmark of the same nodes")
    actual = asdict(config)
    old_reward = asdict(ReducedRidgeExitConfig())
    for key, value in reference["config"].items():
        if actual.get(key) == value:
            continue
        if isinstance(config, ReducedNoveltyConfig) and key in CHANGED_REWARD_FIELDS and value == old_reward[key]:
            continue
        raise ValueError("reference uses different seed, budget, or baseline parameters")
    return reference


def canonical_comparison(reference: dict, node_id: str, result: dict) -> dict:
    previous = next(item for item in reference["runs"] if item["node_id"] == node_id)
    previous_result = json.loads(Path(previous["output"]).read_text(encoding="utf-8"))
    old = {tuple(event["canonical_h_row"]) for event in previous_result["coverage"]["discoveries"]}
    new = {tuple(event["canonical_h_row"]) for event in result["coverage"]["discoveries"]}
    return {
        "reference_covered_count": previous["covered_target_count"],
        "covered_count_delta": result["coverage"]["covered_target_count"] - previous["covered_target_count"],
        "reference_search_seconds": previous["runtime"]["elapsed_seconds"],
        "overlap_canonical_count": len(old & new),
        "reference_only_canonical_h_rows": [list(row) for row in sorted(old - new)],
        "variant_only_canonical_h_rows": [list(row) for row in sorted(new - old)],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4222028)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--baseline-manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="baseline")
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--additional-reference-summary", type=Path, action="append", default=[])
    parser.add_argument("--no-reference", action="store_true", help="Run a fresh reference at a new budget")
    args = parser.parse_args(argv)
    config_class, _engine_class = VARIANTS[args.variant]
    config = config_class(
        iterations=args.iterations, seed=args.seed, progress_interval=args.progress_interval
    )
    output_dir = args.output_dir or (
        RUNS_DIR / "difficult_nodes" / f"{args.variant}_i{args.iterations}_s{args.seed}"
    )
    reference_summary = args.reference_summary
    if args.no_reference and (reference_summary or args.additional_reference_summary):
        parser.error("--no-reference cannot be combined with reference paths")
    if args.variant != "baseline" and reference_summary is None and not args.no_reference:
        reference_variant = ({"ridge_exit": "balanced_continuation", "reduced_novelty": "ridge_exit",
                              "shared_frontier": "reduced_novelty", "multi_exit": "shared_frontier",
                              "frontier_schedule": "multi_exit"}
                             .get(args.variant, "baseline"))
        if args.variant.startswith(("fresh_geometry", "adaptive_")) or args.variant == "diverse_entry_only":
            reference_variant = "multi_exit"
        reference_summary = (
            RUNS_DIR / "difficult_nodes"
            / f"{reference_variant}_i{args.iterations}_s{args.seed}" / "summary.json"
        )
    additional = list(args.additional_reference_summary)
    if args.variant == "ridge_exit" and not additional and not args.no_reference:
        additional.append(RUNS_DIR / "difficult_nodes" / f"baseline_i{args.iterations}_s{args.seed}" / "summary.json")
    if args.variant == "reduced_novelty" and not additional and not args.no_reference:
        additional.extend(RUNS_DIR / "difficult_nodes" / f"{variant}_i{args.iterations}_s{args.seed}" / "summary.json"
                          for variant in ("balanced_continuation", "baseline"))
    summary = run_benchmark(
        handoff_root=args.handoff_root.resolve(),
        output_dir=output_dir.resolve(),
        config=config,
        baseline_manifest=args.baseline_manifest.resolve(),
        variant=args.variant,
        reference_summary=reference_summary.resolve() if reference_summary else None,
        additional_reference_summaries=tuple(path.resolve() for path in additional),
    )
    return 0 if summary["state"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
