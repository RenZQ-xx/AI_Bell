from __future__ import annotations

"""Wall-time or fixed-MCTS-iteration adjacency tests on selected reduced nodes."""

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .atlas import DEFAULT_HANDOFF_ROOT
from .catalog_reproduction import audit_node, node_seed, save_json
from .geometry import FacetOracle
from .recursive_adjacency import audit_adjacency_proof
from .reduced_adjacency_search import (
    ReducedAdjacencyConfig, ReducedAdjacencyExplorer, ReducedAdjacencyMCTS,
)
from .reduced_multi_exit_search import ReducedMultiExitConfig, ReducedMultiExitMCTS
from .reduced_lazy_adjacency_search import ReducedLazyAdjacencyConfig, ReducedLazyAdjacencyMCTS
from .reduced_novelty_search import DeferredLiftGeometry
from .reduced_subgroup_search import ReducedSubgroupProblem, word_from_indices
from .subgroup_benchmark import load_target_class_ids


DEFAULT_NODES = ("BFS422-H-1171", "BFS422-H-0587", "BFS422-H-0596")
MODES = ("v6", "as_only", "v6_as")
AVAILABLE_MODES = (*MODES, "v6_as_lazy")


class TimedFacetOracle(FacetOracle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_seconds = 0.0

    def validate(self, word):
        started = time.perf_counter()
        try:
            return super().validate(word)
        finally:
            self.total_seconds += time.perf_counter() - started


def instrument_geometry(geometry):
    seconds, calls = Counter(), Counter()
    for name in ("sample_supporting_facets", "sample_ridges", "validate", "lift_taken_terminal"):
        original = getattr(geometry, name)
        def wrapper(*args, _original=original, _name=name, **kwargs):
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                seconds[_name] += time.perf_counter() - started
                calls[_name] += 1
        setattr(geometry, name, wrapper)
    return seconds, calls


def audit_reduced_paths(problem, witnesses, handoff_root):
    geometry = DeferredLiftGeometry(problem, handoff_root=handoff_root)
    failures = []
    for witness in witnesses:
        try:
            word = 0
            for action in witness["path"]:
                if action["kind"] == "remove":
                    if not geometry.validate(word).valid:
                        raise ValueError("remove parent is not a facet")
                    retained = word_from_indices(action["retained_vertices"])
                    if not retained or retained & ~word:
                        raise ValueError("invalid removal")
                    word = retained
                elif action["kind"] == "add":
                    word |= word_from_indices(action["add_vertices"])
                else:
                    raise ValueError("unknown action")
                rank = geometry.rank_oracle.rank(word)
                if rank > problem.facet_rank:
                    raise ValueError("support rank exceeds facet dimension")
                if rank == problem.facet_rank:
                    terminal = geometry.validate(word)
                    if not terminal.valid:
                        raise ValueError("non-supporting intermediate terminal")
                    word = terminal.closure_word
            if word != int(witness["reduced_facet_word_hex"], 16) or not geometry.validate(word).valid:
                raise ValueError("reduced terminal witness mismatch")
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"word": witness.get("reduced_facet_word_hex"), "error": str(error)})
    return {"paths_checked": len(witnesses), "failure_count": len(failures), "failures": failures}


def audit_run(problem, result, handoff_root):
    started = time.perf_counter()
    full = audit_node(problem, result["coverage"]["discoveries"], handoff_root,
                      FacetOracle(handoff_root=handoff_root))
    reduced = audit_reduced_paths(problem, result["reduced_facets"]["witnesses"], handoff_root)
    proofs = result.get("recursive_adjacency", {}).get("executed_proofs", ())
    failures = []
    for index, proof in enumerate(proofs):
        try:
            audit_adjacency_proof(problem.irredundant_points, proof)
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"index": index, "error": str(error)})
    return {"full_classes": full, "reduced_paths": reduced,
            "adjacency": {"proofs_checked": len(proofs), "failure_count": len(failures), "failures": failures},
            "failure_count": full["failure_count"] + reduced["failure_count"] + len(failures),
            "elapsed_seconds": time.perf_counter()-started}


def validate_budget(seconds, iterations):
    if (seconds is None) == (iterations is None):
        raise ValueError("choose exactly one wall-time or iteration budget")
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("search wall budget must be finite and positive")
    if iterations is not None and (not isinstance(iterations, int) or iterations <= 0):
        raise ValueError("iteration budget must be a positive integer")


def search_loop(engine, mode, *, seconds=None, iterations=None):
    validate_budget(seconds, iterations)
    if iterations is not None and mode == "as_only":
        raise ValueError("AS-only native iterations are not MCTS iterations")
    engine.started_at = time.perf_counter()
    deadline = engine.started_at + seconds if seconds is not None else math.inf
    if hasattr(engine, "search_deadline"):
        engine.search_deadline = deadline
    engine.termination_reason = "iteration_budget" if iterations is not None else "wall_time_limit"
    timeline, iteration_trace = [], []
    iteration = 0
    next_progress_seconds = 0.0
    while (iterations is None or iteration < iterations) and time.perf_counter() < deadline:
        iteration += 1
        before_classes = len(engine.discovery_events)
        before_reduced = len(engine.reduced_events if mode == "as_only" else engine.reduced_discovery_events)
        engine.run_iteration(iteration)
        elapsed = time.perf_counter() - engine.started_at
        reduced_events = engine.reduced_events if mode == "as_only" else engine.reduced_discovery_events
        for event in engine.discovery_events[before_classes:]:
            event.setdefault("elapsed_seconds", elapsed)
        for event in reduced_events[before_reduced:]:
            event.setdefault("elapsed_seconds", elapsed)
        snapshot = {"iteration": iteration, "elapsed_seconds": elapsed,
                    "classes": len(engine.discovery_events), "reduced_facets": len(reduced_events)}
        if iterations is not None:
            iteration_trace.append({**snapshot, "actual_terminal_arrivals": sum(
                getattr(engine, "reduced_facet_frequencies", {}).values())})
        if (iteration == 1 or (iterations is not None and (iteration % 50 == 0 or iteration == iterations))
                or (iterations is None and elapsed >= next_progress_seconds)):
            timeline.append(snapshot)
            next_progress_seconds = (math.floor(elapsed / 10) + 1) * 10
            print(json.dumps({"mode": mode, **snapshot}), flush=True)
    return {"iterations": iteration, "search_seconds": time.perf_counter()-engine.started_at,
            "timeline": timeline, "iteration_trace": iteration_trace}


def run_case(*, node_id, seed, mode, seconds, cutoff, macro_seconds, handoff_root, output_dir,
             iterations=None):
    validate_budget(seconds, iterations)
    problem = ReducedSubgroupProblem.load(handoff_root, node_id)
    targets = load_target_class_ids(handoff_root, node_id)
    effective_seed = node_seed(seed, node_id)
    common = {"iterations": iterations or 10_000_000, "seed": effective_seed, "progress_interval": 0}
    full_oracle = TimedFacetOracle(handoff_root=handoff_root)
    if mode == "v6":
        config = ReducedMultiExitConfig(**common)
        engine = ReducedMultiExitMCTS(problem, config, handoff_root=handoff_root, full_oracle=full_oracle)
    elif mode == "v6_as_lazy":
        config = ReducedLazyAdjacencyConfig(**common, adjacency_cutoff=cutoff,
                                            adjacency_macro_seconds=macro_seconds)
        engine = ReducedLazyAdjacencyMCTS(problem, config, handoff_root=handoff_root, full_oracle=full_oracle)
    else:
        config = ReducedAdjacencyConfig(**common, adjacency_cutoff=cutoff,
                                        adjacency_macro_seconds=macro_seconds)
        cls = ReducedAdjacencyExplorer if mode == "as_only" else ReducedAdjacencyMCTS
        engine = cls(problem, config, handoff_root=handoff_root, full_oracle=full_oracle)
    geometry_seconds, geometry_calls = instrument_geometry(engine.geometry)
    print(json.dumps({"start": {"node": node_id, "seed": seed, "mode": mode,
                                "seconds": seconds, "iterations": iterations}}), flush=True)
    progress = search_loop(engine, mode, seconds=seconds, iterations=iterations)
    search_seconds, iteration = progress["search_seconds"], progress["iterations"]
    result = engine.result(target_ids=targets)
    for event in result["coverage"]["discoveries"]:
        terminal = engine.geometry.lift_cache[int(event["reduced_facet_word_hex"], 16)]
        event["canonical_h_row"] = list(terminal.full_result.canonical_h_row)
    result["wall_benchmark"] = {
        "mode": mode, "base_seed": seed, "effective_seed": effective_seed,
        "budget_kind": "mcts_iterations" if iterations is not None else "wall_time",
        "budget_iterations": iterations,
        "budget_seconds": seconds, "actual_search_seconds": search_seconds,
        "iteration_boundary_overshoot_seconds": max(0, search_seconds-seconds) if seconds is not None else None,
        "target_coverage_stopping_enabled": False,
        "initialization_and_audit_in_search_budget": False,
        "geometry_seconds_inclusive": dict(geometry_seconds), "geometry_calls": dict(geometry_calls),
        "full_validation_and_canonicalization_seconds_inclusive": full_oracle.total_seconds,
        "timeline": progress["timeline"], "iteration_trace": progress["iteration_trace"],
        "work_counts": {
            "actual_reduced_terminal_arrivals": sum(getattr(engine, "reduced_facet_frequencies", {}).values()),
            "actual_action_steps": sum(getattr(engine, "action_frequencies", {}).values()),
            "source_frontier_requests": sum(getattr(engine, "frontier_requests", {}).values()),
            "support_lp_attempts": engine.geometry.stats.get("support_lp_attempts", 0),
            "ridge_lp_attempts": engine.geometry.stats.get("ridge_lp_attempts", 0),
        },
    }
    name = f"{node_id.lower()}_{mode}_s{seed}"
    result_path = output_dir / f"{name}.json"
    save_json(result, result_path)
    audit = audit_run(problem, result, handoff_root)
    save_json(audit, output_dir / f"{name}.audit.json")
    events = result["coverage"]["discoveries"]
    rows = sorted({tuple(event["canonical_h_row"]) for event in events})
    record = {"node_id": node_id, "mode": mode, "seed": seed, "config": asdict(config),
              "result": str(result_path.resolve()), "iterations": iteration,
              "search_seconds": search_seconds, "audit_seconds": audit["elapsed_seconds"],
              "canonical_classes": len(rows), "target_classes": len(targets),
              "matched_classes": sum(event["catalog_index"] is not None for event in events),
              "unmatched_classes": sum(event["catalog_index"] is None for event in events),
              "target_covered": result["coverage"]["covered_target_count"],
              "reduced_facets": len(result["reduced_facets"]["discovery_events"]),
              "canonical_h_rows": [list(row) for row in rows],
              "audit_failures": audit["failure_count"],
              "wall_benchmark": result["wall_benchmark"],
              "adjacency_geometry": result.get("recursive_adjacency", {}).get("geometry")}
    print(json.dumps({k: v for k, v in record.items() if k in
                      ("node_id", "mode", "seed", "iterations", "canonical_classes", "reduced_facets", "audit_failures")}), flush=True)
    if audit["failure_count"]:
        raise ValueError(f"independent audit failed: {name}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", default=list(DEFAULT_NODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[4222028, 4222029])
    parser.add_argument("--modes", nargs="+", choices=AVAILABLE_MODES, default=list(MODES))
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--seconds", type=float)
    budget.add_argument("--iterations", type=int)
    parser.add_argument("--cutoff", type=int, default=16)
    parser.add_argument("--macro-seconds", type=float, default=1.0)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds is None and args.iterations is None:
        args.seconds = 30.0
    validate_budget(args.seconds, args.iterations)
    if args.iterations is not None and "as_only" in args.modes:
        raise ValueError("fixed MCTS budgets require --modes v6 v6_as; AS-only units differ")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite benchmark directory")
    args.output_dir.mkdir(parents=True)
    summary = {"state": "RUNNING", "config": {"nodes": args.nodes, "seeds": args.seeds,
                "modes": args.modes, "seconds_per_case": args.seconds, "cutoff": args.cutoff,
                "macro_seconds": args.macro_seconds, "iterations_per_case": args.iterations,
                "budget_kind": "mcts_iterations" if args.iterations is not None else "wall_time"}, "runs": [],
               "scope": "fresh isolated reduced-node searches; no global search or target witnesses"}
    started = time.perf_counter()
    path = args.output_dir / "summary.json"
    save_json(summary, path)
    try:
        for seed_index, seed in enumerate(args.seeds):
            for node_index, node_id in enumerate(args.nodes):
                offset = (seed_index + node_index) % len(args.modes)
                order = args.modes[offset:] + args.modes[:offset]
                for mode in order:
                    summary["current"] = {"node_id": node_id, "seed": seed, "mode": mode}
                    save_json(summary, path)
                    summary["runs"].append(run_case(
                        node_id=node_id, seed=seed, mode=mode, seconds=args.seconds,
                        cutoff=args.cutoff, macro_seconds=args.macro_seconds,
                        handoff_root=args.handoff_root, output_dir=args.output_dir,
                        iterations=args.iterations))
                    save_json(summary, path)
        summary["comparisons"] = []
        for seed in args.seeds:
            for node_id in args.nodes:
                cases = {run["mode"]: {tuple(row) for row in run["canonical_h_rows"]}
                         for run in summary["runs"] if run["seed"] == seed and run["node_id"] == node_id}
                if "v6" in cases:
                    for mode, rows in cases.items():
                        summary["comparisons"].append({"node_id": node_id, "seed": seed, "mode": mode,
                            "intersection_with_v6": len(rows & cases["v6"]),
                            "v6_only": len(cases["v6"] - rows), "variant_only": len(rows - cases["v6"])})
        summary["state"] = "COMPLETED"
    except Exception as error:
        summary.update(state="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["total_wall_seconds"] = time.perf_counter() - started
        save_json(summary, path)


if __name__ == "__main__":
    main()
