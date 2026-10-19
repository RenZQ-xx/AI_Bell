from __future__ import annotations

"""Set-accurate post-run analysis of the isolated equal-wall adjacency tests."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .catalog_reproduction import save_json


def reference_counts(rows, baseline):
    if baseline is None:
        return {key: None for key in ("intersection_with_v6", "v6_only", "variant_only")}
    return {"intersection_with_v6": len(rows & baseline),
            "v6_only": len(baseline-rows), "variant_only": len(rows-baseline)}


def terminal_as_proof(event, proofs):
    removal = next((action for action in reversed(event["path"]) if action["kind"] == "remove"), None)
    if removal is None or removal["source"] != "recursive_as_ridge":
        return None
    key = (int(removal["sampled_facet_word_hex"], 16),
           sum(1 << int(v) for v in removal["retained_vertices"]),
           int(event["reduced_facet_word_hex"], 16))
    return proofs.get(key)


def analyze(summary_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["state"] != "COMPLETED":
        raise ValueError("benchmark and independent audits must finish first")
    cases, sets = [], {}
    for run in summary["runs"]:
        result = json.loads(Path(run["result"]).read_text(encoding="utf-8"))
        mode = run["mode"]
        key = run["node_id"], run["seed"], mode
        rows = {tuple(row) for row in run["canonical_h_rows"]}
        sets[key] = rows
        seconds = run["search_seconds"]
        events = result["coverage"]["discoveries"]
        adjacency = result.get("recursive_adjacency", {})
        diagnostics = adjacency.get("geometry", {})
        macro_seconds = diagnostics.get("seconds", {}).get("macro_inclusive", 0.0)
        actual_as = sum(any(action["source"] == "recursive_as_ridge" for action in event["path"])
                        for event in events)
        last = max((event["elapsed_seconds"] for event in events), default=0.0)
        covered_ids = set(result["coverage"]["covered_target_ids"])
        complete_at = max((event["elapsed_seconds"] for event in events
                           if event["class_id"] in covered_ids), default=0.0)
        local_removals = [sum(action["kind"] == "remove" for action in
                              event.get("local_path", event["path"])) for event in events]
        proofs = {(int(proof["chain_word_hex"][1], 16), int(proof["retained_word_hex"], 16),
                   int(proof["destination_word_hex"], 16)): proof
                  for proof in adjacency.get("executed_proofs", ())}
        trace = {point["iteration"]: point for point in run["wall_benchmark"].get("iteration_trace", ())}
        milestones = list(range(50, run["iterations"]+1, 50)) if trace else []
        cases.append({
            "node_id": run["node_id"], "seed": run["seed"], "mode": mode,
            "canonical_classes": len(rows), "target_classes": run["target_classes"],
            "target_covered": run["target_covered"], "reduced_facets": run["reduced_facets"],
            "search_seconds": seconds, "audit_seconds": run["audit_seconds"],
            "iterations": run["iterations"], "audit_failures": run["audit_failures"],
            "budget_kind": run["wall_benchmark"].get("budget_kind", "wall_time"),
            "work_counts": run["wall_benchmark"].get("work_counts", {}),
            "iteration_growth": [trace[iteration] for iteration in milestones],
            "last_class_iteration": max((event["iteration"] for event in events), default=0),
            "classes_per_second": len(rows)/seconds,
            "reduced_facets_per_second": run["reduced_facets"]/seconds,
            "last_class_seconds": last, "trailing_no_class_seconds": seconds-last,
            "catalog_complete_at_seconds": complete_at if run["target_covered"] == run["target_classes"] else None,
            "class_discovery_curve_10_seconds": [
                {"seconds": t, "classes": sum(event["elapsed_seconds"] <= t for event in events)}
                for t in (10, 20, 30)],
            "as_assisted_class_paths": actual_as,
            "local_remove_histogram": dict(Counter(local_removals)),
            "as_macro_seconds_inclusive": macro_seconds,
            "as_macro_fraction_of_search": macro_seconds/seconds,
            "as_geometry": diagnostics,
            "as_actual_terminal_arrivals": adjacency.get("actual_terminal_arrivals"),
            "as_new_reduced_terminal_arrivals": adjacency.get("new_reduced_terminal_arrivals"),
            "executed_as_proof_count": len(adjacency.get("executed_proofs", ())),
            "class_paths": [{"class_id": event["class_id"], "iteration": event["iteration"],
                             "elapsed_seconds": event["elapsed_seconds"],
                             "reduced_facet_word_hex": event["reduced_facet_word_hex"],
                             "remove_count": sum(action["kind"] == "remove" for action in event["path"]),
                             "last_as_chain_depth": (
                                 len(proof["chain_word_hex"])-2
                                 if (proof := terminal_as_proof(event, proofs)) is not None else None)}
                            for event in events],
            "full_validation_seconds_inclusive": run["wall_benchmark"][
                "full_validation_and_canonicalization_seconds_inclusive"],
        })
    aggregates = []
    for seed in summary["config"]["seeds"]:
        for mode in summary["config"]["modes"]:
            selected = [case for case in cases if case["seed"] == seed and case["mode"] == mode]
            rows = set().union(*(sets[(case["node_id"], seed, mode)] for case in selected))
            references = [value for (node, s, m), value in sets.items() if s == seed and m == "v6"]
            baseline = set().union(*references) if references else None
            seconds = sum(case["search_seconds"] for case in selected)
            aggregates.append({"seed": seed, "mode": mode, "canonical_union": len(rows),
                               "per_node_class_sum": sum(case["canonical_classes"] for case in selected),
                               "reduced_facet_sum": sum(case["reduced_facets"] for case in selected),
                               "search_seconds": seconds, "union_classes_per_search_second": len(rows)/seconds,
                               **reference_counts(rows, baseline)})
    all_seeds = []
    for mode in summary["config"]["modes"]:
        rows = set().union(*(value for (node, seed, m), value in sets.items() if m == mode))
        references = [value for (node, seed, m), value in sets.items() if m == "v6"]
        baseline = set().union(*references) if references else None
        all_seeds.append({"mode": mode, "canonical_union_across_nodes_and_seeds": len(rows),
                          **reference_counts(rows, baseline)})
    return {"source": str(summary_path.resolve()), "state": summary["state"],
            "benchmark_wall_seconds": summary["total_wall_seconds"],
            "search_seconds": sum(case["search_seconds"] for case in cases),
            "audit_seconds": sum(case["audit_seconds"] for case in cases),
            "audit_failures": sum(case["audit_failures"] for case in cases),
            "cases": cases, "per_seed": aggregates, "all_seeds": all_seeds,
            "budget_kind": summary["config"].get("budget_kind", "wall_time"),
            "causal_boundary": ("AS-only changes entrance allocation and removes MCTS; recursion alone is not isolated"
                                if "as_only" in summary["config"]["modes"] else
                                "same outer MCTS iteration count is not the same amount of geometric computation"
                                if summary["config"].get("budget_kind") == "mcts_iterations" else
                                "search wall budget excludes initialization and audit; iteration work may differ"),
            "catalog_scope": "partial subgroup-derived checklist, not all Bell 4-2-2 classes"}


def compare_mcts_prefix(summary_path: Path, iterations: int):
    if iterations <= 0:
        raise ValueError("prefix iterations must be positive")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["state"] != "COMPLETED":
        raise ValueError("benchmark and independent audits must finish first")
    indexed = {(run["node_id"], run["seed"], run["mode"]): run for run in summary["runs"]}
    cases = []
    for seed in summary["config"]["seeds"]:
        for node in summary["config"]["nodes"]:
            modes = {}
            for mode in ("v6", "v6_as"):
                run = indexed[(node, seed, mode)]
                if run["iterations"] < iterations:
                    raise ValueError(f"insufficient saved prefix: {node}/{seed}/{mode}")
                result = json.loads(Path(run["result"]).read_text(encoding="utf-8"))
                rows = {tuple(event["canonical_h_row"]) for event in result["coverage"]["discoveries"]
                        if event["iteration"] <= iterations}
                reduced = {event["reduced_facet_word_hex"] for event in
                           result["reduced_facets"]["discovery_events"] if event["iteration"] <= iterations}
                modes[mode] = {"canonical_classes": len(rows), "reduced_facets": len(reduced),
                               "canonical_h_rows": [list(row) for row in sorted(rows)],
                               "saved_iterations": run["iterations"]}
            old = {tuple(row) for row in modes["v6"]["canonical_h_rows"]}
            new = {tuple(row) for row in modes["v6_as"]["canonical_h_rows"]}
            cases.append({"node_id": node, "seed": seed, "iterations": iterations,
                          **modes, "intersection": len(old & new),
                          "v6_only": len(old-new), "v6_as_only": len(new-old)})
    differences = [case["v6_as"]["canonical_classes"]-case["v6"]["canonical_classes"] for case in cases]
    return {"source": str(summary_path.resolve()), "mcts_iteration_prefix": iterations,
            "source_is_saved_trajectory_prefix_not_new_search": True,
            "same_iterations_mean_equal_compute": False,
            "prefix_elapsed_times_available": False,
            "excluded_modes": {"as_only": "one pending outlet or one source expansion is not a full MCTS iteration"},
            "comparison_scope": "same outer MCTS iteration count; AS adds geometric work and retains its per-macro time guard",
            "cases": cases, "higher_class_count": sum(value > 0 for value in differences),
            "equal_class_count": sum(value == 0 for value in differences),
            "lower_class_count": sum(value < 0 for value in differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcts-prefix", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite analysis")
    result = (compare_mcts_prefix(args.summary, args.mcts_prefix)
              if args.mcts_prefix is not None else analyze(args.summary))
    save_json(result, args.output)
    if args.mcts_prefix is not None:
        printed = {key: value for key, value in result.items() if key != "cases"}
        printed["cases"] = [{"node_id": case["node_id"], "seed": case["seed"],
                             **{mode: {key: case[mode][key] for key in ("canonical_classes", "reduced_facets")}
                                for mode in ("v6", "v6_as")}} for case in result["cases"]]
    else:
        printed = {key: result[key] for key in ("search_seconds", "audit_seconds", "audit_failures", "per_seed", "all_seeds")}
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
