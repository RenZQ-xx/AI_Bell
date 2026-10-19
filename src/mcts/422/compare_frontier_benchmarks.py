from __future__ import annotations

"""Compare matched reduced-search variants without float canonical H-rows."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .catalog_reproduction import save_json


POLICY_SWITCHES = {"enable_frontier_anchor_schedule", "enable_corrector_outlet_quota",
                   "enable_fresh_geometry_probes", "enable_adaptive_probe_cooldown",
                   "enable_diverse_continuation"}


def recorded_frontier_gap(source, retained, destination, visited_words, schedule):
    if not schedule:
        return None
    if source is None:
        return {"category": "direct_root_path"}
    if source not in visited_words:
        return {"category": "source_not_visited"}
    if retained is None:
        return {"category": "reference_retained_not_recorded"}
    key = (f"0x{source:x}", f"0x{retained:x}")
    bank = schedule["bank_history"].get(key)
    face = schedule["cached_faces"].get(key)
    outlet = next((item for item in face["outlets"]
                   if int(item["word_hex"], 16) == destination), None) if face else None
    attempts = (bank or face or {}).get("actual_removals", 0)
    if bank is None and face is None:
        category = "retained_not_in_recorded_bank_or_cache"
    elif not attempts:
        category = ("retained_proposed_not_taken" if bank and "first_banked_iteration" in bank
                    and bank["first_banked_iteration"] is None else "retained_banked_not_taken")
    elif face is None:
        category = "retained_taken_without_complete_indexed_cache"
    elif outlet is None:
        category = "reference_destination_not_in_cached_outlets"
    elif not outlet["actual_traversals"]:
        category = "cached_outlet_not_taken"
    else:
        category = "route_taken_without_same_full_class"
    return {
        "category": category,
        "first_banked_iteration": bank.get("first_banked_iteration", bank.get("first_generated_iteration")) if bank else None,
        "first_taken_iteration": bank.get("first_attempted_iteration") if bank else None,
        "actual_removals": attempts,
        "complete_indexed_cache": face is not None,
        "reference_destination_is_cached": outlet is not None,
        "reference_destination_actual_traversals": outlet["actual_traversals"] if outlet else None,
    }


def summarize(paths):
    classes, catalog, reduced = set(), set(), set()
    totals = Counter()
    frontier_totals = Counter()
    lower_totals = Counter()
    schedule_totals = Counter()
    probe_totals = Counter()
    adaptive_totals = Counter()
    probe_modes = {}
    nodes = {}
    windows = {}
    sources = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["state"] != "COMPLETED":
            raise ValueError(f"incomplete benchmark: {path}")
        sources.append(str(path.resolve()))
        totals["suite_wall_seconds"] += summary["elapsed_seconds"]
        for run in summary["runs"]:
            if run["audit"]["failure_count"]:
                raise ValueError("cannot compare a failed audit")
            result = json.loads(Path(run["output"]).read_text(encoding="utf-8"))
            node_id = run["node_id"]
            row = nodes.setdefault(node_id, {"classes": set(), "reduced": set(),
                                             "search_seconds": 0.0, "lp_attempts": 0,
                                             "seed_runs": []})
            events = result["coverage"]["discoveries"]
            reduced_events = result["reduced_facets"]["discovery_events"]
            normals = {tuple(event["canonical_h_row"]) for event in events}
            if any(not isinstance(value, int) for normal in normals for value in normal):
                raise ValueError("canonical normals must retain exact integers")
            classes.update(normals)
            catalog.update(tuple(event["canonical_h_row"]) for event in events
                           if event.get("catalog_index") is not None)
            closed = {int(event["reduced_facet_word_hex"], 16) for event in reduced_events}
            reduced.update((node_id, word) for word in closed)
            row["classes"].update(normals)
            row["reduced"].update(closed)
            stats = result["geometry"]["statistics"]
            lp = stats.get("ridge_lp_attempts", 0) + stats.get("support_lp_attempts", 0)
            seconds = result["runtime"]["elapsed_seconds"]
            row["search_seconds"] += seconds
            row["lp_attempts"] += lp
            totals.update({"search_seconds": seconds, "audit_seconds": run["audit"]["elapsed_seconds"],
                           "class_records": len(events), "reduced_records": len(reduced_events),
                           "lp_attempts": lp, "ridge_lp_attempts": stats.get("ridge_lp_attempts", 0),
                           "support_lp_attempts": stats.get("support_lp_attempts", 0),
                           "iterations": result["runtime"]["iterations_completed"],
                           "full_path_audits": run["audit"].get("paths_checked", len(events)),
                           "reduced_path_audits": run["audit"]["reduced_witness_audit"]["paths_checked"]})
            totals.update({key: stats.get(key, 0) for key in ("successful_full_lifts", "failed_full_lifts")})
            totals.update({key: result.get("search_stats", {}).get(key, 0)
                           for key in ("rollout_steps", "tree_policy_steps")})
            totals["direct_filler_class_records"] += sum(
                not any(action["kind"] == "remove" for action in event.get("path", ())) for event in events)
            totals["corrector_assisted_class_records"] += sum(
                any(action["kind"] == "remove" for action in event.get("path", ())) for event in events)
            totals["deterministic_ridge_assisted_class_records"] += sum(
                any(action.get("source") == "deterministic_ridge_exit"
                    for action in event.get("local_path", event.get("path", ()))) for event in events)
            totals["certified_lower_assisted_class_records"] += sum(
                any(action.get("source") == "certified_lower_face_exit"
                    for action in event.get("local_path", event.get("path", ()))) for event in events)
            totals["certified_lower_terminal_class_records"] += sum(
                bool(event.get("path") and event["path"][-1].get("source") == "certified_lower_face_exit")
                for event in events)
            frontier_totals.update(result.get("face_frontier", {}).get("statistics", {}))
            lower_totals.update(result.get("lower_face_exits", {}).get("statistics", {}))
            schedule = result.get("frontier_schedule", {})
            schedule_totals.update(schedule.get("statistics", {}))
            fresh = result.get("fresh_geometry", {})
            probe_totals.update(fresh.get("statistics", {}))
            probe_totals.update(fresh.get("geometry_statistics", {}))
            adaptive = result.get("adaptive_exploration", {})
            adaptive_totals.update(adaptive.get("statistics", {}))
            totals["fresh_probe_lp_attempts"] += fresh.get("geometry_statistics", {}).get("fresh_probe_lp_attempts", 0)
            for event in fresh.get("probe_events", ()):
                metrics = probe_modes.setdefault(event.get("mode", "unknown"), Counter())
                metrics.update({"requests": 1, "lp_attempts": event.get("lp_attempts", 0),
                                "new_geometry": int(event.get("retained_word_hex") is not None),
                                "executed_proposals": int(event.get("first_taken_iteration") is not None),
                                "exit_arrivals": event.get("exit_arrivals", 0),
                                "new_reduced_exit_arrivals": event.get("new_reduced_exit_arrivals", 0),
                                "terminal_class_discoveries": event.get("terminal_class_discoveries", 0)})
            last = max((event["iteration"] for event in events), default=0)
            row["seed_runs"].append({"seed": summary["config"]["seed"], "class_count": len(normals),
                                     "reduced_count": len(closed), "search_seconds": seconds,
                                     "last_class_iteration": last,
                                     "iterations_since_last_class": result["runtime"]["iterations_completed"] - last,
                                     "face_frontier": result.get("face_frontier"),
                                     "lower_face_exits": result.get("lower_face_exits"),
                                     "frontier_schedule": {key: value for key, value in schedule.items()
                                                           if key not in ("sources", "cached_faces", "bank_history")},
                                     "fresh_geometry": {key: value for key, value in fresh.items() if key != "probe_events"},
                                     "adaptive_exploration": {key: value for key, value in adaptive.items()
                                                              if key not in ("sources", "entrance_events")},
                                     "class_growth_500": [sum(start < event["iteration"] <= start + 500
                                                               for event in events)
                                                          for start in range(0, result["runtime"]["iterations_completed"], 500)]})
            for window in result["entry_diagnostics"]["windows"]:
                end = window["end_iteration_inclusive"]
                metrics = windows.setdefault(end, Counter())
                for entry in window["entries"].values():
                    metrics.update({key: entry.get(key, 0) for key in
                                    ("iterations", "elapsed_seconds", "new_local_classes", "new_reduced_facets",
                                     "ridge_lp_attempts", "support_lp_attempts")})
    totals["canonical_class_union"] = len(classes)
    totals["catalog_matched_union"] = len(classes & catalog)
    totals["catalog_unmatched_union"] = len(classes - catalog)
    totals["node_reduced_closure_union"] = len(reduced)
    totals["canonical_classes_per_search_second"] = len(classes) / totals["search_seconds"]
    totals["class_records_per_1000_lp"] = 1000 * totals["class_records"] / totals["lp_attempts"]
    totals["reduced_records_per_1000_lp"] = 1000 * totals["reduced_records"] / totals["lp_attempts"]
    arrivals = frontier_totals["frontier_terminal_arrivals"]
    return {"sources": sources, "totals": dict(totals),
            "frontier_statistics": dict(frontier_totals),
            "lower_exit_statistics": dict(lower_totals),
            "frontier_schedule_statistics": dict(schedule_totals),
            "fresh_geometry_statistics": dict(probe_totals),
            "adaptive_exploration_statistics": dict(adaptive_totals),
            "fresh_probe_mode_statistics": {mode: dict(row) for mode, row in sorted(probe_modes.items())},
            "repeat_exit_arrival_fraction": (frontier_totals["frontier_repeated_exit_arrivals"] / arrivals
                                              if arrivals else None),
            "canonical_h_rows": [list(row) for row in sorted(classes)],
            "nodes": [{"node_id": node_id, "canonical_class_union": len(row["classes"]),
                       "reduced_closure_union": len(row["reduced"]),
                       "search_seconds": row["search_seconds"], "lp_attempts": row["lp_attempts"],
                       "seed_runs": row["seed_runs"]} for node_id, row in nodes.items()],
            "windows": [{"end_iteration_per_node_seed": end, **dict(row)}
                        for end, row in sorted(windows.items())]}


def compare(reference_paths, variant_paths, *, allow_policy_differences=()):
    allowed = set(allow_policy_differences)
    if not allowed <= POLICY_SWITCHES:
        raise ValueError("only named policy switches can differ; seed, budget and geometry parameters must match")
    if len(reference_paths) != len(variant_paths):
        raise ValueError("each variant run needs a matched reference")
    missing_diagnostics = []
    policy_differences = []
    for reference_path, variant_path in zip(reference_paths, variant_paths):
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        variant = json.loads(variant_path.read_text(encoding="utf-8"))
        if reference["selected_node_ids"] != variant["selected_node_ids"]:
            raise ValueError("node scope mismatch")
        changed = {key: {"reference": value, "variant": variant["config"].get(key)}
                   for key, value in reference["config"].items() if variant["config"].get(key) != value}
        if not set(changed) <= allowed:
            raise ValueError("reference search parameters must match exactly")
        if changed:
            policy_differences.append({"reference": str(reference_path), "variant": str(variant_path), "switches": changed})
        new_runs = {run["node_id"]: run for run in variant["runs"]}
        for run in reference["runs"]:
            old_result = json.loads(Path(run["output"]).read_text(encoding="utf-8"))
            new_result = json.loads(Path(new_runs[run["node_id"]]["output"]).read_text(encoding="utf-8"))
            new_normals = {tuple(event["canonical_h_row"]) for event in new_result["coverage"]["discoveries"]}
            new_words = {int(event["reduced_facet_word_hex"], 16)
                         for event in new_result["reduced_facets"]["discovery_events"]}
            schedule = new_result.get("frontier_schedule")
            if schedule:
                schedule = {name: {(item["source_word_hex"], item["retained_word_hex"]): item
                                   for item in schedule[name]}
                            for name in ("bank_history", "cached_faces")}
            missing = []
            for event in old_result["coverage"]["discoveries"]:
                if tuple(event["canonical_h_row"]) in new_normals:
                    continue
                path = event.get("path", ())
                removal = next((action for action in reversed(path) if action["kind"] == "remove"), None)
                source = int(removal["sampled_facet_word_hex"], 16) if removal else None
                retained_hex = removal.get("retained_word_hex") if removal else None
                retained = int(retained_hex, 16) if retained_hex is not None else None
                if retained is None and removal and "retained_vertices" in removal:
                    retained = sum(1 << index for index in removal["retained_vertices"])
                origin = event.get("continuation_origin")
                origin_word = int(origin["word_hex"], 16) if origin else None
                terminal_word = int(event["reduced_facet_word_hex"], 16) if "reduced_facet_word_hex" in event else None
                missing.append({
                    "class_id": event.get("class_id"), "reference_iteration": event["iteration"],
                    "reference_terminal_completion_source": path[-1].get("source") if path else None,
                    "reference_terminal_source_word_hex": f"0x{source:x}" if source is not None else None,
                    "reference_retained_word_hex": f"0x{retained:x}" if retained is not None else None,
                    "terminal_source_was_visited": source in new_words if source is not None else None,
                    "reference_continuation_origin_word_hex": f"0x{origin_word:x}" if origin_word is not None else None,
                    "continuation_origin_was_visited": origin_word in new_words if origin_word is not None else None,
                    "reference_terminal_was_visited": terminal_word in new_words if terminal_word is not None else None,
                    "recorded_frontier_gap": recorded_frontier_gap(source, retained, terminal_word, new_words, schedule),
                })
            missing_diagnostics.append({
                "node_id": run["node_id"], "seed": reference["config"]["seed"],
                "missing_class_records": len(missing),
                "visited_source_exit_gaps": sum(item["terminal_source_was_visited"] is True for item in missing),
                "unvisited_source_gaps": sum(item["terminal_source_was_visited"] is False for item in missing),
                "direct_root_gaps": sum(item["terminal_source_was_visited"] is None for item in missing),
                "recorded_frontier_gap_counts": dict(Counter(
                    item["recorded_frontier_gap"]["category"] for item in missing
                    if item["recorded_frontier_gap"] is not None)),
                "records": missing,
            })
    old, new = summarize(reference_paths), summarize(variant_paths)
    old_rows = {tuple(row) for row in old["canonical_h_rows"]}
    new_rows = {tuple(row) for row in new["canonical_h_rows"]}
    return {"scope": "same six reduced nodes, matched seed/budget; 4-2-2 class total unknown",
            "intentional_policy_differences": policy_differences,
            "missing_reference_path_diagnostics": missing_diagnostics,
            "missing_path_diagnostic_scope": ("actual closures and optional recorded bank/cache evidence; "
                                              "reference concrete path only, no causal or exhaustive-geometry claims"),
            "reference": old, "variant": new,
            "canonical_overlap_count": len(old_rows & new_rows),
            "reference_only_canonical_h_rows": [list(row) for row in sorted(old_rows - new_rows)],
            "variant_only_canonical_h_rows": [list(row) for row in sorted(new_rows - old_rows)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--variant", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-policy-difference", choices=sorted(POLICY_SWITCHES), action="append", default=[])
    args = parser.parse_args()
    result = compare(args.reference, args.variant, allow_policy_differences=args.allow_policy_difference)
    save_json(result, args.output)
    print(json.dumps({"reference": result["reference"]["totals"],
                      "variant": result["variant"]["totals"],
                      "canonical_overlap_count": result["canonical_overlap_count"]}, indent=2))


if __name__ == "__main__":
    main()
