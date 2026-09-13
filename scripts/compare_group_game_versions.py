"""Audit same-budget algorithm comparisons without assuming equal trajectories."""

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linprog

from compare_group_game_budgets import audit
from baseline.bell322 import generate_bell322_points
from baseline.geometry import affine_rank, validate_facet_support
from mcts.subgroup_patterns import canonical_support_word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = [json.loads(path.read_text(encoding="utf-8"))
                for path in (args.baseline, args.candidate)]
    assert old["config"]["iterations"] == new["config"]["iterations"]
    assert old["config"]["seed"] == new["config"]["seed"]
    audits = [audit(run) for run in (old, new)]
    merge_points = generate_bell322_points()
    for run in (old, new):
        replay = run["tree"]["facet_replay"]
        if "frontier_planned_pairs" in replay:
            planned = replay["frontier_planned_pairs"]
            if replay.get("frontier_executed_rewrite", 0) > planned:
                planned += replay.get("context_frontier_planned_pairs", 0)
            assert (0 <= replay.get("frontier_executed_add", 0)
                    <= replay.get("frontier_executed_rewrite", 0)
                    <= planned <= replay["frontier_episodes"]
                    <= replay["episodes"])
        for state in run["facet_corrector"].get("systematic_ridges", {}).values():
            assert 0 <= state["examined"] <= state["total_candidates"]
            if not state["eligible"]:
                assert state["examined"] == 0
            elif state["finished"]:
                assert state["examined"] == state["total_candidates"]
        graph = run["facet_corrector"].get("endpoint_graph")
        if run["facet_corrector"].get("endpoint_dedup"):
            pairs = graph["pairs"]
            stats = graph["statistics"]
            total = sum(p["executions"] for p in pairs)
            distinct = sum(p["executions"] > 0 for p in pairs)
            assert len(pairs) == graph["unique_source_target_pairs"]
            assert distinct == graph["distinct_executed_pairs"]
            assert total == stats.get("mapped_executions", 0)
            assert total == run["facet_corrector"]["statistics"].get("executed_facet_external_exit", 0)
            assert not stats.get("unmapped_executions", 0)
            assert total - distinct == stats.get("repeat_pair_executions", 0)
            assert sum(p["executions"] for p in pairs if p["source_hex"] == p["target_hex"]) == stats.get("self_pair_executions", 0)
            assert sum(p["executions"] == 0 and p["source_hex"] != p["target_hex"] for p in pairs) == graph["pending_nonself_pairs"]
        for word, frontier in run["facet_corrector"].get("orbit_face_frontiers", {}).items():
            source = int(word, 16)
            assert source == int(frontier["source_word_hex"], 16)
            assert 0 <= frontier["examined"] <= min(frontier["total_candidates"], frontier["max_candidates"])
            assert frontier["finished"] == (frontier["examined"] == min(frontier["total_candidates"], frontier["max_candidates"]))
            assert frontier["exhaustive"] == (frontier["examined"] == frontier["total_candidates"])
            counts = frontier["statistics"]
            assert sum(counts.get(k, 0) for k in ("accepted", "rank_low", "rank_high", "not_supporting")) == frontier["examined"]
            assert len(frontier["emitted"]) == counts.get("accepted", 0)
            for face in frontier["emitted"]:
                retained = int(face["retained_word_hex"], 16)
                assert not retained & ~source
                blocks = [int(b, 16) for b in frontier["pattern_blocks"][face["pattern_id"]]]
                assert all(retained & b in (0, b) for b in blocks)
                assert sum(not retained & b for b in blocks) == face["removed_orbit_count"]
                inside = [i for i in range(64) if retained >> i & 1]
                removed = [i for i in range(64) if (source & ~retained) >> i & 1]
                assert run["config"]["min_corrector_rank"] <= affine_rank(merge_points[inside]) == face["rank"] <= 24
                affine = np.column_stack((merge_points, np.ones(64)))
                certificate = linprog(np.zeros(27), A_eq=affine[inside], b_eq=np.zeros(len(inside)),
                                      A_ub=-affine[removed], b_ub=-np.ones(len(removed)),
                                      bounds=[(None, None)] * 27, method="highs")
                assert certificate.success
        for event in run["facet_corrector"].get("orbit_face_publication_events", []):
            source, retained = (int(event[k], 16) for k in ("source_word_hex", "retained_word_hex"))
            assert not retained & ~source
            assert 0 < event["iteration"] <= run["summary"]["iterations_completed"]
            for word in event["exit_words_hex"]:
                outside = int(word, 16)
                assert outside & ~source
                assert validate_facet_support(merge_points, [i for i in range(64) if (retained | outside) >> i & 1]).valid
        publication = run["facet_corrector"].get("publication_audit", {})
        scheduler = run["facet_corrector"].get("frontier_scheduling", {})
        yield_scheduler = run["facet_corrector"].get("yield_scheduling", {})
        if yield_scheduler.get("enabled"):
            assert run["config"]["yield_frontier"] and not scheduler["enabled"]
            assert not yield_scheduler["class_labels_used"]
            decisions = yield_scheduler["decisions"]
            assert len(decisions) == yield_scheduler["choices"]
            assert len(decisions) <= replay.get("episodes", 0) // yield_scheduler["replay_stride"]
            assert [d["iteration"] for d in decisions] == sorted({d["iteration"] for d in decisions})
            for i, decision in enumerate(decisions):
                assert 0 < decision["iteration"] <= run["summary"]["iterations_completed"]
                assert decision["lane"] in ("ridge", "orbit", "completion", "delivery")
                assert decision["reason"] == ("probe" if i % yield_scheduler["probe_stride"] == 0 else "yield")
                assert all(np.isfinite(decision[k]) and decision[k] >= 0
                           for k in ("score", "observations", "reward_ema", "seconds_ema"))
            for work in yield_scheduler["work"]:
                assert work["observations"] > 0
                assert 0 <= work["new_targets"] <= work["new_pairs"]
                assert all(np.isfinite(work[k]) and work[k] >= 0
                           for k in ("seconds", "seconds_ema", "reward_ema"))
            assert sum(w["new_pairs"] for w in yield_scheduler["work"]) <= sum(
                p["source_hex"] != p["target_hex"] for p in graph["pairs"])
        saturation_offers = replay.get("saturated_facet_repair_offers", 0)
        saturation_skips = replay.get("saturated_facet_repair_skips", 0)
        assert 0 <= saturation_skips <= saturation_offers
        if run["config"].get("endpoint_saturation_gate"):
            assert run["config"]["endpoint_dedup"]
        cache = run["tree"].get("generation_cache", {})
        if cache:
            assert all(value >= 0 for value in cache.values())
            assert cache["coherence_entries"] <= run["config"]["cache_light_max_entries"]
            assert cache["candidate_entries"] <= run["config"]["cache_heavy_max_entries"]
        contexts = run["tree"].get("endpoint_contexts", {})
        if contexts.get("enabled"):
            assert not contexts["class_labels_used"]
            assert contexts["key_fields"] == [
                "canonical_source",
                "canonical_target",
                "repair_pattern_id",
                "corrections_used",
            ]
            rows = contexts["executions"]
            assert len(rows) == contexts["distinct_contexts"]
            assert len({(row["source_hex"], row["target_hex"],
                        row["repair_pattern_id"], row["corrections_used"])
                       for row in rows}) == len(rows)
            context_stats = contexts["statistics"]
            context_mapped = context_stats.get("mapped_executions", 0)
            graph_mapped = graph["statistics"].get("mapped_executions", 0)
            assert sum(row["executions"] for row in rows) == context_mapped
            assert context_stats.get("unmapped_executions", 0) == 0
            assert context_mapped == graph_mapped
            graph_pairs = {(row["source_hex"], row["target_hex"])
                           for row in graph["pairs"]}
            for row in rows:
                assert row["executions"] > 0
                assert 1 <= row["corrections_used"] <= run["config"]["extended_max_corrections"]
                assert (row["source_hex"], row["target_hex"]) in graph_pairs
            assert sum(contexts["distinct_contexts_by_corrections_used"].values()) == len(rows)
        if scheduler.get("enabled"):
            assert run["config"]["fair_frontier"]
            assert not scheduler["class_labels_used"]
            by_lane = Counter()
            for row in scheduler["source_selections"]:
                assert row["count"] > 0
                assert row["scope"] in ("grow", "replay")
                assert row["lane"] in ("ridge", "completion", "orbit", "delivery")
                assert row["scope"] != "grow" or row["lane"] != "delivery"
                by_lane[f"{row['scope']}:{row['lane']}"] += row["count"]
            assert dict(by_lane) == scheduler["selections"]
            for name, count in by_lane.items():
                scope, lane = name.split(":")
                dispatched = run["facet_corrector"]["statistics"].get(f"scheduled_{scope}_{lane}_dispatches", 0)
                assert 0 <= dispatched <= count
                if scope == "grow":
                    assert dispatched == count
            assert sum(n for name, n in by_lane.items() if name.startswith("replay:")) == run["tree"]["facet_replay"].get("frontier_episodes", 0)
            for scope, count in scheduler["pressure_selections"].items():
                assert 0 <= count <= sum(n for name, n in by_lane.items() if name.startswith(scope + ":"))
        if publication.get("enabled"):
            assert publication["unpublished_candidate_action_keys"] == 0
            assert not publication["unpublished_candidate_targets_hex"]
            assert publication["verified_candidate_action_keys"] == publication["published_action_keys"] == graph["verified_action_keys"]
        continuation = run["facet_corrector"].get("lower_face_frontier", {})
        if continuation.get("max_attempts_per_key", 0):
            jobs = continuation["jobs"]
            limit = continuation["max_attempts_per_key"]
            assert not continuation["class_labels_used"]
            assert len({(j["canonical_source_hex"], j["canonical_retained_hex"],
                         j["canonical_pattern_id"]) for j in jobs}) == len(jobs)
            assert sum(j["attempts"] for j in jobs) == run["facet_corrector"]["statistics"].get("completion_lp_attempts", 0)
            assert sum(j["continuation_batches"] for j in jobs) == run["facet_corrector"]["statistics"].get("completion_continuation_batches", 0)
            for job in jobs:
                source = int(job["canonical_source_hex"], 16)
                retained = int(job["canonical_retained_hex"], 16)
                assert not retained & ~source
                assert run["config"]["min_corrector_rank"] <= affine_rank(
                    merge_points[[i for i in range(64) if retained >> i & 1]]) < 24
                assert 0 < job["attempts"] <= limit
                assert 0 <= job["unique_exit_keys"] <= job["attempts"]
                assert job["finished"] == (job["attempts"] == limit)
        for event in run["facet_corrector"].get("local_merge_events", []):
            source = int(event["source_word_hex"], 16)
            retained = int(event["retained_word_hex"], 16)
            additions = [int(w, 16) for w in event["added_exit_words_hex"]]
            assert not retained & ~source
            assert len(set(additions)) == len(additions) == event["new_exit_count"] - event["previous_exit_count"] > 0
            assert 0 < event["iteration"] <= run["summary"]["iterations_completed"]
            for added in additions:
                assert added & ~source
                assert validate_facet_support(merge_points, [i for i in range(64) if (retained | added) >> i & 1]).valid
    a, b = audits
    common = old["config"].keys() & new["config"].keys()
    changed = {k: [old["config"][k], new["config"][k]] for k in sorted(common)
               if old["config"][k] != new["config"][k]}
    timeline_old = {event["class_id"]: event for event in a["timeline"]}
    timeline_new = {event["class_id"]: event for event in b["timeline"]}
    labels = {canonical_support_word(int(d["tight_support_word_hex"], 16)): d["class_id"]
              for d in new["discoveries"]}
    discovered_at = {d["class_id"]: d["iteration"] for d in new["discoveries"]}
    merge_endpoints = []
    for event in new["facet_corrector"].get("local_merge_events", []):
        retained = int(event["retained_word_hex"], 16)
        for outside_hex in event["added_exit_words_hex"]:
            selected = retained | int(outside_hex, 16)
            check = validate_facet_support(merge_points, [i for i in range(64) if selected >> i & 1])
            tight = sum(1 << i for i, distance in enumerate(merge_points @ check.normal + check.offset)
                        if abs(distance) <= 1e-6)
            canonical = canonical_support_word(tight)
            class_id = labels.get(canonical)
            merge_endpoints.append({
                "merge_iteration": event["iteration"],
                "source_word_hex": event["source_word_hex"],
                "target_canonical_word_hex": f"0x{canonical:016x}",
                "target_class_posthoc": class_id,
                "target_first_discovery_iteration": discovered_at.get(class_id),
                "target_observed_at_merge": class_id is not None and discovered_at[class_id] <= event["iteration"],
            })
    enumerators = []
    orbit_endpoint_events = []
    for event in new["facet_corrector"].get("orbit_face_publication_events", []):
        retained = int(event["retained_word_hex"], 16)
        targets = set()
        for outside_hex in event["exit_words_hex"]:
            selected = retained | int(outside_hex, 16)
            check = validate_facet_support(merge_points, [i for i in range(64) if selected >> i & 1])
            tight = sum(1 << i for i, distance in enumerate(merge_points @ check.normal + check.offset)
                        if abs(distance) <= 1e-6)
            targets.add(canonical_support_word(tight))
        orbit_endpoint_events.append({
            "publication_iteration": event["iteration"],
            "source_word_hex": event["source_word_hex"],
            "retained_word_hex": event["retained_word_hex"], "rank": event["rank"],
            "targets": [{"canonical_word_hex": f"0x{target:016x}",
                         "class_id_posthoc": labels.get(target),
                         "first_discovery_iteration": discovered_at.get(labels.get(target))}
                        for target in sorted(targets)],
        })
    for word, state in new["facet_corrector"].get("systematic_ridges", {}).items():
        if state["eligible"]:
            enumerators.append({"canonical_word_hex": word,
                                "class_id_after_terminal_validation": labels.get(int(word, 16)),
                                "support_size": int(word, 16).bit_count(), **state})
    gained = sorted(set(b["observed_classes"]) - set(a["observed_classes"]))
    gained_paths = []
    points = generate_bell322_points()
    for event in new["discoveries"]:
        if event["class_id"] not in gained:
            continue
        word = 0
        repairs = []
        for action in event["path"]:
            if action["kind"] == "add":
                word |= int(action["effective_word_hex"], 16)
            elif action["kind"] == "rewrite":
                word = (word & ~int(action["remove_word_hex"], 16)) | int(action["add_word_hex"], 16)
            elif action["kind"] == "remove":
                word &= ~int(action["effective_word_hex"], 16)
            if action["source"] == "facet_repartition":
                source = int(action["source_facet_word_hex"], 16)
                repairs.append({"source_class_after_validation": labels.get(canonical_support_word(source)),
                                "source_size": source.bit_count(), "retained_size": word.bit_count(),
                                "retained_rank": affine_rank(points[[i for i in range(64) if word >> i & 1]]),
                                "removed_size": int(action["remove_word_hex"], 16).bit_count(),
                                "added_size": int(action["add_word_hex"], 16).bit_count()})
        gained_paths.append({"class_id": event["class_id"], "repairs": repairs})
    completion_jobs = new["facet_corrector"].get("lower_face_frontier", {}).get("jobs", [])
    orbit_frontiers = list(new["facet_corrector"].get("orbit_face_frontiers", {}).values())
    comparison = {
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "algorithm_revisions": [run["metadata"]["algorithm_revision"] for run in (old, new)],
        "changed_common_config": changed,
        "added_config": {k: new["config"][k] for k in sorted(new["config"].keys() - common)},
        "audits": audits,
        "gained_classes": gained,
        "gained_class_repair_paths": gained_paths,
        "local_merge_endpoint_analysis": merge_endpoints,
        "publication_audit": new["facet_corrector"].get("publication_audit"),
        "frontier_scheduling": new["facet_corrector"].get("frontier_scheduling"),
        "yield_scheduling": new["facet_corrector"].get("yield_scheduling"),
        "orbit_publication_endpoint_analysis": orbit_endpoint_events,
        "orbit_face_summary": {
            "sources": len(orbit_frontiers),
            "finished": sum(e["finished"] for e in orbit_frontiers),
            "exhaustive": sum(e["exhaustive"] for e in orbit_frontiers),
            "examined": sum(e["examined"] for e in orbit_frontiers),
            "statistics": {key: sum(e["statistics"].get(key, 0) for e in orbit_frontiers)
                           for key in ("accepted", "rank_low", "rank_high", "not_supporting", "support_lp_attempts")},
            "publications": new["facet_corrector"]["statistics"].get("orbit_face_publications", 0),
            "interpretation": "Orbit-union candidates from observed facets only; bounded streams are not exhaustive facet search.",
        },
        "completion_frontier_summary": {
            "jobs": len(completion_jobs),
            "finished": sum(j["finished"] for j in completion_jobs),
            "pending_jobs": sum(not j["finished"] for j in completion_jobs),
            "remaining_lp_attempts": sum(new["facet_corrector"]["lower_face_frontier"]["max_attempts_per_key"] - j["attempts"]
                                         for j in completion_jobs),
            "lp_attempts": sum(j["attempts"] for j in completion_jobs),
            "continuation_batches": sum(j["continuation_batches"] for j in completion_jobs),
            "without_valid_exit": sum(j["unique_exit_keys"] == 0 for j in completion_jobs),
            "with_multiple_exit_keys": sum(j["unique_exit_keys"] > 1 for j in completion_jobs),
            "interpretation": "Geometric completion work, not novel class count; exhausted budgets are not exhaustive proofs.",
        },
        "lost_classes": sorted(set(a["observed_classes"]) - set(b["observed_classes"])),
        "seconds_difference": b["elapsed_seconds"] - a["elapsed_seconds"],
        "time_change_percent": 100 * (b["elapsed_seconds"] / a["elapsed_seconds"] - 1),
        "class_comparison": [{"class_id": c, "baseline": timeline_old.get(c), "candidate": timeline_new.get(c)}
                             for c in old["config"]["target_classes"]],
        "coverage_checkpoints": [
            {"iteration": i, "baseline": sum(e["iteration"] <= i for e in a["timeline"]),
             "candidate": sum(e["iteration"] <= i for e in b["timeline"])}
            for i in range(1000, new["config"]["iterations"] + 1, 1000)],
        "frontier_work_comparison": [{
            "ridge_candidates": run["facet_corrector"]["statistics"].get("systematic_candidates", 0),
            "ridge_finished": sum(e["eligible"] and e["finished"] for e in run["facet_corrector"].get("systematic_ridges", {}).values()),
            "orbit_candidates": sum(e["examined"] for e in run["facet_corrector"].get("orbit_face_frontiers", {}).values()),
            "completion_pending": sum(not j["finished"] for j in run["facet_corrector"]["lower_face_frontier"]["jobs"]),
            "completion_remaining_lp": sum(run["facet_corrector"]["lower_face_frontier"]["max_attempts_per_key"] - j["attempts"]
                                           for j in run["facet_corrector"]["lower_face_frontier"]["jobs"]),
            "completion_lp_attempts": run["facet_corrector"]["statistics"].get("completion_lp_attempts", 0),
        } if "lower_face_frontier" in run["facet_corrector"] else None for run in (old, new)],
        "eligible_systematic_frontiers": enumerators,
        "interpretation": "Same seed and budget, different trajectories; not a multi-seed causal estimate.",
    }
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps({k: comparison[k] for k in (
        "gained_classes", "lost_classes", "seconds_difference", "time_change_percent",
        "changed_common_config", "added_config", "eligible_systematic_frontiers")}, indent=2))


if __name__ == "__main__":
    main()
