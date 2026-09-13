"""Audit complete group-game reports and compare deterministic budget prefixes."""

import argparse
from collections import Counter
import json
from pathlib import Path

from baseline.bell322 import generate_bell322_points
from baseline.geometry import affine_rank, validate_facet_support
from mcts.subgroup_patterns import move_support_word


def without_times(value):
    if isinstance(value, dict):
        return {key: without_times(item) for key, item in value.items() if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [without_times(item) for item in value]
    return value


def audit(payload):
    summary, tree = payload["summary"], payload["tree"]
    observed = {int(k) for k, count in summary["class_terminal_counts"].items() if count > 0}
    assert observed == set(summary["discovered_class_ids"])
    assert set(summary["missing_target_class_ids"]) == set(payload["config"]["target_classes"]) - observed
    recycled = tree["epoch_recycled_edges"] + tree["stagnation_recycled_edges"] + tree["facet_replay"].get("recycled_edges", 0)
    assert tree["edge_count"] + recycled == tree["tree_expansions"] <= summary["iterations_completed"]
    points = generate_bell322_points()
    anchors = {int(a["tight_support_word_hex"], 16): a for a in payload["facet_corrector"]["anchors"]}
    ranks, sources = Counter(), Counter()
    external_checks = 0
    for discovery in payload["discoveries"]:
        word = 0
        corrections = 0
        for action in discovery["path"]:
            if action["kind"] == "add":
                word |= int(action["effective_word_hex"], 16)
            elif action["kind"] == "remove":
                word &= ~int(action["effective_word_hex"], 16)
                corrections += 1
            elif action["kind"] == "rewrite":
                word = (word & ~int(action["remove_word_hex"], 16)) | int(action["add_word_hex"], 16)
                corrections += 1
            if action["source"] == "facet_repartition":
                source = int(action["source_facet_word_hex"], 16)
                assert not word & ~source
                rank = affine_rank(points[[i for i in range(64) if word >> i & 1]])
                assert payload["config"]["min_corrector_rank"] <= rank <= 24
                ranks[rank] += 1
                for generator in anchors[source]["embedded_generators"][action["pattern_id"]]:
                    assert move_support_word(source, generator) == source
                    assert move_support_word(word, generator) == word
            elif action["source"] == "facet_external_exit":
                assert word & ~int(action["source_facet_word_hex"], 16)
                assert validate_facet_support(points, [i for i in range(64) if word >> i & 1]).valid
                external_checks += 1
        assert word == int(discovery["selected_support_word_hex"], 16)
        assert corrections == discovery["corrections_used"] <= payload["config"]["extended_max_corrections"]
        sources[discovery["path"][-1]["source"]] += 1
    for event in payload["corrector_novelty_credit_events"]:
        assert abs(sum(item["credit"] for item in event["contributors"]) - payload["config"]["new_class_reward"]) < 1e-8
    last = payload["discoveries"][-1] if payload["discoveries"] else {"iteration": 0, "elapsed_seconds": 0}
    return {
        "budget": payload["config"]["iterations"],
        "iterations_completed": summary["iterations_completed"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "observed_classes": sorted(observed), "coverage": len(observed),
        "missing_classes": summary["missing_target_class_ids"],
        "last_discovery_iteration": last["iteration"],
        "last_discovery_seconds": last["elapsed_seconds"],
        "tail_iterations": summary["iterations_completed"] - last["iteration"],
        "tail_seconds": summary["elapsed_seconds"] - last["elapsed_seconds"],
        "discovery_sources": dict(sources), "retained_ranks_on_discovery_paths": dict(ranks),
        "verified_external_steps": external_checks,
        "facet_replay": tree["facet_replay"],
        "geometry_statistics": payload["facet_corrector"]["statistics"],
        "timeline": [{k: d[k] for k in ("class_id", "iteration", "elapsed_seconds", "corrections_used")}
                     for d in payload["discoveries"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    audits = [audit(run) for run in runs]
    comparisons = []
    for before, after, a, b in zip(runs, runs[1:], audits, audits[1:]):
        config_a = {k: v for k, v in before["config"].items() if k != "iterations"}
        config_b = {k: v for k, v in after["config"].items() if k != "iterations"}
        assert config_a == config_b
        prefix = [d for d in after["discoveries"] if d["iteration"] <= a["iterations_completed"]]
        assert without_times(prefix) == without_times(before["discoveries"])
        assert without_times(before["corrector_diagnostics"]) == without_times(after["corrector_diagnostics"])
        gained = sorted(set(b["observed_classes"]) - set(a["observed_classes"]))
        comparisons.append({"from_budget": a["budget"], "to_budget": b["budget"],
                            "prefix_matches": True, "gained_classes": gained,
                            "additional_total_seconds": b["elapsed_seconds"] - a["elapsed_seconds"],
                            "further_budget_supported": bool(gained and b["missing_classes"])})
    output = {"runs": audits, "comparisons": comparisons}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "runs": [{k: a[k] for k in ("budget", "coverage", "elapsed_seconds", "missing_classes", "tail_iterations")}
                               for a in audits], "comparisons": comparisons}))


if __name__ == "__main__":
    main()
