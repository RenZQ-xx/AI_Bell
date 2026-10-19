from __future__ import annotations

"""Summarize observed first-discovery paths without changing the search."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT
from .geometry import FacetOracle, RankOracle


def analyze_paths(run_path: Path, *, handoff_root: Path = DEFAULT_HANDOFF_ROOT,
                  payload: dict | None = None) -> dict:
    if payload is None:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    events = payload["coverage"]["discoveries"]
    anchors = payload["continuation_game"]["anchors"]
    labels = {int(item["support_word_hex"], 16): item["class_id"] for item in anchors}
    ranks = RankOracle()
    oracle = FacetOracle(
        rank_oracle=ranks, handoff_root=handoff_root,
        use_catalog=False, canonicalize=False,
    )
    transitions: dict[tuple, tuple[int, int]] = {}
    rows = []
    root_yields: Counter[int] = Counter()
    class_starts: Counter[str] = Counter()
    class_yields: Counter[str] = Counter()
    class_episodes: dict[str, set[int]] = defaultdict(set)
    anchor_yields: Counter[int] = Counter()
    anchor_episodes: dict[int, set[int]] = defaultdict(set)
    origin_differs_from_parent = 0

    for item in anchors:
        class_starts[item["class_id"]] += item["starts"]

    for event in events:
        word = 0
        steps = []
        facets = []
        for index, action in enumerate(event["path"]):
            kind = action["kind"]
            removed = int(action["remove_word_hex"], 16)
            added = int(action["add_word_hex"], 16)
            before = word
            key = before, kind, removed, added
            if key not in transitions:
                if kind == "remove":
                    word &= ~removed
                elif kind == "add":
                    word |= added
                elif kind == "rewrite":
                    word = (word & ~removed) | added
                rank = ranks.rank(word)
                if rank == 79:
                    facet = oracle.validate(word)
                    if not facet.valid:
                        raise ValueError(f"invalid facet in {event['class_id']}")
                    word = facet.support_word
                if rank > 79:
                    raise ValueError(f"rank overflow in {event['class_id']}")
                transitions[key] = word, rank
            word, rank = transitions[key]
            steps.append({
                "kind": kind, "source": action["source"],
                "pattern_id": action["pattern_id"],
                "rank_before": ranks.rank(before), "rank_after": rank,
                "removed_points": (before & removed).bit_count(),
                "explicit_added_points": (added & ~before).bit_count(),
                "support_before": before.bit_count(), "support_after": word.bit_count(),
            })
            if rank == 79 and (not facets or facets[-1]["word"] != word):
                if word not in labels:
                    raise ValueError(f"missing observed facet anchor: {word:x}")
                facets.append({"word": word, "class_id": labels[word], "step": index})
        if word != int(event["support_word_hex"], 16):
            raise ValueError(f"witness mismatch in {event['class_id']}")
        root = facets[0]
        parent = facets[-2] if len(facets) > 1 else None
        start = parent["step"] + 1 if parent else 0
        last_jump = steps[start:]
        origin = event.get("continuation_origin")
        prefix = origin["historical_path_length"] if origin else 0
        local = steps[prefix:]
        corrections = sum(step["kind"] in ("remove", "rewrite") for step in local)
        if origin:
            class_id = origin["class_id"]
            anchor_word = int(origin["support_word_hex"], 16)
            class_yields[class_id] += 1
            class_episodes[class_id].add(event["iteration"])
            anchor_yields[anchor_word] += 1
            anchor_episodes[anchor_word].add(event["iteration"])
            origin_differs_from_parent += int(parent is not None and class_id != parent["class_id"])
        root_yields[root["word"]] += 1
        rows.append({
            "class_id": event["class_id"], "catalog_index": event["catalog_index"],
            "iteration": event["iteration"], "elapsed_seconds": event["elapsed_seconds"],
            "episode_kind": event["episode_kind"], "rollout_policy": event["rollout_policy"],
            "origin_class": origin["class_id"] if origin else None,
            "origin_support_word_hex": origin["support_word_hex"] if origin else None,
            "immediate_parent_class": parent["class_id"] if parent else None,
            "root_class": root["class_id"], "root_support_word_hex": f"0x{root['word']:064x}",
            "root_pattern": event["path"][root["step"]]["pattern_id"],
            "local_corrections": corrections,
            "full_corrections": sum(step["kind"] in ("remove", "rewrite") for step in steps),
            "local_depth": len(local), "full_depth": len(steps),
            "canonical_hops": sum(left["class_id"] != right["class_id"] for left, right in zip(facets, facets[1:])),
            "concrete_hops": len(facets) - 1,
            "facet_chain": [item["class_id"] for item in facets],
            "final_action_source": steps[-1]["source"],
            "last_jump": last_jump,
            "local_steps": local,
        })

    class_table = sorted(({
        "class_id": class_id, "starts": starts,
        "new_classes": class_yields[class_id],
        "episodes_with_novelty": len(class_episodes[class_id]),
        "classes_per_start": class_yields[class_id] / starts if starts else None,
    } for class_id, starts in class_starts.items()), key=lambda item: (-item["new_classes"], item["class_id"]))
    selected = [item for item in anchors if item["starts"]]
    zero_yield = [item for item in selected if not anchor_yields[int(item["support_word_hex"], 16)]]
    anchor_table = sorted(({
        "class_id": item["class_id"], "support_word_hex": item["support_word_hex"],
        "starts": item["starts"], "new_classes": anchor_yields[int(item["support_word_hex"], 16)],
        "episodes_with_novelty": len(anchor_episodes[int(item["support_word_hex"], 16)]),
    } for item in selected), key=lambda item: (-item["new_classes"], item["support_word_hex"]))

    windows = []
    previous_seconds = 0.0
    previous_local = previous_root = 0
    for window in payload["continuation_game"]["growth_windows"]:
        end = window["iteration"]
        selected_rows = [row for row in rows if end - 1000 < row["iteration"] <= end]
        local_episodes = window["local_continuation_episodes"] - previous_local
        root_episodes = window["root_filler_episodes"] - previous_root
        duration = window["elapsed_seconds"] - previous_seconds
        windows.append({
            "end_iteration": end, "search_seconds": duration,
            "new_classes": len(selected_rows), "classes_per_hour": len(selected_rows) * 3600 / duration,
            "continuation_episodes": local_episodes, "root_episodes": root_episodes,
            "continuation_new_classes": sum(row["episode_kind"] == "continuation" for row in selected_rows),
            "root_episode_new_classes": sum(row["episode_kind"] == "root_filler" for row in selected_rows),
            "direct_lp_new_classes": sum(row["final_action_source"] == "fullspace_supporting_face_lp" for row in selected_rows),
            "catalog_matched": sum(row["catalog_index"] is not None for row in selected_rows),
        })
        previous_seconds = window["elapsed_seconds"]
        previous_local = window["local_continuation_episodes"]
        previous_root = window["root_filler_episodes"]

    jumps = [row["last_jump"] for row in rows if row["immediate_parent_class"]]
    delete_steps = [step for jump in jumps for step in jump if step["kind"] in ("remove", "rewrite")]
    root_patterns = Counter(row["root_pattern"] for row in rows)
    discovery_episodes = Counter(row["episode_kind"] for row in {
        (row["iteration"], row["episode_kind"]): row for row in rows
    }.values())
    ordered = sorted(rows, key=lambda item: item["elapsed_seconds"])
    gaps = sorted(({
        "from_iteration": left["iteration"], "to_iteration": right["iteration"],
        "iteration_gap": right["iteration"] - left["iteration"],
        "seconds": right["elapsed_seconds"] - left["elapsed_seconds"],
    } for left, right in zip(ordered, ordered[1:])), key=lambda item: -item["seconds"])
    return {
        "run_path": str(run_path),
        "scope": "First-discovery provenance and observed anchor starts, not all episode trajectories or wall-time profiling.",
        "class_count": len(rows), "unique_transitions_replayed": len(transitions),
        "final_action_sources": dict(Counter(row["final_action_source"] for row in rows)),
        "novel_episodes": dict(discovery_episodes),
        "local_correction_counts": dict(Counter(row["local_corrections"] for row in rows)),
        "full_correction_counts": dict(Counter(row["full_corrections"] for row in rows)),
        "canonical_hop_counts": dict(Counter(row["canonical_hops"] for row in rows)),
        "origin_differs_from_immediate_parent": origin_differs_from_parent,
        "distinct_immediate_parent_classes": len({row["immediate_parent_class"] for row in rows if row["immediate_parent_class"]}),
        "root_pattern_yields": dict(root_patterns.most_common()),
        "distinct_root_patterns_in_provenance": len(root_patterns),
        "distinct_concrete_root_entrances_in_provenance": len(root_yields),
        "root_entrance_yields": [{"support_word_hex": f"0x{word:064x}", "class_id": labels[word], "new_classes": count} for word, count in root_yields.most_common()],
        "last_jump_rank_paths": dict(Counter(" / ".join(f"{step['kind']}:{step['rank_before']}->{step['rank_after']}" for step in jump) for jump in jumps)),
        "last_jump_delete_sources": dict(Counter(step["source"] for step in delete_steps)),
        "last_jump_deleted_point_counts": dict(Counter(step["removed_points"] for step in delete_steps)),
        "selected_anchors": len(selected), "zero_novelty_selected_anchors": len(zero_yield),
        "starts_from_zero_novelty_anchors": sum(item["starts"] for item in zero_yield),
        "zero_novelty_selected_classes": sum(item["starts"] > 0 and not item["new_classes"] for item in class_table),
        "starts_from_zero_novelty_classes": sum(item["starts"] for item in class_table if not item["new_classes"]),
        "class_origin_yields": class_table, "anchor_origin_yields": anchor_table,
        "windows": windows, "largest_discovery_time_gaps": gaps[:10], "classes": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    args = parser.parse_args(argv)
    result = analyze_paths(args.run_path, handoff_root=args.handoff_root)
    output = args.output or args.run_path.with_suffix(".path_analysis.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in {
        "classes", "class_origin_yields", "anchor_origin_yields", "root_entrance_yields",
    }}
    summary["top_class_origins"] = result["class_origin_yields"][:10]
    summary["top_concrete_roots"] = result["root_entrance_yields"][:10]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
