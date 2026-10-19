from __future__ import annotations

"""Audit canonical terminals and discovery paths from a quotient-search run."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .atlas import DEFAULT_HANDOFF_ROOT
from .geometry import FacetOracle, canonical_h_row
from .problem import deterministic_vertices, named_relabelings, signed_coordinate_action


def _source_family(source: str) -> str:
    if source == "fullspace_supporting_face_lp":
        return "lp_seed"
    if source == "fullspace_ridge_frontier":
        return "ridge_exit"
    if source in ("ridge_carve_exact", "quotient_ridge_basis"):
        return "ridge_carve"
    if source in ("handoff_subgroup", "identity", "online_stabilizer"):
        return "orbit"
    if source == "symmetry_rewrite":
        return "rewrite"
    if source == "rank_tail_enumeration":
        return "tail"
    if source.startswith("quotient_"):
        return "quotient"
    return source


def _counter_rows(counter: Counter, *, limit: int | None = None) -> list[dict]:
    rows = counter.most_common(limit)
    return [{"key": key, "count": count} for key, count in rows]


def _slack_signatures(
    rows: Sequence[tuple[int, ...]],
    augmented: np.ndarray,
    *,
    batch_size: int = 1024,
) -> list[tuple[int, ...]]:
    signatures: list[tuple[int, ...]] = []
    for start in range(0, len(rows), batch_size):
        values = np.asarray(rows[start : start + batch_size], dtype=np.int64) @ (
            augmented.T
        )
        signatures.extend(tuple(map(int, np.sort(value))) for value in values)
    return signatures


def _catalog_rows(handoff_root: Path) -> list[tuple[int, ...]]:
    path = handoff_root / "03_global_facets" / "global_facet_classes.csv"
    rows: list[tuple[int, ...]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            rows.append(tuple(map(int, json.loads(item["canonical_h_row"]))))
    return rows


def _generator_invariance_failures(
    rows: Sequence[tuple[int, ...]],
) -> list[dict[str, object]]:
    generators = tuple(named_relabelings().items())
    failures: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        name, relabeling = generators[index % len(generators)]
        transformed = [0] * 80
        for source, (target, sign) in enumerate(
            signed_coordinate_action(relabeling)
        ):
            transformed[target] = sign * row[source + 1]
        if canonical_h_row((row[0], *transformed)) != row:
            failures.append({"row_index": index, "generator": name})
    return failures


def _numeric_summary(values: Iterable[int]) -> dict[str, float | int | None]:
    items = sorted(map(int, values))
    if not items:
        return {"minimum": None, "median": None, "mean": None, "p90": None, "maximum": None}
    return {
        "minimum": items[0],
        "median": statistics.median(items),
        "mean": statistics.mean(items),
        "p90": items[int(0.9 * (len(items) - 1))],
        "maximum": items[-1],
    }


def analyze_run(run_path: Path, *, handoff_root: Path, payload: dict | None = None) -> dict[str, object]:
    if payload is None:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    events = payload["coverage"]["discoveries"]
    oracle = FacetOracle(handoff_root=handoff_root)
    facets = [
        oracle.validate(int(event["support_word_hex"], 16))
        for event in events
    ]
    valid_pairs = [
        (event, facet)
        for event, facet in zip(events, facets)
        if facet.valid and facet.canonical_h_row is not None
    ]
    canonical_rows = [
        tuple(map(int, facet.canonical_h_row))
        for _event, facet in valid_pairs
    ]
    supports = [facet.support_word for _event, facet in valid_pairs]

    augmented = np.column_stack(
        (
            np.ones(256, dtype=np.int64),
            np.asarray(deterministic_vertices(), dtype=np.int64),
        )
    )
    discovered_signatures = _slack_signatures(canonical_rows, augmented)
    catalog_rows = _catalog_rows(handoff_root)
    catalog_signatures = _slack_signatures(catalog_rows, augmented)
    catalog_signature_set = set(catalog_signatures)

    unmatched_indices = [
        index
        for index, (event, _facet) in enumerate(valid_pairs)
        if event["catalog_index"] is None
    ]
    unmatched_signatures = Counter(
        discovered_signatures[index] for index in unmatched_indices
    )
    exact_canonical_counts = Counter(canonical_rows)
    support_counts = Counter(supports)
    class_id_counts = Counter(event["class_id"] for event in events)
    invariance_failures = _generator_invariance_failures(canonical_rows)

    terminal_sources = Counter(
        _source_family(event["path"][-1]["source"])
        for event in events
    )
    terminal_sources_matched = Counter(
        _source_family(event["path"][-1]["source"])
        for event in events
        if event["catalog_index"] is not None
    )
    terminal_sources_unmatched = Counter(
        _source_family(event["path"][-1]["source"])
        for event in events
        if event["catalog_index"] is None
    )
    window_counts = Counter((int(event["iteration"]) - 1) // 1000 + 1 for event in events)
    seed_patterns = Counter(event["path"][0]["pattern_id"] for event in events)
    seed_structures = Counter(event["path"][0]["structure"] for event in events)
    action_sources = Counter(
        _source_family(action["source"])
        for event in events
        for action in event["path"]
    )
    motifs = Counter(
        " > ".join(_source_family(action["source"]) for action in event["path"])
        for event in events
    )

    unmatched_signature_in_catalog = sum(
        discovered_signatures[index] in catalog_signature_set
        for index in unmatched_indices
    )
    return {
        "run": str(run_path),
        "algorithm": payload.get("algorithm"),
        "runtime": payload.get("runtime"),
        "catalog_scope": {
            "record_count": len(catalog_rows),
            "is_complete": False,
            "total_422_class_count_known": False,
        },
        "canonical_audit": {
            "reported_terminal_count": len(events),
            "revalidated_terminal_count": len(valid_pairs),
            "unique_support_count": len(support_counts),
            "duplicate_support_count": sum(count - 1 for count in support_counts.values()),
            "unique_canonical_h_row_count": len(exact_canonical_counts),
            "duplicate_canonical_h_row_count": sum(
                count - 1 for count in exact_canonical_counts.values()
            ),
            "unique_class_id_count": len(class_id_counts),
            "duplicate_class_id_count": sum(count - 1 for count in class_id_counts.values()),
            "stored_canonical_h_row_mismatch_count": sum(
                tuple(event.get("canonical_h_row") or ()) != facet.canonical_h_row
                for event, facet in valid_pairs
            ),
            "stored_catalog_index_mismatch_count": sum(
                event["catalog_index"] != facet.catalog_index for event, facet in valid_pairs
            ),
            "stored_support_closure_mismatch_count": sum(
                int(event["support_word_hex"], 16) != facet.support_word for event, facet in valid_pairs
            ),
            "nonprimitive_canonical_rows": sum(
                math.gcd(*(abs(value) for value in row)) != 1
                for row in canonical_rows
            ),
            "generator_invariance_rows_checked": len(canonical_rows),
            "generator_invariance_failure_count": len(invariance_failures),
            "generator_invariance_failures": invariance_failures,
            "catalog_matched_count": sum(
                event["catalog_index"] is not None for event in events
            ),
            "catalog_unmatched_count": len(unmatched_indices),
            "unmatched_unique_slack_signature_count": len(unmatched_signatures),
            "unmatched_slack_signature_collision_groups": sum(
                count > 1 for count in unmatched_signatures.values()
            ),
            "largest_unmatched_slack_signature_group": max(
                unmatched_signatures.values(), default=0
            ),
            "unmatched_with_signature_present_in_partial_catalog": (
                unmatched_signature_in_catalog
            ),
            "interpretation": (
                "Slack signatures are coarse invariants only. Equal signatures do not "
                "justify merging classes; exact full-group canonical H-rows do."
            ),
        },
        "discovery_curve": {
            "per_1000": {
                str(window): window_counts[window]
                for window in range(1, max(window_counts, default=0) + 1)
            },
            "last_discovery_iteration": max(
                (int(event["iteration"]) for event in events), default=None
            ),
        },
        "paths": {
            "terminal_sources": dict(terminal_sources),
            "terminal_sources_catalog_matched": dict(terminal_sources_matched),
            "terminal_sources_catalog_unmatched": dict(terminal_sources_unmatched),
            "depth": _numeric_summary(event["depth"] for event in events),
            "support_size": _numeric_summary(
                event["support_size"] for event in events
            ),
            "action_source_occurrences": dict(action_sources),
            "top_motifs": _counter_rows(motifs, limit=20),
            "top_seed_patterns": _counter_rows(seed_patterns, limit=30),
            "seed_structures": dict(seed_structures),
        },
        "runtime_pressure": {
            "rank_oracle": payload.get("rank_oracle", {}),
            "facet_oracle": payload.get("facet_oracle", {}),
            "search_stats": payload.get("search_stats", {}),
            "quotient_stats": payload.get("quotient_game", {}).get(
                "quotient_stats", {}
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_run(
        args.run.resolve(),
        handoff_root=args.handoff_root.resolve(),
    )
    output = args.output or args.run.with_suffix(".analysis.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **report["canonical_audit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
