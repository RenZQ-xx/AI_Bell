#!/usr/bin/env python3
"""Derive block-only reachable classes from the grouped 46x46 matrix summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "class46x46_grouped_min_occupancy_matrix_summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "block_only_reachable_classes_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive P0 block-only reachable class sets from the grouped 46x46 matrix summary."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_payload(summary: dict[str, Any]) -> dict[str, Any]:
    matrix = summary["matrix"]
    row_groups = summary["row_groups"]
    payload: dict[str, Any] = {
        "semantics": {
            "experiment": (
                "For a fixed row-class stabilizer partition, only allow whole-block operations. "
                "A column class is structurally reachable iff the selected cell has total partial "
                "occupancy P=0 under the matrix selection rule."
            ),
            "selection_rule": summary["semantics"]["selection"],
            "rows": summary["semantics"]["rows"],
            "columns": summary["semantics"]["columns"],
        },
        "row_results": {},
    }

    for row_class in map(str, range(1, 47)):
        reachable = []
        nonreachable = []
        for col_class in map(str, range(1, 47)):
            entries = matrix[row_class][col_class]
            total_partial = sum(int(entry["partial_occupied_orbits"]) for entry in entries)
            total_full = sum(int(entry["full_occupied_orbits"]) for entry in entries)
            total_occ = sum(int(entry["occupied_orbits_total"]) for entry in entries)
            record = {
                "class_id": int(col_class),
                "total_partial": total_partial,
                "total_full": total_full,
                "total_occupied_orbits": total_occ,
                "entries": entries,
            }
            if total_partial == 0:
                reachable.append(record)
            else:
                nonreachable.append(record)

        payload["row_results"][row_class] = {
            "row_groups": row_groups[row_class],
            "reachable_classes": reachable,
            "nonreachable_classes": nonreachable,
        }
    return payload


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    payload = build_payload(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    row13 = payload["row_results"]["13"]
    print(
        json.dumps(
            {
                "output_path": str(args.output),
                "row13_reachable_classes": [item["class_id"] for item in row13["reachable_classes"]],
                "row13_num_reachable": len(row13["reachable_classes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
