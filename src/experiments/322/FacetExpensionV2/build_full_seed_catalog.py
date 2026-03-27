from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    EXP322_DIR = CURRENT_DIR.parent
    DIFFUCO_DIR = EXP322_DIR / "DIffUCO"
    V1_DIR = EXP322_DIR / "FacetExpansionV1"
    LRS_DIR = EXP322_DIR / "lrs"
    sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR), str(LRS_DIR)]
    from geometry import generate_points_322
    from seed_bank import row_to_mask
    from facet_reference import parse_hrep_rows, classify_rows, canonicalize_row
else:
    from ..DIffUCO.geometry import generate_points_322
    from ..FacetExpansionV1.seed_bank import row_to_mask
    from ..DIffUCO.facet_reference import parse_hrep_rows, classify_rows, canonicalize_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the full 322 facet seed catalog with class assignments.")
    parser.add_argument("--facets", type=Path, default=Path("data/facets_322.txt"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = generate_points_322()
    rows = parse_hrep_rows(args.facets)
    classes = classify_rows(rows)
    row_to_class: dict[tuple[int, ...], dict[str, object]] = {}
    for class_info in classes:
        canonical = canonicalize_row(class_info["representative"])
        payload = {
            "class_id": int(class_info["class_id"]),
            "class_size": int(class_info["size"]),
            "orbit_unique_rows": int(len(class_info["members"])),
            "canonical_row": list(canonical),
        }
        for member in class_info["members"]:
            row_to_class[member] = payload

    summary_rows = []
    counts: dict[int, int] = {}
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            class_info = row_to_class[row]
            mask = row_to_mask(list(row), points=points)
            support_size = int(sum(mask))
            record = {
                "row_index": int(row_index),
                "class_id": int(class_info["class_id"]),
                "class_size": int(class_info["class_size"]),
                "orbit_unique_rows": int(class_info["orbit_unique_rows"]),
                "canonical_row": list(class_info["canonical_row"]),
                "row": list(row),
                "mask": [int(value) for value in mask],
                "support_size": support_size,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["class_id"]] = counts.get(record["class_id"], 0) + 1

    for class_id in sorted(counts):
        summary_rows.append({"class_id": int(class_id), "size": int(counts[class_id])})
    payload = {
        "facets_path": str(args.facets),
        "num_rows": len(rows),
        "num_classes": len(summary_rows),
        "class_sizes": summary_rows,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
