from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.append(str(CURRENT_DIR))
    from facet_reference import build_reference_database
else:
    from .facet_reference import build_reference_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track coverage of discovered 3-2-2 facets against the 46 known classes.")
    parser.add_argument("--discovered", type=Path, required=True, help="Path to Geometric-DiffUCO inference JSON output.")
    parser.add_argument("--reference", type=Path, default=None, help="Path to the known 3-2-2 facet H-representation file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for the coverage report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = build_reference_database(args.reference)
    payload = json.loads(args.discovered.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    matched_class_ids = sorted({class_id for item in results for class_id in item.get("matched_classes", [])})
    class_hit_counter = Counter(class_id for item in results for class_id in item.get("matched_classes", []))
    canonical_hit_counter = Counter(item.get("canonical_key") for item in results if item.get("canonical_key") is not None)
    all_class_ids = [item["class_id"] for item in reference["classes"]]
    report = {
        "discovered_file": str(args.discovered),
        "num_discovered_facets": len(results),
        "matched_class_ids": matched_class_ids,
        "num_matched_classes": len(matched_class_ids),
        "num_canonical_orbits": len(canonical_hit_counter),
        "num_total_classes": len(all_class_ids),
        "coverage_ratio": len(matched_class_ids) / len(all_class_ids) if all_class_ids else 0.0,
        "uncovered_class_ids": [class_id for class_id in all_class_ids if class_id not in set(matched_class_ids)],
        "class_hit_counter": dict(sorted(class_hit_counter.items())),
        "canonical_orbit_hit_counter": dict(sorted(canonical_hit_counter.items())),
        "matched_facets": [
            {
                "facet": idx + 1,
                "cardinality": item["cardinality"],
                "matched_classes": item.get("matched_classes", []),
                "canonical_integer_row": item.get("canonical_integer_row"),
                "canonical_key": item.get("canonical_key"),
                "orbit_size": item.get("orbit_size"),
                "tier": item.get("tier", "unknown"),
                "recovered_integer_row": item.get("recovered_integer_row"),
                "direction_error": item.get("direction_error"),
                "max_ratio_error": item.get("max_ratio_error"),
                "indices": item["indices"],
            }
            for idx, item in enumerate(results)
            if item.get("matched_classes")
        ],
        "unmatched_facets": [
            {
                "facet": idx + 1,
                "cardinality": item["cardinality"],
                "canonical_integer_row": item.get("canonical_integer_row"),
                "canonical_key": item.get("canonical_key"),
                "orbit_size": item.get("orbit_size"),
                "tier": item.get("tier", "unknown"),
                "recovered_integer_row": item.get("recovered_integer_row"),
                "direction_error": item.get("direction_error"),
                "max_ratio_error": item.get("max_ratio_error"),
                "indices": item["indices"],
            }
            for idx, item in enumerate(results)
            if not item.get("matched_classes")
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
