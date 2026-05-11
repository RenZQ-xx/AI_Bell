from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baseline.bell322 import generate_bell322_points
from baseline.facet_validator import FacetValidator
from baseline.reference_classes import (
    DEFAULT_EXAMPLES_PATH,
    parse_example_rows,
    support_mask_from_row,
)
from baseline.support_masks import vertex_mask_to_indices


@dataclass(frozen=True)
class GeometryCheckConfig:
    """Inputs for the baseline geometry sanity check."""

    examples_path: Path = DEFAULT_EXAMPLES_PATH
    include_details: bool = False


def validate_geometry(config: GeometryCheckConfig | None = None) -> dict[str, Any]:
    """Validate the migrated geometry, point table, and exact-class labels.

    This is intentionally a check, not a search experiment.  It verifies that:

    - the Bell 3-2-2 deterministic point table has the expected 64 x 26 shape;
    - every representative row in `facet_classes_322_examples.txt` induces a
      valid supporting facet;
    - the fitted hyperplane is classified back to the expected exact class.
    """
    cfg = GeometryCheckConfig() if config is None else config
    points = generate_bell322_points()
    example_rows = parse_example_rows(cfg.examples_path)
    validator = FacetValidator(points=points)

    rows_checked = 0
    exact_matches = 0
    invalid_facets: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    classes_seen: set[int] = set()

    for class_id in sorted(example_rows):
        classes_seen.add(class_id)
        for rep_index, row in sorted(example_rows[class_id].items()):
            rows_checked += 1
            mask = support_mask_from_row(row)
            indices = vertex_mask_to_indices(mask)
            label = validator.validate_indices(indices)
            expected = f"exact:class{class_id}"
            item = {
                "class_id": class_id,
                "rep_index": rep_index,
                "expected": expected,
                "label": label.label,
                "cardinality": label.validation.cardinality,
                "affine_rank": label.validation.affine_rank,
                "codimension": label.validation.codimension,
                "coplanar": label.validation.coplanar,
                "supporting": label.validation.supporting,
            }
            if not label.validation.valid:
                invalid_facets.append({**item, "reason": label.validation.reason})
            elif label.label != expected:
                mismatches.append(item)
            else:
                exact_matches += 1
            if cfg.include_details:
                details.append(item)

    class_ids = sorted(classes_seen)
    missing_classes = [class_id for class_id in range(1, 47) if class_id not in classes_seen]
    passed = (
        tuple(points.shape) == (64, 26)
        and len(class_ids) == 46
        and rows_checked > 0
        and exact_matches == rows_checked
        and not invalid_facets
        and not mismatches
        and not missing_classes
    )

    report: dict[str, Any] = {
        "passed": passed,
        "points_shape": list(points.shape),
        "points_dtype": str(points.dtype),
        "points_are_finite": bool(np.isfinite(points).all()),
        "classes_seen": class_ids,
        "missing_classes": missing_classes,
        "example_rows_checked": rows_checked,
        "exact_matches": exact_matches,
        "invalid_facets": invalid_facets,
        "mismatches": mismatches,
    }
    if cfg.include_details:
        report["details"] = details
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the migrated baseline geometry and exact facet labels."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=DEFAULT_EXAMPLES_PATH,
        help="Path to facet_classes_322_examples.txt.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include one compact record per checked representative row.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = validate_geometry(
        GeometryCheckConfig(examples_path=args.examples, include_details=args.details)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
