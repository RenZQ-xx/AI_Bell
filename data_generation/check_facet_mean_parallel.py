#!/usr/bin/env python3
"""Check whether each representative facet's on-plane mean is parallel to its normal."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES = PROJECT_ROOT / "data" / "facet_classes_322_examples.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "facet_classes_322_mean_parallel_check.txt"
EQ_TOL = 1e-9
PARALLEL_TOL = 1e-9


@dataclass(frozen=True)
class FacetExample:
    class_id: int
    rep_id: int
    row: np.ndarray


def points_322() -> np.ndarray:
    rows: list[list[int]] = []
    for a0 in (1, -1):
        for a1 in (1, -1):
            for b0 in (1, -1):
                for b1 in (1, -1):
                    for c0 in (1, -1):
                        for c1 in (1, -1):
                            rows.append(
                                [
                                    a0,
                                    a1,
                                    b0,
                                    b1,
                                    c0,
                                    c1,
                                    a0 * b0,
                                    a0 * b1,
                                    a1 * b0,
                                    a1 * b1,
                                    a0 * c0,
                                    a0 * c1,
                                    a1 * c0,
                                    a1 * c1,
                                    b0 * c0,
                                    b0 * c1,
                                    b1 * c0,
                                    b1 * c1,
                                    a0 * b0 * c0,
                                    a0 * b1 * c0,
                                    a1 * b0 * c0,
                                    a1 * b1 * c0,
                                    a0 * b0 * c1,
                                    a0 * b1 * c1,
                                    a1 * b0 * c1,
                                    a1 * b1 * c1,
                                ]
                            )
    return np.asarray(rows, dtype=np.float64)


def parse_examples(examples_path: Path) -> list[FacetExample]:
    examples: list[FacetExample] = []
    class_id: int | None = None
    with examples_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            class_match = re.match(r"\[class\s+(\d+)\]", line)
            if class_match:
                class_id = int(class_match.group(1))
                continue
            rep_match = re.match(r"rep(\d+):\s+(.+)", line)
            if rep_match and class_id is not None:
                rep_id = int(rep_match.group(1))
                row = np.array([int(value) for value in rep_match.group(2).split()], dtype=np.int64)
                examples.append(FacetExample(class_id=class_id, rep_id=rep_id, row=row))
    return examples


def format_vector(vec: np.ndarray, decimals: int = 6) -> str:
    parts = ["0" if abs(value) < 10 ** (-decimals) else f"{value:.{decimals}f}" for value in vec]
    return "[" + ", ".join(parts) + "]"


def analyze_example(example: FacetExample, points: np.ndarray) -> dict[str, object]:
    intercept = int(example.row[0])
    normal = example.row[1:].astype(np.float64)
    values = intercept + points @ normal
    on_plane_indices = np.flatnonzero(np.isclose(values, 0.0, atol=EQ_TOL))
    on_plane_points = points[on_plane_indices]
    if len(on_plane_points) == 0:
        raise ValueError(f"class {example.class_id} rep {example.rep_id} has no on-plane vertices")

    mean_vector = np.mean(on_plane_points, axis=0)
    normal_norm = np.linalg.norm(normal)
    mean_norm = np.linalg.norm(mean_vector)
    if normal_norm < PARALLEL_TOL:
        raise ValueError(f"class {example.class_id} rep {example.rep_id} has zero normal")

    if mean_norm < PARALLEL_TOL:
        cosine = np.nan
        orthogonal_residual_norm = 0.0
        is_parallel = True
    else:
        unit_normal = normal / normal_norm
        projection = np.dot(mean_vector, unit_normal) * unit_normal
        orthogonal_residual = mean_vector - projection
        orthogonal_residual_norm = float(np.linalg.norm(orthogonal_residual))
        cosine = float(np.dot(mean_vector, normal) / (mean_norm * normal_norm))
        is_parallel = orthogonal_residual_norm <= PARALLEL_TOL

    return {
        "class_id": example.class_id,
        "rep_id": example.rep_id,
        "normal": normal,
        "vertex_count": int(len(on_plane_indices)),
        "on_plane_indices": on_plane_indices.tolist(),
        "mean_vector": mean_vector,
        "cosine": cosine,
        "orthogonal_residual_norm": orthogonal_residual_norm,
        "is_parallel": is_parallel,
    }


def build_report(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    class_ids = sorted({int(result["class_id"]) for result in results})
    all_parallel = all(bool(result["is_parallel"]) for result in results)
    lines.append("3-2-2 facet check: mean vector on hyperplane vs hyperplane normal")
    lines.append(f"classes checked: {len(class_ids)}")
    lines.append(f"examples checked: {len(results)}")
    lines.append(f"all examples parallel: {all_parallel}")
    lines.append("")
    for class_id in class_ids:
        class_results = [result for result in results if int(result["class_id"]) == class_id]
        class_parallel = all(bool(result["is_parallel"]) for result in class_results)
        lines.append(f"[class {class_id}] all_reps_parallel={class_parallel}")
        for result in class_results:
            cosine = result["cosine"]
            cosine_str = "nan" if np.isnan(cosine) else f"{cosine:.12f}"
            lines.append(
                "  "
                f"rep{result['rep_id']}: "
                f"vertices={result['vertex_count']}, "
                f"parallel={result['is_parallel']}, "
                f"cosine={cosine_str}, "
                f"orth_residual={result['orthogonal_residual_norm']:.12e}"
            )
            lines.append(f"    normal      = {format_vector(result['normal'])}")
            lines.append(f"    mean_vector = {format_vector(result['mean_vector'])}")
            lines.append(f"    on_plane_indices = {result['on_plane_indices']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether on-plane vertex means are parallel to facet normals."
    )
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = parse_examples(args.examples)
    points = points_322()
    results = [analyze_example(example, points) for example in examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(results), encoding="utf-8")
    print(f"examples file: {args.examples}")
    print(f"report saved: {args.output}")
    print(f"classes checked: {len({result['class_id'] for result in results})}")
    print(f"examples checked: {len(results)}")
    print(f"parallel examples: {sum(bool(result['is_parallel']) for result in results)}/{len(results)}")


if __name__ == "__main__":
    main()
