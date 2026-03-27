from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aibell.tool import Points_322

EQ_TOL = 1e-9
PARALLEL_TOL = 1e-9


@dataclass
class FacetExample:
    class_id: int
    rep_id: int
    row: np.ndarray


def parse_examples(examples_path: Path) -> list[FacetExample]:
    examples: list[FacetExample] = []
    class_id: int | None = None

    with examples_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
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
                row = np.array([int(x) for x in rep_match.group(2).split()], dtype=np.int64)
                examples.append(FacetExample(class_id=class_id, rep_id=rep_id, row=row))

    return examples


def format_vector(vec: np.ndarray, decimals: int = 6) -> str:
    parts = []
    for value in vec:
        if abs(value) < 10 ** (-decimals):
            parts.append("0")
        else:
            parts.append(f"{value:.{decimals}f}")
    return "[" + ", ".join(parts) + "]"


def analyze_example(example: FacetExample, points: np.ndarray) -> dict[str, object]:
    intercept = int(example.row[0])
    normal = example.row[1:].astype(np.float64)

    values = intercept + points @ normal
    on_plane_mask = np.isclose(values, 0.0, atol=EQ_TOL)
    on_plane_indices = np.flatnonzero(on_plane_mask)
    on_plane_points = points[on_plane_indices]

    if len(on_plane_points) == 0:
        raise ValueError(f"class {example.class_id} rep {example.rep_id} has no on-plane vertices")

    mean_vector = np.mean(on_plane_points, axis=0)
    normal_norm = np.linalg.norm(normal)
    mean_norm = np.linalg.norm(mean_vector)

    if normal_norm < PARALLEL_TOL:
        raise ValueError(f"class {example.class_id} rep {example.rep_id} has zero normal")

    if mean_norm < PARALLEL_TOL:
        is_parallel = True
        cosine = np.nan
        orthogonal_residual_norm = 0.0
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
        "intercept": intercept,
        "normal": normal,
        "vertex_count": int(len(on_plane_indices)),
        "on_plane_indices": on_plane_indices.tolist(),
        "mean_vector": mean_vector,
        "mean_norm": float(mean_norm),
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
        class_results = [result for result in results if result["class_id"] == class_id]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the mean of on-plane vertices is parallel to the facet normal in 3-2-2."
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        help="Path to facet_classes_322_examples.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output report path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    examples_path = (
        args.examples
        if args.examples is not None
        else project_root / "data" / "facet_classes_322_examples.txt"
    )
    output_path = (
        args.output
        if args.output is not None
        else project_root / "data" / "facet_classes_322_mean_parallel_check.txt"
    )

    points = Points_322().astype(np.float64)
    examples = parse_examples(examples_path)
    results = [analyze_example(example, points) for example in examples]
    report = build_report(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    num_parallel = sum(bool(result["is_parallel"]) for result in results)
    print(f"examples file: {examples_path}")
    print(f"report saved: {output_path}")
    print(f"classes checked: {len({result['class_id'] for result in results})}")
    print(f"examples checked: {len(results)}")
    print(f"parallel examples: {num_parallel}/{len(results)}")


if __name__ == "__main__":
    main()
