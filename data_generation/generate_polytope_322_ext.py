#!/usr/bin/env python3
"""Generate the Bell 3-2-2 V-representation file for lrslib."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "polytope_322.ext"


def iter_points_322() -> list[tuple[int, ...]]:
    points: list[tuple[int, ...]] = []
    for a0 in (1, -1):
        for a1 in (1, -1):
            for b0 in (1, -1):
                for b1 in (1, -1):
                    for c0 in (1, -1):
                        for c1 in (1, -1):
                            points.append(
                                (
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
                                )
                            )
    return sorted(set(points))


def write_ext(points: list[tuple[int, ...]], output: Path) -> None:
    if not points:
        raise ValueError("no points generated")
    row_count = len(points)
    dim = len(points[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\r\n") as handle:
        handle.write("my_polytope_26d_322\n")
        handle.write("V-representation\n")
        handle.write("begin\n")
        handle.write(f"{row_count} {dim + 1} rational\n")
        for point in points:
            handle.write("1 " + " ".join(str(value) for value in point) + "\n")
        handle.write("end\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data/polytope_322.ext for lrslib.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = iter_points_322()
    write_ext(points, args.output)
    print(f"saved: {args.output}")
    print(f"points: {len(points)}")
    print(f"dimension: {len(points[0])}")


if __name__ == "__main__":
    main()
