#!/usr/bin/env python3
"""Classify Bell facet inequalities into symmetry-equivalent classes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import reduce
from itertools import permutations, product
from math import gcd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_row(row: tuple[int, ...]) -> tuple[int, ...]:
    divisor = reduce(gcd, (abs(value) for value in row if value != 0), 0)
    if divisor > 1:
        row = tuple(value // divisor for value in row)
    for value in row:
        if value != 0:
            return tuple(-item for item in row) if value < 0 else row
    return row


def parse_hrep_rows(facets_path: Path) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    in_block = False
    with facets_path.open("r", encoding="utf-8") as handle:
        iterator = iter(handle)
        for line in iterator:
            stripped = line.strip()
            if stripped == "begin":
                in_block = True
                next(iterator, None)
                continue
            if stripped == "end":
                break
            if (not in_block) or (not stripped) or stripped.startswith("*"):
                continue
            rows.append(normalize_row(tuple(int(value) for value in stripped.split())))
    return rows


def basis_222() -> list[tuple[tuple[int, int], ...]]:
    return [
        ((0, 0),),
        ((0, 1),),
        ((1, 0),),
        ((1, 1),),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
        ((0, 1), (1, 1)),
    ]


def basis_322() -> list[tuple[tuple[int, int], ...]]:
    return [
        ((0, 0),),
        ((0, 1),),
        ((1, 0),),
        ((1, 1),),
        ((2, 0),),
        ((2, 1),),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 0), (2, 0)),
        ((0, 0), (2, 1)),
        ((0, 1), (2, 0)),
        ((0, 1), (2, 1)),
        ((1, 0), (2, 0)),
        ((1, 0), (2, 1)),
        ((1, 1), (2, 0)),
        ((1, 1), (2, 1)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 1), (2, 0)),
        ((0, 1), (1, 0), (2, 0)),
        ((0, 1), (1, 1), (2, 0)),
        ((0, 0), (1, 0), (2, 1)),
        ((0, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 1)),
    ]


def build_transforms(num_parties: int, basis: list[tuple[tuple[int, int], ...]]):
    monomial_to_index = {tuple(sorted(monomial)): index for index, monomial in enumerate(basis)}
    transforms = []
    for party_perm in permutations(range(num_parties)):
        for setting_swaps in product([0, 1], repeat=num_parties):
            for sign_flips in product([1, -1], repeat=2 * num_parties):
                index_map = [0] * len(basis)
                sign_map = [1] * len(basis)
                for index, monomial in enumerate(basis):
                    sign = 1
                    new_monomial = []
                    for party, setting in monomial:
                        sign *= sign_flips[party * 2 + setting]
                        new_party = party_perm[party]
                        new_setting = setting ^ setting_swaps[party]
                        new_monomial.append((new_party, new_setting))
                    index_map[index] = monomial_to_index[tuple(sorted(new_monomial))]
                    sign_map[index] = sign
                transforms.append((tuple(index_map), tuple(sign_map)))
    return transforms


def apply_transform(
    row: tuple[int, ...],
    index_map: tuple[int, ...],
    sign_map: tuple[int, ...],
) -> tuple[int, ...]:
    bias = row[0]
    coeffs = row[1:]
    new_coeffs = [0] * len(coeffs)
    for index, coeff in enumerate(coeffs):
        if coeff:
            new_coeffs[index_map[index]] += coeff * sign_map[index]
    return normalize_row((bias, *new_coeffs))


def classify_facets(rows: list[tuple[int, ...]], scenario: str):
    if scenario == "222":
        num_parties = 2
        basis = basis_222()
    elif scenario == "322":
        num_parties = 3
        basis = basis_322()
    else:
        raise ValueError(f"unsupported scenario: {scenario}")

    transforms = build_transforms(num_parties, basis)
    row_to_count: dict[tuple[int, ...], int] = defaultdict(int)
    for row in rows:
        row_to_count[row] += 1

    remaining = set(row_to_count)
    classes = []
    class_id = 1
    while remaining:
        representative = next(iter(remaining))
        orbit = set()
        for index_map, sign_map in transforms:
            transformed = apply_transform(representative, index_map, sign_map)
            if transformed in remaining:
                orbit.add(transformed)
        size = sum(row_to_count[row] for row in orbit)
        classes.append(
            {
                "class_id": class_id,
                "size": size,
                "orbit_unique_facets": len(orbit),
                "representative": representative,
                "members": sorted(orbit),
            }
        )
        class_id += 1
        remaining -= orbit

    classes.sort(key=lambda item: (-item["size"], item["class_id"]))
    return classes, len(transforms)


def write_tsv(classes: list[dict[str, object]], output: Path) -> None:
    total = sum(int(item["size"]) for item in classes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\r\n") as handle:
        handle.write("class_id\tsize\n")
        for index, item in enumerate(classes, 1):
            handle.write(f"{index}\t{item['size']}\n")
        handle.write(f"TOTAL\t{total}\n")


def write_examples(
    classes: list[dict[str, object]],
    output: Path,
    *,
    facets_path: Path,
    scenario: str,
    facets_total: int,
    examples_per_class: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    k = max(1, int(examples_per_class))
    with output.open("w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(f"input: {facets_path}\n")
        handle.write(f"scenario: {scenario}\n")
        handle.write(f"facets total: {facets_total}\n")
        handle.write(f"num classes: {len(classes)}\n")
        handle.write(f"examples per class: {k}\n\n")
        for index, item in enumerate(classes, 1):
            handle.write(f"[class {index}] size={item['size']}\n")
            members = item["members"]
            assert isinstance(members, list)
            for rep_index, row in enumerate(members[:k], 1):
                handle.write(f"rep{rep_index}: {' '.join(map(str, row))}\n")
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify Bell facets into symmetry classes.")
    parser.add_argument("--scenario", choices=["222", "322"], default="322")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "facets_322.txt")
    parser.add_argument("--save", type=Path, default=PROJECT_ROOT / "data" / "facet_classes_322.tsv")
    parser.add_argument(
        "--save-detail",
        type=Path,
        default=PROJECT_ROOT / "data" / "facet_classes_322_examples.txt",
    )
    parser.add_argument("--examples-per-class", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = parse_hrep_rows(args.input)
    classes, group_size = classify_facets(rows, args.scenario)
    total = sum(int(item["size"]) for item in classes)
    print(f"input: {args.input}")
    print(f"scenario: {args.scenario}")
    print(f"facets total: {len(rows)}")
    print(f"unique rows normalized: {len(set(rows))}")
    print(f"symmetry group size used: {group_size}")
    print(f"num classes: {len(classes)}")
    print(f"check sum: {total}")
    write_tsv(classes, args.save)
    write_examples(
        classes,
        args.save_detail,
        facets_path=args.input,
        scenario=args.scenario,
        facets_total=len(rows),
        examples_per_class=args.examples_per_class,
    )
    print(f"saved: {args.save}")
    print(f"saved detail: {args.save_detail}")


if __name__ == "__main__":
    main()
