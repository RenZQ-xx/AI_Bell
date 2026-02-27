from __future__ import annotations

import argparse
from collections import defaultdict
from functools import reduce
from itertools import permutations, product
from math import gcd
from pathlib import Path


def _normalize_row(row: tuple[int, ...]) -> tuple[int, ...]:
    g = reduce(gcd, (abs(x) for x in row if x != 0), 0)
    if g > 1:
        row = tuple(x // g for x in row)

    for x in row:
        if x != 0:
            if x < 0:
                row = tuple(-y for y in row)
            break
    return row


def _parse_hrep_rows(facets_path: Path) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    in_block = False
    with open(facets_path, "r") as f:
        it = iter(f)
        for line in it:
            s = line.strip()
            if s == "begin":
                in_block = True
                next(it, None)  # skip "***** N rational"
                continue
            if s == "end":
                break
            if (not in_block) or (not s) or s.startswith("*"):
                continue

            nums = tuple(int(x) for x in s.split())
            rows.append(_normalize_row(nums))
    return rows


def _basis_222():
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


def _basis_322():
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


def _build_transforms(num_parties: int, basis: list[tuple[tuple[int, int], ...]]):
    mon_to_idx = {tuple(sorted(m)): i for i, m in enumerate(basis)}
    transforms = []

    for perm in permutations(range(num_parties)):
        for swap_bits in product([0, 1], repeat=num_parties):
            for flip_bits in product([1, -1], repeat=2 * num_parties):
                idx_map = [0] * len(basis)
                sign_map = [1] * len(basis)
                for i, mon in enumerate(basis):
                    sign = 1
                    new_mon = []
                    for party, setting in mon:
                        sign *= flip_bits[party * 2 + setting]
                        party2 = perm[party]
                        setting2 = setting ^ swap_bits[party]
                        new_mon.append((party2, setting2))
                    idx_map[i] = mon_to_idx[tuple(sorted(new_mon))]
                    sign_map[i] = sign
                transforms.append((tuple(idx_map), tuple(sign_map)))
    return transforms


def _apply_transform(
    row: tuple[int, ...],
    idx_map: tuple[int, ...],
    sign_map: tuple[int, ...],
) -> tuple[int, ...]:
    b = row[0]
    coeffs = row[1:]
    new_coeffs = [0] * len(coeffs)
    for i, c in enumerate(coeffs):
        if c:
            new_coeffs[idx_map[i]] += c * sign_map[i]
    return _normalize_row((b, *new_coeffs))


def classify_facets(rows: list[tuple[int, ...]], scenario: str):
    if scenario == "222":
        num_parties = 2
        basis = _basis_222()
    elif scenario == "322":
        num_parties = 3
        basis = _basis_322()
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    transforms = _build_transforms(num_parties, basis)

    row_to_count: dict[tuple[int, ...], int] = defaultdict(int)
    for r in rows:
        row_to_count[r] += 1

    remaining = set(row_to_count.keys())
    classes = []
    class_id = 1
    while remaining:
        rep = next(iter(remaining))
        orbit = set()
        for idx_map, sign_map in transforms:
            r2 = _apply_transform(rep, idx_map, sign_map)
            if r2 in remaining:
                orbit.add(r2)

        size = sum(row_to_count[r] for r in orbit)
        classes.append(
            {
                "class_id": class_id,
                "size": size,
                "orbit_unique_facets": len(orbit),
                "representative": rep,
                "members": sorted(orbit),
            }
        )
        class_id += 1
        remaining -= orbit

    classes.sort(key=lambda x: (-x["size"], x["class_id"]))
    return classes, len(transforms)


def main():
    parser = argparse.ArgumentParser(
        description="Classify Bell facets into symmetry-equivalent classes."
    )
    parser.add_argument(
        "--scenario",
        choices=["222", "322"],
        required=True,
        help="Scenario type; selects basis and symmetry group.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to facets txt. Defaults to data/facets_<scenario>.txt",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path for class counts table.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=3,
        help="How many representative facets to print for each class.",
    )
    parser.add_argument(
        "--save-detail",
        type=Path,
        default=None,
        help="Optional detailed output (class sizes + representative inequalities).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    facets_path = (
        args.input
        if args.input is not None
        else project_root / "data" / f"facets_{args.scenario}.txt"
    )

    rows = _parse_hrep_rows(facets_path)
    classes, group_size = classify_facets(rows, args.scenario)

    total = sum(c["size"] for c in classes)
    print(f"input: {facets_path}")
    print(f"scenario: {args.scenario}")
    print(f"facets total: {len(rows)}")
    print(f"unique rows (normalized): {len(set(rows))}")
    print(f"symmetry group size used: {group_size}")
    print(f"num classes: {len(classes)}")
    print("")
    print("class_id\tsize")
    for i, c in enumerate(classes, 1):
        print(f"{i}\t{c['size']}")
    print(f"\ncheck sum: {total}")

    k = max(1, args.examples_per_class)
    print("")
    print(f"examples per class (k={k}):")
    for i, c in enumerate(classes, 1):
        print(f"\n[class {i}] size={c['size']}")
        for j, row in enumerate(c["members"][:k], 1):
            print(f"rep{j}: {' '.join(map(str, row))}")

    if args.save is not None:
        out = args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("class_id\tsize\n")
            for i, c in enumerate(classes, 1):
                f.write(f"{i}\t{c['size']}\n")
            f.write(f"TOTAL\t{total}\n")
        print(f"saved: {out}")

    if args.save_detail is not None:
        out = args.save_detail
        out.parent.mkdir(parents=True, exist_ok=True)
        k = max(1, args.examples_per_class)
        with open(out, "w") as f:
            f.write(f"input: {facets_path}\n")
            f.write(f"scenario: {args.scenario}\n")
            f.write(f"facets total: {len(rows)}\n")
            f.write(f"num classes: {len(classes)}\n")
            f.write(f"examples per class: {k}\n\n")
            for i, c in enumerate(classes, 1):
                f.write(f"[class {i}] size={c['size']}\n")
                for j, row in enumerate(c["members"][:k], 1):
                    f.write(f"rep{j}: {' '.join(map(str, row))}\n")
                f.write("\n")
        print(f"saved detail: {out}")


if __name__ == "__main__":
    main()
