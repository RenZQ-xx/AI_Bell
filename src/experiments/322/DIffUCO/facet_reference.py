from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from math import lcm
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def normalize_row(row: Tuple[int, ...]) -> Tuple[int, ...]:
    from functools import reduce
    from math import gcd

    g = reduce(gcd, (abs(x) for x in row if x != 0), 0)
    if g > 1:
        row = tuple(x // g for x in row)
    for x in row:
        if x != 0:
            if x < 0:
                row = tuple(-y for y in row)
            break
    return row


def parse_hrep_rows(facets_path: Path) -> List[Tuple[int, ...]]:
    rows: List[Tuple[int, ...]] = []
    in_block = False
    with open(facets_path, "r", encoding="utf-8") as handle:
        it = iter(handle)
        for line in it:
            s = line.strip()
            if s == "begin":
                in_block = True
                next(it, None)
                continue
            if s == "end":
                break
            if (not in_block) or (not s) or s.startswith("*"):
                continue
            rows.append(normalize_row(tuple(int(x) for x in s.split())))
    return rows


def basis_322() -> List[Tuple[Tuple[int, int], ...]]:
    return [
        ((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),), ((2, 0),), ((2, 1),),
        ((0, 0), (1, 0)), ((0, 0), (1, 1)), ((0, 1), (1, 0)), ((0, 1), (1, 1)),
        ((0, 0), (2, 0)), ((0, 0), (2, 1)), ((0, 1), (2, 0)), ((0, 1), (2, 1)),
        ((1, 0), (2, 0)), ((1, 0), (2, 1)), ((1, 1), (2, 0)), ((1, 1), (2, 1)),
        ((0, 0), (1, 0), (2, 0)), ((0, 0), (1, 1), (2, 0)), ((0, 1), (1, 0), (2, 0)), ((0, 1), (1, 1), (2, 0)),
        ((0, 0), (1, 0), (2, 1)), ((0, 0), (1, 1), (2, 1)), ((0, 1), (1, 0), (2, 1)), ((0, 1), (1, 1), (2, 1)),
    ]


@lru_cache(maxsize=1)
def build_transforms():
    from itertools import permutations, product

    num_parties = 3
    basis = basis_322()
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


def apply_transform(row: Tuple[int, ...], idx_map: Tuple[int, ...], sign_map: Tuple[int, ...]) -> Tuple[int, ...]:
    b = row[0]
    coeffs = row[1:]
    new_coeffs = [0] * len(coeffs)
    for i, c in enumerate(coeffs):
        if c:
            new_coeffs[idx_map[i]] += c * sign_map[i]
    return normalize_row((b, *new_coeffs))


def orbit_members(row: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    members = {apply_transform(row, idx_map, sign_map) for idx_map, sign_map in build_transforms()}
    return sorted(members)


@lru_cache(maxsize=4096)
def _orbit_members_cached(row: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    normalized = normalize_row(row)
    return tuple(orbit_members(normalized))


def canonicalize_row(row: Tuple[int, ...]) -> Tuple[int, ...]:
    normalized = normalize_row(row)
    return min(_orbit_members_cached(normalized))


def classify_rows(rows: List[Tuple[int, ...]]):
    transforms = build_transforms()
    row_to_count: Dict[Tuple[int, ...], int] = defaultdict(int)
    for row in rows:
        row_to_count[row] += 1

    remaining = set(row_to_count.keys())
    classes = []
    class_id = 1
    while remaining:
        rep = next(iter(remaining))
        orbit = set()
        for idx_map, sign_map in transforms:
            transformed = apply_transform(rep, idx_map, sign_map)
            if transformed in remaining:
                orbit.add(transformed)
        size = sum(row_to_count[item] for item in orbit)
        classes.append(
            {
                "class_id": class_id,
                "size": size,
                "representative": rep,
                "members": sorted(orbit),
            }
        )
        remaining -= orbit
        class_id += 1
    classes.sort(key=lambda item: (-item["size"], item["class_id"]))
    for index, item in enumerate(classes, start=1):
        item["class_id"] = index
    return classes


def build_reference_database(facets_path: Path | None = None) -> Dict[str, object]:
    if facets_path is None:
        facets_path = PROJECT_ROOT / "data" / "facets_322.txt"
    rows = parse_hrep_rows(facets_path)
    classes = classify_rows(rows)
    row_to_matches: Dict[Tuple[int, ...], List[Dict[str, object]]] = defaultdict(list)
    canonical_row_to_info: Dict[Tuple[int, ...], Dict[str, object]] = {}
    class_summaries: List[Dict[str, object]] = []
    for class_info in classes:
        canonical_row = canonicalize_row(class_info["representative"])
        for row in class_info["members"]:
            row_to_matches[row].append(
                {
                    "class_id": class_info["class_id"],
                    "row": row,
                    "class_size": class_info["size"],
                    "canonical_row": canonical_row,
                }
            )
        canonical_row_to_info[canonical_row] = {
            "class_id": class_info["class_id"],
            "class_size": class_info["size"],
            "orbit_unique_rows": len(class_info["members"]),
            "canonical_row": canonical_row,
        }
        class_summaries.append(
            {
                "class_id": class_info["class_id"],
                "size": class_info["size"],
                "orbit_unique_rows": len(class_info["members"]),
                "canonical_row": list(canonical_row),
                "representative": list(class_info["representative"]),
            }
        )
    return {
        "classes": class_summaries,
        "row_to_matches": row_to_matches,
        "canonical_row_to_info": canonical_row_to_info,
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def integerize_plane(normal: List[float], offset: float, max_denominator: int = 12, ratio_tol: float = 2e-2) -> List[Dict[str, object]]:
    vector = np.asarray([offset, *normal], dtype=np.float64)
    candidates: Dict[Tuple[int, ...], Dict[str, object]] = {}
    significant = np.flatnonzero(np.abs(vector) > 1e-8)
    if significant.size == 0:
        return []

    for sign in (1.0, -1.0):
        signed = sign * vector
        for pivot in significant:
            base = signed[pivot]
            ratios = signed / base
            numerators: List[int] = []
            denominators: List[int] = []
            max_error = 0.0
            valid = True
            for value in ratios:
                if abs(value) <= 1e-8:
                    numerators.append(0)
                    denominators.append(1)
                    continue
                frac = Fraction(float(value)).limit_denominator(max_denominator)
                approx = frac.numerator / frac.denominator
                max_error = max(max_error, abs(value - approx))
                if abs(value - approx) > ratio_tol:
                    valid = False
                    break
                numerators.append(frac.numerator)
                denominators.append(frac.denominator)
            if not valid:
                continue
            scale = 1
            for denom in denominators:
                scale = lcm(scale, denom)
            row = tuple(int(num * (scale // den)) for num, den in zip(numerators, denominators))
            row = normalize_row(row)
            if all(value == 0 for value in row):
                continue
            direction_error = float(np.linalg.norm(_unit(np.asarray(row, dtype=np.float64)) - _unit(signed)))
            existing = candidates.get(row)
            payload = {
                "row": row,
                "max_ratio_error": max_error,
                "direction_error": direction_error,
                "pivot": int(pivot),
            }
            if existing is None or (direction_error, max_error) < (existing["direction_error"], existing["max_ratio_error"]):
                candidates[row] = payload
    return sorted(candidates.values(), key=lambda item: (item["direction_error"], item["max_ratio_error"]))


def classify_discovered_plane(normal: List[float], offset: float, reference: Dict[str, object]) -> Dict[str, object]:
    candidates = integerize_plane(normal, offset)
    for candidate in candidates:
        matches = reference["row_to_matches"].get(candidate["row"], [])
        if matches:
            class_ids = sorted({match["class_id"] for match in matches})
            canonical_row = canonicalize_row(candidate["row"])
            canonical_info = reference["canonical_row_to_info"].get(canonical_row, {})
            return {
                "tier": "exact_match",
                "matched_classes": class_ids,
                "num_row_matches": len(matches),
                "recovered_integer_row": list(candidate["row"]),
                "canonical_integer_row": list(canonical_row),
                "canonical_key": ",".join(str(value) for value in canonical_row),
                "orbit_size": int(canonical_info.get("class_size", len(matches))),
                "orbit_unique_rows": int(canonical_info.get("orbit_unique_rows", 0)),
                "direction_error": candidate["direction_error"],
                "max_ratio_error": candidate["max_ratio_error"],
            }
    if candidates:
        best = candidates[0]
        canonical_row = canonicalize_row(best["row"])
        orbit_size = len(_orbit_members_cached(normalize_row(best["row"])))
        return {
            "tier": "unknown_supporting_face",
            "matched_classes": [],
            "num_row_matches": 0,
            "recovered_integer_row": list(best["row"]),
            "canonical_integer_row": list(canonical_row),
            "canonical_key": ",".join(str(value) for value in canonical_row),
            "orbit_size": orbit_size,
            "orbit_unique_rows": orbit_size,
            "direction_error": best["direction_error"],
            "max_ratio_error": best["max_ratio_error"],
        }
    return {
        "tier": "unknown_supporting_face",
        "matched_classes": [],
        "num_row_matches": 0,
        "recovered_integer_row": None,
        "canonical_integer_row": None,
        "canonical_key": None,
        "orbit_size": None,
        "orbit_unique_rows": None,
        "direction_error": None,
        "max_ratio_error": None,
    }
