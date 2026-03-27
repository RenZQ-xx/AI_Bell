from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache, reduce
from math import gcd, lcm
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]


Row = Tuple[int, ...]
Monomial = Tuple[Tuple[int, int], ...]


def normalize_row(row: Sequence[int]) -> Row:
    row_tuple = tuple(int(x) for x in row)
    g = reduce(gcd, (abs(x) for x in row_tuple if x != 0), 0)
    if g > 1:
        row_tuple = tuple(x // g for x in row_tuple)
    for value in row_tuple:
        if value != 0:
            if value < 0:
                row_tuple = tuple(-x for x in row_tuple)
            break
    return row_tuple


def basis_422() -> List[Monomial]:
    return [
        ((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),), ((2, 0),), ((2, 1),), ((3, 0),), ((3, 1),),
        ((0, 0), (1, 0)), ((0, 0), (1, 1)), ((0, 1), (1, 0)), ((0, 1), (1, 1)),
        ((0, 0), (2, 0)), ((0, 0), (2, 1)), ((0, 1), (2, 0)), ((0, 1), (2, 1)),
        ((0, 0), (3, 0)), ((0, 0), (3, 1)), ((0, 1), (3, 0)), ((0, 1), (3, 1)),
        ((1, 0), (2, 0)), ((1, 0), (2, 1)), ((1, 1), (2, 0)), ((1, 1), (2, 1)),
        ((1, 0), (3, 0)), ((1, 0), (3, 1)), ((1, 1), (3, 0)), ((1, 1), (3, 1)),
        ((2, 0), (3, 0)), ((2, 0), (3, 1)), ((2, 1), (3, 0)), ((2, 1), (3, 1)),
        ((0, 0), (1, 0), (2, 0)), ((0, 0), (1, 1), (2, 0)), ((0, 1), (1, 0), (2, 0)), ((0, 1), (1, 1), (2, 0)),
        ((0, 0), (1, 0), (2, 1)), ((0, 0), (1, 1), (2, 1)), ((0, 1), (1, 0), (2, 1)), ((0, 1), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (3, 0)), ((0, 0), (1, 1), (3, 0)), ((0, 1), (1, 0), (3, 0)), ((0, 1), (1, 1), (3, 0)),
        ((0, 0), (1, 0), (3, 1)), ((0, 0), (1, 1), (3, 1)), ((0, 1), (1, 0), (3, 1)), ((0, 1), (1, 1), (3, 1)),
        ((0, 0), (2, 0), (3, 0)), ((0, 0), (2, 1), (3, 0)), ((0, 1), (2, 0), (3, 0)), ((0, 1), (2, 1), (3, 0)),
        ((0, 0), (2, 0), (3, 1)), ((0, 0), (2, 1), (3, 1)), ((0, 1), (2, 0), (3, 1)), ((0, 1), (2, 1), (3, 1)),
        ((1, 0), (2, 0), (3, 0)), ((1, 0), (2, 1), (3, 0)), ((1, 1), (2, 0), (3, 0)), ((1, 1), (2, 1), (3, 0)),
        ((1, 0), (2, 0), (3, 1)), ((1, 0), (2, 1), (3, 1)), ((1, 1), (2, 0), (3, 1)), ((1, 1), (2, 1), (3, 1)),
        ((0, 0), (1, 0), (2, 0), (3, 0)), ((0, 0), (1, 1), (2, 0), (3, 0)), ((0, 1), (1, 0), (2, 0), (3, 0)), ((0, 1), (1, 1), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (2, 1), (3, 0)), ((0, 0), (1, 1), (2, 1), (3, 0)), ((0, 1), (1, 0), (2, 1), (3, 0)), ((0, 1), (1, 1), (2, 1), (3, 0)),
        ((0, 0), (1, 0), (2, 0), (3, 1)), ((0, 0), (1, 1), (2, 0), (3, 1)), ((0, 1), (1, 0), (2, 0), (3, 1)), ((0, 1), (1, 1), (2, 0), (3, 1)),
        ((0, 0), (1, 0), (2, 1), (3, 1)), ((0, 0), (1, 1), (2, 1), (3, 1)), ((0, 1), (1, 0), (2, 1), (3, 1)), ((0, 1), (1, 1), (2, 1), (3, 1)),
    ]


def _generator_specs() -> List[Tuple[str, Dict[int, int], Dict[Tuple[int, int], int]]]:
    party_perm_identity = {0: 0, 1: 1, 2: 2, 3: 3}
    output_flip_identity = {(party, setting): 1 for party in range(4) for setting in range(2)}

    def permuted(mapping: Dict[int, int]) -> Dict[int, int]:
        full = dict(party_perm_identity)
        full.update(mapping)
        return full

    def output_flip(party: int, setting: int) -> Dict[Tuple[int, int], int]:
        mapping = dict(output_flip_identity)
        mapping[(party, setting)] = -1
        return mapping

    specs = [
        ("ABswap", permuted({0: 1, 1: 0}), output_flip_identity),
        ("ACswap", permuted({0: 2, 2: 0}), output_flip_identity),
        ("ADswap", permuted({0: 3, 3: 0}), output_flip_identity),
    ]
    for party in range(4):
        setting_swap = {(p, s): (p, s ^ 1) if p == party else (p, s) for p in range(4) for s in range(2)}
        specs.append((f"FlipIn_{'ABCD'[party]}", permuted({}), {k: 1 for k in setting_swap}))
    for party in range(4):
        for setting in range(2):
            specs.append((f"FlipOut_{'ABCD'[party]}{setting}", permuted({}), output_flip(party, setting)))
    return specs


@lru_cache(maxsize=1)
def _generator_transforms() -> Tuple[Tuple[str, Tuple[int, ...], Tuple[int, ...]], ...]:
    basis = basis_422()
    mon_to_idx = {tuple(sorted(mon)): index for index, mon in enumerate(basis)}
    transforms = []
    for name, party_perm, output_signs in _generator_specs():
        idx_map = [0] * len(basis)
        sign_map = [1] * len(basis)
        is_flipin = name.startswith("FlipIn_")
        flipin_party = "ABCD".index(name[-1]) if is_flipin else -1
        for index, mon in enumerate(basis):
            new_mon = []
            sign = 1
            for party, setting in mon:
                new_party = party_perm[party]
                new_setting = setting ^ 1 if is_flipin and party == flipin_party else setting
                sign *= output_signs[(party, setting)]
                new_mon.append((new_party, new_setting))
            idx_map[index] = mon_to_idx[tuple(sorted(new_mon))]
            sign_map[index] = sign
        transforms.append((name, tuple(idx_map), tuple(sign_map)))
    return tuple(transforms)


def apply_transform(row: Sequence[int], idx_map: Tuple[int, ...], sign_map: Tuple[int, ...]) -> Row:
    row = normalize_row(row)
    bias = row[0]
    coeffs = row[1:]
    new_coeffs = [0] * len(coeffs)
    for index, coefficient in enumerate(coeffs):
        if coefficient:
            new_coeffs[idx_map[index]] += coefficient * sign_map[index]
    return normalize_row((bias, *new_coeffs))


def orbit_members(row: Sequence[int]) -> List[Row]:
    generators = _generator_transforms()
    start = normalize_row(row)
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        for _, idx_map, sign_map in generators:
            candidate = apply_transform(current, idx_map, sign_map)
            if candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
    return sorted(seen)


@lru_cache(maxsize=4096)
def _orbit_members_cached(row: Row) -> Tuple[Row, ...]:
    return tuple(orbit_members(row))


def canonicalize_row(row: Sequence[int]) -> Row:
    normalized = normalize_row(row)
    return min(_orbit_members_cached(normalized))


def parse_hrep_rows(facets_path: Path) -> List[Row]:
    rows: List[Row] = []
    with open(facets_path, "r", encoding="utf-8") as handle:
        in_block = False
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


def classify_rows(rows: Iterable[Row]) -> List[Dict[str, object]]:
    row_to_count: Dict[Row, int] = defaultdict(int)
    for row in rows:
        row_to_count[normalize_row(row)] += 1

    remaining = set(row_to_count.keys())
    classes = []
    class_id = 1
    while remaining:
        representative = next(iter(remaining))
        orbit = set(_orbit_members_cached(representative))
        members = sorted(orbit & remaining)
        size = sum(row_to_count[item] for item in members)
        canonical_row = canonicalize_row(representative)
        classes.append(
            {
                "class_id": class_id,
                "size": size,
                "orbit_unique_rows": len(orbit),
                "canonical_row": canonical_row,
                "members": members,
            }
        )
        remaining -= orbit
        class_id += 1
    classes.sort(key=lambda item: (-item["size"], item["class_id"]))
    for index, item in enumerate(classes, start=1):
        item["class_id"] = index
    return classes


def build_reference_database(facets_path: Path | None = None) -> Dict[str, object] | None:
    if facets_path is None:
        candidate = PROJECT_ROOT / "data" / "facets_422.txt"
        if not candidate.exists():
            return None
        facets_path = candidate
    if not Path(facets_path).exists():
        return None

    rows = parse_hrep_rows(Path(facets_path))
    classes = classify_rows(rows)
    row_to_matches: Dict[Row, List[Dict[str, object]]] = defaultdict(list)
    canonical_row_to_info: Dict[Row, Dict[str, object]] = {}
    class_summaries: List[Dict[str, object]] = []
    for class_info in classes:
        canonical_row = class_info["canonical_row"]
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
            "orbit_unique_rows": class_info["orbit_unique_rows"],
            "canonical_row": canonical_row,
        }
        class_summaries.append(
            {
                "class_id": class_info["class_id"],
                "size": class_info["size"],
                "orbit_unique_rows": class_info["orbit_unique_rows"],
                "canonical_row": list(canonical_row),
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


def integerize_plane(normal: List[float], offset: float, max_denominator: int = 16, ratio_tol: float = 2e-2) -> List[Dict[str, object]]:
    vector = np.asarray([offset, *normal], dtype=np.float64)
    candidates: Dict[Row, Dict[str, object]] = {}
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
            for denominator in denominators:
                scale = lcm(scale, denominator)
            row = normalize_row(int(num * (scale // den)) for num, den in zip(numerators, denominators))
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


def classify_discovered_plane(normal: List[float], offset: float, reference: Dict[str, object] | None) -> Dict[str, object]:
    candidates = integerize_plane(normal, offset)
    if reference is not None:
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
                    "orbit_unique_rows": int(canonical_info.get("orbit_unique_rows", len(_orbit_members_cached(canonical_row)))),
                    "direction_error": float(candidate["direction_error"]),
                    "max_ratio_error": float(candidate["max_ratio_error"]),
                }

    if candidates:
        best = candidates[0]
        canonical_row = canonicalize_row(best["row"])
        orbit = _orbit_members_cached(canonical_row)
        return {
            "tier": "unknown_supporting_face",
            "matched_classes": [],
            "num_row_matches": 0,
            "recovered_integer_row": list(best["row"]),
            "canonical_integer_row": list(canonical_row),
            "canonical_key": ",".join(str(value) for value in canonical_row),
            "orbit_size": len(orbit),
            "orbit_unique_rows": len(orbit),
            "direction_error": float(best["direction_error"]),
            "max_ratio_error": float(best["max_ratio_error"]),
        }

    return {
        "tier": "candidate_supporting_face",
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
