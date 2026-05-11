from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache, reduce
from math import gcd, lcm
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bell322 import generate_bell322_points


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACETS_PATH = PROJECT_ROOT / "data" / "facets_322.txt"
DEFAULT_EXAMPLES_PATH = PROJECT_ROOT / "data" / "facet_classes_322_examples.txt"

IntegerRow = tuple[int, ...]


def normalize_row(row: Sequence[int]) -> IntegerRow:
    """Normalize an integer inequality row up to gcd and global sign."""
    values = tuple(int(value) for value in row)
    divisor = reduce(gcd, (abs(value) for value in values if value != 0), 0)
    if divisor > 1:
        values = tuple(value // divisor for value in values)
    for value in values:
        if value != 0:
            if value < 0:
                values = tuple(-item for item in values)
            break
    return values


def parse_hrep_rows(facets_path: Path = DEFAULT_FACETS_PATH) -> list[IntegerRow]:
    """Parse normalized H-representation rows from `facets_322.txt`."""
    rows: list[IntegerRow] = []
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


def parse_example_rows(path: Path = DEFAULT_EXAMPLES_PATH) -> dict[int, dict[int, IntegerRow]]:
    """Parse the small representative-row file with three examples per class."""
    lines = path.read_text(encoding="utf-8").splitlines()
    class_rows: dict[int, dict[int, IntegerRow]] = {}
    current_class: int | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[class "):
            current_class = int(stripped.split("]")[0].split()[1])
            class_rows.setdefault(current_class, {})
        elif stripped.startswith("rep") and current_class is not None:
            head, payload = stripped.split(":", 1)
            rep_index = int(head.replace("rep", ""))
            class_rows[current_class][rep_index] = normalize_row(
                tuple(int(value) for value in payload.strip().split())
            )
    return class_rows


@lru_cache(maxsize=1)
def point_rows() -> tuple[tuple[int, ...], ...]:
    """Integer Bell 322 deterministic vertices as row tuples."""
    points = generate_bell322_points()
    return tuple(tuple(int(round(value)) for value in row) for row in points.tolist())


def support_mask_from_row(
    row: Sequence[int],
    rows: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Return the 64-bit support where an integer inequality is tight."""
    point_table = point_rows() if rows is None else rows
    normalized = normalize_row(row)
    bias = normalized[0]
    coeffs = normalized[1:]
    if len(coeffs) != 26:
        raise ValueError(f"expected 26 coefficients after bias, got {len(coeffs)}")
    mask = np.zeros(64, dtype=np.int64)
    for index, point in enumerate(point_table):
        value = bias + sum(coeff * coordinate for coeff, coordinate in zip(coeffs, point))
        if value == 0:
            mask[index] = 1
    return mask


def basis_322() -> list[tuple[tuple[int, int], ...]]:
    """Coordinate basis used by the 26 nonconstant Bell 322 row entries."""
    return [
        ((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),), ((2, 0),), ((2, 1),),
        ((0, 0), (1, 0)), ((0, 0), (1, 1)), ((0, 1), (1, 0)), ((0, 1), (1, 1)),
        ((0, 0), (2, 0)), ((0, 0), (2, 1)), ((0, 1), (2, 0)), ((0, 1), (2, 1)),
        ((1, 0), (2, 0)), ((1, 0), (2, 1)), ((1, 1), (2, 0)), ((1, 1), (2, 1)),
        ((0, 0), (1, 0), (2, 0)), ((0, 0), (1, 1), (2, 0)),
        ((0, 1), (1, 0), (2, 0)), ((0, 1), (1, 1), (2, 0)),
        ((0, 0), (1, 0), (2, 1)), ((0, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (2, 1)), ((0, 1), (1, 1), (2, 1)),
    ]


@lru_cache(maxsize=1)
def build_row_transforms() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Build row-coordinate transformations induced by Bell 322 symmetries."""
    from itertools import permutations, product

    num_parties = 3
    basis = basis_322()
    monomial_to_index = {tuple(sorted(monomial)): index for index, monomial in enumerate(basis)}
    transforms: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for party_perm in permutations(range(num_parties)):
        for swap_bits in product([0, 1], repeat=num_parties):
            for flip_bits in product([1, -1], repeat=2 * num_parties):
                index_map = [0] * len(basis)
                sign_map = [1] * len(basis)
                for index, monomial in enumerate(basis):
                    sign = 1
                    new_monomial = []
                    for party, setting in monomial:
                        sign *= flip_bits[party * 2 + setting]
                        new_party = party_perm[party]
                        new_setting = setting ^ swap_bits[party]
                        new_monomial.append((new_party, new_setting))
                    index_map[index] = monomial_to_index[tuple(sorted(new_monomial))]
                    sign_map[index] = sign
                transforms.append((tuple(index_map), tuple(sign_map)))
    return tuple(transforms)


def apply_row_transform(
    row: Sequence[int],
    index_map: Sequence[int],
    sign_map: Sequence[int],
) -> IntegerRow:
    """Apply one symmetry transform to an integer inequality row."""
    normalized = normalize_row(row)
    bias = normalized[0]
    coeffs = normalized[1:]
    new_coeffs = [0] * len(coeffs)
    for index, coeff in enumerate(coeffs):
        if coeff:
            new_coeffs[int(index_map[index])] += coeff * int(sign_map[index])
    return normalize_row((bias, *new_coeffs))


def orbit_members(row: Sequence[int]) -> list[IntegerRow]:
    """All normalized row-orbit members under Bell 322 symmetries."""
    normalized = normalize_row(row)
    members = {
        apply_row_transform(normalized, index_map, sign_map)
        for index_map, sign_map in build_row_transforms()
    }
    return sorted(members)


@lru_cache(maxsize=4096)
def _orbit_members_cached(row: IntegerRow) -> tuple[IntegerRow, ...]:
    return tuple(orbit_members(normalize_row(row)))


def canonicalize_row(row: Sequence[int]) -> IntegerRow:
    """Canonical representative of a row symmetry orbit."""
    normalized = normalize_row(row)
    return min(_orbit_members_cached(normalized))


def classify_rows(rows: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    """Group normalized H-rep rows into the 46 symmetry classes."""
    row_to_count: dict[IntegerRow, int] = defaultdict(int)
    for row in rows:
        row_to_count[normalize_row(row)] += 1

    remaining = set(row_to_count.keys())
    classes: list[dict[str, Any]] = []
    class_id = 1
    while remaining:
        representative = next(iter(remaining))
        orbit = set()
        for index_map, sign_map in build_row_transforms():
            transformed = apply_row_transform(representative, index_map, sign_map)
            if transformed in remaining:
                orbit.add(transformed)
        size = sum(row_to_count[item] for item in orbit)
        classes.append(
            {
                "class_id": class_id,
                "size": size,
                "representative": representative,
                "members": sorted(orbit),
            }
        )
        remaining -= orbit
        class_id += 1

    classes.sort(key=lambda item: (-int(item["size"]), int(item["class_id"])))
    for index, item in enumerate(classes, start=1):
        item["class_id"] = index
    return classes


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


@lru_cache(maxsize=1)
def build_reference_database(facets_path: Path = DEFAULT_FACETS_PATH) -> dict[str, Any]:
    """Build the exact-class reference database from all known facet rows."""
    rows = parse_hrep_rows(facets_path)
    classes = classify_rows(rows)
    row_to_matches: dict[IntegerRow, list[dict[str, Any]]] = defaultdict(list)
    canonical_row_to_info: dict[IntegerRow, dict[str, Any]] = {}
    unit_rows: list[dict[str, Any]] = []
    class_summaries: list[dict[str, Any]] = []

    for class_info in classes:
        class_id = int(class_info["class_id"])
        class_size = int(class_info["size"])
        representative = normalize_row(class_info["representative"])
        canonical_row = canonicalize_row(representative)
        members = tuple(normalize_row(row) for row in class_info["members"])
        for row in members:
            row_to_matches[row].append(
                {
                    "class_id": class_id,
                    "row": row,
                    "class_size": class_size,
                    "canonical_row": canonical_row,
                }
            )
            unit_rows.append(
                {
                    "row": row,
                    "class_id": class_id,
                    "canonical_row": canonical_row,
                    "unit_vector": _unit(np.asarray(row, dtype=np.float64)),
                }
            )
        canonical_row_to_info[canonical_row] = {
            "class_id": class_id,
            "class_size": class_size,
            "orbit_unique_rows": len(members),
            "canonical_row": canonical_row,
        }
        class_summaries.append(
            {
                "class_id": class_id,
                "size": class_size,
                "orbit_unique_rows": len(members),
                "canonical_row": list(canonical_row),
                "representative": list(representative),
            }
        )

    return {
        "classes": class_summaries,
        "row_to_matches": row_to_matches,
        "canonical_row_to_info": canonical_row_to_info,
        "unit_rows": unit_rows,
    }


def integerize_plane(
    normal: Sequence[float],
    offset: float,
    *,
    max_denominator: int = 12,
    ratio_tol: float = 2e-2,
) -> list[dict[str, Any]]:
    """Recover candidate integer rows from a fitted real hyperplane."""
    vector = np.asarray([offset, *normal], dtype=np.float64)
    candidates: dict[IntegerRow, dict[str, Any]] = {}
    significant = np.flatnonzero(np.abs(vector) > 1e-8)
    if significant.size == 0:
        return []

    for sign in (1.0, -1.0):
        signed = sign * vector
        for pivot in significant:
            base = signed[pivot]
            ratios = signed / base
            numerators: list[int] = []
            denominators: list[int] = []
            max_error = 0.0
            valid = True
            for value in ratios:
                if abs(value) <= 1e-8:
                    numerators.append(0)
                    denominators.append(1)
                    continue
                fraction = Fraction(float(value)).limit_denominator(max_denominator)
                approx = fraction.numerator / fraction.denominator
                max_error = max(max_error, abs(value - approx))
                if abs(value - approx) > ratio_tol:
                    valid = False
                    break
                numerators.append(fraction.numerator)
                denominators.append(fraction.denominator)
            if not valid:
                continue
            scale = 1
            for denominator in denominators:
                scale = lcm(scale, denominator)
            row = normalize_row(
                tuple(num * (scale // den) for num, den in zip(numerators, denominators))
            )
            if all(value == 0 for value in row):
                continue
            direction_error = float(np.linalg.norm(_unit(np.asarray(row, dtype=np.float64)) - _unit(signed)))
            payload = {
                "row": row,
                "max_ratio_error": max_error,
                "direction_error": direction_error,
                "pivot": int(pivot),
            }
            existing = candidates.get(row)
            if existing is None or (
                direction_error,
                max_error,
            ) < (
                float(existing["direction_error"]),
                float(existing["max_ratio_error"]),
            ):
                candidates[row] = payload
    return sorted(candidates.values(), key=lambda item: (item["direction_error"], item["max_ratio_error"]))


def _row_match_payload(
    candidate_row: IntegerRow,
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any] | None:
    matches = reference["row_to_matches"].get(candidate_row, [])
    if matches:
        class_ids = sorted({int(match["class_id"]) for match in matches})
        canonical_row = canonicalize_row(candidate_row)
        canonical_info = reference["canonical_row_to_info"].get(canonical_row, {})
        return {
            "tier": "exact_match",
            "matched_classes": class_ids,
            "num_row_matches": len(matches),
            "recovered_integer_row": list(candidate_row),
            "canonical_integer_row": list(canonical_row),
            "canonical_key": ",".join(str(value) for value in canonical_row),
            "orbit_size": int(canonical_info.get("class_size", len(matches))),
            "orbit_unique_rows": int(canonical_info.get("orbit_unique_rows", 0)),
            "direction_error": candidate["direction_error"],
            "max_ratio_error": candidate["max_ratio_error"],
            "match_source": "integer_row",
        }

    canonical_row = canonicalize_row(candidate_row)
    canonical_info = reference["canonical_row_to_info"].get(canonical_row)
    if canonical_info is not None:
        return {
            "tier": "exact_match",
            "matched_classes": [int(canonical_info["class_id"])],
            "num_row_matches": 0,
            "recovered_integer_row": list(candidate_row),
            "canonical_integer_row": list(canonical_row),
            "canonical_key": ",".join(str(value) for value in canonical_row),
            "orbit_size": int(canonical_info.get("class_size", 0)),
            "orbit_unique_rows": int(canonical_info.get("orbit_unique_rows", 0)),
            "direction_error": candidate["direction_error"],
            "max_ratio_error": candidate["max_ratio_error"],
            "match_source": "canonical_row",
        }
    return None


def _symmetry_direction_match(
    normal: Sequence[float],
    offset: float,
    reference: dict[str, Any],
    *,
    direction_tol: float = 2e-2,
) -> dict[str, Any] | None:
    plane_vector = _unit(np.asarray([offset, *normal], dtype=np.float64))
    best_item: dict[str, Any] | None = None
    best_error = float("inf")
    for item in reference.get("unit_rows", []):
        row_vector = item["unit_vector"]
        direction_error = float(
            min(
                np.linalg.norm(plane_vector - row_vector),
                np.linalg.norm(plane_vector + row_vector),
            )
        )
        if direction_error < best_error:
            best_error = direction_error
            best_item = item
    if best_item is None or best_error > direction_tol:
        return None
    canonical_row = tuple(best_item["canonical_row"])
    canonical_info = reference["canonical_row_to_info"].get(canonical_row, {})
    return {
        "tier": "exact_match",
        "matched_classes": [int(best_item["class_id"])],
        "num_row_matches": 0,
        "recovered_integer_row": None,
        "canonical_integer_row": list(canonical_row),
        "canonical_key": ",".join(str(value) for value in canonical_row),
        "orbit_size": int(canonical_info.get("class_size", 0)),
        "orbit_unique_rows": int(canonical_info.get("orbit_unique_rows", 0)),
        "direction_error": best_error,
        "max_ratio_error": None,
        "match_source": "symmetry_direction",
    }


def classify_hyperplane(
    normal: Sequence[float],
    offset: float,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a validated facet hyperplane against the known 46 classes."""
    database = build_reference_database() if reference is None else reference
    candidates = integerize_plane(normal, offset)
    for candidate in candidates:
        matched = _row_match_payload(candidate["row"], candidate, database)
        if matched is not None:
            return matched

    symmetry_matched = _symmetry_direction_match(normal, offset, database)
    if symmetry_matched is not None:
        return symmetry_matched

    if candidates:
        best = candidates[0]
        canonical_row = canonicalize_row(best["row"])
        orbit_size = len(_orbit_members_cached(normalize_row(best["row"])))
        return {
            "tier": "unknown_facet",
            "matched_classes": [],
            "num_row_matches": 0,
            "recovered_integer_row": list(best["row"]),
            "canonical_integer_row": list(canonical_row),
            "canonical_key": ",".join(str(value) for value in canonical_row),
            "orbit_size": orbit_size,
            "orbit_unique_rows": orbit_size,
            "direction_error": best["direction_error"],
            "max_ratio_error": best["max_ratio_error"],
            "match_source": "unmatched_integer_row",
        }

    return {
        "tier": "unknown_facet",
        "matched_classes": [],
        "num_row_matches": 0,
        "recovered_integer_row": None,
        "canonical_integer_row": None,
        "canonical_key": None,
        "orbit_size": None,
        "orbit_unique_rows": None,
        "direction_error": None,
        "max_ratio_error": None,
        "match_source": "no_integerization",
    }


def exact_label_from_match(match: dict[str, Any]) -> str:
    """Compact label used by the search/scorer layer."""
    if match.get("tier") != "exact_match":
        return "unknown"
    classes = [int(value) for value in match.get("matched_classes", [])]
    if not classes:
        return "unknown"
    return f"exact:class{classes[0]}"
