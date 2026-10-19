from __future__ import annotations

"""Rank, terminal facet validation, and post-terminal class lookup."""

import csv
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path
from typing import Sequence

import numpy as np

from .atlas import DEFAULT_HANDOFF_ROOT
from .problem import (
    AMBIENT_DIMENSION,
    CORRELATOR_BASIS,
    TARGET_AFFINE_RANK,
    Relabeling,
    deterministic_vertices,
    signed_coordinate_action,
    word_indices,
)


@dataclass(frozen=True)
class FacetResult:
    valid: bool
    reason: str
    selected_word: int
    support_word: int = 0
    affine_rank: int = -1
    h_row: tuple[int, ...] | None = None
    canonical_h_row: tuple[int, ...] | None = None
    class_id: str | None = None
    catalog_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "selected_word_hex": f"0x{self.selected_word:064x}",
            "support_word_hex": f"0x{self.support_word:064x}",
            "support_size": self.support_word.bit_count(),
            "affine_rank": self.affine_rank,
            "h_row": list(self.h_row) if self.h_row is not None else None,
            "canonical_h_row": (
                list(self.canonical_h_row) if self.canonical_h_row is not None else None
            ),
            "class_id": self.class_id,
            "catalog_index": self.catalog_index,
        }


class RankOracle:
    def __init__(self, points: np.ndarray | None = None, *, max_entries: int = 100_000):
        self.points = np.asarray(deterministic_vertices() if points is None else points, dtype=float)
        self.max_entries = int(max_entries)
        self.cache: OrderedDict[int, int] = OrderedDict({0: -1})
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def rank(self, word: int) -> int:
        cached = self.cache.get(int(word))
        if cached is not None:
            self.cache.move_to_end(int(word))
            self.hits += 1
            return cached
        indices = word_indices(word)
        if len(indices) <= 1:
            value = len(indices) - 1
        else:
            selected = self.points[indices]
            value = int(np.linalg.matrix_rank(selected[1:] - selected[0], tol=1e-8))
        self.cache[int(word)] = value
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)
            self.evictions += 1
        self.misses += 1
        return value

    def to_dict(self) -> dict[str, int]:
        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


def _primitive_integer_row(vector: np.ndarray, augmented: np.ndarray) -> tuple[int, ...]:
    scale = float(np.max(np.abs(vector)))
    if scale <= 1e-14:
        raise ValueError("zero null vector")
    fractions = [Fraction(float(value / scale)).limit_denominator(1_000_000) for value in vector]
    denominator = 1
    for value in fractions:
        denominator = math.lcm(denominator, value.denominator)
    integers = [value.numerator * (denominator // value.denominator) for value in fractions]
    divisor = 0
    for value in integers:
        divisor = math.gcd(divisor, abs(value))
    integers = [value // max(1, divisor) for value in integers]
    values = augmented @ np.asarray(integers, dtype=object)
    if any(int(value) < 0 for value in values):
        if all(int(value) <= 0 for value in values):
            integers = [-value for value in integers]
            values = -values
        else:
            raise ValueError("rationalized plane cuts the Bell polytope")
    if any(int(value) != 0 for value in values[np.flatnonzero(np.abs(augmented @ vector) <= 1e-7)]):
        raise ValueError("rationalized row changed its tight support")
    return tuple(map(int, integers))


@lru_cache(maxsize=1)
def _outcome_sign_actions() -> np.ndarray:
    actions = []
    for signs in product((-1, 1), repeat=8):
        actions.append(
            [
                math.prod(signs[2 * party + setting] for party, setting in correlator)
                for correlator in CORRELATOR_BASIS
            ]
        )
    return np.asarray(actions, dtype=np.int8)


@lru_cache(maxsize=1)
def _unsigned_coordinate_targets() -> tuple[np.ndarray, ...]:
    result = []
    for party_permutation in permutations(range(4)):
        for swap_bits in product((0, 1), repeat=4):
            setting_permutations = tuple((1, 0) if bit else (0, 1) for bit in swap_bits)
            action = signed_coordinate_action(
                Relabeling(
                    party_permutation=tuple(party_permutation),
                    setting_permutations=setting_permutations,
                )
            )
            result.append(np.asarray([target for target, _sign in action], dtype=np.int16))
    return tuple(result)


def canonical_h_row(row: Sequence[int]) -> tuple[int, ...]:
    """Return the lexicographically maximal oriented H-row in the full orbit."""
    normalized = tuple(map(int, row))
    if len(normalized) != AMBIENT_DIMENSION + 1:
        raise ValueError("a Bell 4-2-2 H-row must contain 81 integers")
    coefficients = np.asarray(normalized[1:], dtype=np.int64)
    signs = _outcome_sign_actions().astype(np.int64, copy=False)
    best: tuple[int, ...] | None = None
    for targets in _unsigned_coordinate_targets():
        candidates = np.empty((len(signs), AMBIENT_DIMENSION), dtype=np.int64)
        candidates[:, targets] = signs * coefficients
        surviving = np.arange(len(candidates))
        for column in range(AMBIENT_DIMENSION):
            values = candidates[surviving, column]
            maximum = values.max()
            surviving = surviving[values == maximum]
            if len(surviving) == 1:
                break
        candidate = (normalized[0], *map(int, candidates[int(surviving[0])]))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("empty Bell relabeling group")
    return best


class FacetCatalog:
    """Read-only answer catalog used strictly after geometric validation."""

    def __init__(self, handoff_root: Path | str = DEFAULT_HANDOFF_ROOT):
        self.path = Path(handoff_root).resolve() / "03_global_facets" / "global_facet_classes.csv"
        self._rows: dict[tuple[int, ...], tuple[str, int]] | None = None

    def _load(self) -> None:
        if self._rows is not None:
            return
        rows: dict[tuple[int, ...], tuple[str, int]] = {}
        with self.path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[tuple(map(int, json.loads(row["canonical_h_row"])))] = (
                    row["facet_class_id"],
                    int(row["catalog_index"]),
                )
        self._rows = rows

    def lookup(self, row: tuple[int, ...]) -> tuple[str, int] | None:
        self._load()
        assert self._rows is not None
        return self._rows.get(row)

    @property
    def loaded(self) -> bool:
        return self._rows is not None


class FacetOracle:
    def __init__(
        self,
        *,
        rank_oracle: RankOracle | None = None,
        handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
        use_catalog: bool = True,
        canonicalize: bool = True,
    ):
        self.points = np.asarray(deterministic_vertices(), dtype=float)
        self.augmented = np.column_stack((np.ones(len(self.points), dtype=np.int8), self.points.astype(np.int8)))
        self.rank_oracle = rank_oracle or RankOracle(self.points)
        self.catalog = FacetCatalog(handoff_root) if use_catalog else None
        self.canonicalize = bool(canonicalize)
        self.cache: dict[int, FacetResult] = {}
        self.validation_count = 0

    def validate(self, word: int) -> FacetResult:
        cached = self.cache.get(int(word))
        if cached is not None:
            return cached
        self.validation_count += 1
        rank = self.rank_oracle.rank(word)
        if rank != TARGET_AFFINE_RANK:
            result = FacetResult(False, "rank", int(word), affine_rank=rank)
            self.cache[int(word)] = result
            return result
        indices = word_indices(word)
        selected = self.augmented[indices].astype(float)
        _u, _s, vh = np.linalg.svd(selected, full_matrices=True)
        vector = vh[-1]
        signed = self.augmented @ vector
        tolerance = 1e-7 * max(1.0, float(np.max(np.abs(signed))))
        if np.any(signed > tolerance) and np.any(signed < -tolerance):
            result = FacetResult(False, "cuts_polytope", int(word), affine_rank=rank)
            self.cache[int(word)] = result
            return result
        if np.all(signed <= tolerance):
            vector = -vector
            signed = -signed
        try:
            h_row = _primitive_integer_row(vector, self.augmented)
        except ValueError:
            result = FacetResult(False, "integer_normalization", int(word), affine_rank=rank)
            self.cache[int(word)] = result
            return result
        exact_values = self.augmented.astype(object) @ np.asarray(h_row, dtype=object)
        if any(int(value) < 0 for value in exact_values):
            result = FacetResult(False, "cuts_polytope_exact", int(word), affine_rank=rank)
            self.cache[int(word)] = result
            return result
        tight_indices = [index for index, value in enumerate(exact_values) if int(value) == 0]
        support_word = sum(1 << index for index in tight_indices)
        support_rank = self.rank_oracle.rank(support_word)
        if support_rank != TARGET_AFFINE_RANK:
            result = FacetResult(False, "non_facet_face", int(word), affine_rank=support_rank)
            self.cache[int(word)] = result
            return result
        canonical = canonical_h_row(h_row) if self.canonicalize else h_row
        match = self.catalog.lookup(canonical) if self.catalog is not None else None
        if match is None:
            digest = hashlib.sha256(repr(canonical).encode("ascii")).hexdigest()[:12]
            class_id, catalog_index = f"BFS422-NEW-{digest}", None
        else:
            class_id, catalog_index = match
        result = FacetResult(
            True,
            "facet",
            int(word),
            support_word,
            support_rank,
            h_row,
            canonical,
            class_id,
            catalog_index,
        )
        self.cache[int(word)] = result
        self.cache.setdefault(support_word, result)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "validation_count": self.validation_count,
            "cache_entries": len(self.cache),
            "catalog_enabled": self.catalog is not None,
            "catalog_loaded": bool(self.catalog and self.catalog.loaded),
            "canonicalization_enabled": self.canonicalize,
        }
