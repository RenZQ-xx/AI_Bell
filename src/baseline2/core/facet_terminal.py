from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from typing import Sequence

import numpy as np

from baseline2.primitives.geometry import FacetValidation, validate_facet_support
from baseline2.primitives.orbit_blocks import BlockKey
from baseline2.primitives.support_masks import block_key_to_vertex_indices


@dataclass(frozen=True)
class FacetFamily:
    """A geometry-only facet family discovered by the search."""

    family_id: str
    canonical_signature: tuple[int, ...]
    representation: str
    normal: tuple[float, ...]
    offset: float
    first_key: BlockKey
    first_path: tuple[int, ...] | None = None
    first_action: int | None = None


@dataclass(frozen=True)
class FacetTerminalHit:
    """One exact facet hit, attached to the discovered facet family."""

    family: FacetFamily
    key: BlockKey
    validation: FacetValidation
    duplicate: bool
    path: tuple[int, ...] | None = None
    action: int | None = None

    @property
    def family_id(self) -> str:
        return self.family.family_id

    @property
    def label(self) -> str:
        return self.family.representation

    @property
    def is_exact(self) -> bool:
        return True

    def to_dict(self, *, include_normal: bool = False) -> dict:
        data = {
            "label": self.label,
            "is_exact": True,
            "family_id": self.family_id,
            "duplicate": self.duplicate,
            "validation": self.validation.to_dict(include_normal=include_normal),
        }
        if self.path is not None:
            data["path"] = list(self.path)
        if self.action is not None:
            data["action"] = self.action
        return data


@dataclass(frozen=True)
class FacetTerminalMiss:
    """A rank-terminal candidate that is not a valid facet."""

    key: BlockKey
    validation: FacetValidation

    @property
    def label(self) -> str:
        reason = self.validation.reason or "geometry"
        if reason == "cardinality":
            return "invalid:cardinality"
        if reason == "non_coplanar":
            return "invalid:non_coplanar"
        if reason == "cuts_polytope":
            return "invalid:boundary"
        if reason == "non_facet_face":
            rank = self.validation.affine_rank
            return f"invalid:rank{rank}" if rank is not None else "invalid:non_facet_face"
        return "invalid"

    @property
    def is_exact(self) -> bool:
        return False

    def to_dict(self, *, include_normal: bool = False) -> dict:
        return {
            "label": self.label,
            "is_exact": False,
            "validation": self.validation.to_dict(include_normal=include_normal),
        }


@dataclass
class FacetFamilyRegistry:
    """Deduplicate exact facet hits by geometry-derived family signature."""

    families: dict[str, FacetFamily] = field(default_factory=dict)
    hit_counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        *,
        signature: tuple[int, ...],
        normal: Sequence[float],
        offset: float,
        key: BlockKey,
        validation: FacetValidation,
        path: Sequence[int] | None = None,
        action: int | None = None,
    ) -> FacetTerminalHit:
        family_id = family_id_from_signature(signature)
        duplicate = family_id in self.families
        if duplicate:
            family = self.families[family_id]
        else:
            family = FacetFamily(
                family_id=family_id,
                canonical_signature=signature,
                representation=represent_hyperplane(signature),
                normal=tuple(float(value) for value in normal),
                offset=float(offset),
                first_key=tuple(int(value) for value in key),
                first_path=None if path is None else tuple(int(value) for value in path),
                first_action=None if action is None else int(action),
            )
            self.families[family_id] = family
            self.hit_counts[family_id] = 0

        self.hit_counts[family_id] = self.hit_counts.get(family_id, 0) + 1
        return FacetTerminalHit(
            family=family,
            key=tuple(int(value) for value in key),
            validation=validation,
            duplicate=duplicate,
            path=None if path is None else tuple(int(value) for value in path),
            action=None if action is None else int(action),
        )

    def family_hits(self, family_id: str) -> int:
        return int(self.hit_counts.get(family_id, 0))

    def note_first_context(
        self,
        family_id: str,
        *,
        path: Sequence[int] | None = None,
        action: int | None = None,
    ) -> None:
        family = self.families.get(family_id)
        if family is None or (family.first_path is not None and family.first_action is not None):
            return
        self.families[family_id] = FacetFamily(
            family_id=family.family_id,
            canonical_signature=family.canonical_signature,
            representation=family.representation,
            normal=family.normal,
            offset=family.offset,
            first_key=family.first_key,
            first_path=family.first_path if family.first_path is not None else (
                None if path is None else tuple(int(value) for value in path)
            ),
            first_action=family.first_action if family.first_action is not None else (
                None if action is None else int(action)
            ),
        )

    def to_dict(self) -> dict:
        return {
            "family_count": len(self.families),
            "families": [
                {
                    "family_id": family.family_id,
                    "representation": family.representation,
                    "hit_count": self.family_hits(family.family_id),
                    "duplicate_count": max(0, self.family_hits(family.family_id) - 1),
                    "first_key": list(family.first_key),
                    "first_path": None if family.first_path is None else list(family.first_path),
                    "first_action": family.first_action,
                    "normal": list(family.normal),
                    "offset": float(family.offset),
                }
                for family in sorted(self.families.values(), key=lambda item: item.family_id)
            ],
        }


@dataclass
class FacetTerminalOracle:
    """Geometry-only terminal oracle for discovering and deduplicating facets."""

    points: np.ndarray
    blocks: Sequence[Sequence[int]]
    registry: FacetFamilyRegistry = field(default_factory=FacetFamilyRegistry)
    min_cardinality: int = 26
    plane_eps: float = 1e-6
    support_tol: float = 1e-6
    facet_rank_eps: float = 1e-5
    signature_scale: int = 1_000_000_000

    @classmethod
    def from_parts(
        cls,
        *,
        points: np.ndarray,
        blocks: Sequence[Sequence[int]],
        config,
        registry: FacetFamilyRegistry | None = None,
    ) -> "FacetTerminalOracle":
        return cls(
            points=np.asarray(points),
            blocks=blocks,
            registry=registry or FacetFamilyRegistry(),
            min_cardinality=int(getattr(config, "min_terminal_cardinality", 26)),
            plane_eps=float(getattr(config, "plane_eps", 1e-6)),
            support_tol=float(getattr(config, "support_tol", 1e-6)),
            facet_rank_eps=float(getattr(config, "facet_rank_eps", 1e-5)),
        )

    @classmethod
    def from_scorer(cls, scorer, *, registry: FacetFamilyRegistry | None = None) -> "FacetTerminalOracle":
        return cls.from_parts(
            points=scorer.points,
            blocks=scorer.blocks,
            config=scorer.config,
            registry=registry,
        )

    def validate_key(
        self,
        key: BlockKey,
        *,
        path: Sequence[int] | None = None,
        action: int | None = None,
    ) -> FacetTerminalHit | None:
        result = self.evaluate_key(key, path=path, action=action)
        return result if isinstance(result, FacetTerminalHit) else None

    def evaluate_key(
        self,
        key: BlockKey,
        *,
        path: Sequence[int] | None = None,
        action: int | None = None,
    ) -> FacetTerminalHit | FacetTerminalMiss:
        indices = block_key_to_vertex_indices(key, self.blocks)
        return self.evaluate_indices(indices, key=key, path=path, action=action)

    def validate_indices(
        self,
        indices: Sequence[int],
        *,
        key: BlockKey,
        path: Sequence[int] | None = None,
        action: int | None = None,
    ) -> FacetTerminalHit | None:
        result = self.evaluate_indices(indices, key=key, path=path, action=action)
        return result if isinstance(result, FacetTerminalHit) else None

    def evaluate_indices(
        self,
        indices: Sequence[int],
        *,
        key: BlockKey,
        path: Sequence[int] | None = None,
        action: int | None = None,
    ) -> FacetTerminalHit | FacetTerminalMiss:
        validation = validate_facet_support(
            self.points,
            indices,
            min_cardinality=self.min_cardinality,
            plane_eps=self.plane_eps,
            support_tol=self.support_tol,
            facet_rank_eps=self.facet_rank_eps,
        )
        if not validation.valid or validation.normal is None or validation.offset is None:
            return FacetTerminalMiss(key=tuple(int(value) for value in key), validation=validation)

        normal, offset = normalize_hyperplane(validation.normal, validation.offset)
        signature = canonical_hyperplane_signature(
            normal,
            offset,
            scale=self.signature_scale,
        )
        return self.registry.record(
            signature=signature,
            normal=normal,
            offset=offset,
            key=key,
            validation=validation,
            path=path,
            action=action,
        )


def normalize_hyperplane(normal: Sequence[float], offset: float) -> tuple[np.ndarray, float]:
    normal_array = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(normal_array))
    if norm == 0.0:
        raise ValueError("cannot normalize a hyperplane with zero normal")
    normal_array = normal_array / norm
    offset_value = float(offset) / norm

    first_nonzero = next((value for value in normal_array if abs(float(value)) > 0.0), 0.0)
    if float(first_nonzero) < 0.0:
        normal_array = -normal_array
        offset_value = -offset_value
    return normal_array, offset_value


def canonical_hyperplane_signature(
    normal: Sequence[float],
    offset: float,
    *,
    scale: int = 1_000_000_000,
) -> tuple[int, ...]:
    values = [*np.asarray(normal, dtype=float).tolist(), float(offset)]
    return tuple(int(round(value * scale)) for value in values)


def family_id_from_signature(signature: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in signature).encode("ascii")
    return "facet:" + sha1(payload).hexdigest()[:16]


def represent_hyperplane(signature: Sequence[int]) -> str:
    return "exact:" + family_id_from_signature(signature)
