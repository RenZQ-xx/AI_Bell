from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bell322 import generate_bell322_points
from .geometry import FacetValidation, validate_facet_support
from .reference_classes import build_reference_database, classify_hyperplane, exact_label_from_match
from .support_masks import vertex_mask_to_indices


@dataclass(frozen=True)
class FacetLabel:
    """Terminal validation result plus optional exact class information."""

    label: str
    validation: FacetValidation
    match: dict[str, Any] | None = None

    @property
    def is_exact(self) -> bool:
        return self.label.startswith("exact:")

    def to_dict(self, *, include_normal: bool = False) -> dict[str, Any]:
        data = {
            "label": self.label,
            "is_exact": self.is_exact,
            "validation": self.validation.to_dict(include_normal=include_normal),
        }
        if self.match is not None:
            data["match"] = self.match
        return data


class FacetValidator:
    """Validate a 64-bit hard support and assign an exact facet class label."""

    def __init__(
        self,
        *,
        points: np.ndarray | None = None,
        reference: dict[str, Any] | None = None,
        facets_path: Path | None = None,
        min_cardinality: int = 26,
        plane_eps: float = 1e-6,
        support_tol: float = 1e-6,
        facet_rank_eps: float = 1e-5,
    ) -> None:
        self.points = generate_bell322_points() if points is None else np.asarray(points)
        self.reference = (
            build_reference_database() if reference is None and facets_path is None
            else build_reference_database(facets_path) if reference is None
            else reference
        )
        self.min_cardinality = int(min_cardinality)
        self.plane_eps = float(plane_eps)
        self.support_tol = float(support_tol)
        self.facet_rank_eps = float(facet_rank_eps)

    def validate_indices(self, indices: Sequence[int]) -> FacetLabel:
        """Validate selected vertex indices and return an exact/invalid label."""
        validation = validate_facet_support(
            self.points,
            indices,
            min_cardinality=self.min_cardinality,
            plane_eps=self.plane_eps,
            support_tol=self.support_tol,
            facet_rank_eps=self.facet_rank_eps,
        )
        if not validation.valid:
            return FacetLabel(label=self._invalid_label(validation), validation=validation)

        if validation.normal is None or validation.offset is None:
            return FacetLabel(label="invalid:missing_plane", validation=validation)

        match = classify_hyperplane(validation.normal, validation.offset, self.reference)
        label = exact_label_from_match(match)
        if not label.startswith("exact:"):
            label = "unknown_facet"
        return FacetLabel(label=label, validation=validation, match=match)

    def validate_mask(self, mask: Sequence[int]) -> FacetLabel:
        """Validate a 64-bit vertex mask."""
        return self.validate_indices(vertex_mask_to_indices(mask))

    def label_indices(self, indices: Sequence[int]) -> str:
        """Return only the compact label for selected vertex indices."""
        return self.validate_indices(indices).label

    def label_mask(self, mask: Sequence[int]) -> str:
        """Return only the compact label for a 64-bit vertex mask."""
        return self.validate_mask(mask).label

    @staticmethod
    def _invalid_label(validation: FacetValidation) -> str:
        if validation.reason == "cardinality":
            return "invalid:cardinality"
        if validation.reason == "non_coplanar":
            return "invalid:non_coplanar"
        if validation.reason == "cuts_polytope":
            return "invalid:boundary"
        if validation.reason == "non_facet_face":
            rank = validation.affine_rank
            return f"invalid:rank{rank}" if rank is not None else "invalid:non_facet_face"
        return "invalid"
