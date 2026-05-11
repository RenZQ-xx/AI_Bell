from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class BoundarySideMetrics:
    """How a hyperplane separates the full 64-vertex Bell point cloud."""

    positive: int
    negative: int
    closer_side: int
    max_abs_distance: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "closer_side": self.closer_side,
            "max_abs_distance": self.max_abs_distance,
        }


@dataclass(frozen=True)
class FacetValidation:
    """Result of checking whether selected vertices define a valid facet."""

    indices: list[int]
    cardinality: int
    valid: bool
    reason: str | None = None
    plane_eigenvalue: float | None = None
    second_eigenvalue: float | None = None
    affine_rank: int | None = None
    codimension: int | None = None
    normal: np.ndarray | None = None
    offset: float | None = None
    support: BoundarySideMetrics | None = None
    coplanar: bool | None = None
    supporting: bool | None = None

    def to_dict(self, *, include_normal: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "indices": self.indices,
            "cardinality": self.cardinality,
            "valid": self.valid,
        }
        if self.reason is not None:
            data["reason"] = self.reason
        if self.plane_eigenvalue is not None:
            data["plane_eigenvalue"] = self.plane_eigenvalue
        if self.second_eigenvalue is not None:
            data["second_eigenvalue"] = self.second_eigenvalue
        if self.affine_rank is not None:
            data["affine_rank"] = self.affine_rank
        if self.codimension is not None:
            data["codimension"] = self.codimension
        if include_normal and self.normal is not None:
            data["normal"] = self.normal.tolist()
        if self.offset is not None:
            data["offset"] = self.offset
        if self.support is not None:
            data["support"] = self.support.to_dict()
        if self.coplanar is not None:
            data["coplanar"] = self.coplanar
        if self.supporting is not None:
            data["supporting"] = self.supporting
        return data


def as_numpy_points(points: np.ndarray) -> np.ndarray:
    """Return a 2D float array and fail early on malformed point clouds."""
    array = np.asarray(points)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(float)
    if array.ndim != 2:
        raise ValueError(f"points must be a 2D array, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"points must be nonempty, got shape {array.shape}")
    return array


def unique_sorted_indices(indices: Sequence[int], *, upper_bound: int | None = None) -> list[int]:
    """Normalize a support index list before doing linear algebra."""
    unique = sorted(set(int(index) for index in indices))
    if upper_bound is not None:
        bad = [index for index in unique if index < 0 or index >= upper_bound]
        if bad:
            raise IndexError(f"indices out of range for {upper_bound} points: {bad[:5]}")
    return unique


def covariance_eigenvalues(points: np.ndarray) -> np.ndarray:
    """Eigenvalues of the centered covariance of a selected point set."""
    selected = as_numpy_points(points)
    centroid = selected.mean(axis=0, keepdims=True)
    centered = selected - centroid
    covariance = centered.T @ centered / max(len(selected), 1)
    return np.linalg.eigvalsh(covariance)


def affine_rank(points: np.ndarray, *, rank_eps: float = 1e-5) -> int:
    """Affine rank of selected points, computed from covariance eigenvalues."""
    eigenvalues = covariance_eigenvalues(points)
    return int(np.sum(eigenvalues > rank_eps))


def fit_hyperplane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the least-squares hyperplane through selected points by SVD."""
    selected = as_numpy_points(points)
    centroid = selected.mean(axis=0)
    centered = selected - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("cannot fit a hyperplane with a zero normal")
    normal = normal / norm
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def boundary_side_metrics(
    all_points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    *,
    tol: float = 1e-6,
) -> BoundarySideMetrics:
    """Count full point-cloud vertices on each side of a candidate hyperplane."""
    points = as_numpy_points(all_points)
    normal_array = np.asarray(normal, dtype=float)
    if normal_array.shape != (points.shape[1],):
        raise ValueError(
            f"normal shape {normal_array.shape} does not match point dimension {points.shape[1]}"
        )
    signed = points @ normal_array + float(offset)
    positive = int(np.sum(signed > tol))
    negative = int(np.sum(signed < -tol))
    closer_side = min(positive, negative)
    return BoundarySideMetrics(
        positive=positive,
        negative=negative,
        closer_side=closer_side,
        max_abs_distance=float(np.max(np.abs(signed))),
    )


def validate_facet_support(
    all_points: np.ndarray,
    indices: Sequence[int],
    *,
    min_cardinality: int = 26,
    plane_eps: float = 1e-6,
    support_tol: float = 1e-6,
    facet_rank_eps: float = 1e-5,
) -> FacetValidation:
    """Check whether a hard support is an exact codimension-1 supporting facet.

    This is a terminal validator, not a low-rank search heuristic.  It assumes
    the selected support is already a candidate terminal set and verifies:

    - enough selected vertices;
    - selected vertices are coplanar;
    - the fitted plane supports the whole point cloud;
    - selected vertices have affine rank one less than the ambient dimension.
    """
    points = as_numpy_points(all_points)
    unique = unique_sorted_indices(indices, upper_bound=len(points))
    if len(unique) < min_cardinality:
        return FacetValidation(
            indices=unique,
            cardinality=len(unique),
            valid=False,
            reason="cardinality",
        )

    subset = points[unique]
    eigenvalues = covariance_eigenvalues(subset)
    plane_value = float(eigenvalues[0])
    second_eigenvalue = float(eigenvalues[1]) if eigenvalues.shape[0] > 1 else 0.0
    normal, offset = fit_hyperplane(subset)
    side = boundary_side_metrics(points, normal, offset, tol=support_tol)
    support_rank = int(np.sum(eigenvalues > facet_rank_eps))
    codimension = int(points.shape[1] - support_rank)
    supporting = bool(side.closer_side == 0)
    coplanar = bool(plane_value < plane_eps)
    valid = bool(coplanar and supporting and codimension == 1)

    reason: str | None = None
    if not valid:
        if not coplanar:
            reason = "non_coplanar"
        elif not supporting:
            reason = "cuts_polytope"
        elif codimension != 1:
            reason = "non_facet_face"
        else:
            reason = "geometry"

    return FacetValidation(
        indices=unique,
        cardinality=len(unique),
        valid=valid,
        reason=reason,
        plane_eigenvalue=plane_value,
        second_eigenvalue=second_eigenvalue,
        affine_rank=support_rank,
        codimension=codimension,
        normal=normal,
        offset=offset,
        support=side,
        coplanar=coplanar,
        supporting=supporting,
    )


def validate_subset(
    all_points: np.ndarray,
    indices: Sequence[int],
    *,
    min_cardinality: int = 26,
    plane_eps: float = 1e-6,
    support_tol: float = 1e-6,
    facet_rank_eps: float = 1e-5,
) -> dict[str, Any]:
    """Compatibility wrapper matching the old exploratory function name."""
    validation = validate_facet_support(
        all_points,
        indices,
        min_cardinality=min_cardinality,
        plane_eps=plane_eps,
        support_tol=support_tol,
        facet_rank_eps=facet_rank_eps,
    )
    return validation.to_dict()
