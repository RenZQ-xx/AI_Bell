from __future__ import annotations

"""Dimension-independent pivots through codimension-two faces."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgePivotResult:
    facet_words: tuple[int, ...]
    candidate_rays: int
    feasible_rays: int
    rejected_facets: int
    reason: str | None = None


def _indices(word: int, vertex_count: int) -> list[int]:
    return [index for index in range(vertex_count) if word & (1 << index)]


def _affine_rank(points: np.ndarray, *, tolerance: float = 1e-9) -> int:
    if len(points) <= 1:
        return max(-1, len(points) - 1)
    centered = np.asarray(points[1:] - points[0], dtype=float)
    return int(np.linalg.matrix_rank(centered, tol=tolerance))


def _null_space(matrix: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    _u, singular, vh = np.linalg.svd(np.asarray(matrix, dtype=float), full_matrices=True)
    scale = float(singular[0]) if len(singular) else 1.0
    rank = int(np.count_nonzero(singular > tolerance * max(1.0, scale)))
    return vh[rank:].T.copy()


def adjacent_facet_words(
    points: np.ndarray,
    source_word: int,
    ridge_word: int,
    *,
    cone_tol: float = 1e-9,
    tight_tol: float = 1e-7,
) -> RidgePivotResult:
    """Return all supporting facets incident to a verified ridge.

    The algorithm depends on the ridge normal space being two-dimensional, not
    on a fixed Bell scenario. Support words may contain any number of bits.
    """

    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] < 2:
        raise ValueError("points must be a nonempty two-dimensional array")
    vertex_count, dimension = cloud.shape
    if source_word < 0 or ridge_word < 0:
        raise ValueError("support words must be nonnegative")
    if source_word.bit_length() > vertex_count or ridge_word.bit_length() > vertex_count:
        raise ValueError("support word exceeds the point cloud")
    if ridge_word & ~source_word:
        return RidgePivotResult((), 0, 0, 0, "ridge_not_in_source")

    source_indices = _indices(source_word, vertex_count)
    ridge_indices = _indices(ridge_word, vertex_count)
    if len(source_indices) < dimension or len(ridge_indices) < dimension - 1:
        return RidgePivotResult((), 0, 0, 0, "insufficient_vertices")
    if _affine_rank(cloud[source_indices]) != dimension - 1:
        return RidgePivotResult((), 0, 0, 0, "source_not_facet_rank")
    if _affine_rank(cloud[ridge_indices]) != dimension - 2:
        return RidgePivotResult((), 0, 0, 0, "retained_not_ridge_rank")

    augmented = np.column_stack((cloud, np.ones(vertex_count)))
    basis = _null_space(augmented[ridge_indices])
    if basis.shape != (dimension + 1, 2):
        return RidgePivotResult((), 0, 0, 0, "normal_space_not_two_dimensional")

    inequalities = augmented @ basis
    ray_directions: list[np.ndarray] = []
    feasible_rays = 0
    for row in inequalities:
        norm = float(np.linalg.norm(row))
        if norm <= cone_tol:
            continue
        perpendicular = np.asarray((-row[1], row[0]), dtype=float) / norm
        for direction in (perpendicular, -perpendicular):
            slacks = inequalities @ direction
            scale = max(1.0, float(np.max(np.abs(slacks))))
            if float(np.max(slacks)) > cone_tol * scale:
                continue
            feasible_rays += 1
            if any(float(np.dot(direction, old)) >= 1.0 - 1e-8 for old in ray_directions):
                continue
            ray_directions.append(direction)

    facets: set[int] = set()
    rejected = 0
    for direction in ray_directions:
        hyperplane = basis @ direction
        normal_norm = float(np.linalg.norm(hyperplane[:-1]))
        if normal_norm <= cone_tol:
            rejected += 1
            continue
        hyperplane /= normal_norm
        slacks = augmented @ hyperplane
        scale = max(1.0, float(np.max(np.abs(slacks))))
        tight = np.flatnonzero(np.abs(slacks) <= tight_tol * scale)
        tight_word = sum(1 << int(index) for index in tight)
        if ridge_word & ~tight_word:
            rejected += 1
            continue
        tight_points = cloud[tight]
        if _affine_rank(tight_points) != dimension - 1:
            rejected += 1
            continue
        facets.add(tight_word)

    return RidgePivotResult(
        tuple(sorted(facets)),
        len(ray_directions),
        feasible_rays,
        rejected,
        None if facets else "no_valid_incident_facet",
    )
