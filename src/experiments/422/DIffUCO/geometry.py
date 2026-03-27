from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch


def generate_states_422() -> List[tuple[int, ...]]:
    states: List[tuple[int, ...]] = []
    for a0 in (1, -1):
        for a1 in (1, -1):
            for b0 in (1, -1):
                for b1 in (1, -1):
                    for c0 in (1, -1):
                        for c1 in (1, -1):
                            for d0 in (1, -1):
                                for d1 in (1, -1):
                                    states.append((a0, a1, b0, b1, c0, c1, d0, d1))
    return states


def generate_points_422() -> torch.Tensor:
    rows: List[List[float]] = []
    for a0, a1, b0, b1, c0, c1, d0, d1 in generate_states_422():
        rows.append(
            [
                a0, a1, b0, b1, c0, c1, d0, d1,
                a0 * b0, a0 * b1, a1 * b0, a1 * b1,
                a0 * c0, a0 * c1, a1 * c0, a1 * c1,
                a0 * d0, a0 * d1, a1 * d0, a1 * d1,
                b0 * c0, b0 * c1, b1 * c0, b1 * c1,
                b0 * d0, b0 * d1, b1 * d0, b1 * d1,
                c0 * d0, c0 * d1, c1 * d0, c1 * d1,
                a0 * b0 * c0, a0 * b1 * c0, a1 * b0 * c0, a1 * b1 * c0,
                a0 * b0 * c1, a0 * b1 * c1, a1 * b0 * c1, a1 * b1 * c1,
                a0 * b0 * d0, a0 * b1 * d0, a1 * b0 * d0, a1 * b1 * d0,
                a0 * b0 * d1, a0 * b1 * d1, a1 * b0 * d1, a1 * b1 * d1,
                a0 * c0 * d0, a0 * c1 * d0, a1 * c0 * d0, a1 * c1 * d0,
                a0 * c0 * d1, a0 * c1 * d1, a1 * c0 * d1, a1 * c1 * d1,
                b0 * c0 * d0, b0 * c1 * d0, b1 * c0 * d0, b1 * c1 * d0,
                b0 * c0 * d1, b0 * c1 * d1, b1 * c0 * d1, b1 * c1 * d1,
                a0 * b0 * c0 * d0, a0 * b1 * c0 * d0, a1 * b0 * c0 * d0, a1 * b1 * c0 * d0,
                a0 * b0 * c1 * d0, a0 * b1 * c1 * d0, a1 * b0 * c1 * d0, a1 * b1 * c1 * d0,
                a0 * b0 * c0 * d1, a0 * b1 * c0 * d1, a1 * b0 * c0 * d1, a1 * b1 * c0 * d1,
                a0 * b0 * c1 * d1, a0 * b1 * c1 * d1, a1 * b0 * c1 * d1, a1 * b1 * c1 * d1,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


@dataclass
class PlaneStatistics:
    centroid: torch.Tensor
    covariance: torch.Tensor
    eigenvalues: torch.Tensor
    normal: torch.Tensor
    offset: torch.Tensor


def weighted_plane_statistics(points: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> PlaneStatistics:
    normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    centroid = normalized @ points
    centered = points.unsqueeze(0) - centroid.unsqueeze(1)
    covariance = torch.einsum("bn,bnd,bne->bde", normalized, centered, centered)
    covariance = covariance + eps * torch.eye(points.shape[-1], device=points.device).unsqueeze(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[:, :, 0]
    offset = -(centroid * normal).sum(dim=-1)
    return PlaneStatistics(
        centroid=centroid,
        covariance=covariance,
        eigenvalues=eigenvalues,
        normal=normal,
        offset=offset,
    )


def smallest_eigenvalue(points: np.ndarray) -> float:
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(np.min(eigenvalues))


def affine_rank(points: np.ndarray, tol: float = 1e-8) -> int:
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    return int(np.linalg.matrix_rank(centered, tol=tol))


def fit_hyperplane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    offset = -float(np.dot(normal, centroid))
    return normal, offset


def boundary_side_metrics(all_points: np.ndarray, normal: np.ndarray, offset: float, tol: float = 1e-6) -> Dict[str, float]:
    signed = all_points @ normal + offset
    positive = int(np.sum(signed > tol))
    negative = int(np.sum(signed < -tol))
    closer_side = min(positive, negative)
    return {
        "positive": positive,
        "negative": negative,
        "closer_side": closer_side,
        "max_abs_distance": float(np.max(np.abs(signed))),
    }


def validate_subset(
    all_points: np.ndarray,
    indices: Sequence[int],
    min_cardinality: int = 80,
    plane_eps: float = 1e-6,
    support_tol: float = 1e-6,
) -> Dict[str, object]:
    unique_indices = sorted(set(int(index) for index in indices))
    subset = all_points[unique_indices]
    result: Dict[str, object] = {
        "indices": unique_indices,
        "cardinality": len(unique_indices),
        "valid": False,
    }
    if len(unique_indices) < min_cardinality:
        result["reason"] = "cardinality"
        return result

    plane_value = smallest_eigenvalue(subset)
    subset_rank = affine_rank(subset, tol=max(plane_eps, 1e-8))
    target_rank = max(all_points.shape[1] - 1, 0)
    normal, offset = fit_hyperplane(subset)
    side = boundary_side_metrics(all_points, normal, offset, tol=support_tol)
    supporting_face_valid = plane_value < plane_eps and side["closer_side"] == 0
    facet_valid = supporting_face_valid and subset_rank >= target_rank
    result.update(
        {
            "plane_eigenvalue": plane_value,
            "affine_rank": subset_rank,
            "facet_target_rank": target_rank,
            "normal": normal.tolist(),
            "offset": offset,
            "support": side,
            "supporting_face_valid": supporting_face_valid,
            "is_facet": facet_valid,
            "valid": supporting_face_valid,
        }
    )
    if not result["valid"]:
        result["reason"] = "geometry"
    elif not result["is_facet"]:
        result["reason"] = "lower_dim_face"
    return result
