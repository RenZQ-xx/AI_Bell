from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baseline2.primitives.geometry import affine_rank as compute_affine_rank
from baseline2.primitives.orbit_blocks import BlockKey, add_block, unselected_blocks
from baseline2.primitives.support_masks import block_key_to_vertex_indices

from .supportability_oracle import SupportabilityOracle


@dataclass
class GeometryOracle:
    blocks: list[tuple[int, ...]]
    points: np.ndarray
    config: Any
    rank_cache: dict[BlockKey, int]
    vertices_cache: dict[BlockKey, list[int]]
    flat_cache: dict[BlockKey, int]
    affine_hull_cache: dict[BlockKey, tuple[np.ndarray | None, np.ndarray]]
    incremental_affine_hull_enabled: bool
    supportability_oracle: SupportabilityOracle

    @classmethod
    def from_parts(
        cls,
        *,
        blocks: list[tuple[int, ...]],
        points: np.ndarray,
        config: Any,
        rank_cache: dict[BlockKey, int],
        vertices_cache: dict[BlockKey, list[int]],
        flat_cache: dict[BlockKey, int],
        affine_hull_cache: dict[BlockKey, tuple[np.ndarray | None, np.ndarray]],
        incremental_affine_hull_enabled: bool,
        supportability_oracle: SupportabilityOracle,
    ) -> "GeometryOracle":
        oracle = cls(
            blocks=blocks,
            points=points,
            config=config,
            rank_cache=rank_cache,
            vertices_cache=vertices_cache,
            flat_cache=flat_cache,
            affine_hull_cache=affine_hull_cache,
            incremental_affine_hull_enabled=bool(incremental_affine_hull_enabled),
            supportability_oracle=supportability_oracle,
        )
        oracle.supportability_oracle.bind_geometry(
            rank_fn=oracle.affine_rank,
            vertex_indices_fn=oracle.vertex_indices,
        )
        return oracle

    @classmethod
    def from_scorer(
        cls,
        scorer: Any,
        *,
        supportability_oracle: SupportabilityOracle | None = None,
    ) -> "GeometryOracle":
        return cls.from_parts(
            blocks=scorer.blocks,
            points=scorer.points,
            config=scorer.config,
            rank_cache=scorer.rank_cache,
            vertices_cache=scorer.vertices_cache,
            flat_cache=scorer.flat_cache,
            affine_hull_cache=scorer.affine_hull_cache,
            incremental_affine_hull_enabled=bool(scorer.incremental_affine_hull_enabled),
            supportability_oracle=supportability_oracle or SupportabilityOracle.from_scorer(scorer),
        )

    def vertex_indices(self, key: BlockKey) -> list[int]:
        if key not in self.vertices_cache:
            self.vertices_cache[key] = block_key_to_vertex_indices(key, self.blocks)
        return self.vertices_cache[key]

    def affine_rank(self, key: BlockKey) -> int:
        if key not in self.rank_cache:
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                rank = 0
            elif self.incremental_affine_hull_enabled:
                _anchor, basis = self.affine_hull_basis(key)
                rank = int(basis.shape[1])
            else:
                rank = compute_affine_rank(self.points[indices], rank_eps=self.config.rank_tol)
            self.rank_cache[key] = rank
        return self.rank_cache[key]

    def _cached_parent_for_incremental_hull(self, key: BlockKey) -> tuple[BlockKey, int] | None:
        selected_actions = [index for index, value in enumerate(key) if int(value)]
        for action in reversed(selected_actions):
            parent_key = tuple(0 if idx == int(action) else int(value) for idx, value in enumerate(key))
            if parent_key in self.affine_hull_cache or parent_key in self.rank_cache or not any(parent_key):
                return parent_key, int(action)
        return None

    def affine_hull_basis(self, key: BlockKey) -> tuple[np.ndarray | None, np.ndarray]:
        if key not in self.affine_hull_cache:
            indices = self.vertex_indices(key)
            if not indices:
                self.affine_hull_cache[key] = (None, np.zeros((self.points.shape[1], 0), dtype=float))
                self.rank_cache[key] = 0
                return self.affine_hull_cache[key]

            incremental_parent = (
                self._cached_parent_for_incremental_hull(key)
                if self.incremental_affine_hull_enabled
                else None
            )
            if incremental_parent is not None:
                parent_key, added_action = incremental_parent
                parent_anchor, parent_basis = self.affine_hull_basis(parent_key)
                new_points = self.points[list(self.blocks[added_action])]
                anchor, basis = self._extend_affine_hull(parent_anchor, parent_basis, new_points)
                self.affine_hull_cache[key] = (anchor, basis)
                self.rank_cache[key] = int(basis.shape[1])
            else:
                selected = self.points[indices]
                anchor = selected[0].copy()
                centered = selected - anchor
                _u, singulars, vh = np.linalg.svd(centered, full_matrices=False)
                rank = int(np.sum(singulars > self.config.rank_tol))
                basis = np.zeros((self.points.shape[1], 0), dtype=float) if rank <= 0 else vh[:rank].T
                self.affine_hull_cache[key] = (anchor, basis)
                self.rank_cache[key] = rank
        return self.affine_hull_cache[key]

    def _extend_affine_hull(
        self,
        anchor: np.ndarray | None,
        basis: np.ndarray,
        new_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(new_points) <= 0:
            if anchor is None:
                return np.zeros(self.points.shape[1], dtype=float), basis
            return anchor, basis
        if anchor is None:
            anchor = np.asarray(new_points[0], dtype=float).copy()
            points_to_add = new_points[1:]
        else:
            anchor = np.asarray(anchor, dtype=float)
            points_to_add = new_points

        columns = [basis[:, idx].copy() for idx in range(basis.shape[1])]
        for point in points_to_add:
            vector = np.asarray(point, dtype=float) - anchor
            for column in columns:
                vector = vector - float(np.dot(vector, column)) * column
            norm = float(np.linalg.norm(vector))
            if norm > self.config.rank_tol:
                columns.append(vector / norm)
        if not columns:
            return anchor, np.zeros((self.points.shape[1], 0), dtype=float)
        return anchor, np.column_stack(columns)

    def block_lies_in_affine_hull(self, key: BlockKey, action: int) -> bool:
        anchor, basis = self.affine_hull_basis(key)
        if anchor is None:
            return False
        block_points = self.points[list(self.blocks[int(action)])]
        delta = block_points - anchor
        residual = delta - (delta @ basis) @ basis.T if basis.shape[1] > 0 else delta
        return float(np.max(np.linalg.norm(residual, axis=1))) <= 2.0 * self.config.rank_tol

    def flat_capacity(self, key: BlockKey) -> int:
        if key not in self.flat_cache:
            if self.config.flat_capacity_method == "child_rank":
                rank = self.affine_rank(key)
                if self.incremental_affine_hull_enabled and len(self.vertex_indices(key)) > 1:
                    self.flat_cache[key] = sum(
                        1 for action in unselected_blocks(key) if self.block_lies_in_affine_hull(key, action)
                    )
                    return self.flat_cache[key]
                self.flat_cache[key] = sum(
                    1 for action in unselected_blocks(key) if self.affine_rank(add_block(key, action)) == rank
                )
                return self.flat_cache[key]

            rank = self.affine_rank(key)
            indices = self.vertex_indices(key)
            if len(indices) <= 1:
                self.flat_cache[key] = sum(
                    1 for action in unselected_blocks(key) if self.affine_rank(add_block(key, action)) == rank
                )
                return self.flat_cache[key]

            selected = self.points[indices]
            anchor = selected[0]
            centered = selected - anchor
            _u, singulars, vh = np.linalg.svd(centered, full_matrices=False)
            hull_rank = int(np.sum(singulars > self.config.rank_tol))
            basis = np.zeros((self.points.shape[1], 0), dtype=float) if hull_rank <= 0 else vh[:hull_rank].T

            count = 0
            for action in unselected_blocks(key):
                block_points = self.points[list(self.blocks[action])]
                delta = block_points - anchor
                residual = delta - (delta @ basis) @ basis.T if basis.shape[1] > 0 else delta
                if float(np.max(np.linalg.norm(residual, axis=1))) <= 2.0 * self.config.rank_tol:
                    count += 1
            self.flat_cache[key] = count
        return self.flat_cache[key]

    def supportability_metrics(self, key: BlockKey):
        return self.supportability_oracle.metrics(key)
