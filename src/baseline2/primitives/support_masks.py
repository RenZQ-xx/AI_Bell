from __future__ import annotations

from typing import Sequence

import numpy as np

from .orbit_blocks import BlockKey, selected_blocks, selected_vertices_from_blocks


def block_key_to_vertex_indices(key: BlockKey, blocks: Sequence[Sequence[int]]) -> list[int]:
    """Expand selected orbit blocks into sorted deterministic vertex indices."""
    return selected_vertices_from_blocks(key, blocks)


def block_key_to_vertex_mask(
    key: BlockKey,
    blocks: Sequence[Sequence[int]],
    *,
    vertex_count: int = 64,
    dtype: np.dtype | type = np.int64,
) -> np.ndarray:
    """Convert a block-level search state into a vertex-level hard support mask."""
    if vertex_count <= 0:
        raise ValueError(f"vertex_count must be positive, got {vertex_count}")
    mask = np.zeros(vertex_count, dtype=dtype)
    for block_index in selected_blocks(key):
        for vertex in blocks[block_index]:
            index = int(vertex)
            if index < 0 or index >= vertex_count:
                raise IndexError(f"vertex index {index} out of range for {vertex_count} vertices")
            mask[index] = 1
    return mask


def vertex_mask_to_indices(mask: Sequence[int]) -> list[int]:
    """Return selected vertex indices from a 0/1 hard support mask."""
    return [index for index, value in enumerate(mask) if int(value) == 1]


def vertex_mask_to_key(mask: Sequence[int]) -> tuple[int, ...]:
    """Normalize a vertex support mask as an immutable tuple of 0/1 integers."""
    return tuple(1 if int(value) else 0 for value in mask)


def block_key_to_vertex_key(
    key: BlockKey,
    blocks: Sequence[Sequence[int]],
    *,
    vertex_count: int = 64,
) -> tuple[int, ...]:
    """Convert selected orbit blocks into an immutable 64-bit vertex support key."""
    mask = block_key_to_vertex_mask(key, blocks, vertex_count=vertex_count, dtype=np.int64)
    return vertex_mask_to_key(mask)
