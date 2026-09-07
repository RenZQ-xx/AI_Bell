"""Exact-class compatibility counts for orbit-block search states.

A known facet support is compatible with a block partition only when every
block is either wholly inside or wholly outside that support.  A search prefix
is compatible with such a facet when all currently selected blocks occur in
the facet's block mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from baseline.reference_classes import build_reference_database, support_mask_from_row


BlockMask = frozenset[int]
BlockKey = tuple[int, ...]


def selected_block_indices(key: Sequence[int]) -> frozenset[int]:
    """Validate a 0/1 block key and return its selected block indices."""

    normalized = tuple(int(value) for value in key)
    if any(value not in (0, 1) for value in normalized):
        raise ValueError("a block key may contain only 0 and 1")
    return frozenset(index for index, value in enumerate(normalized) if value)


def project_support_to_blocks(
    support: Sequence[int],
    blocks: Sequence[Sequence[int]],
) -> BlockMask | None:
    """Project a vertex support to blocks, or return None for a partial block.

    Returning None is essential: a support containing only part of one orbit
    is not invariant under the subgroup that produced the block partition.
    """

    vertex_count = len(support)
    selected: set[int] = set()
    seen_vertices: set[int] = set()
    for block_index, block in enumerate(blocks):
        normalized_block = tuple(int(vertex) for vertex in block)
        if not normalized_block:
            raise ValueError(f"block {block_index} is empty")
        for vertex in normalized_block:
            if vertex < 0 or vertex >= vertex_count:
                raise IndexError(f"vertex {vertex} in block {block_index} is out of range")
            if vertex in seen_vertices:
                raise ValueError(f"vertex {vertex} occurs in more than one block")
            seen_vertices.add(vertex)
        inside = sum(bool(int(support[vertex])) for vertex in normalized_block)
        if inside == len(normalized_block):
            selected.add(block_index)
        elif inside:
            return None
    if seen_vertices != set(range(vertex_count)):
        raise ValueError("blocks must partition every support vertex exactly once")
    return frozenset(selected)


@dataclass(frozen=True)
class ClassCompatibilityIndex:
    """Precomputed compatible facet masks grouped by exact class."""

    block_count: int
    masks_by_class: Mapping[int, tuple[BlockMask, ...]]

    @classmethod
    def build(
        cls,
        blocks: Sequence[Sequence[int]],
        class_ids: Iterable[int] = range(1, 47),
    ) -> "ClassCompatibilityIndex":
        normalized_blocks = tuple(tuple(int(vertex) for vertex in block) for block in blocks)
        requested = tuple(sorted({int(class_id) for class_id in class_ids}))
        reference = build_reference_database()
        masks: dict[int, set[BlockMask]] = {class_id: set() for class_id in requested}

        for row, matches in reference["row_to_matches"].items():
            matched = {int(match["class_id"]) for match in matches}
            relevant = matched.intersection(masks)
            if not relevant:
                continue
            projected = project_support_to_blocks(
                support_mask_from_row(row),
                normalized_blocks,
            )
            if projected is None:
                continue
            for class_id in relevant:
                masks[class_id].add(projected)

        ordered = {
            class_id: tuple(sorted(class_masks, key=lambda item: (len(item), tuple(sorted(item)))))
            for class_id, class_masks in masks.items()
        }
        return cls(block_count=len(normalized_blocks), masks_by_class=ordered)

    def counts(self, key: Sequence[int]) -> dict[int, int]:
        """Return compat_class_c for every indexed class, including zeroes."""

        if len(key) != self.block_count:
            raise ValueError(f"block key has length {len(key)}; expected {self.block_count}")
        selected = selected_block_indices(key)
        return {
            class_id: sum(selected.issubset(mask) for mask in masks)
            for class_id, masks in sorted(self.masks_by_class.items())
        }

    def nonzero_counts(self, key: Sequence[int]) -> dict[int, int]:
        return {class_id: count for class_id, count in self.counts(key).items() if count}

    def compatible_classes(self, key: Sequence[int]) -> tuple[int, ...]:
        return tuple(self.nonzero_counts(key))
