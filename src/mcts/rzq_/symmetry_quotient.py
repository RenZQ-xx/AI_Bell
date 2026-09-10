"""Parent-local symmetry quotient for MCTS expansion actions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from baseline.orbit_blocks import BlockKey, build_state_group_permutations, unselected_blocks


BlockMap = tuple[int, ...]


def build_partition_block_maps(blocks: Sequence[Sequence[int]]) -> tuple[BlockMap, ...]:
    """Return full Bell symmetries that map the block partition to itself."""

    normalized = tuple(tuple(int(vertex) for vertex in block) for block in blocks)
    block_by_vertices = {
        frozenset(block): index for index, block in enumerate(normalized)
    }
    maps: set[BlockMap] = set()
    for permutation in build_state_group_permutations():
        image: list[int] = []
        for block in normalized:
            mapped = block_by_vertices.get(
                frozenset(int(permutation[vertex]) for vertex in block)
            )
            if mapped is None:
                break
            image.append(int(mapped))
        else:
            maps.add(tuple(image))
    identity = tuple(range(len(normalized)))
    maps.add(identity)
    return tuple(sorted(maps))


def parent_stabilizer_block_maps(
    parent_key: BlockKey,
    block_maps: Sequence[BlockMap],
) -> tuple[BlockMap, ...]:
    """Return block maps fixing the parent state; the empty root uses the full group."""

    selected = {index for index, value in enumerate(parent_key) if int(value)}
    if not selected:
        return tuple(block_maps)
    return tuple(
        block_map
        for block_map in block_maps
        if {int(block_map[index]) for index in selected} == selected
    )


def representative_action_families(
    parent_key: BlockKey,
    block_maps: Sequence[BlockMap],
    actions: Iterable[int] | None = None,
    *,
    use_full_group: bool = False,
) -> dict[int, tuple[int, ...]]:
    """Partition actions into stabilizer orbits, keyed by their minimum action."""

    remaining = set(
        int(action)
        for action in (unselected_blocks(parent_key) if actions is None else actions)
    )
    stabilizer = tuple(block_maps) if use_full_group else parent_stabilizer_block_maps(parent_key, block_maps)
    families: dict[int, tuple[int, ...]] = {}
    while remaining:
        seed = min(remaining)
        family = tuple(sorted({int(block_map[seed]) for block_map in stabilizer} & remaining))
        if not family:
            family = (seed,)
        representative = min(family)
        families[representative] = family
        remaining.difference_update(family)
    return families
