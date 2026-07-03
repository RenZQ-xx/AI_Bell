from __future__ import annotations

from typing import Sequence

from baseline2.primitives.orbit_blocks import build_state_group_permutations


def induced_block_map(vertex_perm, *, orbits, block_by_vertices):
    out = []
    for block in orbits:
        image = frozenset(int(vertex_perm[int(v)]) for v in block)
        mapped = block_by_vertices.get(image)
        if mapped is None:
            return None
        out.append(int(mapped))
    return tuple(out)


def build_partition_block_maps(orbits):
    block_by_vertices = {frozenset(int(v) for v in block): i for i, block in enumerate(orbits)}
    maps = set()
    for vertex_perm in build_state_group_permutations():
        block_map = induced_block_map(vertex_perm, orbits=orbits, block_by_vertices=block_by_vertices)
        if block_map is not None:
            maps.add(tuple(block_map))
    return sorted(maps)


def apply_block_map(key, block_map):
    out = [0] * len(key)
    for index, value in enumerate(key):
        if int(value):
            out[int(block_map[int(index)])] = 1
    return tuple(out)


def canonical_key(key, block_maps):
    return min(apply_block_map(key, block_map) for block_map in block_maps)


def local_stabilizer_block_maps(parent_key, block_maps, *, quotient_root=False):
    identity = tuple(range(len(parent_key)))
    if not quotient_root and not any(int(v) for v in parent_key):
        return [identity]

    selected = {i for i, v in enumerate(parent_key) if int(v)}
    out = []
    for block_map in block_maps:
        image = {int(block_map[i]) for i in selected}
        if image == selected:
            out.append(tuple(int(v) for v in block_map))
    return out or [identity]


def support_penalty(item):
    support = item.supportability
    if support is None:
        return 0.0, 0.0
    return float(support.closer_side), float(support.supporting_shift)


def representative_sort_key(
    item,
    *,
    path,
):
    closer, shift = support_penalty(item)

    return (
        float(item.score),
        -closer,
        -shift,
        tuple(-int(v) for v in path),
    )
