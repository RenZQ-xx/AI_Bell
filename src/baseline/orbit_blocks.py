from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

from .bell322 import iter_deterministic_states_322


State = tuple[int, int, int, int, int, int]
Permutation = tuple[int, ...]
BlockKey = tuple[int, ...]


@dataclass(frozen=True)
class OrbitPattern:
    """A block decomposition induced by the stabilizer of one support."""

    class_id: int | None
    rep_index: int | None
    pattern_index: int
    orbit_sizes: tuple[int, ...]
    target: BlockKey
    orbits: tuple[tuple[int, ...], ...]

    @property
    def block_count(self) -> int:
        return len(self.orbits)


def empty_key(block_count: int) -> BlockKey:
    """The zero-start search state: no orbit block has been selected."""
    if block_count < 0:
        raise ValueError(f"block_count must be nonnegative, got {block_count}")
    return tuple(0 for _ in range(block_count))


def selected_blocks(key: BlockKey) -> list[int]:
    """Return block indices whose bit is 1."""
    return [index for index, value in enumerate(key) if int(value) == 1]


def unselected_blocks(key: BlockKey) -> list[int]:
    """Return block indices still available for add-only expansion."""
    return [index for index, value in enumerate(key) if int(value) == 0]


def add_block(key: BlockKey, block_index: int) -> BlockKey:
    """Return a new key with one additional block selected."""
    index = int(block_index)
    if index < 0 or index >= len(key):
        raise IndexError(f"block_index {index} out of range for {len(key)} blocks")
    if int(key[index]) == 1:
        return tuple(int(value) for value in key)
    values = [int(value) for value in key]
    values[index] = 1
    return tuple(values)


def support_key(support: Sequence[int]) -> tuple[int, ...]:
    """Normalize a 64-bit vertex support to a tuple of 0/1 integers."""
    key = tuple(1 if int(value) else 0 for value in support)
    if len(key) != 64:
        raise ValueError(f"support must have length 64, got {len(key)}")
    return key


def selected_vertices_from_blocks(key: BlockKey, blocks: Sequence[Sequence[int]]) -> list[int]:
    """Expand a block-level key into sorted deterministic vertex indices."""
    if len(key) != len(blocks):
        raise ValueError(f"key has {len(key)} bits but blocks has {len(blocks)} entries")
    vertices: set[int] = set()
    for block_index in selected_blocks(key):
        vertices.update(int(vertex) for vertex in blocks[block_index])
    return sorted(vertices)


def _generator_definitions() -> list[tuple[str, Callable[[State], State]]]:
    idx = {"A0": 0, "A1": 1, "B0": 2, "B1": 3, "C0": 4, "C1": 5}

    def swap(state: State, first: str, second: str) -> State:
        values = list(state)
        pos_a = idx[first]
        pos_b = idx[second]
        values[pos_a], values[pos_b] = values[pos_b], values[pos_a]
        return tuple(values)  # type: ignore[return-value]

    def flip(state: State, name: str) -> State:
        values = list(state)
        values[idx[name]] = -values[idx[name]]
        return tuple(values)  # type: ignore[return-value]

    return [
        ("ABswap", lambda state: swap(swap(state, "A0", "B0"), "A1", "B1")),
        ("ACswap", lambda state: swap(swap(state, "A0", "C0"), "A1", "C1")),
        ("FlipIn_A", lambda state: swap(state, "A0", "A1")),
        ("FlipIn_B", lambda state: swap(state, "B0", "B1")),
        ("FlipIn_C", lambda state: swap(state, "C0", "C1")),
        ("FlipOut_A0", lambda state: flip(state, "A0")),
        ("FlipOut_A1", lambda state: flip(state, "A1")),
        ("FlipOut_B0", lambda state: flip(state, "B0")),
        ("FlipOut_B1", lambda state: flip(state, "B1")),
        ("FlipOut_C0", lambda state: flip(state, "C0")),
        ("FlipOut_C1", lambda state: flip(state, "C1")),
    ]


def _compose_permutations(first: Permutation, second: Permutation) -> Permutation:
    return tuple(second[index] for index in first)


@lru_cache(maxsize=1)
def build_state_group_permutations() -> tuple[Permutation, ...]:
    """Generate the full state permutation group from the Bell 3-2-2 symmetries."""
    states = list(iter_deterministic_states_322())
    state_to_id = {state: index for index, state in enumerate(states)}
    generator_permutations = [
        tuple(state_to_id[transform(state)] for state in states)
        for _name, transform in _generator_definitions()
    ]

    identity = tuple(range(len(states)))
    seen = {identity}
    queue = [identity]
    while queue:
        current = queue.pop()
        for generator in generator_permutations:
            candidate = _compose_permutations(current, generator)
            if candidate not in seen:
                seen.add(candidate)
                queue.append(candidate)
    return tuple(sorted(seen))


def apply_permutation_to_support(support: Sequence[int], perm: Sequence[int]) -> tuple[int, ...]:
    """Move a 64-bit support by a vertex permutation."""
    key = support_key(support)
    if len(perm) != len(key):
        raise ValueError(f"permutation has length {len(perm)} but support has length {len(key)}")
    moved = [0] * len(key)
    for index, value in enumerate(key):
        if value == 1:
            moved[int(perm[index])] = 1
    return tuple(moved)


def stabilizer_orbits(
    support: Sequence[int],
    group: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return vertex orbits under the subgroup that stabilizes a support."""
    key = support_key(support)
    permutations = build_state_group_permutations() if group is None else group
    stabilizer = [perm for perm in permutations if apply_permutation_to_support(key, perm) == key]
    unused = set(range(len(key)))
    orbits: list[tuple[int, ...]] = []
    while unused:
        seed = min(unused)
        orbit = {int(perm[seed]) for perm in stabilizer}
        block = tuple(sorted(orbit))
        unused.difference_update(block)
        orbits.append(block)
    orbits.sort(key=lambda block: (-len(block), block))
    return tuple(orbits)


def build_orbit_patterns_from_support(
    support: Sequence[int],
    *,
    class_id: int | None = None,
    rep_index: int | None = None,
    group: Sequence[Sequence[int]] | None = None,
    max_patterns: int | None = None,
) -> tuple[OrbitPattern, ...]:
    """Build all distinct stabilizer-orbit patterns generated by one support.

    The baseline uses this to choose a block system.  We take one known support,
    move it through the symmetry group, and for each distinct moved support take
    the vertex orbits of its stabilizer as the available search blocks.
    """
    base_key = support_key(support)
    permutations = tuple(build_state_group_permutations() if group is None else group)
    seen: set[tuple[int, ...]] = set()
    patterns: list[OrbitPattern] = []
    for perm in permutations:
        if max_patterns is not None and len(patterns) >= int(max_patterns):
            break
        moved = apply_permutation_to_support(base_key, perm)
        if moved in seen:
            continue
        seen.add(moved)
        orbits = stabilizer_orbits(moved, permutations)
        target = tuple(1 if any(moved[index] == 1 for index in orbit) else 0 for orbit in orbits)
        patterns.append(
            OrbitPattern(
                class_id=class_id,
                rep_index=rep_index,
                pattern_index=len(patterns),
                orbit_sizes=tuple(len(orbit) for orbit in orbits),
                target=target,
                orbits=orbits,
            )
        )
    return tuple(patterns)
