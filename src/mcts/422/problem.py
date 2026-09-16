from __future__ import annotations

"""Self-contained Bell (4,2,2) geometry and relabeling action.

The coordinate order and relabeling convention are copied from the frozen
``422_handoff/01_422_definitions/bell422.py`` contract.  Keeping the small
definition here makes a run reproducible after the handoff is archived.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Iterable, Iterator, Sequence

import numpy as np


VERTEX_COUNT = 256
AMBIENT_DIMENSION = 80
TARGET_AFFINE_RANK = 79
GROUP_ORDER = 98_304
OBSERVABLE_LABELS = ("A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1")
CORRELATOR_BASIS = tuple(
    tuple(zip(parties, settings))
    for size in range(1, 5)
    for parties in combinations(range(4), size)
    for settings in product(range(2), repeat=size)
)


def iter_deterministic_states() -> Iterator[tuple[int, ...]]:
    yield from product((-1, 1), repeat=8)


@lru_cache(maxsize=1)
def deterministic_vertices() -> np.ndarray:
    vertices = [
        [
            np.prod([state[2 * party + setting] for party, setting in correlator])
            for correlator in CORRELATOR_BASIS
        ]
        for state in iter_deterministic_states()
    ]
    result = np.asarray(vertices, dtype=np.int8)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Relabeling:
    party_permutation: tuple[int, int, int, int] = (0, 1, 2, 3)
    setting_permutations: tuple[tuple[int, int], ...] = ((0, 1),) * 4
    outcome_signs: tuple[int, ...] = (1,) * 8

    def __post_init__(self) -> None:
        if sorted(self.party_permutation) != list(range(4)):
            raise ValueError("party_permutation must permute range(4)")
        if len(self.setting_permutations) != 4 or any(
            sorted(item) != [0, 1] for item in self.setting_permutations
        ):
            raise ValueError("four binary setting permutations are required")
        if len(self.outcome_signs) != 8 or any(v not in (-1, 1) for v in self.outcome_signs):
            raise ValueError("outcome_signs must contain eight signs")


def named_relabelings() -> dict[str, Relabeling]:
    def party_swap(first: int, second: int) -> Relabeling:
        value = list(range(4))
        value[first], value[second] = value[second], value[first]
        return Relabeling(party_permutation=tuple(value))

    def setting_swap(party: int) -> Relabeling:
        value = [(0, 1)] * 4
        value[party] = (1, 0)
        return Relabeling(setting_permutations=tuple(value))

    def outcome_flip(party: int, setting: int) -> Relabeling:
        value = [1] * 8
        value[2 * party + setting] = -1
        return Relabeling(outcome_signs=tuple(value))

    result = {
        "s_AB": party_swap(0, 1),
        "s_BC": party_swap(1, 2),
        "s_CD": party_swap(2, 3),
    }
    for party, label in enumerate("ABCD"):
        result[f"i_{label}"] = setting_swap(party)
        for setting in range(2):
            result[f"f_{label}{setting}"] = outcome_flip(party, setting)
    return result


def apply_relabeling_to_state(relabeling: Relabeling, state: Sequence[int]) -> tuple[int, ...]:
    target = [0] * 8
    for party in range(4):
        for setting in range(2):
            target_party = relabeling.party_permutation[party]
            target_setting = relabeling.setting_permutations[party][setting]
            target[2 * target_party + target_setting] = (
                relabeling.outcome_signs[2 * party + setting]
                * int(state[2 * party + setting])
            )
    return tuple(target)


@lru_cache(maxsize=None)
def vertex_permutation(relabeling: Relabeling) -> tuple[int, ...]:
    states = tuple(iter_deterministic_states())
    lookup = {state: index for index, state in enumerate(states)}
    return tuple(lookup[apply_relabeling_to_state(relabeling, state)] for state in states)


@lru_cache(maxsize=1)
def vertex_symmetry_generators() -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, vertex_permutation(value)) for name, value in named_relabelings().items())


@lru_cache(maxsize=None)
def signed_coordinate_action(relabeling: Relabeling) -> tuple[tuple[int, int], ...]:
    lookup = {tuple(sorted(correlator)): i for i, correlator in enumerate(CORRELATOR_BASIS)}
    result = []
    for correlator in CORRELATOR_BASIS:
        image = []
        sign = 1
        for party, setting in correlator:
            image.append(
                (
                    relabeling.party_permutation[party],
                    relabeling.setting_permutations[party][setting],
                )
            )
            sign *= relabeling.outcome_signs[2 * party + setting]
        result.append((lookup[tuple(sorted(image))], sign))
    return tuple(result)


def compose_permutations(first: Sequence[int], second: Sequence[int]) -> tuple[int, ...]:
    if len(first) != len(second):
        raise ValueError("permutations must have equal degree")
    return tuple(int(second[int(index)]) for index in first)


def permutation_closure(
    generators: Sequence[Sequence[int]],
    *,
    max_order: int | None = None,
) -> tuple[tuple[int, ...], ...] | None:
    if not generators:
        raise ValueError("at least one generator is required")
    degree = len(generators[0])
    normalized = tuple(tuple(map(int, generator)) for generator in generators)
    identity = tuple(range(degree))
    seen = {identity}
    stack = [identity]
    while stack:
        current = stack.pop()
        for generator in normalized:
            candidate = compose_permutations(current, generator)
            if candidate in seen:
                continue
            seen.add(candidate)
            if max_order is not None and len(seen) > max_order:
                return None
            stack.append(candidate)
    return tuple(sorted(seen))


def orbit_partition(
    actions: Sequence[Sequence[int]], *, degree: int = VERTEX_COUNT
) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(degree))
    blocks: list[tuple[int, ...]] = []
    while unseen:
        seed = min(unseen)
        orbit = {int(action[seed]) for action in actions}
        unseen.difference_update(orbit)
        blocks.append(tuple(sorted(orbit)))
    return tuple(sorted(blocks, key=lambda block: (-len(block), block)))


def block_word(vertices: Iterable[int]) -> int:
    word = 0
    for vertex in vertices:
        value = int(vertex)
        if not 0 <= value < VERTEX_COUNT:
            raise ValueError(f"vertex index out of range: {value}")
        word |= 1 << value
    return word


def word_indices(word: int) -> list[int]:
    result: list[int] = []
    remaining = int(word)
    while remaining:
        bit = remaining & -remaining
        result.append(bit.bit_length() - 1)
        remaining ^= bit
    return result


def move_word(word: int, permutation: Sequence[int]) -> int:
    moved = 0
    remaining = int(word)
    while remaining:
        bit = remaining & -remaining
        source = bit.bit_length() - 1
        moved |= 1 << int(permutation[source])
        remaining ^= bit
    return moved


def validate_word(word: int) -> int:
    normalized = int(word)
    if normalized < 0 or normalized.bit_length() > VERTEX_COUNT:
        raise ValueError("support word must fit in 256 bits")
    return normalized
