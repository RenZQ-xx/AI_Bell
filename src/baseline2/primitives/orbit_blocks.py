from __future__ import annotations

from collections import Counter
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


@dataclass(frozen=True)
class GroupDescription:
    """Readable summary of a vertex-permutation subgroup."""

    element_count: int
    generator_words: tuple[str, ...]
    generator_orders: tuple[int, ...]
    orbit_sizes: tuple[int, ...]
    element_order_histogram: tuple[tuple[int, int], ...]
    element_words: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "element_count": self.element_count,
            "generator_words": list(self.generator_words),
            "generator_orders": list(self.generator_orders),
            "orbit_sizes": list(self.orbit_sizes),
            "element_order_histogram": [
                {"order": order, "count": count}
                for order, count in self.element_order_histogram
            ],
            "element_words": list(self.element_words),
        }


@dataclass(frozen=True)
class RepresentativeGroupDescription:
    """Best-looking stabilizer representative across a support symmetry orbit."""

    support: tuple[int, ...]
    group: GroupDescription
    score: tuple[int, int, int, int, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "support": list(self.support),
            "group": self.group.to_dict(),
            "score": list(self.score),
        }


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
        ("s_AB", lambda state: swap(swap(state, "A0", "B0"), "A1", "B1")),
        ("s_AC", lambda state: swap(swap(state, "A0", "C0"), "A1", "C1")),
        ("s_BC", lambda state: swap(swap(state, "B0", "C0"), "B1", "C1")),
        ("i_A", lambda state: swap(state, "A0", "A1")),
        ("i_B", lambda state: swap(state, "B0", "B1")),
        ("i_C", lambda state: swap(state, "C0", "C1")),
        ("f_A0", lambda state: flip(state, "A0")),
        ("f_A1", lambda state: flip(state, "A1")),
        ("f_B0", lambda state: flip(state, "B0")),
        ("f_B1", lambda state: flip(state, "B1")),
        ("f_C0", lambda state: flip(state, "C0")),
        ("f_C1", lambda state: flip(state, "C1")),
    ]


def _compose_permutations(first: Permutation, second: Permutation) -> Permutation:
    return tuple(second[index] for index in first)


def _invert_permutation(perm: Permutation) -> Permutation:
    inverse = [0] * len(perm)
    for source, target in enumerate(perm):
        inverse[int(target)] = int(source)
    return tuple(inverse)


def _permutation_order(perm: Permutation) -> int:
    identity = tuple(range(len(perm)))
    current = identity
    for order in range(1, 10_000):
        current = _compose_permutations(current, perm)
        if current == identity:
            return order
    raise ValueError("permutation order search exceeded limit")


def _word_sort_key(word: tuple[str, ...]) -> tuple[int, str]:
    return (len(word), "*".join(word))


def _format_word(word: tuple[str, ...]) -> str:
    return "id" if not word else "*".join(word)


def _split_formatted_word(word: str) -> tuple[str, ...]:
    if word == "id":
        return tuple()
    return tuple(part for part in word.split("*") if part)


def _group_description_score(description: GroupDescription) -> tuple[int, int, int, int, str]:
    tokens = tuple(token for word in description.generator_words for token in _split_formatted_word(word))
    output_flips = sum(1 for token in tokens if token.startswith("f_"))
    input_flips = sum(1 for token in tokens if token.startswith("i_"))
    return (
        len(description.generator_words),
        len(tokens),
        output_flips,
        input_flips,
        ",".join(description.generator_words),
    )


@lru_cache(maxsize=1)
def _named_generator_permutations() -> tuple[tuple[str, Permutation], ...]:
    states = list(iter_deterministic_states_322())
    state_to_id = {state: index for index, state in enumerate(states)}
    return tuple(
        (name, tuple(state_to_id[transform(state)] for state in states))
        for name, transform in _generator_definitions()
    )


GENERATOR_ALIASES = {
    "id": "id",
    "identity": "id",
    "ABswap": "s_AB",
    "ACswap": "s_AC",
    "BCswap": "s_BC",
    "FlipIn_A": "i_A",
    "FlipIn_B": "i_B",
    "FlipIn_C": "i_C",
    "FlipOut_A0": "f_A0",
    "FlipOut_A1": "f_A1",
    "FlipOut_B0": "f_B0",
    "FlipOut_B1": "f_B1",
    "FlipOut_C0": "f_C0",
    "FlipOut_C1": "f_C1",
}


def _normalize_generator_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("empty group generator name")
    return GENERATOR_ALIASES.get(normalized, normalized)


def _generator_lookup() -> dict[str, Permutation]:
    return {name: perm for name, perm in _named_generator_permutations()}


@lru_cache(maxsize=1)
def build_state_group_permutations() -> tuple[Permutation, ...]:
    """Generate the full state permutation group from the Bell 3-2-2 symmetries."""
    generator_permutations = [perm for _name, perm in _named_generator_permutations()]

    identity = tuple(range(len(generator_permutations[0])))
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


@lru_cache(maxsize=1)
def state_group_words() -> tuple[tuple[Permutation, tuple[str, ...]], ...]:
    """Return the full group with one short readable word per element."""
    generators = _named_generator_permutations()
    identity = tuple(range(64))
    words: dict[Permutation, tuple[str, ...]] = {identity: ()}
    queue = [identity]
    while queue:
        current = queue.pop(0)
        current_word = words[current]
        for name, generator in generators:
            candidate = _compose_permutations(current, generator)
            candidate_word = (*current_word, name)
            if candidate not in words or _word_sort_key(candidate_word) < _word_sort_key(words[candidate]):
                words[candidate] = candidate_word
                queue.append(candidate)
    return tuple(sorted(words.items(), key=lambda item: _word_sort_key(item[1])))


def parse_group_generator_words(values: Sequence[str] | None) -> tuple[Permutation, ...]:
    """Parse explicit subgroup generators from CLI strings.

    Each value may be a single generator, a product such as ``s_AB*f_A0``, or a
    comma-separated list of such products.  The returned permutations are the
    generators whose closure defines the subgroup.
    """
    if values is None:
        return tuple()
    lookup = _generator_lookup()
    identity = tuple(range(64))
    parsed: list[Permutation] = []
    raw_items: list[str] = []
    for value in values:
        raw_items.extend(part for part in str(value).replace(";", ",").split(",") if part.strip())
    for item in raw_items:
        word = identity
        for token in item.replace("*", " ").split():
            name = _normalize_generator_name(token)
            if name == "id":
                continue
            if name not in lookup:
                valid = ", ".join(["id", *sorted(lookup)])
                raise ValueError(f"unknown group generator {token!r}; valid generators: {valid}")
            word = _compose_permutations(word, lookup[name])
        parsed.append(word)
    return tuple(parsed)


def subgroup_closure(generators: Sequence[Sequence[int]] | None) -> tuple[Permutation, ...]:
    """Close a subgroup from explicit generator permutations."""
    identity = tuple(range(64))
    generator_perms = [tuple(int(value) for value in perm) for perm in (generators or [])]
    if not generator_perms:
        return (identity,)
    for perm in generator_perms:
        if len(perm) != 64:
            raise ValueError(f"group generator permutation must have length 64, got {len(perm)}")
    moves = tuple({*generator_perms, *(_invert_permutation(perm) for perm in generator_perms)})
    seen = {identity}
    queue = [identity]
    while queue:
        current = queue.pop()
        for move in moves:
            candidate = _compose_permutations(current, move)
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


def stabilizer_group(
    support: Sequence[int],
    group: Sequence[Sequence[int]] | None = None,
) -> tuple[Permutation, ...]:
    """Return the subgroup that fixes a support exactly."""
    key = support_key(support)
    permutations = build_state_group_permutations() if group is None else group
    return tuple(perm for perm in permutations if apply_permutation_to_support(key, perm) == key)


def group_orbits(group: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Return vertex orbits under an already-closed subgroup."""
    permutations = tuple(tuple(int(value) for value in perm) for perm in group)
    unused = set(range(64))
    orbits: list[tuple[int, ...]] = []
    while unused:
        seed = min(unused)
        orbit = {int(perm[seed]) for perm in permutations}
        block = tuple(sorted(orbit))
        unused.difference_update(block)
        orbits.append(block)
    orbits.sort(key=lambda block: (-len(block), block))
    return tuple(orbits)


def describe_group(group: Sequence[Sequence[int]]) -> GroupDescription:
    """Choose a compact readable generating set and basic subgroup statistics."""
    subgroup = tuple(tuple(int(value) for value in perm) for perm in group)
    subgroup_set = set(subgroup)
    identity = tuple(range(64))
    word_lookup = {perm: word for perm, word in state_group_words()}
    candidates = [
        (perm, word_lookup.get(perm, ("<unknown>",)))
        for perm in subgroup
        if perm != identity
    ]
    candidates.sort(key=lambda item: _word_sort_key(item[1]))

    generated = (identity,)
    chosen: list[Permutation] = []
    chosen_words: list[tuple[str, ...]] = []
    for perm, word in candidates:
        if perm in set(generated):
            continue
        trial = subgroup_closure((*chosen, perm))
        if not set(trial).issubset(subgroup_set):
            continue
        chosen.append(perm)
        chosen_words.append(word)
        generated = trial
        if set(generated) == subgroup_set:
            break

    element_words = tuple(
        _format_word(word_lookup.get(perm, ("<unknown>",)))
        for perm in sorted(subgroup, key=lambda perm: _word_sort_key(word_lookup.get(perm, ("<unknown>",))))
    )
    return GroupDescription(
        element_count=len(subgroup),
        generator_words=tuple(_format_word(word) for word in chosen_words),
        generator_orders=tuple(_permutation_order(perm) for perm in chosen),
        orbit_sizes=tuple(len(orbit) for orbit in group_orbits(subgroup)),
        element_order_histogram=tuple(sorted(Counter(_permutation_order(perm) for perm in subgroup).items())),
        element_words=element_words,
    )


def describe_stabilizer_group(support: Sequence[int]) -> GroupDescription:
    """Summarize the support stabilizer subgroup."""
    return describe_group(stabilizer_group(support))


def describe_best_stabilizer_representative(support: Sequence[int]) -> RepresentativeGroupDescription:
    """Pick the simplest stabilizer among all symmetry-equivalent supports."""
    base_key = support_key(support)
    best: RepresentativeGroupDescription | None = None
    seen: set[tuple[int, ...]] = set()
    for perm in build_state_group_permutations():
        moved = apply_permutation_to_support(base_key, perm)
        if moved in seen:
            continue
        seen.add(moved)
        description = describe_stabilizer_group(moved)
        candidate = RepresentativeGroupDescription(
            support=moved,
            group=description,
            score=_group_description_score(description),
        )
        if best is None or candidate.score < best.score:
            best = candidate
    if best is None:
        raise ValueError("support orbit is empty")
    return best


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


def build_orbit_pattern_from_support(
    support: Sequence[int],
    *,
    group: Sequence[Sequence[int]] | None = None,
) -> OrbitPattern:
    """Build the stabilizer-orbit pattern for this exact support.

    Unlike build_orbit_patterns_from_support, this does not move the support
    through the full symmetry group. It is the clean entry point for an explicit
    inequality/support supplied by the caller.
    """
    key = support_key(support)
    permutations = tuple(build_state_group_permutations() if group is None else group)
    orbits = stabilizer_orbits(key, permutations)
    target = tuple(1 if any(key[index] == 1 for index in orbit) else 0 for orbit in orbits)
    return OrbitPattern(
        class_id=None,
        rep_index=None,
        pattern_index=0,
        orbit_sizes=tuple(len(orbit) for orbit in orbits),
        target=target,
        orbits=orbits,
    )


def build_orbit_pattern_from_group(
    group: Sequence[Sequence[int]],
) -> OrbitPattern:
    """Build a block pattern directly from a subgroup's vertex orbits."""
    subgroup = subgroup_closure(group)
    orbits = group_orbits(subgroup)
    return OrbitPattern(
        class_id=None,
        rep_index=None,
        pattern_index=0,
        orbit_sizes=tuple(len(orbit) for orbit in orbits),
        target=tuple(0 for _orbit in orbits),
        orbits=orbits,
    )
