from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from baseline2.core.clean_symmetry import apply_block_map, canonical_key, local_stabilizer_block_maps
from baseline2.primitives.orbit_blocks import BlockKey, add_block, unselected_blocks


BlockMap = tuple[int, ...]


def selected_indices(key: Sequence[int]) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(key) if int(value))


def maps_sending_key_to(
    source_key: Sequence[int],
    target_key: Sequence[int],
    block_maps: Sequence[Sequence[int]],
) -> tuple[BlockMap, ...]:
    source = tuple(int(value) for value in source_key)
    target = tuple(int(value) for value in target_key)
    return tuple(
        tuple(int(value) for value in block_map)
        for block_map in block_maps
        if apply_block_map(source, block_map) == target
    )


def representation_keys(
    canonical_state_key: Sequence[int],
    block_maps: Sequence[Sequence[int]],
) -> tuple[BlockKey, ...]:
    key = tuple(int(value) for value in canonical_state_key)
    return tuple(sorted({apply_block_map(key, block_map) for block_map in block_maps}))


def representation_count(
    canonical_state_key: Sequence[int],
    block_maps: Sequence[Sequence[int]],
) -> int:
    return len({apply_block_map(canonical_state_key, block_map) for block_map in block_maps})


def permutation_order(perm: Sequence[int]) -> int:
    identity = tuple(range(len(perm)))
    current = identity
    move = tuple(int(value) for value in perm)
    for order in range(1, 10_000):
        current = tuple(move[index] for index in current)
        if current == identity:
            return order
    raise ValueError("permutation order search exceeded limit")


def stable_structure_type_id(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def action_family_signature(
    families: Sequence["ActionFamily"],
    *,
    blocks: Sequence[Sequence[int]],
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    rows = []
    for family in families:
        block_size_counts = Counter(len(blocks[int(action)]) for action in family.action_orbit)
        rows.append(
            (
                len(family.action_orbit),
                tuple(sorted((int(size), int(count)) for size, count in block_size_counts.items())),
            )
        )
    return tuple(sorted(rows))


def state_structure_type_payload(
    state: "StateFamily",
    action_families: Sequence["ActionFamily"],
    *,
    blocks: Sequence[Sequence[int]],
) -> dict[str, object]:
    order_histogram = Counter(permutation_order(block_map) for block_map in state.stabilizer_maps)
    return {
        "selected_block_count": len(state.family_id),
        "stabilizer_size": len(state.stabilizer_maps),
        "representation_count": int(state.representation_count),
        "stabilizer_order_histogram": [
            [int(order), int(count)]
            for order, count in sorted(order_histogram.items())
        ],
        "action_family_signature": [
            [int(orbit_size), [[int(size), int(count)] for size, count in block_size_counts]]
            for orbit_size, block_size_counts in action_family_signature(action_families, blocks=blocks)
        ],
    }


@dataclass(frozen=True)
class StateFamily:
    """Quotient-level state identity plus stabilizer metadata in block space."""

    concrete_key: BlockKey
    canonical_key: BlockKey
    stabilizer_maps: tuple[BlockMap, ...]
    concrete_to_canonical_maps: tuple[BlockMap, ...]
    representation_count: int

    @classmethod
    def from_key(
        cls,
        key: Sequence[int],
        block_maps: Sequence[Sequence[int]],
        *,
        quotient_root: bool,
    ) -> "StateFamily":
        concrete = tuple(int(value) for value in key)
        canonical = canonical_key(concrete, block_maps)
        return cls(
            concrete_key=concrete,
            canonical_key=canonical,
            stabilizer_maps=tuple(
                tuple(int(value) for value in block_map)
                for block_map in local_stabilizer_block_maps(canonical, block_maps, quotient_root=quotient_root)
            ),
            concrete_to_canonical_maps=maps_sending_key_to(concrete, canonical, block_maps),
            representation_count=representation_count(canonical, block_maps),
        )

    @property
    def family_id(self) -> tuple[int, ...]:
        return selected_indices(self.canonical_key)

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "concrete_support": list(selected_indices(self.concrete_key)),
            "canonical_support": list(self.family_id),
            "stabilizer_size": len(self.stabilizer_maps),
            "representation_count": int(self.representation_count),
            "concrete_to_canonical_map_count": len(self.concrete_to_canonical_maps),
        }


@dataclass(frozen=True)
class ActionFamily:
    """One orbit of available block actions under a StateFamily stabilizer."""

    action_orbit: tuple[int, ...]
    child_canonical_key: BlockKey | None = None

    @property
    def canonical_action(self) -> int:
        return max(self.action_orbit)

    def to_audit_dict(self, *, blocks: Sequence[Sequence[int]]) -> dict[str, object]:
        row = {
            "actions": list(self.action_orbit),
            "canonical_action": int(self.canonical_action),
            "block_sizes": [len(blocks[int(action)]) for action in self.action_orbit],
            "orbit_size": len(self.action_orbit),
        }
        if self.child_canonical_key is not None:
            row["child_canonical_support"] = list(selected_indices(self.child_canonical_key))
        return row


@dataclass
class FamilyAuditCache:
    canonical_key_cache: dict[BlockKey, BlockKey] = field(default_factory=dict)
    state_family_cache: dict[BlockKey, StateFamily] = field(default_factory=dict)
    action_family_cache: dict[BlockKey, tuple[ActionFamily, ...]] = field(default_factory=dict)


def cached_canonical_key(
    key: Sequence[int],
    block_maps: Sequence[Sequence[int]],
    cache: FamilyAuditCache | None,
) -> BlockKey:
    normalized = tuple(int(value) for value in key)
    if cache is None:
        return canonical_key(normalized, block_maps)
    found = cache.canonical_key_cache.get(normalized)
    if found is None:
        found = canonical_key(normalized, block_maps)
        cache.canonical_key_cache[normalized] = found
    return found


def state_family_from_key(
    key: Sequence[int],
    block_maps: Sequence[Sequence[int]],
    *,
    quotient_root: bool,
    cache: FamilyAuditCache | None = None,
) -> StateFamily:
    concrete = tuple(int(value) for value in key)
    canonical = cached_canonical_key(concrete, block_maps, cache)
    if cache is not None and canonical in cache.state_family_cache:
        base = cache.state_family_cache[canonical]
        if base.concrete_key == concrete:
            return base
        return StateFamily(
            concrete_key=concrete,
            canonical_key=base.canonical_key,
            stabilizer_maps=base.stabilizer_maps,
            concrete_to_canonical_maps=maps_sending_key_to(concrete, base.canonical_key, block_maps),
            representation_count=base.representation_count,
        )
    built = StateFamily.from_key(
        canonical,
        block_maps,
        quotient_root=quotient_root,
    )
    if cache is not None:
        cache.state_family_cache[canonical] = built
    if concrete == canonical:
        return built
    return StateFamily(
        concrete_key=concrete,
        canonical_key=built.canonical_key,
        stabilizer_maps=built.stabilizer_maps,
        concrete_to_canonical_maps=maps_sending_key_to(concrete, built.canonical_key, block_maps),
        representation_count=built.representation_count,
    )


def action_orbits_for_state(
    state: StateFamily,
) -> tuple[ActionFamily, ...]:
    remaining = set(unselected_blocks(state.canonical_key))
    families: list[ActionFamily] = []
    while remaining:
        seed = min(remaining)
        orbit = {
            int(block_map[int(seed)])
            for block_map in state.stabilizer_maps
            if int(block_map[int(seed)]) in remaining
        }
        orbit.add(int(seed))
        remaining.difference_update(orbit)
        families.append(ActionFamily(action_orbit=tuple(sorted(orbit))))
    return tuple(sorted(families, key=lambda family: (-len(family.action_orbit), family.action_orbit)))


def action_families_for_state(
    state: StateFamily,
    block_maps: Sequence[Sequence[int]],
    *,
    compute_child_canonical: bool = False,
    cache: FamilyAuditCache | None = None,
) -> tuple[ActionFamily, ...]:
    if cache is not None and not compute_child_canonical and state.canonical_key in cache.action_family_cache:
        return cache.action_family_cache[state.canonical_key]
    families = action_orbits_for_state(state)
    if not compute_child_canonical:
        if cache is not None:
            cache.action_family_cache[state.canonical_key] = families
        return families
    return tuple(
        ActionFamily(
            action_orbit=family.action_orbit,
            child_canonical_key=canonical_key(
                add_block(state.canonical_key, int(family.canonical_action)),
                state.stabilizer_maps,
            ),
        )
        for family in families
    )


def concrete_action_families(
    key: Sequence[int],
    block_maps: Sequence[Sequence[int]],
    *,
    quotient_root: bool,
) -> tuple[ActionFamily, ...]:
    concrete = tuple(int(value) for value in key)
    local_maps = local_stabilizer_block_maps(concrete, block_maps, quotient_root=quotient_root)
    families: dict[BlockKey, set[int]] = {}
    for action in unselected_blocks(concrete):
        child_key = add_block(concrete, int(action))
        child_canonical = canonical_key(child_key, local_maps)
        families.setdefault(child_canonical, set()).add(int(action))
    return tuple(
        ActionFamily(action_orbit=tuple(sorted(actions)), child_canonical_key=child_canonical)
        for child_canonical, actions in sorted(
            families.items(),
            key=lambda item: (-len(item[1]), tuple(sorted(item[1]))),
        )
    )
