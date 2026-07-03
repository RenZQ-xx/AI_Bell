from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from baseline2.core.candidate_feature_builder import is_supportable
from baseline2.core.clean_symmetry import canonical_key, local_stabilizer_block_maps
from baseline2.core.expansion_facts import ExpansionFactOracle, ExpansionFacts, build_expansion_facts
from baseline2.core.state_family import BlockMap
from baseline2.primitives.orbit_blocks import BlockKey, add_block, empty_key, unselected_blocks
from baseline2.primitives.supportability import SupportabilityMetrics


AffineHull = tuple[np.ndarray | None, np.ndarray]


@dataclass(frozen=True)
class CandidateFamily:
    """Canonical support family plus the next-step action orbits available from it."""

    canonical_support: BlockKey
    support_representations: tuple[BlockKey, ...] | None
    stabilizer_maps: tuple[BlockMap, ...]
    vertices: tuple[int, ...]
    affine_hull: AffineHull
    rank: int
    flat_capacity: int
    supportability: SupportabilityMetrics | None
    supportable: bool
    action_orbits: tuple[tuple[int, ...], ...]
    action_is_rank_gain: tuple[bool, ...]

    @classmethod
    def from_key(
        cls,
        key: Sequence[int],
        *,
        oracle: ExpansionFactOracle,
        block_maps: Sequence[Sequence[int]],
        quotient_root: bool,
        cache: "CandidateFamilyCache | None" = None,
        materialize_actions: bool = True,
    ) -> "CandidateFamily":
        canonical_support = _cached_canonical_key(key, block_maps, cache)
        if cache is not None and canonical_support in cache.family_cache:
            return cache.family_cache[canonical_support]
        if (
            cache is not None
            and not materialize_actions
            and canonical_support in cache.light_family_cache
        ):
            return cache.light_family_cache[canonical_support]

        stabilizer_maps = tuple(
            tuple(int(value) for value in block_map)
            for block_map in local_stabilizer_block_maps(
                canonical_support,
                block_maps,
                quotient_root=quotient_root,
            )
        )
        rank = int(oracle.affine_rank(canonical_support))
        flat_capacity = 0 if rank >= 25 else int(oracle.flat_capacity(canonical_support))
        supportability = _supportability_for_rank(oracle, canonical_support, rank)
        action_orbits = _action_orbits(canonical_support, stabilizer_maps) if materialize_actions else tuple()
        family = cls(
            canonical_support=canonical_support,
            support_representations=None,
            stabilizer_maps=stabilizer_maps,
            vertices=tuple(int(index) for index in oracle.vertex_indices(canonical_support)),
            affine_hull=oracle.affine_hull_basis(canonical_support),
            rank=rank,
            flat_capacity=flat_capacity,
            supportability=supportability,
            supportable=is_supportable(supportability),
            action_orbits=action_orbits,
            action_is_rank_gain=tuple(
                _orbit_is_rank_gain(oracle, canonical_support, rank, orbit)
                for orbit in action_orbits
            ),
        )
        if cache is not None:
            if materialize_actions:
                cache.family_cache[canonical_support] = family
            else:
                cache.light_family_cache[canonical_support] = family
        return family

    @property
    def action_structure_materialized(self) -> bool:
        return bool(self.action_orbits) or not any(int(value) == 0 for value in self.canonical_support)

    def materialize_action_structure(
        self,
        *,
        oracle: ExpansionFactOracle,
        cache: "CandidateFamilyCache | None" = None,
    ) -> "CandidateFamily":
        if self.action_structure_materialized:
            if cache is not None:
                cache.family_cache[self.canonical_support] = self
            return self
        action_orbits = _action_orbits(self.canonical_support, self.stabilizer_maps)
        materialized = CandidateFamily(
            canonical_support=self.canonical_support,
            support_representations=self.support_representations,
            stabilizer_maps=self.stabilizer_maps,
            vertices=self.vertices,
            affine_hull=self.affine_hull,
            rank=self.rank,
            flat_capacity=self.flat_capacity,
            supportability=self.supportability,
            supportable=self.supportable,
            action_orbits=action_orbits,
            action_is_rank_gain=tuple(
                _orbit_is_rank_gain(oracle, self.canonical_support, self.rank, orbit)
                for orbit in action_orbits
            ),
        )
        if cache is not None:
            cache.family_cache[self.canonical_support] = materialized
        return materialized

    @property
    def selected_blocks(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.canonical_support) if int(value))

    def evolve(
        self,
        *,
        oracle: ExpansionFactOracle,
        block_maps: Sequence[Sequence[int]],
        quotient_root: bool,
        cache: "CandidateFamilyCache | None" = None,
        materialize_children: bool = True,
    ) -> tuple["CandidateTransition", ...]:
        parent = self.materialize_action_structure(oracle=oracle, cache=cache)
        return tuple(
            CandidateTransition.from_parent_orbit(
                parent,
                action_orbit=orbit,
                oracle=oracle,
                block_maps=block_maps,
                quotient_root=quotient_root,
                cache=cache,
                materialize_child=materialize_children,
            )
            for orbit in parent.action_orbits
        )

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "canonical_support": list(self.selected_blocks),
            "representation_count": None if self.support_representations is None else len(self.support_representations),
            "stabilizer_size": len(self.stabilizer_maps),
            "vertex_count": len(self.vertices),
            "rank": int(self.rank),
            "flat_capacity": int(self.flat_capacity),
            "supportability": None if self.supportability is None else self.supportability.to_dict(),
            "supportable": bool(self.supportable),
            "action_orbits": [list(orbit) for orbit in self.action_orbits],
            "action_is_rank_gain": [bool(value) for value in self.action_is_rank_gain],
        }


@dataclass(frozen=True)
class CandidateTransition:
    """Read-only bridge from a parent CandidateFamily action orbit to its child family."""

    parent: CandidateFamily
    action_orbit: tuple[int, ...]
    canonical_action: int
    expansion_facts: ExpansionFacts
    child: CandidateFamily

    @classmethod
    def from_parent_orbit(
        cls,
        parent: CandidateFamily,
        *,
        action_orbit: Sequence[int],
        oracle: ExpansionFactOracle,
        block_maps: Sequence[Sequence[int]],
        quotient_root: bool,
        cache: "CandidateFamilyCache | None" = None,
        materialize_child: bool = True,
    ) -> "CandidateTransition":
        orbit = tuple(sorted(int(action) for action in action_orbit))
        canonical_action = max(orbit)
        facts = build_expansion_facts(oracle, parent.canonical_support, canonical_action)
        child = CandidateFamily.from_key(
            facts.key,
            oracle=oracle,
            block_maps=block_maps,
            quotient_root=quotient_root,
            cache=cache,
            materialize_actions=materialize_child,
        )
        return cls(
            parent=parent,
            action_orbit=orbit,
            canonical_action=int(canonical_action),
            expansion_facts=facts,
            child=child,
        )

    def matches_expansion_facts(self) -> bool:
        return (
            self.child.rank == self.expansion_facts.new_rank
            and self.child.flat_capacity == self.expansion_facts.flat_capacity
            and self.child.supportability == self.expansion_facts.supportability
        )


@dataclass
class CandidateFamilyCache:
    canonical_key_cache: dict[BlockKey, BlockKey] = field(default_factory=dict)
    light_family_cache: dict[BlockKey, CandidateFamily] = field(default_factory=dict)
    family_cache: dict[BlockKey, CandidateFamily] = field(default_factory=dict)


def root_candidate_family(
    *,
    block_count: int,
    oracle: ExpansionFactOracle,
    block_maps: Sequence[Sequence[int]],
    quotient_root: bool,
    cache: CandidateFamilyCache | None = None,
    materialize_actions: bool = True,
) -> CandidateFamily:
    return CandidateFamily.from_key(
        empty_key(int(block_count)),
        oracle=oracle,
        block_maps=block_maps,
        quotient_root=quotient_root,
        cache=cache,
        materialize_actions=materialize_actions,
    )


def evolve_candidate_family(
    parent: CandidateFamily,
    *,
    oracle: ExpansionFactOracle,
    block_maps: Sequence[Sequence[int]],
    quotient_root: bool,
    cache: CandidateFamilyCache | None = None,
    materialize_children: bool = True,
) -> tuple[CandidateTransition, ...]:
    return parent.evolve(
        oracle=oracle,
        block_maps=block_maps,
        quotient_root=quotient_root,
        cache=cache,
        materialize_children=materialize_children,
    )


def _cached_canonical_key(
    key: Sequence[int],
    block_maps: Sequence[Sequence[int]],
    cache: CandidateFamilyCache | None,
) -> BlockKey:
    normalized = tuple(int(value) for value in key)
    if cache is None:
        return canonical_key(normalized, block_maps)
    found = cache.canonical_key_cache.get(normalized)
    if found is None:
        found = canonical_key(normalized, block_maps)
        cache.canonical_key_cache[normalized] = found
    return found


def _supportability_for_rank(
    oracle: ExpansionFactOracle,
    key: BlockKey,
    rank: int,
) -> SupportabilityMetrics | None:
    supportability_start_rank = getattr(oracle.config, "supportability_start_rank", None)
    if supportability_start_rank is None or int(rank) < int(supportability_start_rank):
        return None
    return oracle.supportability_metrics(key)


def _action_orbits(
    canonical_support: BlockKey,
    stabilizer_maps: Sequence[BlockMap],
) -> tuple[tuple[int, ...], ...]:
    remaining = set(unselected_blocks(canonical_support))
    families: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        orbit = {
            int(block_map[int(seed)])
            for block_map in stabilizer_maps
            if int(block_map[int(seed)]) in remaining
        }
        orbit.add(int(seed))
        remaining.difference_update(orbit)
        families.append(tuple(sorted(orbit)))
    return tuple(sorted(families, key=lambda orbit: (-len(orbit), orbit)))


def _orbit_is_rank_gain(
    oracle: ExpansionFactOracle,
    key: BlockKey,
    parent_rank: int,
    action_orbit: Sequence[int],
) -> bool:
    canonical_action = max(int(action) for action in action_orbit)
    return int(oracle.affine_rank(add_block(key, canonical_action))) > int(parent_rank)
