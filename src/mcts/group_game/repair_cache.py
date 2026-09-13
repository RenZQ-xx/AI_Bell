from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np

from mcts.subgroup_patterns import conjugate_permutation, inverse_permutation, move_support_word
from .knowledge import PatternSpec
from .endpoint_graph import EndpointGraph


def transported_pattern(pattern: PatternSpec, permutation) -> PatternSpec:
    blocks = tuple(sorted((tuple(sorted(permutation[v] for v in b)) for b in pattern.blocks),
                          key=lambda b: (-len(b), b)))
    fingerprint = hashlib.sha256(repr(blocks).encode("ascii")).hexdigest()[:16]
    return PatternSpec(f"EMBED-{fingerprint}", pattern.level, pattern.structure,
                       blocks, "online_facet_subgroup")


@dataclass(frozen=True)
class CachedRepair:
    retained: int
    exits: tuple[int, ...]
    pattern: PatternSpec
    generators: tuple[tuple[int, ...], ...]


class SymmetryRepairCache:
    """Only verified geometry is transported; class labels/rewards are absent."""

    def __init__(self):
        self.coordinates = {}
        self.pools: dict[int, list[CachedRepair]] = {}
        self.keys: dict[int, set[tuple]] = {}
        self.exit_uses: Counter[tuple[int, int, int]] = Counter()
        self.untried_exits: dict[int, set[tuple[int, int]]] = {}
        self.endpoints = EndpointGraph()

    def register(self, word: int, group: np.ndarray):
        if word not in self.coordinates:
            membership = np.asarray([(word >> i) & 1 for i in range(64)], dtype=np.uint64)
            weights = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
            words = np.sum(membership[group] * weights, axis=1, dtype=np.uint64)
            index = int(np.argmin(words))
            from_canonical = tuple(map(int, group[index]))
            canonical = int(words[index])
            self.coordinates[word] = (canonical, inverse_permutation(from_canonical), from_canonical)
            self.pools.setdefault(canonical, [])
            self.keys.setdefault(canonical, set())
            self.untried_exits.setdefault(canonical, set())
        return self.coordinates[word]

    def publish(self, word, retained, exits, pattern, generators):
        canonical, to_canonical, _ = self.coordinates[word]
        moved_pattern = transported_pattern(pattern, to_canonical)
        moved_retained = move_support_word(retained, to_canonical)
        moved_exits = tuple(sorted(move_support_word(w, to_canonical) for w in exits))
        key = (moved_retained, moved_pattern.pattern_id, moved_exits)
        if key not in self.keys[canonical]:
            self.keys[canonical].add(key)
            self.pools[canonical].append(CachedRepair(
                moved_retained, moved_exits, moved_pattern,
                tuple(conjugate_permutation(g, to_canonical) for g in generators),
            ))
            self.untried_exits[canonical].update(
                (moved_retained, w) for w in moved_exits
                if not self.exit_uses[(canonical, moved_retained, w)])

    def exit_key(self, source, retained, added):
        canonical, to_canonical, _ = self.coordinates[source]
        return canonical, move_support_word(retained, to_canonical), move_support_word(added, to_canonical)

    def uses(self, source, retained, added):
        return self.exit_uses[self.exit_key(source, retained, added)]

    def mark_used(self, source, retained, added):
        key = self.exit_key(source, retained, added)
        self.exit_uses[key] += 1
        self.untried_exits[key[0]].discard(key[1:])
        self.endpoints.mark_used(key)
