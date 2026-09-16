from __future__ import annotations

"""Cheap partition certificates used before expensive group computations."""

from collections import OrderedDict
from collections.abc import Iterable, Sequence


def geometric_state_key(
    support_word: int,
    role: str,
    corrections_used: int,
    repair_source_word: int = 0,
) -> tuple[int, str, int, int]:
    """Return path-independent MCTS identity for one geometric state."""

    return (
        int(support_word),
        str(role),
        int(corrections_used),
        int(repair_source_word),
    )


class PatternRelationIndex:
    """Classify pattern changes without enumerating their permutation groups.

    Orbit partitions are certificates: a union of blocks is invariant under the
    inducing subgroup. Partition refinement is therefore a safe, inexpensive
    proxy for subgroup descent in large permutation actions.
    """

    def __init__(self, vertex_count: int, *, cache_size: int = 32_768) -> None:
        if vertex_count <= 0:
            raise ValueError("vertex_count must be positive")
        self.vertex_count = int(vertex_count)
        self.full_mask = (1 << self.vertex_count) - 1
        self.cache_size = max(0, int(cache_size))
        self._blocks: dict[str, tuple[int, ...]] = {}
        self._labels: dict[str, tuple[int, ...]] = {}
        self._relation_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._directed_relation_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._coherence_cache: OrderedDict[tuple[int, str], float] = OrderedDict()

    def register(self, pattern_id: str, blocks: Iterable[Iterable[int]]) -> None:
        if pattern_id in self._blocks:
            return
        words: list[int] = []
        labels = [-1] * self.vertex_count
        for block_index, block in enumerate(blocks):
            word = 0
            for raw_vertex in block:
                vertex = int(raw_vertex)
                if not 0 <= vertex < self.vertex_count:
                    raise ValueError(f"vertex index out of range: {vertex}")
                if labels[vertex] >= 0:
                    raise ValueError("pattern blocks must be disjoint")
                labels[vertex] = block_index
                word |= 1 << vertex
            if not word:
                raise ValueError("pattern blocks must not be empty")
            words.append(word)
        if any(label < 0 for label in labels):
            raise ValueError("pattern blocks must cover every vertex")
        self._blocks[str(pattern_id)] = tuple(words)
        self._labels[str(pattern_id)] = tuple(labels)

    def registered(self, pattern_id: str) -> bool:
        return str(pattern_id) in self._blocks

    @property
    def pattern_count(self) -> int:
        return len(self._blocks)

    def coherence(self, support_word: int, pattern_id: str) -> float:
        key = int(support_word), str(pattern_id)
        cached = self._coherence_cache.get(key)
        if cached is not None:
            self._coherence_cache.move_to_end(key)
            return cached
        support = int(support_word) & self.full_mask
        if support == 0:
            value = 1.0
        else:
            defect = 0
            for block in self._blocks[str(pattern_id)]:
                selected = (support & block).bit_count()
                defect += min(selected, block.bit_count() - selected)
            value = max(0.0, 1.0 - defect / support.bit_count())
        self._remember(self._coherence_cache, key, value)
        return value

    def relation(self, current_pattern_id: str, target_pattern_id: str) -> str:
        current = str(current_pattern_id)
        target = str(target_pattern_id)
        if current == target:
            return "continue"
        key = current, target
        cached = self._relation_cache.get(key)
        if cached is not None:
            self._relation_cache.move_to_end(key)
            return cached
        refines = self._refines(target, current)
        relation = "refine" if refines else "intersection"
        self._remember(self._relation_cache, key, relation)
        return relation

    def directed_relation(
        self,
        current_pattern_id: str,
        target_pattern_id: str,
    ) -> str:
        """Return the directed refinement relation between two partitions."""

        current = str(current_pattern_id)
        target = str(target_pattern_id)
        if current == target:
            return "continue"
        key = current, target
        cached = self._directed_relation_cache.get(key)
        if cached is not None:
            self._directed_relation_cache.move_to_end(key)
            return cached
        if self._refines(target, current):
            relation = "refine"
        elif self._refines(current, target):
            relation = "coarsen"
        else:
            relation = "intersection"
        self._remember(self._directed_relation_cache, key, relation)
        return relation

    def _refines(self, fine_pattern_id: str, coarse_pattern_id: str) -> bool:
        coarse_labels = self._labels[coarse_pattern_id]
        return all(
            len({coarse_labels[v] for v in self._indices(fine_block)}) == 1
            for fine_block in self._blocks[fine_pattern_id]
        )

    def common_refinement_size(self, pattern_ids: Sequence[str]) -> int:
        normalized = tuple(dict.fromkeys(map(str, pattern_ids)))
        if not normalized:
            return 1
        signatures = {
            tuple(self._labels[pattern_id][vertex] for pattern_id in normalized)
            for vertex in range(self.vertex_count)
        }
        return len(signatures)

    @staticmethod
    def _indices(word: int):
        remaining = int(word)
        while remaining:
            bit = remaining & -remaining
            yield bit.bit_length() - 1
            remaining ^= bit

    def _remember(self, cache: OrderedDict, key, value) -> None:
        cache[key] = value
        if self.cache_size and len(cache) > self.cache_size:
            cache.popitem(last=False)
