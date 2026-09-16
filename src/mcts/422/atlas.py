from __future__ import annotations

"""Facet-free subgroup pattern atlas backed by the 4-2-2 handoff."""

import csv
import json
import random
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HANDOFF_ROOT = PROJECT_ROOT.parent / "422_handoff"
IDENTITY_PATTERN_ID = "BFS422-IDENTITY"
_GENERATOR_TOKEN = re.compile(r"(?:s_[A-D]{2}|i_[A-D]|f_[A-D][01])")


@dataclass(frozen=True)
class PatternRef:
    pattern_id: str
    structure: str
    generator_words: tuple[str, ...]
    generator_tokens: frozenset[str]
    block_count: int
    reduced_vertex_count: int
    irredundant_vertex_count: int
    record_path: Path | None
    source: str = "handoff_subgroup"
    family_id: str = ""

    @property
    def order(self) -> int:
        return {"C2 x C2": 4, "C4": 4, "C6": 6, "S3": 6}.get(self.structure, 1)

    @property
    def statistical_family_id(self) -> str:
        return self.family_id or self.pattern_id

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern_id": self.pattern_id,
            "structure": self.structure,
            "generator_words": list(self.generator_words),
            "block_count": self.block_count,
            "reduced_vertex_count": self.reduced_vertex_count,
            "irredundant_vertex_count": self.irredundant_vertex_count,
            "source": self.source,
            "family_id": self.family_id or self.pattern_id,
        }


class HandoffPatternAtlas:
    """Lazy structural view of the 721 completed subgroup nodes.

    Only ``complete_nodes.csv`` metadata and ``orbit_blocks.csv`` partitions are
    read.  The loader has no API for node ``facets.csv`` files or the global
    facet catalog, so known classes cannot influence Filler or Corrector.
    """

    def __init__(self, root: Path | str = DEFAULT_HANDOFF_ROOT, *, cache_size: int = 96):
        self.root = Path(root).resolve()
        index_path = self.root / "02_complete_nodes" / "complete_nodes.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing 4-2-2 handoff index: {index_path}")
        self.identity = PatternRef(
            IDENTITY_PATTERN_ID,
            "C1",
            (),
            frozenset(),
            256,
            256,
            256,
            None,
            "identity",
        )
        patterns: list[PatternRef] = []
        with index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                generators = json.loads(row["generators"])
                words = tuple(str(item["str"]) for item in generators)
                tokens = frozenset(token for word in words for token in _GENERATOR_TOKEN.findall(word))
                patterns.append(
                    PatternRef(
                        pattern_id=row["node_id"],
                        structure=row["structure"],
                        generator_words=words,
                        generator_tokens=tokens,
                        block_count=int(row["orbit_block_count"]),
                        reduced_vertex_count=int(row["reduced_vertex_count"]),
                        irredundant_vertex_count=int(row["irredundant_vertex_count"]),
                        record_path=self.root / "02_complete_nodes" / row["record_path"],
                    )
                )
        self.patterns = (self.identity, *patterns)
        self.by_id = {pattern.pattern_id: pattern for pattern in self.patterns}
        self._cache_size = max(4, int(cache_size))
        self._block_cache: OrderedDict[str, tuple[tuple[int, ...], ...]] = OrderedDict()
        self._online_blocks: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._neighbors: dict[str, tuple[str, ...]] = {}
        self._structure_ids: dict[str, tuple[str, ...]] = {
            structure: tuple(p.pattern_id for p in patterns if p.structure == structure)
            for structure in ("C2 x C2", "C4", "C6", "S3")
        }
        self.stats = defaultdict(int)

    def __len__(self) -> int:
        return len(self.patterns)

    def pattern(self, pattern_id: str) -> PatternRef:
        return self.by_id[pattern_id]

    def blocks(self, pattern_id: str) -> tuple[tuple[int, ...], ...]:
        cached = self._block_cache.get(pattern_id)
        if cached is not None:
            self._block_cache.move_to_end(pattern_id)
            self.stats["block_cache_hits"] += 1
            return cached
        pattern = self.pattern(pattern_id)
        if pattern_id in self._online_blocks:
            blocks = self._online_blocks[pattern_id]
        elif pattern.record_path is None:
            blocks = tuple((index,) for index in range(256))
        else:
            path = pattern.record_path / "orbit_blocks.csv"
            loaded: list[tuple[int, tuple[int, ...]]] = []
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    loaded.append((int(row["orbit_block_id"]), tuple(map(int, json.loads(row["vertex_indices"])))))
            loaded.sort()
            blocks = tuple(block for _index, block in loaded)
        if len(blocks) != pattern.block_count or sorted(v for block in blocks for v in block) != list(range(256)):
            raise ValueError(f"invalid orbit partition for {pattern_id}")
        self._block_cache[pattern_id] = blocks
        self._block_cache.move_to_end(pattern_id)
        while len(self._block_cache) > self._cache_size:
            self._block_cache.popitem(last=False)
        self.stats["block_cache_misses"] += 1
        return blocks

    def fair_order(self, seed: int) -> tuple[str, ...]:
        """Interleave structures before score-based progressive widening."""
        rng = random.Random(int(seed))
        lanes: dict[str, list[str]] = {
            structure: list(ids) for structure, ids in self._structure_ids.items()
        }
        for ids in lanes.values():
            rng.shuffle(ids)
        order = [IDENTITY_PATTERN_ID]
        cycle = ("C2 x C2", "C4", "C2 x C2", "C6", "C2 x C2", "S3")
        while any(lanes.values()):
            progressed = False
            for structure in cycle:
                if lanes[structure]:
                    order.append(lanes[structure].pop())
                    progressed = True
            if not progressed:
                break
        return tuple(order)

    def neighbors(self, pattern_id: str, *, limit: int = 24) -> tuple[str, ...]:
        cached = self._neighbors.get(pattern_id)
        if cached is not None:
            return cached[:limit]
        source = self.pattern(pattern_id)
        if pattern_id == IDENTITY_PATTERN_ID:
            result = tuple(
                pattern.pattern_id
                for pattern in sorted(
                    self.patterns[1:], key=lambda p: (-p.block_count, p.pattern_id)
                )[:limit]
            )
            self._neighbors[pattern_id] = result
            return result

        def distance(target: PatternRef) -> tuple[float, int, str]:
            overlap = len(source.generator_tokens & target.generator_tokens)
            union = len(source.generator_tokens | target.generator_tokens) or 1
            jaccard = overlap / union
            structure_penalty = 0 if source.structure == target.structure else 1
            block_gap = abs(source.block_count - target.block_count) / 256.0
            return (structure_penalty + block_gap - 1.5 * jaccard, -overlap, target.pattern_id)

        result = tuple(
            target.pattern_id
            for target in sorted(
                (p for p in self.patterns if p.pattern_id != pattern_id), key=distance
            )[: max(limit, 24)]
        )
        self._neighbors[pattern_id] = result
        return result[:limit]

    def register_online_pattern(
        self,
        pattern_id: str,
        blocks: Iterable[Iterable[int]],
        *,
        structure: str,
        family_id: str = "",
    ) -> PatternRef:
        if pattern_id in self.by_id:
            return self.by_id[pattern_id]
        normalized = tuple(tuple(sorted(map(int, block))) for block in blocks)
        pattern = PatternRef(
            pattern_id,
            structure,
            (),
            frozenset(),
            len(normalized),
            len(normalized),
            len(normalized),
            None,
            "online_stabilizer",
            family_id,
        )
        self.patterns = (*self.patterns, pattern)
        self.by_id[pattern_id] = pattern
        self._online_blocks[pattern_id] = normalized
        self._block_cache[pattern_id] = normalized
        self._neighbors.pop(pattern_id, None)
        return pattern

    def family_id(self, pattern_id: str) -> str:
        return self.pattern(pattern_id).statistical_family_id

    def family_embeddings(self, family_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                pattern.pattern_id
                for pattern in self.patterns
                if pattern.statistical_family_id == family_id
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_root": str(self.root),
            "pattern_count": len(self.patterns),
            "structural_pattern_count": sum(p.source == "handoff_subgroup" for p in self.patterns),
            "online_pattern_count": sum(p.source == "online_stabilizer" for p in self.patterns),
            "structures": {
                structure: len(ids) for structure, ids in self._structure_ids.items()
            },
            "loader_stats": dict(self.stats),
            "answer_files_read_by_atlas": [],
        }
