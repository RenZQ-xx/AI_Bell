from __future__ import annotations

"""Bell 322 subgroup patterns derived from a facet-free group atlas.

The atlas contains only permutation-group data.  A class pattern is inferred
after a terminal support has been observed by computing its stabilizer inside
the 3072-element Bell relabeling group.  No class support is stored in, or
loaded by, this module.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping, Sequence

from baseline.orbit_blocks import (
    PartitionKey,
    Permutation,
    build_state_group_permutations,
    canonical_partition_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBGROUP_ATLAS_PATH = PROJECT_ROOT / "data" / "subgroup_search_atlas_322.json"
EXTERNAL_SUBGROUP_ATLAS_PATH = (
    PROJECT_ROOT.parent
    / "322_"
    / "data"
    / "generated"
    / "322"
    / "subgroup_atlas.json"
)
IDENTITY_PATTERN_ID = "BFS322-C1-00"


def compose_permutations(first: Permutation, second: Permutation) -> Permutation:
    """Return ``second o first`` for source-to-target permutation arrays."""
    if len(first) != len(second):
        raise ValueError("permutations must have the same degree")
    return tuple(int(second[index]) for index in first)


def inverse_permutation(permutation: Sequence[int]) -> Permutation:
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[int(target)] = int(source)
    return tuple(inverse)


def conjugate_permutation(
    permutation: Permutation,
    conjugator: Permutation,
    conjugator_inverse: Permutation | None = None,
) -> Permutation:
    """Return ``conjugator * permutation * conjugator^-1``."""
    inverse = (
        inverse_permutation(conjugator)
        if conjugator_inverse is None
        else conjugator_inverse
    )
    return tuple(
        int(conjugator[permutation[inverse[index]]])
        for index in range(len(permutation))
    )


def permutation_group_closure(
    generators: Sequence[Permutation],
    *,
    degree: int = 64,
    max_order: int | None = None,
) -> tuple[Permutation, ...] | None:
    """Generate a finite permutation subgroup, optionally aborting above a cap."""
    identity = tuple(range(int(degree)))
    normalized = tuple(tuple(int(value) for value in item) for item in generators)
    if any(len(item) != degree or set(item) != set(range(degree)) for item in normalized):
        raise ValueError(f"every generator must be a degree-{degree} permutation")

    seen = {identity}
    stack = [identity]
    while stack:
        current = stack.pop()
        for generator in normalized:
            candidate = compose_permutations(current, generator)
            if candidate in seen:
                continue
            seen.add(candidate)
            if max_order is not None and len(seen) > int(max_order):
                return None
            stack.append(candidate)
    return tuple(sorted(seen))


def subgroup_orbit_partition(
    subgroup: Sequence[Permutation],
    *,
    degree: int = 64,
) -> tuple[tuple[int, ...], ...]:
    """Return the point-orbit partition induced by a permutation subgroup."""
    if not subgroup:
        raise ValueError("subgroup must contain at least the identity")
    unused = set(range(int(degree)))
    blocks: list[tuple[int, ...]] = []
    while unused:
        seed = min(unused)
        block = tuple(sorted({int(permutation[seed]) for permutation in subgroup}))
        unused.difference_update(block)
        blocks.append(block)
    blocks.sort(key=lambda block: (-len(block), block))
    return tuple(blocks)


def support_word_from_mask(support: int | Sequence[int]) -> int:
    if isinstance(support, int):
        word = int(support)
        if word < 0 or word.bit_length() > 64:
            raise ValueError("support word must fit in 64 bits")
        return word
    values = tuple(1 if int(value) else 0 for value in support)
    if len(values) != 64:
        raise ValueError(f"support must contain 64 entries, got {len(values)}")
    return sum(1 << index for index, selected in enumerate(values) if selected)


def move_support_word(word: int, permutation: Sequence[int]) -> int:
    moved = 0
    remaining = int(word)
    while remaining:
        bit = remaining & -remaining
        source = bit.bit_length() - 1
        moved |= 1 << int(permutation[source])
        remaining ^= bit
    return moved


@lru_cache(maxsize=4096)
def canonical_support_word(word: int) -> int:
    """Return the minimum support word in the full Bell relabeling orbit."""

    normalized = support_word_from_mask(int(word))
    return min(
        move_support_word(normalized, permutation)
        for permutation in build_state_group_permutations()
    )


def subgroup_fingerprint(subgroup: Sequence[Permutation]) -> tuple[Permutation, ...]:
    return tuple(sorted(tuple(int(value) for value in item) for item in subgroup))


def subgroup_generating_set(subgroup: Sequence[Permutation]) -> tuple[Permutation, ...]:
    """Return a deterministic irredundant generating set for a finite subgroup."""
    if not subgroup:
        raise ValueError("subgroup must not be empty")
    degree = len(subgroup[0])
    identity = tuple(range(degree))
    generated: set[Permutation] = {identity}
    generators: list[Permutation] = []
    for element in sorted(subgroup):
        if element in generated:
            continue
        generators.append(element)
        closure = permutation_group_closure(tuple(generators), degree=degree)
        if closure is None:
            raise RuntimeError("unexpected capped subgroup closure")
        generated = set(closure)
        if len(generated) == len(subgroup):
            break
    return tuple(generators)


@dataclass(frozen=True)
class SubgroupPattern:
    """One conjugacy representative and its induced search partition."""

    pattern_id: str
    level: int
    structure: str
    order: int
    generators: tuple[Permutation, ...]
    generator_labels: tuple[str, ...]
    blocks: tuple[tuple[int, ...], ...]
    parent_root_ids: tuple[str, ...] = ()
    macro_family: str = ""
    action_family: str = ""
    fine_type: str = ""
    reduction_scale: str = ""
    projected_point_count: int | None = None
    fixed_polytope_dimension: int | None = None
    source_count: int = 1
    canonical_partition: PartitionKey | None = None

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(len(block) for block in self.blocks)

    @property
    def is_pair_pattern(self) -> bool:
        return (
            self.level == 1
            and self.structure == "C2"
            and len(self.blocks) == 32
            and all(len(block) == 2 for block in self.blocks)
        )

    def to_dict(self, *, include_blocks: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "pattern_id": self.pattern_id,
            "level": self.level,
            "structure": self.structure,
            "order": self.order,
            "generator_labels": list(self.generator_labels),
            "parent_root_ids": list(self.parent_root_ids),
            "macro_family": self.macro_family,
            "action_family": self.action_family,
            "fine_type": self.fine_type,
            "reduction_scale": self.reduction_scale,
            "block_count": len(self.blocks),
            "block_sizes": list(self.block_sizes),
            "is_pair_pattern": self.is_pair_pattern,
            "projected_point_count": self.projected_point_count,
            "fixed_polytope_dimension": self.fixed_polytope_dimension,
            "source_count": self.source_count,
        }
        if include_blocks:
            payload["blocks"] = [list(block) for block in self.blocks]
        return payload


@dataclass(frozen=True)
class SubgroupGrowthEdge:
    edge_id: str
    parent_id: str
    child_id: str
    relation: str
    extension_label: str = ""


@dataclass(frozen=True)
class SupportPatternAnalysis:
    """Subgroup information inferred from one observed terminal support."""

    support_word: int
    support_size: int
    stabilizer_order: int
    stabilizer_blocks: tuple[tuple[int, ...], ...]
    stabilizer_generators: tuple[Permutation, ...]
    root_pattern_ids: tuple[str, ...]
    root_subgroup_multiplicities: tuple[tuple[str, int], ...]
    level2_pattern_ids: tuple[str, ...] = ()
    level2_partition_aliases: tuple[tuple[str, ...], ...] = ()
    level2_pair_checks: int = 0
    level2_analysis_truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "support_word_hex": f"0x{self.support_word:016x}",
            "support_size": self.support_size,
            "stabilizer_order": self.stabilizer_order,
            "stabilizer_block_count": len(self.stabilizer_blocks),
            "stabilizer_orbit_sizes": [len(block) for block in self.stabilizer_blocks],
            "stabilizer_generators": [
                list(permutation) for permutation in self.stabilizer_generators
            ],
            "root_pattern_ids": list(self.root_pattern_ids),
            "root_subgroup_multiplicities": {
                pattern_id: count
                for pattern_id, count in self.root_subgroup_multiplicities
            },
            "level2_pattern_ids": list(self.level2_pattern_ids),
            "level2_partition_aliases": [
                list(pattern_ids) for pattern_ids in self.level2_partition_aliases
            ],
            "level2_pair_checks": self.level2_pair_checks,
            "level2_analysis_truncated": self.level2_analysis_truncated,
        }


@dataclass
class SubgroupPatternAtlas:
    """Search-ready view of the Bell 322 two-level subgroup DAG."""

    source_path: Path
    patterns: tuple[SubgroupPattern, ...]
    edges: tuple[SubgroupGrowthEdge, ...]
    source_metadata: Mapping[str, object] = field(default_factory=dict)
    _partition_indices: dict[int, dict[PartitionKey, tuple[str, ...]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _root_subgroup_index: dict[tuple[Permutation, ...], str] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        ids = [pattern.pattern_id for pattern in self.patterns]
        if len(ids) != len(set(ids)):
            raise ValueError("subgroup atlas contains duplicate pattern ids")
        node_ids = set(ids)
        for edge in self.edges:
            if edge.parent_id not in node_ids or edge.child_id not in node_ids:
                raise ValueError(f"growth edge {edge.edge_id} references an unknown node")

    @property
    def by_id(self) -> dict[str, SubgroupPattern]:
        return {pattern.pattern_id: pattern for pattern in self.patterns}

    @property
    def identity(self) -> SubgroupPattern:
        return self.by_id[IDENTITY_PATTERN_ID]

    @property
    def roots(self) -> tuple[SubgroupPattern, ...]:
        return tuple(pattern for pattern in self.patterns if pattern.level == 1)

    @property
    def level2(self) -> tuple[SubgroupPattern, ...]:
        return tuple(pattern for pattern in self.patterns if pattern.level == 2)

    @property
    def pair_roots(self) -> tuple[SubgroupPattern, ...]:
        return tuple(pattern for pattern in self.roots if pattern.is_pair_pattern)

    def children(self, root_id: str) -> tuple[tuple[SubgroupPattern, SubgroupGrowthEdge], ...]:
        by_id = self.by_id
        first_edge_by_child: dict[str, SubgroupGrowthEdge] = {}
        for edge in self.edges:
            if edge.parent_id == root_id:
                first_edge_by_child.setdefault(edge.child_id, edge)
        return tuple(
            (by_id[child_id], first_edge_by_child[child_id])
            for child_id in sorted(first_edge_by_child)
        )

    def _partition_index(self, level: int) -> dict[PartitionKey, tuple[str, ...]]:
        normalized_level = int(level)
        with self._lock:
            cached = self._partition_indices.get(normalized_level)
            if cached is not None:
                return cached

        ambient = build_state_group_permutations()
        mutable: dict[PartitionKey, list[str]] = {}
        for pattern in self.patterns:
            if pattern.level != normalized_level:
                continue
            key = (
                canonical_partition_key(pattern.blocks, ambient)
                if pattern.canonical_partition is None
                else pattern.canonical_partition
            )
            mutable.setdefault(key, []).append(pattern.pattern_id)
        built = {
            key: tuple(sorted(pattern_ids))
            for key, pattern_ids in mutable.items()
        }
        with self._lock:
            self._partition_indices[normalized_level] = built
        return built

    def _build_root_subgroup_index(self) -> dict[tuple[Permutation, ...], str]:
        with self._lock:
            if self._root_subgroup_index is not None:
                return self._root_subgroup_index

        ambient = build_state_group_permutations()
        inverses = tuple(inverse_permutation(item) for item in ambient)
        index: dict[tuple[Permutation, ...], str] = {}
        for pattern in self.roots:
            subgroup = permutation_group_closure(pattern.generators)
            if subgroup is None:
                raise RuntimeError("unexpected capped root subgroup closure")
            for conjugator, inverse in zip(ambient, inverses):
                conjugate = tuple(
                    sorted(
                        conjugate_permutation(item, conjugator, inverse)
                        for item in subgroup
                    )
                )
                existing = index.setdefault(conjugate, pattern.pattern_id)
                if existing != pattern.pattern_id:
                    raise ValueError(
                        "distinct root conjugacy representatives overlap: "
                        f"{existing} and {pattern.pattern_id}"
                    )
        with self._lock:
            self._root_subgroup_index = index
        return index

    def analyze_support(
        self,
        support: int | Sequence[int],
        *,
        include_level2: bool = True,
        max_level2_pair_checks: int = 4096,
        max_level2_matches: int = 128,
    ) -> SupportPatternAnalysis:
        """Infer subgroup patterns from an already observed terminal support."""
        word = support_word_from_mask(support)
        ambient = build_state_group_permutations()
        stabilizer = tuple(
            permutation
            for permutation in ambient
            if move_support_word(word, permutation) == word
        )
        stabilizer_blocks = subgroup_orbit_partition(stabilizer)

        root_index = self._build_root_subgroup_index()
        root_instances: dict[str, set[tuple[Permutation, ...]]] = {}
        identity = tuple(range(64))
        for element in stabilizer:
            if element == identity:
                continue
            cyclic = permutation_group_closure((element,), max_order=6)
            if cyclic is None or len(cyclic) not in (2, 3):
                continue
            fingerprint = subgroup_fingerprint(cyclic)
            pattern_id = root_index.get(fingerprint)
            if pattern_id is not None:
                root_instances.setdefault(pattern_id, set()).add(fingerprint)

        level2_ids: set[str] = set()
        aliases: set[tuple[str, ...]] = set()
        pair_checks = 0
        truncated = False
        if (
            include_level2
            and len(stabilizer) >= 4
            and int(max_level2_matches) > 0
        ):
            level2_index = self._partition_index(2)
            candidate_subgroups: set[tuple[Permutation, ...]] = set()

            for element in stabilizer:
                if element == identity:
                    continue
                cyclic = permutation_group_closure((element,), max_order=6)
                if cyclic is not None and len(cyclic) in (4, 6):
                    candidate_subgroups.add(subgroup_fingerprint(cyclic))

            nonidentity = tuple(item for item in stabilizer if item != identity)
            for first, second in combinations_with_replacement(nonidentity, 2):
                if pair_checks >= max(0, int(max_level2_pair_checks)):
                    truncated = True
                    break
                pair_checks += 1
                subgroup = permutation_group_closure(
                    (first, second),
                    max_order=6,
                )
                if subgroup is None or len(subgroup) not in (4, 6):
                    continue
                candidate_subgroups.add(subgroup_fingerprint(subgroup))

            ambient_group = build_state_group_permutations()
            for fingerprint in sorted(candidate_subgroups):
                blocks = subgroup_orbit_partition(fingerprint)
                pattern_ids = level2_index.get(
                    canonical_partition_key(blocks, ambient_group),
                    (),
                )
                if not pattern_ids:
                    continue
                aliases.add(pattern_ids)
                level2_ids.update(pattern_ids)
                if len(level2_ids) >= max(0, int(max_level2_matches)):
                    truncated = True
                    break

        multiplicities = tuple(
            (pattern_id, len(instances))
            for pattern_id, instances in sorted(root_instances.items())
        )
        return SupportPatternAnalysis(
            support_word=word,
            support_size=word.bit_count(),
            stabilizer_order=len(stabilizer),
            stabilizer_blocks=stabilizer_blocks,
            stabilizer_generators=subgroup_generating_set(stabilizer),
            root_pattern_ids=tuple(pattern_id for pattern_id, _ in multiplicities),
            root_subgroup_multiplicities=multiplicities,
            level2_pattern_ids=tuple(sorted(level2_ids)),
            level2_partition_aliases=tuple(sorted(aliases)),
            level2_pair_checks=pair_checks,
            level2_analysis_truncated=truncated,
        )


def _resolve_atlas_path(path: Path | None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"subgroup atlas does not exist: {resolved}")
        return resolved
    for candidate in (DEFAULT_SUBGROUP_ATLAS_PATH, EXTERNAL_SUBGROUP_ATLAS_PATH):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Bell 322 subgroup atlas not found; expected either "
        f"{DEFAULT_SUBGROUP_ATLAS_PATH} or {EXTERNAL_SUBGROUP_ATLAS_PATH}"
    )


def _pattern_from_node(node: Mapping[str, object]) -> SubgroupPattern:
    raw_generators = node.get("generators", [])
    if not isinstance(raw_generators, list):
        raise ValueError("node generators must be a list")
    generators: list[Permutation] = []
    labels: list[str] = []
    for raw_generator in raw_generators:
        if not isinstance(raw_generator, dict):
            raise ValueError("generator record must be an object")
        permutation = tuple(int(value) for value in raw_generator["vertex_permutation"])
        generators.append(permutation)
        labels.append(str(raw_generator.get("str") or "1"))
    subgroup = permutation_group_closure(tuple(generators))
    if subgroup is None:
        raise RuntimeError("unexpected capped subgroup closure")
    blocks = subgroup_orbit_partition(subgroup)

    features = node.get("features", {})
    if not isinstance(features, dict):
        features = {}
    expected_order = int(features.get("order", len(subgroup)))
    if expected_order != len(subgroup):
        raise ValueError(
            f"node {node.get('id')} has order {expected_order}, reconstructed {len(subgroup)}"
        )
    expected_sizes = features.get("vertex_orbit_sizes")
    if expected_sizes is not None and sorted(int(value) for value in expected_sizes) != sorted(
        len(block) for block in blocks
    ):
        raise ValueError(f"node {node.get('id')} has inconsistent vertex orbit sizes")
    raw_canonical_partition = node.get("canonical_partition")
    canonical_partition = (
        None
        if raw_canonical_partition is None
        else tuple(
            tuple(int(vertex) for vertex in block)
            for block in raw_canonical_partition
        )
    )
    if canonical_partition is not None:
        if sorted(vertex for block in canonical_partition for vertex in block) != list(
            range(64)
        ):
            raise ValueError(f"node {node.get('id')} has an invalid canonical partition")
        if sorted(len(block) for block in canonical_partition) != sorted(
            len(block) for block in blocks
        ):
            raise ValueError(
                f"node {node.get('id')} canonical partition has the wrong shape"
            )

    return SubgroupPattern(
        pattern_id=str(node["id"]),
        level=int(node["level"]),
        structure=str(node["structure"]),
        order=len(subgroup),
        generators=tuple(generators),
        generator_labels=tuple(labels),
        blocks=blocks,
        parent_root_ids=tuple(str(value) for value in node.get("parent_root_ids", [])),
        macro_family=str(node.get("macro_family") or ""),
        action_family=str(node.get("action_family") or ""),
        fine_type=str(node.get("fine_type") or ""),
        reduction_scale=str(node.get("reduction_scale") or ""),
        projected_point_count=(
            None
            if features.get("projected_point_count") is None
            else int(features["projected_point_count"])
        ),
        fixed_polytope_dimension=(
            None
            if features.get("fixed_polytope_dimension") is None
            else int(features["fixed_polytope_dimension"])
        ),
        source_count=int(node.get("source_count", 1)),
        canonical_partition=canonical_partition,
    )


@lru_cache(maxsize=4)
def _load_subgroup_pattern_atlas_cached(path_string: str) -> SubgroupPatternAtlas:
    path = Path(path_string)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("scenario")) != "322":
        raise ValueError(f"expected a Bell 322 subgroup atlas, got {payload.get('scenario')!r}")
    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError("subgroup atlas must contain node and edge lists")

    identity = SubgroupPattern(
        pattern_id=IDENTITY_PATTERN_ID,
        level=0,
        structure="C1",
        order=1,
        generators=(),
        generator_labels=(),
        blocks=tuple((index,) for index in range(64)),
        macro_family="C1:identity",
        action_family="identity",
        fine_type="identity",
        reduction_scale="none",
        projected_point_count=64,
        fixed_polytope_dimension=26,
        canonical_partition=tuple((index,) for index in range(64)),
    )
    patterns = (identity, *(_pattern_from_node(node) for node in raw_nodes))
    edges = tuple(
        SubgroupGrowthEdge(
            edge_id=str(edge["id"]),
            parent_id=str(edge["parent_id"]),
            child_id=str(edge["child_id"]),
            relation=str(edge.get("relation") or ""),
            extension_label=str(edge.get("extension_str") or ""),
        )
        for edge in raw_edges
    )
    atlas = SubgroupPatternAtlas(
        source_path=path,
        patterns=patterns,
        edges=edges,
        source_metadata=(
            payload.get("source")
            if isinstance(payload.get("source"), dict)
            else {
                "schema_version": payload.get("schema_version"),
                "created_at": payload.get("created_at"),
            }
        ),
    )
    if len(atlas.roots) != 24 or len(atlas.level2) != 239:
        raise ValueError(
            "unexpected Bell 322 atlas size: "
            f"roots={len(atlas.roots)}, level2={len(atlas.level2)}"
        )
    if len(atlas.pair_roots) != 18:
        raise ValueError(f"expected 18 pair roots, got {len(atlas.pair_roots)}")
    return atlas


def load_subgroup_pattern_atlas(path: Path | None = None) -> SubgroupPatternAtlas:
    return _load_subgroup_pattern_atlas_cached(str(_resolve_atlas_path(path)))


def unique_patterns(patterns: Iterable[SubgroupPattern]) -> tuple[SubgroupPattern, ...]:
    """Deduplicate patterns by their group-canonical vertex partition."""
    ambient = build_state_group_permutations()
    by_partition: dict[PartitionKey, SubgroupPattern] = {}
    for pattern in patterns:
        key = (
            canonical_partition_key(pattern.blocks, ambient)
            if pattern.canonical_partition is None
            else pattern.canonical_partition
        )
        by_partition.setdefault(key, pattern)
    return tuple(sorted(by_partition.values(), key=lambda item: item.pattern_id))
