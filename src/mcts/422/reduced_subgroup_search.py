from __future__ import annotations

"""MCTS over one subgroup's irredundant symmetry-reduced polytope.

Search happens entirely in intrinsic reduced coordinates.  A reduced facet is
lifted to the original 256 deterministic vertices only after it has passed the
reduced supporting-hyperplane test.  Known node facets are used solely as an
external coverage checklist.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog

from .atlas import DEFAULT_HANDOFF_ROOT
from .geometry import FacetOracle, FacetResult
from .subgroup_benchmark import load_target_class_ids


DEFAULT_NODE_IDS = (
    "BFS422-H-0000",  # C2 x C2: d=23, 48 vertices, 16 successful lifts
    "BFS422-H-0022",  # C4:      d=22, 48 vertices, 5 successful lifts
    "BFS422-H-0110",  # C6:      d=16, 38 vertices, 6 successful lifts
    "BFS422-H-0863",  # S3:      d=17, 42 vertices, 4 successful lifts
)
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs" / "reduced_subgroups"
ROLLOUT_CYCLE = ("guided", "guided", "novel", "guided", "random", "novel")


def word_indices(word: int) -> tuple[int, ...]:
    result: list[int] = []
    value = int(word)
    while value:
        low = value & -value
        result.append(low.bit_length() - 1)
        value ^= low
    return tuple(result)


def word_from_indices(indices: Iterable[int]) -> int:
    return sum(1 << int(index) for index in indices)


@dataclass(frozen=True)
class ReducedSearchConfig:
    iterations: int = 1_000
    seed: int = 4_222_025
    max_depth: int = 48
    max_corrections: int = 5
    productive_max_corrections: int = 7
    exploration: float = 1.30
    discount: float = 0.985
    widening_k0: int = 3
    widening_alpha: float = 1.8
    widening_beta: float = 0.50
    max_children: int = 96
    tree_face_samples: int = 6
    rollout_face_samples: int = 2
    corrector_candidates: int = 12
    corrector_face_bank_limit: int = 64
    lower_face_fraction: float = 0.34
    lower_face_pair_trials: int = 36
    rollout_exit_filter_bits: int = 1 << 18
    complete_sampled_face: bool = True
    early_bundle_size: int = 4
    middle_bundle_size: int = 3
    late_bundle_size: int = 2
    new_class_reward: float = 10.0
    known_class_reward: float = 0.08
    new_reduced_facet_reward: float = 0.12
    new_lifted_facet_reward: float = 0.35
    known_reduced_facet_reward: float = -0.02
    failed_lift_reward: float = 0.0
    new_root_basin_reward: float = 0.18
    new_ridge_exit_reward: float = 0.08
    new_lower_face_exit_reward: float = 0.24
    lower_face_prior_boost: float = 0.30
    rank_gain_reward: float = 0.045
    correction_cost: float = 0.015
    source_return_penalty: float = 0.30
    dead_end_reward: float = -0.50
    progress_interval: int = 100
    max_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.max_depth <= 0:
            raise ValueError("iterations and max_depth must be positive")
        if self.max_corrections < 0:
            raise ValueError("max_corrections must be nonnegative")
        if self.productive_max_corrections < self.max_corrections:
            raise ValueError(
                "productive_max_corrections must be at least max_corrections"
            )
        if not 0.0 <= self.lower_face_fraction <= 1.0:
            raise ValueError("lower_face_fraction must be in [0, 1]")
        if (
            self.rollout_exit_filter_bits <= 0
            or self.rollout_exit_filter_bits
            & (self.rollout_exit_filter_bits - 1)
        ):
            raise ValueError("rollout_exit_filter_bits must be a positive power of two")
        if min(
            self.tree_face_samples,
            self.rollout_face_samples,
            self.corrector_candidates,
            self.corrector_face_bank_limit,
            self.lower_face_pair_trials,
            self.early_bundle_size,
            self.middle_bundle_size,
            self.late_bundle_size,
        ) <= 0:
            raise ValueError("candidate budgets must be positive")


@dataclass
class SharedDiscoveryState:
    """Cross-lane novelty state for globally deduplicated lifted facets."""

    discovered_classes: dict[str, dict[str, object]] = field(default_factory=dict)
    class_frequencies: Counter[str] = field(default_factory=Counter)
    discovery_events: list[dict[str, object]] = field(default_factory=list)
    epoch: int = 0
    started_at: float = field(default_factory=time.perf_counter, repr=False)

    def observe(
        self,
        class_id: str,
        event: dict[str, object] | None,
    ) -> tuple[bool, int]:
        previous_hits = self.class_frequencies[class_id]
        self.class_frequencies[class_id] += 1
        is_new = class_id not in self.discovered_classes
        if not is_new:
            return False, previous_hits
        if event is None:
            raise ValueError("a globally new class requires a discovery event")
        recorded = dict(event)
        recorded["elapsed_seconds"] = time.perf_counter() - self.started_at
        self.discovered_classes[class_id] = recorded
        self.discovery_events.append(recorded)
        self.epoch += 1
        return True, previous_hits


@dataclass(frozen=True)
class ReducedSubgroupProblem:
    node_id: str
    structure: str
    intrinsic_dimension: int
    all_points: np.ndarray
    irredundant_points: np.ndarray
    irredundant_to_reduced: tuple[int, ...]
    reduced_to_irredundant: tuple[int | None, ...]
    reduced_original_words: tuple[int, ...]
    orbit_block_count: int
    record_path: Path

    @property
    def vertex_count(self) -> int:
        return len(self.irredundant_points)

    @property
    def reduced_point_count(self) -> int:
        return len(self.all_points)

    @property
    def facet_rank(self) -> int:
        return self.intrinsic_dimension - 1

    @property
    def ridge_rank(self) -> int:
        return self.intrinsic_dimension - 2

    @classmethod
    def load(
        cls,
        handoff_root: Path | str,
        node_id: str,
    ) -> "ReducedSubgroupProblem":
        root = Path(handoff_root).resolve() / "02_complete_nodes"
        metadata = None
        with (root / "complete_nodes.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["node_id"] == node_id:
                    metadata = row
                    break
        if metadata is None:
            raise ValueError(f"unknown complete subgroup node: {node_id}")
        record_path = root / metadata["record_path"]
        reduced_rows: list[dict[str, str]] = []
        with (record_path / "reduced_vertices.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            reduced_rows.extend(csv.DictReader(handle))
        reduced_rows.sort(key=lambda row: int(row["reduced_vertex_id"]))
        all_points = np.asarray(
            [json.loads(row["intrinsic_reduced_vertex"]) for row in reduced_rows],
            dtype=float,
        )
        irredundant_rows = [
            row
            for row in reduced_rows
            if row["is_irredundant_vertex"].strip().lower() == "true"
        ]
        irredundant_rows.sort(key=lambda row: int(row["irredundant_vertex_id"]))
        irredundant_to_reduced = tuple(
            int(row["reduced_vertex_id"]) for row in irredundant_rows
        )
        reduced_to_irredundant: list[int | None] = [None] * len(reduced_rows)
        for irredundant_id, reduced_id in enumerate(irredundant_to_reduced):
            reduced_to_irredundant[reduced_id] = irredundant_id
        reduced_original_words = [0] * len(reduced_rows)
        orbit_count = 0
        with (record_path / "orbit_blocks.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                orbit_count += 1
                reduced_id = int(row["reduced_vertex_id"])
                reduced_original_words[reduced_id] |= word_from_indices(
                    json.loads(row["vertex_indices"])
                )
        dimension = all_points.shape[1]
        irredundant_points = all_points[list(irredundant_to_reduced)]
        affine_rank = int(
            np.linalg.matrix_rank(
                irredundant_points[1:] - irredundant_points[0], tol=1e-8
            )
        )
        if affine_rank != dimension:
            raise ValueError(
                f"{node_id} intrinsic rank mismatch: {affine_rank} != {dimension}"
            )
        return cls(
            node_id=node_id,
            structure=metadata["structure"],
            intrinsic_dimension=dimension,
            all_points=all_points,
            irredundant_points=irredundant_points,
            irredundant_to_reduced=irredundant_to_reduced,
            reduced_to_irredundant=tuple(reduced_to_irredundant),
            reduced_original_words=tuple(reduced_original_words),
            orbit_block_count=orbit_count,
            record_path=record_path,
        )

    def original_support(self, tight_reduced_ids: Iterable[int]) -> int:
        word = 0
        for reduced_id in tight_reduced_ids:
            word |= self.reduced_original_words[int(reduced_id)]
        return word

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "structure": self.structure,
            "intrinsic_dimension": self.intrinsic_dimension,
            "reduced_point_count": self.reduced_point_count,
            "irredundant_vertex_count": self.vertex_count,
            "orbit_block_count": self.orbit_block_count,
            "reduced_facet_affine_rank": self.facet_rank,
            "reduced_ridge_affine_rank": self.ridge_rank,
            "full_lift_affine_rank": 79,
        }


class ReducedRankOracle:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.cache: dict[int, int] = {0: -1}
        self.hits = 0
        self.misses = 0

    def rank(self, word: int) -> int:
        cached = self.cache.get(int(word))
        if cached is not None:
            self.hits += 1
            return cached
        indices = word_indices(word)
        if len(indices) <= 1:
            value = len(indices) - 1
        else:
            selected = self.points[list(indices)]
            value = int(
                np.linalg.matrix_rank(selected[1:] - selected[0], tol=1e-8)
            )
        self.cache[int(word)] = value
        self.misses += 1
        return value

    def to_dict(self) -> dict[str, int]:
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
        }


@dataclass(frozen=True)
class ReducedFacetResult:
    valid: bool
    reason: str
    selected_word: int
    closure_word: int = 0
    tight_reduced_ids: tuple[int, ...] = ()
    original_support_word: int = 0
    full_result: FacetResult | None = None

    @property
    def lifted(self) -> bool:
        return bool(self.full_result and self.full_result.valid)


class ReducedGeometry:
    def __init__(
        self,
        problem: ReducedSubgroupProblem,
        *,
        handoff_root: Path | str,
        full_oracle: FacetOracle | None = None,
    ) -> None:
        self.problem = problem
        self.rank_oracle = ReducedRankOracle(problem.irredundant_points)
        self.full_oracle = full_oracle or FacetOracle(
            handoff_root=handoff_root, use_catalog=True, canonicalize=True
        )
        self.shared_full_oracle = full_oracle is not None
        self.irredundant_augmented = np.column_stack(
            (np.ones(problem.vertex_count), problem.irredundant_points)
        )
        self.all_augmented = np.column_stack(
            (np.ones(problem.reduced_point_count), problem.all_points)
        )
        self.slack_sum_row = self.irredundant_augmented.sum(axis=0)
        self.facet_cache: dict[int, ReducedFacetResult] = {}
        self.ridge_coordinate_cache: dict[int, tuple[tuple[int, ...], np.ndarray]] = {}
        self.stats: Counter[str] = Counter()

    def sample_supporting_facets(
        self,
        support_word: int,
        *,
        count: int,
        rng: random.Random,
    ) -> tuple[int, ...]:
        selected = word_indices(support_word)
        if self.rank_oracle.rank(support_word) >= self.problem.facet_rank:
            result = self.validate(support_word)
            return (result.closure_word,) if result.valid else ()
        equalities = [self.irredundant_augmented[index] for index in selected]
        equalities.append(self.slack_sum_row)
        a_eq = np.asarray(equalities, dtype=float)
        b_eq = np.asarray([0.0] * len(selected) + [1.0], dtype=float)
        a_ub = -self.irredundant_augmented
        b_ub = np.zeros(self.problem.vertex_count, dtype=float)
        facets: list[int] = []
        attempts = max(count * 5, count + 2)
        for _ in range(attempts):
            self.stats["support_lp_attempts"] += 1
            objective = np.asarray(
                [rng.gauss(0.0, 1.0) for _ in range(a_eq.shape[1])],
                dtype=float,
            )
            result = linprog(
                objective,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=[(None, None)] * a_eq.shape[1],
                method="highs",
            )
            if not result.success:
                self.stats["support_lp_failures"] += 1
                continue
            slacks = self.irredundant_augmented @ result.x
            tolerance = 2e-7 * max(1.0, float(np.max(np.abs(slacks))))
            closure = word_from_indices(np.flatnonzero(np.abs(slacks) <= tolerance))
            if support_word & ~closure:
                self.stats["support_lp_lost_selected"] += 1
                continue
            if self.rank_oracle.rank(closure) != self.problem.facet_rank:
                self.stats["support_lp_nonfacet"] += 1
                continue
            if closure not in facets:
                facets.append(closure)
                self.stats["sampled_reduced_facets"] += 1
                if len(facets) >= count:
                    break
            else:
                self.stats["sampled_facet_duplicates"] += 1
        return tuple(facets)

    def sample_ridges(
        self,
        facet_word: int,
        *,
        count: int,
        rng: random.Random,
    ) -> tuple[int, ...]:
        cached = self.ridge_coordinate_cache.get(facet_word)
        if cached is None:
            facet_indices = word_indices(facet_word)
            points = self.problem.irredundant_points[list(facet_indices)]
            centered = points - points[0]
            _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
            rank = int(np.count_nonzero(singular > 1e-8))
            if rank != self.problem.facet_rank:
                self.stats["ridge_coordinate_rank_failures"] += 1
                return ()
            local_points = centered @ vh[:rank].T
            cached = facet_indices, local_points
            self.ridge_coordinate_cache[facet_word] = cached
        facet_indices, local_points = cached
        augmented = np.column_stack((np.ones(len(local_points)), local_points))
        a_eq = augmented.sum(axis=0, keepdims=True)
        b_eq = np.asarray([1.0])
        ridges: list[int] = []
        for _ in range(max(count * 5, count + 2)):
            self.stats["ridge_lp_attempts"] += 1
            objective = np.asarray(
                [rng.gauss(0.0, 1.0) for _ in range(augmented.shape[1])],
                dtype=float,
            )
            result = linprog(
                objective,
                A_ub=-augmented,
                b_ub=np.zeros(len(local_points)),
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=[(None, None)] * augmented.shape[1],
                method="highs",
            )
            if not result.success:
                self.stats["ridge_lp_failures"] += 1
                continue
            slacks = augmented @ result.x
            tolerance = 2e-7 * max(1.0, float(np.max(np.abs(slacks))))
            local_tight = np.flatnonzero(np.abs(slacks) <= tolerance)
            ridge_word = word_from_indices(facet_indices[int(i)] for i in local_tight)
            if self.rank_oracle.rank(ridge_word) != self.problem.ridge_rank:
                self.stats["ridge_lp_nonridge"] += 1
                continue
            if ridge_word not in ridges:
                ridges.append(ridge_word)
                self.stats["sampled_ridges"] += 1
                if len(ridges) >= count:
                    break
            else:
                self.stats["sampled_ridge_duplicates"] += 1
        return tuple(ridges)

    def _lift_reduced_support(self, original_support: int) -> FacetResult | None:
        return self.full_oracle.validate(original_support)

    def validate(self, selected_word: int) -> ReducedFacetResult:
        cached = self.facet_cache.get(int(selected_word))
        if cached is not None:
            self.stats["reduced_facet_cache_hits"] += 1
            return cached
        self.stats["reduced_facet_validations"] += 1
        rank = self.rank_oracle.rank(selected_word)
        if rank != self.problem.facet_rank:
            result = ReducedFacetResult(False, "rank", int(selected_word))
            self.facet_cache[int(selected_word)] = result
            return result
        selected = word_indices(selected_word)
        matrix = self.irredundant_augmented[list(selected)]
        _u, _s, vh = np.linalg.svd(matrix, full_matrices=True)
        normal = vh[-1]
        values = self.irredundant_augmented @ normal
        tolerance = 2e-7 * max(1.0, float(np.max(np.abs(values))))
        if np.any(values > tolerance) and np.any(values < -tolerance):
            result = ReducedFacetResult(False, "cuts_reduced_polytope", int(selected_word))
            self.facet_cache[int(selected_word)] = result
            return result
        all_values = self.all_augmented @ normal
        if np.all(values <= tolerance):
            all_values = -all_values
        tight_reduced_ids = tuple(
            map(int, np.flatnonzero(np.abs(all_values) <= tolerance))
        )
        closure_indices = [
            irredundant_id
            for reduced_id in tight_reduced_ids
            if (
                irredundant_id := self.problem.reduced_to_irredundant[reduced_id]
            )
            is not None
        ]
        closure_word = word_from_indices(closure_indices)
        if self.rank_oracle.rank(closure_word) != self.problem.facet_rank:
            result = ReducedFacetResult(False, "closure_rank", int(selected_word))
            self.facet_cache[int(selected_word)] = result
            return result
        original_support = self.problem.original_support(tight_reduced_ids)
        full_result = self._lift_reduced_support(original_support)
        result = ReducedFacetResult(
            True,
            "reduced_facet",
            int(selected_word),
            closure_word,
            tight_reduced_ids,
            original_support,
            full_result,
        )
        self.facet_cache[int(selected_word)] = result
        self.facet_cache.setdefault(closure_word, result)
        if full_result is not None:
            self.stats["successful_full_lifts" if full_result.valid else "failed_full_lifts"] += 1
            if not full_result.valid:
                self.stats[f"failed_lift_{full_result.reason}"] += 1
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "statistics": dict(self.stats),
            "rank_oracle": self.rank_oracle.to_dict(),
            "reduced_facet_cache_entries": len(self.facet_cache),
            "shared_full_facet_oracle": self.shared_full_oracle,
            "full_facet_oracle": self.full_oracle.to_dict(),
        }


@dataclass(frozen=True)
class ReducedAction:
    kind: str
    add_word: int = 0
    retained_word: int = 0
    sampled_facet_word: int = 0
    source: str = ""

    @property
    def action_id(self) -> str:
        return (
            f"{self.kind}:{self.add_word:x}:{self.retained_word:x}:"
            f"{self.sampled_facet_word:x}:{self.source}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "add_vertices": list(word_indices(self.add_word)),
            "retained_vertices": list(word_indices(self.retained_word)),
            "sampled_facet_word_hex": f"0x{self.sampled_facet_word:x}",
            "source": self.source,
        }


@dataclass(frozen=True)
class ReducedState:
    support_word: int
    role: str = "filler"
    corrections_used: int = 0
    source_facet_word: int = 0

    @property
    def key(self) -> tuple[int, str, int, int]:
        return (
            self.support_word,
            self.role,
            self.corrections_used,
            self.source_facet_word,
        )


@dataclass(frozen=True)
class ReducedTransition:
    state: ReducedState
    rank: int
    terminal: ReducedFacetResult | None = None


@dataclass
class ReducedEdge:
    action: ReducedAction
    child_key: tuple[int, str, int, int]
    rank_before: int
    rank_after: int
    terminal: ReducedFacetResult | None
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    epoch: int = -1
    epoch_visits: int = 0
    epoch_value_sum: float = 0.0
    discovery_hits: int = 0

    def mean(self, epoch: int) -> float:
        if self.epoch == epoch and self.epoch_visits:
            return self.epoch_value_sum / self.epoch_visits
        return self.value_sum / self.visits if self.visits else 0.0

    def update(self, value: float, epoch: int) -> None:
        self.visits += 1
        self.value_sum += value
        if self.epoch != epoch:
            self.epoch = epoch
            self.epoch_visits = 0
            self.epoch_value_sum = 0.0
        self.epoch_visits += 1
        self.epoch_value_sum += value


@dataclass
class ReducedNode:
    state: ReducedState
    rank: int
    visits: int = 0
    value_sum: float = 0.0
    edges: dict[str, ReducedEdge] = field(default_factory=dict)
    pending_actions: list[ReducedAction] = field(default_factory=list)
    generation_round: int = 0


class ReducedSubgroupMCTS:
    def __init__(
        self,
        problem: ReducedSubgroupProblem,
        config: ReducedSearchConfig,
        *,
        handoff_root: Path | str,
        shared_discovery: SharedDiscoveryState | None = None,
        full_oracle: FacetOracle | None = None,
    ) -> None:
        self.problem = problem
        self.config = config
        self.geometry = ReducedGeometry(
            problem,
            handoff_root=handoff_root,
            full_oracle=full_oracle,
        )
        root_state = ReducedState(0)
        self.root = ReducedNode(root_state, -1)
        self.nodes = {root_state.key: self.root}
        self.rng = random.Random(config.seed)
        self.rollout_rngs = {
            policy: random.Random(
                config.seed
                ^ zlib.crc32(f"{problem.node_id}:{policy}".encode("ascii"))
            )
            for policy in set(ROLLOUT_CYCLE)
        }
        self.shared_discovery = shared_discovery or SharedDiscoveryState()
        self.discovered_classes: dict[str, dict[str, object]] = {}
        self.class_frequencies: Counter[str] = Counter()
        self.discovered_reduced_facets: set[int] = set()
        self.reduced_facet_frequencies: Counter[int] = Counter()
        self.vertex_frequencies: Counter[int] = Counter()
        self.action_frequencies: Counter[str] = Counter()
        self.retained_face_attempts: Counter[int] = Counter()
        self.retained_face_exits: dict[int, set[int]] = defaultdict(set)
        self.corrector_ridge_bank: dict[int, list[int]] = defaultdict(list)
        self.corrector_lower_face_bank: dict[int, list[int]] = defaultdict(list)
        self.persistent_retained_faces: set[int] = set()
        self.rollout_exit_filter = bytearray(config.rollout_exit_filter_bits // 8)
        self.root_basin_terminal_hits: Counter[int] = Counter()
        self.root_basin_discovery_hits: Counter[int] = Counter()
        self.discovery_events: list[dict[str, object]] = []
        self.reduced_discovery_events: list[dict[str, object]] = []
        self.stats: Counter[str] = Counter()
        self.current_iteration = 0
        self.current_global_iteration = 0
        self.started_at = 0.0
        self.termination_reason = "not_started"

    def _node(self, transition: ReducedTransition) -> ReducedNode:
        node = self.nodes.get(transition.state.key)
        if node is None:
            node = ReducedNode(transition.state, transition.rank)
            self.nodes[transition.state.key] = node
        return node

    def _child_limit(self, node: ReducedNode) -> int:
        value = self.config.widening_k0 + self.config.widening_alpha * (
            max(1, node.visits) ** self.config.widening_beta
        )
        return min(self.config.max_children, max(1, math.floor(value)))

    def _bundle_size(self, rank: int) -> int:
        ratio = (rank + 1) / max(1, self.problem.intrinsic_dimension)
        if ratio < 0.45:
            return self.config.early_bundle_size
        if ratio < 0.78:
            return self.config.middle_bundle_size
        return self.config.late_bundle_size

    def _independent_bundle(
        self,
        support_word: int,
        candidates: list[int],
        *,
        limit: int,
        rng: random.Random,
    ) -> int:
        rng.shuffle(candidates)
        current = support_word
        rank = self.geometry.rank_oracle.rank(current)
        added: list[int] = []
        for vertex in candidates:
            moved = current | (1 << vertex)
            moved_rank = self.geometry.rank_oracle.rank(moved)
            if moved_rank <= rank:
                continue
            added.append(vertex)
            current = moved
            rank = moved_rank
            if len(added) >= limit or rank >= self.problem.facet_rank:
                break
        return word_from_indices(added)

    def _filler_actions(
        self,
        node: ReducedNode,
        *,
        sample_count: int,
        rng: random.Random,
    ) -> list[ReducedAction]:
        facets = self.geometry.sample_supporting_facets(
            node.state.support_word,
            count=sample_count,
            rng=rng,
        )
        actions: list[ReducedAction] = []
        for facet_word in facets:
            if (
                node.state.source_facet_word
                and facet_word == node.state.source_facet_word
                and len(facets) > 1
            ):
                self.stats["source_facet_samples_skipped"] += 1
                continue
            outside = list(word_indices(facet_word & ~node.state.support_word))
            if not outside:
                continue
            gap = self.problem.facet_rank - node.rank
            bundle = self._independent_bundle(
                node.state.support_word,
                outside,
                limit=(
                    max(1, gap)
                    if self.config.complete_sampled_face
                    else min(max(1, gap), self._bundle_size(node.rank))
                ),
                rng=rng,
            )
            if not bundle:
                continue
            actions.append(
                ReducedAction(
                    "add",
                    add_word=bundle,
                    sampled_facet_word=facet_word,
                    source="supporting_face_lp",
                )
            )
        return list({action.action_id: action for action in actions}.values())

    def _affine_basis(self, facet_word: int, rng: random.Random) -> list[int]:
        candidates = list(word_indices(facet_word))
        rng.shuffle(candidates)
        basis: list[int] = []
        word = 0
        rank = -1
        for vertex in candidates:
            moved = word | (1 << vertex)
            moved_rank = self.geometry.rank_oracle.rank(moved)
            if moved_rank > rank:
                basis.append(vertex)
                word = moved
                rank = moved_rank
                if rank == self.problem.facet_rank:
                    break
        return basis

    def _corrector_actions(
        self,
        node: ReducedNode,
        *,
        count: int,
        rng: random.Random,
        persistent: bool,
    ) -> list[ReducedAction]:
        if node.state.corrections_used >= self.config.productive_max_corrections:
            return []
        facet_word = node.state.support_word
        ridge_bank = (
            self.corrector_ridge_bank[facet_word] if persistent else []
        )
        sampled = self.geometry.sample_ridges(
            facet_word,
            count=max(4, count),
            rng=rng,
        )
        for retained in sampled:
            if retained not in ridge_bank:
                ridge_bank.append(retained)
                if len(ridge_bank) >= self.config.corrector_face_bank_limit:
                    break

        lower_bank = (
            self.corrector_lower_face_bank[facet_word] if persistent else []
        )
        if len(ridge_bank) >= 2 and len(lower_bank) < self.config.corrector_face_bank_limit:
            pair_trials = (
                self.config.lower_face_pair_trials
                if persistent
                else min(self.config.lower_face_pair_trials, max(4, count * 2))
            )
            for _ in range(pair_trials):
                left, right = rng.sample(ridge_bank, 2)
                lower = left & right
                if (
                    lower
                    and lower not in lower_bank
                    and self.geometry.rank_oracle.rank(lower)
                    == self.problem.ridge_rank - 1
                ):
                    lower_bank.append(lower)
                    self.stats[
                        "corrector_lower_faces_banked"
                        if persistent
                        else "rollout_lower_faces_sampled"
                    ] += 1
                    if len(lower_bank) >= self.config.corrector_face_bank_limit:
                        break

        def frontier_key(retained: int) -> tuple[float, float]:
            attempts = self.retained_face_attempts[retained]
            exits = len(self.retained_face_exits.get(retained, ()))
            return (
                attempts / math.sqrt(1.0 + exits),
                rng.random(),
            )

        lower_quota = min(
            len(lower_bank),
            max(1, round(count * self.config.lower_face_fraction)),
        )
        ridge_quota = max(0, count - lower_quota)
        selected_ridges = sorted(ridge_bank, key=frontier_key)[:ridge_quota]
        selected_lower = sorted(lower_bank, key=frontier_key)[:lower_quota]
        actions = [
            ReducedAction(
                "remove",
                retained_word=retained,
                sampled_facet_word=facet_word,
                source="reduced_facet_ridge",
            )
            for retained in selected_ridges
        ]
        actions.extend(
            ReducedAction(
                "remove",
                retained_word=retained,
                sampled_facet_word=facet_word,
                source="reduced_facet_lower_face",
            )
            for retained in selected_lower
        )
        return actions

    def _generate_actions(
        self,
        node: ReducedNode,
        *,
        rollout: bool,
        policy: str,
        rng: random.Random,
    ) -> list[ReducedAction]:
        if node.state.role == "corrector":
            actions = self._corrector_actions(
                node,
                count=(4 if rollout else self.config.corrector_candidates),
                rng=rng,
                persistent=not rollout,
            )
        else:
            actions = self._filler_actions(
                node,
                sample_count=(
                    self.config.rollout_face_samples
                    if rollout
                    else self.config.tree_face_samples
                ),
                rng=rng,
            )
        if policy == "novel":
            actions.sort(
                key=lambda action: (
                    sum(self.vertex_frequencies[index] for index in word_indices(action.add_word)),
                    self.action_frequencies[action.action_id],
                    rng.random(),
                )
            )
        elif policy == "random":
            rng.shuffle(actions)
        else:
            actions.sort(
                key=lambda action: (
                    self.action_frequencies[action.action_id],
                    -action.add_word.bit_count(),
                    rng.random(),
                )
            )
        return actions

    def _transition(
        self,
        state: ReducedState,
        action: ReducedAction,
    ) -> ReducedTransition | None:
        if action.kind == "remove":
            rank = self.geometry.rank_oracle.rank(action.retained_word)
            return ReducedTransition(
                ReducedState(
                    action.retained_word,
                    "filler",
                    state.corrections_used + 1,
                    state.support_word,
                ),
                rank,
            )
        word = state.support_word | action.add_word
        rank = self.geometry.rank_oracle.rank(word)
        if rank > self.problem.facet_rank or rank <= self.geometry.rank_oracle.rank(
            state.support_word
        ):
            return None
        if rank == self.problem.facet_rank:
            terminal = self.geometry.validate(word)
            if not terminal.valid:
                self.stats[f"invalid_reduced_terminal_{terminal.reason}"] += 1
                return None
            return ReducedTransition(
                ReducedState(
                    terminal.closure_word,
                    "corrector",
                    state.corrections_used,
                    state.source_facet_word,
                ),
                self.problem.facet_rank,
                terminal,
            )
        return ReducedTransition(
            ReducedState(
                word,
                "filler",
                state.corrections_used,
                state.source_facet_word,
            ),
            rank,
        )

    def _edge(
        self,
        node: ReducedNode,
        action: ReducedAction,
    ) -> tuple[ReducedEdge, ReducedTransition] | None:
        transition = self._transition(node.state, action)
        if transition is None:
            return None
        gain = transition.rank - node.rank
        prior = 0.25 + 0.08 * max(0, gain)
        if action.kind == "remove" and action.source == "reduced_facet_lower_face":
            prior += self.config.lower_face_prior_boost
        if transition.terminal is not None:
            prior += 0.40 + 0.30 * int(transition.terminal.lifted)
        edge = ReducedEdge(
            action,
            transition.state.key,
            node.rank,
            transition.rank,
            transition.terminal,
            prior,
        )
        return edge, transition

    def _widen(self, node: ReducedNode) -> tuple[ReducedEdge, ReducedNode] | None:
        if len(node.edges) >= self._child_limit(node):
            return None
        if not node.pending_actions:
            node.generation_round += 1
            generated = self._generate_actions(
                node,
                rollout=False,
                policy="novel" if node.generation_round % 3 == 0 else "guided",
                rng=self.rng,
            )
            node.pending_actions.extend(
                action
                for action in generated
                if action.action_id not in node.edges
            )
        while node.pending_actions:
            action = node.pending_actions.pop(0)
            built = self._edge(node, action)
            if built is None:
                continue
            edge, transition = built
            if any(old.child_key == edge.child_key for old in node.edges.values()):
                continue
            child = self._node(transition)
            node.edges[action.action_id] = edge
            self.stats["tree_expansions"] += 1
            self.stats[f"tree_action_{action.kind}"] += 1
            return edge, child
        return None

    def _observe_rollout_exit(self, retained_word: int, reduced_key: int) -> bool:
        mask64 = (1 << 64) - 1
        mixed = (
            retained_word
            ^ (
                reduced_key
                + 0x9E3779B97F4A7C15
                + ((retained_word << 6) & mask64)
                + (retained_word >> 2)
            )
        ) & mask64
        first = mixed & (self.config.rollout_exit_filter_bits - 1)
        mixed ^= mixed >> 33
        mixed = (mixed * 0xFF51AFD7ED558CCD) & mask64
        mixed ^= mixed >> 33
        second = mixed & (self.config.rollout_exit_filter_bits - 1)
        first_byte, first_bit = divmod(first, 8)
        second_byte, second_bit = divmod(second, 8)
        first_mask = 1 << first_bit
        second_mask = 1 << second_bit
        seen = bool(
            self.rollout_exit_filter[first_byte] & first_mask
            and self.rollout_exit_filter[second_byte] & second_mask
        )
        self.rollout_exit_filter[first_byte] |= first_mask
        self.rollout_exit_filter[second_byte] |= second_mask
        return not seen

    def _select(self, node: ReducedNode) -> ReducedEdge | None:
        if not node.edges:
            return None
        log_parent = math.log(max(2, node.visits + 1))
        return max(
            node.edges.values(),
            key=lambda edge: (
                edge.mean(self.shared_discovery.epoch)
                + self.config.exploration
                * edge.prior
                * math.sqrt(log_parent / (edge.visits + 1)),
                -edge.visits,
                edge.action.action_id,
            ),
        )

    def _rollout_edge(
        self,
        node: ReducedNode,
        policy: str,
    ) -> tuple[ReducedEdge, ReducedNode] | None:
        rng = self.rollout_rngs[policy]
        actions = self._generate_actions(
            node,
            rollout=True,
            policy=policy,
            rng=rng,
        )
        for action in actions:
            built = self._edge(node, action)
            if built is None:
                continue
            edge, transition = built
            return edge, ReducedNode(transition.state, transition.rank)
        return None

    def _step_reward(
        self,
        node: ReducedNode,
        edge: ReducedEdge,
        path_actions: Sequence[ReducedAction],
        path_terminals: list[dict[str, object]],
        tree_edge: bool,
    ) -> tuple[float, bool]:
        action = edge.action
        reward = self.config.rank_gain_reward * max(
            0, edge.rank_after - edge.rank_before
        )
        for vertex in word_indices(action.add_word):
            self.vertex_frequencies[vertex] += 1
        self.action_frequencies[action.action_id] += 1
        if action.kind == "remove":
            removed_count = (
                node.state.support_word & ~action.retained_word
            ).bit_count()
            reward -= self.config.correction_cost * math.sqrt(removed_count)
            if tree_edge:
                self.retained_face_attempts[action.retained_word] += 1
                self.persistent_retained_faces.add(action.retained_word)
        terminal = edge.terminal
        if terminal is None:
            return reward, False
        reduced_key = terminal.closure_word
        previous_reduced_hits = self.reduced_facet_frequencies[reduced_key]
        self.reduced_facet_frequencies[reduced_key] += 1
        new_reduced = reduced_key not in self.discovered_reduced_facets
        if new_reduced:
            self.discovered_reduced_facets.add(reduced_key)
            reward += self.config.new_reduced_facet_reward
            self.reduced_discovery_events.append(
                {
                    "iteration": self.current_iteration,
                    "reduced_facet_word_hex": f"0x{reduced_key:x}",
                    "tight_irredundant_vertex_count": reduced_key.bit_count(),
                    "tight_reduced_point_count": len(terminal.tight_reduced_ids),
                    "lifted": terminal.lifted,
                    "full_reason": (
                        terminal.full_result.reason if terminal.full_result else None
                    ),
                }
            )
        else:
            reward += self.config.known_reduced_facet_reward / math.sqrt(
                1 + previous_reduced_hits
            )
        if (
            node.state.source_facet_word
            and reduced_key == node.state.source_facet_word
        ):
            reward -= self.config.source_return_penalty
            self.stats["corrector_source_returns"] += 1
        full = terminal.full_result
        retained_word = node.state.support_word if action.kind == "add" else 0
        retained_rank = node.rank if retained_word else None
        new_retained_exit = False
        if (
            retained_word
            and retained_rank is not None
            and retained_rank <= self.problem.ridge_rank
            and reduced_key != node.state.source_facet_word
        ):
            if retained_word in self.persistent_retained_faces:
                exits = self.retained_face_exits[retained_word]
                new_retained_exit = reduced_key not in exits
                exits.add(reduced_key)
            else:
                new_retained_exit = self._observe_rollout_exit(
                    retained_word,
                    reduced_key,
                )
            if new_retained_exit:
                if retained_rank < self.problem.ridge_rank:
                    reward += self.config.new_lower_face_exit_reward
                    self.stats["new_lower_face_exits"] += 1
                else:
                    reward += self.config.new_ridge_exit_reward
                    self.stats["new_ridge_exits"] += 1

        root_basin_word = next(
            (
                item.sampled_facet_word
                for item in path_actions
                if item.kind == "add" and item.sampled_facet_word
            ),
            0,
        )
        previous_basin_hits = self.root_basin_terminal_hits[root_basin_word]
        if root_basin_word:
            self.root_basin_terminal_hits[root_basin_word] += 1
            if previous_basin_hits == 0:
                reward += self.config.new_root_basin_reward
                self.stats["new_root_basins"] += 1

        terminal_step = {
            "action_depth": len(path_actions),
            "reduced_facet_word_hex": f"0x{reduced_key:x}",
            "original_support_word_hex": f"0x{terminal.original_support_word:064x}",
            "lifted": bool(full and full.valid),
            "class_id": full.class_id if full and full.valid else None,
            "catalog_index": full.catalog_index if full and full.valid else None,
            "new_reduced_facet": new_reduced,
            "retained_face_word_hex": (
                f"0x{retained_word:x}" if retained_word else None
            ),
            "retained_face_rank": retained_rank,
            "new_retained_face_exit": new_retained_exit,
            "new_global_class": False,
        }
        path_terminals.append(terminal_step)
        if full is None or not full.valid or full.class_id is None:
            reward += self.config.failed_lift_reward * int(new_reduced)
            return reward, False
        if new_reduced:
            reward += self.config.new_lifted_facet_reward
        class_id = full.class_id
        previous_local_hits = self.class_frequencies[class_id]
        self.class_frequencies[class_id] += 1
        new_local_class = class_id not in self.discovered_classes
        event: dict[str, object] | None = None
        if new_local_class:
            event = {
                "class_id": class_id,
                "catalog_index": full.catalog_index,
                "node_id": self.problem.node_id,
                "structure": self.problem.structure,
                "iteration": self.current_iteration,
                "global_iteration": self.current_global_iteration,
                "depth": len(path_actions),
                "reduced_facet_word_hex": f"0x{reduced_key:x}",
                "original_support_word_hex": f"0x{terminal.original_support_word:064x}",
                "root_basin_word_hex": f"0x{root_basin_word:x}",
                "terminal_steps": [dict(item) for item in path_terminals],
                "path": [item.to_dict() for item in path_actions],
            }
            self.discovered_classes[class_id] = event
            self.discovery_events.append(event)
            self.stats["new_local_classes"] += 1
        global_event = None
        if event is not None:
            global_event = {
                key: value
                for key, value in event.items()
                if key != "path"
            }
            global_event["path_action_ids"] = [
                item.action_id for item in path_actions
            ]
        new_class, previous_global_hits = self.shared_discovery.observe(
            class_id,
            global_event,
        )
        if new_class:
            reward += self.config.new_class_reward
            terminal_step["new_global_class"] = True
            if event is not None:
                event["terminal_steps"][-1]["new_global_class"] = True
            if global_event is not None:
                global_event["terminal_steps"][-1]["new_global_class"] = True
            if root_basin_word:
                self.root_basin_discovery_hits[root_basin_word] += 1
            self.stats["new_classes"] += 1
            self.stats["new_global_classes"] += 1
        else:
            reward += self.config.known_class_reward / math.sqrt(
                1 + previous_global_hits
            )
            if new_local_class:
                self.stats["globally_known_local_classes"] += 1
            elif previous_local_hits:
                self.stats["repeated_local_class_hits"] += 1
        return reward, new_class

    def _correction_limit(
        self,
        path_actions: Sequence[ReducedAction],
    ) -> int:
        root_basin_word = next(
            (
                item.sampled_facet_word
                for item in path_actions
                if item.kind == "add" and item.sampled_facet_word
            ),
            0,
        )
        if root_basin_word and self.root_basin_discovery_hits[root_basin_word]:
            return self.config.productive_max_corrections
        return self.config.max_corrections

    def run_iteration(
        self,
        iteration: int,
        *,
        global_iteration: int | None = None,
    ) -> None:
        self.current_iteration = iteration
        self.current_global_iteration = int(global_iteration or iteration)
        policy = ROLLOUT_CYCLE[(iteration - 1) % len(ROLLOUT_CYCLE)]
        self.stats[f"rollout_policy_{policy}"] += 1
        node = self.root
        path: list[tuple[ReducedNode, ReducedEdge, float, bool]] = []
        path_actions: list[ReducedAction] = []
        path_terminals: list[dict[str, object]] = []
        persistent = True
        leaf_value = 0.0
        for _depth in range(self.config.max_depth):
            if (
                node.state.role == "corrector"
                and node.state.corrections_used
                >= self._correction_limit(path_actions)
            ):
                break
            if persistent:
                widened = self._widen(node)
                if widened is not None:
                    edge, child = widened
                    expanded = True
                else:
                    edge = self._select(node)
                    if edge is None:
                        leaf_value = self.config.dead_end_reward
                        self.stats["dead_ends"] += 1
                        break
                    child = self.nodes[edge.child_key]
                    expanded = False
                self.stats["tree_policy_steps"] += 1
            else:
                rollout = self._rollout_edge(node, policy)
                if rollout is None:
                    leaf_value = self.config.dead_end_reward
                    self.stats["dead_ends"] += 1
                    break
                edge, child = rollout
                expanded = False
                self.stats["rollout_steps"] += 1
            path_actions.append(edge.action)
            reward, new_class = self._step_reward(
                node,
                edge,
                path_actions,
                path_terminals,
                persistent,
            )
            path.append((node, edge, reward, persistent))
            if new_class:
                edge.discovery_hits += 1
            node = child
            if persistent and expanded:
                persistent = False
        else:
            self.stats["depth_limit_hits"] += 1
        value = leaf_value
        for parent, edge, reward, tree_edge in reversed(path):
            value = reward + self.config.discount * value
            if tree_edge:
                edge.update(value, self.shared_discovery.epoch)
                parent.visits += 1
                parent.value_sum += value
        self.root.visits += int(not path or not path[0][3])
        self.stats["iterations"] += 1

    def run(self, *, target_ids: Sequence[str] = ()) -> dict[str, object]:
        self.started_at = time.perf_counter()
        self.termination_reason = "iteration_budget"
        target_set = set(target_ids)
        for iteration in range(1, self.config.iterations + 1):
            self.run_iteration(iteration)
            covered = target_set & self.discovered_classes.keys()
            if (
                self.config.progress_interval
                and iteration % self.config.progress_interval == 0
            ):
                print(
                    f"[422-reduced] node={self.problem.node_id} "
                    f"iteration={iteration} local_classes={len(self.discovered_classes)} "
                    f"global_classes={len(self.shared_discovery.discovered_classes)} "
                    f"target={len(covered)}/{len(target_set)} "
                    f"reduced_facets={len(self.discovered_reduced_facets)} "
                    f"nodes={len(self.nodes)} "
                    f"elapsed={time.perf_counter() - self.started_at:.1f}s",
                    flush=True,
                )
            if target_set and len(covered) == len(target_set):
                self.termination_reason = "target_coverage"
                break
            if (
                self.config.max_wall_seconds is not None
                and time.perf_counter() - self.started_at
                >= self.config.max_wall_seconds
            ):
                self.termination_reason = "wall_time_limit"
                break
        return self.result(target_ids=target_ids)

    def result(self, *, target_ids: Sequence[str] = ()) -> dict[str, object]:
        elapsed = time.perf_counter() - self.started_at if self.started_at else 0.0
        target_set = set(target_ids)
        covered = [item for item in target_ids if item in self.discovered_classes]
        return {
            "algorithm": "bell422_intrinsic_reduced_subgroup_mcts_v1",
            "problem": self.problem.to_dict(),
            "prior_boundary": {
                "search_reads": [
                    "complete_nodes.csv: selected node metadata",
                    f"nodes/{self.problem.node_id}/reduced_vertices.csv",
                    f"nodes/{self.problem.node_id}/orbit_blocks.csv",
                ],
                "terminal_only_reads": ["global_facet_classes.csv"],
                "evaluation_only_reads": [
                    f"nodes/{self.problem.node_id}/facets.csv: facet_class_id column only"
                ],
                "never_used_by_search": [
                    f"nodes/{self.problem.node_id}/facets.csv: h_row",
                    "known reduced facet supports",
                    "known target vertex selections",
                ],
                "target_labels_affect_search": False,
            },
            "features": {
                "intrinsic_reduced_search": True,
                "one_persistent_mcts_tree": True,
                "progressive_widening": True,
                "supporting_face_lp_filler": True,
                "coherent_sampled_face_completion": self.config.complete_sampled_face,
                "rank_adaptive_vertex_bundles": not self.config.complete_sampled_face,
                "ridge_and_lower_face_corrector": True,
                "persistent_corrector_face_banks": True,
                "retained_face_exit_novelty": True,
                "productive_basin_depth_extension": True,
                "root_basin_novelty_racing": True,
                "dynamic_global_new_class_reward": True,
                "cross_lane_shared_discovery": True,
                "shared_full_facet_oracle": self.geometry.shared_full_oracle,
                "discovery_epoch_ucb": True,
                "full_rank79_validation_only_after_reduced_facet": True,
            },
            "config": asdict(self.config),
            "runtime": {
                "elapsed_seconds": elapsed,
                "iterations_completed": self.current_iteration,
                "planned_iterations": self.config.iterations,
                "termination_reason": self.termination_reason,
                "truncated": self.termination_reason == "wall_time_limit",
            },
            "coverage": {
                "discovered_class_count": len(self.discovered_classes),
                "discovered_class_ids": list(self.discovered_classes),
                "discoveries": self.discovery_events,
                "class_frequencies": dict(self.class_frequencies),
                "target_class_count": len(target_ids),
                "covered_target_count": len(covered),
                "covered_target_ids": covered,
                "missing_target_ids": [
                    item for item in target_ids if item not in self.discovered_classes
                ],
                "non_target_discovered_ids": [
                    item for item in self.discovered_classes if item not in target_set
                ],
                "global_discovered_class_count": len(
                    self.shared_discovery.discovered_classes
                ),
                "global_epoch": self.shared_discovery.epoch,
            },
            "reduced_facets": {
                "unique_count": len(self.discovered_reduced_facets),
                "discovery_events": self.reduced_discovery_events,
            },
            "tree": {
                "node_count": len(self.nodes),
                "edge_count": sum(len(node.edges) for node in self.nodes.values()),
                "root_visits": self.root.visits,
                "root_mean_value": (
                    self.root.value_sum / self.root.visits if self.root.visits else 0.0
                ),
                "rollout_cycle": list(ROLLOUT_CYCLE),
                "root_basin_count": len(self.root_basin_terminal_hits),
                "productive_root_basin_count": len(self.root_basin_discovery_hits),
                "retained_face_count": len(self.retained_face_exits),
                "retained_face_exit_count": sum(
                    len(exits) for exits in self.retained_face_exits.values()
                ),
            },
            "search_stats": dict(self.stats),
            "geometry": self.geometry.to_dict(),
        }

    @staticmethod
    def write_result(result: dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        temporary.replace(path)


def run_node(
    *,
    handoff_root: Path,
    node_id: str,
    config: ReducedSearchConfig,
    output: Path,
) -> dict[str, object]:
    problem = ReducedSubgroupProblem.load(handoff_root, node_id)
    targets = load_target_class_ids(handoff_root, node_id)
    engine = ReducedSubgroupMCTS(
        problem,
        config,
        handoff_root=handoff_root,
    )
    result = engine.run(target_ids=targets)
    engine.write_result(result, output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bell 4-2-2 intrinsic reduced-subgroup Filler-Corrector MCTS"
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--node-id", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=4_222_025)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-wall-hours", type=float, default=2.9)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_wall_hours <= 0 or args.max_wall_hours > 3:
        raise SystemExit("--max-wall-hours must be in (0, 3]")
    node_ids = tuple(dict.fromkeys(args.node_id or DEFAULT_NODE_IDS))
    suite_started = time.perf_counter()
    suite_limit = 3600.0 * args.max_wall_hours
    summaries: list[dict[str, object]] = []
    for index, node_id in enumerate(node_ids):
        elapsed = time.perf_counter() - suite_started
        remaining = suite_limit - elapsed
        if remaining <= 1.0:
            break
        node_seed = args.seed ^ zlib.crc32(node_id.encode("ascii"))
        config = ReducedSearchConfig(
            iterations=args.iterations,
            seed=node_seed,
            progress_interval=args.progress_interval,
            max_wall_seconds=remaining,
        )
        output = args.output_dir / (
            f"{node_id.lower()}_i{args.iterations}_s{node_seed}_reduced_v1.json"
        )
        result = run_node(
            handoff_root=args.handoff_root,
            node_id=node_id,
            config=config,
            output=output,
        )
        summaries.append(
            {
                "node_id": node_id,
                "structure": result["problem"]["structure"],
                "output": str(output),
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "iterations_completed": result["runtime"]["iterations_completed"],
                "reduced_facet_count": result["reduced_facets"]["unique_count"],
                "discovered_class_count": result["coverage"][
                    "discovered_class_count"
                ],
                "target_class_count": result["coverage"]["target_class_count"],
                "covered_target_count": result["coverage"]["covered_target_count"],
                "missing_target_ids": result["coverage"]["missing_target_ids"],
            }
        )
    summary = {
        "algorithm": "bell422_intrinsic_reduced_subgroup_mcts_v1_suite",
        "total_elapsed_seconds": time.perf_counter() - suite_started,
        "wall_limit_seconds": suite_limit,
        "planned_node_ids": list(node_ids),
        "completed_node_count": len(summaries),
        "runs": summaries,
    }
    summary_output = args.summary_output or (
        args.output_dir / f"suite_i{args.iterations}_s{args.seed}_reduced_v1.json"
    )
    ReducedSubgroupMCTS.write_result(summary, summary_output)
    print(json.dumps({"summary_output": str(summary_output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
