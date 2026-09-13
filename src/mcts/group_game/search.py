from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import zlib
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from baseline.scorer import (
    ExpansionScorer,
    ScorerConfig,
    SharedScorerStructureCache,
    SharedTerminalValidationCache,
    exact_class_id,
)
from mcts.subgroup_patterns import (
    IDENTITY_PATTERN_ID,
    SubgroupPattern,
    SubgroupPatternAtlas,
    canonical_support_word,
    load_subgroup_pattern_atlas,
)

from .knowledge import GroupKnowledgeBase, PatternSpec
from .facet_corrector import FacetCorrectorBank
from .model import (
    EdgeStats,
    GameNode,
    MacroAction,
    SymmetryState,
    block_word,
    size_bucket,
    support_word_to_key,
)


DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
SIZE_BUCKETS = ("singleton", "pair", "small", "medium", "large")
FILLER_RELATION_LANES = (
    "continue",
    "neighbor",
    "continue",
    "switch",
    "neighbor",
    "switch",
)
FILLER_ACTION_LANES = (
    "fine",
    "single",
    "single",
    "union",
    "fine",
    "single",
    "union",
    "broad",
)
CORRECTOR_ACTION_LANES = (
    "remove_small",
    "rewrite",
    "remove_broad",
    "remove_small",
    "rewrite",
    "remove_broad",
)
ROLLOUT_POLICY_CYCLE = (
    "mixed",
    "mixed",
    "coherent",
    "mixed",
    "mixed",
    "fine",
    "mixed",
    "mixed",
)
REPAIR_SATURATION_CYCLE = (
    "context",
    "global",
)
CONTEXT_FRONTIER_LANE_CYCLE = (
    "endpoint",
    "endpoint",
    "endpoint",
    "bridge",
)


@dataclass(frozen=True)
class GroupGameConfig:
    """Configuration for the single-tree symmetry Filler-Corrector game."""

    iterations: int = 300
    seed: int = 322_2026
    known_classes: tuple[int, ...] = ()
    target_classes: tuple[int, ...] = tuple(range(1, 47))
    max_depth: int = 72
    max_corrections: int = 4
    extended_max_corrections: int = 8
    facet_replay_interval: int = 4
    frontier_replay_stride: int = 2
    endpoint_dedup: bool = True
    min_corrector_rank: int = 17
    exploration: float = 1.35
    discount: float = 0.985
    pattern_widening_k0: int = 3
    pattern_widening_alpha: float = 1.15
    pattern_widening_beta: float = 0.45
    block_widening_k0: int = 1
    block_widening_alpha: float = 1.25
    block_widening_beta: float = 0.50
    action_widening_k0: int = 4
    action_widening_alpha: float = 2.0
    action_widening_beta: float = 0.50
    max_children_per_node: int = 192
    max_corrector_children_per_node: int = 128
    discovery_epoch_child_reserve: int = 8
    discovery_epoch_child_recycle: int = 16
    stagnation_patience: int = 500
    recycle_visit_interval: int = 128
    recycle_fresh_patterns: int = 4
    rollout_policy_cycle: tuple[str, ...] = ROLLOUT_POLICY_CYCLE
    rank_checks_per_widen: int = 6
    filler_max_orbits_per_action: int = 2
    corrector_max_orbits_per_action: int = 4
    filler_union_candidates_per_pattern: int = 32
    corrector_union_candidates_per_pattern: int = 64
    corrector_rewrite_candidates_per_pattern: int = 48
    corrector_cross_rewrite_candidates_per_pattern: int = 24
    rollout_pattern_candidates: int = 8
    rollout_action_candidates: int = 12
    rollout_random_action_probability: float = 0.18
    rollout_corrector_probability: float = 0.88
    new_class_reward: float = 10.0
    known_class_reward: float = 0.10
    known_penalty_cap: float = 0.25
    support_penalty_cap: float = 0.25
    corrector_exact_utility: float = 0.25
    corrector_exact_frequency_penalty: float = 0.04
    unknown_facet_reward: float = 0.10
    invalid_terminal_reward: float = -1.5
    novel_terminal_support_bonus: float = 0.35
    repeated_terminal_penalty: float = 0.12
    known_class_frequency_penalty: float = 0.18
    rank_gain_reward: float = 0.055
    flat_add_penalty: float = 0.025
    correction_size_cost: float = 0.012
    correction_rank_loss_cost: float = 0.035
    rewrite_size_cost: float = 0.008
    corrector_new_basin_bonus: float = 0.02
    dead_end_reward: float = -0.75
    supportability_start_rank: int = 18
    supportability_direction_samples: int = 64
    supportability_prior_weight: float = 0.11
    include_level2_analysis: bool = True
    max_level2_pair_checks: int = 1024
    max_level2_matches: int = 64
    max_pattern_analyses_per_class: int = 2
    growth_children_per_root: int = 12
    snapshot_interval: int = 0
    cache_heavy_max_entries: int = 6000
    cache_light_max_entries: int = 30000
    max_corrector_diagnostics: int = 2000
    facet_repair_attempts: int = 48
    facet_repair_probability: float = 0.75
    saturated_facet_repair_probability: float = 0.12
    endpoint_saturation_gate: bool = True
    repair_saturation_cycle: tuple[str, ...] = REPAIR_SATURATION_CYCLE
    context_frontier_replay: bool = True
    context_frontier_stagnation_patience: int = 500
    context_frontier_quota_stride: int = 4
    context_frontier_lane_cycle: tuple[str, ...] = CONTEXT_FRONTIER_LANE_CYCLE
    context_frontier_scan_limit: int = 12
    context_frontier_grow_stride: int = 0
    context_geometry_growth: bool = True
    context_geometry_growth_stride: int = 1
    geometry_growth_warmup: int = 2
    geometry_growth_probe_stride: int = 4
    geometry_growth_max_service_lead: int = 8
    geometry_growth_target_lead_bonus: int = 16
    endpoint_context_novelty_reward: float = 0.20
    endpoint_pair_novelty_reward: float = 0.35
    endpoint_target_novelty_reward: float = 0.45
    ridge_max_candidates: int = 4096
    ridge_batch_size: int = 32
    lower_face_max_attempts: int = 6
    orbit_face_max_candidates: int = 256
    orbit_face_batch_size: int = 8
    fair_frontier: bool = False
    yield_frontier: bool = True
    frontier_high_water: int = 24

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.max_corrections < 0:
            raise ValueError("max_corrections must be nonnegative")
        if self.extended_max_corrections < self.max_corrections or self.facet_replay_interval < 0:
            raise ValueError("invalid correction extension or replay interval")
        if self.frontier_replay_stride < 0:
            raise ValueError("frontier replay stride must be nonnegative")
        if self.facet_repair_attempts < 0 or not 0 <= self.facet_repair_probability <= 1:
            raise ValueError("invalid facet repair budget/probability")
        if not 0 <= self.saturated_facet_repair_probability <= self.facet_repair_probability:
            raise ValueError("saturated facet repair probability must be within the base probability")
        if not self.repair_saturation_cycle or not set(self.repair_saturation_cycle) <= {
            "context", "global",
        }:
            raise ValueError("repair_saturation_cycle must contain context/global policies")
        if self.context_frontier_scan_limit <= 0:
            raise ValueError("context_frontier_scan_limit must be positive")
        if self.context_frontier_stagnation_patience < 0:
            raise ValueError("context_frontier_stagnation_patience must be nonnegative")
        if self.context_frontier_quota_stride <= 0:
            raise ValueError("context_frontier_quota_stride must be positive")
        if not self.context_frontier_lane_cycle or not set(
            self.context_frontier_lane_cycle
        ) <= {"endpoint", "bridge"}:
            raise ValueError("context_frontier_lane_cycle must contain endpoint/bridge lanes")
        if self.context_frontier_grow_stride < 0:
            raise ValueError("context_frontier_grow_stride must be nonnegative")
        if self.context_geometry_growth_stride <= 0:
            raise ValueError("context_geometry_growth_stride must be positive")
        if self.geometry_growth_warmup < 0:
            raise ValueError("geometry_growth_warmup must be nonnegative")
        if self.geometry_growth_probe_stride <= 0:
            raise ValueError("geometry_growth_probe_stride must be positive")
        if self.geometry_growth_max_service_lead < 0:
            raise ValueError("geometry_growth_max_service_lead must be nonnegative")
        if self.geometry_growth_target_lead_bonus < 0:
            raise ValueError("geometry_growth_target_lead_bonus must be nonnegative")
        if min(
            self.endpoint_context_novelty_reward,
            self.endpoint_pair_novelty_reward,
            self.endpoint_target_novelty_reward,
        ) < 0:
            raise ValueError("endpoint novelty rewards must be nonnegative")
        if self.ridge_max_candidates < 0 or self.ridge_batch_size <= 0:
            raise ValueError("invalid systematic ridge budget")
        if self.lower_face_max_attempts < 0:
            raise ValueError("lower face completion budget must be nonnegative")
        if self.orbit_face_max_candidates < 0 or self.orbit_face_batch_size <= 0:
            raise ValueError("invalid orbit face budget")
        if self.frontier_high_water <= 0:
            raise ValueError("frontier high water must be positive")
        if not 0 <= self.min_corrector_rank <= 25:
            raise ValueError("min_corrector_rank must be in [0, 25]")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if self.max_children_per_node <= 0:
            raise ValueError("max_children_per_node must be positive")
        if self.max_corrector_children_per_node <= 0:
            raise ValueError("max_corrector_children_per_node must be positive")
        if self.discovery_epoch_child_recycle < 0:
            raise ValueError("discovery_epoch_child_recycle must be nonnegative")
        if self.stagnation_patience < 0 or self.recycle_fresh_patterns < 0:
            raise ValueError("stagnation_patience and recycle_fresh_patterns must be nonnegative")
        if self.recycle_visit_interval <= 0:
            raise ValueError("recycle_visit_interval must be positive")
        if not self.rollout_policy_cycle or not set(self.rollout_policy_cycle) <= {
            "fine", "coherent", "mixed",
        }:
            raise ValueError("rollout_policy_cycle must contain fine/coherent/mixed policies")
        if not 0.0 <= self.rollout_random_action_probability <= 1.0:
            raise ValueError("rollout_random_action_probability must be in [0, 1]")
        if not 0.0 <= self.rollout_corrector_probability <= 1.0:
            raise ValueError("rollout_corrector_probability must be in [0, 1]")


@dataclass(frozen=True)
class OrbitUnionTemplate:
    pattern_id: str
    orbit_indices: tuple[int, ...]
    word: int
    source: str
    structure: str


@dataclass
class PathStep:
    node: GameNode
    edge: EdgeStats
    reward: float
    terminal_exact: bool = False
    terminal_new: bool = False
    corrector_origin_label: str | None = None
    corrector_origin_utility: float | None = None
    corrector_origin_support: int | None = None
    corrector_transition_cost: float = 0.0
    corrector_credit: float | None = None
    corrector_novelty_credit: float = 0.0
    corrector_geometric_credit: float = 0.0
    endpoint_novelty_credit: float = 0.0
    corrector_outcome_exact: bool = False
    corrector_outcome_new: bool = False
    is_tree_edge: bool = True


@dataclass
class GeometryGrowthEstimate:
    observations: int = 0
    reward_ema: float = 0.0
    seconds_ema: float = 0.0
    seconds: float = 0.0
    new_targets: int = 0
    new_pairs: int = 0
    new_plans: int = 0
    consecutive_empty: int = 0
    max_empty_streak: int = 0

    def observe(self, *, seconds: float, targets: int, pairs: int, plans: int) -> None:
        weight = 1.0 if not self.observations else 0.2
        reward = float(targets) + 0.05 * float(pairs)
        self.reward_ema += weight * (reward - self.reward_ema)
        self.seconds_ema += weight * (float(seconds) - self.seconds_ema)
        self.observations += 1
        self.seconds += float(seconds)
        self.new_targets += int(targets)
        self.new_pairs += int(pairs)
        self.new_plans += int(plans)
        if targets or pairs:
            self.consecutive_empty = 0
        else:
            self.consecutive_empty += 1
            self.max_empty_streak = max(
                self.max_empty_streak,
                self.consecutive_empty,
            )

    def score(self, services: int) -> float:
        cumulative_reward = (
            float(self.new_targets)
            + 0.05 * float(self.new_pairs)
        )
        yield_rate = cumulative_reward / max(1, self.observations)
        empty_decay = 1.0 / (1.0 + self.consecutive_empty / 8.0)
        uncertainty = 0.05 / math.sqrt(1 + int(services))
        return yield_rate * empty_decay + uncertainty


@dataclass(frozen=True)
class TerminalOutcome:
    reward: float
    counterfactual_utility: float
    label: str
    class_id: int | None
    exact_hit: bool
    new_class: bool
    valid: bool
    canonical_support_word: int


@dataclass
class DiscoveryEvent:
    event_index: int
    iteration: int
    elapsed_seconds: float
    class_id: int
    selected_support_word: int
    tight_support_word: int
    selected_support_size: int
    tight_support_size: int
    corrections_used: int
    rollout_policy: str
    path: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "iteration": self.iteration,
            "elapsed_seconds": self.elapsed_seconds,
            "class_id": self.class_id,
            "selected_support_word_hex": f"0x{self.selected_support_word:016x}",
            "tight_support_word_hex": f"0x{self.tight_support_word:016x}",
            "selected_support_size": self.selected_support_size,
            "tight_support_size": self.tight_support_size,
            "corrections_used": self.corrections_used,
            "rollout_policy": self.rollout_policy,
            "path": self.path,
        }


def _pattern_spec(pattern: SubgroupPattern, *, source: str = "atlas") -> PatternSpec:
    return PatternSpec(
        pattern_id=pattern.pattern_id,
        level=pattern.level,
        structure=pattern.structure,
        blocks=pattern.blocks,
        source=source,
        parent_root_ids=pattern.parent_root_ids,
        generator_labels=pattern.generator_labels,
    )


def _make_singleton_scorer(config: GroupGameConfig) -> ExpansionScorer:
    """Use singleton coordinates so every 64-bit support is scoreable."""
    scorer_config = ScorerConfig(
        terminal_scoring_mode="dynamic",
        dynamic_new_class_score=float(config.new_class_reward),
        dynamic_known_class_score=float(config.known_class_reward),
        supportability_start_rank=int(config.supportability_start_rank),
        supportability_direction_samples=int(config.supportability_direction_samples),
        supportability_direction_bank="global_cached",
        cache_heavy_max_entries=int(config.cache_heavy_max_entries),
        cache_light_max_entries=int(config.cache_light_max_entries),
    )
    return ExpansionScorer(
        blocks=tuple((index,) for index in range(64)),
        target_classes=set(int(value) for value in config.target_classes),
        rare_target_classes=set(),
        terminal_validation_cache=SharedTerminalValidationCache(),
        structure_cache=SharedScorerStructureCache(max_partitions=1),
        config=scorer_config,
    )


class SingleTreeGroupGame:
    """One MCTS tree whose actions are learned subgroup-orbit macros."""

    def __init__(
        self,
        config: GroupGameConfig,
        *,
        atlas: SubgroupPatternAtlas | None = None,
        scorer: ExpansionScorer | None = None,
    ) -> None:
        self.config = config
        self.atlas = load_subgroup_pattern_atlas() if atlas is None else atlas
        self.scorer = _make_singleton_scorer(config) if scorer is None else scorer
        self.facet_bank = FacetCorrectorBank(
            self.scorer.points, seed=config.seed, max_attempts=config.facet_repair_attempts,
            min_rank=config.min_corrector_rank,
            ridge_max_candidates=config.ridge_max_candidates, ridge_batch_size=config.ridge_batch_size,
            endpoint_dedup=config.endpoint_dedup,
            lower_face_max_attempts=config.lower_face_max_attempts,
            orbit_face_max_candidates=config.orbit_face_max_candidates,
            orbit_face_batch_size=config.orbit_face_batch_size,
            fair_frontier=config.fair_frontier, frontier_high_water=config.frontier_high_water,
            yield_frontier=config.yield_frontier,
        )
        self.endpoint_saturation_gate = bool(
            config.endpoint_saturation_gate and config.endpoint_dedup
        )
        self.novelty_credit_events: list[dict[str, object]] = []
        self.facet_witnesses: dict[int, tuple[MacroAction, ...]] = {}
        self.facet_witness_sources: dict[int, int] = {}
        self.facet_depth_witnesses: dict[
            tuple[int, int], tuple[MacroAction, ...]
        ] = {}
        self.facet_depth_witness_sources: dict[tuple[int, int], int] = {}
        self.frontier_replay_counts: Counter[int] = Counter()
        self.context_frontier_replay_counts: Counter[tuple[int, int]] = Counter()
        self.context_frontier_attempt_counts: Counter[tuple[int, int]] = Counter()
        self.context_frontier_proposals: Counter[
            tuple[int, int, str, int]
        ] = Counter()
        self.geometry_growth_estimates: dict[int, GeometryGrowthEstimate] = {}
        self.geometry_growth_services: Counter[int] = Counter()
        self.geometry_growth_stats: Counter[str] = Counter()
        self.geometry_growth_decisions: list[dict[str, object]] = []
        self.frontier_replay_start: int | None = None
        self.frontier_replay_keys: set[tuple] = set()
        self.facet_replay_counts: Counter[int] = Counter()
        self.iteration_facet_counts: Counter[int] = Counter()
        self.replay_stats: Counter[str] = Counter()
        self.context_endpoint_executions: Counter[
            tuple[int, int, str, int]
        ] = Counter()
        self.context_endpoint_stats: Counter[str] = Counter()
        self.endpoint_target_executions: Counter[int] = Counter()
        self.endpoint_novelty_events: list[dict[str, object]] = []
        self.rng = random.Random(int(config.seed))
        self.rollout_rngs = {
            policy: random.Random(
                int(config.seed)
                ^ zlib.crc32(f"rollout:{policy}".encode("ascii"))
            )
            for policy in ("fine", "coherent", "mixed")
        }

        initial_specs = (
            _pattern_spec(self.atlas.identity),
            *(_pattern_spec(pattern) for pattern in self.atlas.roots),
        )
        self.knowledge = GroupKnowledgeBase(initial_specs)
        for pattern in self.atlas.level2:
            self.knowledge.register_pattern(_pattern_spec(pattern), active=False)

        root_state = SymmetryState(
            support_word=0,
            role="filler",
            corrections_used=0,
            context_pattern_id=IDENTITY_PATTERN_ID,
        )
        self.root = GameNode(state=root_state, rank=self._rank(0))
        self.nodes: dict[tuple[int, str, int, str, int], GameNode] = {
            root_state.key: self.root,
        }
        self.template_cache: dict[tuple[str, str], tuple[OrbitUnionTemplate, ...]] = {}
        self.pattern_block_word_cache: dict[str, tuple[int, ...]] = {}
        self.coherence_cache: OrderedDict[tuple[int, str], float] = OrderedDict()
        self.candidate_action_cache: OrderedDict[
            tuple[str, int, str, str, str], tuple[MacroAction, ...]
        ] = OrderedDict()
        self.generation_cache_stats: Counter[str] = Counter()
        self.pattern_neighbors: dict[str, set[str]] = {
            pattern_id: set() for pattern_id in self.knowledge.patterns
        }
        for pattern in self.knowledge.patterns.values():
            for parent_id in pattern.parent_root_ids:
                self.pattern_neighbors.setdefault(pattern.pattern_id, set()).add(parent_id)
                self.pattern_neighbors.setdefault(parent_id, set()).add(pattern.pattern_id)
        for pattern in self.atlas.roots:
            self.pattern_neighbors.setdefault(IDENTITY_PATTERN_ID, set()).add(
                pattern.pattern_id
            )
            self.pattern_neighbors.setdefault(pattern.pattern_id, set()).add(
                IDENTITY_PATTERN_ID
            )
        self.discovered_classes = set(int(value) for value in config.known_classes)
        self.discovery_events: list[DiscoveryEvent] = []
        self.terminal_label_counts: Counter[str] = Counter()
        self.class_terminal_counts: Counter[int] = Counter()
        self.terminal_support_counts: Counter[int] = Counter()
        self.analyzed_supports: set[int] = set()
        self.class_analysis_counts: Counter[int] = Counter()
        self.growth_activations: list[dict[str, object]] = []
        self.started_at = 0.0
        self.current_iteration = 0
        self.simulations_completed = 0
        self.dead_ends = 0
        self.depth_limit_hits = 0
        self.corrector_outcome_counts: Counter[str] = Counter()
        self.corrector_operation_outcomes: Counter[str] = Counter()
        self.corrector_diagnostics: list[dict[str, object]] = []
        self.tree_expansions = 0
        self.rollout_steps = 0
        self.tree_policy_steps = 0
        self.action_kind_counts: Counter[str] = Counter()
        self.rollout_action_kind_counts: Counter[str] = Counter()
        self.relation_counts: Counter[str] = Counter()
        self.rollout_policy_counts: Counter[str] = Counter()
        self.current_rollout_policy = "mixed"
        self.current_repair_policy = config.repair_saturation_cycle[0]
        self.repair_policy_counts: Counter[str] = Counter()
        self.epoch_recycled_edges = 0
        self.stagnation_recycled_edges = 0
        self.recycle_events: list[dict[str, object]] = []

    def _rank(self, word: int) -> int:
        return int(self.scorer.affine_rank(support_word_to_key(int(word))))

    def _node(self, state: SymmetryState, rank: int | None = None) -> GameNode:
        node = self.nodes.get(state.key)
        if node is None:
            node = GameNode(
                state=state,
                rank=self._rank(state.support_word) if rank is None else int(rank),
            )
            self.nodes[state.key] = node
        return node

    @staticmethod
    def _pattern_bucket(pattern: PatternSpec) -> str:
        return size_bucket(pattern.dominant_size)

    def _pattern_limit(self, node: GameNode) -> int:
        config = self.config
        value = config.pattern_widening_k0 + config.pattern_widening_alpha * (
            max(1, node.visits) ** config.pattern_widening_beta
        )
        return max(1, int(math.floor(value)))

    def _block_limit(self, node: GameNode) -> int:
        config = self.config
        visits_per_pattern = max(1.0, node.visits / max(1, len(node.expanded_patterns)))
        value = config.block_widening_k0 + config.block_widening_alpha * (
            visits_per_pattern ** config.block_widening_beta
        )
        return max(1, int(math.floor(value)))

    def _child_limit(self, node: GameNode) -> int:
        config = self.config
        progressive = config.action_widening_k0 + config.action_widening_alpha * (
            max(1, node.visits) ** config.action_widening_beta
        )
        progressive += config.discovery_epoch_child_reserve * self.knowledge.epoch
        hard_cap = (
            config.max_corrector_children_per_node
            if node.state.role == "corrector"
            else config.max_children_per_node
        )
        return min(hard_cap, max(1, int(math.floor(progressive))))

    def _relation(self, current_pattern_id: str, target_pattern_id: str) -> str:
        if current_pattern_id == target_pattern_id:
            return "continue"
        if target_pattern_id in self.pattern_neighbors.get(current_pattern_id, set()):
            return "neighbor"
        return "switch"

    def _pattern_block_words(self, pattern_id: str) -> tuple[int, ...]:
        cached = self.pattern_block_word_cache.get(pattern_id)
        if cached is not None:
            return cached
        words = tuple(
            block_word(block)
            for block in self.knowledge.patterns[pattern_id].blocks
        )
        self.pattern_block_word_cache[pattern_id] = words
        return words

    def _coherence(self, support_word: int, pattern_id: str) -> float:
        key = int(support_word), pattern_id
        cached = self.coherence_cache.get(key)
        if cached is not None:
            self.generation_cache_stats["coherence_hits"] += 1
            self.coherence_cache.move_to_end(key)
            return cached
        self.generation_cache_stats["coherence_misses"] += 1
        support = int(support_word)
        if support == 0:
            value = 1.0
        else:
            defect = 0
            for word in self._pattern_block_words(pattern_id):
                selected = (support & word).bit_count()
                defect += min(selected, word.bit_count() - selected)
            value = max(0.0, 1.0 - defect / max(1, support.bit_count()))
        limit = self.config.cache_light_max_entries
        self.coherence_cache[key] = value
        if limit > 0 and len(self.coherence_cache) > limit:
            self.coherence_cache.popitem(last=False)
            self.generation_cache_stats["coherence_evictions"] += 1
        return value

    def _choose_new_pattern(self, node: GameNode) -> str | None:
        role = node.state.role
        available = sorted(
            self.knowledge.active_pattern_ids
            - node.expanded_patterns
            - node.exhausted_patterns
        )
        if not available:
            return None

        desired_relation = FILLER_RELATION_LANES[
            node.relation_cursor % len(FILLER_RELATION_LANES)
        ]
        node.relation_cursor += 1
        related = [
            pattern_id
            for pattern_id in available
            if self._relation(
                node.state.context_pattern_id,
                pattern_id,
            )
            == desired_relation
        ]
        relation_candidates = related or available

        bucket_candidates: list[str] = []
        for offset in range(len(SIZE_BUCKETS)):
            bucket = SIZE_BUCKETS[(node.bucket_cursor + offset) % len(SIZE_BUCKETS)]
            bucket_candidates = [
                pattern_id
                for pattern_id in relation_candidates
                if self._pattern_bucket(self.knowledge.patterns[pattern_id]) == bucket
            ]
            if bucket_candidates:
                node.bucket_cursor = (
                    node.bucket_cursor + offset + 1
                ) % len(SIZE_BUCKETS)
                break
        candidates = bucket_candidates or relation_candidates
        return max(
            candidates,
            key=lambda pattern_id: (
                self.knowledge.pattern_score(role, pattern_id)
                + 0.18
                * self.knowledge.transition_score(
                    role,
                    node.state.context_pattern_id,
                    pattern_id,
                )
                + 0.35 * self._coherence(node.state.support_word, pattern_id),
                -self.knowledge.patterns[pattern_id].level,
                pattern_id,
            ),
        )

    def _orbit_unions(
        self,
        pattern: PatternSpec,
        *,
        role: str,
    ) -> tuple[OrbitUnionTemplate, ...]:
        cache_key = (role, pattern.pattern_id)
        cached = self.template_cache.get(cache_key)
        if cached is not None:
            return cached

        orbit_words = tuple(block_word(block) for block in pattern.blocks)
        max_arity = (
            self.config.filler_max_orbits_per_action
            if role == "filler"
            else self.config.corrector_max_orbits_per_action
        )
        union_limit = (
            self.config.filler_union_candidates_per_pattern
            if role == "filler"
            else self.config.corrector_union_candidates_per_pattern
        )
        index_sets: list[tuple[int, ...]] = [(index,) for index in range(len(orbit_words))]
        seen = set(index_sets)
        seed = int(self.config.seed) ^ zlib.crc32(
            f"{role}:{pattern.pattern_id}".encode("ascii", errors="ignore")
        )
        local_rng = random.Random(seed)
        widths = tuple(range(2, min(max_arity, len(orbit_words)) + 1))
        per_width_limit = max(1, math.ceil(union_limit / max(1, len(widths))))
        added_unions = 0
        for width in widths:
            width_added = 0
            deterministic = (
                tuple((start + offset) % len(orbit_words) for offset in range(width))
                for start in range(len(orbit_words))
            )
            for indices in deterministic:
                normalized = tuple(sorted(indices))
                if normalized in seen:
                    continue
                seen.add(normalized)
                index_sets.append(normalized)
                added_unions += 1
                width_added += 1
                if width_added >= per_width_limit or added_unions >= union_limit:
                    break
            attempts = 0
            while (
                width_added < per_width_limit
                and added_unions < union_limit
                and attempts < per_width_limit * 8
            ):
                attempts += 1
                normalized = tuple(sorted(local_rng.sample(range(len(orbit_words)), width)))
                if normalized in seen:
                    continue
                seen.add(normalized)
                index_sets.append(normalized)
                added_unions += 1
                width_added += 1

        templates = tuple(
            OrbitUnionTemplate(
                pattern_id=pattern.pattern_id,
                orbit_indices=indices,
                word=sum(orbit_words[index] for index in indices),
                source=pattern.source,
                structure=pattern.structure,
            )
            for indices in index_sets
        )
        self.template_cache[cache_key] = templates
        return templates

    def candidate_actions_for_pattern(
        self,
        node: GameNode,
        pattern_id: str,
        *,
        lane: str | None = None,
        fallback: bool = True,
    ) -> list[MacroAction]:
        """Expose only the action family requested by progressive widening."""
        role = node.state.role
        if role not in ("filler", "corrector"):
            return []
        lane_key = "all" if lane is None else lane
        cache_key = (
            role,
            node.state.support_word,
            node.state.context_pattern_id,
            pattern_id,
            lane_key,
        )
        cached = self.candidate_action_cache.get(cache_key)
        if cached is not None:
            self.generation_cache_stats["candidate_hits"] += 1
            self.candidate_action_cache.move_to_end(cache_key)
            candidates = list(cached)
            if lane is not None and fallback and not candidates:
                self.generation_cache_stats["lane_fallbacks"] += 1
                return self.candidate_actions_for_pattern(node, pattern_id)
            return candidates
        self.generation_cache_stats["candidate_misses"] += 1
        pattern = self.knowledge.patterns[pattern_id]
        support = node.state.support_word
        candidates: list[MacroAction] = []
        seen_effects: set[int] = set()
        build_orbit_actions = role == "filler" or lane != "rewrite"
        templates = self._orbit_unions(pattern, role=role) if build_orbit_actions else ()
        for template in templates:
            effective = (
                template.word & ~support
                if role == "filler"
                else template.word & support
            )
            effective &= (1 << 64) - 1
            if effective == 0 or effective in seen_effects:
                continue
            if role == "corrector" and effective == support:
                continue
            seen_effects.add(effective)
            arity = len(template.orbit_indices)
            size = effective.bit_count()
            if lane == "fine" and not (role == "filler" and arity == 1 and size <= 2):
                continue
            if lane == "single" and not (role == "filler" and arity == 1):
                continue
            if lane == "union" and not (role == "filler" and arity > 1):
                continue
            if lane == "broad" and not (role == "filler" and size >= 5):
                continue
            if lane == "remove_small" and not (role == "corrector" and size <= 2):
                continue
            if lane == "remove_broad" and not (role == "corrector" and size >= 3):
                continue
            candidates.append(
                MacroAction(
                    kind="add" if role == "filler" else "remove",
                    pattern_id=template.pattern_id,
                    orbit_indices=template.orbit_indices,
                    block_word=template.word,
                    effective_word=effective,
                    source=template.source,
                    structure=template.structure,
                    block_size=size,
                    arity=arity,
                    remove_word=effective if role == "corrector" else 0,
                    add_word=effective if role == "filler" else 0,
                    remove_orbit_indices=(
                        template.orbit_indices if role == "corrector" else ()
                    ),
                    add_orbit_indices=(
                        template.orbit_indices if role == "filler" else ()
                    ),
                    remove_pattern_id=template.pattern_id if role == "corrector" else "",
                    add_pattern_id=template.pattern_id if role == "filler" else "",
                )
            )
        if role == "corrector" and lane in (None, "rewrite"):
            candidates.extend(self._rewrite_actions(node, pattern))
            context_id = node.state.context_pattern_id
            if self._relation(context_id, pattern_id) == "neighbor":
                candidates.extend(self._paired_rewrite_actions(
                    node,
                    remove_pattern=self.knowledge.patterns[context_id],
                    add_pattern=pattern,
                    limit=self.config.corrector_cross_rewrite_candidates_per_pattern,
                ))
        limit = self.config.cache_heavy_max_entries
        self.candidate_action_cache[cache_key] = tuple(candidates)
        if limit > 0 and len(self.candidate_action_cache) > limit:
            self.candidate_action_cache.popitem(last=False)
            self.generation_cache_stats["candidate_evictions"] += 1
        if lane is not None and fallback and not candidates:
            self.generation_cache_stats["lane_fallbacks"] += 1
            return self.candidate_actions_for_pattern(node, pattern_id)
        return candidates

    def _rewrite_actions(
        self,
        node: GameNode,
        pattern: PatternSpec,
    ) -> list[MacroAction]:
        return self._paired_rewrite_actions(
            node,
            remove_pattern=pattern,
            add_pattern=pattern,
            limit=self.config.corrector_rewrite_candidates_per_pattern,
        )

    def _paired_rewrite_actions(
        self,
        node: GameNode,
        *,
        remove_pattern: PatternSpec,
        add_pattern: PatternSpec,
        limit: int,
    ) -> list[MacroAction]:
        if limit <= 0:
            return []
        support = node.state.support_word
        removable: list[tuple[OrbitUnionTemplate, int]] = []
        addable: list[tuple[OrbitUnionTemplate, int]] = []
        for template in self._orbit_unions(remove_pattern, role="corrector"):
            remove_word = template.word & support
            if remove_word and remove_word != support:
                removable.append((template, remove_word))
        for template in self._orbit_unions(add_pattern, role="corrector"):
            add_word = template.word & ~support & ((1 << 64) - 1)
            if add_word:
                addable.append((template, add_word))
        removable.sort(
            key=lambda item: (item[1].bit_count(), len(item[0].orbit_indices), item[0].orbit_indices)
        )
        addable.sort(
            key=lambda item: (item[1].bit_count(), len(item[0].orbit_indices), item[0].orbit_indices)
        )
        if not removable or not addable:
            return []

        candidates: list[MacroAction] = []
        seen: set[tuple[int, int]] = set()
        ranked_additions_by_size: dict[int, list[tuple[OrbitUnionTemplate, int]]] = {}
        # Interleave removal sizes so the bounded pool reaches broad rewrites.
        by_size: dict[int, list[tuple[OrbitUnionTemplate, int]]] = {}
        for item in removable:
            by_size.setdefault(item[1].bit_count(), []).append(item)
        removable = [
            group[index]
            for index in range(max(map(len, by_size.values())))
            for group in by_size.values()
            if index < len(group)
        ]
        for remove_template, remove_word in removable:
            remove_size = remove_word.bit_count()
            ranked_additions = ranked_additions_by_size.get(remove_size)
            if ranked_additions is None:
                ranked_additions = sorted(
                    addable,
                    key=lambda item: (
                        abs(item[1].bit_count() - remove_size),
                        len(item[0].orbit_indices),
                        item[0].orbit_indices,
                    ),
                )
                ranked_additions_by_size[remove_size] = ranked_additions
            for add_template, add_word in ranked_additions[:4]:
                signature = (remove_word, add_word)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(
                    MacroAction(
                        kind="rewrite",
                        pattern_id=add_pattern.pattern_id,
                        orbit_indices=(
                            *remove_template.orbit_indices,
                            *add_template.orbit_indices,
                        ),
                        block_word=remove_template.word | add_template.word,
                        effective_word=remove_word | add_word,
                        source=add_pattern.source,
                        structure=(
                            add_pattern.structure
                            if remove_pattern.pattern_id == add_pattern.pattern_id
                            else f"{remove_pattern.structure}->{add_pattern.structure}"
                        ),
                        block_size=remove_word.bit_count() + add_word.bit_count(),
                        arity=(
                            len(remove_template.orbit_indices)
                            + len(add_template.orbit_indices)
                        ),
                        remove_word=remove_word,
                        add_word=add_word,
                        remove_orbit_indices=remove_template.orbit_indices,
                        add_orbit_indices=add_template.orbit_indices,
                        remove_pattern_id=remove_pattern.pattern_id,
                        add_pattern_id=add_pattern.pattern_id,
                    )
                )
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def _approximate_action_score(self, node: GameNode, action: MacroAction) -> float:
        pattern_score = self.knowledge.pattern_score(node.state.role, action.pattern_id)
        return (pattern_score + self._action_shape_score(node, action)
                + {"continue": 0.30, "neighbor": 0.12, "switch": -0.04}[
                    self._relation(node.state.context_pattern_id, action.pattern_id)]
                + 0.45 * self._coherence(node.state.support_word, action.pattern_id))

    @staticmethod
    def _action_shape_score(node: GameNode, action: MacroAction) -> float:
        if node.state.role == "filler":
            desired = max(1, min(8, 25 - node.rank + 1))
            size_term = -0.08 * abs(action.block_size - desired)
            union_term = 0.04 * (action.arity - 1)
        else:
            desired = 1 + min(5, node.state.corrections_used + 1)
            size_term = -0.035 * abs(action.block_size - desired)
            union_term = 0.025 * min(action.arity - 1, 3)
        return size_term + union_term

    def _materialize_action(
        self,
        node: GameNode,
        action: MacroAction,
    ) -> tuple[SymmetryState, int] | None:
        state = node.state
        if action.kind == "add":
            new_word = state.support_word | action.effective_word
            new_rank = self._rank(new_word)
            if new_rank > 25:
                return None
            if new_rank == 25 and state.repair_source_word and not (new_word & ~state.repair_source_word):
                self.facet_bank.stats["source_facet_closure_blocked"] += 1
                return None
            new_role = "corrector" if new_rank == 25 else "filler"
            return (
                SymmetryState(
                    support_word=new_word,
                    role=new_role,
                    corrections_used=state.corrections_used,
                    context_pattern_id=action.pattern_id,
                    repair_source_word=state.repair_source_word if new_rank < 25 else 0,
                ),
                new_rank,
            )
        if action.kind == "remove":
            new_word = state.support_word & ~action.effective_word
            new_rank = self._rank(new_word)
            if new_rank < self.config.min_corrector_rank:
                return None
            corrections = state.corrections_used + 1
            new_role = "corrector" if new_rank == 25 else "filler"
            return (
                SymmetryState(
                    support_word=new_word,
                    role=new_role,
                    corrections_used=corrections,
                    context_pattern_id=action.pattern_id,
                ),
                new_rank,
            )
        if action.kind == "rewrite":
            new_word = (state.support_word & ~action.remove_word) | action.add_word
            if new_word == state.support_word:
                return None
            new_rank = self._rank(new_word)
            if new_rank < self.config.min_corrector_rank or new_rank > 25:
                return None
            corrections = state.corrections_used + 1
            new_role = "corrector" if new_rank == 25 else "filler"
            return (
                SymmetryState(
                    support_word=new_word,
                    role=new_role,
                    corrections_used=corrections,
                    context_pattern_id=action.pattern_id,
                    repair_source_word=action.source_facet_word if new_rank < 25 else 0,
                ),
                new_rank,
            )
        if action.kind == "stop":
            return (
                SymmetryState(
                    support_word=state.support_word,
                    role="done",
                    corrections_used=state.corrections_used,
                    context_pattern_id=state.context_pattern_id,
                    repair_source_word=state.repair_source_word,
                ),
                node.rank,
            )
        raise ValueError(f"unsupported action kind {action.kind!r}")

    def _geometry_prior(self, *, word: int, rank: int) -> float:
        if rank < self.config.supportability_start_rank or rank >= 25:
            return 0.0
        metrics_fn = getattr(self.scorer, "supportability_metrics", None)
        if metrics_fn is None:
            return 0.0
        metrics = metrics_fn(support_word_to_key(word))
        obstruction = float(metrics.closer_side) + math.log1p(
            max(0.0, float(metrics.supporting_shift))
        )
        return -self.config.supportability_prior_weight * min(12.0, obstruction)

    def _add_stop_edge(self, node: GameNode) -> EdgeStats | None:
        if "stop" in node.edges:
            return None
        action = MacroAction(
            kind="stop",
            pattern_id="__stop__",
            orbit_indices=(),
            block_word=0,
            effective_word=0,
            source="control",
            structure="stop",
            block_size=0,
            arity=0,
        )
        materialized = self._materialize_action(node, action)
        if materialized is None:
            return None
        child_state, child_rank = materialized
        child = self._node(child_state, child_rank)
        prior = 2.0 if self._correction_limit_reached(node) else 0.25
        edge = EdgeStats(
            action=action,
            child_key=child.state.key,
            prior=prior,
            rank_before=node.rank,
            rank_after=child_rank,
        )
        node.edges["stop"] = edge
        node.child_state_keys.add(child.state.key)
        return edge

    def _next_action_lane(self, node: GameNode) -> str:
        lanes = (
            FILLER_ACTION_LANES
            if node.state.role == "filler"
            else CORRECTOR_ACTION_LANES
        )
        lane = lanes[node.action_lane_cursor % len(lanes)]
        node.action_lane_cursor += 1
        return lane

    @staticmethod
    def _matches_action_lane(action: MacroAction, lane: str) -> bool:
        if lane == "fine":
            return action.kind == "add" and action.arity == 1 and action.block_size <= 2
        if lane == "single":
            return action.kind == "add" and action.arity == 1
        if lane == "union":
            return action.kind == "add" and action.arity > 1
        if lane == "broad":
            return action.kind == "add" and action.block_size >= 5
        if lane == "rewrite":
            return action.kind == "rewrite"
        if lane == "remove_small":
            return action.kind == "remove" and action.block_size <= 2
        if lane == "remove_broad":
            return action.kind == "remove" and action.block_size >= 3
        return True

    def _refresh_node_epoch(self, node: GameNode) -> None:
        epoch = self.knowledge.epoch
        previous_epoch = node.knowledge_epoch_seen
        node.knowledge_epoch_seen = epoch
        last_discovery = self.discovery_events[-1].iteration if self.discovery_events else 0
        stagnant = (
            self.config.stagnation_patience > 0
            and self.current_iteration - last_discovery >= self.config.stagnation_patience
            and node.visits - node.last_recycle_visit >= self.config.recycle_visit_interval
        )
        epoch_changed = previous_epoch >= 0 and previous_epoch != epoch
        if not (epoch_changed or stagnant):
            return
        if self.config.discovery_epoch_child_recycle <= 0 or len(node.edges) < self._child_limit(node):
            return
        unseen_patterns = self.knowledge.active_pattern_ids - node.expanded_patterns
        if not unseen_patterns:
            return
        candidates = sorted(
            (
                edge for edge in node.edges.values()
                if edge.action.kind != "stop" and edge.visits >= 8
                and edge.mean_value <= 0.0
                and edge.discovery_hits == 0
            ),
            key=lambda edge: (edge.mean_value, -edge.visits, edge.action.action_id),
        )
        retired = 0
        for edge in candidates:
            pattern_id = edge.action.pattern_id
            if node.pattern_action_counts.get(pattern_id, 0) <= 1:
                continue
            del node.edges[edge.action.action_id]
            node.retired_action_ids.add(edge.action.action_id)
            node.pattern_action_counts[pattern_id] -= 1
            retired += 1
            if retired >= self.config.discovery_epoch_child_recycle:
                break
        if retired:
            node.child_state_keys = {edge.child_key for edge in node.edges.values()}
            node.child_support_words = {key[0] for key in node.child_state_keys}
            node.last_recycle_visit = node.visits
            fresh_patterns = []
            for _ in range(min(retired, self.config.recycle_fresh_patterns)):
                pattern_id = self._choose_new_pattern(node)
                if pattern_id is None:
                    break
                node.expanded_patterns.add(pattern_id)
                fresh_patterns.append(pattern_id)
            reason = "discovery_epoch" if epoch_changed else "stagnation"
            if epoch_changed:
                self.epoch_recycled_edges += retired
            else:
                self.stagnation_recycled_edges += retired
            self.recycle_events.append({
                "iteration": self.current_iteration,
                "elapsed_seconds": time.perf_counter() - self.started_at,
                "reason": reason,
                "support_word_hex": f"0x{node.state.support_word:016x}",
                "role": node.state.role,
                "context_pattern_id": node.state.context_pattern_id,
                "visits": node.visits,
                "iterations_since_discovery": self.current_iteration - last_discovery,
                "retired_edges": retired,
                "fresh_pattern_ids": fresh_patterns,
            })

    def _correction_limit_reached(self, node: GameNode) -> bool:
        used = node.state.corrections_used
        anchor = self.facet_bank.anchors.get(node.facet_word)
        canonical = self.facet_bank.shared.coordinates[node.facet_word][0] if anchor else None
        repeats = self.iteration_facet_counts[canonical] if canonical is not None else 0
        context_open = self._has_open_context_exit(node)
        if repeats >= 3 and not context_open:
            return True
        if used < self.config.max_corrections:
            return False
        if self.config.max_corrections == 0 or used >= self.config.extended_max_corrections:
            return True
        return (
            anchor is None
            or (repeats > 1 and not context_open)
            or not (self.facet_bank.has_frontier(node.facet_word) or context_open)
        )

    def _facet_actions(self, node: GameNode, *, grow: bool = True) -> list[MacroAction]:
        state = node.state
        if state.repair_source_word:
            words = self.facet_bank.exits.get((
                state.repair_source_word, state.support_word, state.context_pattern_id,
            ), ())
            pattern = self.knowledge.patterns[state.context_pattern_id]
            return [MacroAction(
                kind="add", pattern_id=pattern.pattern_id,
                orbit_indices=tuple(i for i, b in enumerate(pattern.blocks) if block_word(b) & word),
                block_word=word, effective_word=word, source="facet_external_exit",
                structure=pattern.structure, block_size=word.bit_count(),
                arity=sum(bool(block_word(b) & word) for b in pattern.blocks),
                source_facet_word=state.repair_source_word,
            ) for word in words]
        anchor = self.facet_bank.anchors.get(node.facet_word)
        if state.role != "corrector" or anchor is None:
            return []
        if grow:
            self.facet_bank.grow(node.facet_word)
        for pattern in anchor.patterns:
            self.knowledge.register_pattern(pattern, active=False)
        actions = []
        for plan in anchor.plans:
            removed = state.support_word & ~plan.retained_word
            added = plan.retained_word & ~state.support_word
            if not removed:
                continue
            pattern = self.knowledge.patterns[plan.pattern_id]
            actions.append(MacroAction(
                kind="rewrite", pattern_id=plan.pattern_id, orbit_indices=(),
                block_word=plan.retained_word, effective_word=removed | added,
                source="facet_repartition", structure=pattern.structure,
                block_size=(removed | added).bit_count(),
                arity=sum(bool(block_word(b) & (anchor.word & ~plan.retained_word)) for b in pattern.blocks),
                remove_word=removed, add_word=added,
                remove_pattern_id=f"FACET-H-{anchor.word:016x}",
                add_pattern_id=plan.pattern_id, source_facet_word=anchor.word,
            ))
        return actions

    def _facet_edge(self, node: GameNode, *, rng: random.Random, persistent: bool):
        if not node.state.repair_source_word and (
            node.state.role != "corrector" or node.facet_word not in self.facet_bank.anchors
        ):
            return None
        if not node.state.repair_source_word and rng.random() >= self.config.facet_repair_probability:
            return None
        actions = self._facet_actions(node)
        if self.endpoint_saturation_gate and not node.state.repair_source_word and actions:
            mode = self.current_repair_policy
            self.replay_stats[f"{mode}_repair_policy_offers"] += 1
            source_saturated = self._facet_source_saturated(node.facet_word)
            if mode == "context":
                context_open = [
                    action for action in actions
                    if self._facet_action_context_uses(node, action) == 0
                ]
                if context_open:
                    if source_saturated:
                        self.replay_stats["global_saturated_context_open_offers"] += 1
                    self.replay_stats["context_open_facet_repair_offers"] += 1
                    actions = context_open
                else:
                    self.replay_stats["saturated_facet_repair_offers"] += 1
                    keep = (self.config.saturated_facet_repair_probability
                            / max(self.config.facet_repair_probability, 1e-12))
                    if rng.random() >= keep:
                        self.replay_stats["saturated_facet_repair_skips"] += 1
                        return None
            elif source_saturated:
                self.replay_stats["global_policy_saturated_offers"] += 1
                keep = (self.config.saturated_facet_repair_probability
                        / max(self.config.facet_repair_probability, 1e-12))
                if rng.random() >= keep:
                    self.replay_stats["global_policy_saturated_skips"] += 1
                    return None
            else:
                self.replay_stats["global_policy_open_offers"] += 1
        elif actions and not node.state.repair_source_word:
            self.replay_stats["open_facet_repair_offers"] += 1
        if persistent:
            actions = [a for a in actions if a.action_id not in node.edges
                       and a.action_id not in node.retired_action_ids]
        if not actions:
            return None
        # Geometry chooses admissibility; online action values choose among
        # admissible transformations, with a random share for unseen ridges.
        rng.shuffle(actions)
        if rng.random() >= self.config.rollout_random_action_probability:
            def priority(action):
                context_priority = (
                    self.endpoint_saturation_gate
                    and self.current_repair_policy == "context"
                )
                if node.state.repair_source_word:
                    uses = self.facet_bank.shared.uses(node.state.repair_source_word,
                                                      node.state.support_word, action.effective_word)
                    context_uses = (
                        self._external_action_context_uses(node, action)
                        if context_priority
                        else 0
                    )
                else:
                    retained = (node.state.support_word & ~action.remove_word) | action.add_word
                    exits = self.facet_bank.exits.get(
                        (node.facet_word, retained, action.pattern_id),
                        (),
                    )
                    uses = min(
                        (
                            self.facet_bank.shared.uses(node.facet_word, retained, word)
                            for word in exits
                        ),
                        default=0,
                    )
                    context_uses = (
                        self._facet_action_context_uses(node, action)
                        if context_priority
                        else 0
                    )
                pattern_score = self.knowledge.pattern_score(
                    node.state.role,
                    action.pattern_id,
                )
                if context_priority:
                    return -context_uses, -uses, pattern_score
                return 0, -uses, pattern_score
            actions.sort(key=priority, reverse=True)
        for action in actions:
            candidate = self._ephemeral_edge(node, action)
            if candidate is not None and (not persistent or candidate[1].key not in node.child_state_keys):
                return candidate
        return None

    def _endpoint_context_key(
        self,
        raw_source: int,
        retained: int,
        outside: int,
        pattern_id: str,
        corrections_used: int,
    ) -> tuple[int, int, str, int] | None:
        key = self.facet_bank.shared.exit_key(raw_source, retained, outside)
        target = self.facet_bank.shared.endpoints.targets.get(key)
        if target is None:
            return None
        return key[0], target, pattern_id, int(corrections_used)

    def _facet_action_context_uses(self, node: GameNode, action: MacroAction) -> int:
        retained = (node.state.support_word & ~action.remove_word) | action.add_word
        exits = self.facet_bank.exits.get(
            (node.facet_word, retained, action.pattern_id),
            (),
        )
        uses = []
        for outside in exits:
            key = self._endpoint_context_key(
                node.facet_word,
                retained,
                outside,
                action.pattern_id,
                node.state.corrections_used + 1,
            )
            if key is not None:
                uses.append(self.context_endpoint_executions[key])
        return min(uses) if uses else 0

    def _external_action_context_uses(
        self,
        node: GameNode,
        action: MacroAction,
    ) -> int:
        key = self._endpoint_context_key(
            node.state.repair_source_word,
            node.state.support_word,
            action.effective_word,
            node.state.context_pattern_id,
            node.state.corrections_used,
        )
        return 0 if key is None else self.context_endpoint_executions[key]

    def _has_open_context_exit(self, node: GameNode) -> bool:
        if (
            not self.endpoint_saturation_gate
            or not self._context_frontier_active()
            or node.state.role != "corrector"
            or node.facet_word not in self.facet_bank.anchors
            or not hasattr(self.facet_bank.anchors[node.facet_word], "plans")
        ):
            return False
        for action in self._facet_actions(node, grow=False):
            if (
                self._facet_action_context_uses(node, action) == 0
                and self._materialize_action(node, action) is not None
            ):
                return True
        return False

    def _context_frontier_active(self) -> bool:
        if not self.config.context_frontier_replay:
            return False
        last_discovery = self.discovery_events[-1].iteration if self.discovery_events else 0
        return (
            self.current_iteration - last_discovery
            >= self.config.context_frontier_stagnation_patience
        )

    def _record_endpoint_context(self, node: GameNode, action: MacroAction) -> float:
        key = self._endpoint_context_key(
            node.state.repair_source_word,
            node.state.support_word,
            action.effective_word,
            node.state.context_pattern_id,
            node.state.corrections_used,
        )
        if key is None:
            self.context_endpoint_stats["unmapped_executions"] += 1
            return 0.0
        source, target, _pattern_id, _corrections = key
        graph = self.facet_bank.shared.endpoints
        pair = source, target
        first_context = not self.context_endpoint_executions[key]
        first_pair = not graph.executions[pair]
        first_target = not self.endpoint_target_executions[target]
        if not first_context:
            self.context_endpoint_stats["repeat_context_executions"] += 1
        self.context_endpoint_executions[key] += 1
        self.endpoint_target_executions[target] += 1
        self.context_endpoint_stats["mapped_executions"] += 1
        if target == source:
            self.context_endpoint_stats["self_endpoint_no_novelty_reward"] += 1
            return 0.0
        components = {
            "context": self.config.endpoint_context_novelty_reward if first_context else 0.0,
            "pair": self.config.endpoint_pair_novelty_reward if first_pair else 0.0,
            "target": self.config.endpoint_target_novelty_reward if first_target else 0.0,
        }
        reward = float(sum(components.values()))
        for name, value in components.items():
            if value:
                self.context_endpoint_stats[f"novel_{name}_executions"] += 1
        if reward and len(self.endpoint_novelty_events) < self.config.max_corrector_diagnostics:
            self.endpoint_novelty_events.append({
                "iteration": self.current_iteration,
                "source_hex": f"0x{source:016x}",
                "target_hex": f"0x{target:016x}",
                "repair_pattern_id": key[2],
                "corrections_used": key[3],
                "reward": reward,
                "components": components,
            })
        return reward

    def _facet_source_saturated(self, raw_source: int) -> bool:
        coordinates = self.facet_bank.shared.coordinates.get(raw_source)
        if coordinates is None:
            return False
        canonical = coordinates[0]
        graph = self.facet_bank.shared.endpoints
        return (not graph.pending[canonical]
                and any(source == canonical for source, _target in graph.pairs))

    def _widen_once(self, node: GameNode) -> EdgeStats | None:
        if node.state.role == "done":
            return None
        new_stop: EdgeStats | None = None
        if node.state.role == "corrector":
            new_stop = self._add_stop_edge(node)
            if new_stop is not None:
                self.tree_expansions += 1
                return new_stop
            if self._correction_limit_reached(node):
                return None
        if self.config.discovery_epoch_child_recycle > 0:
            self._refresh_node_epoch(node)
        if len(node.edges) >= self._child_limit(node):
            return None

        repaired = self._facet_edge(node, rng=self.rng, persistent=True)
        if repaired is not None:
            edge, state, rank = repaired
            self._node(state, rank)
            node.edges[edge.action.action_id] = edge
            node.child_state_keys.add(state.key)
            node.child_support_words.add(state.support_word)
            pattern_id = edge.action.pattern_id
            node.pattern_action_counts[pattern_id] = node.pattern_action_counts.get(pattern_id, 0) + 1
            self.tree_expansions += 1
            self.action_kind_counts[edge.action.kind] += 1
            return edge
        if node.state.repair_source_word:
            return None

        desired_patterns = min(
            self._pattern_limit(node),
            len(self.knowledge.active_pattern_ids),
        )
        if len(node.expanded_patterns) < desired_patterns:
            pattern_id = self._choose_new_pattern(node)
            if pattern_id is not None:
                node.expanded_patterns.add(pattern_id)

        desired_lane = self._next_action_lane(node)
        attempts = max(1, len(node.expanded_patterns) + 2)
        for _ in range(attempts):
            block_limit = self._block_limit(node)
            eligible = [
                pattern_id
                for pattern_id in node.expanded_patterns
                if pattern_id not in node.exhausted_patterns
                and node.pattern_action_counts.get(pattern_id, 0) < block_limit
            ]
            if not eligible:
                if len(node.expanded_patterns) < desired_patterns:
                    pattern_id = self._choose_new_pattern(node)
                    if pattern_id is not None:
                        node.expanded_patterns.add(pattern_id)
                        continue
                return None
            pattern_id = max(
                eligible,
                key=lambda value: (
                    -node.pattern_action_counts.get(value, 0),
                    self.knowledge.pattern_score(node.state.role, value)
                    + 0.25
                    * self._coherence(node.state.support_word, value)
                    + 0.12
                    * self.knowledge.transition_score(
                        node.state.role,
                        node.state.context_pattern_id,
                        value,
                    ),
                    value,
                ),
            )
            candidates = [
                action
                for action in self.candidate_actions_for_pattern(
                    node,
                    pattern_id,
                    lane=desired_lane,
                    fallback=False,
                )
                if action.action_id not in node.edges
                and action.action_id not in node.rejected_action_ids
                and action.action_id not in node.retired_action_ids
            ]
            if not candidates:
                self.generation_cache_stats["lane_fallbacks"] += 1
                candidates = [
                    action
                    for action in self.candidate_actions_for_pattern(node, pattern_id)
                    if action.action_id not in node.edges
                    and action.action_id not in node.rejected_action_ids
                    and action.action_id not in node.retired_action_ids
                ]
            if not candidates:
                node.exhausted_patterns.add(pattern_id)
                continue
            candidates.sort(
                key=lambda action: (
                    self._action_shape_score(node, action),
                    -action.block_size,
                    action.action_id,
                ),
                reverse=True,
            )

            best: tuple[float, MacroAction, SymmetryState, int] | None = None
            checks = 0
            for action in candidates:
                materialized = self._materialize_action(node, action)
                checks += 1
                if materialized is None:
                    node.rejected_action_ids.add(action.action_id)
                else:
                    child_state, child_rank = materialized
                    if child_state.key in node.child_state_keys:
                        node.rejected_action_ids.add(action.action_id)
                    else:
                        relation = self._relation(
                            node.state.context_pattern_id,
                            action.pattern_id,
                        )
                        prior = self.knowledge.action_prior(
                            node.state.role,
                            action,
                            rank_before=node.rank,
                            rank_after=child_rank,
                            context_pattern_id=node.state.context_pattern_id,
                            coherence=self._coherence(
                                child_state.support_word,
                                action.pattern_id,
                            ),
                            relation=relation,
                        )
                        prior = max(
                            0.05,
                            prior
                            + self._geometry_prior(
                                word=child_state.support_word,
                                rank=child_rank,
                            ),
                        )
                        score = prior + 0.02 * self.rng.random()
                        candidate = (score, action, child_state, child_rank)
                        if best is None or candidate[0] > best[0]:
                            best = candidate
                if checks >= self.config.rank_checks_per_widen:
                    break

            if best is None:
                if checks >= len(candidates):
                    node.exhausted_patterns.add(pattern_id)
                continue
            prior, action, child_state, child_rank = best
            child = self._node(child_state, child_rank)
            node.edges[action.action_id] = EdgeStats(
                action=action,
                child_key=child.state.key,
                prior=prior,
                rank_before=node.rank,
                rank_after=child_rank,
            )
            node.child_support_words.add(child_state.support_word)
            node.child_state_keys.add(child_state.key)
            node.pattern_action_counts[pattern_id] = (
                node.pattern_action_counts.get(pattern_id, 0) + 1
            )
            self.tree_expansions += 1
            self.action_kind_counts[action.kind] += 1
            self.relation_counts[
                self._relation(node.state.context_pattern_id, action.pattern_id)
            ] += 1
            return node.edges[action.action_id]
        return None

    def _select_edge(self, node: GameNode) -> EdgeStats:
        if node.state.role == "corrector" and self._correction_limit_reached(node):
            return node.edges["stop"]
        epoch = self.knowledge.epoch
        sqrt_parent = math.sqrt(node.visits + 1.0)

        def score(edge: EdgeStats) -> tuple[float, str]:
            epoch_mean = edge.epoch_mean(epoch)
            q_value = 0.0 if epoch_mean is None else epoch_mean
            epoch_visits = edge.epoch_visits if edge.epoch == epoch else 0
            exploration = (
                self.config.exploration
                * edge.prior
                * sqrt_parent
                / (1.0 + epoch_visits)
            )
            return q_value + exploration + 1e-9 * self.rng.random(), edge.action.action_id

        return max(node.edges.values(), key=score)

    def _transition_reward(self, edge: EdgeStats) -> float:
        if edge.action.kind == "add":
            gain = edge.rank_after - edge.rank_before
            return (
                self.config.rank_gain_reward * gain
                - (self.config.flat_add_penalty if gain <= 0 else 0.0)
            )
        if edge.action.kind == "remove":
            rank_loss = max(0, edge.rank_before - edge.rank_after)
            return -(
                self.config.correction_size_cost * edge.action.block_size
                + self.config.correction_rank_loss_cost * rank_loss
            )
        if edge.action.kind == "rewrite":
            rank_loss = max(0, edge.rank_before - edge.rank_after)
            changed_size = (
                edge.action.remove_word.bit_count()
                + edge.action.add_word.bit_count()
            )
            return -(
                self.config.rewrite_size_cost * changed_size
                + self.config.correction_rank_loss_cost * rank_loss
            )
        return 0.0

    def _tight_support_word(self, selected_word: int, label: object) -> int:
        validation = getattr(label, "validation", None)
        normal = getattr(validation, "normal", None)
        offset = getattr(validation, "offset", None)
        if normal is None or offset is None:
            return int(selected_word)
        signed = self.scorer.points @ np.asarray(normal) + float(offset)
        tight = np.flatnonzero(np.abs(signed) <= self.scorer.config.support_tol)
        word = 0
        for vertex in tight:
            word |= 1 << int(vertex)
        return word

    def _activate_growth_children(self, root_pattern_ids: Sequence[str]) -> None:
        limit = max(0, int(self.config.growth_children_per_root))
        for root_id in root_pattern_ids:
            children = [pattern.pattern_id for pattern, _edge in self.atlas.children(root_id)]
            inactive = [
                pattern_id
                for pattern_id in children
                if pattern_id not in self.knowledge.active_pattern_ids
            ]
            chosen = sorted(
                inactive,
                key=lambda pattern_id: (
                    self.knowledge.pattern_score("filler", pattern_id),
                    pattern_id,
                ),
                reverse=True,
            )[:limit]
            if not chosen:
                continue
            self.knowledge.activate(chosen)
            self.growth_activations.append(
                {
                    "iteration": self.current_iteration,
                    "root_pattern_id": root_id,
                    "activated_children": chosen,
                }
            )

    def _learn_terminal_pattern(
        self,
        *,
        tight_word: int,
        class_id: int | None,
    ) -> None:
        canonical_word = canonical_support_word(int(tight_word))
        if canonical_word in self.analyzed_supports:
            record = self.knowledge.observed_supports.get(canonical_word)
            if record is not None:
                record.hit_count += 1
                if class_id is not None:
                    record.class_ids.add(int(class_id))
            return
        if class_id is not None and (
            self.class_analysis_counts[class_id]
            >= self.config.max_pattern_analyses_per_class
        ):
            return

        analysis = self.atlas.analyze_support(
            tight_word,
            include_level2=self.config.include_level2_analysis,
            max_level2_pair_checks=self.config.max_level2_pair_checks,
            max_level2_matches=self.config.max_level2_matches,
        )
        observed_id = f"OBS-{canonical_word:016x}"
        observed = PatternSpec(
            pattern_id=observed_id,
            level=3,
            structure=f"Stab({analysis.stabilizer_order})",
            blocks=analysis.stabilizer_blocks,
            source="observed_stabilizer",
            parent_root_ids=analysis.root_pattern_ids,
            generator_labels=(),
        )
        self.knowledge.record_support(
            canonical_word=canonical_word,
            class_id=class_id,
            stabilizer_pattern=observed,
            stabilizer_order=analysis.stabilizer_order,
            root_pattern_ids=analysis.root_pattern_ids,
            root_subgroup_multiplicities=analysis.root_subgroup_multiplicities,
            level2_pattern_ids=analysis.level2_pattern_ids,
        )
        self.pattern_neighbors.setdefault(observed_id, set()).update(
            analysis.root_pattern_ids
        )
        for root_id in analysis.root_pattern_ids:
            self.pattern_neighbors.setdefault(root_id, set()).add(observed_id)
        self._activate_growth_children(analysis.root_pattern_ids)
        self.analyzed_supports.add(canonical_word)
        if class_id is not None:
            self.class_analysis_counts[class_id] += 1

    def _observe_terminal(
        self,
        node: GameNode,
        *,
        path: Sequence[PathStep],
    ) -> TerminalOutcome:
        key = support_word_to_key(node.state.support_word)
        label = self.scorer.terminal_label(key)
        label_name = str(label.label)
        self.terminal_label_counts[label_name] += 1
        class_id = exact_class_id(label_name) if bool(label.is_exact) else None
        previous_class_hits = (
            0 if class_id is None else self.class_terminal_counts[class_id]
        )
        if class_id is not None:
            self.class_terminal_counts[class_id] += 1
        tight_word = self._tight_support_word(node.state.support_word, label)
        valid_terminal = bool(label.validation.valid)
        if valid_terminal and getattr(label.validation, "normal", None) is not None:
            node.facet_word = tight_word
            self.facet_bank.observe(tight_word, label.validation.normal, label.validation.offset)
        canonical_tight = (
            canonical_support_word(tight_word)
            if valid_terminal
            else node.state.support_word
        )
        if node.facet_word:
            self.iteration_facet_counts[canonical_tight] += 1
            witness = tuple(step.edge.action for step in path)
            old = self.facet_witnesses.get(canonical_tight)
            def cost(actions):
                return sum(a.kind in ("remove", "rewrite") for a in actions), len(actions)
            if witness and (old is None or cost(witness) < cost(old)):
                self.facet_witnesses[canonical_tight] = witness
                self.facet_witness_sources[canonical_tight] = node.facet_word
                self.replay_stats["witness_updates"] += 1
            depth_key = canonical_tight, int(node.state.corrections_used)
            old_depth = self.facet_depth_witnesses.get(depth_key)
            if witness and (old_depth is None or len(witness) < len(old_depth)):
                self.facet_depth_witnesses[depth_key] = witness
                self.facet_depth_witness_sources[depth_key] = node.facet_word
                self.replay_stats["depth_witness_updates"] += 1
        previous_support_hits = self.terminal_support_counts[canonical_tight]
        self.terminal_support_counts[canonical_tight] += 1

        exact_hit = class_id is not None
        new_class = exact_hit and class_id not in self.discovered_classes
        if new_class:
            self.discovered_classes.add(int(class_id))
            self._assign_corrector_novelty(path, int(class_id))
            self.knowledge.begin_discovery_epoch()
            self.discovery_events.append(
                DiscoveryEvent(
                    event_index=len(self.discovery_events) + 1,
                    iteration=self.current_iteration,
                    elapsed_seconds=time.perf_counter() - self.started_at,
                    class_id=int(class_id),
                    selected_support_word=node.state.support_word,
                    tight_support_word=tight_word,
                    selected_support_size=node.state.support_size,
                    tight_support_size=tight_word.bit_count(),
                    corrections_used=node.state.corrections_used,
                    rollout_policy=self.current_rollout_policy,
                    path=[step.edge.action.to_dict() for step in path],
                )
            )

        if bool(label.is_exact):
            base_utility = (
                self.config.new_class_reward
                if new_class
                else self.config.known_class_reward
                - min(self.config.known_penalty_cap, self.config.known_class_frequency_penalty
                      * math.log1p(previous_class_hits))
            )
            counterfactual_utility = (
                self.config.corrector_exact_utility
                - min(0.20, self.config.corrector_exact_frequency_penalty
                      * math.log1p(previous_class_hits + 1))
            )
        elif valid_terminal:
            base_utility = self.config.unknown_facet_reward
            counterfactual_utility = self.config.unknown_facet_reward
        else:
            base_utility = self.config.invalid_terminal_reward
            counterfactual_utility = self.config.invalid_terminal_reward
        reward = float(base_utility)
        if previous_support_hits == 0:
            if valid_terminal:
                reward += self.config.novel_terminal_support_bonus
                self._learn_terminal_pattern(tight_word=tight_word, class_id=class_id)
        elif valid_terminal:
            reward -= min(self.config.support_penalty_cap,
                          self.config.repeated_terminal_penalty * math.log1p(previous_support_hits))
        return TerminalOutcome(
            reward=float(reward),
            counterfactual_utility=float(counterfactual_utility),
            label=label_name,
            class_id=class_id,
            exact_hit=exact_hit,
            new_class=bool(new_class),
            valid=valid_terminal,
            canonical_support_word=canonical_tight,
        )

    def _assign_corrector_novelty(self, path: Sequence[PathStep], class_id: int) -> None:
        contributors = [(i, step) for i, step in enumerate(path)
                        if step.corrector_origin_utility is not None]
        if not contributors:
            return
        weights = [self.config.discount ** (len(path) - 1 - i) for i, _ in contributors]
        total = sum(weights)
        shares = []
        for (index, step), weight in zip(contributors, weights):
            share = self.config.new_class_reward * weight / total
            step.corrector_novelty_credit += share
            shares.append({"path_index": index, "credit": share, "source": step.edge.action.source})
        self.novelty_credit_events.append({
            "iteration": self.current_iteration, "class_id": class_id,
            "total_credit": sum(item["credit"] for item in shares), "contributors": shares,
        })

    def _resolve_corrector_credit(
        self,
        step: PathStep,
        *,
        outcome: TerminalOutcome | None,
        fallback_value: float | None = None,
    ) -> None:
        baseline = step.corrector_origin_utility
        if baseline is None:
            return
        if outcome is None:
            fallback = (
                self.config.dead_end_reward
                if fallback_value is None
                else float(fallback_value)
            )
            result_utility = min(float(baseline), fallback)
            outcome_label = "no_terminal"
            downstream_new_class = False
        else:
            result_utility = outcome.counterfactual_utility
            outcome_label = outcome.label
            downstream_new_class = outcome.new_class
        basin_bonus = 0.0
        if (
            outcome is not None
            and step.corrector_origin_support is not None
            and outcome.canonical_support_word != step.corrector_origin_support
        ):
            basin_bonus = self.config.corrector_new_basin_bonus
        credit = (
            result_utility
            - baseline
            + basin_bonus
            + step.corrector_transition_cost
        )
        step.corrector_credit = float(credit)
        step.corrector_outcome_exact = outcome is not None and outcome.exact_hit
        step.corrector_outcome_new = downstream_new_class

        if downstream_new_class:
            category = "new_class"
        elif credit > 0.20:
            category = "improved"
        elif credit < -0.20:
            category = "worse"
        else:
            category = "neutral"
        self.corrector_outcome_counts[category] += 1
        operation = step.edge.action.kind
        if operation == "rewrite":
            operation = (
                "cross_rewrite"
                if step.edge.action.remove_pattern_id != step.edge.action.add_pattern_id
                else "same_rewrite"
            )
        self.corrector_operation_outcomes[f"{operation}:{category}"] += 1
        if len(self.corrector_diagnostics) < self.config.max_corrector_diagnostics:
            self.corrector_diagnostics.append(
                {
                    "iteration": self.current_iteration,
                    "action": step.edge.action.to_dict(),
                    "origin_label": step.corrector_origin_label,
                    "outcome_label": outcome_label,
                    "origin_counterfactual_utility": baseline,
                    "outcome_utility": result_utility,
                    "basin_bonus": basin_bonus,
                    "transition_cost": step.corrector_transition_cost,
                    "counterfactual_credit": credit,
                    "geometric_novelty_credit": step.corrector_geometric_credit,
                    "category": category,
                }
            )

    def _leaf_value(self, node: GameNode) -> float:
        if node.state.role == "done":
            return 0.0
        size_excess = max(0, node.state.support_size - 30)
        return 0.018 * node.rank - 0.006 * size_excess

    @staticmethod
    def _stop_action() -> MacroAction:
        return MacroAction(
            kind="stop",
            pattern_id="__stop__",
            orbit_indices=(),
            block_word=0,
            effective_word=0,
            source="control",
            structure="stop",
            block_size=0,
            arity=0,
        )

    def _rollout_lane(self, role: str, policy: str = "mixed") -> str:
        draw = self.rollout_rngs[policy].random()
        if role == "filler":
            if policy == "fine":
                return "fine"
            if policy == "coherent":
                return "single"
            if draw < 0.50:
                return "single"
            if draw < 0.85:
                return "union"
            return "broad"
        if draw < 0.35:
            return "remove_small"
        if draw < 0.65:
            return "remove_broad"
        return "rewrite"

    def _rollout_pattern_ids(self, node: GameNode, policy: str = "mixed") -> list[str]:
        rng = self.rollout_rngs[policy]
        active = sorted(self.knowledge.active_pattern_ids)
        if policy == "fine" and node.state.role == "filler":
            active = [
                pattern_id for pattern_id in active
                if min(self.knowledge.patterns[pattern_id].block_sizes) <= 2
            ]
        if not active:
            return []
        context = node.state.context_pattern_id
        candidates: set[str] = set()
        if context in active:
            candidates.add(context)
        neighbors = sorted(
            self.pattern_neighbors.get(context, set())
            & set(active)
        )
        if neighbors:
            candidates.update(rng.sample(neighbors, min(3, len(neighbors))))
        top = sorted(
            active,
            key=lambda pattern_id: (
                self.knowledge.pattern_score(node.state.role, pattern_id),
                pattern_id,
            ),
            reverse=True,
        )[:3]
        candidates.update(top)
        remaining = [pattern_id for pattern_id in active if pattern_id not in candidates]
        if remaining:
            candidates.update(rng.sample(remaining, min(3, len(remaining))))
        if policy == "fine":
            candidates.add(IDENTITY_PATTERN_ID)
        ranked = sorted(
            candidates,
            key=lambda pattern_id: (
                int(policy == "coherent" and pattern_id == context)
                + int(policy == "fine" and pattern_id == IDENTITY_PATTERN_ID),
                self.knowledge.pattern_score(node.state.role, pattern_id)
                + 0.30 * self._coherence(node.state.support_word, pattern_id)
                + {
                    "continue": 0.30,
                    "neighbor": 0.12,
                    "switch": 0.0,
                }[
                    self._relation(context, pattern_id)
                ],
                pattern_id,
            ),
            reverse=True,
        )
        return ranked[: self.config.rollout_pattern_candidates]

    def _ephemeral_edge(
        self,
        node: GameNode,
        action: MacroAction,
    ) -> tuple[EdgeStats, SymmetryState, int] | None:
        materialized = self._materialize_action(node, action)
        if materialized is None:
            return None
        child_state, child_rank = materialized
        relation = self._relation(
            node.state.context_pattern_id,
            action.pattern_id,
        )
        prior = self.knowledge.action_prior(
            node.state.role,
            action,
            rank_before=node.rank,
            rank_after=child_rank,
            context_pattern_id=node.state.context_pattern_id,
            coherence=(
                0.0
                if action.kind == "stop"
                else self._coherence(child_state.support_word, action.pattern_id)
            ),
            relation=relation,
        )
        prior = max(
            0.05,
            prior
            + self._geometry_prior(
                word=child_state.support_word,
                rank=child_rank,
            ),
        )
        return (
            EdgeStats(
                action=action,
                child_key=child_state.key,
                prior=prior,
                rank_before=node.rank,
                rank_after=child_rank,
            ),
            child_state,
            child_rank,
        )

    def _rollout_edge(
        self,
        node: GameNode,
        policy: str = "mixed",
    ) -> tuple[EdgeStats, SymmetryState, int] | None:
        rng = self.rollout_rngs[policy]
        if node.state.role == "corrector":
            if (
                self._correction_limit_reached(node)
                or (node.state.corrections_used < self.config.max_corrections
                    and rng.random() > self.config.rollout_corrector_probability)
            ):
                return self._ephemeral_edge(node, self._stop_action())

        repaired = self._facet_edge(node, rng=rng, persistent=False)
        if repaired is not None:
            return repaired
        if node.state.repair_source_word:
            return None

        lane = self._rollout_lane(node.state.role, policy)
        rough_candidates: list[MacroAction] = []
        for pattern_id in self._rollout_pattern_ids(node, policy):
            selected = self.candidate_actions_for_pattern(
                node,
                pattern_id,
                lane=lane,
                fallback=lane != "fine",
            )
            selected.sort(
                key=lambda action: (
                    self._action_shape_score(node, action),
                    -action.block_size,
                    action.action_id,
                ),
                reverse=True,
            )
            rough_candidates.extend(selected[:1])
            if len(selected) > 1:
                rough_candidates.append(rng.choice(selected[1:]))
        if not rough_candidates:
            if node.state.role == "corrector":
                return self._ephemeral_edge(node, self._stop_action())
            return None

        if rng.random() < self.config.rollout_random_action_probability:
            rng.shuffle(rough_candidates)
        else:
            rough_candidates.sort(
                key=lambda action: (
                    int(policy == "coherent" and action.pattern_id == node.state.context_pattern_id),
                    self._approximate_action_score(node, action),
                ),
                reverse=True,
            )
        materialized: list[tuple[float, EdgeStats, SymmetryState, int]] = []
        for action in rough_candidates[: self.config.rollout_action_candidates]:
            candidate = self._ephemeral_edge(node, action)
            if candidate is None:
                continue
            edge, child_state, child_rank = candidate
            materialized.append(
                (
                    edge.prior + 0.08 * rng.random(),
                    edge,
                    child_state,
                    child_rank,
                )
            )
        if not materialized:
            if node.state.role == "corrector":
                return self._ephemeral_edge(node, self._stop_action())
            return None
        if policy == "coherent" and node.state.role == "filler":
            continued = [
                item for item in materialized
                if item[1].action.pattern_id == node.state.context_pattern_id
            ]
            if continued and rng.random() < 0.85:
                materialized = continued
        if rng.random() < self.config.rollout_random_action_probability:
            _score, edge, child_state, child_rank = rng.choice(materialized)
        else:
            _score, edge, child_state, child_rank = max(
                materialized,
                key=lambda item: (item[0], item[1].action.action_id),
            )
        return edge, child_state, child_rank

    def _make_path_step(
        self,
        *,
        node: GameNode,
        edge: EdgeStats,
        current_terminal: TerminalOutcome | None,
        is_tree_edge: bool,
    ) -> PathStep:
        transition_reward = self._transition_reward(edge)
        endpoint_novelty_credit = 0.0
        if (node.state.key, edge.action.action_id) in self.frontier_replay_keys:
            self.replay_stats[f"frontier_executed_{edge.action.kind}"] += 1
            self.frontier_replay_keys.discard((node.state.key, edge.action.action_id))
        if edge.action.source in ("facet_repartition", "facet_external_exit"):
            self.facet_bank.stats[f"executed_{edge.action.source}"] += 1
        if edge.action.source == "facet_external_exit":
            endpoint_novelty_credit = self._record_endpoint_context(node, edge.action)
            transition_reward += endpoint_novelty_credit
            self.facet_bank.shared.mark_used(node.state.repair_source_word,
                                             node.state.support_word, edge.action.effective_word)
        if edge.action.kind in ("remove", "rewrite") and node.state.corrections_used >= self.config.max_corrections:
            self.replay_stats["extended_corrections"] += 1
        step = PathStep(
            node=node,
            edge=edge,
            reward=transition_reward,
            corrector_transition_cost=(
                transition_reward
                if edge.action.kind in ("remove", "rewrite")
                else 0.0
            ),
            endpoint_novelty_credit=endpoint_novelty_credit,
            is_tree_edge=is_tree_edge,
        )
        if (
            edge.action.kind in ("remove", "rewrite")
            and current_terminal is not None
        ):
            step.corrector_origin_label = current_terminal.label
            step.corrector_origin_utility = (
                current_terminal.counterfactual_utility
            )
            step.corrector_origin_support = (
                current_terminal.canonical_support_word
            )
        return step

    def _choose_replay(self, iteration: int) -> tuple[MacroAction, ...]:
        self.frontier_replay_start = None
        self.frontier_replay_keys.clear()
        interval = self.config.facet_replay_interval
        if not interval or iteration % interval:
            return ()
        eligible = [word for word, path in self.facet_witnesses.items()
                    if len(path) + 2 < self.config.max_depth
                    and sum(a.kind in ("remove", "rewrite") for a in path) < self.config.extended_max_corrections
                    and self.facet_bank.has_frontier(self.facet_witness_sources[word])]
        stride = self.config.frontier_replay_stride
        if stride and (iteration // interval) % stride == 0:
            pending = [w for w in eligible if self.facet_bank.has_delivery_frontier(self.facet_witness_sources[w])]
            if pending:
                lane = None
                if self.config.fair_frontier:
                    choice = self.facet_bank.choose_replay_frontier(
                        {w: self.facet_witness_sources[w] for w in pending})
                    if choice is not None:
                        lane, word = choice
                    else:
                        word = min(pending, key=lambda w: (self.frontier_replay_counts[w], w))
                else:
                    word = min(pending, key=lambda w: (self.frontier_replay_counts[w], w))
                    if (self.facet_bank.yield_frontier
                            and (self.replay_stats["frontier_episodes"] + 1)
                            % self.facet_bank.yield_scheduler.replay_stride == 0):
                        choice = self.facet_bank.choose_yield_frontier(
                            {w: self.facet_witness_sources[w] for w in pending})
                        if choice is not None:
                            lane, word = choice
                self.frontier_replay_counts[word] += 1
                extended = self._extend_frontier_witness(word, lane=lane)
                if self.frontier_replay_start is not None:
                    self.facet_replay_counts[word] += 1
                    self.replay_stats["episodes"] += 1
                    self.replay_stats["frontier_episodes"] += 1
                    return extended
            if self.endpoint_saturation_gate and self._context_frontier_active():
                self.replay_stats["context_frontier_opportunities"] += 1
                growth_stride = self.config.context_geometry_growth_stride
                if (
                    self.config.context_geometry_growth
                    and self.replay_stats["context_frontier_opportunities"] % growth_stride == 0
                ):
                    self._run_context_geometry_growth()
                quota_stride = self.config.context_frontier_quota_stride
                if self.replay_stats["context_frontier_opportunities"] % quota_stride == 0:
                    extended = self._choose_context_frontier_replay()
                    if extended is not None:
                        self.replay_stats["episodes"] += 1
                        self.replay_stats["frontier_episodes"] += 1
                        self.replay_stats["context_frontier_episodes"] += 1
                        return extended
                else:
                    self.replay_stats["context_frontier_quota_skips"] += 1
        if not eligible:
            return ()
        word = min(eligible, key=lambda w: (self.facet_replay_counts[w], w))
        self.facet_replay_counts[word] += 1
        self.replay_stats["episodes"] += 1
        return self.facet_witnesses[word]

    def _depth_witness_entries(
        self,
    ) -> list[tuple[tuple[int, int], tuple[MacroAction, ...], int]]:
        witnesses = dict(self.facet_depth_witnesses)
        sources = dict(self.facet_depth_witness_sources)
        for word, path in self.facet_witnesses.items():
            key = word, sum(a.kind in ("remove", "rewrite") for a in path)
            witnesses.setdefault(key, path)
            sources.setdefault(key, self.facet_witness_sources[word])
        return [
            (key, path, sources[key])
            for key, path in witnesses.items()
            if len(path) + 2 < self.config.max_depth
            and key[1] < self.config.extended_max_corrections
            and sources.get(key) in self.facet_bank.anchors
        ]

    def _choose_context_frontier_replay(self) -> tuple[MacroAction, ...] | None:
        entries = self._depth_witness_entries()
        if not entries:
            self.replay_stats["context_frontier_no_witness"] += 1
            return None
        lane_cycle = self.config.context_frontier_lane_cycle
        lane = lane_cycle[
            self.replay_stats["context_frontier_lane_requests"] % len(lane_cycle)
        ]
        self.replay_stats["context_frontier_lane_requests"] += 1
        self.replay_stats[f"context_frontier_{lane}_requests"] += 1
        entries.sort(key=lambda item: (
            self.context_frontier_attempt_counts[item[0]],
            self.context_frontier_replay_counts[item[0]],
            item[0][1],
            item[0][0],
        ))
        scan_limit = min(len(entries), self.config.context_frontier_scan_limit)
        for key, witness, raw_source in entries[:scan_limit]:
            self.context_frontier_attempt_counts[key] += 1
            stride = self.config.context_frontier_grow_stride
            grow = bool(stride and self.context_frontier_attempt_counts[key] % stride == 0)
            extended = self._extend_context_witness(
                key,
                witness,
                raw_source,
                grow=grow,
                lane=lane,
            )
            if extended is None:
                continue
            self.context_frontier_replay_counts[key] += 1
            self.facet_replay_counts[key[0]] += 1
            return extended
        self.replay_stats["context_frontier_scan_misses"] += 1
        return None

    def _run_context_geometry_growth(self) -> int | None:
        grouped: dict[int, set[int]] = {}
        for key, _witness, raw_source in self._depth_witness_entries():
            if not self.facet_bank.has_growth_frontier(raw_source):
                continue
            canonical = self.facet_bank.shared.coordinates[raw_source][0]
            grouped.setdefault(canonical, set()).add(raw_source)
        if not grouped:
            self.geometry_growth_stats["no_frontier"] += 1
            return None
        for canonical in grouped:
            self.geometry_growth_estimates.setdefault(
                canonical, GeometryGrowthEstimate()
            )

        warmup = self.config.geometry_growth_warmup
        warm = [
            canonical for canonical in grouped
            if self.geometry_growth_estimates.get(
                canonical, GeometryGrowthEstimate()
            ).observations < warmup
        ]
        next_call = self.geometry_growth_stats["calls"] + 1
        probe = next_call % self.config.geometry_growth_probe_stride == 0
        if warm:
            canonical = min(warm, key=lambda word: (
                self.geometry_growth_estimates.get(
                    word, GeometryGrowthEstimate()
                ).observations,
                self.geometry_growth_services[word],
                word,
            ))
            reason = "warmup"
        elif probe:
            canonical = min(grouped, key=lambda word: (
                self.geometry_growth_services[word],
                word,
            ))
            reason = "probe"
        else:
            service_floor = min(
                self.geometry_growth_services[word]
                for word in grouped
            )
            eligible = [
                word for word in grouped
                if self.geometry_growth_services[word]
                <= service_floor
                + self.config.geometry_growth_max_service_lead
                + self.config.geometry_growth_target_lead_bonus
                * int(bool(self.geometry_growth_estimates[word].new_targets))
            ]
            if len(eligible) < len(grouped):
                self.geometry_growth_stats["service_lead_limited_choices"] += 1
            canonical = max(eligible, key=lambda word: (
                self.geometry_growth_estimates[word].score(
                    self.geometry_growth_services[word]
                ),
                -self.geometry_growth_services[word],
                -word,
            ))
            reason = "yield"

        raw_source = min(grouped[canonical], key=lambda word: (
            self.facet_bank.anchors[word].requests,
            self.facet_bank.anchors[word].attempts,
            len(self.facet_bank.anchors[word].plans),
            word,
        ))
        graph = self.facet_bank.shared.endpoints
        before_pairs = set(graph.pairs)
        known_targets = {target for _source, target in before_pairs}
        known_targets.update(self.facet_bank.shared.pools)
        before_plans = sum(map(len, self.facet_bank.shared.pools.values()))
        started = time.perf_counter()
        self.facet_bank.grow(raw_source)
        seconds = time.perf_counter() - started
        fresh_pairs = {
            pair for pair in graph.pairs - before_pairs
            if pair[0] != pair[1]
        }
        new_targets = len({target for _source, target in fresh_pairs} - known_targets)
        new_plans = max(
            0,
            sum(map(len, self.facet_bank.shared.pools.values())) - before_plans,
        )
        estimate = self.geometry_growth_estimates.setdefault(
            canonical, GeometryGrowthEstimate()
        )
        estimate.observe(
            seconds=seconds,
            targets=new_targets,
            pairs=len(fresh_pairs),
            plans=new_plans,
        )
        self.geometry_growth_services[canonical] += 1
        self.geometry_growth_stats["calls"] += 1
        self.geometry_growth_stats[f"{reason}_calls"] += 1
        self.geometry_growth_stats["new_targets"] += new_targets
        self.geometry_growth_stats["new_pairs"] += len(fresh_pairs)
        self.geometry_growth_stats["new_plans"] += new_plans
        if len(self.geometry_growth_decisions) < self.config.max_corrector_diagnostics:
            self.geometry_growth_decisions.append({
                "iteration": self.current_iteration,
                "reason": reason,
                "canonical_source_hex": f"0x{canonical:016x}",
                "raw_source_hex": f"0x{raw_source:016x}",
                "seconds": seconds,
                "new_targets": new_targets,
                "new_pairs": len(fresh_pairs),
                "new_plans": new_plans,
                "score_after": estimate.score(
                    self.geometry_growth_services[canonical]
                ),
            })
        return canonical

    def _reconstruct_witness(
        self,
        witness: tuple[MacroAction, ...],
        raw_source: int,
    ) -> GameNode | None:
        node = self.root
        for action in witness:
            materialized = self._materialize_action(node, action)
            if materialized is None:
                return None
            node = GameNode(*materialized)
        node.facet_word = raw_source
        return node

    def _extend_context_witness(
        self,
        witness_key: tuple[int, int],
        witness: tuple[MacroAction, ...],
        raw_source: int,
        *,
        grow: bool,
        lane: str,
    ) -> tuple[MacroAction, ...] | None:
        node = self._reconstruct_witness(witness, raw_source)
        if node is None or node.state.role != "corrector":
            self.replay_stats["context_frontier_invalid_witness"] += 1
            return None
        if node.state.corrections_used != witness_key[1]:
            self.replay_stats["context_frontier_depth_mismatch"] += 1
            return None
        candidates = []
        graph = self.facet_bank.shared.endpoints
        for repair in self._facet_actions(node, grow=grow):
            if self._facet_action_context_uses(node, repair) != 0:
                continue
            materialized = self._materialize_action(node, repair)
            if materialized is None:
                continue
            retained_node = GameNode(*materialized)
            for completion in self._facet_actions(retained_node, grow=False):
                context_key = self._endpoint_context_key(
                    retained_node.state.repair_source_word,
                    retained_node.state.support_word,
                    completion.effective_word,
                    retained_node.state.context_pattern_id,
                    retained_node.state.corrections_used,
                )
                if (
                    context_key is None
                    or self.context_endpoint_executions[context_key]
                    or self._materialize_action(retained_node, completion) is None
                ):
                    continue
                source, target, _pattern_id, _depth = context_key
                if target == source:
                    continue
                pair_uses = graph.executions[(source, target)]
                if lane == "bridge" and not pair_uses:
                    continue
                lane_priority = (
                    (
                        -self.context_frontier_proposals[context_key],
                        -pair_uses,
                        int(not self.endpoint_target_executions[target]),
                    )
                    if lane == "bridge"
                    else (
                        int(not self.endpoint_target_executions[target]),
                        int(not pair_uses),
                        -self.context_frontier_proposals[context_key],
                        -pair_uses,
                    )
                )
                candidates.append((
                    *lane_priority,
                    self.knowledge.pattern_score("corrector", repair.pattern_id),
                    repair.action_id,
                    completion.action_id,
                    repair,
                    retained_node,
                    completion,
                    context_key,
                ))
        if not candidates:
            self.replay_stats["context_frontier_no_exit"] += 1
            return None
        (*_priority, repair, retained_node, completion, context_key) = max(candidates)
        self.context_frontier_proposals[context_key] += 1
        self.frontier_replay_start = len(witness)
        self.frontier_replay_keys = {
            (node.state.key, repair.action_id),
            (retained_node.state.key, completion.action_id),
        }
        self.replay_stats["context_frontier_planned_pairs"] += 1
        self.replay_stats[f"context_frontier_{lane}_planned_pairs"] += 1
        self.replay_stats["frontier_planned_pairs"] += 1
        if grow:
            self.replay_stats["context_frontier_growth_requests"] += 1
        return witness + (repair, completion)

    def _extend_frontier_witness(self, word: int, *, lane: str | None = None) -> tuple[MacroAction, ...]:
        witness = self.facet_witnesses[word]
        node = self._reconstruct_witness(witness, self.facet_witness_sources[word])
        if node is None:
            self.replay_stats["frontier_invalid_witness"] += 1
            return witness
        proposed = (self.facet_bank.next_frontier_exit(node.facet_word, lane=lane) if lane is not None
                    else self.facet_bank.next_frontier_exit(node.facet_word))
        if proposed is None:
            self.replay_stats["frontier_no_exit"] += 1
            return witness
        plan, outside = proposed
        repair = next((a for a in self._facet_actions(node, grow=False)
                       if a.pattern_id == plan.pattern_id and a.block_word == plan.retained_word), None)
        if repair is None:
            self.replay_stats["frontier_no_repair"] += 1
            return witness
        materialized = self._materialize_action(node, repair)
        if materialized is None:
            self.replay_stats["frontier_invalid_repair"] += 1
            return witness
        retained_node = GameNode(*materialized)
        completion = next((a for a in self._facet_actions(retained_node, grow=False)
                           if a.effective_word == outside), None)
        if completion is None or self._materialize_action(retained_node, completion) is None:
            self.replay_stats["frontier_invalid_completion"] += 1
            return witness
        self.frontier_replay_start = len(witness)
        self.frontier_replay_keys = {(node.state.key, repair.action_id),
                                     (retained_node.state.key, completion.action_id)}
        self.replay_stats["frontier_planned_pairs"] += 1
        return witness + (repair, completion)

    def _frontier_replay_blocked(self, node: GameNode, depth: int) -> bool:
        blocked = (self.frontier_replay_start is not None and depth == self.frontier_replay_start
                   and node.state.role == "corrector" and self._correction_limit_reached(node))
        if blocked:
            self.replay_stats["frontier_correction_limit"] += 1
            self.frontier_replay_start = None
            self.frontier_replay_keys.clear()
        return blocked

    def _replay_edge(self, node: GameNode, action: MacroAction) -> tuple[EdgeStats | None, bool]:
        if action.action_id in node.edges:
            return node.edges[action.action_id], False
        candidate = self._ephemeral_edge(node, action)
        if candidate is None:
            self.replay_stats["invalid_witness"] += 1
            return None, False
        if len(node.edges) >= self._child_limit(node):
            removable = [e for e in node.edges.values() if not e.discovery_hits and e.action.kind != "stop"]
            if not removable:
                self.replay_stats["protected_capacity_block"] += 1
                return None, False
            retired = min(removable, key=lambda e: (e.mean_value, -e.visits, e.action.action_id))
            del node.edges[retired.action.action_id]
            node.retired_action_ids.add(retired.action.action_id)
            pid = retired.action.pattern_id
            node.pattern_action_counts[pid] = max(0, node.pattern_action_counts.get(pid, 0) - 1)
            node.child_state_keys = {e.child_key for e in node.edges.values()}
            node.child_support_words = {key[0] for key in node.child_state_keys}
            self.replay_stats["recycled_edges"] += 1
        edge, state, rank = candidate
        self._node(state, rank)
        node.edges[action.action_id] = edge
        node.retired_action_ids.discard(action.action_id)
        node.child_state_keys.add(state.key)
        node.child_support_words.add(state.support_word)
        node.expanded_patterns.add(action.pattern_id)
        node.pattern_action_counts[action.pattern_id] = node.pattern_action_counts.get(action.pattern_id, 0) + 1
        self.tree_expansions += 1
        self.action_kind_counts[action.kind] += 1
        self.replay_stats["persistent_witness_edges"] += 1
        return edge, True

    def run_iteration(self, iteration: int) -> None:
        self.current_iteration = int(iteration)
        self.facet_bank.current_iteration = int(iteration)
        self.iteration_facet_counts.clear()
        repair_cycle = self.config.repair_saturation_cycle
        self.current_repair_policy = repair_cycle[(iteration - 1) % len(repair_cycle)]
        self.repair_policy_counts[self.current_repair_policy] += 1
        replay = self._choose_replay(iteration)
        cycle = self.config.rollout_policy_cycle
        self.current_rollout_policy = cycle[(iteration - 1) % len(cycle)]
        self.rollout_policy_counts[self.current_rollout_policy] += 1
        node = self.root
        path: list[PathStep] = []
        pending_corrector_index: int | None = None
        leaf_value: float | None = None
        depth = 0
        tree_edge_count = 0
        tree_leaf = self.root
        expanded = False
        last_observed_state_key: tuple[int, str, int, str, int] | None = None

        while depth < self.config.max_depth:
            if node.state.role == "done":
                break
            current_terminal: TerminalOutcome | None = None
            if node.state.role == "corrector":
                current_terminal = self._observe_terminal(node, path=path)
                last_observed_state_key = node.state.key
                if path:
                    path[-1].reward += current_terminal.reward
                    path[-1].terminal_exact = current_terminal.exact_hit
                    path[-1].terminal_new = current_terminal.new_class
                if pending_corrector_index is not None:
                    self._resolve_corrector_credit(
                        path[pending_corrector_index],
                        outcome=current_terminal,
                    )
                    pending_corrector_index = None

            replay_edge = None
            replay_expanded = False
            if self._frontier_replay_blocked(node, depth):
                replay = ()
            if depth < len(replay):
                replay_edge, replay_expanded = self._replay_edge(node, replay[depth])
                if replay_edge is None:
                    replay = ()
            expanded_edge = None if replay_edge is not None else self._widen_once(node)
            if replay_edge is not None:
                edge = replay_edge
                expanded = replay_expanded
            elif expanded_edge is not None:
                edge = expanded_edge
                expanded = True
            elif node.edges:
                edge = self._select_edge(node)
            else:
                self.dead_ends += 1
                leaf_value = self.config.dead_end_reward
                break

            step = self._make_path_step(
                node=node,
                edge=edge,
                current_terminal=current_terminal,
                is_tree_edge=True,
            )
            if step.endpoint_novelty_credit and pending_corrector_index is not None:
                path[pending_corrector_index].corrector_geometric_credit += (
                    step.endpoint_novelty_credit
                )
            if step.corrector_origin_utility is not None:
                pending_corrector_index = len(path)
            path.append(step)
            node = self.nodes[edge.child_key]
            tree_leaf = node
            tree_edge_count += 1
            self.tree_policy_steps += 1
            depth += 1
            if expanded:
                break

        if expanded and node.state.role != "done":
            while depth < self.config.max_depth:
                current_terminal = None
                if node.state.role == "corrector":
                    current_terminal = self._observe_terminal(node, path=path)
                    last_observed_state_key = node.state.key
                    if path:
                        path[-1].reward += current_terminal.reward
                        path[-1].terminal_exact = current_terminal.exact_hit
                        path[-1].terminal_new = current_terminal.new_class
                    if pending_corrector_index is not None:
                        self._resolve_corrector_credit(
                            path[pending_corrector_index],
                            outcome=current_terminal,
                        )
                        pending_corrector_index = None
                if node.state.role == "done":
                    break
                if self._frontier_replay_blocked(node, depth):
                    replay = ()
                rollout = (self._ephemeral_edge(node, replay[depth]) if depth < len(replay)
                           else self._rollout_edge(node, self.current_rollout_policy))
                if rollout is None:
                    self.dead_ends += 1
                    leaf_value = self.config.dead_end_reward
                    break
                edge, child_state, child_rank = rollout
                step = self._make_path_step(
                    node=node,
                    edge=edge,
                    current_terminal=current_terminal,
                    is_tree_edge=False,
                )
                if step.endpoint_novelty_credit and pending_corrector_index is not None:
                    path[pending_corrector_index].corrector_geometric_credit += (
                        step.endpoint_novelty_credit
                    )
                if step.corrector_origin_utility is not None:
                    pending_corrector_index = len(path)
                path.append(step)
                self.rollout_action_kind_counts[edge.action.kind] += 1
                self.rollout_steps += 1
                node = GameNode(state=child_state, rank=child_rank)
                depth += 1
                if node.state.role == "done":
                    break

        if (
            node.state.role == "corrector"
            and node.state.key != last_observed_state_key
        ):
            final_terminal = self._observe_terminal(node, path=path)
            if path:
                path[-1].reward += final_terminal.reward
                path[-1].terminal_exact = final_terminal.exact_hit
                path[-1].terminal_new = final_terminal.new_class
            if pending_corrector_index is not None:
                self._resolve_corrector_credit(
                    path[pending_corrector_index],
                    outcome=final_terminal,
                )
                pending_corrector_index = None

        if depth >= self.config.max_depth and node.state.role != "done":
            self.depth_limit_hits += 1
        if node.state.role == "done":
            leaf_value = 0.0
        elif leaf_value is None:
            leaf_value = self._leaf_value(node)
        if pending_corrector_index is not None:
            self._resolve_corrector_credit(
                path[pending_corrector_index],
                outcome=None,
                fallback_value=leaf_value,
            )

        epoch = self.knowledge.epoch
        returns = [0.0] * len(path)
        value = leaf_value
        suffix_new_class = False
        for index in range(len(path) - 1, -1, -1):
            step = path[index]
            value = step.reward + self.config.discount * value
            returns[index] = value
            suffix_new_class = suffix_new_class or step.terminal_new
            learned_value = (
                step.corrector_credit
                + step.corrector_novelty_credit
                + step.corrector_geometric_credit
                if step.edge.action.kind in ("remove", "rewrite")
                and step.corrector_credit is not None
                else value
            )
            if step.is_tree_edge:
                step.edge.update(learned_value, epoch)
                step.edge.discovery_hits += int(suffix_new_class)
                step.node.visits += 1
                step.node.value_sum += value
            if step.edge.action.kind in ("add", "remove", "rewrite"):
                role = (
                    "filler"
                    if step.edge.action.kind == "add"
                    else "corrector"
                )
                direct_exact = (
                    step.corrector_outcome_exact
                    if role == "corrector"
                    else step.terminal_exact
                )
                direct_new_class = (
                    step.corrector_outcome_new
                    if role == "corrector"
                    else step.terminal_new
                )
                self.knowledge.update_action(
                    role,
                    step.edge.action,
                    rank_before=step.edge.rank_before,
                    rank_after=step.edge.rank_after,
                    reward=learned_value,
                    exact_hit=direct_exact,
                    new_class=direct_new_class,
                    context_pattern_id=step.node.state.context_pattern_id,
                )

        if tree_edge_count < len(path):
            tree_leaf_value = returns[tree_edge_count]
        else:
            tree_leaf_value = leaf_value
        tree_leaf.visits += 1
        tree_leaf.value_sum += tree_leaf_value
        self.simulations_completed += 1

    def _tree_summary(self) -> dict[str, object]:
        role_counts = Counter(node.state.role for node in self.nodes.values())
        context_counts = Counter(
            node.state.context_pattern_id for node in self.nodes.values()
        )
        ranks = Counter(node.rank for node in self.nodes.values())
        edge_count = sum(len(node.edges) for node in self.nodes.values())
        return {
            "node_count": len(self.nodes),
            "edge_count": edge_count,
            "role_counts": dict(sorted(role_counts.items())),
            "context_counts": dict(context_counts.most_common()),
            "rank_counts": {str(rank): count for rank, count in sorted(ranks.items())},
            "root_visits": self.root.visits,
            "root_value": self.root.mean_value,
            "root_edges": [
                edge.to_dict(epoch=self.knowledge.epoch)
                for edge in sorted(
                    self.root.edges.values(),
                    key=lambda item: (item.visits, item.mean_value),
                    reverse=True,
                )
            ],
            "dead_ends": self.dead_ends,
            "depth_limit_hits": self.depth_limit_hits,
            "tree_expansions": self.tree_expansions,
            "tree_policy_steps": self.tree_policy_steps,
            "rollout_steps": self.rollout_steps,
            "mean_persistent_nodes_per_iteration": (
                (len(self.nodes) - 1) / max(1, self.simulations_completed)
            ),
            "root_child_limit": self._child_limit(self.root),
            "tree_expansion_action_kinds": dict(self.action_kind_counts),
            "rollout_action_kinds": dict(self.rollout_action_kind_counts),
            "rollout_policy_counts": dict(self.rollout_policy_counts),
            "repair_policy_counts": dict(self.repair_policy_counts),
            "epoch_recycled_edges": self.epoch_recycled_edges,
            "stagnation_recycled_edges": self.stagnation_recycled_edges,
            "recycle_event_count": len(self.recycle_events),
            "facet_replay": dict(self.replay_stats),
            "facet_witness_count": len(self.facet_witnesses),
            "facet_depth_witness_count": len(self.facet_depth_witnesses),
            "facet_depth_witnesses_by_corrections_used": {
                str(depth): count
                for depth, count in sorted(Counter(
                    key[1] for key in self.facet_depth_witnesses
                ).items())
            },
            "facet_replay_counts": {f"0x{w:016x}": count for w, count in self.facet_replay_counts.items()},
            "context_frontier_replay_counts": [
                {
                    "source_hex": f"0x{word:016x}",
                    "corrections_used": depth,
                    "attempts": self.context_frontier_attempt_counts[(word, depth)],
                    "executed_replays": count,
                }
                for (word, depth), count in sorted(self.context_frontier_replay_counts.items())
            ],
            "context_geometry_growth": {
                "enabled": self.config.context_geometry_growth,
                "class_labels_used": False,
                "selection_signal": "deterministic_endpoint_yield_with_fair_service_lead",
                "statistics": dict(self.geometry_growth_stats),
                "total_seconds": sum(
                    estimate.seconds
                    for estimate in self.geometry_growth_estimates.values()
                ),
                "sources": [
                    {
                        "canonical_source_hex": f"0x{source:016x}",
                        "services": self.geometry_growth_services[source],
                        **asdict(estimate),
                        "score": estimate.score(self.geometry_growth_services[source]),
                    }
                    for source, estimate in sorted(self.geometry_growth_estimates.items())
                ],
                "decisions": self.geometry_growth_decisions,
            },
            "tree_expansion_relations": dict(self.relation_counts),
            "generation_cache": {
                **dict(self.generation_cache_stats),
                "coherence_entries": len(self.coherence_cache),
                "candidate_entries": len(self.candidate_action_cache),
            },
            "endpoint_contexts": {
                "enabled": self.endpoint_saturation_gate,
                "repair_saturation_cycle": list(self.config.repair_saturation_cycle),
                "context_frontier_replay": self.config.context_frontier_replay,
                "class_labels_used": False,
                "key_fields": [
                    "canonical_source",
                    "canonical_target",
                    "repair_pattern_id",
                    "corrections_used",
                ],
                "distinct_contexts": len(self.context_endpoint_executions),
                "statistics": dict(self.context_endpoint_stats),
                "distinct_executed_targets": len(self.endpoint_target_executions),
                "novelty_reward_total": (
                    self.context_endpoint_stats["novel_context_executions"]
                    * self.config.endpoint_context_novelty_reward
                    + self.context_endpoint_stats["novel_pair_executions"]
                    * self.config.endpoint_pair_novelty_reward
                    + self.context_endpoint_stats["novel_target_executions"]
                    * self.config.endpoint_target_novelty_reward
                ),
                "distinct_contexts_by_corrections_used": {
                    str(depth): count
                    for depth, count in sorted(Counter(
                        key[3] for key in self.context_endpoint_executions
                    ).items())
                },
                "executions": [
                    {
                        "source_hex": f"0x{source:016x}",
                        "target_hex": f"0x{target:016x}",
                        "repair_pattern_id": pattern_id,
                        "corrections_used": corrections,
                        "executions": count,
                    }
                    for (source, target, pattern_id, corrections), count
                    in sorted(self.context_endpoint_executions.items())
                ],
            },
        }

    def result(self, elapsed_seconds: float) -> dict[str, object]:
        targets = set(int(value) for value in self.config.target_classes)
        observed = {class_id for class_id, hits in self.class_terminal_counts.items() if hits > 0}
        return {
            "metadata": {
                "algorithm": "single_tree_symmetry_filler_corrector_mcts",
                "algorithm_revision": "deterministic_fair_growth_bridge_v20",
                "one_mcts_tree": True,
                "persistent_expansions_per_simulation": 1,
                "group_theory_location": "state_action_generator_and_online_knowledge",
                "facet_examples_used_for_action_generation": False,
                "online_validated_facets_used_for_repairs": True,
                "unseen_class_examples_used": False,
                "coverage_basis": "positive_terminal_class_hits_this_run",
                "terminal_classification_only_after_rank25": True,
                "atlas_path": str(self.atlas.source_path),
            },
            "config": asdict(self.config),
            "summary": {
                "iterations_completed": self.simulations_completed,
                "elapsed_seconds": float(elapsed_seconds),
                "known_classes_at_start": sorted(self.config.known_classes),
                "discovered_class_ids": sorted(observed),
                "discovered_class_count": len(observed),
                "reward_known_class_ids": sorted(self.discovered_classes),
                "new_class_ids": sorted(
                    observed - set(self.config.known_classes)
                ),
                "missing_target_class_ids": sorted(targets - observed),
                "new_class_count": len(
                    observed - set(self.config.known_classes)
                ),
                "terminal_label_counts": dict(self.terminal_label_counts),
                "class_terminal_counts": {
                    str(class_id): count
                    for class_id, count in sorted(self.class_terminal_counts.items())
                },
                "unique_terminal_supports": len(self.terminal_support_counts),
                "corrector_outcome_counts": dict(self.corrector_outcome_counts),
                "corrector_operation_outcomes": dict(self.corrector_operation_outcomes),
            },
            "discoveries": [event.to_dict() for event in self.discovery_events],
            "growth_activations": self.growth_activations,
            "recycle_events": self.recycle_events,
            "corrector_diagnostics": self.corrector_diagnostics,
            "corrector_novelty_credit_events": self.novelty_credit_events,
            "endpoint_novelty_events": self.endpoint_novelty_events,
            "facet_corrector": self.facet_bank.to_dict(),
            "tree": self._tree_summary(),
            "knowledge_base": self.knowledge.to_dict(),
        }

    def run(self, *, output_path: Path | None = None) -> dict[str, object]:
        self.started_at = time.perf_counter()
        for iteration in range(1, self.config.iterations + 1):
            self.run_iteration(iteration)
            if all(self.class_terminal_counts[class_id] > 0 for class_id in self.config.target_classes):
                break
            if (
                output_path is not None
                and self.config.snapshot_interval > 0
                and iteration % self.config.snapshot_interval == 0
            ):
                elapsed = time.perf_counter() - self.started_at
                try:
                    _write_json(output_path, self.result(elapsed))
                except OSError as error:
                    print(f"Snapshot at iteration {iteration} deferred: {error}", file=sys.stderr)
        elapsed = time.perf_counter() - self.started_at
        payload = self.result(elapsed)
        if output_path is not None:
            _write_json(output_path, payload)
        return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Windows readers may briefly deny replacement while decoding a large report.
    for attempt in range(7):
        try:
            temporary.replace(output)
            break
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.05 * 2**attempt)


def run_group_game_search(
    config: GroupGameConfig,
    *,
    output_path: Path | None = None,
    atlas: SubgroupPatternAtlas | None = None,
    scorer: ExpansionScorer | None = None,
) -> dict[str, object]:
    engine = SingleTreeGroupGame(config, atlas=atlas, scorer=scorer)
    return engine.run(output_path=output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-tree symmetry Filler-Corrector MCTS for Bell 322",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=322_2026)
    parser.add_argument("--known-classes", type=int, nargs="*", default=[])
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--max-depth", type=int, default=72)
    parser.add_argument("--max-corrections", type=int, default=4)
    parser.add_argument("--extended-max-corrections", type=int, default=8)
    parser.add_argument("--facet-replay-interval", type=int, default=4)
    parser.add_argument("--frontier-replay-stride", type=int, default=2)
    parser.add_argument("--no-endpoint-dedup", action="store_true")
    parser.add_argument("--ridge-max-candidates", type=int, default=4096)
    parser.add_argument("--ridge-batch-size", type=int, default=32)
    parser.add_argument("--lower-face-max-attempts", type=int, default=6)
    parser.add_argument("--orbit-face-max-candidates", type=int, default=256)
    parser.add_argument("--orbit-face-batch-size", type=int, default=8)
    frontier_mode = parser.add_mutually_exclusive_group()
    frontier_mode.add_argument("--no-fair-frontier", action="store_true", help="restore v12 work order")
    frontier_mode.add_argument("--fair-frontier", action="store_true", help="enable v13 quota scheduling")
    frontier_mode.add_argument("--no-yield-frontier", action="store_true", help="restore v12 work order")
    parser.add_argument("--no-endpoint-saturation-gate", action="store_true", help="restore v14 repair choice")
    parser.add_argument("--saturated-facet-repair-probability", type=float, default=0.12)
    parser.add_argument("--repair-saturation-cycle", nargs="+", choices=("context", "global"),
                        default=list(REPAIR_SATURATION_CYCLE))
    parser.add_argument("--no-context-frontier-replay", action="store_true")
    parser.add_argument("--context-frontier-stagnation-patience", type=int, default=500)
    parser.add_argument("--context-frontier-quota-stride", type=int, default=4)
    parser.add_argument("--context-frontier-lane-cycle", nargs="+",
                        choices=("endpoint", "bridge"),
                        default=list(CONTEXT_FRONTIER_LANE_CYCLE))
    parser.add_argument("--context-frontier-scan-limit", type=int, default=12)
    parser.add_argument("--context-frontier-grow-stride", type=int, default=0)
    parser.add_argument("--no-context-geometry-growth", action="store_true")
    parser.add_argument("--context-geometry-growth-stride", type=int, default=1)
    parser.add_argument("--geometry-growth-warmup", type=int, default=2)
    parser.add_argument("--geometry-growth-probe-stride", type=int, default=4)
    parser.add_argument("--geometry-growth-max-service-lead", type=int, default=8)
    parser.add_argument("--geometry-growth-target-lead-bonus", type=int, default=16)
    parser.add_argument("--endpoint-context-novelty-reward", type=float, default=0.20)
    parser.add_argument("--endpoint-pair-novelty-reward", type=float, default=0.35)
    parser.add_argument("--endpoint-target-novelty-reward", type=float, default=0.45)
    parser.add_argument("--frontier-high-water", type=int, default=24)
    parser.add_argument("--min-corrector-rank", type=int, default=17)
    parser.add_argument("--snapshot-interval", type=int, default=0)
    parser.add_argument("--stagnation-patience", type=int, default=500)
    parser.add_argument("--rollout-policy-cycle", nargs="+", choices=("fine", "coherent", "mixed"),
                        default=list(ROLLOUT_POLICY_CYCLE))
    parser.add_argument("--no-level2-analysis", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(None if argv is None else list(argv))
    config = GroupGameConfig(
        iterations=int(args.iterations),
        seed=int(args.seed),
        known_classes=tuple(int(value) for value in args.known_classes),
        target_classes=tuple(int(value) for value in args.target_classes),
        max_depth=int(args.max_depth),
        max_corrections=int(args.max_corrections),
        extended_max_corrections=int(args.extended_max_corrections),
        facet_replay_interval=int(args.facet_replay_interval),
        frontier_replay_stride=int(args.frontier_replay_stride),
        endpoint_dedup=not args.no_endpoint_dedup,
        ridge_max_candidates=int(args.ridge_max_candidates),
        ridge_batch_size=int(args.ridge_batch_size),
        lower_face_max_attempts=int(args.lower_face_max_attempts),
        orbit_face_max_candidates=int(args.orbit_face_max_candidates),
        orbit_face_batch_size=int(args.orbit_face_batch_size),
        fair_frontier=args.fair_frontier, frontier_high_water=int(args.frontier_high_water),
        yield_frontier=not (args.no_fair_frontier or args.no_yield_frontier or args.fair_frontier),
        endpoint_saturation_gate=not (args.no_endpoint_dedup or args.no_fair_frontier
                                     or args.no_yield_frontier or args.fair_frontier
                                     or args.no_endpoint_saturation_gate),
        saturated_facet_repair_probability=float(args.saturated_facet_repair_probability),
        repair_saturation_cycle=tuple(args.repair_saturation_cycle),
        context_frontier_replay=not bool(args.no_context_frontier_replay),
        context_frontier_stagnation_patience=int(args.context_frontier_stagnation_patience),
        context_frontier_quota_stride=int(args.context_frontier_quota_stride),
        context_frontier_lane_cycle=tuple(args.context_frontier_lane_cycle),
        context_frontier_scan_limit=int(args.context_frontier_scan_limit),
        context_frontier_grow_stride=int(args.context_frontier_grow_stride),
        context_geometry_growth=not bool(args.no_context_geometry_growth),
        context_geometry_growth_stride=int(args.context_geometry_growth_stride),
        geometry_growth_warmup=int(args.geometry_growth_warmup),
        geometry_growth_probe_stride=int(args.geometry_growth_probe_stride),
        geometry_growth_max_service_lead=int(args.geometry_growth_max_service_lead),
        geometry_growth_target_lead_bonus=int(args.geometry_growth_target_lead_bonus),
        endpoint_context_novelty_reward=float(args.endpoint_context_novelty_reward),
        endpoint_pair_novelty_reward=float(args.endpoint_pair_novelty_reward),
        endpoint_target_novelty_reward=float(args.endpoint_target_novelty_reward),
        min_corrector_rank=int(args.min_corrector_rank),
        include_level2_analysis=not bool(args.no_level2_analysis),
        snapshot_interval=int(args.snapshot_interval),
        stagnation_patience=int(args.stagnation_patience),
        rollout_policy_cycle=tuple(args.rollout_policy_cycle),
    )
    output = args.output
    if output is None:
        output = DEFAULT_RUNS_DIR / f"single_tree_group_game_i{config.iterations}.json"
    payload = run_group_game_search(config, output_path=output)
    summary = payload["summary"]
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_seconds": summary["elapsed_seconds"],
                "new_class_ids": summary["new_class_ids"],
                "missing_target_class_ids": summary["missing_target_class_ids"],
                "node_count": payload["tree"]["node_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return payload


if __name__ == "__main__":
    main()
