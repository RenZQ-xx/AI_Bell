from __future__ import annotations

"""Bell 4-2-2 port of the v25 single-tree Filler-Corrector MCTS."""

import argparse
import hashlib
import json
import math
import random
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from mcts.symmetry_game.patterns import PatternRelationIndex

from .atlas import DEFAULT_HANDOFF_ROOT, HandoffPatternAtlas, IDENTITY_PATTERN_ID
from .filler_corrector import FillerCorrector, TailCompletion
from .geometry import FacetOracle, FacetResult, RankOracle
from .model import EdgeStats, GameNode, MacroAction, SearchState, size_bucket
from .problem import TARGET_AFFINE_RANK, VERTEX_COUNT
from .v25_knowledge import V25OnlineKnowledge


DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
SIZE_BUCKETS = ("singleton", "pair", "small", "medium", "large")
RELATION_LANES = (
    "continue",
    "reembed",
    "coarsen",
    "continue",
    "refine",
    "reembed",
    "intersection",
    "identity",
    "coarsen",
    "refine",
    "intersection",
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


@dataclass(frozen=True)
class V25SearchConfig:
    iterations: int = 300
    seed: int = 4_222_025
    max_depth: int = 112
    max_corrections: int = 4
    extended_max_corrections: int = 8
    min_corrector_rank: int = 60
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
    rank_checks_per_widen: int = 6
    filler_candidate_limit: int = 32
    corrector_candidate_limit: int = 48
    rollout_pattern_candidates: int = 8
    rollout_action_candidates: int = 12
    rollout_random_action_probability: float = 0.18
    rollout_corrector_probability: float = 0.88
    tail_checks: int = 24
    tail_repair_probability: float = 0.75
    stabilizer_probes: int = 192
    max_pattern_embeddings_per_family: int = 4
    max_pattern_analyses_per_class: int = 4
    facet_replay_interval: int = 4
    adaptive_replay_class_scale: int = 8
    adaptive_replay_min_coverage: float = 0.50
    free_filler_initial_ratio: float = 0.75
    free_filler_floor: float = 0.10
    free_filler_decay_power: float = 2.0
    new_class_reward: float = 10.0
    known_class_reward: float = 0.10
    known_penalty_cap: float = 0.25
    support_penalty_cap: float = 0.25
    corrector_exact_utility: float = 0.25
    corrector_exact_frequency_penalty: float = 0.04
    invalid_terminal_reward: float = -1.50
    novel_terminal_support_bonus: float = 0.35
    repeated_terminal_penalty: float = 0.12
    known_class_frequency_penalty: float = 0.18
    rank_gain_reward: float = 0.055
    flat_add_penalty: float = 0.025
    correction_size_cost: float = 0.012
    correction_rank_loss_cost: float = 0.035
    rewrite_size_cost: float = 0.008
    corrector_new_basin_bonus: float = 0.02
    endpoint_context_novelty_reward: float = 0.20
    endpoint_pair_novelty_reward: float = 0.35
    endpoint_target_novelty_reward: float = 0.45
    dead_end_reward: float = -0.75
    use_catalog: bool = True
    canonicalize_classes: bool = True
    progress_interval: int = 25
    max_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.max_depth <= 0:
            raise ValueError("iterations and max_depth must be positive")
        if not 0 <= self.min_corrector_rank < TARGET_AFFINE_RANK:
            raise ValueError("min_corrector_rank must be in [0, 78]")
        if not 0 <= self.max_corrections <= self.extended_max_corrections:
            raise ValueError("invalid correction limits")
        if min(
            self.rank_checks_per_widen,
            self.tail_checks,
            self.rollout_pattern_candidates,
            self.rollout_action_candidates,
            self.adaptive_replay_class_scale,
        ) <= 0:
            raise ValueError("search budgets must be positive")
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive when provided")


@dataclass(frozen=True)
class Transition:
    state: SearchState
    rank: int
    facet: FacetResult | None = None


@dataclass(frozen=True)
class TerminalOutcome:
    reward: float
    counterfactual_utility: float
    class_id: str | None
    exact_hit: bool
    new_class: bool
    valid: bool
    canonical_key: tuple[int, ...] | tuple[str, int]


@dataclass
class PathStep:
    node: GameNode
    edge: EdgeStats
    reward: float
    is_tree_edge: bool
    terminal_exact: bool = False
    terminal_new: bool = False
    corrector_origin_utility: float | None = None
    corrector_origin_key: tuple[int, ...] | tuple[str, int] | None = None
    corrector_credit: float | None = None
    corrector_novelty_credit: float = 0.0
    corrector_geometric_credit: float = 0.0
    corrector_outcome_exact: bool = False
    corrector_outcome_new: bool = False


class Bell422V25Search:
    """Context-sensitive one-tree v25 search over 256-bit supports."""

    def __init__(
        self,
        config: V25SearchConfig,
        *,
        handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
        atlas: HandoffPatternAtlas | None = None,
        rank_oracle: RankOracle | None = None,
        facet_oracle: FacetOracle | None = None,
    ) -> None:
        self.config = config
        self.handoff_root = Path(handoff_root).resolve()
        self.atlas = atlas or HandoffPatternAtlas(self.handoff_root)
        self.rank_oracle = rank_oracle or RankOracle()
        self.facet_oracle = facet_oracle or FacetOracle(
            rank_oracle=self.rank_oracle,
            handoff_root=self.handoff_root,
            use_catalog=config.use_catalog,
            canonicalize=config.canonicalize_classes,
        )
        self.actions = FillerCorrector(
            self.atlas,
            self.rank_oracle,
            self.facet_oracle,
            seed=config.seed,
            filler_candidate_limit=config.filler_candidate_limit,
            corrector_candidate_limit=config.corrector_candidate_limit,
            tail_checks=config.tail_checks,
            stabilizer_probes=config.stabilizer_probes,
            context_free_cache_keys=True,
            ridge_pivot=True,
            filler_max_orbits_per_action=2,
            corrector_max_orbits_per_action=4,
            max_pattern_embeddings_per_family=(
                config.max_pattern_embeddings_per_family
            ),
        )
        self.knowledge = V25OnlineKnowledge(self.atlas)
        self.pattern_order = self.atlas.fair_order(config.seed)
        root_state = SearchState(0, "filler", 0, IDENTITY_PATTERN_ID)
        self.root = GameNode(root_state, self.rank_oracle.rank(0))
        self.nodes: dict[tuple, GameNode] = {root_state.key: self.root}
        self.pattern_relations = PatternRelationIndex(VERTEX_COUNT)
        self.pattern_bucket_cache: dict[str, str] = {}
        self.rng = random.Random(config.seed)
        self.rollout_rngs = {
            policy: random.Random(
                config.seed ^ zlib.crc32(f"422-v25:{policy}".encode("ascii"))
            )
            for policy in ("fine", "coherent", "mixed")
        }
        self.discovered: dict[str, dict[str, object]] = {}
        self.discovery_events: list[dict[str, object]] = []
        self.class_frequencies: Counter[str] = Counter()
        self.support_frequencies: Counter[tuple] = Counter()
        self.class_analysis_counts: Counter[str] = Counter()
        self.facet_witnesses: dict[tuple, tuple[MacroAction, ...]] = {}
        self.facet_replay_counts: Counter[tuple] = Counter()
        self.endpoint_context_counts: Counter[tuple] = Counter()
        self.endpoint_pair_counts: Counter[tuple] = Counter()
        self.endpoint_target_counts: Counter[tuple] = Counter()
        self.endpoint_novelty_events: list[dict[str, object]] = []
        self.corrector_credit_events: list[dict[str, object]] = []
        self.recycle_events: list[dict[str, object]] = []
        self.iteration_facet_counts: Counter[tuple] = Counter()
        self.stats: Counter[str] = Counter()
        self.current_iteration = 0
        self.termination_reason = "not_started"
        self.current_rollout_policy = "mixed"
        self.adaptive_replay_balance = 0.0
        self.started_at = 0.0

    @staticmethod
    def _family_key(facet: FacetResult) -> tuple[int, ...]:
        if facet.canonical_h_row is not None:
            return tuple(map(int, facet.canonical_h_row))
        if facet.h_row is not None:
            return tuple(map(int, facet.h_row))
        return (int(facet.support_word),)

    @classmethod
    def _family_id(cls, facet: FacetResult) -> str:
        digest = hashlib.blake2b(
            repr(cls._family_key(facet)).encode("ascii"), digest_size=8
        ).hexdigest()
        return f"OBS422-{digest}"

    def _node(self, state: SearchState, rank: int) -> GameNode:
        node = self.nodes.get(state.key)
        if node is None:
            node = GameNode(state, rank)
            self.nodes[state.key] = node
        return node

    def _ensure_relation(self, pattern_id: str) -> None:
        if not self.pattern_relations.registered(pattern_id):
            self.pattern_relations.register(pattern_id, self.atlas.blocks(pattern_id))

    def _relation_from(self, context: str, target: str) -> str:
        if context == target:
            return "continue"
        if target == IDENTITY_PATTERN_ID:
            return "identity"
        if self.atlas.family_id(context) == self.atlas.family_id(target):
            return "reembed"
        self._ensure_relation(context)
        self._ensure_relation(target)
        return self.pattern_relations.directed_relation(context, target)

    def _node_relation(self, node: GameNode, target: str) -> str:
        relations = [
            self._relation_from(context, target)
            for context in node.context_pattern_ids
            if context in self.atlas.by_id
        ]
        for relation in (
            "continue",
            "reembed",
            "refine",
            "coarsen",
            "intersection",
            "identity",
        ):
            if relation in relations:
                return relation
        return "intersection"

    def _node_context(self, node: GameNode, target: str) -> str:
        order = {
            "continue": 5,
            "reembed": 4,
            "refine": 4,
            "coarsen": 3,
            "intersection": 2,
            "identity": 1,
        }
        return max(
            node.context_pattern_ids,
            key=lambda context: (
                order.get(self._relation_from(context, target), 0),
                self.knowledge.transition_score(
                    node.state.role, context, target
                ),
                context,
            ),
        )

    def _coherence(self, word: int, pattern_id: str) -> float:
        self._ensure_relation(pattern_id)
        return self.pattern_relations.coherence(word, pattern_id)

    def _pattern_bucket(self, pattern_id: str) -> str:
        cached = self.pattern_bucket_cache.get(pattern_id)
        if cached is not None:
            return cached
        pattern = self.atlas.pattern(pattern_id)
        approximate_size = max(1, round(VERTEX_COUNT / pattern.block_count))
        cached = size_bucket(approximate_size)
        self.pattern_bucket_cache[pattern_id] = cached
        return cached

    def _pattern_limit(self, node: GameNode) -> int:
        value = self.config.pattern_widening_k0 + self.config.pattern_widening_alpha * (
            max(1, node.visits) ** self.config.pattern_widening_beta
        )
        return max(1, math.floor(value))

    def _block_limit(self, node: GameNode) -> int:
        visits = max(1.0, node.visits / max(1, len(node.expanded_patterns)))
        value = self.config.block_widening_k0 + self.config.block_widening_alpha * (
            visits ** self.config.block_widening_beta
        )
        return max(1, math.floor(value))

    def _child_limit(self, node: GameNode) -> int:
        value = self.config.action_widening_k0 + self.config.action_widening_alpha * (
            max(1, node.visits) ** self.config.action_widening_beta
        )
        value += self.config.discovery_epoch_child_reserve * self.knowledge.epoch
        hard_cap = (
            self.config.max_corrector_children_per_node
            if node.state.role == "corrector"
            else self.config.max_children_per_node
        )
        return min(hard_cap, max(1, math.floor(value)))

    def _candidate_pattern_ids(self, node: GameNode) -> tuple[str, ...]:
        preferred: list[str] = sorted(node.context_pattern_ids)
        preferred.extend(self.actions.facet_patterns.get(node.state.support_word, ()))
        for context in sorted(node.context_pattern_ids):
            if context in self.atlas.by_id:
                preferred.extend(self.atlas.neighbors(context, limit=12))
                family = self.atlas.family_id(context)
                preferred.extend(self.atlas.family_embeddings(family))
        preferred.extend(
            pattern.pattern_id
            for pattern in self.atlas.patterns
            if pattern.source == "online_stabilizer"
        )
        start = node.pattern_cursor % len(self.pattern_order)
        preferred.extend(
            self.pattern_order[(start + offset) % len(self.pattern_order)]
            for offset in range(48)
        )
        return tuple(
            pattern_id
            for pattern_id in dict.fromkeys(preferred)
            if pattern_id in self.atlas.by_id
        )

    def _choose_new_pattern(self, node: GameNode) -> str | None:
        available = [
            pattern_id
            for pattern_id in self._candidate_pattern_ids(node)
            if pattern_id not in node.expanded_patterns
            and pattern_id not in node.exhausted_patterns
        ]
        if not available:
            return None
        desired = RELATION_LANES[node.relation_cursor % len(RELATION_LANES)]
        node.relation_cursor += 1
        window = available[:16]
        related = [
            pattern_id
            for pattern_id in window
            if self._node_relation(node, pattern_id) == desired
        ]
        candidates = related or window
        expanded_families = {
            self.atlas.family_id(pattern_id)
            for pattern_id in node.expanded_patterns
        }
        unseen_families = [
            pattern_id
            for pattern_id in candidates
            if self.atlas.family_id(pattern_id) not in expanded_families
        ]
        if unseen_families and desired != "reembed":
            candidates = unseen_families
        if desired != "reembed" or not related:
            representatives: dict[str, str] = {}
            for pattern_id in candidates:
                family = self.atlas.family_id(pattern_id)
                old = representatives.get(family)
                if old is None or self.knowledge.pattern_score(
                    node.state.role, pattern_id
                ) > self.knowledge.pattern_score(node.state.role, old):
                    representatives[family] = pattern_id
            candidates = list(representatives.values())
        bucket_candidates: list[str] = []
        for offset in range(len(SIZE_BUCKETS)):
            bucket = SIZE_BUCKETS[(node.bucket_cursor + offset) % len(SIZE_BUCKETS)]
            bucket_candidates = [
                pattern_id
                for pattern_id in candidates
                if self._pattern_bucket(pattern_id) == bucket
            ]
            if bucket_candidates:
                node.bucket_cursor = (node.bucket_cursor + offset + 1) % len(
                    SIZE_BUCKETS
                )
                break
        choices = bucket_candidates or candidates
        selected = max(
            choices,
            key=lambda pattern_id: (
                self.knowledge.pattern_score(node.state.role, pattern_id)
                + 0.18
                * self.knowledge.transition_score(
                    node.state.role,
                    self._node_context(node, pattern_id),
                    pattern_id,
                )
                + 0.35 * self._coherence(node.state.support_word, pattern_id),
                -self.atlas.pattern(pattern_id).order,
                pattern_id,
            ),
        )
        node.pattern_cursor += 1
        node.expanded_patterns.add(selected)
        self.stats[f"pattern_relation_{self._node_relation(node, selected)}"] += 1
        self.stats["patterns_expanded"] += 1
        return selected

    @staticmethod
    def _shape_score(node: GameNode, action: MacroAction) -> float:
        if node.state.role == "filler":
            desired = max(1, min(8, TARGET_AFFINE_RANK - node.rank + 1))
            return -0.08 * abs(action.block_size - desired) + 0.04 * (
                action.arity - 1
            )
        desired = 1 + min(5, node.state.corrections_used + 1)
        return -0.035 * abs(action.block_size - desired) + 0.025 * min(
            action.arity - 1, 3
        )

    def _approximate_action_score(
        self, node: GameNode, action: MacroAction
    ) -> float:
        relation = self._node_relation(node, action.pattern_id)
        return (
            self.knowledge.pattern_score(node.state.role, action.pattern_id)
            + self._shape_score(node, action)
            + {
                "continue": 0.30,
                "reembed": 0.20,
                "refine": 0.18,
                "coarsen": 0.14,
                "intersection": -0.01,
                "identity": -0.04,
            }.get(relation, 0.0)
            + 0.45 * self._coherence(node.state.support_word, action.pattern_id)
        )

    def _actions_for_pattern(
        self,
        node: GameNode,
        pattern_id: str,
        lane: str,
    ) -> list[MacroAction]:
        if node.state.role == "filler":
            actions = self.actions.filler_actions(node.state, pattern_id, lane=lane)
            if not actions and lane != "single":
                actions = self.actions.filler_actions(
                    node.state, pattern_id, lane="single"
                )
        else:
            actions = self.actions.corrector_actions(
                node.state, pattern_id, lane=lane
            )
            if lane == "rewrite":
                actions.extend(
                    self.actions.cross_pattern_rewrites(
                        node.state, pattern_id, limit=24
                    )
                )
            if not actions:
                actions = self.actions.corrector_actions(
                    node.state, pattern_id, lane="remove_small"
                )
        return [
            action
            for action in actions
            if action.action_id not in node.edges
            and action.action_id not in node.rejected_action_ids
            and action.action_id not in node.retired_action_ids
        ]

    def _transition(self, state: SearchState, action: MacroAction) -> Transition | None:
        if action.kind == "stop":
            return Transition(
                SearchState(
                    state.support_word,
                    "done",
                    state.corrections_used,
                    state.context_pattern_id,
                    state.source_facet_word,
                ),
                self.rank_oracle.rank(state.support_word),
            )
        word, rank = self.actions.apply(state, action)
        if word == state.support_word or rank > TARGET_AFFINE_RANK:
            return None
        corrections = state.corrections_used + int(
            action.kind in ("remove", "rewrite")
        )
        if rank == TARGET_AFFINE_RANK:
            facet = self.facet_oracle.validate(word)
            if facet.valid:
                return Transition(
                    SearchState(
                        facet.support_word,
                        "corrector",
                        corrections,
                        action.pattern_id,
                        facet.support_word,
                    ),
                    TARGET_AFFINE_RANK,
                    facet,
                )
            return Transition(
                SearchState(
                    word,
                    "done",
                    corrections,
                    action.pattern_id,
                    state.source_facet_word,
                ),
                rank,
                facet,
            )
        if action.kind in ("remove", "rewrite") and rank < self.config.min_corrector_rank:
            return None
        return Transition(
            SearchState(
                word,
                "filler",
                corrections,
                action.pattern_id,
                state.source_facet_word,
            ),
            rank,
        )

    def _edge_for(
        self,
        node: GameNode,
        action: MacroAction,
        transition: Transition | None = None,
    ) -> tuple[EdgeStats, Transition] | None:
        moved = transition or self._transition(node.state, action)
        if moved is None:
            return None
        relation = (
            "continue"
            if action.kind == "stop"
            else self._node_relation(node, action.pattern_id)
        )
        context = (
            node.state.context_pattern_id
            if action.kind == "stop"
            else self._node_context(node, action.pattern_id)
        )
        prior = self.knowledge.action_prior(
            node.state.role,
            action,
            rank_before=node.rank,
            rank_after=moved.rank,
            context_pattern_id=context,
            coherence=(
                0.0
                if action.kind == "stop"
                else self._coherence(moved.state.support_word, action.pattern_id)
            ),
            relation=relation,
        )
        return (
            EdgeStats(action, moved.state.key, prior, node.rank, moved.rank),
            moved,
        )

    @staticmethod
    def _stop_action() -> MacroAction:
        return MacroAction(
            "stop", "__stop__", (), 0, "control", "stop"
        )

    def _correction_limit_reached(self, node: GameNode) -> bool:
        if node.state.corrections_used >= self.config.extended_max_corrections:
            return True
        facet = self.facet_oracle.validate(node.state.support_word)
        if facet.valid:
            key = self._family_key(facet)
            if self.iteration_facet_counts[key] >= 3:
                return True
        return False

    def _add_stop_edge(self, node: GameNode) -> EdgeStats | None:
        if "stop" in node.edges:
            return None
        candidate = self._edge_for(node, self._stop_action())
        if candidate is None:
            return None
        edge, moved = candidate
        edge.prior = 2.0 if self._correction_limit_reached(node) else 0.25
        self._node(moved.state, moved.rank)
        node.edges["stop"] = edge
        node.child_keys.add(moved.state.key)
        return edge

    def _tail_candidate(
        self,
        node: GameNode,
        *,
        rng: random.Random,
    ) -> tuple[MacroAction, Transition] | None:
        if node.state.role != "filler" or node.rank not in (
            TARGET_AFFINE_RANK - 2,
            TARGET_AFFINE_RANK - 1,
        ):
            return None
        if rng.random() >= self.config.tail_repair_probability:
            return None
        pattern_ids = tuple(node.context_pattern_ids) + tuple(
            sorted(node.expanded_patterns)
        )
        completion = self.actions.tail_complete(
            node.state,
            pattern_ids,
            max_checks=self.config.tail_checks,
        )
        if completion is None:
            return None
        moved = self._transition(node.state, completion.action)
        if moved is None or moved.facet is None or not moved.facet.valid:
            return None
        return completion.action, moved

    def _refresh_node_epoch(self, node: GameNode) -> None:
        epoch = self.knowledge.epoch
        old_epoch = node.knowledge_epoch_seen
        node.knowledge_epoch_seen = epoch
        last_discovery = (
            self.discovery_events[-1]["iteration"] if self.discovery_events else 0
        )
        stagnant = (
            self.current_iteration - int(last_discovery)
            >= self.config.stagnation_patience
            and node.visits - node.last_recycle_visit
            >= self.config.recycle_visit_interval
        )
        epoch_changed = old_epoch >= 0 and old_epoch != epoch
        if not (epoch_changed or stagnant):
            return
        if len(node.edges) < self._child_limit(node):
            return
        candidates = sorted(
            (
                edge
                for edge in node.edges.values()
                if edge.action.kind != "stop"
                and edge.visits >= 8
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
        if not retired:
            return
        node.child_keys = {edge.child_key for edge in node.edges.values()}
        node.last_recycle_visit = node.visits
        fresh: list[str] = []
        for _ in range(min(retired, self.config.recycle_fresh_patterns)):
            pattern_id = self._choose_new_pattern(node)
            if pattern_id is None:
                break
            fresh.append(pattern_id)
        self.recycle_events.append(
            {
                "iteration": self.current_iteration,
                "reason": "discovery_epoch" if epoch_changed else "stagnation",
                "support_word_hex": f"0x{node.state.support_word:064x}",
                "retired_edges": retired,
                "fresh_pattern_ids": fresh,
            }
        )

    def _install_edge(
        self,
        node: GameNode,
        edge: EdgeStats,
        moved: Transition,
    ) -> EdgeStats:
        self._node(moved.state, moved.rank)
        node.edges[edge.action.action_id] = edge
        node.child_keys.add(moved.state.key)
        node.pattern_action_counts[edge.action.pattern_id] = (
            node.pattern_action_counts.get(edge.action.pattern_id, 0) + 1
        )
        self.stats["tree_expansions"] += 1
        self.stats[f"tree_action_{edge.action.kind}"] += 1
        return edge

    def _widen_once(self, node: GameNode) -> EdgeStats | None:
        if node.state.role == "done":
            return None
        if node.state.role == "corrector":
            stop = self._add_stop_edge(node)
            if stop is not None:
                self.stats["tree_expansions"] += 1
                return stop
            if self._correction_limit_reached(node):
                return None
        self._refresh_node_epoch(node)
        if len(node.edges) >= self._child_limit(node):
            return None

        tail = self._tail_candidate(node, rng=self.rng)
        if tail is not None:
            action, moved = tail
            if action.action_id not in node.edges and moved.state.key not in node.child_keys:
                candidate = self._edge_for(node, action, moved)
                if candidate is not None:
                    return self._install_edge(node, *candidate)

        desired_patterns = min(
            self._pattern_limit(node), len(self._candidate_pattern_ids(node))
        )
        if len(node.expanded_patterns) < desired_patterns:
            self._choose_new_pattern(node)
        lanes = (
            FILLER_ACTION_LANES
            if node.state.role == "filler"
            else CORRECTOR_ACTION_LANES
        )
        lane = lanes[node.action_lane_cursor % len(lanes)]
        node.action_lane_cursor += 1
        attempted_patterns: set[str] = set()
        for _ in range(max(1, len(node.expanded_patterns) + 2)):
            limit = self._block_limit(node)
            eligible = [
                pattern_id
                for pattern_id in node.expanded_patterns
                if node.pattern_action_counts.get(pattern_id, 0) < limit
                and pattern_id not in attempted_patterns
            ]
            if not eligible:
                pattern_id = self._choose_new_pattern(node)
                if pattern_id is None:
                    return None
                eligible = [pattern_id]
            pattern_id = max(
                eligible,
                key=lambda value: (
                    -node.pattern_action_counts.get(value, 0),
                    self.knowledge.pattern_score(node.state.role, value)
                    + 0.25 * self._coherence(node.state.support_word, value),
                    value,
                ),
            )
            candidates = self._actions_for_pattern(node, pattern_id, lane)
            if not candidates:
                attempted_patterns.add(pattern_id)
                continue
            candidates.sort(
                key=lambda action: (
                    self._approximate_action_score(node, action),
                    -action.block_size,
                    action.action_id,
                ),
                reverse=True,
            )
            for action in candidates[: self.config.rank_checks_per_widen]:
                moved = self._transition(node.state, action)
                if moved is None:
                    node.rejected_action_ids.add(action.action_id)
                    self.stats["actions_rejected"] += 1
                    continue
                if moved.state.key in node.child_keys:
                    node.rejected_action_ids.add(action.action_id)
                    self.stats["duplicate_children"] += 1
                    continue
                candidate = self._edge_for(node, action, moved)
                if candidate is not None:
                    return self._install_edge(node, *candidate)
        return None

    def _select_edge(self, node: GameNode) -> EdgeStats | None:
        if not node.edges:
            return None
        if node.state.role == "corrector" and self._correction_limit_reached(node):
            return node.edges.get("stop")
        epoch = self.knowledge.epoch
        root = math.sqrt(node.visits + 1.0)

        def score(edge: EdgeStats) -> tuple[float, str]:
            recent = edge.epoch_mean(epoch)
            q_value = 0.0 if recent is None else recent
            epoch_visits = edge.epoch_visits if edge.epoch == epoch else 0
            exploration = (
                self.config.exploration
                * edge.prior
                * root
                / (1.0 + epoch_visits)
            )
            return q_value + exploration + 1e-9 * self.rng.random(), edge.action.action_id

        return max(node.edges.values(), key=score)

    def _terminal_outcome(
        self,
        facet: FacetResult,
        *,
        path: Sequence[PathStep],
        depth: int,
        source: str,
    ) -> TerminalOutcome:
        if not facet.valid or facet.class_id is None:
            return TerminalOutcome(
                self.config.invalid_terminal_reward,
                self.config.invalid_terminal_reward,
                None,
                False,
                False,
                False,
                ("invalid", int(facet.selected_word)),
            )
        class_id = facet.class_id
        canonical = self._family_key(facet)
        previous_class_hits = self.class_frequencies[class_id]
        previous_support_hits = self.support_frequencies[canonical]
        self.class_frequencies[class_id] += 1
        self.support_frequencies[canonical] += 1
        new_class = class_id not in self.discovered

        family_id = self._family_id(facet)
        if (
            facet.support_word not in self.actions.facet_patterns
            and self.class_analysis_counts[class_id]
            < self.config.max_pattern_analyses_per_class
        ):
            patterns = self.actions.observe_facet(
                facet.support_word,
                path[-1].edge.action.pattern_id if path else IDENTITY_PATTERN_ID,
                family_id=family_id,
            )
            self.class_analysis_counts[class_id] += 1
        else:
            patterns = self.actions.facet_patterns.get(
                facet.support_word,
                (path[-1].edge.action.pattern_id if path else IDENTITY_PATTERN_ID,),
            )

        witness = tuple(step.edge.action for step in path)
        old = self.facet_witnesses.get(canonical)
        witness_cost = (
            sum(action.kind in ("remove", "rewrite") for action in witness),
            len(witness),
        )
        if witness and (
            old is None
            or witness_cost
            < (
                sum(action.kind in ("remove", "rewrite") for action in old),
                len(old),
            )
        ):
            self.facet_witnesses[canonical] = witness

        self.iteration_facet_counts[canonical] += 1
        if new_class:
            self.knowledge.begin_discovery_epoch()
            self._assign_corrector_novelty(path, class_id)
            event = {
                "class_id": class_id,
                "catalog_index": facet.catalog_index,
                "iteration": self.current_iteration,
                "depth": depth,
                "elapsed_seconds": time.perf_counter() - self.started_at,
                "source": source,
                "rollout_policy": self.current_rollout_policy,
                "support_word_hex": f"0x{facet.support_word:064x}",
                "support_size": facet.support_word.bit_count(),
                "family_id": family_id,
                "online_pattern_ids": list(patterns),
                "path": [step.edge.action.to_dict() for step in path],
            }
            self.discovered[class_id] = event
            self.discovery_events.append(event)
            self.stats["new_classes"] += 1

        class_penalty = min(
            self.config.known_penalty_cap,
            self.config.known_class_frequency_penalty
            * math.log1p(previous_class_hits),
        )
        reward = (
            self.config.new_class_reward
            if new_class
            else self.config.known_class_reward - class_penalty
        )
        if previous_support_hits == 0:
            reward += self.config.novel_terminal_support_bonus
        else:
            reward -= min(
                self.config.support_penalty_cap,
                self.config.repeated_terminal_penalty
                * math.log1p(previous_support_hits),
            )
        utility = self.config.corrector_exact_utility - min(
            0.20,
            self.config.corrector_exact_frequency_penalty
            * math.log1p(previous_class_hits + 1),
        )
        return TerminalOutcome(
            float(reward),
            float(utility),
            class_id,
            True,
            new_class,
            True,
            canonical,
        )

    def _observe_terminal(
        self,
        node: GameNode,
        *,
        path: Sequence[PathStep],
        depth: int,
        source: str,
    ) -> TerminalOutcome:
        facet = self.facet_oracle.validate(node.state.support_word)
        return self._terminal_outcome(
            facet, path=path, depth=depth, source=source
        )

    def _assign_corrector_novelty(
        self, path: Sequence[PathStep], class_id: str
    ) -> None:
        contributors = [
            (index, step)
            for index, step in enumerate(path)
            if step.corrector_origin_utility is not None
        ]
        if not contributors:
            return
        weights = [
            self.config.discount ** (len(path) - 1 - index)
            for index, _step in contributors
        ]
        total = sum(weights)
        credits = []
        for (index, step), weight in zip(contributors, weights):
            credit = self.config.new_class_reward * weight / total
            step.corrector_novelty_credit += credit
            credits.append({"path_index": index, "credit": credit})
        self.corrector_credit_events.append(
            {
                "iteration": self.current_iteration,
                "class_id": class_id,
                "credits": credits,
            }
        )

    def _resolve_corrector_credit(
        self,
        step: PathStep,
        outcome: TerminalOutcome | None,
        *,
        fallback: float = 0.0,
    ) -> None:
        baseline = step.corrector_origin_utility
        if baseline is None:
            return
        if outcome is None:
            result = min(float(baseline), float(fallback))
        else:
            result = outcome.counterfactual_utility
        basin_bonus = (
            self.config.corrector_new_basin_bonus
            if outcome is not None
            and step.corrector_origin_key is not None
            and outcome.canonical_key != step.corrector_origin_key
            else 0.0
        )
        step.corrector_credit = (
            result - baseline + basin_bonus + self._transition_reward(step.edge)
        )
        step.corrector_outcome_exact = bool(outcome and outcome.exact_hit)
        step.corrector_outcome_new = bool(outcome and outcome.new_class)

    def _transition_reward(self, edge: EdgeStats) -> float:
        action = edge.action
        if action.kind == "add":
            gain = edge.rank_after - edge.rank_before
            return self.config.rank_gain_reward * gain - (
                self.config.flat_add_penalty if gain <= 0 else 0.0
            )
        rank_loss = max(0, edge.rank_before - edge.rank_after)
        if action.kind == "remove":
            return -(
                self.config.correction_size_cost * action.block_size
                + self.config.correction_rank_loss_cost * rank_loss
            )
        if action.kind == "rewrite":
            changed = action.remove_word.bit_count() + action.add_word.bit_count()
            return -(
                self.config.rewrite_size_cost * changed
                + self.config.correction_rank_loss_cost * rank_loss
            )
        return 0.0

    def _endpoint_key(self, word: int) -> tuple[int, ...] | None:
        if not word:
            return None
        facet = self.facet_oracle.validate(word)
        return self._family_key(facet) if facet.valid else None

    def _endpoint_novelty(
        self,
        node: GameNode,
        edge: EdgeStats,
        child_state: SearchState,
    ) -> float:
        source = self._endpoint_key(node.state.source_facet_word)
        target = self._endpoint_key(child_state.source_facet_word)
        if source is None or target is None or source == target:
            return 0.0
        context = (
            source,
            target,
            edge.action.pattern_id,
            child_state.corrections_used,
        )
        pair = source, target
        first_context = self.endpoint_context_counts[context] == 0
        first_pair = self.endpoint_pair_counts[pair] == 0
        first_target = self.endpoint_target_counts[target] == 0
        self.endpoint_context_counts[context] += 1
        self.endpoint_pair_counts[pair] += 1
        self.endpoint_target_counts[target] += 1
        reward = (
            self.config.endpoint_context_novelty_reward * first_context
            + self.config.endpoint_pair_novelty_reward * first_pair
            + self.config.endpoint_target_novelty_reward * first_target
        )
        if reward:
            self.endpoint_novelty_events.append(
                {
                    "iteration": self.current_iteration,
                    "source_class_key": hashlib.blake2b(
                        repr(source).encode("ascii"), digest_size=6
                    ).hexdigest(),
                    "target_class_key": hashlib.blake2b(
                        repr(target).encode("ascii"), digest_size=6
                    ).hexdigest(),
                    "pattern_id": edge.action.pattern_id,
                    "reward": reward,
                }
            )
        return float(reward)

    def _make_step(
        self,
        node: GameNode,
        edge: EdgeStats,
        child_state: SearchState,
        *,
        current_terminal: TerminalOutcome | None,
        is_tree_edge: bool,
    ) -> PathStep:
        endpoint = self._endpoint_novelty(node, edge, child_state)
        step = PathStep(
            node,
            edge,
            self._transition_reward(edge) + endpoint,
            is_tree_edge,
            corrector_geometric_credit=0.0,
        )
        if edge.action.kind in ("remove", "rewrite") and current_terminal is not None:
            step.corrector_origin_utility = current_terminal.counterfactual_utility
            step.corrector_origin_key = current_terminal.canonical_key
        return step

    def _rollout_pattern_ids(
        self, node: GameNode, policy: str
    ) -> list[str]:
        rng = self.rollout_rngs[policy]
        active = [pattern.pattern_id for pattern in self.atlas.patterns]
        contexts = {
            pattern_id
            for pattern_id in node.context_pattern_ids
            if pattern_id in self.atlas.by_id
        }
        candidates = set(contexts)
        neighbors = sorted(
            {
                neighbor
                for context in contexts
                for neighbor in self.atlas.neighbors(context, limit=8)
            }
        )
        if neighbors:
            candidates.update(rng.sample(neighbors, min(3, len(neighbors))))
        siblings = [
            pattern_id
            for pattern_id in active
            if pattern_id not in contexts
            and any(
                self.atlas.family_id(pattern_id) == self.atlas.family_id(context)
                for context in contexts
            )
        ]
        if siblings:
            candidates.update(rng.sample(siblings, min(2, len(siblings))))
        representatives: dict[str, str] = {}
        for pattern_id in active:
            family = self.atlas.family_id(pattern_id)
            old = representatives.get(family)
            if old is None or self.knowledge.pattern_score(
                node.state.role, pattern_id
            ) > self.knowledge.pattern_score(node.state.role, old):
                representatives[family] = pattern_id
        top = sorted(
            representatives.values(),
            key=lambda pattern_id: (
                self.knowledge.pattern_score(node.state.role, pattern_id),
                pattern_id,
            ),
            reverse=True,
        )[:3]
        candidates.update(top)
        remaining = [item for item in active if item not in candidates]
        if remaining:
            candidates.update(rng.sample(remaining, min(3, len(remaining))))
        if policy == "fine":
            candidates.add(IDENTITY_PATTERN_ID)
            candidates = {
                pattern_id
                for pattern_id in candidates
                if self._pattern_bucket(pattern_id) in ("singleton", "pair")
            }
        ranked = sorted(
            candidates,
            key=lambda pattern_id: (
                int(policy == "coherent" and pattern_id in contexts)
                + int(policy == "fine" and pattern_id == IDENTITY_PATTERN_ID),
                self.knowledge.pattern_score(node.state.role, pattern_id)
                + 0.30 * self._coherence(node.state.support_word, pattern_id)
                + {
                    "continue": 0.30,
                    "reembed": 0.24,
                    "refine": 0.22,
                    "coarsen": 0.18,
                    "intersection": 0.10,
                    "identity": 0.04,
                }.get(self._node_relation(node, pattern_id), 0.0),
                pattern_id,
            ),
            reverse=True,
        )
        return ranked[: self.config.rollout_pattern_candidates]

    def _rollout_lane(self, role: str, policy: str) -> str:
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

    def _rollout_edge(
        self, node: GameNode, policy: str
    ) -> tuple[EdgeStats, Transition] | None:
        rng = self.rollout_rngs[policy]
        if node.state.role == "corrector" and (
            self._correction_limit_reached(node)
            or (
                node.state.corrections_used < self.config.max_corrections
                and rng.random() > self.config.rollout_corrector_probability
            )
        ):
            return self._edge_for(node, self._stop_action())
        tail = self._tail_candidate(node, rng=rng)
        if tail is not None:
            action, moved = tail
            return self._edge_for(node, action, moved)

        lane = self._rollout_lane(node.state.role, policy)
        rough: list[MacroAction] = []
        for pattern_id in self._rollout_pattern_ids(node, policy):
            selected = self._actions_for_pattern(node, pattern_id, lane)
            selected.sort(
                key=lambda action: (
                    self._shape_score(node, action),
                    -action.block_size,
                    action.action_id,
                ),
                reverse=True,
            )
            rough.extend(selected[:1])
            if len(selected) > 1:
                rough.append(rng.choice(selected[1:]))
        if not rough:
            return (
                self._edge_for(node, self._stop_action())
                if node.state.role == "corrector"
                else None
            )
        if rng.random() < self.config.rollout_random_action_probability:
            rng.shuffle(rough)
        else:
            rough.sort(
                key=lambda action: (
                    int(
                        policy == "coherent"
                        and action.pattern_id in node.context_pattern_ids
                    ),
                    self._approximate_action_score(node, action),
                ),
                reverse=True,
            )
        materialized: list[tuple[float, EdgeStats, Transition]] = []
        for action in rough[: self.config.rollout_action_candidates]:
            candidate = self._edge_for(node, action)
            if candidate is None:
                continue
            edge, moved = candidate
            materialized.append((edge.prior + 0.08 * rng.random(), edge, moved))
        if not materialized:
            return (
                self._edge_for(node, self._stop_action())
                if node.state.role == "corrector"
                else None
            )
        if policy == "coherent" and node.state.role == "filler":
            continued = [
                item
                for item in materialized
                if item[1].action.pattern_id in node.context_pattern_ids
            ]
            if continued and rng.random() < 0.85:
                materialized = continued
        if rng.random() < self.config.rollout_random_action_probability:
            _score, edge, moved = rng.choice(materialized)
        else:
            _score, edge, moved = max(
                materialized,
                key=lambda item: (item[0], item[1].action.action_id),
            )
        return edge, moved

    def _free_filler_ratio(self) -> float:
        coverage = min(
            1.0,
            len(self.discovered) / self.config.adaptive_replay_class_scale,
        )
        if coverage < self.config.adaptive_replay_min_coverage:
            return self.config.free_filler_initial_ratio
        span = self.config.free_filler_initial_ratio - self.config.free_filler_floor
        return self.config.free_filler_floor + span * (
            (1.0 - coverage) ** self.config.free_filler_decay_power
        )

    def _choose_replay(self, iteration: int) -> tuple[MacroAction, ...]:
        mandatory = bool(
            self.config.facet_replay_interval
            and iteration % self.config.facet_replay_interval == 0
        )
        target_replay = 1.0 - self._free_filler_ratio()
        periodic = (
            0.0
            if not self.config.facet_replay_interval
            else 1.0 / self.config.facet_replay_interval
        )
        self.adaptive_replay_balance += max(0.0, target_replay - periodic)
        due = mandatory
        if not due and self.adaptive_replay_balance >= 1.0:
            self.adaptive_replay_balance -= 1.0
            due = True
            self.stats["adaptive_replay_slots"] += 1
        elif mandatory:
            self.stats["periodic_replay_slots"] += 1
        if not due or not self.facet_witnesses:
            self.stats["free_filler_slots"] += int(not due)
            return ()
        eligible = [
            key
            for key, witness in self.facet_witnesses.items()
            if len(witness) + 2 < self.config.max_depth
        ]
        if not eligible:
            return ()
        key = min(
            eligible,
            key=lambda item: (
                self.facet_replay_counts[item],
                self.endpoint_target_counts[item],
                repr(item),
            ),
        )
        self.facet_replay_counts[key] += 1
        self.stats["replay_episodes"] += 1
        return self.facet_witnesses[key]

    def _replay_edge(
        self, node: GameNode, action: MacroAction
    ) -> tuple[EdgeStats | None, bool]:
        existing = node.edges.get(action.action_id)
        if existing is not None:
            return existing, False
        candidate = self._edge_for(node, action)
        if candidate is None:
            self.stats["invalid_replay_actions"] += 1
            return None, False
        edge, moved = candidate
        for old in node.edges.values():
            if old.child_key == moved.state.key:
                return old, False
        if len(node.edges) >= self._child_limit(node):
            removable = [
                edge
                for edge in node.edges.values()
                if edge.action.kind != "stop" and edge.discovery_hits == 0
            ]
            if not removable:
                return None, False
            retired = min(
                removable,
                key=lambda item: (item.mean_value, -item.visits, item.action.action_id),
            )
            del node.edges[retired.action.action_id]
            node.retired_action_ids.add(retired.action.action_id)
            node.pattern_action_counts[retired.action.pattern_id] = max(
                0,
                node.pattern_action_counts.get(retired.action.pattern_id, 0) - 1,
            )
            node.child_keys = {item.child_key for item in node.edges.values()}
            self.stats["replay_recycled_edges"] += 1
        self._install_edge(node, edge, moved)
        return edge, True

    def run_iteration(self, iteration: int) -> None:
        self.current_iteration = iteration
        self.iteration_facet_counts.clear()
        self.current_rollout_policy = ROLLOUT_POLICY_CYCLE[
            (iteration - 1) % len(ROLLOUT_POLICY_CYCLE)
        ]
        self.stats[f"rollout_policy_{self.current_rollout_policy}"] += 1
        replay = self._choose_replay(iteration)
        node = self.root
        path: list[PathStep] = []
        pending_corrector: int | None = None
        persistent = True
        tree_leaf = self.root
        depth = 0
        leaf_value = 0.0
        last_observed_key: tuple | None = None

        while depth < self.config.max_depth:
            current_terminal: TerminalOutcome | None = None
            if node.state.role == "done":
                if path and path[-1].edge.action.kind != "stop":
                    current_terminal = TerminalOutcome(
                        self.config.invalid_terminal_reward,
                        self.config.invalid_terminal_reward,
                        None,
                        False,
                        False,
                        False,
                        ("invalid", int(node.state.support_word)),
                    )
                    path[-1].reward += current_terminal.reward
                    if pending_corrector is not None:
                        self._resolve_corrector_credit(
                            path[pending_corrector], current_terminal
                        )
                        pending_corrector = None
                break
            if node.state.role == "corrector":
                current_terminal = self._observe_terminal(
                    node,
                    path=path,
                    depth=depth,
                    source="tree" if persistent else "rollout",
                )
                last_observed_key = node.state.key
                if path:
                    path[-1].reward += current_terminal.reward
                    path[-1].terminal_exact = current_terminal.exact_hit
                    path[-1].terminal_new = current_terminal.new_class
                if pending_corrector is not None:
                    self._resolve_corrector_credit(
                        path[pending_corrector], current_terminal
                    )
                    pending_corrector = None

            if persistent:
                replay_edge = None
                replay_expanded = False
                if depth < len(replay):
                    replay_edge, replay_expanded = self._replay_edge(
                        node, replay[depth]
                    )
                    if replay_edge is None:
                        replay = ()
                widened = None if replay_edge is not None else self._widen_once(node)
                if replay_edge is not None:
                    edge = replay_edge
                    expanded = replay_expanded
                elif widened is not None:
                    edge = widened
                    expanded = True
                else:
                    edge = self._select_edge(node)
                    expanded = False
                if edge is None:
                    leaf_value = self.config.dead_end_reward
                    self.stats["dead_ends"] += 1
                    break
                child = self.nodes[edge.child_key]
                moved = Transition(child.state, child.rank)
                child_node = child
            else:
                rollout = self._rollout_edge(node, self.current_rollout_policy)
                if rollout is None:
                    leaf_value = self.config.dead_end_reward
                    self.stats["dead_ends"] += 1
                    break
                edge, moved = rollout
                expanded = False
                child_node = GameNode(moved.state, moved.rank)

            step = self._make_step(
                node,
                edge,
                moved.state,
                current_terminal=current_terminal,
                is_tree_edge=persistent,
            )
            endpoint_credit = step.reward - self._transition_reward(edge)
            if endpoint_credit and pending_corrector is not None:
                path[pending_corrector].corrector_geometric_credit += endpoint_credit
            if step.corrector_origin_utility is not None:
                pending_corrector = len(path)
            path.append(step)
            node = child_node
            depth += 1
            if persistent:
                tree_leaf = node
                self.stats["tree_policy_steps"] += 1
                if expanded:
                    persistent = False
            else:
                self.stats["rollout_steps"] += 1

        if (
            depth >= self.config.max_depth
            and node.state.role != "done"
        ):
            self.stats["depth_limit_hits"] += 1
        if (
            node.state.role == "corrector"
            and node.state.key != last_observed_key
        ):
            final = self._observe_terminal(
                node,
                path=path,
                depth=depth,
                source="tree" if persistent else "rollout",
            )
            if path:
                path[-1].reward += final.reward
                path[-1].terminal_exact = final.exact_hit
                path[-1].terminal_new = final.new_class
            if pending_corrector is not None:
                self._resolve_corrector_credit(path[pending_corrector], final)
                pending_corrector = None
        if pending_corrector is not None:
            self._resolve_corrector_credit(
                path[pending_corrector], None, fallback=leaf_value
            )

        epoch = self.knowledge.epoch
        value = leaf_value
        suffix_new_class = False
        returns = [0.0] * len(path)
        for index in range(len(path) - 1, -1, -1):
            step = path[index]
            value = step.reward + self.config.discount * value
            returns[index] = value
            suffix_new_class = suffix_new_class or step.terminal_new
            learned = (
                step.corrector_credit
                + step.corrector_novelty_credit
                + step.corrector_geometric_credit
                if step.edge.action.kind in ("remove", "rewrite")
                and step.corrector_credit is not None
                else value
            )
            if step.is_tree_edge:
                step.edge.update(learned, epoch)
                step.edge.discovery_hits += int(suffix_new_class)
                step.node.visits += 1
                step.node.value_sum += value
            if step.edge.action.kind in ("add", "remove", "rewrite"):
                role = "filler" if step.edge.action.kind == "add" else "corrector"
                relation = self._node_relation(step.node, step.edge.action.pattern_id)
                self.knowledge.update_action(
                    role,
                    step.edge.action,
                    rank_before=step.edge.rank_before,
                    rank_after=step.edge.rank_after,
                    reward=learned,
                    exact_hit=(
                        step.terminal_exact
                        if role == "filler"
                        else step.corrector_outcome_exact
                    ),
                    new_class=(
                        step.terminal_new
                        if role == "filler"
                        else step.corrector_outcome_new
                    ),
                    context_pattern_id=self._node_context(
                        step.node, step.edge.action.pattern_id
                    ),
                    relation=relation,
                )
        tree_edge_count = sum(step.is_tree_edge for step in path)
        tree_value = (
            returns[tree_edge_count]
            if tree_edge_count < len(returns)
            else leaf_value
        )
        tree_leaf.visits += 1
        tree_leaf.value_sum += tree_value
        self.stats["iterations"] += 1

    @staticmethod
    def _write_result(result: dict[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        temporary.replace(path)

    def run(
        self,
        *,
        checkpoint_paths: Mapping[int, Path | str] | None = None,
    ) -> dict[str, object]:
        self.started_at = time.perf_counter()
        self.termination_reason = "iteration_budget"
        checkpoints = {
            int(iteration): Path(path)
            for iteration, path in (checkpoint_paths or {}).items()
        }
        for iteration in range(1, self.config.iterations + 1):
            self.run_iteration(iteration)
            if (
                self.config.progress_interval
                and iteration % self.config.progress_interval == 0
            ):
                print(
                    f"[422-v25] iteration={iteration} "
                    f"classes={len(self.discovered)} nodes={len(self.nodes)} "
                    f"elapsed={time.perf_counter() - self.started_at:.1f}s",
                    flush=True,
                )
            checkpoint_path = checkpoints.get(iteration)
            if checkpoint_path is not None:
                snapshot = self.result()
                snapshot["config"]["iterations"] = iteration
                snapshot["runtime"]["planned_iterations"] = (
                    self.config.iterations
                )
                write_started = time.perf_counter()
                self._write_result(snapshot, checkpoint_path)
                self.started_at += time.perf_counter() - write_started
                print(
                    f"[422-v25] checkpoint={iteration} "
                    f"output={checkpoint_path}",
                    flush=True,
                )
            if (
                self.config.max_wall_seconds is not None
                and time.perf_counter() - self.started_at
                >= self.config.max_wall_seconds
            ):
                self.termination_reason = "wall_time_limit"
                print(
                    f"[422-v25] wall-time limit reached at "
                    f"iteration={iteration}",
                    flush=True,
                )
                break
        return self.result()

    def result(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self.started_at if self.started_at else 0.0
        edge_count = sum(len(node.edges) for node in self.nodes.values())
        return {
            "algorithm": "bell422_independent_embedding_relation_mcts_v25",
            "problem": {
                "scenario": "4-2-2",
                "vertex_count": VERTEX_COUNT,
                "ambient_dimension": 80,
                "target_affine_rank": TARGET_AFFINE_RANK,
                "catalog_class_count": 34_836,
                "relabeling_group_order": 98_304,
            },
            "prior_boundary": {
                "search_reads": [
                    "complete_nodes.csv",
                    "nodes/*/orbit_blocks.csv",
                ],
                "terminal_only_reads": ["global_facet_classes.csv"],
                "never_read": ["nodes/*/facets.csv"],
                "initial_discovered_classes": [],
                "unseen_class_examples_used": False,
            },
            "features": {
                "one_context_sensitive_tree": True,
                "persistent_expansions_per_simulation": 1,
                "multilevel_progressive_widening": True,
                "size_and_action_bucket_quotas": True,
                "directed_partition_relations": True,
                "independent_concrete_embedding_values": True,
                "share_embedding_family_values": False,
                "discovery_epoch_puct": True,
                "adaptive_corrector_replay": True,
                "corrector_counterfactual_credit": True,
                "endpoint_novelty_credit": True,
                "rank78_ridge_pivot": True,
                "nonterminal_leaf_value": 0.0,
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
                "discovered_class_count": len(self.discovered),
                "discovered_class_ids": list(self.discovered),
                "discoveries": self.discovery_events,
                "class_frequencies": dict(self.class_frequencies),
                "unique_canonical_terminals": len(self.support_frequencies),
            },
            "tree": {
                "node_count": len(self.nodes),
                "edge_count": edge_count,
                "root_visits": self.root.visits,
                "root_mean_value": self.root.mean_value,
                "tree_policy_steps": self.stats["tree_policy_steps"],
                "rollout_steps": self.stats["rollout_steps"],
                "depth_limit_hits": self.stats["depth_limit_hits"],
                "recycle_events": self.recycle_events,
                "relation_lanes": list(RELATION_LANES),
                "filler_action_lanes": list(FILLER_ACTION_LANES),
                "corrector_action_lanes": list(CORRECTOR_ACTION_LANES),
                "rollout_policy_cycle": list(ROLLOUT_POLICY_CYCLE),
            },
            "replay": {
                "witness_count": len(self.facet_witnesses),
                "replay_counts": {
                    hashlib.blake2b(
                        repr(key).encode("ascii"), digest_size=6
                    ).hexdigest(): count
                    for key, count in self.facet_replay_counts.items()
                },
                "free_filler_ratio": self._free_filler_ratio(),
            },
            "endpoint_graph": {
                "context_count": len(self.endpoint_context_counts),
                "pair_count": len(self.endpoint_pair_counts),
                "target_count": len(self.endpoint_target_counts),
                "novelty_events": self.endpoint_novelty_events,
            },
            "search_stats": dict(self.stats),
            "rank_oracle": self.rank_oracle.to_dict(),
            "facet_oracle": self.facet_oracle.to_dict(),
            "atlas": self.atlas.to_dict(),
            "filler_corrector": self.actions.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "corrector_credit_events": self.corrector_credit_events,
        }


def run_search(
    config: V25SearchConfig,
    *,
    handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    output_path: Path | str | None = None,
    checkpoint_paths: Mapping[int, Path | str] | None = None,
) -> dict[str, object]:
    engine = Bell422V25Search(config, handoff_root=handoff_root)
    result = engine.run(checkpoint_paths=checkpoint_paths)
    if output_path is not None:
        engine._write_result(result, Path(output_path))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bell 4-2-2 v25 Filler-Corrector MCTS"
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4_222_025)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("ITERATION", "PATH"),
        default=[],
        help="write a full result snapshot at an intermediate iteration",
    )
    parser.add_argument("--max-depth", type=int, default=112)
    parser.add_argument("--min-corrector-rank", type=int, default=60)
    parser.add_argument("--tail-checks", type=int, default=24)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument(
        "--max-wall-hours",
        type=float,
        help="gracefully stop and save results after this many wall-clock hours",
    )
    parser.add_argument("--no-catalog", action="store_true")
    parser.add_argument("--no-canonicalize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = V25SearchConfig(
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.max_depth,
        min_corrector_rank=args.min_corrector_rank,
        tail_checks=args.tail_checks,
        progress_interval=args.progress_interval,
        max_wall_seconds=(
            None
            if args.max_wall_hours is None
            else 3600.0 * args.max_wall_hours
        ),
        use_catalog=not args.no_catalog,
        canonicalize_classes=not args.no_canonicalize,
    )
    output = args.output or (
        DEFAULT_RUNS_DIR / f"bell422_group_game_i{config.iterations}_v25.json"
    )
    checkpoints = {int(iteration): path for iteration, path in args.checkpoint}
    invalid = sorted(
        iteration
        for iteration in checkpoints
        if not 0 < iteration < config.iterations
    )
    if invalid:
        raise SystemExit(
            "checkpoint iterations must be inside the run: "
            + ", ".join(map(str, invalid))
        )
    result = run_search(
        config,
        handoff_root=args.handoff_root,
        output_path=output,
        checkpoint_paths=checkpoints,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "node_count": result["tree"]["node_count"],
                "discovered_class_count": result["coverage"][
                    "discovered_class_count"
                ],
                "discovered_class_ids": result["coverage"][
                    "discovered_class_ids"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
