from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT, HandoffPatternAtlas, IDENTITY_PATTERN_ID
from .filler_corrector import CORRECTOR_LANES, FILLER_LANES, FillerCorrector
from .geometry import FacetOracle, FacetResult, RankOracle
from .knowledge import OnlineKnowledge
from .model import EdgeStats, GameNode, MacroAction, SearchState, rank_band
from .problem import TARGET_AFFINE_RANK
from mcts.symmetry_game.patterns import PatternRelationIndex


DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
ROLLOUT_POLICIES = (
    "coherent",
    "coherent",
    "neighbor",
    "coherent",
    "mixed",
    "coherent",
    "neighbor",
    "broad",
)
SYMMETRY_RELATION_LANES = (
    "continue",
    "refine",
    "continue",
    "intersection",
    "refine",
    "identity",
    "intersection",
)


@dataclass(frozen=True)
class SearchConfig:
    iterations: int = 300
    seed: int = 422_2026
    max_depth: int = 112
    rollout_depth: int = 96
    max_corrections: int = 5
    min_corrector_rank: int = 60
    exploration: float = 1.35
    discount: float = 0.985
    pattern_widening_k0: int = 3
    pattern_widening_alpha: float = 1.15
    pattern_widening_beta: float = 0.45
    action_widening_k0: int = 4
    action_widening_alpha: float = 2.0
    action_widening_beta: float = 0.50
    max_children_per_node: int = 160
    max_corrector_children_per_node: int = 112
    rank_checks_per_widen: int = 10
    filler_candidate_limit: int = 72
    corrector_candidate_limit: int = 64
    rollout_action_candidates: int = 12
    rollout_random_probability: float = 0.16
    tail_checks: int = 24
    stabilizer_probes: int = 192
    new_class_reward: float = 10.0
    repeated_class_reward: float = 0.10
    repeated_frequency_penalty: float = 0.025
    invalid_terminal_reward: float = -1.25
    dead_end_reward: float = -0.60
    rank_gain_reward: float = 0.035
    correction_cost: float = 0.008
    canonicalize_classes: bool = True
    use_catalog: bool = True
    progress_interval: int = 25
    geometric_transpositions: bool = False
    hierarchical_pattern_policy: bool = False
    rollout_pattern_candidates: int = 8
    ridge_pivot: bool = False
    context_free_action_cache: bool = False
    structural_novelty_reward: float = 0.0
    repeated_basin_penalty_cap: float = 0.0

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.max_depth <= 0 or self.rollout_depth <= 0:
            raise ValueError("depth limits must be positive")
        if not 0 <= self.min_corrector_rank < TARGET_AFFINE_RANK:
            raise ValueError("min_corrector_rank must be in [0, 78]")
        if self.max_corrections < 0:
            raise ValueError("max_corrections must be nonnegative")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if min(self.rank_checks_per_widen, self.tail_checks) <= 0:
            raise ValueError("rank and tail check budgets must be positive")
        if self.rollout_pattern_candidates <= 0:
            raise ValueError("rollout_pattern_candidates must be positive")
        if min(self.structural_novelty_reward, self.repeated_basin_penalty_cap) < 0:
            raise ValueError("novelty reward and basin penalty must be nonnegative")


@dataclass(frozen=True)
class Transition:
    state: SearchState
    rank: int
    facet: FacetResult | None


@dataclass
class PathStep:
    node: GameNode
    edge: EdgeStats
    role: str
    context_pattern_id: str


class Bell422Search:
    """One persistent, alternating Filler-Corrector MCTS tree."""

    def __init__(
        self,
        config: SearchConfig,
        *,
        handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
        atlas: HandoffPatternAtlas | None = None,
        rank_oracle: RankOracle | None = None,
        facet_oracle: FacetOracle | None = None,
    ):
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
        self.knowledge = OnlineKnowledge()
        self.actions = FillerCorrector(
            self.atlas,
            self.rank_oracle,
            self.facet_oracle,
            seed=config.seed,
            filler_candidate_limit=config.filler_candidate_limit,
            corrector_candidate_limit=config.corrector_candidate_limit,
            tail_checks=config.tail_checks,
            stabilizer_probes=config.stabilizer_probes,
            context_free_cache_keys=config.context_free_action_cache,
            ridge_pivot=config.ridge_pivot,
        )
        self.rng = random.Random(config.seed)
        self.pattern_order = self.atlas.fair_order(config.seed)
        root_state = SearchState(0, "filler", 0, IDENTITY_PATTERN_ID)
        self.root = GameNode(root_state, -1)
        self.nodes = {self._state_key(root_state): self.root}
        self.pattern_relations = PatternRelationIndex(256)
        self.pattern_relation_counts: Counter[str] = Counter()
        self.structural_cells: set[tuple] = set()
        self.discovered: dict[str, dict[str, object]] = {}
        self.discovery_events: list[dict[str, object]] = []
        self.terminal_events: list[dict[str, object]] = []
        self.stats: Counter[str] = Counter()
        self.started_at = 0.0
        self.current_iteration = 0

    def _state_key(self, state: SearchState) -> tuple:
        return state.geometry_key if self.config.geometric_transpositions else state.key

    def _ensure_pattern_relation(self, pattern_id: str) -> None:
        if not self.pattern_relations.registered(pattern_id):
            self.pattern_relations.register(pattern_id, self.atlas.blocks(pattern_id))

    def _pattern_relation(self, node: GameNode, pattern_id: str) -> str:
        if pattern_id == IDENTITY_PATTERN_ID:
            return "continue" if pattern_id in node.context_pattern_ids else "identity"
        self._ensure_pattern_relation(pattern_id)
        relations = []
        for context in node.context_pattern_ids:
            if context not in self.atlas.by_id:
                continue
            self._ensure_pattern_relation(context)
            relations.append(self.pattern_relations.relation(context, pattern_id))
        for relation in ("continue", "refine", "intersection"):
            if relation in relations:
                return relation
        return "intersection"

    def _pattern_limit(self, node: GameNode) -> int:
        cfg = self.config
        return max(1, int(cfg.pattern_widening_k0 + cfg.pattern_widening_alpha * (node.visits + 1) ** cfg.pattern_widening_beta))

    def _action_limit(self, node: GameNode) -> int:
        cfg = self.config
        cap = cfg.max_corrector_children_per_node if node.state.role == "corrector" else cfg.max_children_per_node
        value = int(cfg.action_widening_k0 + cfg.action_widening_alpha * (node.visits + 1) ** cfg.action_widening_beta)
        return min(cap, max(1, value))

    def _candidate_pattern_ids(self, node: GameNode) -> tuple[str, ...]:
        preferred: list[str] = sorted(node.context_pattern_ids)
        if node.state.role == "corrector":
            preferred.extend(self.actions.facet_patterns.get(node.state.support_word, ()))
        for context in sorted(node.context_pattern_ids):
            preferred.extend(self.atlas.neighbors(context, limit=12))
        preferred.extend(self.knowledge.observed_pattern_ids)
        start = node.pattern_cursor % len(self.pattern_order)
        preferred.extend(self.pattern_order[(start + offset) % len(self.pattern_order)] for offset in range(32))
        return tuple(pattern_id for pattern_id in dict.fromkeys(preferred) if pattern_id in self.atlas.by_id)

    def _choose_new_pattern(self, node: GameNode) -> str | None:
        available = [
            pattern_id
            for pattern_id in self._candidate_pattern_ids(node)
            if pattern_id not in node.expanded_patterns and pattern_id not in node.exhausted_patterns
        ]
        if not available:
            return None
        role = node.state.role
        window = available[:16]
        if self.config.hierarchical_pattern_policy:
            lane = SYMMETRY_RELATION_LANES[
                node.pattern_cursor % len(SYMMETRY_RELATION_LANES)
            ]
            matched = [pattern_id for pattern_id in window if self._pattern_relation(node, pattern_id) == lane]
            if matched:
                window = matched
                self.stats[f"pattern_lane_{lane}"] += 1
            else:
                self.stats[f"pattern_lane_{lane}_fallback"] += 1
        selected = max(
            enumerate(window),
            key=lambda item: (
                self.knowledge.pattern_score(role, item[1]),
                -item[0],
            ),
        )[1]
        node.pattern_cursor += 1
        node.expanded_patterns.add(selected)
        self.stats["patterns_expanded"] += 1
        return selected

    def _action_pool(self, node: GameNode, lane: str) -> list[MacroAction]:
        pool: list[MacroAction] = []
        for pattern_id in tuple(node.expanded_patterns):
            if node.state.role == "filler":
                pool.extend(self.actions.filler_actions(node.state, pattern_id, lane=lane))
            else:
                pool.extend(self.actions.corrector_actions(node.state, pattern_id, lane=lane))
                if lane == "rewrite":
                    pool.extend(self.actions.cross_pattern_rewrites(node.state, pattern_id, limit=16))
        return [
            action for action in pool
            if action.action_id not in node.edges and action.action_id not in node.rejected_action_ids
        ]

    def _approximate_action_score(self, node: GameNode, action: MacroAction) -> float:
        if node.state.role == "filler":
            remaining = TARGET_AFFINE_RANK - node.rank
            cardinality = action.add_word.bit_count()
            closeness = -abs(cardinality - max(1, min(cardinality, remaining)))
            relation = self._pattern_relation(node, action.pattern_id)
            continuity = {
                "continue": 2.0,
                "refine": 1.1,
                "intersection": 0.25,
                "identity": 0.0,
            }[relation]
            tail_bonus = 2.0 if node.rank >= TARGET_AFFINE_RANK - 3 and cardinality <= 4 else 0.0
            return continuity + tail_bonus + 0.08 * cardinality + 0.05 * closeness
        target_rank = TARGET_AFFINE_RANK - 1
        size = action.remove_word.bit_count()
        rewrite = 0.8 if action.kind == "rewrite" else 0.0
        online = 0.8 if self.atlas.pattern(action.pattern_id).source == "online_stabilizer" else 0.0
        return rewrite + online - 0.025 * size - 0.02 * abs(node.rank - target_rank)

    def _transition(self, state: SearchState, action: MacroAction) -> Transition | None:
        word, rank = self.actions.apply(state, action)
        if word == state.support_word or rank > TARGET_AFFINE_RANK:
            return None
        facet: FacetResult | None = None
        corrections = state.corrections_used + int(state.role == "corrector")
        if rank == TARGET_AFFINE_RANK:
            facet = self.facet_oracle.validate(word)
            if facet.valid:
                word = facet.support_word
                next_state = SearchState(word, "corrector", corrections, action.pattern_id, word)
                return Transition(next_state, TARGET_AFFINE_RANK, facet)
            return Transition(SearchState(word, "done", corrections, action.pattern_id), rank, facet)
        if state.role == "corrector" and rank < self.config.min_corrector_rank:
            return None
        return Transition(
            SearchState(word, "filler", corrections, action.pattern_id, state.source_facet_word),
            rank,
            None,
        )

    def _widen_once(self, node: GameNode) -> tuple[EdgeStats, bool] | None:
        if len(node.expanded_patterns) < self._pattern_limit(node):
            self._choose_new_pattern(node)
        if not node.expanded_patterns:
            return None
        lanes = FILLER_LANES if node.state.role == "filler" else CORRECTOR_LANES
        lane = lanes[node.bucket_cursor % len(lanes)]
        node.bucket_cursor += 1
        pool = self._action_pool(node, lane)
        if not pool:
            pattern_id = self._choose_new_pattern(node)
            if pattern_id is not None:
                pool = self._action_pool(node, lane)
        pool.sort(key=lambda action: (self._approximate_action_score(node, action), action.action_id), reverse=True)
        for action in pool[: self.config.rank_checks_per_widen]:
            transition = self._transition(node.state, action)
            if transition is None:
                node.rejected_action_ids.add(action.action_id)
                self.stats["actions_rejected"] += 1
                continue
            child_key = self._state_key(transition.state)
            if child_key in node.child_keys:
                node.rejected_action_ids.add(action.action_id)
                self.stats["duplicate_children"] += 1
                continue
            prior = self.knowledge.action_prior(
                node.state.role,
                action,
                rank_before=node.rank,
                rank_after=transition.rank,
                context_pattern_id=node.state.context_pattern_id,
            )
            edge = EdgeStats(action, child_key, prior, node.rank, transition.rank)
            node.edges[action.action_id] = edge
            node.child_keys.add(child_key)
            child = self.nodes.get(child_key)
            is_new = child is None
            if child is None:
                child = GameNode(transition.state, transition.rank)
                self.nodes[child_key] = child
            else:
                before = len(child.context_pattern_ids)
                child.context_pattern_ids.add(transition.state.context_pattern_id)
                if len(child.context_pattern_ids) > before:
                    self.stats["merged_pattern_contexts"] += 1
            self.stats["edges_created"] += 1
            return edge, is_new
        return None

    def _select_edge(self, node: GameNode) -> EdgeStats | None:
        if not node.edges:
            return None
        epoch = self.knowledge.epoch
        root = math.sqrt(node.visits + 1.0)

        def score(edge: EdgeStats) -> tuple[float, str]:
            recent = edge.epoch_mean(epoch)
            q = edge.mean_value * 0.25 if recent is None else recent
            u = self.config.exploration * edge.prior * root / (1.0 + edge.visits)
            return q + u, edge.action.action_id

        return max(node.edges.values(), key=score)

    def _record_facet(
        self,
        facet: FacetResult,
        *,
        depth: int,
        source: str,
        context_pattern_id: str,
    ) -> tuple[float, bool]:
        assert facet.valid and facet.class_id is not None
        class_id = facet.class_id
        frequency = self.knowledge.class_frequencies[class_id]
        is_new = class_id not in self.discovered
        self.knowledge.class_frequencies[class_id] += 1
        patterns = self.actions.observe_facet(facet.support_word, context_pattern_id)
        for pattern_id in patterns:
            if self.atlas.pattern(pattern_id).source == "online_stabilizer":
                self.knowledge.observe_pattern(facet.support_word, pattern_id)
        if is_new:
            self.knowledge.begin_discovery_epoch()
            event = {
                "class_id": class_id,
                "catalog_index": facet.catalog_index,
                "iteration": self.current_iteration,
                "depth": depth,
                "elapsed_seconds": time.perf_counter() - self.started_at,
                "source": source,
                "context_pattern_id": context_pattern_id,
                "support_word_hex": f"0x{facet.support_word:064x}",
                "support_size": facet.support_word.bit_count(),
                "h_row": list(facet.h_row or ()),
                "canonical_h_row": list(facet.canonical_h_row or ()),
                "online_pattern_ids": list(patterns),
            }
            self.discovered[class_id] = event
            self.discovery_events.append(event)
            self.stats["new_classes"] += 1
            return self.config.new_class_reward, True
        reward = max(
            -self.config.repeated_basin_penalty_cap,
            self.config.repeated_class_reward - self.config.repeated_frequency_penalty * math.log1p(frequency),
        )
        self.stats["repeated_classes"] += 1
        return reward, False

    def _rollout_patterns(self, state: SearchState, policy: str, rng: random.Random) -> tuple[str, ...]:
        context = state.context_pattern_id
        if policy == "coherent":
            result = (context,)
            return result[: self.config.rollout_pattern_candidates]
        if policy == "neighbor":
            result = tuple(self.atlas.neighbors(context, limit=8)) or (context,)
            return result[: self.config.rollout_pattern_candidates]
        if policy == "broad":
            candidates = self.atlas.neighbors(context, limit=12)
            result = tuple(sorted(candidates, key=lambda p: self.atlas.pattern(p).block_count)[:6]) or (context,)
            return result[: self.config.rollout_pattern_candidates]
        candidates = list(self.pattern_order)
        return tuple(rng.sample(candidates, min(self.config.rollout_pattern_candidates, len(candidates))))

    def _structural_reward(
        self,
        state: SearchState,
        rank: int,
        action: MacroAction,
        moved: Transition,
    ) -> float:
        if self.config.structural_novelty_reward <= 0:
            return 0.0
        cell = (
            rank_band(moved.rank),
            state.role,
            action.structure,
            action.bucket,
            action.arity,
            self._support_scale(moved.state.support_word),
        )
        if cell in self.structural_cells:
            return 0.0
        self.structural_cells.add(cell)
        self.stats["new_structural_cells"] += 1
        return self.config.structural_novelty_reward

    @staticmethod
    def _support_scale(word: int) -> str:
        size = int(word).bit_count()
        if size < 64:
            return "s00_63"
        if size < 128:
            return "s64_127"
        if size < 192:
            return "s128_191"
        return "s192_256"

    def _rollout(self, state: SearchState, rank: int, *, start_depth: int) -> tuple[float, bool, bool]:
        policy = ROLLOUT_POLICIES[(self.current_iteration + start_depth) % len(ROLLOUT_POLICIES)]
        policy_seed = self.config.seed ^ (self.current_iteration * 1_000_003) ^ (start_depth * 97) ^ sum(map(ord, policy))
        rng = random.Random(policy_seed)
        reward = 0.0
        discovered = False
        exact = False
        max_rank = rank
        for offset in range(self.config.rollout_depth):
            depth = start_depth + offset
            if state.role == "done":
                reward += self.config.invalid_terminal_reward
                self.stats["rollout_invalid_terminals"] += 1
                break
            if state.role == "corrector":
                facet = self.facet_oracle.validate(state.support_word)
                if facet.valid:
                    value, new = self._record_facet(
                        facet,
                        depth=depth,
                        source="rollout",
                        context_pattern_id=state.context_pattern_id,
                    )
                    reward += value
                    discovered |= new
                    exact = True
                if state.corrections_used >= self.config.max_corrections:
                    break
                pattern_ids = self.actions.facet_patterns.get(
                    state.support_word, (state.context_pattern_id,)
                )
                lane = CORRECTOR_LANES[(self.current_iteration + offset) % len(CORRECTOR_LANES)]
                candidates: list[MacroAction] = []
                for pattern_id in pattern_ids[:6]:
                    candidates.extend(self.actions.corrector_actions(state, pattern_id, lane=lane))
                    if lane == "rewrite":
                        candidates.extend(self.actions.cross_pattern_rewrites(state, pattern_id, limit=8))
                if not candidates:
                    break
                rng.shuffle(candidates)
                candidates.sort(key=lambda action: self._approximate_rollout_score(state, rank, action), reverse=True)
                transition = self._first_valid_transition(state, candidates[: self.config.rollout_action_candidates])
                if transition is None:
                    break
                action, moved = transition
                reward -= self.config.correction_cost * action.remove_word.bit_count()
                reward += self._structural_reward(state, rank, action, moved)
                state, rank = moved.state, moved.rank
                continue

            pattern_ids = self._rollout_patterns(state, policy, rng)
            if rank >= TARGET_AFFINE_RANK - 2:
                completion = self.actions.tail_complete(state, pattern_ids, max_checks=self.config.tail_checks)
                if completion is not None:
                    state = SearchState(
                        completion.facet.support_word,
                        "corrector",
                        state.corrections_used,
                        completion.action.pattern_id,
                        completion.facet.support_word,
                    )
                    rank = TARGET_AFFINE_RANK
                    value, new = self._record_facet(
                        completion.facet,
                        depth=depth,
                        source="rank_tail_enumeration",
                        context_pattern_id=completion.action.pattern_id,
                    )
                    reward += value
                    discovered |= new
                    exact = True
                    continue
            lane = "broad" if policy == "broad" else FILLER_LANES[(self.current_iteration + offset) % len(FILLER_LANES)]
            candidates = []
            for pattern_id in pattern_ids[:8]:
                candidates.extend(self.actions.filler_actions(state, pattern_id, lane=lane))
            if not candidates:
                reward += self.config.dead_end_reward
                break
            rng.shuffle(candidates)
            candidates.sort(key=lambda action: self._approximate_rollout_score(state, rank, action), reverse=True)
            sample = candidates[: self.config.rollout_action_candidates]
            if rng.random() < self.config.rollout_random_probability:
                rng.shuffle(sample)
            transition = self._first_valid_transition(state, sample)
            if transition is None:
                reward += self.config.dead_end_reward
                break
            action, moved = transition
            reward += self.config.rank_gain_reward * max(0, moved.rank - rank)
            reward += self._structural_reward(state, rank, action, moved)
            max_rank = max(max_rank, moved.rank)
            state, rank = moved.state, moved.rank
            if moved.facet is not None:
                if moved.facet.valid:
                    value, new = self._record_facet(
                        moved.facet,
                        depth=depth,
                        source="rollout_add",
                        context_pattern_id=action.pattern_id,
                    )
                    reward += value
                    discovered |= new
                    exact = True
                else:
                    reward += self.config.invalid_terminal_reward
                    break
        self.stats[f"rollout_policy_{policy}"] += 1
        self.stats[f"rollout_max_rank_{max_rank}"] += 1
        return reward, exact, discovered

    def _approximate_rollout_score(self, state: SearchState, rank: int, action: MacroAction) -> float:
        if state.role == "corrector":
            return (1.0 if action.kind == "rewrite" else 0.0) - 0.02 * action.remove_word.bit_count()
        continuity = 1.5 if action.pattern_id == state.context_pattern_id else 0.0
        target_size = max(1, TARGET_AFFINE_RANK - rank)
        size = action.add_word.bit_count()
        return continuity + 0.10 * min(size, target_size) - 0.05 * max(0, size - target_size)

    def _first_valid_transition(
        self, state: SearchState, candidates: Sequence[MacroAction]
    ) -> tuple[MacroAction, Transition] | None:
        for action in candidates:
            transition = self._transition(state, action)
            if transition is not None:
                return action, transition
        return None

    def _backpropagate(
        self,
        visited: list[GameNode],
        path: list[PathStep],
        reward: float,
        *,
        exact: bool,
        discovery: bool,
    ) -> None:
        value = float(reward)
        for node in reversed(visited):
            node.visits += 1
            node.value_sum += value
            value *= self.config.discount
        value = float(reward)
        for step in reversed(path):
            step.edge.update(value, self.knowledge.epoch)
            if discovery:
                step.edge.discovery_hits += 1
            self.knowledge.update(
                step.role,
                step.edge.action,
                rank_before=step.edge.rank_before,
                rank_after=step.edge.rank_after,
                reward=value,
                exact=exact,
                discovery=discovery,
                context_pattern_id=step.context_pattern_id,
            )
            value *= self.config.discount

    def run(self) -> dict[str, object]:
        self.started_at = time.perf_counter()
        for iteration in range(1, self.config.iterations + 1):
            self.current_iteration = iteration
            node = self.root
            visited = [node]
            path: list[PathStep] = []
            reward = 0.0
            exact = False
            discovery = False
            depth = 0
            while depth < self.config.max_depth:
                state = node.state
                if state.role == "done":
                    reward += self.config.invalid_terminal_reward
                    break
                if state.role == "corrector":
                    facet = self.facet_oracle.validate(state.support_word)
                    if facet.valid:
                        value, new = self._record_facet(
                            facet,
                            depth=depth,
                            source="tree",
                            context_pattern_id=state.context_pattern_id,
                        )
                        reward += value
                        exact = True
                        discovery |= new
                    if state.corrections_used >= self.config.max_corrections:
                        break
                widened: tuple[EdgeStats, bool] | None = None
                if len(node.edges) < self._action_limit(node):
                    widened = self._widen_once(node)
                if widened is None:
                    edge = self._select_edge(node)
                    is_new = False
                else:
                    edge, is_new = widened
                if edge is None:
                    reward += self.config.dead_end_reward
                    break
                path.append(PathStep(node, edge, state.role, state.context_pattern_id))
                node = self.nodes[edge.child_key]
                visited.append(node)
                depth += 1
                if is_new:
                    rollout_reward, rollout_exact, rollout_discovery = self._rollout(
                        node.state, node.rank, start_depth=depth
                    )
                    reward += rollout_reward
                    exact |= rollout_exact
                    discovery |= rollout_discovery
                    break
            self._backpropagate(visited, path, reward, exact=exact, discovery=discovery)
            self.stats["iterations"] += 1
            if self.config.progress_interval and iteration % self.config.progress_interval == 0:
                elapsed = time.perf_counter() - self.started_at
                print(
                    f"[422] iteration={iteration} classes={len(self.discovered)} "
                    f"nodes={len(self.nodes)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        return self.result()

    def result(self) -> dict[str, object]:
        elapsed = time.perf_counter() - self.started_at if self.started_at else 0.0
        edge_count = sum(len(node.edges) for node in self.nodes.values())
        root_edges = sorted(
            self.root.edges.values(), key=lambda edge: (edge.visits, edge.mean_value), reverse=True
        )[:24]
        return {
            "algorithm": (
                "bell422_scalable_symmetry_filler_corrector_v2"
                if self.config.hierarchical_pattern_policy or self.config.geometric_transpositions
                else "bell422_single_tree_symmetry_filler_corrector_v1"
            ),
            "problem": {
                "scenario": "4-2-2",
                "vertex_count": 256,
                "ambient_dimension": 80,
                "target_affine_rank": TARGET_AFFINE_RANK,
                "relabeling_group_order": 98_304,
            },
            "prior_boundary": {
                "search_reads": ["complete_nodes.csv", "nodes/*/orbit_blocks.csv"],
                "terminal_only_reads": ["global_facet_classes.csv"],
                "never_read": ["nodes/*/facets.csv"],
                "initial_discovered_classes": [],
            },
            "config": asdict(self.config),
            "runtime": {
                "elapsed_seconds": elapsed,
                "iterations_completed": self.current_iteration,
                "logical_iterations": self.stats["iterations"],
            },
            "coverage": {
                "discovered_class_count": len(self.discovered),
                "discovered_class_ids": list(self.discovered),
                "discoveries": self.discovery_events,
            },
            "tree": {
                "node_count": len(self.nodes),
                "edge_count": edge_count,
                "root_visits": self.root.visits,
                "root_mean_value": self.root.mean_value,
                "merged_pattern_context_count": sum(
                    max(0, len(node.context_pattern_ids) - 1)
                    for node in self.nodes.values()
                ),
                "root_edges": [
                    {
                        "action": edge.action.to_dict(),
                        "visits": edge.visits,
                        "mean_value": edge.mean_value,
                        "prior": edge.prior,
                        "rank_after": edge.rank_after,
                        "discovery_hits": edge.discovery_hits,
                    }
                    for edge in root_edges
                ],
            },
            "search_stats": dict(self.stats),
            "symmetry_policy": {
                "geometric_transpositions": self.config.geometric_transpositions,
                "hierarchical_pattern_policy": self.config.hierarchical_pattern_policy,
                "relation_lanes": list(SYMMETRY_RELATION_LANES),
                "registered_partitions": self.pattern_relations.pattern_count,
                "structural_cells": len(self.structural_cells),
            },
            "rank_oracle": self.rank_oracle.to_dict(),
            "facet_oracle": self.facet_oracle.to_dict(),
            "atlas": self.atlas.to_dict(),
            "filler_corrector": self.actions.to_dict(),
            "knowledge": self.knowledge.to_dict(),
        }


def run_search(
    config: SearchConfig,
    *,
    handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    output_path: Path | str | None = None,
) -> dict[str, object]:
    engine = Bell422Search(config, handoff_root=handoff_root)
    result = engine.run()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bell 4-2-2 single-tree Filler-Corrector MCTS")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=422_2026)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-depth", type=int, default=112)
    parser.add_argument("--rollout-depth", type=int, default=96)
    parser.add_argument("--max-corrections", type=int, default=5)
    parser.add_argument("--min-corrector-rank", type=int, default=60)
    parser.add_argument("--tail-checks", type=int, default=24)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--no-catalog", action="store_true")
    parser.add_argument("--no-canonicalize", action="store_true")
    parser.add_argument("--geometric-transpositions", action="store_true")
    parser.add_argument("--hierarchical-pattern-policy", action="store_true")
    parser.add_argument("--rollout-pattern-candidates", type=int, default=8)
    parser.add_argument("--ridge-pivot", action="store_true")
    parser.add_argument("--context-free-action-cache", action="store_true")
    parser.add_argument("--filler-candidate-limit", type=int, default=72)
    parser.add_argument("--corrector-candidate-limit", type=int, default=64)
    parser.add_argument("--rollout-action-candidates", type=int, default=12)
    parser.add_argument("--structural-novelty-reward", type=float, default=0.0)
    parser.add_argument("--repeated-basin-penalty-cap", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SearchConfig(
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.max_depth,
        rollout_depth=args.rollout_depth,
        max_corrections=args.max_corrections,
        min_corrector_rank=args.min_corrector_rank,
        tail_checks=args.tail_checks,
        progress_interval=args.progress_interval,
        use_catalog=not args.no_catalog,
        canonicalize_classes=not args.no_canonicalize,
        geometric_transpositions=args.geometric_transpositions,
        hierarchical_pattern_policy=args.hierarchical_pattern_policy,
        rollout_pattern_candidates=args.rollout_pattern_candidates,
        ridge_pivot=args.ridge_pivot,
        context_free_action_cache=args.context_free_action_cache,
        filler_candidate_limit=args.filler_candidate_limit,
        corrector_candidate_limit=args.corrector_candidate_limit,
        rollout_action_candidates=args.rollout_action_candidates,
        structural_novelty_reward=args.structural_novelty_reward,
        repeated_basin_penalty_cap=args.repeated_basin_penalty_cap,
    )
    output = args.output or DEFAULT_RUNS_DIR / f"bell422_group_game_i{config.iterations}.json"
    result = run_search(config, handoff_root=args.handoff_root, output_path=output)
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "node_count": result["tree"]["node_count"],
                "discovered_class_count": result["coverage"]["discovered_class_count"],
                "discovered_class_ids": result["coverage"]["discovered_class_ids"],
            },
            indent=2,
        )
    )
    return 0
