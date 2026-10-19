from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .model import MacroAction, rank_band


@dataclass
class OnlineStats:
    visits: int = 0
    reward_sum: float = 0.0
    rank_delta_sum: float = 0.0
    exact_hits: int = 0
    new_classes: int = 0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0

    @property
    def mean_rank_delta(self) -> float:
        return self.rank_delta_sum / self.visits if self.visits else 0.0

    def update(
        self,
        *,
        reward: float,
        rank_delta: int,
        exact_hit: bool,
        new_class: bool,
    ) -> None:
        self.visits += 1
        self.reward_sum += float(reward)
        self.rank_delta_sum += float(rank_delta)
        self.exact_hits += int(bool(exact_hit))
        self.new_classes += int(bool(new_class))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mean_reward"] = self.mean_reward
        payload["mean_rank_delta"] = self.mean_rank_delta
        return payload


@dataclass(frozen=True)
class PatternSpec:
    pattern_id: str
    level: int
    structure: str
    blocks: tuple[tuple[int, ...], ...]
    source: str
    parent_root_ids: tuple[str, ...] = ()
    generator_labels: tuple[str, ...] = ()
    family_id: str = ""

    @property
    def statistical_family_id(self) -> str:
        return self.family_id or self.pattern_id

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(len(block) for block in self.blocks)

    @property
    def dominant_size(self) -> int:
        sizes = sorted(self.block_sizes)
        return sizes[len(sizes) // 2] if sizes else 1

    def to_dict(self, *, include_blocks: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "pattern_id": self.pattern_id,
            "level": self.level,
            "structure": self.structure,
            "source": self.source,
            "parent_root_ids": list(self.parent_root_ids),
            "generator_labels": list(self.generator_labels),
            "block_count": len(self.blocks),
            "block_sizes": list(self.block_sizes),
        }
        if self.family_id:
            payload["family_id"] = self.family_id
        if include_blocks:
            payload["blocks"] = [list(block) for block in self.blocks]
        return payload


@dataclass
class ObservedSupportKnowledge:
    canonical_support_word: int
    class_ids: set[int] = field(default_factory=set)
    hit_count: int = 0
    stabilizer_order: int = 1
    stabilizer_pattern_id: str = ""
    stabilizer_pattern_ids: set[str] = field(default_factory=set)
    root_pattern_ids: tuple[str, ...] = ()
    root_subgroup_multiplicities: tuple[tuple[str, int], ...] = ()
    level2_pattern_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_support_word_hex": (
                f"0x{self.canonical_support_word:016x}"
            ),
            "class_ids": sorted(self.class_ids),
            "hit_count": self.hit_count,
            "stabilizer_order": self.stabilizer_order,
            "stabilizer_pattern_id": self.stabilizer_pattern_id,
            "stabilizer_pattern_ids": sorted(self.stabilizer_pattern_ids),
            "embedding_count": len(self.stabilizer_pattern_ids),
            "root_pattern_ids": list(self.root_pattern_ids),
            "root_subgroup_multiplicities": {
                pattern_id: count
                for pattern_id, count in self.root_subgroup_multiplicities
            },
            "level2_pattern_ids": list(self.level2_pattern_ids),
        }


class GroupKnowledgeBase:
    """Online structural memory shared by both roles in one MCTS tree."""

    def __init__(
        self,
        patterns: Iterable[PatternSpec],
        *,
        share_family_values: bool = False,
    ) -> None:
        self.share_family_values = bool(share_family_values)
        self.patterns = {pattern.pattern_id: pattern for pattern in patterns}
        self.active_pattern_ids = set(self.patterns)
        self.pattern_stats: dict[tuple[str, str], OnlineStats] = {}
        self.family_stats: dict[tuple[str, str], OnlineStats] = {}
        self.context_stats: dict[tuple[str, str, str, str, int], OnlineStats] = {}
        self.transition_stats: dict[tuple[str, str, str], OnlineStats] = {}
        self.family_transition_stats: dict[tuple[str, str, str], OnlineStats] = {}
        self.relation_stats: dict[tuple[str, str], OnlineStats] = {}
        self.observed_supports: dict[int, ObservedSupportKnowledge] = {}
        self.total_updates = 0
        self.epoch = 0

    def register_pattern(self, pattern: PatternSpec, *, active: bool = True) -> None:
        self.patterns.setdefault(pattern.pattern_id, pattern)
        if active:
            self.active_pattern_ids.add(pattern.pattern_id)

    def activate(self, pattern_ids: Iterable[str]) -> None:
        for pattern_id in pattern_ids:
            if pattern_id in self.patterns:
                self.active_pattern_ids.add(pattern_id)

    def family_id(self, pattern_id: str) -> str:
        pattern = self.patterns.get(pattern_id)
        return pattern_id if pattern is None else pattern.statistical_family_id

    def family_embeddings(self, family_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                pattern_id
                for pattern_id, pattern in self.patterns.items()
                if pattern.statistical_family_id == family_id
            )
        )

    def _context_key(
        self,
        role: str,
        action: MacroAction,
        rank_before: int,
    ) -> tuple[str, str, str, str, int]:
        return (
            role,
            action.structure,
            rank_band(rank_before),
            action.bucket,
            action.arity,
        )

    def pattern_score(self, role: str, pattern_id: str) -> float:
        stats = self.pattern_stats.get((role, pattern_id))
        embedding_score = self._pattern_stats_score(stats)
        family_id = self.family_id(pattern_id)
        if not self.share_family_values or family_id == pattern_id:
            return embedding_score
        family = self.family_stats.get((role, family_id))
        family_score = self._pattern_stats_score(family)
        if stats is None or stats.visits == 0:
            return family_score + 0.12
        return 0.70 * family_score + 0.30 * embedding_score

    def _pattern_stats_score(self, stats: OnlineStats | None) -> float:
        if stats is None or stats.visits == 0:
            return 1.25
        optimism = math.sqrt(math.log(self.total_updates + 2.0) / stats.visits)
        novelty_rate = stats.new_classes / stats.visits
        return stats.mean_reward + 2.0 * novelty_rate + 0.45 * optimism

    def transition_score(
        self,
        role: str,
        current_pattern_id: str,
        target_pattern_id: str,
    ) -> float:
        stats = self.transition_stats.get((role, current_pattern_id, target_pattern_id))
        embedding_score = self._transition_stats_score(stats)
        current_family = self.family_id(current_pattern_id)
        target_family = self.family_id(target_pattern_id)
        if not self.share_family_values or (
            current_family == current_pattern_id
            and target_family == target_pattern_id
        ):
            return embedding_score
        family = self.family_transition_stats.get(
            (role, current_family, target_family)
        )
        family_score = self._transition_stats_score(family)
        if stats is None or stats.visits == 0:
            return family_score
        return 0.70 * family_score + 0.30 * embedding_score

    def _transition_stats_score(self, stats: OnlineStats | None) -> float:
        if stats is None or stats.visits == 0:
            return 0.35
        optimism = math.sqrt(math.log(self.total_updates + 2.0) / stats.visits)
        return stats.mean_reward + 0.30 * optimism

    def relation_score(self, role: str, relation: str) -> float:
        stats = self.relation_stats.get((role, relation))
        if stats is None or stats.visits == 0:
            return 0.20
        optimism = math.sqrt(math.log(self.total_updates + 2.0) / stats.visits)
        bounded_reward = min(2.0, max(-1.0, stats.mean_reward))
        return bounded_reward + 0.30 * optimism

    def action_prior(
        self,
        role: str,
        action: MacroAction,
        *,
        rank_before: int,
        rank_after: int,
        context_pattern_id: str | None = None,
        coherence: float = 0.0,
        relation: str = "switch",
        learn_relation_value: bool = False,
    ) -> float:
        pattern_value = self.pattern_score(role, action.pattern_id)
        context = self.context_stats.get(self._context_key(role, action, rank_before))
        context_value = 0.0 if context is None else context.mean_reward
        transition_value = self.transition_score(
            role,
            action.pattern_id if context_pattern_id is None else context_pattern_id,
            action.pattern_id,
        )
        if learn_relation_value:
            relation_bonus = {
                "continue": 0.28,
                "refine": 0.18,
                "reembed": 0.20,
                "coarsen": 0.14,
                "neighbor": 0.12,
                "intersection": -0.02,
                "identity": -0.04,
                "switch": -0.04,
            }.get(relation, 0.0)
            learned_relation_value = 0.12 * self.relation_score(role, relation)
        else:
            relation_bonus = {
                "continue": 0.28,
                "neighbor": 0.12,
                "switch": -0.04,
            }.get(relation, 0.0)
            learned_relation_value = 0.0
        rank_delta = rank_after - rank_before
        if role == "filler":
            geometry = 0.12 * max(rank_delta, -2)
            oversize_penalty = 0.02 * max(0, action.block_size - (25 - rank_before + 3))
        else:
            geometry = 0.05 * min(0, rank_delta)
            oversize_penalty = 0.015 * action.block_size
        raw = (
            0.85
            + 0.18 * pattern_value
            + 0.12 * context_value
            + 0.10 * transition_value
            + 0.42 * float(coherence)
            + relation_bonus
            + learned_relation_value
            + geometry
            - oversize_penalty
        )
        return min(3.0, max(0.05, raw))

    def update_action(
        self,
        role: str,
        action: MacroAction,
        *,
        rank_before: int,
        rank_after: int,
        reward: float,
        exact_hit: bool,
        new_class: bool,
        context_pattern_id: str | None = None,
        relation: str = "switch",
        update_relation: bool = False,
    ) -> None:
        self.total_updates += 1
        rank_delta = rank_after - rank_before
        pattern = self.pattern_stats.setdefault((role, action.pattern_id), OnlineStats())
        context = self.context_stats.setdefault(
            self._context_key(role, action, rank_before),
            OnlineStats(),
        )
        transition = self.transition_stats.setdefault(
            (
                role,
                action.pattern_id
                if context_pattern_id is None
                else context_pattern_id,
                action.pattern_id,
            ),
            OnlineStats(),
        )
        for stats in (pattern, context, transition):
            stats.update(
                reward=reward,
                rank_delta=rank_delta,
                exact_hit=exact_hit,
                new_class=new_class,
            )
        target_family = self.family_id(action.pattern_id)
        context_id = (
            action.pattern_id if context_pattern_id is None else context_pattern_id
        )
        context_family = self.family_id(context_id)
        if self.share_family_values and target_family != action.pattern_id:
            family_pattern = self.family_stats.setdefault(
                (role, target_family),
                OnlineStats(),
            )
            family_transition = self.family_transition_stats.setdefault(
                (role, context_family, target_family),
                OnlineStats(),
            )
            for stats in (family_pattern, family_transition):
                stats.update(
                    reward=reward,
                    rank_delta=rank_delta,
                    exact_hit=exact_hit,
                    new_class=new_class,
                )
        if update_relation:
            self.relation_stats.setdefault((role, relation), OnlineStats()).update(
                reward=reward,
                rank_delta=rank_delta,
                exact_hit=exact_hit,
                new_class=new_class,
            )

    def begin_discovery_epoch(self) -> int:
        """Invalidate edge-local novelty values without discarding experience."""
        self.epoch += 1
        return self.epoch

    def record_support(
        self,
        *,
        canonical_word: int,
        class_id: int | None,
        stabilizer_pattern: PatternSpec,
        stabilizer_order: int,
        root_pattern_ids: tuple[str, ...],
        root_subgroup_multiplicities: tuple[tuple[str, int], ...],
        level2_pattern_ids: tuple[str, ...],
    ) -> ObservedSupportKnowledge:
        """Store symmetry learned from a terminal reached by the search itself."""
        self.register_pattern(stabilizer_pattern)
        record = self.observed_supports.setdefault(
            int(canonical_word),
            ObservedSupportKnowledge(canonical_support_word=int(canonical_word)),
        )
        if class_id is not None:
            record.class_ids.add(int(class_id))
        record.hit_count += 1
        record.stabilizer_order = int(stabilizer_order)
        if not record.stabilizer_pattern_id:
            record.stabilizer_pattern_id = stabilizer_pattern.pattern_id
        record.stabilizer_pattern_ids.add(stabilizer_pattern.pattern_id)
        record.root_pattern_ids = tuple(root_pattern_ids)
        record.root_subgroup_multiplicities = tuple(root_subgroup_multiplicities)
        record.level2_pattern_ids = tuple(level2_pattern_ids)
        self.activate((*root_pattern_ids, *level2_pattern_ids))
        return record

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "share_family_values": self.share_family_values,
            "active_pattern_ids": sorted(self.active_pattern_ids),
            "patterns": {
                pattern_id: pattern.to_dict(
                    include_blocks=pattern.source == "observed_stabilizer"
                )
                for pattern_id, pattern in sorted(self.patterns.items())
            },
            "pattern_stats": {
                f"{role}:{pattern_id}": stats.to_dict()
                for (role, pattern_id), stats in sorted(self.pattern_stats.items())
            },
            "family_stats": {
                f"{role}:{family_id}": stats.to_dict()
                for (role, family_id), stats in sorted(self.family_stats.items())
            },
            "context_stats": {
                ":".join((role, structure, rank_value, bucket, str(arity))): stats.to_dict()
                for (role, structure, rank_value, bucket, arity), stats in sorted(
                    self.context_stats.items()
                )
            },
            "transition_stats": {
                ":".join((role, current_pattern, target_pattern)): stats.to_dict()
                for (
                    role,
                    current_pattern,
                    target_pattern,
                ), stats in sorted(self.transition_stats.items())
            },
            "family_transition_stats": {
                ":".join((role, current_family, target_family)): stats.to_dict()
                for (
                    role,
                    current_family,
                    target_family,
                ), stats in sorted(self.family_transition_stats.items())
            },
            "relation_stats": {
                f"{role}:{relation}": stats.to_dict()
                for (role, relation), stats in sorted(self.relation_stats.items())
            },
            "observed_supports": {
                f"0x{word:016x}": record.to_dict()
                for word, record in sorted(self.observed_supports.items())
            },
        }
