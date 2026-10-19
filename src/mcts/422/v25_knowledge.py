from __future__ import annotations

"""Independent concrete-embedding knowledge for the Bell 4-2-2 v25 port."""

import math
from dataclasses import asdict, dataclass

from .atlas import HandoffPatternAtlas
from .model import MacroAction, rank_band
from .problem import TARGET_AFFINE_RANK


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


class V25OnlineKnowledge:
    """Role-separated structural memory without family-level value sharing."""

    def __init__(self, atlas: HandoffPatternAtlas) -> None:
        self.atlas = atlas
        self.pattern_stats: dict[tuple[str, str], OnlineStats] = {}
        self.context_stats: dict[tuple[str, str, str, str, int], OnlineStats] = {}
        self.transition_stats: dict[tuple[str, str, str], OnlineStats] = {}
        self.relation_stats: dict[tuple[str, str], OnlineStats] = {}
        self.total_updates = 0
        self.epoch = 0

    def family_id(self, pattern_id: str) -> str:
        return self.atlas.family_id(pattern_id)

    def _stats_score(self, stats: OnlineStats | None, *, pattern: bool) -> float:
        if stats is None or stats.visits == 0:
            return 1.25 if pattern else 0.35
        optimism = math.sqrt(math.log(self.total_updates + 2.0) / stats.visits)
        novelty_rate = stats.new_classes / stats.visits
        return (
            stats.mean_reward
            + (2.0 * novelty_rate if pattern else 0.0)
            + (0.45 if pattern else 0.30) * optimism
        )

    def pattern_score(self, role: str, pattern_id: str) -> float:
        return self._stats_score(
            self.pattern_stats.get((role, pattern_id)),
            pattern=True,
        )

    def transition_score(
        self,
        role: str,
        current_pattern_id: str,
        target_pattern_id: str,
    ) -> float:
        return self._stats_score(
            self.transition_stats.get(
                (role, current_pattern_id, target_pattern_id)
            ),
            pattern=False,
        )

    def relation_score(self, role: str, relation: str) -> float:
        stats = self.relation_stats.get((role, relation))
        if stats is None or stats.visits == 0:
            return 0.20
        optimism = math.sqrt(math.log(self.total_updates + 2.0) / stats.visits)
        bounded = min(2.0, max(-1.0, stats.mean_reward))
        return bounded + 0.30 * optimism

    @staticmethod
    def _context_key(
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

    def action_prior(
        self,
        role: str,
        action: MacroAction,
        *,
        rank_before: int,
        rank_after: int,
        context_pattern_id: str,
        coherence: float,
        relation: str,
    ) -> float:
        context = self.context_stats.get(
            self._context_key(role, action, rank_before)
        )
        context_value = 0.0 if context is None else context.mean_reward
        relation_bonus = {
            "continue": 0.28,
            "reembed": 0.20,
            "refine": 0.18,
            "coarsen": 0.14,
            "intersection": -0.02,
            "identity": -0.04,
        }.get(relation, 0.0)
        rank_delta = rank_after - rank_before
        if role == "filler":
            geometry = 0.12 * max(rank_delta, -2)
            oversize = 0.02 * max(
                0, action.block_size - (TARGET_AFFINE_RANK - rank_before + 3)
            )
        else:
            geometry = 0.05 * min(0, rank_delta)
            oversize = 0.015 * action.block_size
        raw = (
            0.85
            + 0.18 * self.pattern_score(role, action.pattern_id)
            + 0.12 * context_value
            + 0.10
            * self.transition_score(role, context_pattern_id, action.pattern_id)
            + 0.42 * float(coherence)
            + relation_bonus
            + 0.12 * self.relation_score(role, relation)
            + geometry
            - oversize
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
        context_pattern_id: str,
        relation: str,
    ) -> None:
        self.total_updates += 1
        rank_delta = rank_after - rank_before
        targets = (
            self.pattern_stats.setdefault(
                (role, action.pattern_id), OnlineStats()
            ),
            self.context_stats.setdefault(
                self._context_key(role, action, rank_before), OnlineStats()
            ),
            self.transition_stats.setdefault(
                (role, context_pattern_id, action.pattern_id), OnlineStats()
            ),
            self.relation_stats.setdefault((role, relation), OnlineStats()),
        )
        for stats in targets:
            stats.update(
                reward=reward,
                rank_delta=rank_delta,
                exact_hit=exact_hit,
                new_class=new_class,
            )

    def begin_discovery_epoch(self) -> int:
        self.epoch += 1
        return self.epoch

    @staticmethod
    def _table_payload(table: dict[tuple, OnlineStats]) -> dict[str, object]:
        return {
            ":".join(map(str, key)): {
                **asdict(stats),
                "mean_reward": stats.mean_reward,
                "mean_rank_delta": stats.mean_rank_delta,
            }
            for key, stats in sorted(table.items())
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "total_updates": self.total_updates,
            "share_embedding_family_values": False,
            "pattern_stats": self._table_payload(self.pattern_stats),
            "context_stats": self._table_payload(self.context_stats),
            "transition_stats": self._table_payload(self.transition_stats),
            "relation_stats": self._table_payload(self.relation_stats),
        }
