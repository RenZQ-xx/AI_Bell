from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .model import MacroAction, rank_band


@dataclass
class OnlineStats:
    visits: int = 0
    reward_sum: float = 0.0
    rank_delta_sum: float = 0.0
    exact_hits: int = 0
    discoveries: int = 0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0

    def update(self, reward: float, rank_delta: int, exact: bool, discovery: bool) -> None:
        self.visits += 1
        self.reward_sum += float(reward)
        self.rank_delta_sum += int(rank_delta)
        self.exact_hits += int(exact)
        self.discoveries += int(discovery)


class OnlineKnowledge:
    """Class-blind structural experience shared by Filler and Corrector."""

    def __init__(self) -> None:
        self.pattern_stats: dict[tuple[str, str], OnlineStats] = {}
        self.context_stats: dict[tuple[str, str, str, str], OnlineStats] = {}
        self.transition_stats: dict[tuple[str, str, str], OnlineStats] = {}
        self.observed_pattern_ids: set[str] = set()
        self.observed_supports: dict[int, str] = {}
        self.total_updates = 0
        self.epoch = 0
        self.class_frequencies: Counter[str] = Counter()

    def pattern_score(self, role: str, pattern_id: str) -> float:
        stats = self.pattern_stats.get((role, pattern_id))
        if stats is None or not stats.visits:
            return 1.4
        optimism = math.sqrt(math.log(self.total_updates + 2.0) / stats.visits)
        novelty = stats.discoveries / stats.visits
        return stats.mean_reward + 2.5 * novelty + 0.4 * optimism

    def action_prior(
        self,
        role: str,
        action: MacroAction,
        *,
        rank_before: int,
        rank_after: int,
        context_pattern_id: str,
    ) -> float:
        pattern = self.pattern_score(role, action.pattern_id)
        context_key = (role, rank_band(rank_before), action.bucket, action.kind)
        context = self.context_stats.get(context_key)
        context_value = context.mean_reward if context is not None else 0.0
        transition = self.transition_stats.get((role, context_pattern_id, action.pattern_id))
        transition_value = transition.mean_reward if transition is not None else 0.0
        delta = rank_after - rank_before
        geometry = 0.09 * max(-2, delta) if role == "filler" else -0.015 * action.block_size
        continuity = 0.25 if action.pattern_id == context_pattern_id else 0.0
        raw = 0.8 + 0.18 * pattern + 0.08 * context_value + 0.08 * transition_value + geometry + continuity
        return min(3.0, max(0.03, raw))

    def update(
        self,
        role: str,
        action: MacroAction,
        *,
        rank_before: int,
        rank_after: int,
        reward: float,
        exact: bool,
        discovery: bool,
        context_pattern_id: str,
    ) -> None:
        self.total_updates += 1
        keys = (
            (self.pattern_stats, (role, action.pattern_id)),
            (self.context_stats, (role, rank_band(rank_before), action.bucket, action.kind)),
            (self.transition_stats, (role, context_pattern_id, action.pattern_id)),
        )
        for table, key in keys:
            table.setdefault(key, OnlineStats()).update(
                reward, rank_after - rank_before, exact, discovery
            )

    def begin_discovery_epoch(self) -> int:
        self.epoch += 1
        return self.epoch

    def observe_pattern(self, support_word: int, pattern_id: str) -> None:
        self.observed_supports[int(support_word)] = pattern_id
        self.observed_pattern_ids.add(pattern_id)

    @staticmethod
    def _stats_payload(table: dict[tuple[str, ...], OnlineStats]) -> dict[str, object]:
        return {
            ":".join(key): {
                "visits": value.visits,
                "mean_reward": value.mean_reward,
                "rank_delta_sum": value.rank_delta_sum,
                "exact_hits": value.exact_hits,
                "discoveries": value.discoveries,
            }
            for key, value in sorted(table.items())
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "total_updates": self.total_updates,
            "observed_pattern_ids": sorted(self.observed_pattern_ids),
            "observed_support_count": len(self.observed_supports),
            "class_frequencies": dict(self.class_frequencies),
            "pattern_stats": self._stats_payload(self.pattern_stats),
            "context_stats": self._stats_payload(self.context_stats),
            "transition_stats": self._stats_payload(self.transition_stats),
        }
