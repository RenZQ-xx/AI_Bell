from __future__ import annotations

"""Quarter-budget continuation with entrance-local rollout cycles.

All geometry, rewards, actions, widening, and RNG streams come from the previous
continuation experiment. Only entrance allocation and rollout-slot assignment
change. Diagnostics are observational and never participate in selection.
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .reduced_continuation_search import (
    ReducedContinuationConfig,
    ReducedContinuationMCTS,
)
from .reduced_subgroup_search import ROLLOUT_CYCLE


@dataclass(frozen=True)
class ReducedBalancedContinuationConfig(ReducedContinuationConfig):
    continuation_interval: int = 4
    entry_local_rollout_cycle: bool = True
    record_entry_diagnostics: bool = True
    diagnostic_window: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.diagnostic_window <= 0:
            raise ValueError("diagnostic_window must be positive")


class ReducedBalancedContinuationMCTS(ReducedContinuationMCTS):
    def __init__(self, problem, config: ReducedBalancedContinuationConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.entry_episodes: Counter[str] = Counter()
        self.entry_policy_counts: dict[str, Counter] = defaultdict(Counter)
        self.entry_metrics: dict[str, Counter] = defaultdict(Counter)
        self.entry_windows: dict[int, dict[str, Counter]] = {}
        self.current_entry_policy: str | None = None

    @property
    def entry_kind(self) -> str:
        return "continuation" if self.episode_origin else "empty_root"

    def _rollout_policy(self, iteration: int) -> str:
        kind = self.entry_kind
        self.entry_episodes[kind] += 1
        ordinal = (
            self.entry_episodes[kind]
            if self.config.entry_local_rollout_cycle else iteration
        )
        policy = ROLLOUT_CYCLE[(ordinal - 1) % len(ROLLOUT_CYCLE)]
        self.entry_policy_counts[kind][policy] += 1
        self.current_entry_policy = policy
        return policy

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        before = len(self.discovery_events)
        reward, new_class = super()._step_reward(
            node, edge, path_actions, path_terminals, tree_edge
        )
        if (self.config.enable_local_continuation and self.config.entry_local_rollout_cycle
                and len(self.discovery_events) != before):
            event = self.discovery_events[-1]
            event["rollout_policy"] = self.current_entry_policy
            event["entry_episode_ordinal"] = self.entry_episodes[self.entry_kind]
            if new_class:
                recorded = self.shared_discovery.discovered_classes[event["class_id"]]
                recorded["rollout_policy"] = event["rollout_policy"]
                recorded["entry_episode_ordinal"] = event["entry_episode_ordinal"]
        return reward, new_class

    def _diagnostic_snapshot(self) -> dict[str, int]:
        geometry = self.geometry.stats
        return {
            "new_local_classes": len(self.discovery_events),
            "new_global_classes": self.stats.get("new_global_classes", 0),
            "support_lp_attempts": geometry.get("support_lp_attempts", 0),
            "ridge_lp_attempts": geometry.get("ridge_lp_attempts", 0),
            "source_samples_skipped": self.stats.get("source_facet_samples_skipped", 0),
            "source_returns": self.stats.get("corrector_source_returns", 0),
            "reduced_facet_validations": geometry.get("reduced_facet_validations", 0),
            "successful_full_lifts": geometry.get("successful_full_lifts", 0),
            "failed_full_lifts": geometry.get("failed_full_lifts", 0),
            "rank_cache_misses": self.geometry.rank_oracle.misses,
            "tree_policy_steps": self.stats.get("tree_policy_steps", 0),
            "rollout_steps": self.stats.get("rollout_steps", 0),
        }

    def run_iteration(self, iteration: int, *, global_iteration: int | None = None) -> None:
        before = self._diagnostic_snapshot() if self.config.record_entry_diagnostics else None
        started = time.perf_counter() if before is not None else None
        super().run_iteration(iteration, global_iteration=global_iteration)
        if not self.config.enable_local_continuation:
            # Disabled continuation delegates directly to the unchanged baseline.
            self.entry_episodes["empty_root"] += 1
            policy = ROLLOUT_CYCLE[(iteration - 1) % len(ROLLOUT_CYCLE)]
            self.entry_policy_counts["empty_root"][policy] += 1
        if before is not None:
            elapsed = time.perf_counter() - started
            after = self._diagnostic_snapshot()
            delta = {key: after[key] - value for key, value in before.items()}
            delta.update({"iterations": 1, "elapsed_seconds": elapsed})
            self.entry_metrics[self.entry_kind].update(delta)
            window = (iteration - 1) // self.config.diagnostic_window
            entries = self.entry_windows.setdefault(window, defaultdict(Counter))
            entries[self.entry_kind].update(delta)

    @staticmethod
    def _metric_result(metrics: Counter) -> dict:
        result = dict(metrics)
        seconds = metrics.get("elapsed_seconds", 0.0)
        iterations = metrics.get("iterations", 0)
        result["new_classes_per_iteration"] = (
            metrics.get("new_global_classes", 0) / iterations if iterations else None
        )
        result["new_classes_per_second"] = (
            metrics.get("new_global_classes", 0) / seconds if seconds else None
        )
        result["lp_attempts"] = (
            metrics.get("support_lp_attempts", 0) + metrics.get("ridge_lp_attempts", 0)
        )
        return result

    def result(self, *, target_ids: Sequence[str] = ()) -> dict[str, object]:
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_entry_balanced_reduced_continuation_v2"
        result["features"]["entry_local_rollout_cycle"] = (
            self.config.enable_local_continuation and self.config.entry_local_rollout_cycle
        )
        result["entry_rollout"] = {
            "cycle": list(ROLLOUT_CYCLE),
            "cycle_index": "per_entrance_episode_count" if self.config.entry_local_rollout_cycle
                           and self.config.enable_local_continuation else "global_iteration",
            "policy_rngs": "unchanged shared policy streams; no entrance-specific RNG",
            "planned_continuation_fraction": (
                1 / self.config.continuation_interval if self.config.enable_local_continuation else 0.0
            ),
            "episode_counts": dict(self.entry_episodes),
            "policy_episode_counts": {kind: dict(counts) for kind, counts in self.entry_policy_counts.items()},
        }
        result["entry_diagnostics"] = {
            "enabled": self.config.record_entry_diagnostics,
            "selection_reads_diagnostics": False,
            "lift_count_scope": "cache-miss geometry validations, including unexecuted candidates",
            "source_returns_scope": "actual terminal hits equal to the correction source",
            "entries": {kind: self._metric_result(metrics) for kind, metrics in self.entry_metrics.items()},
            "window_size": self.config.diagnostic_window,
            "windows": [
                {
                    "start_iteration_exclusive": window * self.config.diagnostic_window,
                    "end_iteration_inclusive": min(
                        (window + 1) * self.config.diagnostic_window, self.current_iteration
                    ),
                    "entries": {kind: self._metric_result(metrics) for kind, metrics in entries.items()},
                }
                for window, entries in sorted(self.entry_windows.items())
            ],
        }
        return result
