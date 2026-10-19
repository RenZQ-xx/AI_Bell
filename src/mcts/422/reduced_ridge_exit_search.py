from __future__ import annotations

"""Certified reduced-ridge opposite exits, isolated from prior experiments.

A ridge's supporting-normal cone has dimension two. Intersecting it with the
existing slack-sum normalization gives a segment whose endpoints are the two
incident reduced facets. Uncertified cases retain the random-LP fallback.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .reduced_balanced_continuation_search import (
    ReducedBalancedContinuationConfig,
    ReducedBalancedContinuationMCTS,
)
from .reduced_subgroup_search import (
    ReducedAction, ReducedGeometry, word_from_indices, word_indices,
)


@dataclass(frozen=True)
class ReducedRidgeExitConfig(ReducedBalancedContinuationConfig):
    enable_deterministic_ridge_exits: bool = True
    ridge_slope_tolerance: float = 1e-10
    ridge_interval_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        super().__post_init__()
        for value in (self.ridge_slope_tolerance, self.ridge_interval_tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("ridge tolerances must be finite and positive")


@dataclass(frozen=True)
class CertifiedRidgeExits:
    reason: str
    facet_words: tuple[int, ...] = ()
    closed_ridge_word: int = 0
    parameter_interval: tuple[float, float] | None = None

    @property
    def complete(self) -> bool:
        return self.reason == "complete" and len(self.facet_words) == 2


class RidgeExitGeometry(ReducedGeometry):
    def __init__(self, problem, *, slope_tolerance=1e-10, interval_tolerance=1e-10, **kwargs):
        super().__init__(problem, **kwargs)
        self.slope_tolerance = slope_tolerance
        self.interval_tolerance = interval_tolerance
        self.ridge_exit_cache: dict[int, CertifiedRidgeExits] = {}

    def certified_ridge_exits(self, retained_word: int) -> CertifiedRidgeExits:
        self.stats["deterministic_ridge_queries"] += 1
        cached = self.ridge_exit_cache.get(retained_word)
        if cached is not None:
            self.stats["deterministic_ridge_cache_hits"] += 1
            return cached
        self.stats["deterministic_ridge_computations"] += 1
        try:
            result = self._compute_ridge_exits(retained_word)
        except np.linalg.LinAlgError:
            result = CertifiedRidgeExits("svd_failure")
        self.ridge_exit_cache[retained_word] = result
        self.stats[f"deterministic_ridge_result_{result.reason}"] += 1
        return result

    def _compute_ridge_exits(self, retained_word: int) -> CertifiedRidgeExits:
        if self.rank_oracle.rank(retained_word) != self.problem.ridge_rank:
            return CertifiedRidgeExits("not_ridge_rank")
        selected = self.irredundant_augmented[list(word_indices(retained_word))]
        _u, singular, vh = np.linalg.svd(selected, full_matrices=True)
        rank = int(np.count_nonzero(singular > 1e-8))
        null_basis = vh[rank:].T
        if null_basis.shape[1] != 2:
            return CertifiedRidgeExits("normal_space_rank")
        normalization = self.slack_sum_row @ null_basis
        norm = float(np.linalg.norm(normalization))
        if not math.isfinite(norm) or norm <= 1e-12 * max(1.0, float(np.linalg.norm(self.slack_sum_row))):
            return CertifiedRidgeExits("zero_normalization")
        origin = null_basis @ (normalization / (norm * norm))
        direction = null_basis @ (np.asarray([-normalization[1], normalization[0]]) / norm)
        offsets = self.irredundant_augmented @ origin
        slopes = self.irredundant_augmented @ direction
        slope_tol = self.slope_tolerance * max(1.0, float(np.max(np.abs(slopes))))
        constant_tol = 2e-9 * max(1.0, float(np.max(np.abs(offsets))))
        positive, negative = slopes > slope_tol, slopes < -slope_tol
        if np.any(offsets[~(positive | negative)] < -constant_tol):
            return CertifiedRidgeExits("infeasible_constants")
        lower = float(np.max(-offsets[positive] / slopes[positive])) if np.any(positive) else -math.inf
        upper = float(np.min(-offsets[negative] / slopes[negative])) if np.any(negative) else math.inf
        if not math.isfinite(lower) or not math.isfinite(upper):
            return CertifiedRidgeExits("unbounded_interval")
        width_tol = self.interval_tolerance * max(1.0, abs(lower), abs(upper))
        if lower > upper + width_tol:
            return CertifiedRidgeExits("empty_interval")
        if upper - lower <= width_tol:
            return CertifiedRidgeExits("collapsed_interval")
        facets = []
        for parameter in (lower, upper):
            slacks = offsets + parameter * slopes
            tolerance = 2e-7 * max(1.0, float(np.max(np.abs(slacks))))
            if np.any(slacks < -tolerance):
                return CertifiedRidgeExits("endpoint_cuts_polytope")
            word = word_from_indices(np.flatnonzero(np.abs(slacks) <= tolerance))
            if retained_word & ~word:
                return CertifiedRidgeExits("endpoint_lost_retained")
            terminal = self.validate(word)
            if not terminal.valid or retained_word & ~terminal.closure_word:
                return CertifiedRidgeExits("invalid_endpoint_facet")
            facets.append(terminal.closure_word)
        if facets[0] == facets[1]:
            return CertifiedRidgeExits("non_distinct_endpoints")
        intersection = facets[0] & facets[1]
        if self.rank_oracle.rank(intersection) != self.problem.ridge_rank:
            return CertifiedRidgeExits("intersection_not_ridge")
        return CertifiedRidgeExits("complete", tuple(facets), intersection, (lower, upper))

    def exit_cache_summary(self) -> dict:
        return {
            "entries": len(self.ridge_exit_cache),
            "complete_pairs": sum(result.complete for result in self.ridge_exit_cache.values()),
            "reasons": dict(Counter(result.reason for result in self.ridge_exit_cache.values())),
            "failed_results_are_pruning": False,
        }


class ReducedRidgeExitMCTS(ReducedBalancedContinuationMCTS):
    def __init__(self, problem, config: ReducedRidgeExitConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        if config.enable_deterministic_ridge_exits:
            original = self.geometry
            self.geometry = RidgeExitGeometry(
                problem, handoff_root=kwargs["handoff_root"], full_oracle=original.full_oracle,
                slope_tolerance=config.ridge_slope_tolerance,
                interval_tolerance=config.ridge_interval_tolerance,
            )
            self.geometry.shared_full_oracle = original.shared_full_oracle

    def _filler_actions(self, node, *, sample_count, rng):
        if (not self.config.enable_deterministic_ridge_exits
                or node.rank != self.problem.ridge_rank or not node.state.source_facet_word):
            return super()._filler_actions(node, sample_count=sample_count, rng=rng)
        self.stats["ridge_exit_requests"] += 1
        result = self.geometry.certified_ridge_exits(node.state.support_word)
        source = node.state.source_facet_word
        if result.complete and source in result.facet_words:
            opposite = next(word for word in result.facet_words if word != source)
            outside = list(word_indices(opposite & ~node.state.support_word))
            # A certified ridge needs exactly one independent vertex to reach a facet.
            bundle = self._independent_bundle(node.state.support_word, outside, limit=1, rng=rng)
            if bundle:
                self.stats["ridge_exit_actions_generated"] += 1
                return [ReducedAction(
                    "add", add_word=bundle, sampled_facet_word=opposite,
                    source="deterministic_ridge_exit",
                )]
            self.stats["ridge_exit_missing_rank_gain"] += 1
        elif result.complete:
            self.stats["ridge_exit_source_mismatches"] += 1
        self.stats["ridge_exit_lp_fallbacks"] += 1
        return super()._filler_actions(node, sample_count=sample_count, rng=rng)

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        before = len(self.discovery_events)
        reward, new_class = super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)
        if self.config.enable_deterministic_ridge_exits and len(self.discovery_events) != before:
            event = self.discovery_events[-1]
            count = sum(action.source == "deterministic_ridge_exit" for action in path_actions)
            event["deterministic_ridge_actions_in_local_path"] = count
            event["terminal_completion_source"] = edge.action.source
            self.stats["ridge_exit_assisted_discoveries"] += int(count > 0)
            self.stats["ridge_exit_terminal_discoveries"] += int(edge.action.source == "deterministic_ridge_exit")
            if new_class:
                recorded = self.shared_discovery.discovered_classes[event["class_id"]]
                recorded["deterministic_ridge_actions_in_local_path"] = count
                recorded["terminal_completion_source"] = edge.action.source
        return reward, new_class

    def _diagnostic_snapshot(self) -> dict[str, int]:
        snapshot = super()._diagnostic_snapshot()
        for key in ("deterministic_ridge_queries", "deterministic_ridge_cache_hits",
                    "deterministic_ridge_computations", "deterministic_ridge_result_complete"):
            snapshot[key] = self.geometry.stats.get(key, 0)
        for key in ("ridge_exit_actions_generated", "ridge_exit_lp_fallbacks",
                    "ridge_exit_source_mismatches", "ridge_exit_assisted_discoveries",
                    "ridge_exit_terminal_discoveries"):
            snapshot[key] = self.stats.get(key, 0)
        return snapshot

    def result(self, *, target_ids: Sequence[str] = ()) -> dict[str, object]:
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_certified_reduced_ridge_exit_continuation_v3"
        result["features"]["deterministic_reduced_ridge_exits"] = self.config.enable_deterministic_ridge_exits
        result["ridge_exits"] = {
            "enabled": self.config.enable_deterministic_ridge_exits,
            "solver": "two-dimensional normal space / normalized feasible interval",
            "certification": "two distinct supporting facets containing retained points; intersection rank d-2",
            "scope": "reduced adjacency only; no original rank-78 adjacency claim",
            "candidate_validation_is_discovery": False,
            "lift_failures_remain_searchable": True,
            "incomplete_pairs_use_random_lp": True,
            "lower_face_policy_changed": False,
            "cache": self.geometry.exit_cache_summary() if self.config.enable_deterministic_ridge_exits else None,
        }
        return result


def audit_ridge_exits(problem, events, handoff_root, full_oracle) -> dict:
    """Replay adjacency with the base geometry, independently of the interval solver."""
    geometry = ReducedGeometry(problem, handoff_root=handoff_root, full_oracle=full_oracle)
    failures = []
    checked = set()
    occurrences = 0
    for event in events:
        try:
            word = source = 0
            for action in event["path"]:
                if action["kind"] == "remove":
                    source = word
                    word = word_from_indices(action["retained_vertices"])
                elif action["kind"] == "add":
                    added = word_from_indices(action["add_vertices"])
                    if action.get("source") == "deterministic_ridge_exit":
                        occurrences += 1
                        sampled = int(action["sampled_facet_word_hex"], 16)
                        key = (word, source, added, sampled)
                        if key not in checked:
                            first, second = geometry.validate(source), geometry.validate(sampled)
                            if not first.valid or not second.valid or first.closure_word == second.closure_word:
                                raise ValueError("ridge exits are not two distinct reduced facets")
                            if (word & ~first.closure_word or word & ~second.closure_word
                                    or geometry.rank_oracle.rank(word) != problem.ridge_rank
                                    or geometry.rank_oracle.rank(first.closure_word & second.closure_word) != problem.ridge_rank):
                                raise ValueError("retained support/intersection is not their common ridge")
                            terminal = geometry.validate(word | added)
                            if added.bit_count() != 1 or not terminal.valid or terminal.closure_word != second.closure_word:
                                raise ValueError("ridge completion did not reach the opposite facet")
                            checked.add(key)
                    word |= added
                else:
                    raise ValueError("unknown reduced action")
                if geometry.rank_oracle.rank(word) == problem.facet_rank:
                    terminal = geometry.validate(word)
                    if not terminal.valid:
                        raise ValueError("invalid intermediate facet")
                    word = terminal.closure_word
        except (ValueError, KeyError, TypeError, IndexError) as error:
            failures.append({"class_id": event.get("class_id"), "error": str(error)})
    return {
        "paths_checked": len(events), "ridge_action_occurrences": occurrences,
        "unique_ridge_actions_checked": len(checked), "failure_count": len(failures),
        "failures": failures, "uses_interval_solver": False,
        "scope": "reduced adjacency, not original-space rank-78 adjacency",
    }
