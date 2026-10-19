from __future__ import annotations

"""Certified lower-face outlets and bounded sampling backoff, isolated as v6."""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .reduced_frontier_search import ReducedFrontierConfig, ReducedFrontierMCTS
from .reduced_novelty_search import DeferredLiftGeometry
from .reduced_subgroup_search import ReducedAction, ReducedGeometry, word_from_indices, word_indices


MULTI_STAT_KEYS = (
    "lower_exit_requests", "lower_exit_actions_generated", "lower_exit_lp_fallbacks",
    "lower_exit_source_mismatches", "lower_exit_unvisited_selected",
    "lower_exit_terminal_arrivals", "lower_exit_new_reduced_arrivals",
    "lower_exit_assisted_class_discoveries", "lower_exit_terminal_class_discoveries",
)
GEOMETRY_STAT_KEYS = (
    "lower_exit_queries", "lower_exit_cache_hits", "lower_exit_computations",
    "lower_exit_result_complete", "lower_exit_boundary_pairs",
    "ridge_backoff_opportunities", "ridge_backoff_skipped", "ridge_backoff_probes",
    "ridge_backoff_zero_increment_samples", "ridge_backoff_new_ridges",
)


@dataclass(frozen=True)
class ReducedMultiExitConfig(ReducedFrontierConfig):
    enable_lower_face_exits: bool = True
    enable_ridge_sampling_backoff: bool = True
    ridge_backoff_max_interval: int = 32

    def __post_init__(self):
        super().__post_init__()
        if self.ridge_backoff_max_interval < 1:
            raise ValueError("backoff interval must be positive")


@dataclass(frozen=True)
class CertifiedLowerFaceExits:
    reason: str
    facet_words: tuple[int, ...] = ()
    closed_face_word: int = 0

    @property
    def complete(self):
        return self.reason == "complete" and len(self.facet_words) >= 3


class MultiExitGeometry(DeferredLiftGeometry):
    def __init__(self, *args, sampling_backoff=True, backoff_max_interval=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower_exit_cache: dict[int, CertifiedLowerFaceExits] = {}
        self.sampling_backoff = sampling_backoff
        self.backoff_max_interval = backoff_max_interval
        self.ridge_sample_seen: dict[int, set[int]] = {}
        self.ridge_sample_opportunities: Counter[int] = Counter()
        self.ridge_zero_streak: Counter[int] = Counter()
        self.ridge_next_probe: Counter[int] = Counter()

    def sample_ridges(self, facet_word, *, count, rng):
        if not self.sampling_backoff:
            return super().sample_ridges(facet_word, count=count, rng=rng)
        self.stats["ridge_backoff_opportunities"] += 1
        self.ridge_sample_opportunities[facet_word] += 1
        opportunity = self.ridge_sample_opportunities[facet_word]
        if opportunity < self.ridge_next_probe[facet_word]:
            self.stats["ridge_backoff_skipped"] += 1
            return ()
        self.stats["ridge_backoff_probes"] += 1
        sampled = super().sample_ridges(facet_word, count=count, rng=rng)
        seen = self.ridge_sample_seen.setdefault(facet_word, set())
        new = set(sampled) - seen
        seen.update(sampled)
        self.stats["ridge_backoff_new_ridges"] += len(new)
        if new:
            self.ridge_zero_streak[facet_word] = 0
        else:
            self.ridge_zero_streak[facet_word] += 1
            self.stats["ridge_backoff_zero_increment_samples"] += 1
        interval = min(self.backoff_max_interval,
                       2 ** min(20, max(0, self.ridge_zero_streak[facet_word] - 1)))
        self.ridge_next_probe[facet_word] = opportunity + interval
        return sampled

    def certified_lower_face_exits(self, retained_word):
        self.stats["lower_exit_queries"] += 1
        if retained_word in self.lower_exit_cache:
            self.stats["lower_exit_cache_hits"] += 1
            return self.lower_exit_cache[retained_word]
        self.stats["lower_exit_computations"] += 1
        try:
            result = self._compute_lower_exits(retained_word)
        except np.linalg.LinAlgError:
            result = CertifiedLowerFaceExits("svd_failure")
        self.lower_exit_cache[retained_word] = result
        self.stats[f"lower_exit_result_{result.reason}"] += 1
        return result

    def _compute_lower_exits(self, retained_word):
        if self.rank_oracle.rank(retained_word) != self.problem.facet_rank - 2:
            return CertifiedLowerFaceExits("not_lower_face_rank")
        selected = self.irredundant_augmented[list(word_indices(retained_word))]
        _u, singular, vh = np.linalg.svd(selected, full_matrices=True)
        basis = vh[int(np.count_nonzero(singular > 1e-8)):].T
        if basis.shape[1] != 3:
            return CertifiedLowerFaceExits("normal_space_rank")
        normalization = self.slack_sum_row @ basis
        norm = float(np.linalg.norm(normalization))
        if not math.isfinite(norm) or norm <= 1e-12 * max(1.0, np.linalg.norm(self.slack_sum_row)):
            return CertifiedLowerFaceExits("zero_normalization")
        origin = basis @ (normalization / norm ** 2)
        _u, _s, plane = np.linalg.svd(normalization[None, :], full_matrices=True)
        directions = basis @ plane[1:].T
        offsets = self.irredundant_augmented @ origin
        slopes = self.irredundant_augmented @ directions
        lengths = np.linalg.norm(slopes, axis=1)
        active = lengths > self.slope_tolerance * max(1.0, float(np.max(lengths)))
        if np.any(offsets[~active] < -2e-9):
            return CertifiedLowerFaceExits("infeasible_constants")
        normals = slopes[active] / lengths[active, None]
        constants = offsets[active] / lengths[active]
        if len(normals) < 3:
            return CertifiedLowerFaceExits("too_few_boundaries")
        # A nonzero recession direction exists iff all inward normals fit in a semicircle.
        angles = np.sort(np.arctan2(normals[:, 1], normals[:, 0]))
        gaps = np.diff(np.concatenate((angles, angles[:1] + 2 * np.pi)))
        if float(np.max(gaps)) >= np.pi - 1e-9:
            return CertifiedLowerFaceExits("unbounded_or_uncertain_section")
        left, right = np.triu_indices(len(normals), 1)
        determinant = (normals[left, 0] * normals[right, 1]
                       - normals[left, 1] * normals[right, 0])
        usable = np.abs(determinant) > self.slope_tolerance
        left, right, determinant = left[usable], right[usable], determinant[usable]
        self.stats["lower_exit_boundary_pairs"] += len(left)
        parameters = np.column_stack((
            (-constants[left] * normals[right, 1] + constants[right] * normals[left, 1]) / determinant,
            (-normals[left, 0] * constants[right] + normals[right, 0] * constants[left]) / determinant,
        ))
        slacks = offsets[None, :] + parameters @ slopes.T
        tolerance = 2e-7 * np.maximum(1.0, np.max(np.abs(slacks), axis=1))
        feasible = np.all(slacks >= -tolerance[:, None], axis=1)
        if not np.any(feasible):
            return CertifiedLowerFaceExits("empty_section")
        vertices = parameters[feasible]
        if np.linalg.matrix_rank(vertices - vertices[0], tol=1e-9) != 2:
            return CertifiedLowerFaceExits("collapsed_section")
        words = {word_from_indices(np.flatnonzero(np.abs(row) <= tol))
                 for row, tol in zip(slacks[feasible], tolerance[feasible])}
        facets = set()
        for word in words:
            if retained_word & ~word:
                return CertifiedLowerFaceExits("endpoint_lost_retained")
            terminal = self.validate(word)
            if not terminal.valid or retained_word & ~terminal.closure_word:
                return CertifiedLowerFaceExits("invalid_endpoint_facet")
            facets.add(terminal.closure_word)
        if len(facets) < 3:
            return CertifiedLowerFaceExits("too_few_facets")
        closed = (1 << len(self.problem.irredundant_points)) - 1
        for word in facets:
            closed &= word
        if self.rank_oracle.rank(closed) != self.problem.facet_rank - 2:
            return CertifiedLowerFaceExits("intersection_not_lower_face")
        return CertifiedLowerFaceExits("complete", tuple(sorted(facets)), closed)


class ReducedMultiExitMCTS(ReducedFrontierMCTS):
    def __init__(self, problem, config: ReducedMultiExitConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        original = self.geometry
        self.geometry = MultiExitGeometry(
            problem, handoff_root=kwargs["handoff_root"], full_oracle=original.full_oracle,
            slope_tolerance=config.ridge_slope_tolerance,
            interval_tolerance=config.ridge_interval_tolerance,
            sampling_backoff=config.enable_ridge_sampling_backoff,
            backoff_max_interval=config.ridge_backoff_max_interval,
        )
        self.geometry.shared_full_oracle = original.shared_full_oracle

    def _filler_actions(self, node, *, sample_count, rng):
        source, retained = node.state.source_facet_word, node.state.support_word
        if (not self.config.enable_lower_face_exits or not source
                or node.rank != self.problem.facet_rank - 2):
            return super()._filler_actions(node, sample_count=sample_count, rng=rng)
        self.stats["lower_exit_requests"] += 1
        result = self.geometry.certified_lower_face_exits(retained)
        if result.complete and source in result.facet_words:
            choices = [word for word in result.facet_words if word != source]
            destination = min(choices, key=lambda word: (
                int((self.problem.node_id, word) in self.shared_discovery.reduced_keys),
                self.frontier_traversals[(source, retained, word)], rng.random(),
            ))
            outside = list(word_indices(destination & ~retained))
            bundle = self._independent_bundle(retained, outside, limit=2, rng=rng)
            if bundle.bit_count() == 2:
                self.stats["lower_exit_actions_generated"] += 1
                self.stats["lower_exit_unvisited_selected"] += int(
                    (self.problem.node_id, destination) not in self.shared_discovery.reduced_keys)
                return [ReducedAction("add", add_word=bundle, sampled_facet_word=destination,
                                      source="certified_lower_face_exit")]
        elif result.complete:
            self.stats["lower_exit_source_mismatches"] += 1
        self.stats["lower_exit_lp_fallbacks"] += 1
        return super()._filler_actions(node, sample_count=sample_count, rng=rng)

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        terminal = edge.terminal
        new_reduced = bool(terminal and (self.problem.node_id, terminal.closure_word)
                           not in self.shared_discovery.reduced_keys)
        before = len(self.discovery_events)
        result = super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)
        if terminal and edge.action.source == "certified_lower_face_exit":
            self.stats["lower_exit_terminal_arrivals"] += 1
            self.stats["lower_exit_new_reduced_arrivals"] += int(new_reduced)
        if len(self.discovery_events) > before:
            count = sum(action.source == "certified_lower_face_exit" for action in path_actions)
            self.discovery_events[-1]["lower_exit_actions_in_local_path"] = count
            self.stats["lower_exit_assisted_class_discoveries"] += int(count > 0)
            self.stats["lower_exit_terminal_class_discoveries"] += int(
                edge.action.source == "certified_lower_face_exit")
        return result

    def _diagnostic_snapshot(self):
        snapshot = super()._diagnostic_snapshot()
        snapshot.update({key: self.stats[key] for key in MULTI_STAT_KEYS})
        snapshot.update({key: self.geometry.stats[key] for key in GEOMETRY_STAT_KEYS})
        return snapshot

    def result(self, *, target_ids=()):
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_certified_reduced_lower_face_multi_exit_v6"
        result["ridge_exits"]["lower_face_policy_changed"] = self.config.enable_lower_face_exits
        cache = self.geometry.lower_exit_cache
        result["lower_face_exits"] = {
            "enabled": self.config.enable_lower_face_exits,
            "solver": "three-dimensional normal cone / bounded two-dimensional halfspace polygon",
            "selection": "unvisited reduced closure, then least actual source/face/exit traversals",
            "candidate_queries_are_discoveries": False,
            "incomplete_results_use_random_lp": True,
            "exhausted_exits_are_facet_completeness": False,
            "reward_and_productivity_unchanged": True,
            "cache_entries": len(cache),
            "complete_faces": sum(item.complete for item in cache.values()),
            "outlet_count_histogram": dict(Counter(len(item.facet_words) for item in cache.values() if item.complete)),
            "cache_reasons": dict(Counter(item.reason for item in cache.values())),
            "sampling_backoff_enabled": self.config.enable_ridge_sampling_backoff,
            "sampling_backoff_max_interval": self.config.ridge_backoff_max_interval,
            "periodic_sampling_preserved": True,
            "statistics": {**{key: self.stats[key] for key in MULTI_STAT_KEYS},
                           **{key: self.geometry.stats[key] for key in GEOMETRY_STAT_KEYS}},
        }
        return result


def audit_lower_face_exits(problem, events, handoff_root, full_oracle=None):
    """Replay actual exits with base facet validation, without the polygon solver."""
    class ReducedOnlyGeometry(ReducedGeometry):
        def _lift_reduced_support(self, original_support):
            return None

    geometry = ReducedOnlyGeometry(problem, handoff_root=handoff_root, full_oracle=full_oracle)
    failures, checked, occurrences = [], set(), 0
    for event in events:
        try:
            word = source = 0
            for action in event["path"]:
                if action["kind"] == "remove":
                    source = word
                    word = word_from_indices(action["retained_vertices"])
                elif action["kind"] == "add":
                    added = word_from_indices(action["add_vertices"])
                    if action.get("source") == "certified_lower_face_exit":
                        occurrences += 1
                        sampled = int(action["sampled_facet_word_hex"], 16)
                        key = (word, source, added, sampled)
                        if key not in checked:
                            first, second = geometry.validate(source), geometry.validate(sampled)
                            if not first.valid or not second.valid or first.closure_word == second.closure_word:
                                raise ValueError("lower exits are not distinct supporting facets")
                            if (word & ~first.closure_word or word & ~second.closure_word
                                    or geometry.rank_oracle.rank(word) != problem.facet_rank - 2):
                                raise ValueError("exits lost their common rank-(d-3) face")
                            terminal = geometry.validate(word | added)
                            if (added.bit_count() != 2 or added & word or not terminal.valid
                                    or terminal.closure_word != second.closure_word):
                                raise ValueError("lower completion did not reach the selected facet")
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
    return {"paths_checked": len(events), "lower_action_occurrences": occurrences,
            "unique_lower_actions_checked": len(checked), "failure_count": len(failures),
            "failures": failures, "uses_polygon_solver": False, "uses_full_lifting": False,
            "scope": "reduced supporting facets sharing a rank-(d-3) retained face"}
