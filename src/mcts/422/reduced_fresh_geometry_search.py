from __future__ import annotations

"""Sparse outlet service and genuinely fresh ridge probes, isolated as v8."""

import random
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .reduced_frontier_schedule_search import ReducedFrontierScheduleConfig, ReducedFrontierScheduleMCTS
from .reduced_multi_exit_search import MultiExitGeometry, ReducedMultiExitMCTS
from .reduced_subgroup_search import ReducedAction, word_from_indices, word_indices


SPARSE_CYCLE = ("regular",) * 6 + ("remaining_exit", "fresh_geometry")
PROBE_MODES = ("isotropic", "vertex", "vertex", "pair")
SPARSE_STATS = (
    "sparse_regular_removals", "sparse_remaining_exit_removals", "sparse_fresh_geometry_removals",
    "sparse_remaining_exit_fulfilled", "sparse_fresh_geometry_fulfilled",
    "sparse_reserved_candidates", "sparse_existing_edge_reservations", "sparse_probe_pw_skipped",
    "sparse_probe_ticket_reuses", "sparse_probe_bank_insertions", "sparse_probe_bank_rejections",
    "fresh_probe_executed_proposals", "fresh_probe_exit_arrivals",
    "fresh_probe_new_reduced_exit_arrivals", "fresh_probe_terminal_class_discoveries",
)
PROBE_STATS = (
    "fresh_probe_requests", "fresh_probe_lp_attempts", "fresh_probe_lp_failures",
    "fresh_probe_nonridge", "fresh_probe_seen_geometry", "fresh_probe_new_geometry",
    "fresh_probe_isotropic_attempts", "fresh_probe_vertex_attempts", "fresh_probe_pair_attempts",
    "fresh_probe_coordinate_failures", "fresh_probe_backoff_resets",
)


@dataclass(frozen=True)
class ReducedFreshGeometryConfig(ReducedFrontierScheduleConfig):
    enable_frontier_anchor_schedule: bool = False
    enable_fresh_geometry_probes: bool = True
    fresh_probe_lp_limit: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.fresh_probe_lp_limit < 1:
            raise ValueError("fresh geometry LP limit must be positive")
        if self.enable_fresh_geometry_probes and not self.enable_shared_face_frontier:
            raise ValueError("fresh geometry probes require the shared face frontier")


@dataclass(frozen=True)
class ReducedFreshProbeOnlyConfig(ReducedFreshGeometryConfig):
    enable_corrector_outlet_quota: bool = False


@dataclass(frozen=True)
class ReducedSparseQuotaOnlyConfig(ReducedFreshGeometryConfig):
    enable_fresh_geometry_probes: bool = False


class FreshProbeGeometry(MultiExitGeometry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_vertex_attempts = {}
        self.last_probe = {}

    def sample_ridges(self, facet_word, *, count, rng):
        sampled = super().sample_ridges(facet_word, count=count, rng=rng)
        self.ridge_sample_seen.setdefault(facet_word, set()).update(sampled)
        return sampled

    def _probe_coordinates(self, source):
        cached = self.ridge_coordinate_cache.get(source)
        if cached is not None:
            return cached
        indices = word_indices(source)
        if not indices:
            return None
        points = self.problem.irredundant_points[list(indices)]
        centered = points - points[0]
        _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
        rank = int(np.count_nonzero(singular > 1e-8))
        if rank != self.problem.facet_rank or rank < 1:
            return None
        cached = indices, centered @ vh[:rank].T
        self.ridge_coordinate_cache[source] = cached
        return cached

    def probe_fresh_ridge(self, source, *, seen, lp_limit, rng):
        mode = PROBE_MODES[self.stats["fresh_probe_requests"] % len(PROBE_MODES)]
        self.stats["fresh_probe_requests"] += 1
        self.last_probe = {"mode": mode, "condition_vertices": [], "lp_attempts": 0}
        try:
            cached = self._probe_coordinates(source)
        except np.linalg.LinAlgError:
            cached = None
        if cached is None:
            self.stats["fresh_probe_coordinate_failures"] += 1
            return ()
        indices, points = cached
        augmented = np.column_stack((np.ones(len(points)), points))
        seed_count = min({"isotropic": 0, "vertex": 1, "pair": 2}[mode], self.problem.facet_rank)
        local_ids = sorted(range(len(indices)), key=lambda index: (
            self.condition_vertex_attempts.get((source, indices[index]), 0), rng.random(),
        ))[:seed_count] if seed_count else []
        self.last_probe["condition_vertices"] = [indices[index] for index in local_ids]
        equalities = [augmented.sum(axis=0)] + [augmented[index] for index in local_ids]
        a_eq = np.asarray(equalities)
        b_eq = np.asarray([1.0] + [0.0] * len(local_ids))
        for _ in range(lp_limit):
            self.stats["ridge_lp_attempts"] += 1
            self.stats["fresh_probe_lp_attempts"] += 1
            self.stats[f"fresh_probe_{mode}_attempts"] += 1
            self.last_probe["lp_attempts"] += 1
            for index in local_ids:
                key = source, indices[index]
                self.condition_vertex_attempts[key] = self.condition_vertex_attempts.get(key, 0) + 1
            objective = np.asarray([rng.gauss(0.0, 1.0) for _ in range(augmented.shape[1])])
            result = linprog(objective, A_ub=-augmented, b_ub=np.zeros(len(points)),
                             A_eq=a_eq, b_eq=b_eq, bounds=[(None, None)] * augmented.shape[1],
                             method="highs")
            if not result.success:
                self.stats["ridge_lp_failures"] += 1
                self.stats["fresh_probe_lp_failures"] += 1
                if result.status == 2:
                    break
                continue
            slacks = augmented @ result.x
            tolerance = 2e-7 * max(1.0, float(np.max(np.abs(slacks))))
            word = word_from_indices(indices[int(index)] for index in np.flatnonzero(np.abs(slacks) <= tolerance))
            if (np.any(slacks < -tolerance)
                    or any(not word & (1 << indices[index]) for index in local_ids)
                    or self.rank_oracle.rank(word) != self.problem.ridge_rank):
                self.stats["ridge_lp_nonridge"] += 1
                self.stats["fresh_probe_nonridge"] += 1
                continue
            if word in seen:
                self.stats["fresh_probe_seen_geometry"] += 1
                continue
            self.stats["sampled_ridges"] += 1
            self.stats["fresh_probe_new_geometry"] += 1
            self.ridge_sample_seen.setdefault(source, set()).add(word)
            self.ridge_zero_streak[source] = 0
            self.ridge_next_probe[source] = self.ridge_sample_opportunities[source] + 1
            self.stats["fresh_probe_backoff_resets"] += 1
            return (word,)
        return ()


class ReducedFreshGeometryMCTS(ReducedFrontierScheduleMCTS):
    def __init__(self, problem, config: ReducedFreshGeometryConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        original = self.geometry
        self.geometry = FreshProbeGeometry(
            problem, handoff_root=kwargs["handoff_root"], full_oracle=original.full_oracle,
            slope_tolerance=config.ridge_slope_tolerance, interval_tolerance=config.ridge_interval_tolerance,
            sampling_backoff=config.enable_ridge_sampling_backoff,
            backoff_max_interval=config.ridge_backoff_max_interval,
        )
        self.geometry.shared_full_oracle = original.shared_full_oracle
        self.fresh_probe_rng = random.Random(config.seed ^ 0xF8E5A713)
        self.removal_clock = 0
        self.probe_tickets = {}
        self.probe_events = []
        self.probe_face_events = {}
        self.face_first_banked = {}

    def _slot(self):
        return SPARSE_CYCLE[self.removal_clock % len(SPARSE_CYCLE)]

    def _filler_actions(self, node, *, sample_count, rng):
        actions = ReducedMultiExitMCTS._filler_actions(self, node, sample_count=sample_count, rng=rng)
        self._remember_cached_face(node.state.source_facet_word, node.state.support_word)
        return actions

    def _corrector_actions(self, node, *, count, rng, persistent):
        source = node.state.support_word
        before = self._bank_words(source)
        actions = ReducedMultiExitMCTS._corrector_actions(self, node, count=count, rng=rng, persistent=persistent)
        for word in self._bank_words(source) - before:
            self.face_first_generated.setdefault((source, word), self.current_iteration)
            self.face_first_banked.setdefault((source, word), self.current_iteration)
        return actions

    def _remaining_action(self, node, *, rng, allowed=None):
        source = node.state.support_word
        words = [word for word in self.cached_source_faces.get(source, ())
                 if self._outlet_frontier(source, word)[0] and (allowed is None or word in allowed)]
        if not words:
            return None
        retained = min(sorted(words), key=lambda word: (
            -len(self._outlet_frontier(source, word)[0]), self.frontier_attempts[(source, word)], rng.random(),
        ))
        lower = self.geometry.rank_oracle.rank(retained) == self.problem.facet_rank - 2
        return ReducedAction("remove", retained_word=retained, sampled_facet_word=source,
                             source="reduced_facet_lower_face" if lower else "reduced_facet_ridge")

    def _probe_action(self, node, *, rollout):
        source = node.state.support_word
        if source in self.probe_tickets:
            self.stats["sparse_probe_ticket_reuses"] += 1
            return self.probe_tickets[source]
        seen = (self.geometry.ridge_sample_seen.get(source, set()) | self._bank_words(source)
                | self.cached_source_faces.get(source, set()))
        words = self.geometry.probe_fresh_ridge(source, seen=seen, lp_limit=self.config.fresh_probe_lp_limit,
                                               rng=self.fresh_probe_rng)
        action = None
        if words:
            retained = words[0]
            self.face_first_generated.setdefault((source, retained), self.current_iteration)
            capacity = min(self.config.corrector_face_bank_limit,
                           max(self.frontier_capacities.get(source, 0), len(self.corrector_ridge_bank[source]) + 1))
            self.frontier_capacities[source] = capacity
            banked = self._bank_face(self.corrector_ridge_bank[source], retained, source, capacity, refresh=True)
            self.stats["sparse_probe_bank_insertions" if banked else "sparse_probe_bank_rejections"] += 1
            if banked:
                self.face_first_banked.setdefault((source, retained), self.current_iteration)
            action = ReducedAction("remove", retained_word=retained, sampled_facet_word=source,
                                   source="reduced_facet_ridge")
        self.probe_tickets[source] = action
        event = {
            "iteration": self.current_iteration, "removal_clock": self.removal_clock,
            "source_word_hex": f"0x{source:x}", "rollout_query": rollout,
            "retained_word_hex": f"0x{words[0]:x}" if words else None,
            "stored_in_bank": words[0] in self._bank_words(source) if words else False,
            "first_taken_iteration": None, "exit_arrivals": 0, "new_reduced_exit_arrivals": 0,
            "terminal_class_discoveries": 0,
            **self.geometry.last_probe,
        }
        self.probe_events.append(event)
        if words:
            self.probe_face_events[(source, words[0])] = event
        return action

    def _planned_action(self, node, *, rollout, rng, allowed=None):
        if node.state.corrections_used >= self.config.productive_max_corrections:
            return None
        slot = self._slot()
        if slot == "remaining_exit" and self.config.enable_corrector_outlet_quota:
            return self._remaining_action(node, rng=rng, allowed=allowed)
        if slot == "fresh_geometry" and self.config.enable_fresh_geometry_probes and allowed is None:
            return self._probe_action(node, rollout=rollout)
        return None

    def _generate_actions(self, node, *, rollout, policy, rng):
        reserved = self._planned_action(node, rollout=rollout, rng=rng) if node.state.role == "corrector" else None
        if reserved is not None and self._slot() == "fresh_geometry":
            self.stats["sparse_reserved_candidates"] += 1
            return [reserved]
        regular = ReducedMultiExitMCTS._generate_actions(self, node, rollout=rollout, policy=policy, rng=rng)
        if reserved is None:
            return regular
        self.stats["sparse_reserved_candidates"] += 1
        return [reserved] + [action for action in regular if action.action_id != reserved.action_id]

    def _widen(self, node):
        if node.state.role != "corrector":
            return ReducedMultiExitMCTS._widen(self, node)
        self._tree_quota_choices.pop(node.state.key, None)
        if self._slot() == "fresh_geometry" and len(node.edges) >= self._child_limit(node):
            self.stats["sparse_probe_pw_skipped"] += int(self.config.enable_fresh_geometry_probes)
            return None
        reserved = self._planned_action(node, rollout=False, rng=self.rng)
        if reserved is not None:
            existing = next((edge for edge in node.edges.values()
                             if edge.action.retained_word == reserved.retained_word), None)
            if existing is not None:
                self._tree_quota_choices[node.state.key] = reserved.retained_word
                self.stats["sparse_existing_edge_reservations"] += 1
                return None
            if len(node.edges) < self._child_limit(node):
                node.pending_actions[:] = [reserved] + [action for action in node.pending_actions
                                                       if action.action_id != reserved.action_id]
        return ReducedMultiExitMCTS._widen(self, node)

    def _select(self, node):
        if node.state.role == "corrector":
            planned = self._tree_quota_choices.pop(node.state.key, None)
            if planned is not None:
                return next(edge for edge in node.edges.values() if edge.action.retained_word == planned)
            allowed = {edge.action.retained_word for edge in node.edges.values()}
            reserved = self._planned_action(node, rollout=False, rng=self.rng, allowed=allowed)
            if reserved is not None:
                return next(edge for edge in node.edges.values() if edge.action.retained_word == reserved.retained_word)
        return ReducedMultiExitMCTS._select(self, node)

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        if edge.action.kind == "remove":
            source, retained = node.state.support_word, edge.action.retained_word
            slot = self._slot()
            self.stats[f"sparse_{slot}_removals"] += 1
            if slot == "remaining_exit":
                self.stats["sparse_remaining_exit_fulfilled"] += int(
                    self.config.enable_corrector_outlet_quota and bool(self._outlet_frontier(source, retained)[0]))
            elif slot == "fresh_geometry":
                planned = self.probe_tickets.get(source)
                self.stats["sparse_fresh_geometry_fulfilled"] += int(
                    planned is not None and planned.retained_word == retained)
            self.source_removals[source] += 1
            self.face_first_attempted.setdefault((source, retained), self.current_iteration)
            event = self.probe_face_events.get((source, retained))
            if event is not None and event["first_taken_iteration"] is None:
                event["first_taken_iteration"] = self.current_iteration
                self.stats["fresh_probe_executed_proposals"] += 1
            self.removal_clock += 1
            self.probe_tickets.clear()
        event = None
        if edge.terminal is not None and node.state.source_facet_word:
            removal = next((action for action in reversed(path_actions) if action.kind == "remove"), None)
            if removal is not None and removal.sampled_facet_word == node.state.source_facet_word:
                event = self.probe_face_events.get((removal.sampled_facet_word, removal.retained_word))
        new_reduced = bool(edge.terminal and (self.problem.node_id, edge.terminal.closure_word)
                           not in self.shared_discovery.reduced_keys)
        before = len(self.discovery_events)
        outcome = ReducedMultiExitMCTS._step_reward(self, node, edge, path_actions, path_terminals, tree_edge)
        if event is not None:
            new_class = len(self.discovery_events) - before
            event["exit_arrivals"] += 1
            event["new_reduced_exit_arrivals"] += int(new_reduced)
            event["terminal_class_discoveries"] += new_class
            self.stats["fresh_probe_exit_arrivals"] += 1
            self.stats["fresh_probe_new_reduced_exit_arrivals"] += int(new_reduced)
            self.stats["fresh_probe_terminal_class_discoveries"] += new_class
        return outcome

    def _diagnostic_snapshot(self):
        snapshot = super()._diagnostic_snapshot()
        snapshot.update({key: self.stats[key] for key in SPARSE_STATS})
        snapshot.update({key: self.geometry.stats[key] for key in PROBE_STATS})
        return snapshot

    def result(self, *, target_ids=()):
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_reduced_sparse_outlet_fresh_geometry_v8"
        report = result["frontier_schedule"]
        report.update(corrector_cycle=list(SPARSE_CYCLE), quota_clock="global actual removals, tree and rollout shared",
                      fresh_face_scope="bounded new ridge LP probes, not bank reselection",
                      remaining_exit_scope="globally unvisited reduced closures only; known routes use regular policy",
                      bank_history_includes_unbanked_probe_proposals=True,
                      statistics={key: self.stats[key] for key in SPARSE_STATS})
        for row in report["bank_history"]:
            key = int(row["source_word_hex"], 16), int(row["retained_word_hex"], 16)
            row["first_banked_iteration"] = self.face_first_banked.get(key)
        result["fresh_geometry"] = {
            "outlet_quota_enabled": self.config.enable_corrector_outlet_quota,
            "probe_enabled": self.config.enable_fresh_geometry_probes,
            "cycle": list(SPARSE_CYCLE), "constraint_modes": list(PROBE_MODES),
            "lp_limit_per_probe": self.config.fresh_probe_lp_limit,
            "probe_rng_is_independent": True, "probe_queries_are_discoveries": False,
            "probe_scope": "current facet coordinates; least-probed conditioning vertices; no target witnesses",
            "normal_backoff_is_bypassed_only_by_bounded_probes": True,
            "successful_probe_can_grow_bank_by_one_up_to_existing_limit": True,
            "same_logical_iteration_and_local_correction_budgets": True,
            "same_lp_workload": False,
            "statistics": {key: self.stats[key] for key in SPARSE_STATS},
            "geometry_statistics": {key: self.geometry.stats[key] for key in PROBE_STATS},
            "probe_events": self.probe_events,
        }
        return result
