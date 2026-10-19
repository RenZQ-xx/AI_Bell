from __future__ import annotations

"""Corrector outlet quotas and fair frontier-aware continuation, isolated as v7."""

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import islice

from .reduced_multi_exit_search import ReducedMultiExitConfig, ReducedMultiExitMCTS
from .reduced_subgroup_search import ReducedAction


CORRECTOR_CYCLE = ("remaining_exit", "untried_face", "fresh_face")
ANCHOR_CYCLE = ("fair", "frontier", "underexplored")
SCHEDULE_STATS = (
    "schedule_remaining_exit_attempts", "schedule_untried_face_attempts",
    "schedule_fresh_face_attempts", "schedule_remaining_exit_fulfilled",
    "schedule_untried_face_fulfilled", "schedule_fresh_face_fulfilled",
    "schedule_reserved_candidates", "schedule_existing_edge_reservations",
    "schedule_fair_entries", "schedule_frontier_entries", "schedule_underexplored_entries",
    "schedule_anchor_candidates_scored", "schedule_cached_faces_indexed",
)


@dataclass(frozen=True)
class ReducedFrontierScheduleConfig(ReducedMultiExitConfig):
    enable_corrector_outlet_quota: bool = True
    enable_frontier_anchor_schedule: bool = True
    frontier_anchor_candidate_window: int = 32

    def __post_init__(self):
        super().__post_init__()
        if self.frontier_anchor_candidate_window < 1:
            raise ValueError("anchor candidate window must be positive")
        if ((self.enable_corrector_outlet_quota or self.enable_frontier_anchor_schedule)
                and not self.enable_shared_face_frontier):
            raise ValueError("frontier scheduling requires shared actual face traversal tracking")


class ReducedFrontierScheduleMCTS(ReducedMultiExitMCTS):
    def __init__(self, problem, config: ReducedFrontierScheduleConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.corrector_schedule_attempts = Counter()
        self.source_removals = Counter()
        self.cached_source_faces = defaultdict(set)
        self.fresh_source_faces = defaultdict(set)
        self.face_first_generated = {}
        self.face_first_attempted = {}
        self.frontier_anchor_ring = deque()
        self.frontier_anchor_members = set()
        self._tree_quota_choices = {}

    def _cached_outlets(self, source, retained):
        result = (self.geometry.lower_exit_cache.get(retained)
                  or self.geometry.ridge_exit_cache.get(retained))
        if result is None or not result.complete or source not in result.facet_words:
            return ()
        return tuple(word for word in result.facet_words if word != source)

    def _outlet_frontier(self, source, retained):
        outlets = self._cached_outlets(source, retained)
        unknown = tuple(word for word in outlets
                        if (self.problem.node_id, word) not in self.shared_discovery.reduced_keys)
        untraversed = tuple(word for word in outlets
                            if not self.frontier_traversals[(source, retained, word)])
        return unknown, untraversed

    def _bank_words(self, source):
        return set(self.corrector_ridge_bank.get(source, ())) | set(self.corrector_lower_face_bank.get(source, ()))

    def _remember_cached_face(self, source, retained):
        if not source or not self._cached_outlets(source, retained):
            return
        if retained not in self.cached_source_faces[source]:
            self.cached_source_faces[source].add(retained)
            self.stats["schedule_cached_faces_indexed"] += 1
        if source not in self.frontier_anchor_members:
            self.frontier_anchor_members.add(source)
            self.frontier_anchor_ring.append(source)

    def _filler_actions(self, node, *, sample_count, rng):
        actions = super()._filler_actions(node, sample_count=sample_count, rng=rng)
        if self.config.enable_corrector_outlet_quota or self.config.enable_frontier_anchor_schedule:
            self._remember_cached_face(node.state.source_facet_word, node.state.support_word)
        return actions

    def _corrector_actions(self, node, *, count, rng, persistent):
        source = node.state.support_word
        before = self._bank_words(source) if self.config.enable_corrector_outlet_quota else set()
        actions = super()._corrector_actions(node, count=count, rng=rng, persistent=persistent)
        if self.config.enable_corrector_outlet_quota:
            added = self._bank_words(source) - before
            self.fresh_source_faces[source].update(added)
            for retained in added:
                self.face_first_generated.setdefault((source, retained), self.current_iteration)
        return actions

    def _quota_kind(self, source, *, rollout):
        # Quotas advance only when a removal is actually taken, not on lookahead.
        return CORRECTOR_CYCLE[self.corrector_schedule_attempts[(source, rollout)] % len(CORRECTOR_CYCLE)]

    def _quota_faces(self, source, kind):
        if kind == "remaining_exit":
            return [word for word in self.cached_source_faces.get(source, ())
                    if any(self._outlet_frontier(source, word))]
        bank = self._bank_words(source)
        if kind == "fresh_face":
            bank &= self.fresh_source_faces.get(source, set())
        return [word for word in bank if not self.frontier_attempts[(source, word)]]

    def _reserved_action(self, node, *, rollout, rng, allowed=None):
        if node.state.corrections_used >= self.config.productive_max_corrections:
            return None
        source = node.state.support_word
        kind = self._quota_kind(source, rollout=rollout)
        words = self._quota_faces(source, kind)
        if allowed is not None:
            words = [word for word in words if word in allowed]
        if not words:
            return None

        def priority(word):
            unknown, untraversed = self._outlet_frontier(source, word)
            return (int(not unknown), -len(unknown), -len(untraversed),
                    self.frontier_attempts[(source, word)], rng.random())

        retained = min(sorted(words), key=priority)
        lower = self.geometry.rank_oracle.rank(retained) == self.problem.facet_rank - 2
        return ReducedAction("remove", retained_word=retained, sampled_facet_word=source,
                             source="reduced_facet_lower_face" if lower else "reduced_facet_ridge")

    def _generate_actions(self, node, *, rollout, policy, rng):
        regular = super()._generate_actions(node, rollout=rollout, policy=policy, rng=rng)
        if not self.config.enable_corrector_outlet_quota or node.state.role != "corrector":
            return regular
        reserved = self._reserved_action(node, rollout=rollout, rng=rng)
        if reserved is None:
            return regular
        self.stats["schedule_reserved_candidates"] += 1
        return [reserved] + [action for action in regular if action.action_id != reserved.action_id]

    def _widen(self, node):
        if not self.config.enable_corrector_outlet_quota or node.state.role != "corrector":
            return super()._widen(node)
        self._tree_quota_choices.pop(node.state.key, None)
        reserved = self._reserved_action(node, rollout=False, rng=self.rng)
        if reserved is not None:
            existing = next((edge for edge in node.edges.values()
                             if edge.action.retained_word == reserved.retained_word), None)
            if existing is not None:
                # Returning an existing edge as an expansion would falsely end tree selection.
                self._tree_quota_choices[node.state.key] = reserved.retained_word
                self.stats["schedule_existing_edge_reservations"] += 1
                return None
            if len(node.edges) < self._child_limit(node):
                node.pending_actions[:] = [reserved] + [action for action in node.pending_actions
                                                       if action.action_id != reserved.action_id]
        return super()._widen(node)

    def _select(self, node):
        if not self.config.enable_corrector_outlet_quota or node.state.role != "corrector":
            return super()._select(node)
        planned = self._tree_quota_choices.pop(node.state.key, None)
        if planned is not None:
            return next(edge for edge in node.edges.values() if edge.action.retained_word == planned)
        allowed = {edge.action.retained_word for edge in node.edges.values()}
        reserved = self._reserved_action(node, rollout=False, rng=self.rng, allowed=allowed)
        if reserved is not None:
            return next(edge for edge in node.edges.values()
                        if edge.action.retained_word == reserved.retained_word)
        return super()._select(node)

    def _source_frontier(self, source):
        unknown, routes = set(), 0
        for retained in self.cached_source_faces.get(source, ()):
            unseen, untraversed = self._outlet_frontier(source, retained)
            unknown.update(unseen)
            routes += len(untraversed)
        untried = sum(not self.frontier_attempts[(source, word)] for word in self._bank_words(source))
        return len(unknown), routes, untried

    def _anchor_candidates(self):
        window = self.config.frontier_anchor_candidate_window
        candidates = list(islice(self.anchor_queue, window))
        for _ in range(min(window, len(self.frontier_anchor_ring))):
            word = self.frontier_anchor_ring.popleft()
            self.frontier_anchor_ring.append(word)
            if word in self.anchors:
                candidates.append(word)
        return list(dict.fromkeys(candidates))

    def _choose_anchor(self, kind):
        if kind == "fair":
            return self.anchor_queue[0]
        candidates = self._anchor_candidates()
        self.stats["schedule_anchor_candidates_scored"] += len(candidates)
        if kind == "underexplored":
            return min(candidates, key=lambda word: (
                self.anchors[word].starts, self.source_removals[word],
                self.anchors[word].observed_iteration, word,
            ))

        def score(word):
            unknown, routes, untried = self._source_frontier(word)
            starts = self.anchors[word].starts
            scale = math.sqrt(1 + starts)
            return (unknown / scale, routes / scale, untried / scale,
                    -starts, -self.source_removals[word], -self.anchors[word].observed_iteration, -word)

        return max(candidates, key=score)

    def _iteration_start(self, iteration):
        if (self.config.enable_frontier_anchor_schedule and self.config.enable_local_continuation
                and self.anchor_queue and iteration % self.config.continuation_interval == 0):
            kind = ANCHOR_CYCLE[self.stats["continuation_episodes"] % len(ANCHOR_CYCLE)]
            word = self._choose_anchor(kind)
            self.stats[f"schedule_{kind}_entries"] += 1
            # The parent still owns proof boundaries, budget resets and episode bookkeeping.
            self.anchor_queue.remove(word)
            self.anchor_queue.appendleft(word)
        return super()._iteration_start(iteration)

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        if edge.action.kind == "remove":
            source, retained = node.state.support_word, edge.action.retained_word
            self.source_removals[source] += 1
            if self.config.enable_corrector_outlet_quota:
                kind = self._quota_kind(source, rollout=not tree_edge)
                eligible = retained in self._quota_faces(source, kind)
                self.stats[f"schedule_{kind}_attempts"] += 1
                self.stats[f"schedule_{kind}_fulfilled"] += int(eligible)
                self.corrector_schedule_attempts[(source, not tree_edge)] += 1
                self.face_first_attempted.setdefault((source, retained), self.current_iteration)
                self.fresh_source_faces[source].discard(retained)
        return super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)

    def _diagnostic_snapshot(self):
        snapshot = super()._diagnostic_snapshot()
        snapshot.update({key: self.stats[key] for key in SCHEDULE_STATS})
        return snapshot

    def result(self, *, target_ids=()):
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_reduced_outlet_quota_fair_frontier_schedule_v7"
        result["continuation"]["selection"] = ("fair / cached-frontier / underexplored, bounded candidate windows"
                                              if self.config.enable_frontier_anchor_schedule else
                                              result["continuation"]["selection"])
        result["frontier_schedule"] = {
            "corrector_quota_enabled": self.config.enable_corrector_outlet_quota,
            "anchor_schedule_enabled": self.config.enable_frontier_anchor_schedule,
            "corrector_cycle": list(CORRECTOR_CYCLE), "anchor_cycle": list(ANCHOR_CYCLE),
            "quota_clock": "actual removals, per concrete source and tree/rollout context",
            "fresh_face_scope": "newly banked unattempted faces; unchanged v6 sampler/backoff",
            "remaining_exit_scope": "globally unseen closures first, then untaken source/face/exit routes",
            "anchor_candidate_window": self.config.frontier_anchor_candidate_window,
            "anchor_scope": "within one reduced node; no cross-subgroup lane scheduling",
            "reward_productivity_budget_lifting_unchanged": True,
            "selection_uses_lifting_or_catalog_labels": False,
            "known_routes_are_pruned": False,
            "statistics": {key: self.stats[key] for key in SCHEDULE_STATS},
            "sources": [], "cached_faces": [], "bank_history": [],
        }
        report = result["frontier_schedule"]
        for word, anchor in self.anchors.items():
            unknown, routes, untried = self._source_frontier(word)
            report["sources"].append({
                "word_hex": f"0x{word:x}", "starts": anchor.starts,
                "actual_removals": self.source_removals[word],
                "cached_faces": len(self.cached_source_faces.get(word, ())),
                "unknown_destinations": unknown, "untaken_routes": routes, "untried_bank_faces": untried,
                "ridge_bank_word_hex": [f"0x{face:x}" for face in self.corrector_ridge_bank.get(word, ())],
                "lower_bank_word_hex": [f"0x{face:x}" for face in self.corrector_lower_face_bank.get(word, ())],
            })
        for source, faces in self.cached_source_faces.items():
            for retained in sorted(faces):
                report["cached_faces"].append({
                    "source_word_hex": f"0x{source:x}", "retained_word_hex": f"0x{retained:x}",
                    "actual_removals": self.frontier_attempts[(source, retained)],
                    "outlets": [{"word_hex": f"0x{word:x}",
                                 "actual_traversals": self.frontier_traversals[(source, retained, word)],
                                 "globally_visited": (self.problem.node_id, word) in self.shared_discovery.reduced_keys}
                                for word in self._cached_outlets(source, retained)],
                })
        for (source, retained), first in self.face_first_generated.items():
            report["bank_history"].append({
                "source_word_hex": f"0x{source:x}", "retained_word_hex": f"0x{retained:x}",
                "first_generated_iteration": first,
                "first_attempted_iteration": self.face_first_attempted.get((source, retained)),
                "actual_removals": self.frontier_attempts[(source, retained)],
            })
        return result
