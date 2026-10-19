from __future__ import annotations

"""Lazy, shared Corrector face frontiers on the reduced-novelty search core.

Only geometry and actual traversal history are shared between tree and rollout.
Node values, reward computation, lifting, and continuation budgets remain v4.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .reduced_novelty_search import ReducedNoveltyConfig, ReducedNoveltyMCTS
from .reduced_subgroup_search import ReducedAction


FRONTIER_STAT_KEYS = (
    "frontier_sampled_bank_duplicates", "frontier_bank_replacements",
    "frontier_untried_bank_preserved", "frontier_tree_requests", "frontier_rollout_requests",
    "frontier_ridge_sampling_calls", "frontier_refresh_sampling_calls", "frontier_ridge_lp_attempts",
    "frontier_ridges_banked", "frontier_ridge_sampling_skipped", "frontier_lower_pair_trials",
    "frontier_lower_faces_banked", "frontier_cached_candidates_selected",
    "frontier_untried_candidates_selected", "frontier_certified_exit_reuses",
    "frontier_tree_attempts", "frontier_rollout_attempts", "frontier_terminal_arrivals",
    "frontier_repeated_exit_arrivals", "frontier_new_reduced_exit_arrivals",
    "frontier_source_return_arrivals",
)


@dataclass(frozen=True)
class ReducedFrontierConfig(ReducedNoveltyConfig):
    enable_shared_face_frontier: bool = True
    frontier_refresh_interval: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.frontier_refresh_interval <= 0:
            raise ValueError("frontier refresh interval must be positive")


class ReducedFrontierMCTS(ReducedNoveltyMCTS):
    def __init__(self, problem, config: ReducedFrontierConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.frontier_requests: Counter[int] = Counter()
        self.frontier_capacities: dict[int, int] = {}
        self.frontier_attempts: Counter[tuple[int, int]] = Counter()
        self.frontier_exits: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.frontier_novel_exits: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.frontier_traversals: Counter[tuple[int, int, int]] = Counter()

    def _frontier_key(self, facet, retained, rng):
        key = (facet, retained)
        attempts = self.frontier_attempts[key]
        novelty = len(self.frontier_novel_exits.get(key, ()))
        return (int(attempts > 0), attempts / math.sqrt(1 + novelty), rng.random())

    def _bank_face(self, bank, retained, facet, capacity, *, refresh):
        if retained in bank:
            self.stats["frontier_sampled_bank_duplicates"] += 1
            return False
        if len(bank) < capacity:
            bank.append(retained)
            return True
        if refresh:
            tried = [word for word in bank if self.frontier_attempts[(facet, word)]]
            if tried:
                evicted = max(tried, key=lambda word: self.frontier_attempts[(facet, word)])
                bank[bank.index(evicted)] = retained
                self.stats["frontier_bank_replacements"] += 1
                return True
        self.stats["frontier_untried_bank_preserved"] += 1
        return False

    def _corrector_actions(self, node, *, count, rng, persistent):
        if not self.config.enable_shared_face_frontier:
            return super()._corrector_actions(node, count=count, rng=rng, persistent=persistent)
        if node.state.corrections_used >= self.config.productive_max_corrections:
            return []
        facet = node.state.support_word
        self.frontier_requests[facet] += 1
        requests = self.frontier_requests[facet]
        self.stats["frontier_tree_requests" if persistent else "frontier_rollout_requests"] += 1
        widening = math.ceil(self.config.widening_k0 + self.config.widening_alpha
                             * requests ** self.config.widening_beta)
        capacity = min(self.config.corrector_face_bank_limit,
                       max(count, widening, self.frontier_capacities.get(facet, 0)))
        self.frontier_capacities[facet] = capacity
        ridges = self.corrector_ridge_bank[facet]
        lower = self.corrector_lower_face_bank[facet]
        refresh = requests % self.config.frontier_refresh_interval == 0
        grow = len(ridges) < capacity
        ridge_changed = False
        if grow or refresh:
            sample_count = min(max(4, count), capacity - len(ridges)) if grow else 4
            self.stats["frontier_ridge_sampling_calls"] += 1
            self.stats["frontier_refresh_sampling_calls"] += int(refresh)
            before = self.geometry.stats["ridge_lp_attempts"]
            sampled = self.geometry.sample_ridges(facet, count=sample_count, rng=rng)
            self.stats["frontier_ridge_lp_attempts"] += self.geometry.stats["ridge_lp_attempts"] - before
            for retained in sampled:
                if self._bank_face(ridges, retained, facet, capacity, refresh=refresh):
                    ridge_changed = True
                    self.stats["frontier_ridges_banked"] += 1
        else:
            self.stats["frontier_ridge_sampling_skipped"] += 1

        # Intersections are geometry-only candidates, never discovery events.
        if len(ridges) >= 2 and (ridge_changed or refresh or not lower):
            trials = (self.config.lower_face_pair_trials if persistent else
                      min(self.config.lower_face_pair_trials, max(4, count * 2)))
            for _ in range(trials):
                left, right = rng.sample(ridges, 2)
                retained = left & right
                self.stats["frontier_lower_pair_trials"] += 1
                if (retained and retained not in lower
                        and self.geometry.rank_oracle.rank(retained) == self.problem.ridge_rank - 1
                        and self._bank_face(lower, retained, facet, capacity, refresh=refresh)):
                    self.stats["frontier_lower_faces_banked"] += 1
                if len(lower) >= capacity:
                    break

        lower_quota = min(len(lower), max(1, round(count * self.config.lower_face_fraction)))
        ridge_quota = max(0, count - lower_quota)
        selected = [(word, "reduced_facet_ridge") for word in
                    sorted(ridges, key=lambda word: self._frontier_key(facet, word, rng))[:ridge_quota]]
        selected += [(word, "reduced_facet_lower_face") for word in
                     sorted(lower, key=lambda word: self._frontier_key(facet, word, rng))[:lower_quota]]
        for retained, _source in selected:
            self.stats["frontier_cached_candidates_selected"] += 1
            self.stats["frontier_untried_candidates_selected"] += int(
                not self.frontier_attempts[(facet, retained)])
        return [ReducedAction("remove", retained_word=retained, sampled_facet_word=facet, source=source)
                for retained, source in selected]

    def _filler_actions(self, node, *, sample_count, rng):
        if self.config.enable_shared_face_frontier:
            cached = self.geometry.ridge_exit_cache.get(node.state.support_word)
            if (cached is not None and cached.complete
                    and node.state.source_facet_word in cached.facet_words):
                self.stats["frontier_certified_exit_reuses"] += 1
        return super()._filler_actions(node, sample_count=sample_count, rng=rng)

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        if not self.config.enable_shared_face_frontier:
            return super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)
        if edge.action.kind == "remove":
            self.frontier_attempts[(node.state.support_word, edge.action.retained_word)] += 1
            self.stats["frontier_tree_attempts" if tree_edge else "frontier_rollout_attempts"] += 1
        terminal = edge.terminal
        globally_new = bool(terminal and (self.problem.node_id, terminal.closure_word)
                            not in self.shared_discovery.reduced_keys)
        outcome = super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)
        if terminal is not None and node.state.source_facet_word:
            removal = next((action for action in reversed(path_actions) if action.kind == "remove"), None)
            if removal is not None and removal.sampled_facet_word == node.state.source_facet_word:
                source, retained, destination = (node.state.source_facet_word,
                                                removal.retained_word, terminal.closure_word)
                traversal = (source, retained, destination)
                self.stats["frontier_terminal_arrivals"] += 1
                self.stats["frontier_repeated_exit_arrivals"] += int(bool(self.frontier_traversals[traversal]))
                self.frontier_traversals[traversal] += 1
                if destination != source:
                    self.frontier_exits[(source, retained)].add(destination)
                    if globally_new:
                        self.frontier_novel_exits[(source, retained)].add(destination)
                        self.stats["frontier_new_reduced_exit_arrivals"] += 1
                else:
                    self.stats["frontier_source_return_arrivals"] += 1
        return outcome

    def _diagnostic_snapshot(self):
        snapshot = super()._diagnostic_snapshot()
        snapshot.update({key: self.stats[key] for key in FRONTIER_STAT_KEYS})
        return snapshot

    def result(self, *, target_ids=()):
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_shared_lazy_reduced_face_frontier_v5"
        arrivals = self.stats["frontier_terminal_arrivals"]
        result["face_frontier"] = {
            "enabled": self.config.enable_shared_face_frontier,
            "scope": "one concrete reduced facet; shared geometry across tree and rollout",
            "capacity": "min(bank_limit, max(candidate_count, k0 + alpha * requests**beta))",
            "refresh_interval": self.config.frontier_refresh_interval,
            "reward_and_productivity_unchanged": True,
            "exhausted_bank_is_facet_completeness": False,
            "known_exits_remain_traversable": True,
            "untried_faces_are_not_evicted": True,
            "actual_attempt_keys": len(self.frontier_attempts),
            "actual_exit_keys": len(self.frontier_exits),
            "actual_traversal_keys": len(self.frontier_traversals),
            "ridge_bank_facets": len(self.corrector_ridge_bank),
            "ridge_bank_entries": sum(map(len, self.corrector_ridge_bank.values())),
            "lower_bank_entries": sum(map(len, self.corrector_lower_face_bank.values())),
            "repeat_exit_arrival_fraction": (self.stats["frontier_repeated_exit_arrivals"] / arrivals
                                              if arrivals else None),
            "statistics": {key: value for key, value in self.stats.items() if key.startswith("frontier_")},
        }
        return result
