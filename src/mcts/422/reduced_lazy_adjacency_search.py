from __future__ import annotations

"""Selected, resumable AS operators on the unchanged v6 reward/rollout core."""

import math
import random
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass

from .reduced_adjacency_search import ReducedAdjacencyConfig, ReducedAdjacencyMCTS
from .reduced_multi_exit_search import ReducedMultiExitMCTS
from .reduced_ridge_exit_search import CertifiedRidgeExits
from .reduced_subgroup_search import ReducedAction, ReducedNode
from .resumable_adjacency import ResumableAdjacency


@dataclass(frozen=True)
class ReducedLazyAdjacencyConfig(ReducedAdjacencyConfig):
    as_operator_probability: float = 0.20
    as_time_fraction: float = 0.20
    as_slice_seconds: float = 0.04
    as_min_slice_seconds: float = 0.02
    as_bootstrap_seconds: float = 0.04
    as_outputs_per_advance: int = 2
    as_max_live_jobs: int = 256

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.as_operator_probability <= 1 or not 0 < self.as_time_fraction < 1:
            raise ValueError("invalid AS probability/time fraction")
        for value in (self.as_slice_seconds, self.as_min_slice_seconds, self.as_bootstrap_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("AS time slices must be finite and positive")
        if self.as_min_slice_seconds > self.as_slice_seconds:
            raise ValueError("AS minimum slice exceeds maximum slice")
        if min(self.as_outputs_per_advance, self.as_max_live_jobs) < 1:
            raise ValueError("AS output/job limits must be positive")


class ReducedLazyAdjacencyMCTS(ReducedAdjacencyMCTS):
    def __init__(self, problem, config: ReducedLazyAdjacencyConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.adjacency = ResumableAdjacency(problem.irredundant_points, config.adjacency_config())
        self.as_policy_rng = random.Random(config.seed ^ 0x1A2B3C4D)
        self.as_jobs = OrderedDict()
        self.as_consumed = Counter()
        self.as_compute_seconds = 0.0
        self.as_budget_started = None

    def _corrector_actions(self, node, *, count, rng, persistent):
        # Native candidate generation never pays for an unselected AS operator.
        return ReducedMultiExitMCTS._corrector_actions(
            self, node, count=count, rng=rng, persistent=persistent)

    def _allowance(self):
        now = time.perf_counter()
        if self.as_budget_started is None:
            self.as_budget_started = self.started_at if self.started_at > 0 else now
        credit = (self.config.as_bootstrap_seconds
                  + self.config.as_time_fraction * (now-self.as_budget_started)
                  - self.as_compute_seconds)
        return min(self.config.as_slice_seconds, credit, self.search_deadline-now)

    def _install_proof(self, source, proof):
        retained = int(proof["retained_word_hex"], 16)
        destination = int(proof["destination_word_hex"], 16)
        if int(proof["chain_word_hex"][1], 16) != source or source & destination != retained:
            raise ValueError("AS cache proof/source mismatch")
        first, second = self.geometry.validate(source), self.geometry.validate(destination)
        if (not first.valid or first.closure_word != source or not second.valid
                or second.closure_word != destination or source == destination
                or self.geometry.rank_oracle.rank(retained) != self.problem.ridge_rank):
            raise ValueError("AS cache endpoints are not certified reduced ridge neighbors")
        pair = CertifiedRidgeExits("complete", (source, destination), retained)
        existing = self.geometry.ridge_exit_cache.get(retained)
        if existing is not None and existing.complete and set(existing.facet_words) != set(pair.facet_words):
            raise ValueError("inconsistent certified ridge cache")
        self.geometry.ridge_exit_cache[retained] = pair
        bank = self.as_bank.setdefault(source, {})
        if retained not in bank:
            bank[retained] = proof
            self.stats["lazy_as_new_banked_ridges"] += 1
        self.stats["lazy_as_imported_exit_pairs"] += 1

    def _pending(self, source, node, tree):
        existing = {edge.action.retained_word for edge in node.edges.values()} if tree else set()
        return [retained for retained in self.as_bank.get(source, {})
                if not self.as_consumed[(source, retained)] and retained not in existing]

    def _as_action(self, node, *, tree):
        if (not self.config.enable_recursive_adjacency or node.state.role != "corrector"
                or node.state.corrections_used >= self.config.productive_max_corrections):
            return None
        self.stats["lazy_as_operator_opportunities"] += 1
        if self.as_policy_rng.random() >= self.config.as_operator_probability:
            return None
        self.stats["lazy_as_operator_selections"] += 1
        source = node.state.support_word
        pending = self._pending(source, node, tree)
        if pending:
            self.stats["lazy_as_refill_avoided"] += 1
        else:
            allowance = self._allowance()
            if allowance < self.config.as_min_slice_seconds:
                self.stats["lazy_as_budget_deferrals"] += 1
                return None
            job = self.as_jobs.get(source)
            if job is None or job.done:
                job = self.adjacency.new_job(source)
                self.as_jobs[source] = job
            self.as_jobs.move_to_end(source)
            while len(self.as_jobs) > self.config.as_max_live_jobs:
                _, evicted = self.as_jobs.popitem(last=False)
                self.stats["lazy_as_unfinished_jobs_evicted"] += int(not evicted.done)
            started = time.perf_counter()
            proofs = self.adjacency.advance(
                job, self.as_rng, max_outputs=self.config.as_outputs_per_advance,
                seconds=allowance, deadline=self.search_deadline)
            for proof in proofs:
                self._install_proof(source, proof)
            self.as_compute_seconds += time.perf_counter()-started
            pending = self._pending(source, node, tree)
        if not pending:
            self.stats["lazy_as_selected_without_ready_exit"] += 1
            return None
        retained = min(pending, key=lambda word: (
            int(self.as_bank[source][word]["destination_word_hex"], 16) in self.discovered_reduced_facets,
            word))
        return ReducedAction("remove", retained_word=retained,
                             sampled_facet_word=source, source="recursive_as_ridge")

    def _widen(self, node):
        if len(node.edges) < self._child_limit(node):
            action = self._as_action(node, tree=True)
            if action is not None:
                built = self._edge(node, action)
                if built is not None:
                    edge, transition = built
                    if not any(old.child_key == edge.child_key for old in node.edges.values()):
                        node.edges[action.action_id] = edge
                        self.stats["tree_expansions"] += 1
                        self.stats["tree_action_remove"] += 1
                        self.stats["lazy_as_tree_expansions"] += 1
                        return edge, self._node(transition)
        return super()._widen(node)

    def _rollout_edge(self, node, policy):
        action = self._as_action(node, tree=False)
        if action is not None:
            built = self._edge(node, action)
            if built is not None:
                edge, transition = built
                return edge, ReducedNode(transition.state, transition.rank)
        return super()._rollout_edge(node, policy)

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        if edge.terminal is not None and node.state.source_facet_word:
            source, retained = node.state.source_facet_word, node.state.support_word
            proof = self.as_bank.get(source, {}).get(retained)
            if proof is not None and int(proof["destination_word_hex"], 16) == edge.terminal.closure_word:
                self.as_consumed[(source, retained)] += 1
                self.stats["lazy_as_cached_filler_completions"] += 1
        return super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)

    def result(self, *, target_ids=()):
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_v6_selected_resumable_adjacency_v2"
        recursive = result["recursive_adjacency"]
        recursive.pop("proposal_interval", None)
        recursive["lazy_operator"] = {
            "probability": self.config.as_operator_probability,
            "target_compute_time_fraction": self.config.as_time_fraction,
            "slice_seconds": self.config.as_slice_seconds,
            "minimum_slice_seconds": self.config.as_min_slice_seconds,
            "outputs_per_advance": self.config.as_outputs_per_advance,
            "compute_seconds_including_cache_import": self.as_compute_seconds,
            "time_cap_scope": "AS work only; atomic geometry may overshoot a slice",
            "live_jobs": len(self.as_jobs),
            "unfinished_jobs": sum(not job.done for job in self.as_jobs.values()),
            "unconsumed_banked_ridges": sum(
                not self.as_consumed[(source, retained)]
                for source, bank in self.as_bank.items() for retained in bank),
            "consumed_distinct_ridges": len(self.as_consumed),
            "generation_on_operator_selection_only": True,
            "resumes_chain_leaf_enumeration_and_rotations": True,
            "ridge_pair_cache_shared_with_filler": True,
            "adaptive_operator_reward_enabled": False,
        }
        return result
