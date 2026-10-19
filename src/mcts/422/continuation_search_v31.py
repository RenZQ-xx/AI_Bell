from __future__ import annotations

"""Full-space v30 with selected, resumable adjacency-sampling operators."""

import argparse
import json
import math
import random
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

from mcts.symmetry_game.ridge import RidgePivotResult

from .atlas import DEFAULT_HANDOFF_ROOT, IDENTITY_PATTERN_ID
from .continuation_search_v30 import Bell422EarlyNoopSearch, EarlyNoopSearchConfig
from .model import MacroAction
from .problem import TARGET_AFFINE_RANK, word_indices
from .recursive_adjacency import AdjacencyConfig, FacetCertificate
from .resumable_adjacency import ResumableAdjacency


@dataclass(frozen=True)
class FullspaceASConfig(EarlyNoopSearchConfig):
    enable_lazy_as: bool = True
    as_operator_probability: float = 0.15
    as_time_fraction: float = 0.10
    as_slice_seconds: float = 0.15
    as_min_slice_seconds: float = 0.05
    as_bootstrap_seconds: float = 0.15
    as_outputs_per_advance: int = 2
    as_max_live_jobs: int = 64
    as_max_active_jobs: int = 4
    as_cutoff_vertices: int = 16
    as_max_chain_outputs: int = 32

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.as_operator_probability <= 1 or not 0 < self.as_time_fraction < 1:
            raise ValueError("invalid AS operator probability/time fraction")
        for value in (self.as_slice_seconds, self.as_min_slice_seconds, self.as_bootstrap_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("AS time budgets must be finite and positive")
        if self.as_min_slice_seconds > self.as_slice_seconds:
            raise ValueError("minimum AS slice exceeds maximum slice")
        if min(self.as_outputs_per_advance, self.as_max_live_jobs, self.as_max_active_jobs) < 1:
            raise ValueError("AS output/job limits must be positive")
        if self.as_max_active_jobs > self.as_max_live_jobs:
            raise ValueError('AS active-job limit exceeds retained-job limit')
        self.adjacency_config()

    def adjacency_config(self):
        return AdjacencyConfig(cutoff_vertices=self.as_cutoff_vertices,
                               max_outputs=self.as_max_chain_outputs,
                               max_macro_seconds=self.as_slice_seconds)


class Bell422LazyASSearch(Bell422EarlyNoopSearch):
    def __init__(self, config: FullspaceASConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.adjacency = ResumableAdjacency(self.facet_oracle.points, config.adjacency_config())
        self.as_rng = random.Random(config.seed ^ 0xA5FACADE)
        self.as_policy_rng = random.Random(config.seed ^ 0x1A2B3C4D)
        self.as_jobs = OrderedDict()
        self.as_bank = {}
        self.as_pairs = {}
        self.as_consumed = Counter()
        self.as_observed_supports = set()
        self.as_executed_proofs = {}
        self.as_compute_seconds = 0.0
        self.as_budget_started = None

    def _as_allowance(self):
        now = time.perf_counter()
        if self.as_budget_started is None:
            self.as_budget_started = self.started_at if self.started_at else now
        elapsed = now-self.as_budget_started
        credit = self.config.as_bootstrap_seconds + self.config.as_time_fraction*elapsed - self.as_compute_seconds
        remaining = (math.inf if self.config.max_wall_seconds is None else
                     self.config.max_wall_seconds-(now-self.started_at))
        return min(self.config.as_slice_seconds, credit, remaining)

    def _seed_source_certificate(self, source):
        facet = self.facet_oracle.cache.get(source)
        if (source not in self.as_observed_supports or facet is None or not facet.valid
                or facet.support_word != source or facet.affine_rank != TARGET_AFFINE_RANK
                or facet.h_row is None):
            return False
        parent = self.adjacency.frame(self.adjacency.full_word)
        row = (facet.h_row[0], *(facet.h_row[axis+1] for axis in parent.axes))
        self.adjacency.certificates[(parent.word, source)] = FacetCertificate(parent.word, source, row)
        return True

    def _as_cache_output(self, source, proof):
        ridge = int(proof['retained_word_hex'], 16)
        target = int(proof['destination_word_hex'], 16)
        if (int(proof['chain_word_hex'][1], 16) != source or source == target
                or source & target != ridge):
            raise ValueError("inconsistent full-space AS endpoints")
        if (self.adjacency.certify(self.adjacency.full_word, target).word != target
                or self.adjacency.certify(source, ridge).word != ridge
                or self.rank_oracle.rank(ridge) != TARGET_AFFINE_RANK-1):
            raise ValueError("AS did not produce a closed original-space rank-78 ridge")
        pair = RidgePivotResult((source, target), 0, 0, 0)
        for endpoint in pair.facet_words:
            key = endpoint, ridge
            old = self.actions._ridge_cache.get(key)
            if old is not None and set(old.facet_words)-set(pair.facet_words):
                raise ValueError("AS pair conflicts with native ridge geometry")
            self.actions._ridge_cache[key] = pair
        self.as_pairs[ridge] = pair.facet_words
        self.as_bank.setdefault(source, {}).setdefault(ridge, proof)
        self.stats['as_shared_ridge_pairs'] += 1

    def _as_has_pending(self, source):
        return any(not self.as_consumed[(source, ridge)] for ridge in self.as_bank.get(source, {}))

    def _as_work_source(self, current):
        job = self.as_jobs.get(current)
        if job is not None and not job.done:
            return current
        active = [source for source, job in self.as_jobs.items()
                  if not job.done and not self._as_has_pending(source)]
        if len(active) < self.config.as_max_active_jobs:
            return current
        # Geometric service can finish an already-selected observed-facet job.
        # It never redirects the current MCTS edge to that other source.
        self.stats['as_cooperative_advances'] += 1
        return active[0]

    def _as_remove_action(self, node, *, tree):
        source = node.state.support_word
        if not self._seed_source_certificate(source):
            self.stats['as_unobserved_or_uncertified_sources_skipped'] += 1
            return None
        existing = {edge.action.remove_word for edge in node.edges.values()
                    if edge.action.kind == 'remove'} if tree else set()
        def pending():
            return [ridge for ridge in self.as_bank.get(source, {})
                    if not self.as_consumed[(source, ridge)] and (source & ~ridge) not in existing]
        ready = pending()
        if ready:
            self.stats['as_refill_avoided'] += 1
        else:
            allowance = self._as_allowance()
            if allowance < self.config.as_min_slice_seconds:
                self.stats['as_budget_deferrals'] += 1
                return None
            work_source = self._as_work_source(source)
            job = self.as_jobs.get(work_source)
            if job is None or job.done:
                job = self.adjacency.new_job(work_source)
                self.as_jobs[work_source] = job
            self.as_jobs.move_to_end(work_source)
            while len(self.as_jobs) > self.config.as_max_live_jobs:
                _, old = self.as_jobs.popitem(last=False)
                self.stats['as_unfinished_jobs_evicted'] += int(not old.done)
            started = time.perf_counter()
            outputs = self.adjacency.advance(job, self.as_rng,
                max_outputs=self.config.as_outputs_per_advance, seconds=allowance)
            for proof in outputs:
                self._as_cache_output(work_source, proof)
            self.as_compute_seconds += time.perf_counter()-started
            ready = pending()
        if not ready:
            return None
        ridge = min(ready, key=lambda word: (
            int(self.as_bank[source][word]['destination_word_hex'], 16) in self.as_observed_supports, word))
        removed = source & ~ridge
        return MacroAction('remove', IDENTITY_PATTERN_ID, tuple(word_indices(removed)), removed,
                           'fullspace_as_remove', 'certified_recursive_adjacency', remove_word=removed)

    def _as_completion(self, node):
        if node.state.role != 'filler' or node.rank != TARGET_AFFINE_RANK-1:
            return None
        pair = self.as_pairs.get(node.state.support_word)
        source = node.state.source_facet_word
        if pair is None or source not in pair:
            return None
        target = next(word for word in pair if word != source)
        outside = target & ~node.state.support_word
        add = outside & -outside
        if not add:
            raise ValueError('AS target has no completing vertex')
        action = MacroAction('add', IDENTITY_PATTERN_ID, tuple(word_indices(add)), add,
                             'fullspace_as_completion', 'certified_recursive_adjacency', add_word=add)
        candidate = self._edge_for(node, action)
        if candidate is None or candidate[1].facet is None or not candidate[1].facet.valid or candidate[1].state.support_word != target:
            raise ValueError('AS Filler completion disagrees with its certified endpoint')
        return candidate

    def _as_selected(self, node):
        if (not self.config.enable_lazy_as or node.state.role != 'corrector'
                or self._correction_limit_reached(node)):
            return False
        self.stats['as_operator_opportunities'] += 1
        if self._as_has_pending(node.state.support_word):
            self.stats['as_ready_exit_priority_selections'] += 1
            return True
        if self.as_policy_rng.random() >= self.config.as_operator_probability:
            return False
        self.stats['as_operator_selections'] += 1
        return True

    def _widen_once(self, node):
        if not self.config.enable_lazy_as:
            return super()._widen_once(node)
        if len(node.edges) < self._child_limit(node):
            candidate = self._as_completion(node)
            if candidate is None and (node.state.role != 'corrector' or 'stop' in node.edges):
                if self._as_selected(node):
                    action = self._as_remove_action(node, tree=True)
                    candidate = self._edge_for(node, action) if action is not None else None
            if candidate is not None:
                edge, moved = candidate
                if (edge.action.action_id not in node.edges
                        and edge.action.action_id not in node.retired_action_ids
                        and self._state_key(moved.state) not in node.child_keys):
                    return self._install_edge(node, edge, moved)
        return super()._widen_once(node)

    def _rollout_edge(self, node, policy):
        if not self.config.enable_lazy_as:
            return super()._rollout_edge(node, policy)
        completion = self._as_completion(node)
        if completion is not None:
            return completion
        if self._as_selected(node):
            rng = self.rollout_rngs[policy]
            saved = rng.getstate()
            if (node.state.corrections_used < self.config.max_corrections
                    and rng.random() > self.config.rollout_corrector_probability):
                return self._edge_for(node, self._stop_action())
            action = self._as_remove_action(node, tree=False)
            if action is not None:
                return self._edge_for(node, action)
            # Native fallback must not take a second continuation/stop draw.
            rng.setstate(saved)
        return super()._rollout_edge(node, policy)

    def _terminal_outcome(self, facet, *, path, depth, source):
        outcome = super()._terminal_outcome(facet, path=path, depth=depth, source=source)
        if self.config.enable_lazy_as and outcome.valid:
            self.as_observed_supports.add(facet.support_word)
            if path:
                previous = path[-1].node.state
                ridge, origin = previous.support_word, previous.source_facet_word
                proof = self.as_bank.get(origin, {}).get(ridge)
                if proof is not None and int(proof['destination_word_hex'], 16) == facet.support_word:
                    self.as_consumed[(origin, ridge)] += 1
                    self.as_executed_proofs[(origin, ridge, facet.support_word)] = proof
                    self.stats['as_actual_terminal_arrivals'] += 1
            if outcome.new_class:
                event = self.discovered[facet.class_id]
                event['as_actions_in_local_path'] = sum(
                    step.edge.action.source.startswith('fullspace_as_') for step in path)
        return outcome

    def result(self):
        result = super().result()
        result['algorithm'] = 'bell422_v30_fullspace_lazy_as_v31'
        result['features'].update(search_policy_unchanged_from_v29=not self.config.enable_lazy_as,
            fullspace_resumable_adjacency=self.config.enable_lazy_as,
            reduced_to_full_lift=False, v30_reward_and_counterfactual_credit_preserved=True)
        result['fullspace_as'] = {
            'config': {key:value for key,value in asdict(self.config).items() if key.startswith('as_')},
            'enabled': self.config.enable_lazy_as, 'geometry': self.adjacency.diagnostics(),
            'compute_seconds': self.as_compute_seconds,
            'banked_ridges': sum(map(len, self.as_bank.values())),
            'consumed_distinct_ridges': len(self.as_consumed),
            'executed_proofs': list(self.as_executed_proofs.values()),
            'unfinished_jobs': sum(not job.done for job in self.as_jobs.values()),
            'active_jobs_without_ready_exits': sum(not job.done and not self._as_has_pending(source)
                                                   for source, job in self.as_jobs.items()),
            'observed_concrete_facets': len(self.as_observed_supports),
            'candidate_generation_is_discovery': False, 'catalog_guided': False,
            'geometry_scope': 'original 256 vertices, affine dimension 80; true rank-78 ridges',
        }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=4222025)
    parser.add_argument('--reference-config', type=Path)
    parser.add_argument('--no-as', action='store_true')
    parser.add_argument('--handoff-root', type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite a search result')
    values = asdict(FullspaceASConfig())
    if args.reference_config:
        saved = json.loads(args.reference_config.read_text(encoding='utf-8'))['config']
        values.update({k:v for k,v in saved.items() if k in EarlyNoopSearchConfig.__dataclass_fields__})
    values.update(iterations=args.iterations, seed=args.seed, enable_lazy_as=not args.no_as)
    engine = Bell422LazyASSearch(FullspaceASConfig(**values), handoff_root=args.handoff_root)
    result = engine.run()
    engine._write_result(result, args.output)
    print(json.dumps({'output':str(args.output), 'runtime':result['runtime'],
                      'classes':result['coverage']['discovered_class_count'],
                      'as_compute_seconds':result['fullspace_as']['compute_seconds']}))


if __name__ == '__main__':
    main()
