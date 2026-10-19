from __future__ import annotations

"""v25 with isolated v29/v30/v31 mechanisms; no answer-guided actions."""

import argparse
import importlib
import math
import random
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from mcts.subgroup_patterns import IDENTITY_PATTERN_ID, canonical_support_word

from ..model import MacroAction, SymmetryState
from ..search import GroupGameConfig, SingleTreeGroupGame

adjacency_module = importlib.import_module('mcts.422.recursive_adjacency')
resumable_module = importlib.import_module('mcts.422.resumable_adjacency')


@dataclass(frozen=True)
class TransferConfig(GroupGameConfig):
    adaptive_corrector_replay: bool = True
    ridge_pivot: bool = True
    eager_ridge_focus: bool = True
    hierarchical_pattern_policy: bool = True
    directed_pattern_relations: bool = True
    learn_relation_values: bool = True
    retain_pattern_embeddings: bool = True
    embedding_relation_policy: bool = True
    leaf_value_shaping: bool = False
    ridge_max_candidates: int = 32768
    ridge_batch_size: int = 256
    ridge_focus_batches: int = 48
    max_pattern_analyses_per_class: int = 4
    enable_local_continuation: bool = True
    enable_early_noop_guard: bool = True
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
            raise ValueError('invalid AS probability/time credit')
        if not all(math.isfinite(v) and v > 0 for v in (
            self.as_slice_seconds, self.as_min_slice_seconds, self.as_bootstrap_seconds)):
            raise ValueError('AS slices must be finite and positive')
        if self.as_min_slice_seconds > self.as_slice_seconds:
            raise ValueError('minimum AS slice exceeds maximum')
        if not 1 <= self.as_max_active_jobs <= self.as_max_live_jobs or self.as_outputs_per_advance < 1:
            raise ValueError('invalid AS job/output limits')
        self.adjacency_config()

    def adjacency_config(self):
        return adjacency_module.AdjacencyConfig(cutoff_vertices=self.as_cutoff_vertices,
            max_outputs=self.as_max_chain_outputs, max_macro_seconds=self.as_slice_seconds)


@dataclass
class ObservedAnchor:
    class_id: int
    word: int
    witness: tuple[MacroAction, ...]
    resets: tuple[int, ...] = ()
    patterns: set[str] = field(default_factory=set)
    starts: int = 0


def witness_cost(actions):
    return sum(a.kind in ('remove', 'rewrite') for a in actions), len(actions)


class TransferredGroupGame(SingleTreeGroupGame):
    def __init__(self, config: TransferConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.transfer_stats = Counter()
        self.continuation_anchors = defaultdict(dict)
        self.continuation_starts = Counter()
        self.anchor_pattern_starts = Counter()
        self.episode_origin = None
        self.episode_prefix = ()
        self.episode_resets = ()
        self.discovery_origins = {}
        self.as_rng = random.Random(config.seed ^ 0xA5FACADE)
        self.as_policy_rng = random.Random(config.seed ^ 0x1A2B3C4D)
        self.adjacency = resumable_module.ResumableAdjacency(self.scorer.points, config.adjacency_config())
        self.as_jobs = OrderedDict()
        self.as_bank = {}
        self.as_pairs = {}
        self.as_consumed = Counter()
        self.as_executed_proofs = {}
        self.as_compute_seconds = 0.0
        self.as_canonical = {}

    def _validation_witness(self, path):
        return self.episode_prefix + super()._validation_witness(path)

    def _iteration_start(self, iteration):
        self.episode_origin, self.episode_prefix, self.episode_resets = None, (), ()
        if not self.config.enable_local_continuation:
            return super()._iteration_start(iteration)
        self.frontier_replay_start = None
        self.frontier_replay_keys.clear()
        due, _reason = self._corrector_replay_due(iteration)
        if not due or not self.continuation_anchors:
            self.transfer_stats['root_episodes'] += 1
            return self.root, ()
        family = min(self.continuation_anchors, key=lambda c: (self.continuation_starts[c], c))
        anchor = min(self.continuation_anchors[family].values(),
            key=lambda a: (a.starts, witness_cost(a.witness), a.word))
        pattern = min(anchor.patterns,
            key=lambda p: (self.anchor_pattern_starts[anchor.word, p], p))
        node = self._node(SymmetryState(anchor.word, 'corrector', 0, pattern), 25)
        node.facet_word = anchor.word
        node.context_pattern_ids.update(anchor.patterns)
        if self._add_stop_edge(node) is not None:
            self.tree_expansions += 1
        self.episode_origin, self.episode_prefix = anchor, anchor.witness
        self.episode_resets = tuple(sorted(set((*anchor.resets, len(anchor.witness)))))
        self.continuation_starts[family] += 1
        self.anchor_pattern_starts[anchor.word, pattern] += 1
        anchor.starts += 1
        self.transfer_stats['continuation_episodes'] += 1
        self.transfer_stats['historical_steps_saved'] += len(anchor.witness)
        self.transfer_stats['historical_corrections_reset'] += witness_cost(anchor.witness)[0]
        return node, ()

    def _observe_terminal(self, node, *, path):
        old_count = len(self.discovery_events)
        outcome = super()._observe_terminal(node, path=path)
        if outcome.valid and outcome.class_id is not None:
            word = node.facet_word
            witness = self._validation_witness(path)
            family = self.continuation_anchors[outcome.class_id]
            anchor = family.get(word)
            if anchor is None:
                anchor = family[word] = ObservedAnchor(outcome.class_id, word, witness, self.episode_resets)
            elif witness and witness_cost(witness) < witness_cost(anchor.witness):
                anchor.witness = witness
                anchor.resets = self.episode_resets
            anchor.patterns.update(node.context_pattern_ids)
        if len(self.discovery_events) > old_count:
            origin = self.episode_origin
            self.discovery_origins[old_count + 1] = {
                'episode_kind': 'continuation' if origin else 'root_filler',
                'historical_path_length': len(self.episode_prefix),
                'historical_resets': list(self.episode_resets),
                'class_id': origin.class_id if origin else None,
                'support_word_hex': hex(origin.word) if origin else None,
                'pattern_id': path[0].node.state.context_pattern_id if path else node.state.context_pattern_id,
                'local_path': [s.edge.action.to_dict() for s in path],
            }
        if self.config.enable_lazy_as and outcome.valid and path:
            previous = path[-1].node.state
            key = previous.repair_source_word, previous.support_word
            proof = self.as_bank.get(key[0], {}).get(key[1])
            if proof and int(proof['destination_word_hex'], 16) == node.facet_word:
                self.as_consumed[key] += 1
                self.as_executed_proofs[(*key, node.facet_word)] = proof
                self.transfer_stats['as_actual_arrivals'] += 1
        return outcome

    def _materialize_action(self, node, action):
        moved = super()._materialize_action(node, action)
        if moved is None:
            return None
        state, rank = moved
        if (self.config.enable_early_noop_guard and action.kind in ('remove', 'rewrite')
                and node.facet_word in self.facet_bank.anchors and rank == 25
                and state.support_word != node.state.support_word
                and not state.support_word & ~node.state.support_word):
            node.context_pattern_ids.add(action.pattern_id)
            self.transfer_stats['certified_noop_skips'] += 1
            return None
        if action.source == 'fullspace_as_remove':
            state = replace(state, repair_source_word=action.source_facet_word)
        elif action.source == 'fullspace_as_completion':
            target = self.as_pairs[node.state.repair_source_word, node.state.support_word]
            if rank != 25 or state.support_word & ~target or self._rank(target) != 25:
                raise ValueError('AS completion disagrees with certified endpoint')
            state = replace(state, support_word=target)
        return state, rank

    def _cache_as_output(self, source, proof):
        ridge = int(proof['retained_word_hex'], 16)
        target = int(proof['destination_word_hex'], 16)
        if (source & target != ridge or source == target
                or int(proof['chain_word_hex'][1], 16) != source
                or self._rank(ridge) != 24
                or self.adjacency.certify(self.adjacency.full_word, target).word != target
                or self.adjacency.certify(source, ridge).word != ridge):
            raise ValueError('invalid original-space AS pair')
        self.as_pairs[source, ridge] = target
        self.as_pairs[target, ridge] = source
        self.as_bank.setdefault(source, {}).setdefault(ridge, proof)

    def _as_pending(self, source):
        return any(not self.as_consumed[source, r] for r in self.as_bank.get(source, {}))

    def _as_selected(self, node):
        if (not self.config.enable_lazy_as or node.state.role != 'corrector'
                or self._correction_limit_reached(node)
                or node.facet_word not in self.facet_bank.anchors):
            return False
        return self._as_pending(node.facet_word) or self.as_policy_rng.random() < self.config.as_operator_probability

    def _as_remove(self, node, *, persistent):
        source = node.facet_word
        if source not in self.facet_bank.anchors:
            return None
        def ready():
            return [r for r in self.as_bank.get(source, {}) if not self.as_consumed[source, r]]
        pending = ready()
        if not pending:
            credit = (self.config.as_bootstrap_seconds
                + self.config.as_time_fraction * (time.perf_counter() - self.started_at)
                - self.as_compute_seconds)
            if credit < self.config.as_min_slice_seconds:
                self.transfer_stats['as_budget_deferrals'] += 1
                return None
            job = self.as_jobs.get(source)
            active = [w for w, j in self.as_jobs.items() if not j.done and not self._as_pending(w)]
            work = (active[0] if (job is None or job.done)
                    and len(active) >= self.config.as_max_active_jobs else source)
            started = time.perf_counter()
            job = self.as_jobs.get(work)
            if job is None or job.done:
                job = self.adjacency.new_job(work)
                self.as_jobs[work] = job
            self.as_jobs.move_to_end(work)
            while len(self.as_jobs) > self.config.as_max_live_jobs:
                _, old = self.as_jobs.popitem(last=False)
                self.transfer_stats['as_unfinished_jobs_evicted'] += int(not old.done)
            for proof in self.adjacency.advance(job, self.as_rng,
                    max_outputs=self.config.as_outputs_per_advance,
                    seconds=min(credit, self.config.as_slice_seconds)):
                self._cache_as_output(work, proof)
            self.as_compute_seconds += time.perf_counter() - started
            pending = ready()
        for ridge in sorted(pending):
            removed = node.state.support_word & ~ridge
            added = ridge & ~node.state.support_word
            if not removed:
                continue
            action = MacroAction('rewrite' if added else 'remove', IDENTITY_PATTERN_ID, (),
                removed | added, removed | added, 'fullspace_as_remove', 'certified_adjacency',
                (removed | added).bit_count(), removed.bit_count() + added.bit_count(),
                remove_word=removed, add_word=added, source_facet_word=source)
            if not persistent or (action.action_id not in node.edges
                    and action.action_id not in node.retired_action_ids):
                return self._ephemeral_edge(node, action)
        return None

    def _as_completion(self, node):
        if node.state.role != 'filler' or node.rank != 24:
            return None
        source, ridge = node.state.repair_source_word, node.state.support_word
        target = self.as_pairs.get((source, ridge))
        if target is None:
            return None
        outside = target & ~ridge
        added = outside & -outside
        action = MacroAction('add', IDENTITY_PATTERN_ID, (), added, added,
            'fullspace_as_completion', 'certified_adjacency', 1, 1,
            add_word=added, source_facet_word=source)
        candidate = self._ephemeral_edge(node, action)
        if candidate is not None:
            if target not in self.as_canonical:
                self.as_canonical[target] = canonical_support_word(target)
            key = self.facet_bank.shared.exit_key(source, ridge, added)
            self.facet_bank.shared.endpoints.register(key, self.as_canonical[target])
        return candidate

    def _widen_once(self, node):
        if not self.config.enable_lazy_as:
            return super()._widen_once(node)
        if (node.state.role == 'done' or (node.state.role == 'corrector'
                and ('stop' not in {e.action.kind for e in node.edges.values()}
                     or self._correction_limit_reached(node)))):
            return super()._widen_once(node)
        if len(node.edges) < self._child_limit(node):
            candidate = self._as_completion(node)
            if candidate is None and self._as_selected(node):
                candidate = self._as_remove(node, persistent=True)
            if candidate is not None:
                edge, state, rank = candidate
                if (edge.action.action_id not in node.edges
                        and edge.action.action_id not in node.retired_action_ids
                        and self._state_key(state) not in node.child_state_keys):
                    self._node(state, rank)
                    node.edges[edge.action.action_id] = edge
                    node.child_state_keys.add(self._state_key(state))
                    node.child_support_words.add(state.support_word)
                    node.pattern_action_counts[edge.action.pattern_id] = node.pattern_action_counts.get(edge.action.pattern_id, 0) + 1
                    self.tree_expansions += 1
                    self.action_kind_counts[edge.action.kind] += 1
                    return edge
        return super()._widen_once(node)

    def _rollout_edge(self, node, policy='mixed'):
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
                return self._ephemeral_edge(node, self._stop_action())
            candidate = self._as_remove(node, persistent=False)
            if candidate is not None:
                return candidate
            rng.setstate(saved)
        return super()._rollout_edge(node, policy)

    def _make_path_step(self, *, node, edge, current_terminal, is_tree_edge):
        step = super()._make_path_step(node=node, edge=edge,
            current_terminal=current_terminal, is_tree_edge=is_tree_edge)
        if edge.action.source == 'fullspace_as_completion':
            credit = self._record_endpoint_context(node, edge.action)
            step.reward += credit
            step.endpoint_novelty_credit = credit
            self.facet_bank.shared.mark_used(node.state.repair_source_word,
                node.state.support_word, edge.action.effective_word)
            self.filler_execution_counts['free_fill'] -= 1
            self.filler_execution_counts['corrector_guided_completion'] += 1
        return step

    def result(self, elapsed_seconds):
        result = super().result(elapsed_seconds)
        result['metadata'].update(algorithm_revision='v26_422_mechanism_transfer',
            baseline='322_v25', transferred_from='422_v29_v30_v31', reduced_to_full_lift=False)
        for event in result['discoveries']:
            origin = self.discovery_origins[event['event_index']]
            event['continuation_origin'] = {k: v for k, v in origin.items() if k != 'local_path'}
            event['local_path'] = origin['local_path']
        result['transfer422'] = {
            'statistics': dict(self.transfer_stats), 'continuation_class_starts': dict(self.continuation_starts),
            'anchor_count': sum(map(len, self.continuation_anchors.values())),
            'as_compute_seconds': self.as_compute_seconds,
            'as_banked_ridges': sum(map(len, self.as_bank.values())),
            'as_consumed_distinct_ridges': len(self.as_executed_proofs),
            'as_executed_proofs': list(self.as_executed_proofs.values()),
            'as_geometry': self.adjacency.diagnostics(),
            'candidate_generation_is_discovery': False, 'unseen_class_examples_used': False,
        }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=3222026)
    parser.add_argument('--profile', choices=('baseline', 'continuation', 'as_only', 'full'), default='full')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite result')
    config = TransferConfig(iterations=args.iterations, seed=args.seed,
        enable_local_continuation=args.profile in ('continuation', 'full'),
        enable_early_noop_guard=args.profile in ('continuation', 'full'),
        enable_lazy_as=args.profile in ('as_only', 'full'))
    engine = (SingleTreeGroupGame(GroupGameConfig(**{
        k: getattr(config, k) for k in GroupGameConfig.__dataclass_fields__}))
        if args.profile == 'baseline' else TransferredGroupGame(config))
    original_iteration = engine.run_iteration
    def progress(iteration):
        original_iteration(iteration)
        if iteration % 250 == 0:
            print(f'iteration={iteration} classes={len(engine.discovery_events)} '
                  f'seconds={time.perf_counter()-engine.started_at:.3f}', flush=True)
    engine.run_iteration = progress
    result = engine.run(output_path=args.output)
    print(result['summary'], flush=True)


if __name__ == '__main__':
    main()
