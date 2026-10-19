from __future__ import annotations

"""Isolated v6 + recursive adjacency, and an AS-only observed-facet explorer."""

import math
import random
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace

from .recursive_adjacency import AdjacencyConfig, RecursiveAdjacency
from .reduced_multi_exit_search import ReducedMultiExitConfig, ReducedMultiExitMCTS
from .reduced_novelty_search import DeferredLiftGeometry
from .reduced_subgroup_search import ReducedAction


@dataclass(frozen=True)
class ReducedAdjacencyConfig(ReducedMultiExitConfig):
    adjacency_cutoff: int = 16
    adjacency_interval: int = 4
    adjacency_macro_seconds: float = 1.0
    adjacency_max_outputs: int = 32
    enable_recursive_adjacency: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.adjacency_config()
        if self.adjacency_interval < 1:
            raise ValueError("adjacency interval must be positive")

    def adjacency_config(self):
        return AdjacencyConfig(cutoff_vertices=self.adjacency_cutoff,
                               max_macro_seconds=self.adjacency_macro_seconds,
                               max_outputs=self.adjacency_max_outputs)


class ReducedAdjacencyMCTS(ReducedMultiExitMCTS):
    def __init__(self, problem, config: ReducedAdjacencyConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.adjacency = RecursiveAdjacency(problem.irredundant_points, config.adjacency_config())
        self.as_requests = Counter()
        self.as_bank: dict[int, dict[int, dict]] = {}
        self.as_attempts = Counter()
        self.as_executed_proofs: dict[tuple[int, int, int], dict] = {}
        self.as_rng = random.Random(config.seed ^ 0xA5FACADE)
        self.search_deadline = math.inf

    def _corrector_actions(self, node, *, count, rng, persistent):
        actions = super()._corrector_actions(node, count=count, rng=rng, persistent=persistent)
        if (not self.config.enable_recursive_adjacency
                or node.state.corrections_used >= self.config.productive_max_corrections):
            return actions
        source = node.state.support_word
        self.as_requests[source] += 1
        bank = self.as_bank.setdefault(source, {})
        if (self.as_requests[source] - 1) % self.config.adjacency_interval == 0:
            batch = self.adjacency.sample(source, self.as_rng, deadline=self.search_deadline)
            for proof in batch.outputs:
                retained = int(proof["retained_word_hex"], 16)
                bank.setdefault(retained, proof)
        if not bank:
            return actions
        retained = min(bank, key=lambda word: (self.as_attempts[(source, word)], rng.random()))
        candidate = ReducedAction("remove", retained_word=retained,
                                  sampled_facet_word=source, source="recursive_as_ridge")
        # Preserve the existing action quota; MCTS still chooses whether to use AS.
        actions = [action for action in actions if action.retained_word != retained]
        return [candidate, *actions[:max(0, count-1)]]

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        if edge.action.source == "recursive_as_ridge":
            source, retained = node.state.support_word, edge.action.retained_word
            self.as_attempts[(source, retained)] += 1
        terminal = edge.terminal
        if terminal is not None and node.state.source_facet_word:
            source = node.state.source_facet_word
            removal = next((action for action in reversed(path_actions) if action.kind == "remove"), None)
            if removal is not None and removal.source == "recursive_as_ridge":
                proof = self.as_bank.get(source, {}).get(removal.retained_word)
                if proof is None or int(proof["destination_word_hex"], 16) != terminal.closure_word:
                    raise ValueError("executed AS ridge disagrees with certified rotation endpoint")
                key = source, removal.retained_word, terminal.closure_word
                self.as_executed_proofs[key] = proof
                self.stats["as_actual_terminal_arrivals"] += 1
                self.stats["as_new_reduced_terminal_arrivals"] += int(
                    terminal.closure_word not in self.discovered_reduced_facets)
        return super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)

    def result(self, *, target_ids=()):
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_v6_recursive_adjacency_proposals_v1"
        result["recursive_adjacency"] = {
            "config": asdict(self.config.adjacency_config()),
            "proposal_interval": self.config.adjacency_interval,
            "geometry": self.adjacency.diagnostics(),
            "banked_top_ridges": sum(map(len, self.as_bank.values())),
            "actual_terminal_arrivals": self.stats["as_actual_terminal_arrivals"],
            "new_reduced_terminal_arrivals": self.stats["as_new_reduced_terminal_arrivals"],
            "executed_proofs": list(self.as_executed_proofs.values()),
            "candidate_generation_is_discovery": False,
            "reward_rollout_and_local_correction_budget_changed": False,
        }
        return result


class ReducedAdjacencyExplorer:
    """FIFO repeated AS from actually visited reduced facets, without MCTS."""

    def __init__(self, problem, config, *, handoff_root, full_oracle=None):
        self.problem, self.config = problem, config
        self.geometry = DeferredLiftGeometry(problem, handoff_root=handoff_root, full_oracle=full_oracle)
        self.adjacency = RecursiveAdjacency(problem.irredundant_points, config.adjacency_config())
        self.rng = random.Random(config.seed ^ 0xA5FACADE)
        self.queue = deque()
        self.pending = deque()
        self.witnesses = {}
        self.reduced_events, self.discovery_events = [], []
        self.discovered_classes = {}
        self.class_frequencies = Counter()
        self.executed_proofs = []
        self.iterations = 0
        self.started_at = 0.0
        self.search_deadline = math.inf
        self.stats = Counter()

    def observe(self, word, path, proof=None):
        terminal = self.geometry.validate(word)
        if not terminal.valid or terminal.closure_word != word:
            raise ValueError("AS actual terminal is not a closed reduced facet")
        if word in self.witnesses:
            self.stats["repeated_reduced_arrivals"] += 1
            return
        self.witnesses[word] = path
        self.queue.append(word)
        terminal = self.geometry.lift_taken_terminal(terminal)
        elapsed = time.perf_counter() - self.started_at
        record = {"iteration": self.iterations, "elapsed_seconds": elapsed,
                  "reduced_facet_word_hex": hex(word), "lifted": terminal.lifted,
                  "path": [action.to_dict() for action in path]}
        self.reduced_events.append(record)
        if proof is not None:
            self.executed_proofs.append(proof)
        if terminal.lifted:
            full = terminal.full_result
            self.class_frequencies[full.class_id] += 1
            if full.class_id not in self.discovered_classes:
                event = {**record, "class_id": full.class_id, "catalog_index": full.catalog_index,
                         "canonical_h_row": list(full.canonical_h_row),
                         "original_support_word_hex": hex(terminal.original_support_word),
                         "node_id": self.problem.node_id}
                self.discovered_classes[full.class_id] = event
                self.discovery_events.append(event)

    def run_iteration(self, iteration):
        self.iterations = iteration
        if not self.queue:
            samples = self.geometry.sample_supporting_facets(0, count=1, rng=self.rng)
            if samples:
                word = samples[0]
                self.observe(word, (ReducedAction("add", add_word=word,
                                                  sampled_facet_word=word, source="as_initial_lp"),))
            return
        if not self.pending:
            source = self.queue.popleft()
            self.queue.append(source)
            batch = self.adjacency.sample(source, self.rng, deadline=self.search_deadline)
            outputs = sorted(batch.outputs, key=lambda proof: (
                int(proof["destination_word_hex"], 16) in self.witnesses,
                int(proof["destination_word_hex"], 16)))
            self.pending.extend((source, proof) for proof in outputs)
            self.stats["source_expansions"] += 1
        if self.pending:
            source, proof = self.pending.popleft()
            destination = int(proof["destination_word_hex"], 16)
            retained = int(proof["retained_word_hex"], 16)
            actions = (ReducedAction("remove", retained_word=retained,
                                      sampled_facet_word=source, source="recursive_as_ridge"),
                       ReducedAction("add", add_word=destination & ~retained,
                                      sampled_facet_word=destination, source="recursive_as_rotation"))
            self.observe(destination, self.witnesses[source] + actions, proof)

    def result(self, *, target_ids=()):
        elapsed = time.perf_counter() - self.started_at
        covered = sorted(set(target_ids) & self.discovered_classes.keys())
        return {"algorithm": "bell422_repeated_fifo_recursive_adjacency_only_v1",
                "problem": self.problem.to_dict(), "config": asdict(self.config),
                "runtime": {"iterations_completed": self.iterations, "elapsed_seconds": elapsed},
                "coverage": {"discoveries": self.discovery_events,
                             "covered_target_count": len(covered), "covered_target_ids": covered,
                             "discovered_class_count": len(self.discovery_events),
                             "class_frequencies": dict(self.class_frequencies)},
                "reduced_facets": {"discovery_events": self.reduced_events,
                                   "witnesses": self.reduced_events,
                                   "unique_count": len(self.reduced_events)},
                "geometry": self.geometry.to_dict(), "statistics": dict(self.stats),
                "recursive_adjacency": {"geometry": self.adjacency.diagnostics(),
                                        "executed_proofs": self.executed_proofs,
                                        "queue_size": len(self.queue), "pending_outputs": len(self.pending)},
                "prior_boundary": {"target_labels_affect_search": False,
                                   "unseen_catalog_witnesses_used": False,
                                   "lifting_is_reporting_only": True,
                                   "queue_revisits_observed_facets": True}}
