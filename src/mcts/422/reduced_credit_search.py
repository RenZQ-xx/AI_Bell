from __future__ import annotations

"""Counterfactual Corrector credit, isolated from the reduced baseline."""

import math
from dataclasses import dataclass, field
from typing import Sequence

from .reduced_subgroup_search import (
    ReducedEdge,
    ReducedFacetResult,
    ReducedNode,
    ReducedSearchConfig,
    ReducedSubgroupMCTS,
)


@dataclass(frozen=True)
class ReducedCreditConfig(ReducedSearchConfig):
    enable_counterfactual_credit: bool = True
    corrector_exact_utility: float = 0.25
    corrector_utility_frequency_penalty: float = 0.04
    corrector_utility_penalty_cap: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all(math.isfinite(value) and value >= 0 for value in (
            self.corrector_exact_utility,
            self.corrector_utility_frequency_penalty,
            self.corrector_utility_penalty_cap,
        )):
            raise ValueError("Corrector utility parameters must be finite and nonnegative")
        if self.corrector_utility_penalty_cap > self.corrector_exact_utility:
            raise ValueError("utility penalty cap exceeds exact utility")


@dataclass
class CorrectionCredit:
    path_index: int
    source_word: int
    source_utility: float
    operation_reward: float
    persistent: bool
    outcome_utility: float | None = None
    geometric_credit: float = 0.0
    novelty_credit: float = 0.0
    target_word: int | None = None

    @property
    def value(self) -> float:
        outcome = (
            self.outcome_utility
            if self.outcome_utility is not None
            else min(self.source_utility, 0.0)
        )
        return (
            outcome - self.source_utility + self.operation_reward
            + self.geometric_credit + self.novelty_credit
        )


@dataclass
class CreditEdge(ReducedEdge):
    credit: CorrectionCredit | None = field(default=None, repr=False)

    def update(self, value: float, epoch: int) -> None:
        super().update(self.credit.value if self.credit is not None else value, epoch)


def distribute_novelty(
    credits: Sequence[CorrectionCredit], *, terminal_index: int,
    reward: float, discount: float,
) -> list[float]:
    if not credits:
        return []
    weights = [discount ** (terminal_index - credit.path_index) for credit in credits]
    total = sum(weights)
    shares = [reward * weight / total for weight in weights]
    for credit, share in zip(credits, shares):
        credit.novelty_credit += share
    return shares


class ReducedCreditMCTS(ReducedSubgroupMCTS):
    config: ReducedCreditConfig

    def __init__(self, problem, config: ReducedCreditConfig, **kwargs) -> None:
        super().__init__(problem, config, **kwargs)
        self.episode_credits: list[CorrectionCredit] = []
        self.pending_credit: CorrectionCredit | None = None
        self.credit_examples: list[dict] = []
        self.credit_discovery_events: list[dict] = []

    def _edge(self, node, action):
        built = super()._edge(node, action)
        if built is None:
            return None
        edge, transition = built
        return CreditEdge(**vars(edge)), transition

    def _utility(self, terminal: ReducedFacetResult) -> float:
        full = terminal.full_result
        if not terminal.valid or not full or not full.valid or full.class_id is None:
            return 0.0
        hits = self.shared_discovery.class_frequencies[full.class_id]
        return self.config.corrector_exact_utility - min(
            self.config.corrector_utility_penalty_cap,
            self.config.corrector_utility_frequency_penalty * math.log1p(hits),
        )

    def _fail_pending(self) -> None:
        if self.pending_credit is not None:
            self.pending_credit.outcome_utility = min(
                self.pending_credit.source_utility, self.config.dead_end_reward
            )
            self.pending_credit = None

    def _select(self, node):
        edge = super()._select(node)
        if edge is None:
            self._fail_pending()
        return edge

    def _rollout_edge(self, node, policy):
        moved = super()._rollout_edge(node, policy)
        if moved is None:
            self._fail_pending()
        return moved

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        reward, new_class = super()._step_reward(
            node, edge, path_actions, path_terminals, tree_edge
        )
        if not self.config.enable_counterfactual_credit:
            return reward, new_class
        if edge.action.kind == "remove":
            source = self.geometry.validate(node.state.support_word)
            credit = CorrectionCredit(
                len(path_actions) - 1, node.state.support_word, self._utility(source),
                reward, tree_edge,
            )
            self.episode_credits.append(credit)
            self.pending_credit = credit
            edge.credit = credit
        elif edge.terminal is not None:
            terminal = edge.terminal
            if self.pending_credit is not None:
                credit = self.pending_credit
                credit.outcome_utility = self._utility(terminal)
                credit.target_word = terminal.closure_word
                observed = path_terminals[-1]
                if observed["new_retained_face_exit"]:
                    credit.geometric_credit += (
                        self.config.new_lower_face_exit_reward
                        if observed["retained_face_rank"] < self.problem.ridge_rank
                        else self.config.new_ridge_exit_reward
                    )
                if terminal.closure_word == credit.source_word:
                    credit.geometric_credit -= self.config.source_return_penalty
                self.pending_credit = None
            if new_class:
                shares = distribute_novelty(
                    self.episode_credits, terminal_index=len(path_actions) - 1,
                    reward=self.config.new_class_reward, discount=self.config.discount,
                )
                if shares:
                    self.credit_discovery_events.append({
                        "iteration": self.current_iteration,
                        "class_id": terminal.full_result.class_id,
                        "reward_pool": self.config.new_class_reward,
                        "contributors": [
                            {"path_index": credit.path_index, "share": share}
                            for credit, share in zip(self.episode_credits, shares)
                        ],
                    })
        return reward, new_class

    def run_iteration(self, iteration: int, *, global_iteration: int | None = None) -> None:
        self.episode_credits = []
        self.pending_credit = None
        super().run_iteration(iteration, global_iteration=global_iteration)
        for credit in self.episode_credits:
            self.stats["corrector_credit_observations"] += 1
            self.stats["corrector_credit_tree_updates"] += int(credit.persistent)
            self.stats["corrector_credit_unresolved"] += int(credit.outcome_utility is None)
            self.stats["corrector_credit_positive"] += int(credit.value > 0)
            if len(self.credit_examples) < 64:
                self.credit_examples.append({
                    "iteration": iteration,
                    "path_index": credit.path_index,
                    "source_word_hex": hex(credit.source_word),
                    "target_word_hex": hex(credit.target_word) if credit.target_word else None,
                    "source_utility": credit.source_utility,
                    "outcome_utility": credit.outcome_utility,
                    "operation_reward": credit.operation_reward,
                    "geometric_credit": credit.geometric_credit,
                    "novelty_credit": credit.novelty_credit,
                    "learned_credit": credit.value,
                    "tree_edge": credit.persistent,
                })

    def result(self, *, target_ids: Sequence[str] = ()) -> dict:
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_reduced_counterfactual_corrector_mcts_v1"
        result["features"]["corrector_counterfactual_credit"] = (
            self.config.enable_counterfactual_credit
        )
        result["corrector_credit"] = {
            "scope": "per_node_observed_classes_only",
            "filler_return_unchanged": True,
            "leaf_heuristic_unchanged": True,
            "failed_lifts_remain_searchable": True,
            "no_extra_counterfactual_rollouts": True,
            "unresolved_outcome_utility": 0.0,
            "examples": self.credit_examples,
            "discovery_events": self.credit_discovery_events,
        }
        return result
