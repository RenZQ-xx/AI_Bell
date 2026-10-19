from __future__ import annotations

"""Single-tree MCTS with low-dimensional symmetric completion proposals."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .atlas import DEFAULT_HANDOFF_ROOT
from .geometry import FacetResult
from .model import EdgeStats, GameNode
from .quotient_search_v27 import (
    AdaptiveQuotientSearchConfig,
    Bell422AdaptiveQuotientSearch,
    build_parser as build_v27_parser,
)
from .symmetric_actions import RelaxationPatternAtlas, SymmetricActionGenerator
from .v25_search import DEFAULT_RUNS_DIR, PathStep, TerminalOutcome, Transition


@dataclass(frozen=True)
class SymmetricSearchConfig(AdaptiveQuotientSearchConfig):
    conditional_completion_period: int = 4
    conditional_completions_per_iteration: int = 1
    conditional_min_rank: int = 60
    conditional_max_rank: int = 77
    relaxation_trials: int = 2
    conditional_cache_size: int = 8_192

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(
            self.conditional_completion_period,
            self.conditional_completions_per_iteration,
            self.conditional_cache_size,
        ) <= 0:
            raise ValueError("conditional completion budgets must be positive")
        if not 0 <= self.conditional_min_rank <= self.conditional_max_rank <= 77:
            raise ValueError("conditional completion ranks must be inside [0, 77]")
        if self.relaxation_trials < 0:
            raise ValueError("relaxation_trials must be nonnegative")


class Bell422SymmetricSearch(Bell422AdaptiveQuotientSearch):
    config: SymmetricSearchConfig
    actions: SymmetricActionGenerator

    def __init__(
        self, config: SymmetricSearchConfig, *, handoff_root=DEFAULT_HANDOFF_ROOT,
        atlas: RelaxationPatternAtlas | None = None,
    ) -> None:
        super().__init__(
            config, handoff_root=handoff_root,
            atlas=atlas or RelaxationPatternAtlas(handoff_root),
        )
        self.actions = SymmetricActionGenerator(
            self.atlas, self.rank_oracle, self.facet_oracle,
            seed=config.seed,
            filler_candidate_limit=config.filler_candidate_limit,
            corrector_candidate_limit=config.corrector_candidate_limit,
            tail_checks=config.tail_checks,
            stabilizer_probes=config.stabilizer_probes,
            candidate_cache_size=config.candidate_cache_size,
            context_free_cache_keys=True,
            ridge_pivot=True,
            filler_max_orbits_per_action=2,
            corrector_max_orbits_per_action=4,
            max_pattern_embeddings_per_family=config.max_pattern_embeddings_per_family,
            quotient_candidate_limit=config.quotient_candidate_limit,
            quotient_secondary_limit=config.quotient_secondary_limit,
            partial_rank_checks=config.partial_rank_checks,
            ridge_rank_checks=config.ridge_rank_checks,
            ridge_basis_trials=config.ridge_basis_trials,
            ridge_basis_rank_checks=config.ridge_basis_rank_checks,
            refinement_cache_size=config.refinement_cache_size,
            quotient_cache_size=config.quotient_cache_size,
            ridge_exit_scan=config.ridge_exit_scan,
            unseen_exit_first=config.enable_unseen_exit_first,
            canonical_frontier_dedup=False,
            conditional_cache_size=config.conditional_cache_size,
        )
        self._conditional_iteration = -1
        self._conditional_calls = 0
        self._conditional_served: dict[tuple, int] = {}

    def _conditional_edge(
        self, node: GameNode
    ) -> tuple[EdgeStats, Transition] | None:
        if (
            node is self.root
            or node.state.role != "filler"
            or not node.state.source_facet_word
            or not self.config.conditional_min_rank <= node.rank <= self.config.conditional_max_rank
            or node.visits % self.config.conditional_completion_period
        ):
            return None
        key = self._state_key(node.state)
        if self._conditional_served.get(key) == node.visits:
            return None
        if self._conditional_iteration != self.current_iteration:
            self._conditional_iteration = self.current_iteration
            self._conditional_calls = 0
        if self._conditional_calls >= self.config.conditional_completions_per_iteration:
            return None
        self._conditional_calls += 1
        self._conditional_served[key] = node.visits
        primary = node.state.context_pattern_id
        if primary not in self.atlas.by_id:
            return None
        self.stats["conditional_completion_calls"] += 1
        actions = self.actions.conditional_completion_actions(
            node.state, primary,
            nonce=node.visits + node.action_lane_cursor,
            relaxation_trials=self.config.relaxation_trials,
        )
        for action in actions:
            if (
                action.action_id in node.edges
                or action.action_id in node.rejected_action_ids
                or action.action_id in node.retired_action_ids
            ):
                continue
            moved = self._transition(node.state, action)
            if moved is None or (moved.facet is not None and not moved.facet.valid):
                node.rejected_action_ids.add(action.action_id)
                continue
            if self._state_key(moved.state) in node.child_keys:
                node.rejected_action_ids.add(action.action_id)
                self.stats["conditional_duplicate_children"] += 1
                continue
            candidate = self._edge_for(node, action, moved)
            if candidate is not None:
                self.stats["conditional_completion_edges"] += 1
                return candidate
        return None

    def _widen_once(self, node: GameNode) -> EdgeStats | None:
        if len(node.edges) < self._child_limit(node):
            candidate = self._conditional_edge(node)
            if candidate is not None:
                return self._install_edge(node, *candidate)
        return super()._widen_once(node)

    def _rollout_edge(self, node: GameNode, policy: str):
        candidate = self._conditional_edge(node)
        return candidate if candidate is not None else super()._rollout_edge(node, policy)

    def _terminal_outcome(
        self, facet: FacetResult, *, path: Sequence[PathStep], depth: int, source: str
    ) -> TerminalOutcome:
        outcome = super()._terminal_outcome(facet, path=path, depth=depth, source=source)
        if outcome.new_class:
            credited = {
                step.edge.action.pattern_id for step in path
                if step.edge.action.source in (
                    "fullspace_supporting_face_lp", "conditional_symmetric_lp"
                )
            }
            for pattern_id in credited:
                self.actions.proposal_discoveries[pattern_id] += 1
        return outcome

    def result(self) -> dict[str, object]:
        result = super().result()
        result["algorithm"] = "bell422_symmetric_completion_mcts_v28"
        result["features"].update({
            "reduced_coefficient_space_lp": True,
            "conditional_symmetric_completion": True,
            "verified_proper_subgroup_relaxation": True,
            "global_fullspace_terminal_reward": True,
        })
        result["quotient_game"]["v28_config"] = {
            key: value for key, value in asdict(self.config).items()
            if key.startswith("conditional_") or key == "relaxation_trials"
        }
        return result


def run_search(
    config: SymmetricSearchConfig, *, handoff_root=DEFAULT_HANDOFF_ROOT,
    output_path: Path | str | None = None,
    checkpoint_paths: Mapping[int, Path | str] | None = None,
) -> dict[str, object]:
    engine = Bell422SymmetricSearch(config, handoff_root=handoff_root)
    result = engine.run(checkpoint_paths=checkpoint_paths)
    if output_path is not None:
        engine._write_result(result, Path(output_path))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = build_v27_parser()
    parser.description = "Bell 4-2-2 symmetric completion MCTS v28"
    parser.add_argument("--conditional-completion-period", type=int, default=4)
    parser.add_argument("--conditional-completions-per-iteration", type=int, default=1)
    parser.add_argument("--conditional-min-rank", type=int, default=60)
    parser.add_argument("--conditional-max-rank", type=int, default=77)
    parser.add_argument("--relaxation-trials", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_keys = set(SymmetricSearchConfig.__dataclass_fields__)
    settings = {
        key: value for key, value in vars(args).items() if key in config_keys
    }
    settings.update({
        "use_catalog": not args.no_catalog,
        "canonicalize_classes": not args.no_canonicalize,
        "enable_adaptive_schedule": args.enable_late_adaptive_schedule,
        "frontier_action_score_weight": 0.70 if args.enable_frontier_priority else 0.0,
        "frontier_tree_score_weight": 0.10 if args.enable_frontier_priority else 0.0,
        "max_wall_seconds": None if args.max_wall_hours is None else args.max_wall_hours * 3600.0,
    })
    config = SymmetricSearchConfig(**settings)
    output = args.output or DEFAULT_RUNS_DIR / f"bell422_symmetric_i{config.iterations}_v28.json"
    checkpoints = {int(iteration): path for iteration, path in args.checkpoint}
    if any(not 0 < iteration < config.iterations for iteration in checkpoints):
        raise SystemExit("checkpoint iterations must be inside the run")
    result = run_search(
        config, handoff_root=args.handoff_root, output_path=output,
        checkpoint_paths=checkpoints,
    )
    print(json.dumps({
        "output": str(output),
        "elapsed_seconds": result["runtime"]["elapsed_seconds"],
        "classes": result["coverage"]["discovered_class_count"],
        "nodes": result["tree"]["node_count"],
        "symmetric_stats": result["quotient_game"]["symmetric_lp"]["stats"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
