from __future__ import annotations

"""Adaptive frontier version of the full-space quotient-ridge MCTS."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .atlas import DEFAULT_HANDOFF_ROOT, HandoffPatternAtlas
from .geometry import FacetOracle, FacetResult, RankOracle
from .model import EdgeStats, GameNode, MacroAction, SearchState
from .quotient_actions import QuotientActionGenerator
from .quotient_search import (
    Bell422QuotientSearch,
    QuotientSearchConfig,
    build_parser as build_v26_parser,
)
from .v25_search import DEFAULT_RUNS_DIR, PathStep, TerminalOutcome


@dataclass(frozen=True)
class AdaptiveQuotientSearchConfig(QuotientSearchConfig):
    rank_cache_size: int = 400_000
    candidate_cache_size: int = 40_000
    refinement_cache_size: int = 8_192
    quotient_cache_size: int = 40_000
    ridge_exit_scan: int = 1
    enable_unseen_exit_first: bool = False
    enable_frontier_priority: bool = False
    frontier_action_score_weight: float = 0.0
    frontier_tree_score_weight: float = 0.0
    enable_adaptive_schedule: bool = False
    adaptive_schedule_min_iteration: int = 2_000
    filler_discovery_stagnation: int = 1_000
    recent_yield_window: int = 750
    recent_yield_min_discoveries: int = 4
    corrector_dominance_threshold: float = 0.80
    productive_corrector_filler_floor: float = 0.10
    filler_warmup_iterations: int = 0
    filler_warmup_ratio: float = 0.0
    reseed_interval: int = 2_000
    reseed_span: int = 125
    reseed_filler_ratio: float = 0.30

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(
            self.rank_cache_size,
            self.candidate_cache_size,
            self.refinement_cache_size,
            self.ridge_exit_scan,
            self.recent_yield_window,
            self.recent_yield_min_discoveries,
            self.adaptive_schedule_min_iteration,
            self.filler_discovery_stagnation,
        ) <= 0:
            raise ValueError("v27 cache and online-yield budgets must be positive")
        if not 0.0 <= self.corrector_dominance_threshold <= 1.0:
            raise ValueError("corrector_dominance_threshold must be in [0, 1]")
        for value in (
            self.productive_corrector_filler_floor,
            self.filler_warmup_ratio,
            self.reseed_filler_ratio,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Filler ratios must be in [0, 1]")
        if self.reseed_interval < 0 or self.reseed_span < 0:
            raise ValueError("reseed scheduling values must be nonnegative")


class Bell422AdaptiveQuotientSearch(Bell422QuotientSearch):
    """v27: canonical ridge-frontier dedup and online yield scheduling."""

    config: AdaptiveQuotientSearchConfig
    actions: QuotientActionGenerator

    def __init__(
        self,
        config: AdaptiveQuotientSearchConfig,
        *,
        handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
        atlas: HandoffPatternAtlas | None = None,
        rank_oracle: RankOracle | None = None,
        facet_oracle: FacetOracle | None = None,
    ) -> None:
        effective_rank_oracle = rank_oracle or RankOracle(
            max_entries=config.rank_cache_size
        )
        super().__init__(
            config,
            handoff_root=handoff_root,
            atlas=atlas,
            rank_oracle=effective_rank_oracle,
            facet_oracle=facet_oracle,
        )
        self.actions = QuotientActionGenerator(
            self.atlas,
            self.rank_oracle,
            self.facet_oracle,
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
            max_pattern_embeddings_per_family=(
                config.max_pattern_embeddings_per_family
            ),
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
        )

    @staticmethod
    def _retained_word(state: SearchState, action: MacroAction) -> int:
        return (state.support_word & ~action.remove_word) | action.add_word

    def _frontier_priority(
        self,
        state: SearchState,
        action: MacroAction,
    ) -> float:
        if (
            not self.config.enable_frontier_priority
            or action.source not in ("ridge_carve_exact", "quotient_ridge_basis")
        ):
            return 0.0
        retained = self._retained_word(state, action)
        return self.actions.ridge_frontier_priority(
            state.source_facet_word or state.support_word,
            retained,
        )

    def _approximate_action_score(
        self,
        node: GameNode,
        action: MacroAction,
    ) -> float:
        return super()._approximate_action_score(node, action) + (
            self.config.frontier_action_score_weight
            * self._frontier_priority(node.state, action)
        )

    def _select_edge(self, node: GameNode) -> EdgeStats | None:
        if not node.edges:
            return None
        if node.state.role == "corrector" and self._correction_limit_reached(node):
            return node.edges.get("stop")
        epoch = self.knowledge.epoch
        root = math.sqrt(node.visits + 1.0)

        def score(edge: EdgeStats) -> tuple[float, str]:
            recent = edge.epoch_mean(epoch)
            q_value = 0.0 if recent is None else recent
            epoch_visits = edge.epoch_visits if edge.epoch == epoch else 0
            exploration = (
                self.config.exploration
                * edge.prior
                * root
                / (1.0 + epoch_visits)
            )
            frontier = self._frontier_priority(node.state, edge.action)
            frontier /= 1.0 + 0.05 * epoch_visits
            return (
                q_value
                + exploration
                + self.config.frontier_tree_score_weight * frontier
                + 1e-9 * self.rng.random(),
                edge.action.action_id,
            )

        return max(node.edges.values(), key=score)

    def _terminal_outcome(
        self,
        facet: FacetResult,
        *,
        path: Sequence[PathStep],
        depth: int,
        source: str,
    ) -> TerminalOutcome:
        outcome = super()._terminal_outcome(
            facet,
            path=path,
            depth=depth,
            source=source,
        )
        self.actions.observe_terminal(facet)
        if outcome.new_class:
            for step in path:
                action = step.edge.action
                if action.source not in (
                    "ridge_carve_exact",
                    "quotient_ridge_basis",
                ):
                    continue
                self.actions.record_frontier_discovery(
                    step.node.state.source_facet_word
                    or step.node.state.support_word,
                    self._retained_word(step.node.state, action),
                )
        return outcome

    @staticmethod
    def _corrector_assisted(event: dict[str, object]) -> bool:
        return any(
            action.get("kind") in ("remove", "rewrite")
            for action in event.get("path", ())
        )

    def _free_filler_ratio(self) -> float:
        ratio = super()._free_filler_ratio()
        if (
            not self.config.enable_adaptive_schedule
            or self.current_iteration < self.config.adaptive_schedule_min_iteration
        ):
            return ratio
        if self.current_iteration <= self.config.filler_warmup_iterations:
            return max(ratio, self.config.filler_warmup_ratio)

        pure_filler_discoveries = [
            int(event["iteration"])
            for event in self.discovery_events
            if not self._corrector_assisted(event)
        ]
        last_filler_discovery = max(pure_filler_discoveries, default=0)
        if (
            self.current_iteration - last_filler_discovery
            < self.config.filler_discovery_stagnation
        ):
            return ratio

        cutoff = max(0, self.current_iteration - self.config.recent_yield_window)
        recent = [
            event
            for event in self.discovery_events
            if int(event["iteration"]) > cutoff
        ]
        if len(recent) >= self.config.recent_yield_min_discoveries:
            corrector_share = sum(map(self._corrector_assisted, recent)) / len(recent)
            if corrector_share >= self.config.corrector_dominance_threshold:
                ratio = min(
                    ratio,
                    self.config.productive_corrector_filler_floor,
                )

        if (
            self.config.reseed_interval
            and self.config.reseed_span
            and self.current_iteration % self.config.reseed_interval
            < self.config.reseed_span
        ):
            ratio = max(ratio, self.config.reseed_filler_ratio)
        return ratio

    def result(self) -> dict[str, object]:
        result = super().result()
        result["algorithm"] = "bell422_adaptive_quotient_ridge_mcts_v27"
        result["features"].update(
            {
                "canonical_ridge_endpoint_pair_tracking": False,
                "concrete_embedding_bridges_preserved": True,
                "unseen_ridge_exit_first": self.config.enable_unseen_exit_first,
                "online_frontier_yield_priority": (
                    self.config.enable_frontier_priority
                ),
                "late_corrector_dominance_schedule": (
                    self.config.enable_adaptive_schedule
                ),
                "periodic_filler_reseed": bool(
                    self.config.enable_adaptive_schedule
                    and self.config.reseed_interval
                    and self.config.reseed_span
                ),
            }
        )
        result["quotient_game"]["v27_config"] = {
            key: value
            for key, value in asdict(self.config).items()
            if key
            in {
                "rank_cache_size",
                "candidate_cache_size",
                "refinement_cache_size",
                "quotient_cache_size",
                "ridge_exit_scan",
                "enable_unseen_exit_first",
                "enable_frontier_priority",
                "frontier_action_score_weight",
                "frontier_tree_score_weight",
                "enable_adaptive_schedule",
                "adaptive_schedule_min_iteration",
                "filler_discovery_stagnation",
                "recent_yield_window",
                "recent_yield_min_discoveries",
                "corrector_dominance_threshold",
                "productive_corrector_filler_floor",
                "filler_warmup_iterations",
                "filler_warmup_ratio",
                "reseed_interval",
                "reseed_span",
                "reseed_filler_ratio",
            }
        }
        return result


def run_search(
    config: AdaptiveQuotientSearchConfig,
    *,
    handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    output_path: Path | str | None = None,
    checkpoint_paths: Mapping[int, Path | str] | None = None,
) -> dict[str, object]:
    engine = Bell422AdaptiveQuotientSearch(config, handoff_root=handoff_root)
    result = engine.run(checkpoint_paths=checkpoint_paths)
    if output_path is not None:
        engine._write_result(result, Path(output_path))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = build_v26_parser()
    parser.description = "Bell 4-2-2 adaptive quotient-ridge MCTS v27"
    parser.add_argument("--rank-cache-size", type=int, default=400_000)
    parser.add_argument("--candidate-cache-size", type=int, default=40_000)
    parser.add_argument("--ridge-exit-scan", type=int, default=1)
    parser.add_argument("--enable-unseen-exit-first", action="store_true")
    parser.add_argument("--enable-frontier-priority", action="store_true")
    parser.add_argument("--enable-late-adaptive-schedule", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AdaptiveQuotientSearchConfig(
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.max_depth,
        min_corrector_rank=args.min_corrector_rank,
        tail_checks=args.tail_checks,
        progress_interval=args.progress_interval,
        max_wall_seconds=(
            None
            if args.max_wall_hours is None
            else 3600.0 * args.max_wall_hours
        ),
        quotient_candidate_limit=args.quotient_candidate_limit,
        quotient_secondary_limit=args.quotient_secondary_limit,
        partial_rank_checks=args.partial_rank_checks,
        ridge_rank_checks=args.ridge_rank_checks,
        ridge_basis_trials=args.ridge_basis_trials,
        ridge_basis_rank_checks=args.ridge_basis_rank_checks,
        supporting_face_lp_actions=args.supporting_face_lp_actions,
        supporting_face_max_rank=args.supporting_face_max_rank,
        use_catalog=not args.no_catalog,
        canonicalize_classes=not args.no_canonicalize,
        rank_cache_size=args.rank_cache_size,
        candidate_cache_size=args.candidate_cache_size,
        ridge_exit_scan=args.ridge_exit_scan,
        enable_unseen_exit_first=args.enable_unseen_exit_first,
        enable_frontier_priority=args.enable_frontier_priority,
        frontier_action_score_weight=(
            0.70 if args.enable_frontier_priority else 0.0
        ),
        frontier_tree_score_weight=(
            0.10 if args.enable_frontier_priority else 0.0
        ),
        enable_adaptive_schedule=args.enable_late_adaptive_schedule,
    )
    output = args.output or (
        DEFAULT_RUNS_DIR
        / f"bell422_adaptive_quotient_i{config.iterations}_v27.json"
    )
    checkpoints = {int(iteration): path for iteration, path in args.checkpoint}
    invalid = sorted(
        iteration
        for iteration in checkpoints
        if not 0 < iteration < config.iterations
    )
    if invalid:
        raise SystemExit(
            "checkpoint iterations must be inside the run: "
            + ", ".join(map(str, invalid))
        )
    result = run_search(
        config,
        handoff_root=args.handoff_root,
        output_path=output,
        checkpoint_paths=checkpoints,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "node_count": result["tree"]["node_count"],
                "discovered_class_count": result["coverage"][
                    "discovered_class_count"
                ],
                "catalog_match_count": sum(
                    event["catalog_index"] is not None
                    for event in result["coverage"]["discoveries"]
                ),
                "quotient_stats": result["quotient_game"]["quotient_stats"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
