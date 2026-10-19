from __future__ import annotations

"""Quotient-guided full-space Filler-Corrector MCTS for Bell 4-2-2."""

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .atlas import DEFAULT_HANDOFF_ROOT, HandoffPatternAtlas
from .geometry import FacetOracle, RankOracle
from .model import EdgeStats, GameNode, MacroAction, SearchState
from .quotient_actions import QuotientActionGenerator
from .v25_search import (
    DEFAULT_RUNS_DIR,
    Bell422V25Search,
    V25SearchConfig,
)


@dataclass(frozen=True)
class QuotientSearchConfig(V25SearchConfig):
    quotient_candidate_limit: int = 8
    quotient_secondary_limit: int = 2
    partial_rank_checks: int = 8
    ridge_rank_checks: int = 24
    ridge_basis_trials: int = 3
    ridge_basis_rank_checks: int = 192
    supporting_face_lp_actions: int = 1
    supporting_face_max_rank: int = -1
    root_pattern_widening_k0: int = 8
    root_pattern_widening_alpha: float = 4.0
    root_pattern_widening_beta: float = 0.5
    root_action_widening_k0: int = 8
    root_action_widening_alpha: float = 4.0
    root_action_widening_beta: float = 0.5
    root_max_children: int = 256
    root_lp_pattern_trials: int = 8
    quotient_cache_size: int = 20_000
    facet_replay_interval: int = 4
    adaptive_replay_class_scale: int = 4
    adaptive_replay_min_coverage: float = 0.25
    free_filler_initial_ratio: float = 0.78
    free_filler_floor: float = 0.15
    unproductive_corrector_filler_floor: float = 0.68
    progress_interval: int = 100

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(
            self.quotient_candidate_limit,
            self.quotient_secondary_limit,
            self.ridge_rank_checks,
            self.ridge_basis_trials,
            self.ridge_basis_rank_checks,
            self.supporting_face_lp_actions,
            self.root_pattern_widening_k0,
            self.root_action_widening_k0,
            self.root_max_children,
            self.root_lp_pattern_trials,
            self.quotient_cache_size,
        ) <= 0:
            raise ValueError("quotient search budgets must be positive")
        if self.partial_rank_checks < 0:
            raise ValueError("partial_rank_checks must be nonnegative")
        if not -1 <= self.supporting_face_max_rank < 79:
            raise ValueError("supporting_face_max_rank must be in [-1, 78]")
        if not 0 <= self.unproductive_corrector_filler_floor <= 1:
            raise ValueError("unproductive_corrector_filler_floor must be in [0, 1]")


class Bell422QuotientSearch(Bell422V25Search):
    """One full-space graph with subgroup partitions used as action charts."""

    config: QuotientSearchConfig
    actions: QuotientActionGenerator

    def __init__(
        self,
        config: QuotientSearchConfig,
        *,
        handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
        atlas: HandoffPatternAtlas | None = None,
        rank_oracle: RankOracle | None = None,
        facet_oracle: FacetOracle | None = None,
    ) -> None:
        super().__init__(
            config,
            handoff_root=handoff_root,
            atlas=atlas,
            rank_oracle=rank_oracle,
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
            quotient_cache_size=config.quotient_cache_size,
        )
        self.root_lp_attempt_counts: Counter[str] = Counter()

    def _state_key(self, state: SearchState) -> tuple:
        return state.geometry_key

    def _pattern_limit(self, node: GameNode) -> int:
        if node is self.root:
            value = (
                self.config.root_pattern_widening_k0
                + self.config.root_pattern_widening_alpha
                * max(1, node.visits) ** self.config.root_pattern_widening_beta
            )
            return max(1, min(len(self.pattern_order), math.floor(value)))
        return super()._pattern_limit(node)

    def _child_limit(self, node: GameNode) -> int:
        if node is self.root:
            value = (
                self.config.root_action_widening_k0
                + self.config.root_action_widening_alpha
                * max(1, node.visits) ** self.config.root_action_widening_beta
            )
            return min(self.config.root_max_children, max(1, math.floor(value)))
        return super()._child_limit(node)

    def _secondary_charts(
        self,
        node: GameNode,
        primary_id: str,
    ) -> tuple[str, ...]:
        candidates: list[str] = []
        candidates.extend(sorted(node.context_pattern_ids))
        candidates.extend(
            sorted(
                node.expanded_patterns,
                key=lambda pattern_id: (
                    self.knowledge.pattern_score(node.state.role, pattern_id),
                    pattern_id,
                ),
                reverse=True,
            )
        )
        if primary_id in self.atlas.by_id:
            candidates.extend(self.atlas.neighbors(primary_id, limit=6))
        return tuple(
            pattern_id
            for pattern_id in dict.fromkeys(candidates)
            if pattern_id != primary_id and pattern_id in self.atlas.by_id
        )[: self.config.quotient_secondary_limit]

    def _widen_root_supporting_face(self) -> EdgeStats | None:
        node = self.root
        if len(node.edges) >= self._child_limit(node):
            return None
        desired = min(
            self._pattern_limit(node),
            len(self._candidate_pattern_ids(node)),
        )
        while len(node.expanded_patterns) < desired:
            if self._choose_new_pattern(node) is None:
                break
        patterns = sorted(
            node.expanded_patterns,
            key=lambda pattern_id: (
                self.root_lp_attempt_counts[pattern_id],
                node.pattern_action_counts.get(pattern_id, 0),
                -self.knowledge.pattern_score("filler", pattern_id),
                pattern_id,
            ),
        )
        for trial, pattern_id in enumerate(
            patterns[: self.config.root_lp_pattern_trials]
        ):
            self.root_lp_attempt_counts[pattern_id] += 1
            nonce = (
                node.visits * self.config.root_lp_pattern_trials
                + trial
                + node.pattern_action_counts.get(pattern_id, 0)
            )
            actions = self.actions.supporting_face_actions(
                node.state,
                pattern_id,
                nonce=nonce,
                count=self.config.supporting_face_lp_actions,
            )
            for action in actions:
                if (
                    action.action_id in node.edges
                    or action.action_id in node.rejected_action_ids
                    or action.action_id in node.retired_action_ids
                ):
                    continue
                moved = self._transition(node.state, action)
                if moved is None:
                    node.rejected_action_ids.add(action.action_id)
                    continue
                if self._state_key(moved.state) in node.child_keys:
                    node.rejected_action_ids.add(action.action_id)
                    continue
                candidate = self._edge_for(node, action, moved)
                if candidate is not None:
                    self.stats["root_supporting_face_expansions"] += 1
                    return self._install_edge(node, *candidate)
        self.stats["root_supporting_face_fallbacks"] += 1
        return None

    def _widen_once(self, node: GameNode) -> EdgeStats | None:
        if node is self.root:
            edge = self._widen_root_supporting_face()
            if edge is not None:
                return edge
        return super()._widen_once(node)

    def _actions_for_pattern(
        self,
        node: GameNode,
        pattern_id: str,
        lane: str,
    ) -> list[MacroAction]:
        actions = super()._actions_for_pattern(node, pattern_id, lane)
        secondaries = self._secondary_charts(node, pattern_id)
        if node.state.role == "filler":
            actions.extend(
                self.actions.quotient_filler_actions(
                    node.state,
                    pattern_id,
                    secondaries,
                    lane=lane,
                )
            )
            if (
                lane != "fine"
                and node.rank <= self.config.supporting_face_max_rank
            ):
                actions.extend(
                    self.actions.supporting_face_actions(
                        node.state,
                        pattern_id,
                        nonce=node.visits + node.action_lane_cursor,
                        count=self.config.supporting_face_lp_actions,
                    )
                )
        else:
            pattern = self.atlas.pattern(pattern_id)
            if (
                pattern_id in node.context_pattern_ids
                or pattern.source == "online_stabilizer"
            ):
                actions.extend(
                    self.actions.ridge_carve_actions(
                        node.state,
                        pattern_id,
                        secondaries,
                    )
                )
        return list(
            {
                action.action_id: action
                for action in actions
                if action.action_id not in node.edges
                and action.action_id not in node.rejected_action_ids
                and action.action_id not in node.retired_action_ids
            }.values()
        )

    def _approximate_action_score(
        self,
        node: GameNode,
        action: MacroAction,
    ) -> float:
        score = super()._approximate_action_score(node, action)
        if action.source == "ridge_carve_exact":
            return score + 1.35
        if action.source == "quotient_ridge_basis":
            return score + 1.45
        if action.source == "fullspace_supporting_face_lp":
            return score + 2.00
        if action.source.startswith("quotient_lower_face"):
            return score + 0.30
        if action.source.startswith("quotient_rank_slice"):
            return score + 0.28
        if action.source.startswith("quotient_intersection"):
            return score + 0.16
        if action.source.startswith("quotient_union"):
            return score + 0.10
        return score

    def _free_filler_ratio(self) -> float:
        if not self.discovered:
            return self.config.free_filler_initial_ratio
        corrector_discoveries = sum(
            any(
                action["kind"] in ("remove", "rewrite")
                for action in event.get("path", ())
            )
            for event in self.discovery_events
        )
        if corrector_discoveries == 0:
            return max(
                self.config.unproductive_corrector_filler_floor,
                self.config.free_filler_floor,
            )
        progress = 1.0 - math.exp(
            -corrector_discoveries / self.config.adaptive_replay_class_scale
        )
        ridge_stats = self.actions.quotient_stats
        attempts = max(1, ridge_stats["ridge_frontiers_opened"])
        ridge_yield = min(1.0, ridge_stats["ridge_exits_consumed"] / attempts)
        progress = min(1.0, progress + 0.20 * ridge_yield)
        span = self.config.free_filler_initial_ratio - self.config.free_filler_floor
        return self.config.free_filler_initial_ratio - span * progress

    def result(self) -> dict[str, object]:
        result = super().result()
        result["algorithm"] = "bell422_fullspace_quotient_ridge_mcts_v26"
        result["features"].update(
            {
                "fullspace_geometric_transpositions": True,
                "pattern_ids_are_action_charts": True,
                "quotient_common_refinement_actions": True,
                "partial_orbit_rank_slices": True,
                "fullspace_supporting_face_lp": True,
                "rank_directed_ridge_carving": True,
                "persistent_unseen_ridge_exits": True,
                "reduced_to_full_lift": False,
            }
        )
        result["prior_boundary"]["search_reads"] = [
            "complete_nodes.csv: subgroup metadata",
            "nodes/*/orbit_blocks.csv: action charts only",
        ]
        result["tree"]["geometric_context_merges"] = self.stats[
            "merged_pattern_contexts"
        ]
        result["tree"]["subgroup_lanes"] = False
        result["quotient_game"] = {
            "config": {
                key: value
                for key, value in asdict(self.config).items()
                if key
                in {
                    "quotient_candidate_limit",
                    "quotient_secondary_limit",
                    "partial_rank_checks",
                    "ridge_rank_checks",
                    "ridge_basis_trials",
                    "ridge_basis_rank_checks",
                    "supporting_face_lp_actions",
                    "supporting_face_max_rank",
                    "root_pattern_widening_k0",
                    "root_pattern_widening_alpha",
                    "root_pattern_widening_beta",
                    "root_action_widening_k0",
                    "root_action_widening_alpha",
                    "root_action_widening_beta",
                    "root_max_children",
                    "root_lp_pattern_trials",
                    "quotient_cache_size",
                }
            },
            "root_lp_attempt_counts": dict(self.root_lp_attempt_counts),
            **self.actions.to_dict(),
        }
        return result


def run_search(
    config: QuotientSearchConfig,
    *,
    handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    output_path: Path | str | None = None,
    checkpoint_paths: Mapping[int, Path | str] | None = None,
) -> dict[str, object]:
    engine = Bell422QuotientSearch(config, handoff_root=handoff_root)
    result = engine.run(checkpoint_paths=checkpoint_paths)
    if output_path is not None:
        engine._write_result(result, Path(output_path))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bell 4-2-2 quotient-guided full-space MCTS"
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=4_222_026)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("ITERATION", "PATH"),
        default=[],
    )
    parser.add_argument("--max-depth", type=int, default=112)
    parser.add_argument("--min-corrector-rank", type=int, default=60)
    parser.add_argument("--tail-checks", type=int, default=24)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-wall-hours", type=float)
    parser.add_argument("--quotient-candidate-limit", type=int, default=8)
    parser.add_argument("--quotient-secondary-limit", type=int, default=2)
    parser.add_argument("--partial-rank-checks", type=int, default=8)
    parser.add_argument("--ridge-rank-checks", type=int, default=24)
    parser.add_argument("--ridge-basis-trials", type=int, default=3)
    parser.add_argument("--ridge-basis-rank-checks", type=int, default=192)
    parser.add_argument("--supporting-face-lp-actions", type=int, default=1)
    parser.add_argument("--supporting-face-max-rank", type=int, default=-1)
    parser.add_argument("--no-catalog", action="store_true")
    parser.add_argument("--no-canonicalize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = QuotientSearchConfig(
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
    )
    output = args.output or (
        DEFAULT_RUNS_DIR
        / f"bell422_fullspace_quotient_i{config.iterations}_v26.json"
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
                "discovered_class_ids": result["coverage"][
                    "discovered_class_ids"
                ],
                "quotient_stats": result["quotient_game"]["quotient_stats"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
