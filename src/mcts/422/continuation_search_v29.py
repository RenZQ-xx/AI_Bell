from __future__ import annotations

"""Observed-facet continuation and renewable Filler entrances for Bell 4-2-2."""

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .atlas import DEFAULT_HANDOFF_ROOT, IDENTITY_PATTERN_ID
from .geometry import FacetResult
from .model import EdgeStats, GameNode, MacroAction, SearchState
from .quotient_search_v27 import (
    AdaptiveQuotientSearchConfig,
    Bell422AdaptiveQuotientSearch,
    build_parser as build_v27_parser,
)
from .v25_search import DEFAULT_RUNS_DIR, PathStep, TerminalOutcome, Transition


@dataclass(frozen=True)
class ContinuationSearchConfig(AdaptiveQuotientSearchConfig):
    enable_local_continuation: bool = True
    skip_geometry_noop_corrections: bool = True
    fresh_root_entrances: bool = True
    noop_rollout_refills: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.noop_rollout_refills < 0:
            raise ValueError("noop_rollout_refills must be nonnegative")


@dataclass
class FacetAnchor:
    class_id: str
    canonical_key: tuple
    support_word: int
    witness: tuple[MacroAction, ...]
    discovered_iteration: int
    pattern_ids: set[str] = field(default_factory=set)
    starts: int = 0


def _witness_cost(witness: Sequence[MacroAction]) -> tuple[int, int]:
    return sum(action.kind in ("remove", "rewrite") for action in witness), len(witness)


class Bell422ContinuationSearch(Bell422AdaptiveQuotientSearch):
    config: ContinuationSearchConfig

    def __init__(self, config: ContinuationSearchConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.anchors: dict[tuple, dict[int, FacetAnchor]] = defaultdict(dict)
        self.anchor_pattern_starts: Counter[tuple[int, str]] = Counter()
        self.episode_origin: FacetAnchor | None = None
        self.episode_prefix: tuple[MacroAction, ...] = ()
        self.episode_pattern_id = IDENTITY_PATTERN_ID
        self.noop_actions: set[tuple[int, str]] = set()
        self.noop_charts: dict[tuple, set[str]] = defaultdict(set)
        self.noop_examples: list[dict[str, object]] = []
        self.archived_root_edges: dict[str, EdgeStats] = {}
        self.root_seen_child_keys: set[tuple] = set()
        self.root_edge_birth: dict[str, int] = {}
        self.root_edge_last_novelty: dict[str, int] = {}
        self.root_archive_events: list[dict[str, object]] = []
        self.growth_windows: list[dict[str, object]] = []
        self._last_window_classes = 0

    def _continuation_due(self, iteration: int) -> bool:
        mandatory = bool(
            self.config.facet_replay_interval
            and iteration % self.config.facet_replay_interval == 0
        )
        periodic = (
            1.0 / self.config.facet_replay_interval
            if self.config.facet_replay_interval else 0.0
        )
        self.adaptive_replay_balance += max(0.0, 1.0 - self._free_filler_ratio() - periodic)
        if mandatory:
            self.stats["periodic_replay_slots"] += 1
            return True
        if self.adaptive_replay_balance >= 1.0:
            self.adaptive_replay_balance -= 1.0
            self.stats["adaptive_replay_slots"] += 1
            return True
        self.stats["free_filler_slots"] += 1
        return False

    def _iteration_start(
        self, iteration: int
    ) -> tuple[GameNode, tuple[MacroAction, ...]]:
        self.episode_origin = None
        self.episode_prefix = ()
        self.episode_pattern_id = IDENTITY_PATTERN_ID
        if not self.config.enable_local_continuation:
            return super()._iteration_start(iteration)
        if not self._continuation_due(iteration) or not self.anchors:
            self.stats["root_filler_episodes"] += 1
            return self.root, ()
        family = min(
            self.anchors,
            key=lambda key: (
                self.facet_replay_counts[key], self.endpoint_target_counts[key], repr(key)
            ),
        )
        anchor = min(
            self.anchors[family].values(),
            key=lambda item: (item.starts, _witness_cost(item.witness), item.support_word),
        )
        pattern = min(
            anchor.pattern_ids,
            key=lambda value: (self.anchor_pattern_starts[anchor.support_word, value], value),
        )
        state = SearchState(anchor.support_word, "corrector", 0, pattern, anchor.support_word)
        node = self._node(state, 79)
        node.context_pattern_ids.update(anchor.pattern_ids)
        # A control edge should not consume the first continuation visit.
        if self._add_stop_edge(node) is not None:
            self.stats["tree_expansions"] += 1
        self.episode_origin = anchor
        self.episode_prefix = anchor.witness
        self.episode_pattern_id = pattern
        anchor.starts += 1
        self.anchor_pattern_starts[anchor.support_word, pattern] += 1
        self.facet_replay_counts[family] += 1
        self.stats["replay_episodes"] += 1
        self.stats["local_continuation_episodes"] += 1
        self.stats["historical_replay_steps_saved"] += len(anchor.witness)
        self.stats["historical_corrections_reset"] += _witness_cost(anchor.witness)[0]
        return node, ()

    def _merge_noop_charts(self, node: GameNode) -> None:
        charts = self.noop_charts.get(self._state_key(node.state), ())
        before = len(node.context_pattern_ids)
        node.context_pattern_ids.update(charts)
        self.stats["noop_chart_context_merges"] += len(node.context_pattern_ids) - before

    def _node(self, state: SearchState, rank: int) -> GameNode:
        node = super()._node(state, rank)
        if hasattr(self, "noop_charts"):
            self._merge_noop_charts(node)
        return node

    def _preserve_noop_chart(self, state: SearchState, action: MacroAction) -> None:
        charts = self.noop_charts[self._state_key(state)]
        before = len(charts)
        charts.add(action.pattern_id)
        self.stats["noop_chart_registrations"] += len(charts) - before
        node = self.nodes.get(self._state_key(state))
        if node is not None:
            self._merge_noop_charts(node)

    def _transition(self, state: SearchState, action: MacroAction) -> Transition | None:
        check_noop = (
            self.config.skip_geometry_noop_corrections
            and state.role == "corrector"
            and action.kind in ("remove", "rewrite")
        )
        key = state.support_word, action.action_id
        if check_noop and key in self.noop_actions:
            self._preserve_noop_chart(state, action)
            self.stats["geometry_noop_cache_hits"] += 1
            return None
        moved = super()._transition(state, action)
        if (
            check_noop and moved is not None
            and moved.state.role == "corrector"
            and moved.state.support_word == state.support_word
        ):
            self.noop_actions.add(key)
            self._preserve_noop_chart(state, action)
            self.stats["geometry_noop_corrections_skipped"] += 1
            if len(self.noop_examples) < 25:
                self.noop_examples.append({
                    "iteration": self.current_iteration,
                    "support_word_hex": f"0x{state.support_word:064x}",
                    "corrections_used": state.corrections_used,
                    "action": action.to_dict(),
                })
            return None
        return moved

    def _actions_for_pattern(self, node: GameNode, pattern_id: str, lane: str) -> list[MacroAction]:
        self._merge_noop_charts(node)
        actions = super()._actions_for_pattern(node, pattern_id, lane)
        if not self.config.skip_geometry_noop_corrections or node.state.role != "corrector":
            return actions
        viable = []
        for action in actions:
            if (node.state.support_word, action.action_id) in self.noop_actions:
                self._preserve_noop_chart(node.state, action)
                self.stats["noop_candidates_filtered"] += 1
            else:
                viable.append(action)
        return viable

    def _rollout_edge(self, node: GameNode, policy: str) -> tuple[EdgeStats, Transition] | None:
        if not self.config.skip_geometry_noop_corrections:
            return super()._rollout_edge(node, policy)
        for attempt in range(self.config.noop_rollout_refills + 1):
            before = len(self.noop_actions), self.stats["noop_candidates_filtered"]
            result = super()._rollout_edge(node, policy)
            after = len(self.noop_actions), self.stats["noop_candidates_filtered"]
            if result is not None and result[0].action.kind != "stop":
                return result
            if after == before:
                return result
            if attempt == self.config.noop_rollout_refills:
                self.stats["noop_rollout_refill_limit_hits"] += 1
                return result
            # Rejected geometry must not occupy all of the policy's candidate slots.
            self._merge_noop_charts(node)
            self.stats["noop_rollout_refills"] += 1
        return None

    def _pattern_limit(self, node: GameNode) -> int:
        if node is self.root and self.config.fresh_root_entrances:
            value = self.config.root_pattern_widening_k0 + self.config.root_pattern_widening_alpha * (
                max(1, self.current_iteration - 1) ** self.config.root_pattern_widening_beta
            )
            return max(1, min(len(self.pattern_order), math.floor(value)))
        return super()._pattern_limit(node)

    def _child_limit(self, node: GameNode) -> int:
        if node is self.root and self.config.fresh_root_entrances:
            value = self.config.root_action_widening_k0 + self.config.root_action_widening_alpha * (
                max(1, self.current_iteration - 1) ** self.config.root_action_widening_beta
            )
            return min(self.config.root_max_children, max(1, math.floor(value)))
        return super()._child_limit(node)

    def _archive_root_edge(self) -> None:
        node = self.root
        edge = min(node.edges.values(), key=lambda item: (
            self.root_edge_last_novelty.get(item.action.action_id, 0),
            self.root_edge_birth.get(item.action.action_id, 0),
            -item.visits, item.action.action_id,
        ))
        action_id = edge.action.action_id
        del node.edges[action_id]
        node.retired_action_ids.add(action_id)
        node.pattern_action_counts[edge.action.pattern_id] = max(
            0, node.pattern_action_counts.get(edge.action.pattern_id, 0) - 1
        )
        node.child_keys = {item.child_key for item in node.edges.values()}
        self.archived_root_edges[action_id] = edge
        self.stats["root_entrances_archived"] += 1
        self.root_archive_events.append({
            "iteration": self.current_iteration,
            "action_id": action_id,
            "pattern_id": edge.action.pattern_id,
            "visits": edge.visits,
            "discovery_hits": edge.discovery_hits,
            "last_novelty_iteration": self.root_edge_last_novelty.get(action_id, 0),
        })

    def _install_edge(self, node: GameNode, edge: EdgeStats, moved: Transition) -> EdgeStats:
        installed = super()._install_edge(node, edge, moved)
        if node is self.root:
            self.root_seen_child_keys.add(edge.child_key)
            self.root_edge_birth[edge.action.action_id] = self.current_iteration
        return installed

    def _widen_root_supporting_face(self) -> EdgeStats | None:
        if not self.config.fresh_root_entrances:
            return super()._widen_root_supporting_face()
        node = self.root
        desired = min(self._pattern_limit(node), len(self._candidate_pattern_ids(node)))
        while len(node.expanded_patterns) < desired:
            if self._choose_new_pattern(node) is None:
                break
        patterns = sorted(node.expanded_patterns, key=lambda pattern_id: (
            self.root_lp_attempt_counts[pattern_id],
            node.pattern_action_counts.get(pattern_id, 0),
            -self.knowledge.pattern_score("filler", pattern_id), pattern_id,
        ))
        for pattern_id in patterns[:self.config.root_lp_pattern_trials]:
            nonce = self.stats["fresh_root_lp_attempts"]
            self.stats["fresh_root_lp_attempts"] += 1
            self.root_lp_attempt_counts[pattern_id] += 1
            actions = self.actions.supporting_face_actions(
                node.state, pattern_id, nonce=nonce,
                count=self.config.supporting_face_lp_actions,
            )
            for action in actions:
                if action.action_id in node.edges or action.action_id in node.rejected_action_ids or action.action_id in node.retired_action_ids:
                    continue
                moved = self._transition(node.state, action)
                if moved is None or self._state_key(moved.state) in self.root_seen_child_keys:
                    node.rejected_action_ids.add(action.action_id)
                    continue
                candidate = self._edge_for(node, action, moved)
                if candidate is None:
                    continue
                # Make room only after finding a genuinely new concrete entrance.
                while len(node.edges) >= self._child_limit(node):
                    self._archive_root_edge()
                self.stats["root_supporting_face_expansions"] += 1
                self.stats["fresh_root_entrances"] += 1
                return self._install_edge(node, *candidate)
        self.stats["root_supporting_face_fallbacks"] += 1
        return None

    def _terminal_outcome(
        self, facet: FacetResult, *, path: Sequence[PathStep], depth: int, source: str
    ) -> TerminalOutcome:
        canonical = self._family_key(facet) if facet.valid and facet.class_id else None
        old_witness = self.facet_witnesses.get(canonical)
        outcome = super()._terminal_outcome(facet, path=path, depth=depth, source=source)
        if not outcome.valid or canonical is None:
            return outcome
        local_actions = tuple(step.edge.action for step in path)
        full_witness = self.episode_prefix + local_actions
        if full_witness:
            self.facet_witnesses[canonical] = min(
                (witness for witness in (old_witness, full_witness) if witness),
                key=_witness_cost,
            )
        patterns = set(self.actions.facet_patterns.get(facet.support_word, ()))
        patterns.add(local_actions[-1].pattern_id if local_actions else (
            self.episode_pattern_id
        ))
        patterns.intersection_update(self.atlas.by_id)
        if not patterns:
            patterns.add(IDENTITY_PATTERN_ID)
        anchor = self.anchors[canonical].get(facet.support_word)
        if anchor is None:
            anchor = FacetAnchor(
                facet.class_id, canonical, facet.support_word, full_witness,
                self.current_iteration, patterns,
            )
            self.anchors[canonical][facet.support_word] = anchor
            self.stats["concrete_facet_anchors"] += 1
        else:
            anchor.pattern_ids.update(patterns)
            if full_witness and _witness_cost(full_witness) < _witness_cost(anchor.witness):
                anchor.witness = full_witness
        if outcome.new_class:
            event = self.discovered[facet.class_id]
            event["local_path"] = event["path"]
            event["local_depth"] = depth
            event["path"] = [action.to_dict() for action in full_witness]
            event["depth"] = len(full_witness)
            event["episode_kind"] = "continuation" if self.episode_origin else "root_filler"
            if self.episode_origin:
                event["continuation_origin"] = {
                    "class_id": self.episode_origin.class_id,
                    "support_word_hex": f"0x{self.episode_origin.support_word:064x}",
                    "anchor_observed_iteration": self.episode_origin.discovered_iteration,
                    "historical_path_length": len(self.episode_prefix),
                    "historical_corrections": _witness_cost(self.episode_prefix)[0],
                    "local_corrections_reset_to": 0,
                }
                self.stats["continuation_new_classes"] += 1
            else:
                self.stats["root_filler_new_classes"] += 1
                if path and path[0].node is self.root:
                    self.root_edge_last_novelty[path[0].edge.action.action_id] = self.current_iteration
        return outcome

    def run_iteration(self, iteration: int) -> None:
        super().run_iteration(iteration)
        if iteration % 1000 == 0 or iteration == self.config.iterations:
            self.growth_windows.append({
                "iteration": iteration,
                "class_count": len(self.discovered),
                "new_classes": len(self.discovered) - self._last_window_classes,
                "elapsed_seconds": time.perf_counter() - self.started_at,
                "local_continuation_episodes": self.stats["local_continuation_episodes"],
                "root_filler_episodes": self.stats["root_filler_episodes"],
                "root_lp_attempts": self.stats["fresh_root_lp_attempts"],
                "root_lp_expansions": self.stats["root_supporting_face_expansions"],
                "noop_corrections_skipped": self.stats["geometry_noop_corrections_skipped"],
            })
            self._last_window_classes = len(self.discovered)

    def result(self) -> dict[str, object]:
        result = super().result()
        result["algorithm"] = "bell422_observed_facet_continuation_mcts_v29"
        result["features"].update({
            "observed_facet_local_continuation": self.config.enable_local_continuation,
            "geometry_noop_corrections_are_chart_only": self.config.skip_geometry_noop_corrections,
            "renewable_root_filler_entrances": self.config.fresh_root_entrances,
            "historical_witness_is_provenance_only": self.config.enable_local_continuation,
        })
        result["continuation_game"] = {
            "class_anchor_count": len(self.anchors),
            "concrete_anchor_count": sum(map(len, self.anchors.values())),
            "episode_correction_limit": self.config.extended_max_corrections,
            "noop_rollout_refills_per_step": self.config.noop_rollout_refills,
            "growth_windows": self.growth_windows,
            "noop_examples": self.noop_examples,
            "unique_noop_actions": len(self.noop_actions),
            "active_root_edges": len(self.root.edges),
            "archived_root_edges": len(self.archived_root_edges),
            "root_archive_events": self.root_archive_events,
            "anchors": [{
                "class_id": anchor.class_id,
                "support_word_hex": f"0x{anchor.support_word:064x}",
                "starts": anchor.starts,
                "witness_length": len(anchor.witness),
                "witness_corrections": _witness_cost(anchor.witness)[0],
                "pattern_ids": sorted(anchor.pattern_ids),
            } for group in self.anchors.values() for anchor in group.values()],
        }
        return result


def run_search(
    config: ContinuationSearchConfig, *, handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    output_path: Path | str | None = None,
    checkpoint_paths: Mapping[int, Path | str] | None = None,
) -> dict[str, object]:
    engine = Bell422ContinuationSearch(config, handoff_root=handoff_root)
    result = engine.run(checkpoint_paths=checkpoint_paths)
    if output_path is not None:
        engine._write_result(result, Path(output_path))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_v27_parser()
    parser.description = "Bell 4-2-2 observed-facet continuation MCTS v29"
    parser.set_defaults(iterations=5000, seed=4_222_025)
    parser.add_argument("--no-local-continuation", action="store_true")
    parser.add_argument("--keep-noop-corrections", action="store_true")
    parser.add_argument("--no-fresh-root-entrances", action="store_true")
    parser.add_argument("--noop-rollout-refills", type=int, default=2)
    args = parser.parse_args(argv)
    values = asdict(ContinuationSearchConfig())
    values.update({key: getattr(args, key) for key in values if hasattr(args, key)})
    values.update(
        use_catalog=not args.no_catalog,
        canonicalize_classes=not args.no_canonicalize,
        max_wall_seconds=None if args.max_wall_hours is None else args.max_wall_hours * 3600.0,
        enable_adaptive_schedule=args.enable_late_adaptive_schedule,
        frontier_action_score_weight=0.70 if args.enable_frontier_priority else 0.0,
        frontier_tree_score_weight=0.10 if args.enable_frontier_priority else 0.0,
        enable_local_continuation=not args.no_local_continuation,
        skip_geometry_noop_corrections=not args.keep_noop_corrections,
        fresh_root_entrances=not args.no_fresh_root_entrances,
    )
    config = ContinuationSearchConfig(**values)
    checkpoints = {int(iteration): path for iteration, path in args.checkpoint}
    if any(not 0 < iteration < config.iterations for iteration in checkpoints):
        parser.error("checkpoint iterations must be inside the run")
    output = args.output or DEFAULT_RUNS_DIR / f"bell422_continuation_i{config.iterations}_s{config.seed}_v29.json"
    result = run_search(config, handoff_root=args.handoff_root, output_path=output, checkpoint_paths=checkpoints)
    print(json.dumps({
        "output": str(output),
        "elapsed_seconds": result["runtime"]["elapsed_seconds"],
        "discovered_class_count": result["coverage"]["discovered_class_count"],
        "node_count": result["tree"]["node_count"],
        "growth_windows": result["continuation_game"]["growth_windows"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
