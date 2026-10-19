from __future__ import annotations

"""v29 search with a certified early guard for facet-preserving corrections."""

import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .atlas import DEFAULT_HANDOFF_ROOT
from .continuation_search_v29 import Bell422ContinuationSearch, ContinuationSearchConfig
from .model import MacroAction, SearchState
from .problem import TARGET_AFFINE_RANK
from .quotient_search_v27 import build_parser as build_v27_parser
from .v25_search import DEFAULT_RUNS_DIR, Transition


@dataclass(frozen=True)
class EarlyNoopSearchConfig(ContinuationSearchConfig):
    enable_early_noop_guard: bool = True
    noop_effect_cache_size: int = 400_000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.noop_effect_cache_size < 0:
            raise ValueError("noop_effect_cache_size must be nonnegative")


class Bell422EarlyNoopSearch(Bell422ContinuationSearch):
    config: EarlyNoopSearchConfig

    def __init__(self, config: EarlyNoopSearchConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.noop_effects: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.early_noop_guard_seconds = 0.0

    def _remember_noop_effect(self, key: tuple[int, int]) -> None:
        if not self.config.noop_effect_cache_size:
            return
        self.noop_effects[key] = None
        self.noop_effects.move_to_end(key)
        if len(self.noop_effects) > self.config.noop_effect_cache_size:
            self.noop_effects.popitem(last=False)
            self.stats["noop_effect_cache_evictions"] += 1

    def _record_early_noop(self, state: SearchState, action: MacroAction) -> None:
        self.noop_actions.add((state.support_word, action.action_id))
        self._preserve_noop_chart(state, action)
        self.stats["geometry_noop_corrections_skipped"] += 1
        self.stats["early_noop_actions_skipped"] += 1
        if len(self.noop_examples) < 25:
            self.noop_examples.append({
                "iteration": self.current_iteration,
                "support_word_hex": f"0x{state.support_word:064x}",
                "corrections_used": state.corrections_used,
                "action": action.to_dict(),
            })

    def _is_certified_noop(self, state: SearchState, action: MacroAction) -> bool:
        parent = self.facet_oracle.cache.get(state.support_word)
        if (
            parent is None or not parent.valid
            or parent.support_word != state.support_word
            or parent.affine_rank != TARGET_AFFINE_RANK
        ):
            self.stats["early_noop_uncertified_parents"] += 1
            return False
        retained = state.support_word & ~action.remove_word
        if action.kind == "rewrite":
            retained |= action.add_word
        if retained == state.support_word or retained & ~state.support_word:
            return False
        key = state.support_word, retained
        if key in self.noop_effects:
            self.noop_effects.move_to_end(key)
            self.stats["noop_effect_cache_hits"] += 1
            return True
        self.stats["early_noop_rank_checks"] += 1
        if self.rank_oracle.rank(retained) != TARGET_AFFINE_RANK:
            return False
        # A full-rank subset of a validated facet spans the same supporting plane.
        self._remember_noop_effect(key)
        self.stats["early_noop_certificates"] += 1
        self.stats["uncached_facet_validations_avoided"] += int(retained not in self.facet_oracle.cache)
        return True

    def _transition(self, state: SearchState, action: MacroAction) -> Transition | None:
        if (
            self.config.enable_early_noop_guard
            and self.config.skip_geometry_noop_corrections
            and state.role == "corrector"
            and action.kind in ("remove", "rewrite")
            and (state.support_word, action.action_id) not in self.noop_actions
        ):
            started = time.perf_counter()
            is_noop = self._is_certified_noop(state, action)
            self.early_noop_guard_seconds += time.perf_counter() - started
            if is_noop:
                self._record_early_noop(state, action)
                return None
        return super()._transition(state, action)

    def result(self) -> dict[str, object]:
        result = super().result()
        result["algorithm"] = "bell422_early_noop_continuation_mcts_v30"
        result["features"].update({
            "certified_early_geometry_noop_guard": self.config.enable_early_noop_guard,
            "search_policy_unchanged_from_v29": True,
        })
        result["early_noop_guard"] = {
            "enabled": self.config.enable_early_noop_guard,
            "effect_cache_entries": len(self.noop_effects),
            "effect_cache_limit": self.config.noop_effect_cache_size,
            "rank_checks": self.stats["early_noop_rank_checks"],
            "certificates": self.stats["early_noop_certificates"],
            "effect_cache_hits": self.stats["noop_effect_cache_hits"],
            "effect_cache_evictions": self.stats["noop_effect_cache_evictions"],
            "actions_skipped": self.stats["early_noop_actions_skipped"],
            "uncached_facet_validations_avoided": self.stats["uncached_facet_validations_avoided"],
            "uncertified_parents": self.stats["early_noop_uncertified_parents"],
            "guard_seconds": self.early_noop_guard_seconds,
        }
        return result


def run_search(
    config: EarlyNoopSearchConfig, *, handoff_root: Path | str = DEFAULT_HANDOFF_ROOT,
    output_path: Path | str | None = None,
    checkpoint_paths: Mapping[int, Path | str] | None = None,
) -> dict[str, object]:
    engine = Bell422EarlyNoopSearch(config, handoff_root=handoff_root)
    result = engine.run(checkpoint_paths=checkpoint_paths)
    if output_path is not None:
        engine._write_result(result, Path(output_path))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_v27_parser()
    parser.description = __doc__
    parser.set_defaults(iterations=1000, seed=4_222_025)
    parser.add_argument("--no-local-continuation", action="store_true")
    parser.add_argument("--keep-noop-corrections", action="store_true")
    parser.add_argument("--no-fresh-root-entrances", action="store_true")
    parser.add_argument("--noop-rollout-refills", type=int, default=2)
    parser.add_argument("--no-early-noop-guard", action="store_true")
    parser.add_argument("--noop-effect-cache-size", type=int, default=400_000)
    args = parser.parse_args(argv)
    values = asdict(EarlyNoopSearchConfig())
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
        enable_early_noop_guard=not args.no_early_noop_guard,
    )
    config = EarlyNoopSearchConfig(**values)
    checkpoints = {int(iteration): path for iteration, path in args.checkpoint}
    if any(not 0 < iteration < config.iterations for iteration in checkpoints):
        parser.error("checkpoint iterations must be inside the run")
    output = args.output or DEFAULT_RUNS_DIR / f"bell422_continuation_i{config.iterations}_s{config.seed}_v30.json"
    result = run_search(config, handoff_root=args.handoff_root, output_path=output, checkpoint_paths=checkpoints)
    print(json.dumps({
        "output": str(output), "elapsed_seconds": result["runtime"]["elapsed_seconds"],
        "discovered_class_count": result["coverage"]["discovered_class_count"],
        "node_count": result["tree"]["node_count"],
        "early_noop_guard": result["early_noop_guard"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
