from __future__ import annotations

"""Facet-blind structural subset probes for the Bell 4-2-2 v25 search."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT, HandoffPatternAtlas
from .v25_search import Bell422V25Search, V25SearchConfig


STRUCTURES = ("C2 x C2", "C4", "C6", "S3")
PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    "default": {},
    "depth64": {
        "max_depth": 64,
    },
    "candidate_half": {
        "filler_candidate_limit": 16,
        "corrector_candidate_limit": 24,
    },
    "rollout_half": {
        "rollout_pattern_candidates": 4,
        "rollout_action_candidates": 6,
    },
    "tail12": {
        "tail_checks": 12,
    },
    "novelty_strong": {
        "known_class_reward": 0.0,
        "known_penalty_cap": 0.75,
        "support_penalty_cap": 0.75,
        "repeated_terminal_penalty": 0.30,
        "known_class_frequency_penalty": 0.35,
    },
    "exploration_high": {
        "exploration": 1.80,
        "rollout_random_action_probability": 0.30,
    },
    "corrector50": {
        "facet_replay_interval": 2,
        "rollout_corrector_probability": 0.96,
    },
    "balanced": {
        "max_depth": 80,
        "rank_checks_per_widen": 4,
        "filler_candidate_limit": 24,
        "corrector_candidate_limit": 32,
        "rollout_pattern_candidates": 6,
        "rollout_action_candidates": 8,
        "tail_checks": 16,
        "stabilizer_probes": 128,
    },
    "efficient": {
        "max_depth": 64,
        "rank_checks_per_widen": 4,
        "filler_candidate_limit": 16,
        "corrector_candidate_limit": 24,
        "rollout_pattern_candidates": 4,
        "rollout_action_candidates": 6,
        "tail_checks": 12,
        "stabilizer_probes": 96,
    },
}


def select_pattern_ids(
    atlas: HandoffPatternAtlas,
    *,
    structures: Sequence[str],
    limit: int | None,
    seed: int,
) -> tuple[str, ...]:
    selected = [
        pattern
        for pattern in atlas.patterns
        if pattern.source == "handoff_subgroup"
        and (not structures or pattern.structure in structures)
    ]
    if limit is None or len(selected) <= limit:
        return tuple(pattern.pattern_id for pattern in selected)

    lanes = {
        structure: [
            pattern
            for pattern in selected
            if pattern.structure == structure
        ]
        for structure in STRUCTURES
    }
    rng = random.Random(seed)
    for patterns in lanes.values():
        patterns.sort(
            key=lambda pattern: (
                pattern.block_count,
                pattern.irredundant_vertex_count,
                pattern.pattern_id,
            )
        )
        rng.shuffle(patterns)
    result: list[str] = []
    while len(result) < limit and any(lanes.values()):
        for structure in STRUCTURES:
            if lanes[structure]:
                result.append(lanes[structure].pop().pattern_id)
                if len(result) >= limit:
                    break
    return tuple(result)


def build_config(
    *,
    iterations: int,
    seed: int,
    profile: str,
    progress_interval: int,
) -> V25SearchConfig:
    values: dict[str, object] = {
        "iterations": iterations,
        "seed": seed,
        "progress_interval": progress_interval,
    }
    values.update(PROFILE_OVERRIDES[profile])
    return V25SearchConfig(**values)


def run_probe(
    *,
    handoff_root: Path,
    iterations: int,
    seed: int,
    profile: str,
    structures: Sequence[str],
    pattern_limit: int | None,
    progress_interval: int,
    output: Path,
) -> dict[str, object]:
    full_atlas = HandoffPatternAtlas(handoff_root)
    selected = select_pattern_ids(
        full_atlas,
        structures=structures,
        limit=pattern_limit,
        seed=seed,
    )
    atlas = HandoffPatternAtlas(
        handoff_root,
        allowed_pattern_ids=selected,
    )
    config = build_config(
        iterations=iterations,
        seed=seed,
        profile=profile,
        progress_interval=progress_interval,
    )
    engine = Bell422V25Search(
        config,
        handoff_root=handoff_root,
        atlas=atlas,
    )
    result = engine.run()
    result["subset_probe"] = {
        "selection_uses_facet_labels": False,
        "selection_fields": [
            "structure",
            "pattern_id",
            "seed",
        ],
        "structures": list(structures) if structures else list(STRUCTURES),
        "pattern_limit": pattern_limit,
        "selected_pattern_count": len(selected),
        "profile": profile,
        "profile_overrides": PROFILE_OVERRIDES[profile],
        "effective_config": asdict(config),
    }
    engine._write_result(result, output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a facet-blind structural subset probe for Bell 4-2-2 v25"
    )
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4_222_025)
    parser.add_argument("--profile", choices=tuple(PROFILE_OVERRIDES), default="default")
    parser.add_argument("--structure", action="append", choices=STRUCTURES, default=[])
    parser.add_argument("--pattern-limit", type=int)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pattern_limit is not None and args.pattern_limit <= 0:
        raise SystemExit("--pattern-limit must be positive")
    result = run_probe(
        handoff_root=args.handoff_root,
        iterations=args.iterations,
        seed=args.seed,
        profile=args.profile,
        structures=args.structure,
        pattern_limit=args.pattern_limit,
        progress_interval=args.progress_interval,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile": args.profile,
                "structures": args.structure or list(STRUCTURES),
                "selected_pattern_count": result["subset_probe"][
                    "selected_pattern_count"
                ],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
                "discovered_class_ids": result["coverage"][
                    "discovered_class_ids"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
