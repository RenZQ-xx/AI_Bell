from __future__ import annotations

"""Run the modular MCTS strict-log probe."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from baseline.orbit_blocks import build_orbit_patterns_from_support
from baseline.reference_classes import parse_example_rows, support_mask_from_row
from baseline.scorer import ExpansionScorer, ScorerConfig, exact_class_id

from .search import MCTSConfig, MCTSResult, run_mcts_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the restored strict-log MCTS baseline.")
    parser.add_argument("--row-class", type=int, default=1)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--rare-target-classes", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260502])
    parser.add_argument("--restarts-per-seed", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--exploration-constant", type=float, default=1.4)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-temperature", type=float, default=0.85)
    parser.add_argument("--expansion-candidate-pool", type=int, default=16)
    parser.add_argument("--rollout-candidate-pool", type=int, default=4)
    parser.add_argument("--widening-score-batch", type=int, default=4)
    parser.add_argument("--widening-score-batch-max", type=int, default=16)
    parser.add_argument("--widening-score-batch-scale", type=float, default=1.0)
    parser.add_argument("--widening-score-batch-beta", type=float, default=0.5)
    parser.add_argument("--rollout-score-batch", type=int, default=8)
    parser.add_argument("--progressive-k0", type=int, default=1)
    parser.add_argument("--progressive-alpha", type=float, default=1.0)
    parser.add_argument("--progressive-beta", type=float, default=0.5)
    parser.add_argument("--rank24-entrance-exists-weight", type=float, default=6.0)
    parser.add_argument("--terminal-scoring-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--dynamic-new-class-score", type=float, default=100.0)
    parser.add_argument("--dynamic-known-class-score", type=float, default=10.0)
    parser.add_argument("--dynamic-frequent-class-score", type=float, default=-5.0)
    parser.add_argument("--dynamic-frequent-class-threshold", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("src/mcts/runs/modular_mcts_seed20260502_probe4.json"))
    return parser.parse_args()


def compact_label(label: str) -> str:
    class_id = exact_class_id(label)
    if class_id is not None:
        return f"exact:{class_id}"
    return label


def summarize(results: Sequence[MCTSResult], rare_target_classes: set[int]) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    encountered: Counter[str] = Counter()
    exact_class_counts: Counter[int] = Counter()
    for result in results:
        if result.best is not None:
            label_counts[compact_label(result.best.label)] += 1
        for label, count in result.encountered_label_counts.items():
            compact = compact_label(label)
            encountered[compact] += int(count)
            class_id = exact_class_id(label)
            if class_id is not None:
                exact_class_counts[class_id] += 1
    opened_rare = sorted(
        class_id
        for label in label_counts
        if (class_id := exact_class_id(label)) in rare_target_classes
    )
    encountered_rare = sorted(
        class_id
        for label, count in encountered.items()
        if count > 0 and (class_id := exact_class_id(label)) in rare_target_classes
    )
    return {
        "label_counts": dict(sorted(label_counts.items())),
        "exact_class_counts": {str(key): value for key, value in sorted(exact_class_counts.items())},
        "opened_rare_target_classes": opened_rare,
        "rare_target_coverage_count": len(opened_rare),
        "encountered_label_counts": dict(sorted(encountered.items())),
        "encountered_rare_target_classes": encountered_rare,
        "encountered_rare_target_coverage_count": len(encountered_rare),
    }


def main() -> None:
    args = parse_args()
    examples = parse_example_rows()
    support = support_mask_from_row(examples[int(args.row_class)][int(args.rep_index)])
    patterns = build_orbit_patterns_from_support(
        support,
        class_id=int(args.row_class),
        rep_index=int(args.rep_index),
        max_patterns=int(args.pattern_index) + 1,
    )
    pattern = patterns[int(args.pattern_index)]
    rare = {int(value) for value in args.rare_target_classes}
    target = {int(value) for value in args.target_classes}
    scorer = ExpansionScorer(
        blocks=pattern.orbits,
        rare_target_classes=rare,
        target_classes=target,
        config=ScorerConfig(
            rank24_entrance_exists_weight=float(args.rank24_entrance_exists_weight),
            terminal_scoring_mode=str(args.terminal_scoring_mode),
            dynamic_new_class_score=float(args.dynamic_new_class_score),
            dynamic_known_class_score=float(args.dynamic_known_class_score),
            dynamic_frequent_class_score=float(args.dynamic_frequent_class_score),
            dynamic_frequent_class_threshold=int(args.dynamic_frequent_class_threshold),
        ),
    )

    results: list[MCTSResult] = []
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        for restart_index in range(int(args.restarts_per_seed)):
            cfg = MCTSConfig(
                iterations=int(args.iterations),
                max_depth=int(args.max_depth),
                exploration_constant=float(args.exploration_constant),
                discount=float(args.discount),
                prior_temperature=float(args.prior_temperature),
                rollout_temperature=float(args.rollout_temperature),
                expansion_candidate_pool=int(args.expansion_candidate_pool),
                rollout_candidate_pool=int(args.rollout_candidate_pool),
                widening_score_batch=int(args.widening_score_batch),
                widening_score_batch_max=int(args.widening_score_batch_max),
                widening_score_batch_scale=float(args.widening_score_batch_scale),
                widening_score_batch_beta=float(args.widening_score_batch_beta),
                rollout_score_batch=int(args.rollout_score_batch),
                progressive_k0=int(args.progressive_k0),
                progressive_alpha=float(args.progressive_alpha),
                progressive_beta=float(args.progressive_beta),
                seed=int(seed) + 1009 * int(restart_index),
            )
            result = run_mcts_search(scorer, config=cfg)
            results.append(result)
            payload = result.to_dict()
            payload["seed"] = int(seed)
            payload["restart_index"] = int(restart_index)
            if result.best is not None:
                payload["label"] = compact_label(result.best.label)
            runs.append(payload)

    output = {
        "meta": {
            "script": "src/mcts/run_mcts_probe.py",
            "row_class": int(args.row_class),
            "rep_index": int(args.rep_index),
            "pattern_index": int(args.pattern_index),
            "seeds": [int(value) for value in args.seeds],
            "restarts_per_seed": int(args.restarts_per_seed),
            "iterations": int(args.iterations),
            "max_depth": int(args.max_depth),
            "exploration_constant": float(args.exploration_constant),
            "discount": float(args.discount),
            "prior_temperature": float(args.prior_temperature),
            "rollout_temperature": float(args.rollout_temperature),
            "expansion_candidate_pool": int(args.expansion_candidate_pool),
            "rollout_candidate_pool": int(args.rollout_candidate_pool),
            "widening_score_batch": int(args.widening_score_batch),
            "widening_score_batch_max": int(args.widening_score_batch_max),
            "widening_score_batch_scale": float(args.widening_score_batch_scale),
            "widening_score_batch_beta": float(args.widening_score_batch_beta),
            "rollout_score_batch": int(args.rollout_score_batch),
            "progressive_k0": int(args.progressive_k0),
            "progressive_alpha": float(args.progressive_alpha),
            "progressive_beta": float(args.progressive_beta),
            "rank24_entrance_exists_weight": float(args.rank24_entrance_exists_weight),
            "terminal_scoring_mode": str(args.terminal_scoring_mode),
            "dynamic_new_class_score": float(args.dynamic_new_class_score),
            "dynamic_known_class_score": float(args.dynamic_known_class_score),
            "dynamic_frequent_class_score": float(args.dynamic_frequent_class_score),
            "dynamic_frequent_class_threshold": int(args.dynamic_frequent_class_threshold),
        },
        "runs": runs,
        "summary": summarize(results, rare),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
