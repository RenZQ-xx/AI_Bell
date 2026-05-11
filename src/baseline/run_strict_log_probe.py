from __future__ import annotations

"""Run the modular strict-log baseline probe.

This is the clean baseline entry point.  The larger
`legacy_phase_aware_search.py` remains as a historical reproduction snapshot;
this runner uses the modular baseline pieces in `orbit_blocks.py`, `scorer.py`,
and `search.py`.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .orbit_blocks import build_orbit_patterns_from_support
from .reference_classes import parse_example_rows, support_mask_from_row
from .scorer import ExpansionScorer, ScorerConfig, exact_class_id
from .search import BeamSearchConfig, SearchResult, run_beam_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the restored strict-log modular baseline.")
    parser.add_argument("--row-class", type=int, default=1)
    parser.add_argument("--rep-index", type=int, default=1)
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--rare-target-classes", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--target-classes", type=int, nargs="*", default=list(range(1, 47)))
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260502])
    parser.add_argument("--restarts-per-seed", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--candidate-pool", type=int, default=16)
    parser.add_argument("--samples-per-state", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-blocks", type=int, default=60)
    parser.add_argument("--rank24-entrance-exists-weight", type=float, default=6.0)
    parser.add_argument("--terminal-scoring-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--dynamic-new-class-score", type=float, default=100.0)
    parser.add_argument("--dynamic-known-class-score", type=float, default=10.0)
    parser.add_argument("--dynamic-frequent-class-score", type=float, default=-5.0)
    parser.add_argument("--dynamic-frequent-class-threshold", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("src/baseline/runs/modular_strict_log_seed20260502_probe4.json"))
    return parser.parse_args()


def compact_label(label: str) -> str:
    class_id = exact_class_id(label)
    if class_id is not None:
        return f"exact:{class_id}"
    return label


def summarize(results: Sequence[SearchResult], rare_target_classes: set[int]) -> dict[str, object]:
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

    results: list[SearchResult] = []
    runs: list[dict[str, object]] = []
    for seed in args.seeds:
        for restart_index in range(int(args.restarts_per_seed)):
            cfg = BeamSearchConfig(
                beam_width=int(args.beam_width),
                candidate_pool=int(args.candidate_pool),
                samples_per_state=int(args.samples_per_state),
                max_blocks=int(args.max_blocks),
                stochastic=False,
                temperature=float(args.temperature),
                seed=int(seed) + 1009 * int(restart_index),
                beam_diversity_slots=16,
                beam_diversity_rank_start=17,
                beam_diversity_rank_end=21,
                beam_diversity_temperature=0.75,
            )
            result = run_beam_search(scorer, config=cfg)
            results.append(result)
            payload = result.to_dict()
            payload["seed"] = int(seed)
            payload["restart_index"] = int(restart_index)
            if result.best is not None:
                payload["label"] = compact_label(result.best.label)
            runs.append(payload)

    output = {
        "meta": {
            "script": "src/baseline/run_strict_log_probe.py",
            "row_class": int(args.row_class),
            "rep_index": int(args.rep_index),
            "pattern_index": int(args.pattern_index),
            "seeds": [int(value) for value in args.seeds],
            "restarts_per_seed": int(args.restarts_per_seed),
            "beam_width": int(args.beam_width),
            "candidate_pool": int(args.candidate_pool),
            "samples_per_state": int(args.samples_per_state),
            "temperature": float(args.temperature),
            "max_blocks": int(args.max_blocks),
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
