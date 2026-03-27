from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys as _sys

    CURRENT_DIR = Path(__file__).resolve().parent
    DIFFUCO_DIR = CURRENT_DIR.parent / "DIffUCO"
    _sys.path[:0] = [str(DIFFUCO_DIR), str(CURRENT_DIR)]
    from geometry import generate_points_322
    from seed_bank import load_seed_bank
else:
    from ..DIffUCO.geometry import generate_points_322
    from .seed_bank import load_seed_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate long-horizon controlled inference across multiple seeds.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--seed-class-id", type=int, default=None)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--distill-scorer", type=Path, default=None)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-tiebreak-threshold", type=float, default=0.15)
    parser.add_argument(
        "--distill-mode",
        type=str,
        choices=("additive", "tiebreak", "bucket_tiebreak", "critical_bucket_tiebreak"),
        default="additive",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_inference(args: argparse.Namespace, inference_script: Path, seed_index: int, output_path: Path) -> dict:
    command = [
        sys.executable,
        str(inference_script),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        args.device,
        "--seed-index",
        str(seed_index),
        "--archive-size",
        str(args.archive_size),
        "--frontier-size",
        str(args.frontier_size),
        "--rounds",
        str(args.rounds),
        "--num-samples",
        str(args.num_samples),
        "--temperature",
        str(args.temperature),
        "--output",
        str(output_path),
    ]
    if args.distill_scorer is not None:
        command.extend(
            [
                "--distill-scorer",
                str(args.distill_scorer),
                "--distill-weight",
                str(args.distill_weight),
                "--distill-tiebreak-threshold",
                str(args.distill_tiebreak_threshold),
                "--distill-mode",
                args.distill_mode,
            ]
        )
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference_script = Path(__file__).resolve().parent / "infer_longhorizon.py"

    points = generate_points_322().to(torch.device("cpu"))
    seed_items = load_seed_bank(points=points)
    if args.seed_class_id is None:
        seed_indices = list(range(args.seed_start, min(args.seed_start + args.seed_count, len(seed_items))))
    else:
        matching_indices = [index for index, item in enumerate(seed_items) if int(item.class_id) == int(args.seed_class_id)]
        seed_indices = matching_indices[args.seed_start : args.seed_start + args.seed_count]

    per_seed = []
    non44_seed_indices = []
    escaped_seed_indices = []
    class_histogram: dict[str, int] = {}
    escaped_class_histogram: dict[str, int] = {}
    canonical_histogram: dict[str, int] = {}

    for seed_index in seed_indices:
        output_path = args.output_dir / f"seed_{seed_index:03d}.json"
        payload = run_inference(args, inference_script=inference_script, seed_index=seed_index, output_path=output_path)
        summary = payload["summary"]
        seed_class_id = int(summary["seed_class_id"])
        matched_classes = [int(class_id) for class_id in summary["matched_classes"]]
        non44_classes = [class_id for class_id in matched_classes if class_id != 44]
        escaped_classes = [class_id for class_id in matched_classes if class_id != seed_class_id]
        for class_id in matched_classes:
            key = str(class_id)
            class_histogram[key] = class_histogram.get(key, 0) + 1
        for class_id in escaped_classes:
            key = str(class_id)
            escaped_class_histogram[key] = escaped_class_histogram.get(key, 0) + 1
        for result in payload["results"]:
            canonical_key = result.get("canonical_key")
            if canonical_key is None:
                continue
            canonical_histogram[canonical_key] = canonical_histogram.get(canonical_key, 0) + 1
        if non44_classes:
            non44_seed_indices.append(seed_index)
        if escaped_classes:
            escaped_seed_indices.append(seed_index)
        first_escape_round = None
        for round_summary in summary["round_summaries"]:
            round_matches = [int(class_id) for class_id in round_summary.get("matched_classes", [])]
            if any(class_id != seed_class_id for class_id in round_matches):
                first_escape_round = int(round_summary["round"])
                break
        per_seed.append(
            {
                "seed_index": seed_index,
                "seed_class_id": seed_class_id,
                "exact_count": summary["exact_count"],
                "canonical_count": summary["canonical_count"],
                "matched_classes": matched_classes,
                "non44_classes": non44_classes,
                "escaped_classes": escaped_classes,
                "escaped_initial_class": bool(escaped_classes),
                "first_escape_round": first_escape_round,
                "round_summaries": summary["round_summaries"],
                "output": str(output_path),
            }
        )
        print(
            json.dumps(
                {
                    "seed_index": seed_index,
                    "seed_class_id": seed_class_id,
                    "matched_classes": matched_classes,
                    "non44_classes": non44_classes,
                    "escaped_classes": escaped_classes,
                    "first_escape_round": first_escape_round,
                    "canonical_count": summary["canonical_count"],
                },
                ensure_ascii=False,
            )
        )

    aggregate = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "seed_start": args.seed_start,
            "seed_count": len(seed_indices),
            "seed_class_id": args.seed_class_id,
            "archive_size": args.archive_size,
            "frontier_size": args.frontier_size,
            "rounds": args.rounds,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "distill_scorer": str(args.distill_scorer) if args.distill_scorer is not None else None,
            "distill_weight": args.distill_weight,
            "distill_tiebreak_threshold": args.distill_tiebreak_threshold,
            "distill_mode": args.distill_mode,
        },
        "aggregate": {
            "total_seeds_evaluated": len(per_seed),
            "num_seeds_with_non44": len(non44_seed_indices),
            "fraction_seeds_with_non44": (len(non44_seed_indices) / len(per_seed)) if per_seed else 0.0,
            "non44_seed_indices": non44_seed_indices,
            "num_seeds_escaped_initial_class": len(escaped_seed_indices),
            "fraction_seeds_escaped_initial_class": (len(escaped_seed_indices) / len(per_seed)) if per_seed else 0.0,
            "escaped_seed_indices": escaped_seed_indices,
            "class_histogram": {key: class_histogram[key] for key in sorted(class_histogram, key=lambda item: int(item))},
            "escaped_class_histogram": {
                key: escaped_class_histogram[key] for key in sorted(escaped_class_histogram, key=lambda item: int(item))
            },
            "top_canonical_histogram": sorted(
                canonical_histogram.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10],
        },
        "per_seed": per_seed,
    }

    summary_path = args.output_dir / "multiseed_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
