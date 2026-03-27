from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys as _sys

    CURRENT_DIR = Path(__file__).resolve().parent
    DIFFUCO_DIR = CURRENT_DIR.parent / "DIffUCO"
    V1_DIR = CURRENT_DIR.parent / "FacetExpansionV1"
    _sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR)]
    from geometry import generate_points_322
    from seed_bank import load_seed_bank
else:
    from ..DIffUCO.geometry import generate_points_322
    from ..FacetExpansionV1.seed_bank import load_seed_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2 jump-proposal inference across seeds.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jump-bank", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-class-id", type=int, default=44)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--jump-top-k", type=int, default=3)
    parser.add_argument("--jump-rounds", type=int, default=6)
    parser.add_argument("--jump-quality-bonus", type=float, default=2.5)
    parser.add_argument("--allow-nonexact-jumps", action="store_true")
    parser.add_argument("--max-bridge-children", type=int, default=2)
    parser.add_argument("--bridge-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--escape-scorer", type=Path, default=None)
    parser.add_argument(
        "--escape-mode",
        type=str,
        choices=("additive", "tiebreak", "bucket_tiebreak"),
        default="bucket_tiebreak",
    )
    parser.add_argument("--escape-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_cli_path(path: Path) -> Path:
    if os.name == "nt" and path.is_absolute() and str(path).startswith(("/home/", "/mnt/")):
        return Path("\\\\wsl$\\Ubuntu") / str(path).lstrip("/").replace("/", "\\")
    return path


def run_inference(args: argparse.Namespace, script: Path, seed_index: int, output_path: Path) -> dict:
    command = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(args.checkpoint),
        "--jump-bank",
        str(args.jump_bank),
        "--device",
        args.device,
        "--seed-index",
        str(seed_index),
        "--num-samples",
        str(args.num_samples),
        "--archive-size",
        str(args.archive_size),
        "--rounds",
        str(args.rounds),
        "--frontier-size",
        str(args.frontier_size),
        "--temperature",
        str(args.temperature),
        "--jump-top-k",
        str(args.jump_top_k),
        "--jump-rounds",
        str(args.jump_rounds),
        "--jump-quality-bonus",
        str(args.jump_quality_bonus),
        "--max-bridge-children",
        str(args.max_bridge_children),
        "--bridge-similarity-threshold",
        str(args.bridge_similarity_threshold),
    ]
    if args.allow_nonexact_jumps:
        command.append("--allow-nonexact-jumps")
    if args.escape_scorer is not None:
        command += [
            "--escape-scorer",
            str(args.escape_scorer),
            "--escape-mode",
            str(args.escape_mode),
            "--escape-weight",
            str(args.escape_weight),
        ]
    command += [
        "--output",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main() -> None:
    args = parse_args()
    args.checkpoint = normalize_cli_path(args.checkpoint)
    args.jump_bank = normalize_cli_path(args.jump_bank)
    args.output_dir = normalize_cli_path(args.output_dir)
    if args.escape_scorer is not None:
        args.escape_scorer = normalize_cli_path(args.escape_scorer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "infer_jump_escape.py"

    points = generate_points_322().to(torch.device("cpu"))
    seed_items = load_seed_bank(points=points)
    matching = [index for index, item in enumerate(seed_items) if int(item.class_id) == int(args.seed_class_id)]
    seed_indices = matching[args.seed_start : args.seed_start + args.seed_count]

    per_seed = []
    escaped_seed_indices = []
    escaped_class_histogram: dict[str, int] = {}

    for seed_index in seed_indices:
        output_path = args.output_dir / f"seed_{seed_index:03d}.json"
        payload = run_inference(args, script=script, seed_index=seed_index, output_path=output_path)
        summary = payload["summary"]
        seed_class_id = int(summary["seed_class_id"])
        matched_classes = [int(class_id) for class_id in summary["matched_classes"]]
        escaped_classes = [class_id for class_id in matched_classes if class_id != seed_class_id]
        if escaped_classes:
            escaped_seed_indices.append(seed_index)
            for class_id in escaped_classes:
                key = str(class_id)
                escaped_class_histogram[key] = escaped_class_histogram.get(key, 0) + 1
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
                "matched_classes": matched_classes,
                "escaped_classes": escaped_classes,
                "first_escape_round": first_escape_round,
                "canonical_count": summary["canonical_count"],
                "output": str(output_path),
            }
        )
        print(json.dumps(per_seed[-1], ensure_ascii=False))

    aggregate = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "jump_bank": str(args.jump_bank),
            "seed_class_id": args.seed_class_id,
            "seed_count": len(seed_indices),
            "rounds": args.rounds,
            "num_samples": args.num_samples,
            "jump_top_k": args.jump_top_k,
            "jump_rounds": args.jump_rounds,
            "jump_quality_bonus": args.jump_quality_bonus,
            "allow_nonexact_jumps": args.allow_nonexact_jumps,
            "max_bridge_children": args.max_bridge_children,
            "bridge_similarity_threshold": args.bridge_similarity_threshold,
        },
        "aggregate": {
            "total_seeds_evaluated": len(per_seed),
            "num_seeds_escaped_initial_class": len(escaped_seed_indices),
            "fraction_seeds_escaped_initial_class": (len(escaped_seed_indices) / len(per_seed)) if per_seed else 0.0,
            "escaped_seed_indices": escaped_seed_indices,
            "escaped_class_histogram": {
                key: escaped_class_histogram[key] for key in sorted(escaped_class_histogram, key=lambda item: int(item))
            },
        },
        "per_seed": per_seed,
    }
    (args.output_dir / "multiseed_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
