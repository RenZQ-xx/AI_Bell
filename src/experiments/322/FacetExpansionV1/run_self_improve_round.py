from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one self-improvement round: rollout -> train -> eval.")
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
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--bootstrap-scorer", type=Path, default=None)
    parser.add_argument("--distill-mode", type=str, default="critical_bucket_tiebreak")
    parser.add_argument("--distill-tiebreak-threshold", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    current_dir = Path(__file__).resolve().parent
    builder = current_dir / "build_self_improve_dataset.py"
    trainer = current_dir / "train_search_distill_scorer.py"
    evaluator = current_dir / "eval_multiseed_longhorizon.py"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "self_improve.jsonl"
    scorer_path = args.output_dir / "self_improve_scorer.pt"
    eval_dir = args.output_dir / "eval16_self_improve"

    build_cmd = [
        sys.executable,
        str(builder),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        args.device,
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
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
        "--history-length",
        str(args.history_length),
        "--distill-mode",
        args.distill_mode,
        "--distill-tiebreak-threshold",
        str(args.distill_tiebreak_threshold),
        "--export-embeddings",
        "--output",
        str(dataset_path),
    ]
    if args.seed_class_id is not None:
        build_cmd.extend(["--seed-class-id", str(args.seed_class_id)])
    if args.bootstrap_scorer is not None:
        build_cmd.extend(["--distill-scorer", str(args.bootstrap_scorer)])
    run_command(build_cmd)

    train_cmd = [
        sys.executable,
        str(trainer),
        "--dataset",
        str(dataset_path),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--objective",
        "pairwise",
        "--use-learned-embeddings",
        "--critical-pairs-only",
        "--min-target-gap",
        "0.5",
        "--min-coverage-gap",
        "1.0",
        "--device",
        args.device,
        "--output",
        str(scorer_path),
    ]
    run_command(train_cmd)

    eval_cmd = [
        sys.executable,
        str(evaluator),
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        args.device,
        "--seed-start",
        str(args.seed_start),
        "--seed-count",
        str(args.seed_count),
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
        "--distill-scorer",
        str(scorer_path),
        "--distill-mode",
        args.distill_mode,
        "--distill-tiebreak-threshold",
        str(args.distill_tiebreak_threshold),
        "--output-dir",
        str(eval_dir),
    ]
    if args.seed_class_id is not None:
        eval_cmd.extend(["--seed-class-id", str(args.seed_class_id)])
    run_command(eval_cmd)

    summary = {
        "dataset": str(dataset_path),
        "dataset_summary": str(dataset_path.with_suffix(".summary.json")),
        "scorer": str(scorer_path),
        "scorer_summary": str(scorer_path.with_suffix(".json")),
        "eval_summary": str(eval_dir / "multiseed_summary.json"),
    }
    (args.output_dir / "round_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
