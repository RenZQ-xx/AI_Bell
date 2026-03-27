from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep FacetExpansionV1 stability/diversity hyperparameters.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-every", type=int, default=6)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--frontier-sample-size", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_command(command: list[str], workdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=workdir, capture_output=True, text=True, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    current_dir = Path(__file__).resolve().parent
    train_script = current_dir / "train.py"
    eval_script = current_dir / "eval_multiseed.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = [
        {"name": "base", "edit_min_penalty_scale": 3.0, "entropy_floor_weight": 0.05},
        {"name": "lighter_floor", "edit_min_penalty_scale": 3.0, "entropy_floor_weight": 0.02},
        {"name": "strong_edit", "edit_min_penalty_scale": 4.0, "entropy_floor_weight": 0.05},
        {"name": "balanced", "edit_min_penalty_scale": 4.0, "entropy_floor_weight": 0.02},
    ]

    summaries = []
    for config in sweep:
        run_dir = args.output_dir / config["name"]
        train_dir = run_dir / "train"
        eval_dir = run_dir / "eval"
        train_dir.mkdir(parents=True, exist_ok=True)

        train_command = [
            sys.executable,
            str(train_script),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--sample-every",
            str(args.sample_every),
            "--rollout-steps",
            str(args.rollout_steps),
            "--frontier-sample-size",
            str(args.frontier_sample_size),
            "--prefer-novel-frontier",
            "--frontier-repeat-cap",
            "1",
            "--edit-min-penalty-scale",
            str(config["edit_min_penalty_scale"]),
            "--entropy-floor-weight",
            str(config["entropy_floor_weight"]),
            "--output-dir",
            str(train_dir),
        ]
        run_command(train_command, workdir=current_dir)
        training_summary = load_json(train_dir / "training_summary.json")
        history = training_summary["history"]
        last_epoch = history[-1]
        status = last_epoch.get("status", "ok")
        is_stable = "status" not in last_epoch

        eval_summary = None
        if is_stable:
            eval_command = [
                sys.executable,
                str(eval_script),
                "--checkpoint",
                str(train_dir / "facet_expander_v1.pt"),
                "--device",
                args.device,
                "--stochastic",
                "--prefer-novel",
                "--temperature",
                str(args.temperature),
                "--rounds",
                str(args.rounds),
                "--num-samples",
                str(args.num_samples),
                "--seed-count",
                str(args.seed_count),
                "--output-dir",
                str(eval_dir),
            ]
            run_command(eval_command, workdir=current_dir)
            eval_summary = load_json(eval_dir / "multiseed_summary.json")

        summary = {
            "name": config["name"],
            "edit_min_penalty_scale": config["edit_min_penalty_scale"],
            "entropy_floor_weight": config["entropy_floor_weight"],
            "stable": is_stable,
            "train_last_epoch": last_epoch,
            "eval_aggregate": eval_summary["aggregate"] if eval_summary is not None else None,
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir) if eval_summary is not None else None,
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    summary_path = args.output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
