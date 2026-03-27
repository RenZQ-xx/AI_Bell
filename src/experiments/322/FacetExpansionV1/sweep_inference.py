from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep rollout inference settings for FacetExpansionV1.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_script = Path(__file__).resolve().parent / "inference.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = [
        {"name": "det_r6_n16", "rounds": 6, "num_samples": 16, "stochastic": False, "temperature": 1.0},
        {"name": "stoch_t12_r6_n16", "rounds": 6, "num_samples": 16, "stochastic": True, "temperature": 1.2},
        {"name": "stoch_t15_r6_n16", "rounds": 6, "num_samples": 16, "stochastic": True, "temperature": 1.5},
        {"name": "stoch_t18_r6_n16", "rounds": 6, "num_samples": 16, "stochastic": True, "temperature": 1.8},
        {"name": "stoch_t15_r8_n24", "rounds": 8, "num_samples": 24, "stochastic": True, "temperature": 1.5},
        {"name": "stoch_t20_r8_n24", "rounds": 8, "num_samples": 24, "stochastic": True, "temperature": 2.0},
    ]

    summaries = []
    for config in sweep:
        output_path = args.output_dir / f"{config['name']}.json"
        command = [
            sys.executable,
            str(inference_script),
            "--checkpoint",
            str(args.checkpoint),
            "--device",
            args.device,
            "--seed-index",
            str(args.seed_index),
            "--archive-size",
            str(args.archive_size),
            "--frontier-size",
            str(args.frontier_size),
            "--rounds",
            str(config["rounds"]),
            "--num-samples",
            str(config["num_samples"]),
            "--temperature",
            str(config["temperature"]),
            "--output",
            str(output_path),
        ]
        if config["stochastic"]:
            command.append("--stochastic")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        summary = payload["summary"]
        summaries.append(
            {
                "name": config["name"],
                "stochastic": config["stochastic"],
                "temperature": config["temperature"],
                "rounds": config["rounds"],
                "num_samples": config["num_samples"],
                "exact_count": summary["exact_count"],
                "matched_classes": summary["matched_classes"],
                "canonical_count": summary["canonical_count"],
                "round_summaries": summary["round_summaries"],
                "output": str(output_path),
            }
        )
        print(json.dumps(summaries[-1], ensure_ascii=False))

    summary_path = args.output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
