from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize novelty-score behavior from long-horizon inference outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Path to infer_longhorizon JSON output.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload.get("results", [])

    by_round: dict[int, list[dict]] = {}
    for item in results:
        by_round.setdefault(int(item.get("round", 0)), []).append(item)

    round_summary = []
    class_to_scores: dict[int, list[float]] = {}
    for round_index in sorted(by_round):
        round_items = by_round[round_index]
        novelty_scores = [float(item.get("novelty_score", 0.0)) for item in round_items]
        hard_energies = [float(item.get("hard_energy", 0.0)) for item in round_items]
        matched_classes = sorted({int(class_id) for item in round_items for class_id in item.get("matched_classes", [])})
        for item in round_items:
            score = float(item.get("novelty_score", 0.0))
            for class_id in item.get("matched_classes", []):
                class_to_scores.setdefault(int(class_id), []).append(score)
        round_summary.append(
            {
                "round": round_index,
                "count": len(round_items),
                "mean_novelty_score": mean(novelty_scores),
                "min_novelty_score": min(novelty_scores) if novelty_scores else 0.0,
                "max_novelty_score": max(novelty_scores) if novelty_scores else 0.0,
                "mean_hard_energy": mean(hard_energies),
                "matched_classes": matched_classes,
            }
        )

    class_summary = [
        {
            "class_id": class_id,
            "count": len(scores),
            "mean_novelty_score": mean(scores),
            "min_novelty_score": min(scores),
            "max_novelty_score": max(scores),
        }
        for class_id, scores in sorted(class_to_scores.items())
    ]

    output = {
        "input": str(args.input),
        "summary": payload.get("summary", {}),
        "round_summary": round_summary,
        "class_summary": class_summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
