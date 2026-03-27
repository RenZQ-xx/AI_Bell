from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster 44-boundary state cloud into unsupervised boundary prototypes.")
    parser.add_argument("--state-cloud", type=Path, required=True)
    parser.add_argument("--source-class-id", type=int, default=44)
    parser.add_argument("--top-k-per-source", type=int, default=12)
    parser.add_argument("--max-prototypes-per-source", type=int, default=8)
    parser.add_argument("--min-jaccard-distance", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jaccard_distance(left: list[int], right: list[int]) -> float:
    left_on = {index for index, value in enumerate(left) if int(value) != 0}
    right_on = {index for index, value in enumerate(right) if int(value) != 0}
    union = left_on | right_on
    if not union:
        return 0.0
    intersection = left_on & right_on
    return 1.0 - (len(intersection) / len(union))


def candidate_rank(record: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(record["boundary_score"]),
        -float(record["hard_energy"]),
        float(record["hamming_distance"]),
    )


def build_prototype_entry(record: dict[str, object], prototype_index: int) -> dict[str, object]:
    return {
        "prototype_index": int(prototype_index),
        "source_seed_index": int(record["source_seed_index"]),
        "source_class_id": int(record["source_class_id"]),
        "candidate_type": str(record["candidate_type"]),
        "sample_index": int(record["sample_index"]),
        "boundary_score": float(record["boundary_score"]),
        "hard_energy": float(record["hard_energy"]),
        "hamming_distance": int(record["hamming_distance"]),
        "jaccard_to_source": float(record["jaccard_to_source"]),
        "cardinality": int(record["cardinality"]),
        "mask": list(record["mask"]),
    }


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    with args.state_cloud.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["source_class_id"]) != int(args.source_class_id):
                continue
            if bool(record["exact"]):
                continue
            records.append(record)

    grouped: dict[int, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(int(record["source_seed_index"]), []).append(record)

    prototypes_by_source: list[dict[str, object]] = []
    total_selected = 0
    for source_seed_index in sorted(grouped):
        source_records = sorted(grouped[source_seed_index], key=candidate_rank, reverse=True)[: args.top_k_per_source]
        selected: list[dict[str, object]] = []
        for record in source_records:
            mask = list(record["mask"])
            if any(jaccard_distance(mask, list(existing["mask"])) < float(args.min_jaccard_distance) for existing in selected):
                continue
            selected.append(record)
            if len(selected) >= args.max_prototypes_per_source:
                break
        prototypes = [build_prototype_entry(record, prototype_index=index) for index, record in enumerate(selected)]
        total_selected += len(prototypes)
        prototypes_by_source.append(
            {
                "source_seed_index": int(source_seed_index),
                "num_candidates_considered": len(source_records),
                "num_selected_prototypes": len(prototypes),
                "prototypes": prototypes,
            }
        )

    payload = {
        "source_class_id": int(args.source_class_id),
        "state_cloud": str(args.state_cloud),
        "num_records": len(records),
        "num_sources": len(prototypes_by_source),
        "num_selected_prototypes": total_selected,
        "parameters": {
            "top_k_per_source": int(args.top_k_per_source),
            "max_prototypes_per_source": int(args.max_prototypes_per_source),
            "min_jaccard_distance": float(args.min_jaccard_distance),
        },
        "prototypes_by_source": prototypes_by_source,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(
        {
            "source_class_id": payload["source_class_id"],
            "num_records": payload["num_records"],
            "num_sources": payload["num_sources"],
            "num_selected_prototypes": payload["num_selected_prototypes"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
