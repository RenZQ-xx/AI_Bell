from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a jump bank from unsupervised boundary prototypes.")
    parser.add_argument("--boundary-prototypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.boundary_prototypes.read_text(encoding="utf-8"))
    jump_proposals: list[dict[str, object]] = []
    total_targets = 0
    for source_entry in payload.get("prototypes_by_source", []):
        targets: list[dict[str, object]] = []
        for prototype in source_entry.get("prototypes", []):
            targets.append(
                {
                    "proposal_type": "boundary_prototype",
                    "prototype_index": int(prototype["prototype_index"]),
                    "source_seed_index": int(source_entry["source_seed_index"]),
                    "boundary_score": float(prototype["boundary_score"]),
                    "hard_energy": float(prototype["hard_energy"]),
                    "hamming_distance": int(prototype["hamming_distance"]),
                    "jaccard_to_source": float(prototype["jaccard_to_source"]),
                    "cardinality": int(prototype["cardinality"]),
                    "mask": list(prototype["mask"]),
                }
            )
        total_targets += len(targets)
        jump_proposals.append(
            {
                "source_seed_index": int(source_entry["source_seed_index"]),
                "targets": targets,
            }
        )

    output_payload = {
        "source_class_id": int(payload["source_class_id"]),
        "boundary_prototypes": str(args.boundary_prototypes),
        "num_source_seeds": len(jump_proposals),
        "num_targets": total_targets,
        "jump_proposals": jump_proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(
        {
            "source_class_id": output_payload["source_class_id"],
            "num_source_seeds": output_payload["num_source_seeds"],
            "num_targets": output_payload["num_targets"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
