from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    EXP322_DIR = CURRENT_DIR.parent
    DIFFUCO_DIR = EXP322_DIR / "DIffUCO"
    V1_DIR = EXP322_DIR / "FacetExpansionV1"
    sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR)]
    from geometry import generate_points_322
    from seed_bank import load_seed_bank
else:
    from ..DIffUCO.geometry import generate_points_322
    from ..FacetExpansionV1.seed_bank import load_seed_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V2 inventory for 44-only escape corridor experiments.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--source-class-id", type=int, default=44)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    points = generate_points_322().to(device)
    seed_items = load_seed_bank(points=points)

    source_seeds = []
    target_by_class: dict[int, list[dict[str, object]]] = defaultdict(list)

    for seed_index, item in enumerate(seed_items):
        payload = {
            "seed_index": seed_index,
            "class_id": int(item.class_id),
            "example_id": int(item.example_id),
            "cardinality": int(item.cardinality),
            "row": list(item.row),
        }
        if int(item.class_id) == int(args.source_class_id):
            source_seeds.append(payload)
        else:
            target_by_class[int(item.class_id)].append(payload)

    inventory = {
        "source_class_id": int(args.source_class_id),
        "num_seed_items": len(seed_items),
        "num_source_seeds": len(source_seeds),
        "source_seed_indices": [item["seed_index"] for item in source_seeds],
        "source_seeds": source_seeds,
        "target_class_ids": sorted(target_by_class.keys()),
        "num_target_classes": len(target_by_class),
        "targets_by_class": {
            str(class_id): target_by_class[class_id] for class_id in sorted(target_by_class.keys())
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(
        {
            "source_class_id": inventory["source_class_id"],
            "num_source_seeds": inventory["num_source_seeds"],
            "source_seed_indices": inventory["source_seed_indices"],
            "num_target_classes": inventory["num_target_classes"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
