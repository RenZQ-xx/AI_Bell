from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Select first-pass corridor prototypes near the class-44 basin boundary.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--source-class-id", type=int, default=44)
    parser.add_argument("--top-k-global", type=int, default=24)
    parser.add_argument("--top-k-per-class", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jaccard_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_mask = left >= 0.5
    right_mask = right >= 0.5
    intersection = (left_mask & right_mask).sum().item()
    union = (left_mask | right_mask).sum().item()
    if union == 0:
        return 0.0
    return 1.0 - float(intersection / union)


def hamming_distance(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left >= 0.5).ne(right >= 0.5).sum().item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    points = generate_points_322().to(device)
    seed_items = load_seed_bank(points=points)
    seed_masks = torch.tensor([item.mask for item in seed_items], dtype=torch.float32)

    source_indices = [index for index, item in enumerate(seed_items) if int(item.class_id) == int(args.source_class_id)]
    target_indices = [index for index, item in enumerate(seed_items) if int(item.class_id) != int(args.source_class_id)]

    if not source_indices:
        raise ValueError(f"No source seeds found for class {args.source_class_id}")

    candidates: list[dict[str, object]] = []
    for target_index in target_indices:
        target_item = seed_items[target_index]
        target_mask = seed_masks[target_index]
        distances = []
        for source_index in source_indices:
            source_item = seed_items[source_index]
            source_mask = seed_masks[source_index]
            distances.append(
                {
                    "source_seed_index": source_index,
                    "source_example_id": int(source_item.example_id),
                    "jaccard_distance": jaccard_distance(source_mask, target_mask),
                    "hamming_distance": hamming_distance(source_mask, target_mask),
                }
            )
        best = min(distances, key=lambda item: (item["jaccard_distance"], item["hamming_distance"]))
        candidates.append(
            {
                "seed_index": target_index,
                "class_id": int(target_item.class_id),
                "example_id": int(target_item.example_id),
                "cardinality": int(target_item.cardinality),
                "best_source_seed_index": int(best["source_seed_index"]),
                "best_source_example_id": int(best["source_example_id"]),
                "best_jaccard_distance": float(best["jaccard_distance"]),
                "best_hamming_distance": int(best["hamming_distance"]),
                "all_source_distances": distances,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["best_jaccard_distance"],
            item["best_hamming_distance"],
            item["class_id"],
            item["example_id"],
        ),
    )

    per_class_counts: dict[int, int] = {}
    selected = []
    for item in ranked:
        class_id = int(item["class_id"])
        if per_class_counts.get(class_id, 0) >= args.top_k_per_class:
            continue
        selected.append(item)
        per_class_counts[class_id] = per_class_counts.get(class_id, 0) + 1
        if len(selected) >= args.top_k_global:
            break

    payload = {
        "source_class_id": int(args.source_class_id),
        "source_seed_indices": source_indices,
        "num_source_seeds": len(source_indices),
        "num_target_candidates": len(candidates),
        "top_k_global": args.top_k_global,
        "top_k_per_class": args.top_k_per_class,
        "selected_prototypes": selected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_class_id": payload["source_class_id"],
                "num_source_seeds": payload["num_source_seeds"],
                "num_target_candidates": payload["num_target_candidates"],
                "num_selected_prototypes": len(selected),
                "selected_classes": sorted({int(item["class_id"]) for item in selected}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
