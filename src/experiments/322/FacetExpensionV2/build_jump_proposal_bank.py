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
    parser = argparse.ArgumentParser(description="Build a first-pass jump proposal bank from source seeds to corridor prototypes.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize_transition(source_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, object]:
    source_bool = source_mask >= 0.5
    target_bool = target_mask >= 0.5
    add_mask = (~source_bool) & target_bool
    remove_mask = source_bool & (~target_bool)
    keep_mask = source_bool & target_bool
    return {
        "num_add": int(add_mask.sum().item()),
        "num_remove": int(remove_mask.sum().item()),
        "num_keep": int(keep_mask.sum().item()),
        "add_indices": torch.nonzero(add_mask, as_tuple=False).view(-1).cpu().tolist(),
        "remove_indices": torch.nonzero(remove_mask, as_tuple=False).view(-1).cpu().tolist(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    points = generate_points_322().to(device)
    seed_items = load_seed_bank(points=points)
    seed_masks = torch.tensor([item.mask for item in seed_items], dtype=torch.float32)

    prototype_payload = json.loads(args.prototypes.read_text(encoding="utf-8"))
    source_seed_indices = [int(index) for index in prototype_payload["source_seed_indices"]]
    selected_prototypes = list(prototype_payload["selected_prototypes"])

    proposals = []
    for source_seed_index in source_seed_indices:
        source_item = seed_items[source_seed_index]
        source_mask = seed_masks[source_seed_index]
        ranked_targets = []
        for proto in selected_prototypes:
            target_seed_index = int(proto["seed_index"])
            target_item = seed_items[target_seed_index]
            target_mask = seed_masks[target_seed_index]
            transition = summarize_transition(source_mask, target_mask)
            ranked_targets.append(
                {
                    "target_seed_index": target_seed_index,
                    "target_class_id": int(target_item.class_id),
                    "target_example_id": int(target_item.example_id),
                    "prototype_jaccard_distance": float(proto["best_jaccard_distance"]),
                    "prototype_hamming_distance": int(proto["best_hamming_distance"]),
                    **transition,
                }
            )
        ranked_targets.sort(
            key=lambda item: (
                item["prototype_jaccard_distance"],
                item["prototype_hamming_distance"],
                item["num_add"] + item["num_remove"],
                item["target_class_id"],
            )
        )
        proposals.append(
            {
                "source_seed_index": source_seed_index,
                "source_class_id": int(source_item.class_id),
                "source_example_id": int(source_item.example_id),
                "targets": ranked_targets,
            }
        )

    payload = {
        "source_class_id": int(prototype_payload["source_class_id"]),
        "source_seed_indices": source_seed_indices,
        "num_prototype_targets": len(selected_prototypes),
        "jump_proposals": proposals,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_class_id": payload["source_class_id"],
                "num_source_seeds": len(source_seed_indices),
                "num_prototype_targets": len(selected_prototypes),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
