from __future__ import annotations

import argparse
import json
from collections import Counter
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
    from seed_bank import load_seed_bank, seed_tensor
else:
    from ..DIffUCO.geometry import generate_points_322
    from ..FacetExpansionV1.seed_bank import load_seed_bank, seed_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a targeted 44->43 jump bank from reverse-collapse templates.")
    parser.add_argument("--reverse-collapse", type=Path, required=True)
    parser.add_argument("--target-class-id", type=int, default=43)
    parser.add_argument("--source-class-id", type=int, default=44)
    parser.add_argument("--template-limit", type=int, default=8)
    parser.add_argument("--hotspot-limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_target_section(path: Path, target_class_id: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload["target_classes"]:
        if int(item["target_class_id"]) == int(target_class_id):
            return item
    raise ValueError(f"target_class_id={target_class_id} not found in {path}")


def main() -> None:
    args = parse_args()
    target = load_target_section(args.reverse_collapse, target_class_id=args.target_class_id)

    points = generate_points_322()
    seed_items = load_seed_bank(points=points)
    seed_masks = seed_tensor(seed_items, device=torch.device("cpu"))
    source_indices = [index for index, item in enumerate(seed_items) if int(item.class_id) == int(args.source_class_id)]

    hotspot_remove = [int(item["index"]) for item in target.get("top_remove_indices", [])[: args.hotspot_limit]]
    hotspot_add = [int(item["index"]) for item in target.get("top_add_indices", [])[: args.hotspot_limit]]

    templates = []
    seen_templates: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for item in target.get("closest_examples", []):
        remove = tuple(sorted(int(index) for index in item["remove_indices"]))
        add = tuple(sorted(int(index) for index in item["add_indices"]))
        key = (remove, add)
        if key in seen_templates:
            continue
        seen_templates.add(key)
        templates.append(
            {
                "template_type": "closest_pair",
                "target_row_index": int(item["target_row_index"]),
                "nearest_44_row_index": int(item["nearest_44_row_index"]),
                "hamming_to_44": int(item["hamming_to_44"]),
                "jaccard_to_44": float(item["jaccard_to_44"]),
                "remove_indices": list(remove),
                "add_indices": list(add),
            }
        )
        if len(templates) >= args.template_limit:
            break

    # Add a couple of hotspot-only templates for a broader class-level prior.
    hotspot_templates = [
        {
            "template_type": "hotspot_prefix",
            "remove_indices": hotspot_remove[:4],
            "add_indices": hotspot_add[:2],
        },
        {
            "template_type": "hotspot_prefix",
            "remove_indices": hotspot_remove[:6],
            "add_indices": hotspot_add[:4],
        },
    ]
    for item in hotspot_templates:
        remove = tuple(sorted(int(index) for index in item["remove_indices"]))
        add = tuple(sorted(int(index) for index in item["add_indices"]))
        key = (remove, add)
        if key in seen_templates:
            continue
        seen_templates.add(key)
        templates.append(item)

    jump_proposals = []
    for source_seed_index in source_indices:
        source_mask = seed_masks[source_seed_index].clone()
        targets = []
        for proposal_index, template in enumerate(templates):
            mask = source_mask.clone()
            for index in template["remove_indices"]:
                if 0 <= int(index) < mask.shape[0]:
                    mask[int(index)] = 0.0
            for index in template["add_indices"]:
                if 0 <= int(index) < mask.shape[0]:
                    mask[int(index)] = 1.0
            targets.append(
                {
                    "proposal_type": "target43_template",
                    "proposal_index": int(proposal_index),
                    "source_seed_index": int(source_seed_index),
                    "target_class_id": int(args.target_class_id),
                    "template_type": template["template_type"],
                    "hamming_to_44": int(template.get("hamming_to_44", len(template["remove_indices"]) + len(template["add_indices"]))),
                    "template_score": float(template.get("jaccard_to_44", 0.0)),
                    "remove_indices": [int(index) for index in template["remove_indices"]],
                    "add_indices": [int(index) for index in template["add_indices"]],
                    "num_remove": len(template["remove_indices"]),
                    "num_add": len(template["add_indices"]),
                    "mask": mask.int().tolist(),
                }
            )
        jump_proposals.append(
            {
                "source_seed_index": int(source_seed_index),
                "source_class_id": int(args.source_class_id),
                "targets": targets,
            }
        )

    payload = {
        "source_class_id": int(args.source_class_id),
        "target_class_id": int(args.target_class_id),
        "source_seed_indices": source_indices,
        "num_templates": len(templates),
        "hotspot_remove_indices": hotspot_remove,
        "hotspot_add_indices": hotspot_add,
        "jump_proposals": jump_proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"num_templates": len(templates), "num_source_seeds": len(source_indices)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
