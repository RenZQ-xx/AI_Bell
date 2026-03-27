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
    from seed_bank import load_seed_bank, seed_tensor
else:
    from ..DIffUCO.geometry import generate_points_322
    from ..FacetExpansionV1.seed_bank import load_seed_bank, seed_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anti-44 jump proposals from the unsupervised 44-boundary state cloud.")
    parser.add_argument("--state-cloud", type=Path, required=True)
    parser.add_argument("--source-class-id", type=int, default=44)
    parser.add_argument("--top-records-per-source", type=int, default=96)
    parser.add_argument("--jump-sizes", type=str, default="4,8,12,16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_jump_sizes(spec: str) -> list[int]:
    sizes = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    if not sizes:
        raise ValueError("At least one jump size is required.")
    return sorted(set(sizes))


def load_records(path: Path, source_class_id: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["source_class_id"]) != int(source_class_id):
                continue
            if bool(record["exact"]):
                continue
            records.append(record)
    return records


def score_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(records, key=lambda item: float(item["boundary_score"]), reverse=True)
    total = max(len(ranked), 1)
    for rank, record in enumerate(ranked):
        # Rank weight keeps all top boundary states positive even when raw scores are negative.
        record["_anti44_rank_weight"] = float(total - rank)
    return ranked


def build_proposals_for_source(
    *,
    source_seed_index: int,
    source_mask: torch.Tensor,
    records: list[dict[str, object]],
    jump_sizes: list[int],
) -> list[dict[str, object]]:
    feature_dim = int(source_mask.shape[0])
    remove_scores = [0.0 for _ in range(feature_dim)]
    add_scores = [0.0 for _ in range(feature_dim)]

    source_bool = (source_mask >= 0.5).cpu()
    for record in records:
        mask = torch.tensor(record["mask"], dtype=torch.float32)
        mask_bool = mask >= 0.5
        weight = float(record["_anti44_rank_weight"])
        for index in range(feature_dim):
            if bool(source_bool[index]) and not bool(mask_bool[index]):
                remove_scores[index] += weight
            if (not bool(source_bool[index])) and bool(mask_bool[index]):
                add_scores[index] += weight

    ranked_remove = [
        index
        for index in sorted(range(feature_dim), key=lambda idx: remove_scores[idx], reverse=True)
        if bool(source_bool[index]) and remove_scores[index] > 0.0
    ]
    ranked_add = [
        index
        for index in sorted(range(feature_dim), key=lambda idx: add_scores[idx], reverse=True)
        if (not bool(source_bool[index])) and add_scores[index] > 0.0
    ]

    proposals: list[dict[str, object]] = []
    for proposal_index, jump_size in enumerate(jump_sizes):
        actual_size = min(int(jump_size), len(ranked_remove), len(ranked_add))
        if actual_size <= 0:
            continue
        remove_indices = ranked_remove[:actual_size]
        add_indices = ranked_add[:actual_size]
        mask = source_mask.clone().cpu()
        mask[remove_indices] = 0.0
        mask[add_indices] = 1.0
        anti44_score = float(sum(remove_scores[idx] for idx in remove_indices) + sum(add_scores[idx] for idx in add_indices))
        proposals.append(
            {
                "proposal_type": "anti44_swap",
                "proposal_index": int(proposal_index),
                "source_seed_index": int(source_seed_index),
                "jump_size": int(actual_size),
                "proposal_score": anti44_score / max(actual_size, 1),
                "anti44_score": anti44_score,
                "remove_indices": [int(idx) for idx in remove_indices],
                "add_indices": [int(idx) for idx in add_indices],
                "num_remove": int(actual_size),
                "num_add": int(actual_size),
                "mask": mask.int().tolist(),
            }
        )
    return proposals


def main() -> None:
    args = parse_args()
    jump_sizes = parse_jump_sizes(args.jump_sizes)
    records = score_records(load_records(args.state_cloud, source_class_id=args.source_class_id))

    points = generate_points_322()
    seed_items = load_seed_bank(points=points)
    seed_masks = seed_tensor(seed_items, device=torch.device("cpu"))
    source_indices = [index for index, item in enumerate(seed_items) if int(item.class_id) == int(args.source_class_id)]

    grouped: dict[int, list[dict[str, object]]] = {index: [] for index in source_indices}
    for record in records:
        grouped.setdefault(int(record["source_seed_index"]), []).append(record)

    jump_proposals: list[dict[str, object]] = []
    total_targets = 0
    for source_seed_index in source_indices:
        source_records = grouped.get(source_seed_index, [])[: args.top_records_per_source]
        proposals = build_proposals_for_source(
            source_seed_index=source_seed_index,
            source_mask=seed_masks[source_seed_index],
            records=source_records,
            jump_sizes=jump_sizes,
        )
        total_targets += len(proposals)
        jump_proposals.append(
            {
                "source_seed_index": int(source_seed_index),
                "targets": proposals,
            }
        )

    payload = {
        "source_class_id": int(args.source_class_id),
        "state_cloud": str(args.state_cloud),
        "top_records_per_source": int(args.top_records_per_source),
        "jump_sizes": jump_sizes,
        "num_source_seeds": len(jump_proposals),
        "num_targets": total_targets,
        "jump_proposals": jump_proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_class_id": payload["source_class_id"],
                "num_source_seeds": payload["num_source_seeds"],
                "num_targets": payload["num_targets"],
                "jump_sizes": payload["jump_sizes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
