from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    DIFFUCO_DIR = CURRENT_DIR.parent / "DIffUCO"
    sys.path[:0] = [str(DIFFUCO_DIR), str(CURRENT_DIR)]
    from energy import EnergyConfig, GeometricHyperplaneEnergy
    from facet_reference import build_reference_database
    from geometry import generate_points_322
    from inference import classify_hard_mask
    from editor_model import ConditionalEditModel
    from seed_bank import load_seed_bank, seed_tensor
    from symmetry_graph import build_symmetry_graph
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .editor_model import ConditionalEditModel
    from .seed_bank import load_seed_bank, seed_tensor


@dataclass
class TeacherBranch:
    branch_id: int
    mask: torch.Tensor
    history_masks: list[torch.Tensor]
    depth: int
    lineage_record_ids: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a search-distillation dataset from long-horizon teacher rollouts.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--max-children-per-parent", type=int, default=2)
    parser.add_argument("--max-same-class", type=int, default=2)
    parser.add_argument("--export-embeddings", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def candidate_from_edit(seed_mask: torch.Tensor, edit_prob: torch.Tensor) -> torch.Tensor:
    return seed_mask * (1.0 - edit_prob) + (1.0 - seed_mask) * edit_prob


def build_archive_context(seed_mask: torch.Tensor, archive_masks: torch.Tensor | None) -> torch.Tensor:
    if archive_masks is None or archive_masks.numel() == 0:
        return torch.full_like(seed_mask, 0.5)
    mean_archive = archive_masks.float().mean(dim=0, keepdim=True)
    return mean_archive.expand(seed_mask.shape[0], -1)


def build_branch_history_batch(
    branches: list[TeacherBranch],
    num_nodes: int,
    device: torch.device,
    num_samples: int,
    history_length: int,
) -> torch.Tensor:
    history_batch = []
    for branch in branches:
        history = [mask.detach().cpu().float() for mask in branch.history_masks[-history_length:]]
        if len(history) < history_length:
            history = [history[0]] * (history_length - len(history)) + history
        stacked = torch.stack(history, dim=0)
        history_batch.append(stacked.unsqueeze(0).expand(num_samples, -1, -1))
    if not history_batch:
        return torch.zeros(0, history_length, num_nodes, device=device)
    return torch.cat(history_batch, dim=0).to(device)


def jaccard_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_mask = left >= 0.5
    right_mask = right >= 0.5
    intersection = (left_mask & right_mask).sum().item()
    union = (left_mask | right_mask).sum().item()
    if union == 0:
        return 1.0
    return float(intersection / union)


def select_teacher_frontier(
    candidates: list[dict[str, object]],
    seen_canonical: set[str],
    seen_classes: set[int],
    frontier_size: int,
    max_children_per_parent: int,
    max_same_class: int,
) -> list[dict[str, object]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(bool(item["exact"])),
            int(item["canonical_key"] is not None and item["canonical_key"] not in seen_canonical),
            int(any(class_id not in seen_classes for class_id in item["matched_classes"])),
            len([class_id for class_id in item["matched_classes"] if class_id not in seen_classes]),
            float(item["quality_score"]),
        ),
        reverse=True,
    )

    selected: list[dict[str, object]] = []
    used_keys: set[str] = set()
    parent_counts: dict[int, int] = {}
    class_counts: dict[int, int] = {}
    for item in ranked:
        if not item["exact"]:
            continue
        canonical_key = item["canonical_key"]
        if canonical_key is None:
            continue
        if canonical_key in used_keys:
            continue
        parent_id = int(item["parent_branch_id"])
        if parent_counts.get(parent_id, 0) >= max_children_per_parent:
            continue
        if item["matched_classes"]:
            primary_class = int(item["matched_classes"][0])
            if class_counts.get(primary_class, 0) >= max_same_class:
                continue
        selected.append(item)
        used_keys.add(canonical_key)
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        for class_id in item["matched_classes"]:
            class_counts[int(class_id)] = class_counts.get(int(class_id), 0) + 1
        if len(selected) >= frontier_size:
            break
    return selected


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)
    energy = GeometricHyperplaneEnergy(points=points, config=EnergyConfig())
    reference = build_reference_database()

    seed_items = load_seed_bank(points=points)
    seed_masks = seed_tensor(seed_items, device=device)
    seed_end = min(args.seed_start + args.seed_count, len(seed_items))

    model = ConditionalEditModel(
        point_dim=points.shape[1],
        edge_types=len(graph.generator_names),
        hidden_dim=payload["model_config"]["hidden_dim"],
        layers=payload["model_config"]["layers"],
        dropout=payload["model_config"]["dropout"],
        history_dim=(
            payload["model_config"].get("history_dim")
            if "history_dim" in payload["model_config"]
            else 0
        ),
        history_layers=payload["model_config"].get("history_layers", 2),
        max_history=payload["model_config"].get("max_history", args.history_length),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "seed_start": args.seed_start,
        "seed_count": seed_end - args.seed_start,
        "rounds": args.rounds,
        "num_samples": args.num_samples,
    }

    next_record_id = 0
    for seed_index in range(args.seed_start, seed_end):
        seed_mask = seed_masks[seed_index].detach().cpu()
        archive_list = [seed_masks[index].detach().cpu() for index in range(min(args.archive_size, seed_masks.shape[0]))]
        frontier = [TeacherBranch(branch_id=0, mask=seed_mask, history_masks=[seed_mask], depth=0, lineage_record_ids=[])]
        seen_canonical: set[str] = set()
        seen_classes: set[int] = set()
        next_branch_id = 1

        with torch.no_grad():
            for round_index in range(1, args.rounds + 1):
                archive_masks = torch.stack(archive_list[-args.archive_size :], dim=0).to(device)
                batch_seed = torch.cat(
                    [branch.mask.to(device).unsqueeze(0).expand(args.num_samples, -1) for branch in frontier],
                    dim=0,
                )
                batch_history = build_branch_history_batch(
                    branches=frontier,
                    num_nodes=seed_masks.shape[1],
                    device=device,
                    num_samples=args.num_samples,
                    history_length=args.history_length,
                )
                archive_context = build_archive_context(batch_seed, archive_masks=archive_masks)
                edit_logits, _ = model(
                    points,
                    batch_seed,
                    archive_context,
                    edge_index,
                    edge_type,
                    trajectory_masks=batch_history,
                )
                edit_prob = torch.sigmoid(edit_logits / max(float(args.temperature), 1e-6))
                sampled_edit = torch.bernoulli(edit_prob)
                candidate_prob = candidate_from_edit(batch_seed, sampled_edit)
                candidate_hard_batch = (candidate_prob >= 0.5).float()
                _, novelty_logits = model(
                    points,
                    batch_seed,
                    archive_context,
                    edge_index,
                    edge_type,
                    trajectory_masks=batch_history,
                    novelty_candidate_mask=candidate_prob,
                )
                embedding_payload = None
                if args.export_embeddings:
                    embedding_payload = model.extract_candidate_embeddings(
                        point_features=points,
                        seed_mask=batch_seed,
                        archive_context=archive_context,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        candidate_mask=candidate_hard_batch,
                        trajectory_masks=batch_history,
                    )

                candidate_items: list[dict[str, object]] = []
                round_record_ids: list[int] = []
                for branch_index, branch in enumerate(frontier):
                    start = branch_index * args.num_samples
                    end = start + args.num_samples
                    for sample_offset in range(args.num_samples):
                        candidate_index = start + sample_offset
                        candidate_mask = candidate_prob[candidate_index].detach().cpu()
                        hard_mask = (candidate_mask >= 0.5).float()
                        validation = classify_hard_mask(hard_mask >= 0.5, energy=energy, reference=reference)
                        exact = validation is not None
                        canonical_key = None if validation is None else validation.get("canonical_key")
                        matched_classes = [] if validation is None else [int(class_id) for class_id in validation.get("matched_classes", [])]
                        hard_energy = float(energy.energy_of_hard_mask(hard_mask >= 0.5))
                        candidate_record = {
                            "record_id": next_record_id,
                            "seed_index": seed_index,
                            "round_index": round_index,
                            "parent_branch_id": branch.branch_id,
                            "parent_depth": branch.depth,
                            "history_length": len(branch.history_masks),
                            "sample_index": sample_offset,
                            "exact": exact,
                            "canonical_key": canonical_key,
                            "matched_classes": matched_classes,
                            "candidate_cardinality": int(hard_mask.sum().item()),
                            "hard_energy": hard_energy,
                            "novelty_score": float(torch.sigmoid(novelty_logits[candidate_index]).item()),
                            "seed_jaccard": jaccard_similarity(hard_mask, branch.mask),
                            "archive_mean_absdiff": float(torch.abs(candidate_mask - archive_context[candidate_index].detach().cpu()).mean().item()),
                            "teacher_selected": 0,
                            "future_non44_hit": 0,
                            "future_first_non44_hit": 0,
                            "future_new_class_count": 0,
                            "future_rare_class_count": 0,
                            "future_new_canonical_count": 0,
                            "future_survival_rounds": 0,
                            "lineage_parents": list(branch.lineage_record_ids),
                            "quality_score": float(torch.sigmoid(novelty_logits[candidate_index]).item() - hard_energy),
                        }
                        if embedding_payload is not None:
                            candidate_record["history_embedding"] = (
                                embedding_payload["history_embedding"][candidate_index].detach().cpu().tolist()
                            )
                            candidate_record["global_summary"] = (
                                embedding_payload["global_summary"][candidate_index].detach().cpu().tolist()
                            )
                            candidate_record["candidate_summary"] = (
                                embedding_payload["candidate_summary"][candidate_index].detach().cpu().tolist()
                            )
                            candidate_record["seed_summary"] = (
                                embedding_payload["seed_summary"][candidate_index].detach().cpu().tolist()
                            )
                            candidate_record["archive_summary"] = (
                                embedding_payload["archive_summary"][candidate_index].detach().cpu().tolist()
                            )
                        records.append(candidate_record)
                        round_record_ids.append(next_record_id)
                        candidate_items.append(
                            {
                                **candidate_record,
                                "record_ref": candidate_record,
                                "mask": hard_mask,
                                "branch": branch,
                            }
                        )
                        next_record_id += 1

                selected = select_teacher_frontier(
                    candidates=candidate_items,
                    seen_canonical=seen_canonical,
                    seen_classes=seen_classes,
                    frontier_size=args.frontier_size,
                    max_children_per_parent=args.max_children_per_parent,
                    max_same_class=args.max_same_class,
                )

                next_frontier: list[TeacherBranch] = []
                for item in selected:
                    item["record_ref"]["teacher_selected"] = 1
                    canonical_key = item["canonical_key"]
                    matched_classes = item["matched_classes"]
                    novel_classes = [class_id for class_id in matched_classes if class_id not in seen_classes]
                    novel_non44_classes = [class_id for class_id in novel_classes if class_id != 44]
                    first_non44_discovery = bool(novel_non44_classes)
                    lineage_updates = item["branch"].lineage_record_ids + [int(item["record_id"])]
                    for record_id in lineage_updates:
                        record = records[record_id]
                        record["future_survival_rounds"] += 1
                        if canonical_key is not None and canonical_key not in seen_canonical:
                            record["future_new_canonical_count"] += 1
                        record["future_new_class_count"] += len(novel_classes)
                        record["future_rare_class_count"] += len(novel_non44_classes)
                        if first_non44_discovery:
                            record["future_non44_hit"] = 1
                            if record["future_first_non44_hit"] == 0:
                                record["future_first_non44_hit"] = 1
                    history_masks = item["branch"].history_masks + [item["mask"]]
                    history_masks = history_masks[-args.history_length :]
                    next_frontier.append(
                        TeacherBranch(
                            branch_id=next_branch_id,
                            mask=item["mask"],
                            history_masks=history_masks,
                            depth=item["branch"].depth + 1,
                            lineage_record_ids=item["branch"].lineage_record_ids + [int(item["record_id"])],
                        )
                    )
                    next_branch_id += 1
                    if canonical_key is not None:
                        seen_canonical.add(str(canonical_key))
                    for class_id in matched_classes:
                        seen_classes.add(int(class_id))
                    archive_list.append(item["mask"])

                if len(archive_list) > args.archive_size:
                    archive_list = archive_list[-args.archive_size :]
                frontier = next_frontier if next_frontier else [
                    TeacherBranch(
                        branch_id=next_branch_id,
                        mask=frontier[0].mask,
                        history_masks=list(frontier[0].history_masks),
                        depth=frontier[0].depth,
                        lineage_record_ids=list(frontier[0].lineage_record_ids),
                    )
                ]
                if not next_frontier:
                    next_branch_id += 1

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary["num_records"] = len(records)
    summary["num_exact_records"] = sum(1 for item in records if item["exact"])
    summary["num_selected_records"] = sum(1 for item in records if item["teacher_selected"])
    summary["num_non44_records"] = sum(1 for item in records if item["future_non44_hit"])
    summary["num_first_non44_records"] = sum(1 for item in records if item["future_first_non44_hit"])
    summary["export_embeddings"] = args.export_embeddings
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
