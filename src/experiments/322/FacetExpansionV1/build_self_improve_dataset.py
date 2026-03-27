from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

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
    from long_horizon_controller import CandidateRecord, LongHorizonController
    from seed_bank import load_seed_bank, seed_tensor
    from symmetry_graph import build_symmetry_graph
    from infer_longhorizon import build_distill_feature_row
    from train_search_distill_scorer import BASE_FEATURE_NAMES
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .editor_model import ConditionalEditModel
    from .infer_longhorizon import build_distill_feature_row
    from .long_horizon_controller import CandidateRecord, LongHorizonController
    from .seed_bank import load_seed_bank, seed_tensor
    from .train_search_distill_scorer import BASE_FEATURE_NAMES


class DistilledCandidateScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class RolloutBranch:
    branch_id: int
    mask: torch.Tensor
    history_masks: list[torch.Tensor]
    depth: int
    lineage_record_ids: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build self-improvement rollout data from the current search policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--seed-class-id", type=int, default=None)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--distill-scorer", type=Path, default=None)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument(
        "--distill-mode",
        type=str,
        choices=("additive", "tiebreak", "bucket_tiebreak", "critical_bucket_tiebreak"),
        default="critical_bucket_tiebreak",
    )
    parser.add_argument("--distill-tiebreak-threshold", type=float, default=0.15)
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
    branches: list[RolloutBranch],
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
        history_batch.append(torch.stack(history, dim=0).unsqueeze(0).expand(num_samples, -1, -1))
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


def load_distill_scorer(
    scorer_path: Path | None,
    device: torch.device,
) -> tuple[DistilledCandidateScorer | None, torch.Tensor | None, torch.Tensor | None, list[str] | None]:
    if scorer_path is None:
        return None, None, None, None
    payload = torch.load(scorer_path, map_location=device, weights_only=False)
    hidden_dim = int(payload.get("hidden_dim", payload["model_state"]["network.0.weight"].shape[0]))
    scorer = DistilledCandidateScorer(input_dim=len(payload["feature_names"]), hidden_dim=hidden_dim).to(device)
    scorer.load_state_dict(payload["model_state"])
    scorer.eval()
    mean = torch.tensor(payload["feature_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(payload["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return scorer, mean, std, list(payload["feature_names"])


def select_policy_frontier(
    controller: LongHorizonController,
    frontier: list[RolloutBranch],
    candidates: list[dict[str, object]],
    seen_canonical: set[str],
    seen_classes: set[int],
) -> list[dict[str, object]]:
    ranked = sorted(
        candidates,
        key=lambda item: controller.score_candidate(
            item["candidate_record"],
            seen_canonical=seen_canonical,
            seen_classes=seen_classes,
        ),
        reverse=True,
    )

    selected: list[dict[str, object]] = []
    parent_counts: dict[int, int] = {}
    class_counts: dict[int, int] = {}
    used_keys: set[str] = set()
    bridge_children = 0

    for item in ranked:
        candidate = item["candidate_record"]
        candidate_key = candidate.canonical_key or f"bridge:{candidate.parent_branch_id}:{candidate.round_index}:{len(selected)}"
        if candidate_key in used_keys:
            continue
        if parent_counts.get(candidate.parent_branch_id, 0) >= controller.max_children_per_parent:
            continue
        if not candidate.exact and bridge_children >= controller.max_bridge_children:
            continue
        if not candidate.exact and any(
            controller.mask_similarity(candidate.mask, existing["candidate_record"].mask) >= controller.bridge_similarity_threshold
            for existing in selected
            if not existing["candidate_record"].exact
        ):
            continue
        if candidate.matched_classes:
            primary_class = candidate.matched_classes[0]
            if class_counts.get(primary_class, 0) >= controller.max_same_class:
                continue
        selected.append(item)
        used_keys.add(candidate_key)
        parent_counts[candidate.parent_branch_id] = parent_counts.get(candidate.parent_branch_id, 0) + 1
        for class_id in candidate.matched_classes:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        if not candidate.exact:
            bridge_children += 1
        if len(selected) >= controller.frontier_size:
            break

    if not selected and frontier:
        return []
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
    if args.seed_class_id is None:
        seed_indices = list(range(args.seed_start, min(args.seed_start + args.seed_count, len(seed_items))))
    else:
        matching_indices = [index for index, item in enumerate(seed_items) if int(item.class_id) == int(args.seed_class_id)]
        seed_indices = matching_indices[args.seed_start : args.seed_start + args.seed_count]

    model = ConditionalEditModel(
        point_dim=points.shape[1],
        edge_types=len(graph.generator_names),
        hidden_dim=payload["model_config"]["hidden_dim"],
        layers=payload["model_config"]["layers"],
        dropout=payload["model_config"]["dropout"],
        history_dim=payload["model_config"].get("history_dim", 0),
        history_layers=payload["model_config"].get("history_layers", 2),
        max_history=payload["model_config"].get("max_history", args.history_length),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()

    distill_scorer, distill_mean, distill_std, distill_feature_names = load_distill_scorer(args.distill_scorer, device=device)
    controller = LongHorizonController(frontier_size=args.frontier_size, history_limit=args.history_length)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    discovery_counts: dict[int, int] = {}
    next_record_id = 0
    next_branch_id = 1

    with torch.no_grad():
        for seed_index in seed_indices:
            seed_mask = seed_masks[seed_index].detach().cpu()
            seed_class_id = int(seed_items[seed_index].class_id)
            frontier = [RolloutBranch(branch_id=0, mask=seed_mask, history_masks=[seed_mask], depth=0, lineage_record_ids=[])]
            archive_list = [seed_masks[index].detach().cpu() for index in range(min(args.archive_size, seed_masks.shape[0]))]
            seen_canonical: set[str] = set()
            seen_classes: set[int] = set()
            episode_id = seed_index

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
                if args.export_embeddings or (
                    distill_feature_names is not None and any(name not in BASE_FEATURE_NAMES for name in distill_feature_names)
                ):
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
                        novelty_score = float(torch.sigmoid(novelty_logits[candidate_index]).item())
                        record = {
                            "record_id": next_record_id,
                            "episode_id": episode_id,
                            "seed_index": seed_index,
                            "seed_class_id": seed_class_id,
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
                            "novelty_score": novelty_score,
                            "seed_jaccard": jaccard_similarity(hard_mask, branch.mask),
                            "archive_mean_absdiff": float(torch.abs(candidate_mask - archive_context[candidate_index].detach().cpu()).mean().item()),
                            "teacher_selected": 0,
                            "policy_selected": 0,
                            "future_non44_hit": 0,
                            "future_first_non44_hit": 0,
                            "future_escape_initial_hit": 0,
                            "future_unique_class_gain": 0.0,
                            "future_rare_class_gain": 0.0,
                            "future_new_class_count": 0.0,
                            "future_new_canonical_count": 0.0,
                            "future_survival_rounds": 0.0,
                            "future_total_return": 0.0,
                            "lineage_parents": list(branch.lineage_record_ids),
                            "quality_score": novelty_score,
                        }
                        if args.export_embeddings and embedding_payload is not None:
                            record["history_embedding"] = embedding_payload["history_embedding"][candidate_index].detach().cpu().tolist()
                            record["global_summary"] = embedding_payload["global_summary"][candidate_index].detach().cpu().tolist()
                            record["candidate_summary"] = embedding_payload["candidate_summary"][candidate_index].detach().cpu().tolist()
                            record["seed_summary"] = embedding_payload["seed_summary"][candidate_index].detach().cpu().tolist()
                            record["archive_summary"] = embedding_payload["archive_summary"][candidate_index].detach().cpu().tolist()

                        quality_score = novelty_score
                        tiebreak_score = 0.0
                        if distill_scorer is not None and distill_mean is not None and distill_std is not None and distill_feature_names is not None:
                            feature_row = torch.tensor(
                                build_distill_feature_row(
                                    distill_feature_names,
                                    round_index=round_index,
                                    parent_depth=branch.depth,
                                    history_length=len(branch.history_masks),
                                    candidate_mask=hard_mask,
                                    hard_energy=hard_energy,
                                    novelty_score=novelty_score,
                                    seed_mask=branch.mask.float(),
                                    archive_context=archive_context[start].detach().cpu(),
                                    embedding_payload=embedding_payload,
                                    embedding_index=candidate_index,
                                ),
                                dtype=torch.float32,
                                device=device,
                            )
                            normalized = (feature_row - distill_mean) / distill_std
                            distill_score = float(distill_scorer(normalized.unsqueeze(0)).item())
                            record["distill_score"] = distill_score
                            if args.distill_mode == "additive":
                                quality_score += args.distill_weight * distill_score
                            elif args.distill_mode == "tiebreak":
                                tiebreak_score = distill_score
                            elif args.distill_mode == "bucket_tiebreak":
                                quality_score = 0.0
                                tiebreak_score = distill_score
                            elif exact and abs(quality_score) <= args.distill_tiebreak_threshold:
                                quality_score = 0.0
                                tiebreak_score = distill_score

                        candidate_record = CandidateRecord(
                            branch_id=next_record_id,
                            parent_branch_id=branch.branch_id,
                            mask=hard_mask,
                            canonical_key=canonical_key,
                            matched_classes=matched_classes,
                            tier="exact_match" if exact else "bridge",
                            round_index=round_index,
                            exact=exact,
                            quality_score=quality_score,
                            tiebreak_score=tiebreak_score,
                            source_index=next_record_id,
                        )
                        records.append(record)
                        candidate_items.append(
                            {
                                "candidate_record": candidate_record,
                                "record_ref": record,
                                "record_id": next_record_id,
                                "branch": branch,
                                "mask": hard_mask,
                            }
                        )
                        next_record_id += 1

                selected = select_policy_frontier(
                    controller=controller,
                    frontier=frontier,
                    candidates=candidate_items,
                    seen_canonical=seen_canonical,
                    seen_classes=seen_classes,
                )

                next_frontier: list[RolloutBranch] = []
                for item in selected:
                    record = item["record_ref"]
                    candidate = item["candidate_record"]
                    record["teacher_selected"] = 1
                    record["policy_selected"] = 1
                    novel_classes = [class_id for class_id in candidate.matched_classes if class_id not in seen_classes]
                    novel_non44 = [class_id for class_id in novel_classes if class_id != 44]
                    escaped_initial_classes = [class_id for class_id in candidate.matched_classes if class_id != seed_class_id]
                    novel_escape_classes = [class_id for class_id in novel_classes if class_id != seed_class_id]
                    rare_gain = sum(1.0 / max(discovery_counts.get(class_id, 0) + 1, 1) for class_id in novel_non44)
                    canonical_gain = 1.0 if candidate.canonical_key is not None and candidate.canonical_key not in seen_canonical else 0.0
                    late_round_gain = float(round_index >= 4 and bool(escaped_initial_classes))
                    total_return = (
                        4.0 * len(novel_escape_classes)
                        + 2.0 * float(bool(escaped_initial_classes))
                        + 2.0 * len(novel_non44)
                        + 2.0 * rare_gain
                        + 1.0 * canonical_gain
                        + 0.5 * late_round_gain
                    )
                    lineage_updates = item["branch"].lineage_record_ids + [int(item["record_id"])]
                    for lineage_record_id in lineage_updates:
                        lineage_record = records[lineage_record_id]
                        lineage_record["future_survival_rounds"] += 1.0
                        lineage_record["future_escape_initial_hit"] = max(
                            int(lineage_record["future_escape_initial_hit"]),
                            int(bool(escaped_initial_classes)),
                        )
                        lineage_record["future_unique_class_gain"] += float(len(novel_non44))
                        lineage_record["future_rare_class_gain"] += float(rare_gain)
                        lineage_record["future_new_class_count"] += float(len(novel_classes))
                        lineage_record["future_new_canonical_count"] += canonical_gain
                        lineage_record["future_total_return"] += total_return
                        if novel_non44:
                            lineage_record["future_non44_hit"] = 1
                            if lineage_record["future_first_non44_hit"] == 0:
                                lineage_record["future_first_non44_hit"] = 1
                    for class_id in novel_non44:
                        discovery_counts[class_id] = discovery_counts.get(class_id, 0) + 1

                    history_masks = (item["branch"].history_masks + [item["mask"]])[-args.history_length :]
                    next_frontier.append(
                        RolloutBranch(
                            branch_id=next_branch_id,
                            mask=item["mask"],
                            history_masks=history_masks,
                            depth=item["branch"].depth + 1,
                            lineage_record_ids=item["branch"].lineage_record_ids + [int(item["record_id"])],
                        )
                    )
                    next_branch_id += 1
                    if candidate.canonical_key is not None:
                        seen_canonical.add(str(candidate.canonical_key))
                    for class_id in candidate.matched_classes:
                        seen_classes.add(int(class_id))
                    archive_list.append(item["mask"])

                if len(archive_list) > args.archive_size:
                    archive_list = archive_list[-args.archive_size :]
                if next_frontier:
                    frontier = next_frontier

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": str(args.checkpoint),
        "seed_start": args.seed_start,
        "seed_count": len(seed_indices),
        "seed_class_id": args.seed_class_id,
        "rounds": args.rounds,
        "num_samples": args.num_samples,
        "distill_scorer": str(args.distill_scorer) if args.distill_scorer else None,
        "distill_mode": args.distill_mode,
        "num_records": len(records),
        "num_exact_records": sum(1 for record in records if record["exact"]),
        "num_policy_selected_records": sum(1 for record in records if record["policy_selected"]),
        "num_non44_records": sum(1 for record in records if record["future_non44_hit"]),
        "num_unique_gain_records": sum(1 for record in records if float(record["future_unique_class_gain"]) > 0),
        "export_embeddings": args.export_embeddings,
        "discovery_histogram": {str(key): value for key, value in sorted(discovery_counts.items())},
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
