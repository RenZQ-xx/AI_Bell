from __future__ import annotations

import argparse
import json
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
    from long_horizon_controller import BranchState, CandidateRecord, LongHorizonController, seed_branch
    from seed_bank import load_seed_bank, seed_tensor
    from symmetry_graph import build_symmetry_graph
    from train_search_distill_scorer import BASE_FEATURE_NAMES
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .editor_model import ConditionalEditModel
    from .long_horizon_controller import BranchState, CandidateRecord, LongHorizonController, seed_branch
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


def candidate_from_edit(seed_mask: torch.Tensor, edit_prob: torch.Tensor) -> torch.Tensor:
    return seed_mask * (1.0 - edit_prob) + (1.0 - seed_mask) * edit_prob


def build_archive_context(seed_mask: torch.Tensor, archive_masks: torch.Tensor | None) -> torch.Tensor:
    if archive_masks is None or archive_masks.numel() == 0:
        return torch.full_like(seed_mask, 0.5)
    mean_archive = archive_masks.float().mean(dim=0, keepdim=True)
    return mean_archive.expand(seed_mask.shape[0], -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-horizon controlled facet expansion inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--distill-scorer", type=Path, default=None)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-tiebreak-threshold", type=float, default=0.15)
    parser.add_argument(
        "--distill-mode",
        type=str,
        choices=("additive", "tiebreak", "bucket_tiebreak", "critical_bucket_tiebreak"),
        default="additive",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_branch_history_batch(
    branches: list[BranchState],
    num_nodes: int,
    device: torch.device,
    num_samples: int,
    history_length: int,
) -> torch.Tensor:
    history_batch = []
    for branch in branches:
        history = list(branch.history_masks)
        if not history:
            history = [branch.mask]
        history = [mask.detach().cpu().float() for mask in history[-history_length:]]
        if len(history) < history_length:
            history = [history[0]] * (history_length - len(history)) + history
        stacked = torch.stack(history, dim=0)
        history_batch.append(stacked.unsqueeze(0).expand(num_samples, -1, -1))
    if not history_batch:
        return torch.zeros(0, history_length, num_nodes, device=device)
    return torch.cat(history_batch, dim=0).to(device)


def analyze_with_scores(
    probabilities: torch.Tensor,
    novelty_scores: torch.Tensor,
    energy: GeometricHyperplaneEnergy,
    reference: dict[str, object],
) -> list[dict[str, object]]:
    analyzed = []
    for index, candidate in enumerate(probabilities):
        validation = classify_hard_mask(candidate >= 0.5, energy=energy, reference=reference)
        if validation is None:
            continue
        validation["hard_energy"] = energy.energy_of_hard_mask(candidate >= 0.5)
        validation["novelty_score"] = float(novelty_scores[index].item())
        validation["sample_index"] = index
        analyzed.append(validation)
    return analyzed


def build_distill_feature_row(
    feature_names: list[str],
    *,
    round_index: int,
    parent_depth: int,
    history_length: int,
    candidate_mask: torch.Tensor,
    hard_energy: float,
    novelty_score: float,
    seed_mask: torch.Tensor,
    archive_context: torch.Tensor,
    embedding_payload: dict[str, torch.Tensor] | None = None,
    embedding_index: int | None = None,
) -> list[float]:
    candidate_hard = (candidate_mask >= 0.5).float()
    intersection = torch.minimum(candidate_hard, seed_mask).sum().item()
    union = torch.maximum(candidate_hard, seed_mask).sum().item()
    seed_jaccard = float(intersection / max(union, 1.0))
    archive_mean_absdiff = float(torch.abs(candidate_mask - archive_context).mean().item())
    values = {
        "round_index": float(round_index),
        "parent_depth": float(parent_depth),
        "history_length": float(history_length),
        "candidate_cardinality": float(candidate_hard.sum().item()),
        "hard_energy": float(hard_energy),
        "novelty_score": float(novelty_score),
        "seed_jaccard": seed_jaccard,
        "archive_mean_absdiff": archive_mean_absdiff,
    }
    if embedding_payload is None or embedding_index is None:
        embedding_vectors: dict[str, list[float]] = {}
    else:
        embedding_vectors = {
            "history_embedding": embedding_payload["history_embedding"][embedding_index].detach().cpu().tolist(),
            "global_summary": embedding_payload["global_summary"][embedding_index].detach().cpu().tolist(),
            "candidate_summary": embedding_payload["candidate_summary"][embedding_index].detach().cpu().tolist(),
            "seed_summary": embedding_payload["seed_summary"][embedding_index].detach().cpu().tolist(),
            "archive_summary": embedding_payload["archive_summary"][embedding_index].detach().cpu().tolist(),
        }
    derived_vectors = {}
    if embedding_vectors:
        derived_vectors["candidate_minus_seed"] = [
            left - right for left, right in zip(embedding_vectors["candidate_summary"], embedding_vectors["seed_summary"])
        ]
        derived_vectors["candidate_minus_archive"] = [
            left - right
            for left, right in zip(embedding_vectors["candidate_summary"], embedding_vectors["archive_summary"])
        ]
        derived_vectors["candidate_minus_history"] = [
            left - right
            for left, right in zip(embedding_vectors["candidate_summary"], embedding_vectors["history_embedding"])
        ]

    row = []
    for name in feature_names:
        if name in values:
            row.append(values[name])
            continue
        prefix, _, raw_index = name.partition(":")
        if not raw_index:
            row.append(0.0)
            continue
        index = int(raw_index)
        source = embedding_vectors.get(prefix)
        if source is None:
            source = derived_vectors.get(prefix)
        if source is None or index >= len(source):
            row.append(0.0)
            continue
        row.append(float(source[index]))
    return row


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
    seed_index = max(0, min(args.seed_index, seed_masks.shape[0] - 1))
    seed_item = seed_items[seed_index]

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
        max_history=payload["model_config"].get("max_history", args.history_length or 6),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()

    distill_scorer = None
    distill_mean = None
    distill_std = None
    distill_feature_names = None
    if args.distill_scorer is not None:
        scorer_payload = torch.load(args.distill_scorer, map_location=device, weights_only=False)
        hidden_dim = int(
            scorer_payload.get(
                "hidden_dim",
                scorer_payload["model_state"]["network.0.weight"].shape[0],
            )
        )
        distill_scorer = DistilledCandidateScorer(
            input_dim=len(scorer_payload["feature_names"]),
            hidden_dim=hidden_dim,
        ).to(device)
        distill_scorer.load_state_dict(scorer_payload["model_state"])
        distill_scorer.eval()
        distill_feature_names = list(scorer_payload["feature_names"])
        distill_mean = torch.tensor(scorer_payload["feature_mean"], dtype=torch.float32, device=device)
        distill_std = torch.tensor(scorer_payload["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)

    history_length = args.history_length or payload["model_config"].get("max_history", 6)
    archive_list = [seed_masks[index].detach().cpu() for index in range(min(args.archive_size, seed_masks.shape[0]))]
    seen_canonical: set[str] = set()
    seen_classes: set[int] = set()
    controller = LongHorizonController(frontier_size=args.frontier_size, history_limit=history_length)
    next_branch_id = 1
    frontier: list[BranchState] = [seed_branch(seed_masks[seed_index], branch_id=0, history_limit=history_length)]
    all_results = []
    round_summaries = []

    with torch.no_grad():
        for round_index in range(1, args.rounds + 1):
            archive_masks = torch.stack(archive_list[-args.archive_size :], dim=0).to(device)
            candidate_records: list[CandidateRecord] = []
            round_results = []
            batch_seed = torch.cat(
                [branch.mask.to(device).unsqueeze(0).expand(args.num_samples, -1) for branch in frontier],
                dim=0,
            )
            batch_history = build_branch_history_batch(
                branches=frontier,
                num_nodes=seed_masks.shape[1],
                device=device,
                num_samples=args.num_samples,
                history_length=history_length,
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
            safe_temperature = max(float(args.temperature), 1e-6)
            edit_prob = torch.sigmoid(edit_logits / safe_temperature)
            sampled_edit = torch.bernoulli(edit_prob)
            candidate_prob = candidate_from_edit(batch_seed, sampled_edit)
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
            if distill_scorer is not None and distill_feature_names is not None:
                requires_embeddings = any(name not in BASE_FEATURE_NAMES for name in distill_feature_names)
                if requires_embeddings:
                    embedding_payload = model.extract_candidate_embeddings(
                        point_features=points,
                        seed_mask=batch_seed,
                        archive_context=archive_context,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        candidate_mask=(candidate_prob >= 0.5).float(),
                        trajectory_masks=batch_history,
                    )
            for branch_index, branch in enumerate(frontier):
                start = branch_index * args.num_samples
                end = start + args.num_samples
                analyzed = analyze_with_scores(
                    probabilities=candidate_prob[start:end].detach().cpu(),
                    novelty_scores=torch.sigmoid(novelty_logits[start:end]).detach().cpu(),
                    energy=energy,
                    reference=reference,
                )
                branch.times_expanded += 1
                for item in analyzed:
                    item["round"] = round_index
                    item["parent_branch_id"] = branch.branch_id
                    round_results.append(item)
                    matched_classes = [int(class_id) for class_id in item.get("matched_classes", [])]
                    novelty_score = float(item.get("novelty_score", 0.0))
                    quality_score = novelty_score
                    tiebreak_score = 0.0
                    exact_candidate = item.get("tier") == "exact_match"
                    if distill_scorer is not None and distill_mean is not None and distill_std is not None:
                        sample_index = start + int(item.get("sample_index", 0))
                        feature_row = torch.tensor(
                            build_distill_feature_row(
                                distill_feature_names or BASE_FEATURE_NAMES,
                                round_index=round_index,
                                parent_depth=branch.depth,
                                history_length=len(branch.history_masks),
                                candidate_mask=torch.tensor(
                                    [1.0 if index in item["indices"] else 0.0 for index in range(seed_masks.shape[1])],
                                    dtype=torch.float32,
                                ),
                                hard_energy=float(item.get("hard_energy", 0.0)),
                                novelty_score=float(item.get("novelty_score", 0.0)),
                                seed_mask=branch.mask.float(),
                                archive_context=archive_context[start].detach().cpu(),
                                embedding_payload=embedding_payload,
                                embedding_index=sample_index,
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        normalized = (feature_row - distill_mean) / distill_std
                        distill_score = float(distill_scorer(normalized.unsqueeze(0)).item())
                        item["distill_score"] = distill_score
                        if args.distill_mode == "additive":
                            quality_score = quality_score + args.distill_weight * distill_score
                        elif args.distill_mode == "tiebreak":
                            tiebreak_score = distill_score
                        elif args.distill_mode == "bucket_tiebreak":
                            quality_score = 0.0
                            tiebreak_score = distill_score
                        else:
                            if exact_candidate and abs(quality_score) <= args.distill_tiebreak_threshold:
                                quality_score = 0.0
                                tiebreak_score = distill_score
                    elif args.distill_mode in ("bucket_tiebreak", "critical_bucket_tiebreak"):
                        quality_score = 0.0
                    candidate_records.append(
                        CandidateRecord(
                            branch_id=-1,
                            parent_branch_id=branch.branch_id,
                            mask=torch.tensor(
                                [1.0 if index in item["indices"] else 0.0 for index in range(seed_masks.shape[1])],
                                dtype=torch.float32,
                            ),
                            canonical_key=item.get("canonical_key"),
                            matched_classes=matched_classes,
                            tier=str(item.get("tier", "")),
                            round_index=round_index,
                            exact=item.get("tier") == "exact_match",
                            quality_score=quality_score,
                            tiebreak_score=tiebreak_score,
                        )
                    )

            frontier, next_branch_id = controller.update_frontier(
                frontier=frontier,
                candidates=candidate_records,
                seen_canonical=seen_canonical,
                seen_classes=seen_classes,
                next_branch_id=next_branch_id,
                round_index=round_index,
            )

            for branch in frontier:
                if branch.canonical_key is not None:
                    seen_canonical.add(branch.canonical_key)
                for class_id in branch.matched_classes:
                    seen_classes.add(class_id)
                archive_list.append(branch.mask.detach().cpu())
            if len(archive_list) > args.archive_size:
                archive_list = archive_list[-args.archive_size :]

            round_summaries.append(
                {
                    "round": round_index,
                    "frontier_size": len(frontier),
                    "exact_count": sum(1 for item in round_results if item.get("tier") == "exact_match"),
                    "matched_classes": sorted({class_id for item in round_results for class_id in item.get("matched_classes", [])}),
                    "seen_exact_classes": sorted(seen_classes),
                }
            )
            all_results.extend(round_results)

    summary = {
        "seed_index": seed_index,
        "seed_class_id": seed_item.class_id,
        "seed_example_id": seed_item.example_id,
        "seed_cardinality": seed_item.cardinality,
        "rounds": args.rounds,
        "num_candidates": len(all_results),
        "exact_count": sum(1 for item in all_results if item.get("tier") == "exact_match"),
        "matched_classes": sorted({class_id for item in all_results for class_id in item.get("matched_classes", [])}),
        "canonical_count": len({item.get("canonical_key") for item in all_results if item.get("canonical_key") is not None}),
        "round_summaries": round_summaries,
    }
    output = {
        "checkpoint": str(args.checkpoint),
        "summary": summary,
        "results": all_results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
