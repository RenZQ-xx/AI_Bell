from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    EXP322_DIR = CURRENT_DIR.parent
    DIFFUCO_DIR = EXP322_DIR / "DIffUCO"
    V1_DIR = EXP322_DIR / "FacetExpansionV1"
    sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR), str(CURRENT_DIR)]
    from energy import EnergyConfig, GeometricHyperplaneEnergy
    from facet_reference import build_reference_database
    from geometry import generate_points_322
    from inference import classify_hard_mask
    from editor_model import ConditionalEditModel
    from long_horizon_controller import CandidateRecord, LongHorizonController, seed_branch
    from seed_bank import load_seed_bank, seed_tensor
    from symmetry_graph import build_symmetry_graph
    from graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from ..FacetExpansionV1.editor_model import ConditionalEditModel
    from ..FacetExpansionV1.long_horizon_controller import CandidateRecord, LongHorizonController, seed_branch
    from ..FacetExpansionV1.seed_bank import load_seed_bank, seed_tensor
    from .graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer


def normalize_cli_path(path: Path) -> Path:
    if os.name == "nt" and path.is_absolute() and str(path).startswith(("/home/", "/mnt/")):
        return Path("\\\\wsl$\\Ubuntu") / str(path).lstrip("/").replace("/", "\\")
    return path


class EscapeScorerWrapper(nn.Module):
    def __init__(self, checkpoint: Path, device: torch.device):
        super().__init__()
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        architecture = payload["model_config"].get("architecture", "set")
        model_kwargs = dict(payload["model_config"])
        model_kwargs.pop("architecture", None)
        if architecture == "graph":
            model = GraphEscapeScorer(**model_kwargs)
        else:
            model = SetEscapeScorer(**model_kwargs)
        model.load_state_dict(payload["model_state"])
        self.model = model.to(device)
        self.model.eval()
        self.architecture = architecture

    def score(
        self,
        point_features: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        energy_value: float,
    ) -> float:
        with torch.no_grad():
            logits = self.model(
                point_features,
                source_mask.unsqueeze(0).float(),
                candidate_mask.unsqueeze(0).float(),
                edge_index,
                edge_type,
                energy_value=torch.tensor([energy_value], dtype=torch.float32, device=point_features.device),
            )
            return float(torch.sigmoid(logits).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental V2 jump-proposal inference for 44-only escape.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jump-bank", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--jump-top-k", type=int, default=3)
    parser.add_argument("--jump-rounds", type=int, default=6)
    parser.add_argument("--jump-quality-bonus", type=float, default=2.5)
    parser.add_argument("--allow-nonexact-jumps", action="store_true")
    parser.add_argument("--max-bridge-children", type=int, default=2)
    parser.add_argument("--bridge-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--escape-scorer", type=Path, default=None)
    parser.add_argument(
        "--escape-mode",
        type=str,
        choices=("additive", "tiebreak", "bucket_tiebreak"),
        default="bucket_tiebreak",
    )
    parser.add_argument("--escape-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_archive_context(seed_mask: torch.Tensor, archive_masks: torch.Tensor | None) -> torch.Tensor:
    if archive_masks is None or archive_masks.numel() == 0:
        return torch.full_like(seed_mask, 0.5)
    mean_archive = archive_masks.float().mean(dim=0, keepdim=True)
    return mean_archive.expand(seed_mask.shape[0], -1)


def candidate_from_edit(seed_mask: torch.Tensor, edit_prob: torch.Tensor) -> torch.Tensor:
    return seed_mask * (1.0 - edit_prob) + (1.0 - seed_mask) * edit_prob


def main() -> None:
    args = parse_args()
    args.checkpoint = normalize_cli_path(args.checkpoint)
    args.jump_bank = normalize_cli_path(args.jump_bank)
    if args.escape_scorer is not None:
        args.escape_scorer = normalize_cli_path(args.escape_scorer)
    if args.output is not None:
        args.output = normalize_cli_path(args.output)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    jump_bank = json.loads(args.jump_bank.read_text(encoding="utf-8"))

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)
    energy = GeometricHyperplaneEnergy(points=points, config=EnergyConfig())
    reference = build_reference_database()
    seed_items = load_seed_bank(points=points)
    seed_masks = seed_tensor(seed_items, device=device)

    seed_index = int(args.seed_index)
    seed_item = seed_items[seed_index]

    model = ConditionalEditModel(
        point_dim=points.shape[1],
        edge_types=len(graph.generator_names),
        hidden_dim=payload["model_config"]["hidden_dim"],
        layers=payload["model_config"]["layers"],
        dropout=payload["model_config"]["dropout"],
        history_dim=payload["model_config"].get("history_dim", 0),
        history_layers=payload["model_config"].get("history_layers", 2),
        max_history=payload["model_config"].get("max_history", 6),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()
    escape_scorer = EscapeScorerWrapper(args.escape_scorer, device=device) if args.escape_scorer is not None else None

    proposals_by_source = {
        int(item["source_seed_index"]): list(item["targets"]) for item in jump_bank.get("jump_proposals", [])
    }
    source_proposals = proposals_by_source.get(seed_index, [])

    controller = LongHorizonController(
        frontier_size=args.frontier_size,
        history_limit=6,
        allow_bridge_candidates=args.allow_nonexact_jumps,
        max_bridge_children=args.max_bridge_children,
        bridge_similarity_threshold=args.bridge_similarity_threshold,
    )
    frontier = [seed_branch(seed_masks[seed_index], branch_id=0, history_limit=6)]
    next_branch_id = 1
    archive_list = [seed_masks[index].detach().cpu() for index in range(min(args.archive_size, seed_masks.shape[0]))]
    seen_canonical: set[str] = set()
    seen_classes: set[int] = set()
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
            archive_context = build_archive_context(batch_seed, archive_masks=archive_masks)
            edit_logits, _ = model(points, batch_seed, archive_context, edge_index, edge_type)
            edit_prob = torch.sigmoid(edit_logits / max(float(args.temperature), 1e-6))
            sampled_edit = torch.bernoulli(edit_prob)
            candidate_prob = candidate_from_edit(batch_seed, sampled_edit)

            for branch_index, branch in enumerate(frontier):
                start = branch_index * args.num_samples
                end = start + args.num_samples
                for local_index, candidate in enumerate(candidate_prob[start:end].detach().cpu()):
                    hard = (candidate >= 0.5).float()
                    validation = classify_hard_mask(hard >= 0.5, energy=energy, reference=reference)
                    if validation is None:
                        continue
                    matched_classes = [int(class_id) for class_id in validation.get("matched_classes", [])]
                    hard_energy = float(energy.energy_of_hard_mask(hard >= 0.5))
                    round_results.append(
                        {
                            "candidate_type": "local_edit",
                            "round": round_index,
                            "parent_branch_id": branch.branch_id,
                            "matched_classes": matched_classes,
                            "canonical_key": validation.get("canonical_key"),
                            "hard_energy": hard_energy,
                        }
                    )
                    escape_score = 0.0
                    quality_score = -hard_energy
                    tiebreak_score = 0.0
                    if escape_scorer is not None:
                        escape_score = escape_scorer.score(
                            points,
                            branch.mask.to(device),
                            hard.to(device),
                            edge_index,
                            edge_type,
                            energy_value=hard_energy,
                        )
                        round_results[-1]["escape_score"] = escape_score
                        if args.escape_mode == "additive":
                            quality_score = quality_score + args.escape_weight * escape_score
                        elif args.escape_mode == "tiebreak":
                            tiebreak_score = args.escape_weight * escape_score
                        else:
                            quality_score = 0.0
                            tiebreak_score = args.escape_weight * escape_score
                    candidate_records.append(
                        CandidateRecord(
                            branch_id=-1,
                            parent_branch_id=branch.branch_id,
                            mask=hard,
                            canonical_key=validation.get("canonical_key"),
                            matched_classes=matched_classes,
                            tier="exact_match",
                            round_index=round_index,
                            exact=True,
                            quality_score=quality_score,
                            tiebreak_score=tiebreak_score,
                        )
                    )

                if round_index <= args.jump_rounds:
                    for jump in source_proposals[: args.jump_top_k]:
                        if "target_seed_index" in jump:
                            jump_mask = seed_masks[int(jump["target_seed_index"])].detach().cpu()
                        else:
                            jump_mask = torch.tensor(jump["mask"], dtype=torch.float32)
                        validation = classify_hard_mask(jump_mask >= 0.5, energy=energy, reference=reference)
                        hard_energy = float(energy.energy_of_hard_mask(jump_mask >= 0.5))
                        matched_classes = [] if validation is None else [int(class_id) for class_id in validation.get("matched_classes", [])]
                        round_result = {
                            "candidate_type": "jump_proposal",
                            "round": round_index,
                            "parent_branch_id": branch.branch_id,
                            "matched_classes": matched_classes,
                            "canonical_key": None if validation is None else validation.get("canonical_key"),
                            "hard_energy": hard_energy,
                            "proposal_type": jump.get("proposal_type", "seed_target"),
                        }
                        if "target_seed_index" in jump:
                            round_result.update(
                                {
                                    "jump_target_seed_index": int(jump["target_seed_index"]),
                                    "jump_target_class_id": int(jump["target_class_id"]),
                                    "jump_num_add": int(jump["num_add"]),
                                    "jump_num_remove": int(jump["num_remove"]),
                                }
                            )
                        else:
                            round_result.update(
                                {
                                    "boundary_score": float(jump.get("boundary_score", 0.0)),
                                    "jump_hamming_distance": int(jump.get("hamming_distance", 0)),
                                    "jump_jaccard_to_source": float(jump.get("jaccard_to_source", 1.0)),
                                }
                            )
                        round_results.append(round_result)
                        if validation is None and not args.allow_nonexact_jumps:
                            continue
                        escape_score = 0.0
                        quality_score = (
                            args.jump_quality_bonus
                            - hard_energy
                            + float(jump.get("proposal_score", jump.get("anti44_score", jump.get("boundary_score", 0.0))))
                        )
                        tiebreak_score = 0.0
                        if escape_scorer is not None:
                            escape_score = escape_scorer.score(
                                points,
                                branch.mask.to(device),
                                jump_mask.to(device),
                                edge_index,
                                edge_type,
                                energy_value=hard_energy,
                            )
                            round_result["escape_score"] = escape_score
                            if args.escape_mode == "additive":
                                quality_score = quality_score + args.escape_weight * escape_score
                            elif args.escape_mode == "tiebreak":
                                tiebreak_score = args.escape_weight * escape_score
                            else:
                                quality_score = 0.0 if validation is not None else quality_score
                                tiebreak_score = args.escape_weight * escape_score
                        candidate_records.append(
                            CandidateRecord(
                                branch_id=-1,
                                parent_branch_id=branch.branch_id,
                                mask=jump_mask,
                                canonical_key=None if validation is None else validation.get("canonical_key"),
                                matched_classes=matched_classes,
                                tier="exact_match" if validation is not None else "bridge_jump",
                                round_index=round_index,
                                exact=validation is not None,
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
                    seen_classes.add(int(class_id))
                archive_list.append(branch.mask.detach().cpu())
            if len(archive_list) > args.archive_size:
                archive_list = archive_list[-args.archive_size :]

            round_summaries.append(
                {
                    "round": round_index,
                    "frontier_size": len(frontier),
                    "exact_count": len(round_results),
                    "matched_classes": sorted({class_id for item in round_results for class_id in item.get("matched_classes", [])}),
                    "seen_exact_classes": sorted(seen_classes),
                }
            )
            all_results.extend(round_results)

    summary = {
        "seed_index": seed_index,
        "seed_class_id": int(seed_item.class_id),
        "seed_example_id": int(seed_item.example_id),
        "num_candidates": len(all_results),
        "matched_classes": sorted({class_id for item in all_results for class_id in item.get("matched_classes", [])}),
        "canonical_count": len({item.get("canonical_key") for item in all_results if item.get("canonical_key") is not None}),
        "round_summaries": round_summaries,
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "jump_bank": str(args.jump_bank),
        "summary": summary,
        "results": all_results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
