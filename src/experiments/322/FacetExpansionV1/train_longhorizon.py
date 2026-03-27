from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import optim

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
    from seed_bank import load_seed_bank, seed_tensor, summarize_seed_bank
    from symmetry_graph import build_symmetry_graph
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .editor_model import ConditionalEditModel
    from .long_horizon_controller import BranchState, CandidateRecord, LongHorizonController, seed_branch
    from .seed_bank import load_seed_bank, seed_tensor, summarize_seed_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FacetExpansionV1 with a long-horizon multi-branch controller in the loop.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--samples-per-branch", type=int, default=6)
    parser.add_argument("--archive-capacity", type=int, default=256)
    parser.add_argument("--controller-max-children-per-parent", type=int, default=2)
    parser.add_argument("--controller-max-same-class", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--bridge-candidates", action="store_true")
    parser.add_argument("--max-bridge-children", type=int, default=1)
    parser.add_argument("--bridge-energy-weight", type=float, default=1.0)
    parser.add_argument("--bridge-repulsion-weight", type=float, default=0.8)
    parser.add_argument("--bridge-edit-weight", type=float, default=0.3)
    parser.add_argument("--bridge-similarity-threshold", type=float, default=0.85)
    parser.add_argument("--novelty-head-weight", type=float, default=0.2)
    parser.add_argument("--bridge-novelty-weight", type=float, default=0.6)
    parser.add_argument("--edit-min", type=float, default=3.0)
    parser.add_argument("--edit-max", type=float, default=18.0)
    parser.add_argument("--edit-band-weight", type=float, default=0.8)
    parser.add_argument("--edit-min-penalty-scale", type=float, default=3.0)
    parser.add_argument("--archive-repulsion-weight", type=float, default=0.6)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--entropy-floor", type=float, default=8.0)
    parser.add_argument("--entropy-floor-weight", type=float, default=0.05)
    parser.add_argument("--logit-clip", type=float, default=8.0)
    parser.add_argument("--train-temperature", type=float, default=1.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/experiments/322/FacetExpansionV1/outputs_longhorizon"),
    )
    return parser.parse_args()


def candidate_from_edit(seed_mask: torch.Tensor, edit_prob: torch.Tensor) -> torch.Tensor:
    return seed_mask * (1.0 - edit_prob) + (1.0 - seed_mask) * edit_prob


def soft_jaccard_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection = torch.minimum(left, right).sum(dim=-1)
    union = torch.maximum(left, right).sum(dim=-1).clamp_min(1e-6)
    return intersection / union


def archive_repulsion(candidate_prob: torch.Tensor, archive_masks: torch.Tensor | None) -> torch.Tensor:
    if archive_masks is None or archive_masks.numel() == 0:
        return torch.zeros(candidate_prob.shape[0], device=candidate_prob.device, dtype=candidate_prob.dtype)
    similarities = []
    for prototype in archive_masks:
        proto = prototype.unsqueeze(0).expand(candidate_prob.shape[0], -1)
        similarities.append(soft_jaccard_similarity(candidate_prob, proto))
    stacked = torch.stack(similarities, dim=0)
    return stacked.max(dim=0).values


def build_archive_context(seed_mask: torch.Tensor, archive_masks: torch.Tensor | None) -> torch.Tensor:
    if archive_masks is None or archive_masks.numel() == 0:
        return torch.full_like(seed_mask, 0.5)
    mean_archive = archive_masks.float().mean(dim=0, keepdim=True)
    return mean_archive.expand(seed_mask.shape[0], -1)


def entropy_bonus(probabilities: torch.Tensor) -> torch.Tensor:
    probs = probabilities.clamp(1e-5, 1.0 - 1e-5)
    return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log()).sum(dim=-1)


def add_to_archive(archive_items: list[tuple[torch.Tensor, str]], candidate_masks: torch.Tensor, analyzed: list[dict[str, object]], capacity: int) -> int:
    added = 0
    for mask_tensor, item in zip(candidate_masks, analyzed):
        if item.get("tier") != "exact_match":
            continue
        canonical_key = item.get("canonical_key")
        if canonical_key is None:
            continue
        if any(existing[1] == str(canonical_key) for existing in archive_items):
            continue
        archive_items.append((mask_tensor.detach().cpu(), str(canonical_key)))
        added += 1
    if len(archive_items) > capacity:
        del archive_items[:-capacity]
    return added


def hard_exact_masks(
    candidate_masks: torch.Tensor,
    energy: GeometricHyperplaneEnergy,
    reference: dict[str, object],
    parent_ids: list[int],
) -> tuple[list[dict[str, object]], list[torch.Tensor], list[int], list[int]]:
    analyzed = []
    aligned_masks = []
    aligned_parent_ids = []
    aligned_candidate_indices = []
    for index, hard_mask in enumerate(candidate_masks):
        validation = classify_hard_mask(hard_mask >= 0.5, energy=energy, reference=reference)
        if validation is None:
            continue
        validation["hard_energy"] = energy.energy_of_hard_mask(hard_mask >= 0.5)
        analyzed.append(validation)
        aligned_masks.append(hard_mask.detach().cpu())
        aligned_parent_ids.append(int(parent_ids[index]))
        aligned_candidate_indices.append(index)
    return analyzed, aligned_masks, aligned_parent_ids, aligned_candidate_indices


def build_candidate_records(
    analyzed: list[dict[str, object]],
    aligned_masks: list[torch.Tensor],
    aligned_parent_ids: list[int],
    aligned_candidate_indices: list[int],
    round_index: int,
) -> list[CandidateRecord]:
    records = []
    for item, mask_tensor, parent_id, candidate_index in zip(analyzed, aligned_masks, aligned_parent_ids, aligned_candidate_indices):
        matched_classes = [int(class_id) for class_id in item.get("matched_classes", [])]
        records.append(
            CandidateRecord(
                branch_id=-1,
                parent_branch_id=int(parent_id),
                mask=mask_tensor.detach().cpu(),
                canonical_key=item.get("canonical_key"),
                matched_classes=matched_classes,
                tier=str(item.get("tier", "")),
                round_index=round_index,
                exact=item.get("tier") == "exact_match",
                quality_score=float(item.get("selection_score", 0.0)),
                source_index=int(candidate_index),
            )
        )
    return records


def build_bridge_candidate_records(
    candidate_masks: torch.Tensor,
    batch_seed: torch.Tensor,
    parent_ids: list[int],
    energy_terms: torch.Tensor,
    repulsion_terms: torch.Tensor,
    expected_edit: torch.Tensor,
    novelty_scores: torch.Tensor,
    round_index: int,
    topk: int,
    energy_weight: float,
    repulsion_weight: float,
    edit_weight: float,
    novelty_weight: float,
) -> list[CandidateRecord]:
    if candidate_masks.numel() == 0 or topk <= 0:
        return []
    topk = min(int(topk), candidate_masks.shape[0])
    normalized_edit = (expected_edit / max(batch_seed.shape[1], 1)).detach().cpu()
    candidate_quality = (
        (-energy_weight * energy_terms)
        + (repulsion_weight * repulsion_terms)
        + (edit_weight * normalized_edit)
        + (novelty_weight * novelty_scores.detach().cpu())
    ).detach().cpu()
    ranked_indices = torch.argsort(candidate_quality, descending=True).tolist()
    selected_indices: list[int] = []
    used_per_parent: dict[int, int] = {}
    for index in ranked_indices:
        parent_id = int(parent_ids[index])
        if used_per_parent.get(parent_id, 0) >= 1:
            continue
        if any(
            float(torch.minimum(candidate_masks[index], candidate_masks[chosen]).sum().item())
            / max(float(torch.maximum(candidate_masks[index], candidate_masks[chosen]).sum().item()), 1.0)
            >= 0.85
            for chosen in selected_indices
        ):
            continue
        selected_indices.append(index)
        used_per_parent[parent_id] = used_per_parent.get(parent_id, 0) + 1
        if len(selected_indices) >= topk:
            break
    records = []
    for index in selected_indices:
        records.append(
            CandidateRecord(
                branch_id=-1,
                parent_branch_id=int(parent_ids[index]),
                mask=candidate_masks[index].detach().cpu(),
                canonical_key=None,
                matched_classes=[],
                tier="bridge_candidate",
                round_index=round_index,
                exact=False,
                quality_score=float(candidate_quality[index].item()),
                source_index=int(index),
            )
        )
    return records


def build_branch_history_batch(
    branches: list[BranchState],
    num_nodes: int,
    device: torch.device,
    branch_batch_size: int,
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
        history_batch.append(stacked.unsqueeze(0).expand(branch_batch_size, -1, -1))
    if not history_batch:
        return torch.zeros(0, history_length, num_nodes, device=device)
    return torch.cat(history_batch, dim=0).to(device)


def summarize_frontier_diversity(frontier: list[BranchState]) -> dict[str, float]:
    if len(frontier) <= 1:
        bridge_count = sum(1 for branch in frontier if branch.canonical_key is None)
        return {
            "bridge_frontier_count": float(bridge_count),
            "bridge_frontier_fraction": float(bridge_count / len(frontier)) if frontier else 0.0,
            "frontier_mean_similarity": 1.0 if frontier else 0.0,
        }
    similarities = []
    for index, left in enumerate(frontier):
        for right in frontier[index + 1 :]:
            left_mask = left.mask >= 0.5
            right_mask = right.mask >= 0.5
            intersection = (left_mask & right_mask).sum().item()
            union = (left_mask | right_mask).sum().item()
            similarities.append(float(intersection / union) if union else 1.0)
    bridge_count = sum(1 for branch in frontier if branch.canonical_key is None)
    return {
        "bridge_frontier_count": float(bridge_count),
        "bridge_frontier_fraction": float(bridge_count / len(frontier)),
        "frontier_mean_similarity": float(sum(similarities) / len(similarities)) if similarities else 1.0,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)
    energy = GeometricHyperplaneEnergy(points=points, config=EnergyConfig())
    reference = build_reference_database()

    seed_items = load_seed_bank(points=points)
    seed_masks = seed_tensor(seed_items, device=device)
    seed_summary = summarize_seed_bank(seed_items)

    model = ConditionalEditModel(
        point_dim=points.shape[1],
        edge_types=len(graph.generator_names),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        history_layers=args.history_layers,
        max_history=args.history_length,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    controller = LongHorizonController(
        frontier_size=args.frontier_size,
        max_children_per_parent=args.controller_max_children_per_parent,
        max_same_class=args.controller_max_same_class,
        history_limit=args.history_length,
        allow_bridge_candidates=args.bridge_candidates,
        max_bridge_children=args.max_bridge_children,
        bridge_similarity_threshold=args.bridge_similarity_threshold,
    )

    archive_items: list[tuple[torch.Tensor, str]] = [(mask.cpu(), f"seed:{index}") for index, mask in enumerate(seed_masks)]
    frontier = [seed_branch(seed_masks[0], branch_id=0, history_limit=args.history_length)]
    next_branch_id = 1
    seen_canonical: set[str] = set()
    seen_classes: set[int] = set()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "facet_expander_v1.pt"
    summary_path = args.output_dir / "training_summary.json"

    history = []
    best_loss = float("inf")
    best_payload = None
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        archive_tensor = torch.stack([item[0] for item in archive_items[-args.archive_capacity :]], dim=0).to(device)
        loss_terms = []
        step_metrics = []
        epoch_new_archive = 0
        epoch_exact_classes: set[int] = set()
        epoch_candidate_records = 0
        local_frontier = frontier[:]
        local_next_branch_id = next_branch_id

        for step_index in range(1, args.rollout_steps + 1):
            if not local_frontier:
                seed_index = (epoch + step_index - 2) % seed_masks.shape[0]
                local_frontier = [seed_branch(seed_masks[seed_index], branch_id=local_next_branch_id, history_limit=args.history_length)]
                local_next_branch_id += 1

            active_frontier = local_frontier[: args.frontier_size]
            branch_batch_size = max(1, min(args.samples_per_branch, max(1, args.batch_size // max(1, len(active_frontier)))))
            seed_batches = []
            parent_ids = []
            for branch in active_frontier:
                expanded = branch.mask.to(device).unsqueeze(0).expand(branch_batch_size, -1)
                seed_batches.append(expanded)
                parent_ids.extend([branch.branch_id] * branch_batch_size)
            batch_seed = torch.cat(seed_batches, dim=0)
            batch_history = build_branch_history_batch(
                branches=active_frontier,
                num_nodes=seed_masks.shape[1],
                device=device,
                branch_batch_size=branch_batch_size,
                history_length=args.history_length,
            )

            archive_context = build_archive_context(batch_seed, archive_masks=archive_tensor)
            edit_logits, _ = model(
                point_features=points,
                seed_mask=batch_seed,
                archive_context=archive_context,
                edge_index=edge_index,
                edge_type=edge_type,
                trajectory_masks=batch_history,
            )
            edit_logits = edit_logits.clamp(-args.logit_clip, args.logit_clip)
            safe_temperature = max(float(args.train_temperature), 1e-6)
            edit_prob = torch.sigmoid(edit_logits / safe_temperature)
            candidate_prob = candidate_from_edit(batch_seed, edit_prob)

            energy_terms = energy(candidate_prob)
            expected_edit = edit_prob.sum(dim=-1)
            low_violation = torch.relu(args.edit_min - expected_edit)
            high_violation = torch.relu(expected_edit - args.edit_max)
            low_edit_loss = low_violation.pow(2)
            high_edit_loss = high_violation.pow(2)
            edit_band_loss = args.edit_min_penalty_scale * low_edit_loss + high_edit_loss
            repulsion = archive_repulsion(candidate_prob, archive_masks=archive_tensor)
            entropy = entropy_bonus(edit_prob)
            entropy_floor_loss = torch.relu(args.entropy_floor - entropy)
            rollout_edit = torch.bernoulli(edit_prob.detach())
            hard_candidates = candidate_from_edit(batch_seed.detach(), rollout_edit).float()
            _, novelty_logits = model(
                point_features=points,
                seed_mask=batch_seed,
                archive_context=archive_context,
                edge_index=edge_index,
                edge_type=edge_type,
                trajectory_masks=batch_history,
                novelty_candidate_mask=hard_candidates,
            )
            novelty_targets = torch.zeros_like(novelty_logits)

            step_loss = (
                energy_terms["total"].mean()
                + args.edit_band_weight * edit_band_loss.mean()
                + args.archive_repulsion_weight * repulsion.mean()
                + args.entropy_floor_weight * entropy_floor_loss.mean()
                - args.entropy_weight * entropy.mean()
            )
            step_metrics.append(
                {
                    "energy": energy_terms["total"].mean().detach(),
                    "expected_edit": expected_edit.mean().detach(),
                    "repulsion": repulsion.mean().detach(),
                    "entropy": entropy.mean().detach(),
                    "entropy_floor_loss": entropy_floor_loss.mean().detach(),
                    "novelty_logit": novelty_logits.mean().detach(),
                }
            )
            analyzed, aligned_masks, aligned_parent_ids, aligned_candidate_indices = hard_exact_masks(
                hard_candidates,
                energy=energy,
                reference=reference,
                parent_ids=parent_ids,
            )
            for item, candidate_index in zip(analyzed, aligned_candidate_indices):
                canonical_key = item.get("canonical_key")
                matched_classes = [int(class_id) for class_id in item.get("matched_classes", [])]
                is_novel = (
                    (canonical_key is not None and canonical_key not in seen_canonical)
                    or any(class_id not in seen_classes for class_id in matched_classes)
                )
                if is_novel:
                    novelty_targets[candidate_index] = 1.0
            novelty_loss = torch.nn.functional.binary_cross_entropy_with_logits(novelty_logits, novelty_targets)
            step_loss = step_loss + args.novelty_head_weight * novelty_loss
            loss_terms.append(step_loss)
            candidate_records = build_candidate_records(
                analyzed=analyzed,
                aligned_masks=aligned_masks,
                aligned_parent_ids=aligned_parent_ids,
                aligned_candidate_indices=aligned_candidate_indices,
                round_index=step_index,
            )
            if args.bridge_candidates:
                candidate_records.extend(
                    build_bridge_candidate_records(
                        candidate_masks=hard_candidates,
                        batch_seed=batch_seed,
                        parent_ids=parent_ids,
                        energy_terms=energy_terms["total"].detach(),
                        repulsion_terms=repulsion.detach(),
                        expected_edit=expected_edit.detach(),
                        novelty_scores=torch.sigmoid(novelty_logits.detach()),
                        round_index=step_index,
                        topk=max(args.max_bridge_children * max(1, len(active_frontier)), args.max_bridge_children),
                        energy_weight=args.bridge_energy_weight,
                        repulsion_weight=args.bridge_repulsion_weight,
                        edit_weight=args.bridge_edit_weight,
                        novelty_weight=args.bridge_novelty_weight,
                    )
                )
            epoch_candidate_records += len(candidate_records)
            epoch_exact_classes.update(int(class_id) for item in analyzed for class_id in item.get("matched_classes", []))
            local_frontier, local_next_branch_id = controller.update_frontier(
                frontier=active_frontier,
                candidates=candidate_records,
                seen_canonical=seen_canonical,
                seen_classes=seen_classes,
                next_branch_id=local_next_branch_id,
                round_index=step_index,
            )
            selected_bridge_indices = {
                record.source_index
                for record in candidate_records
                if not record.exact
                and any(torch.equal(branch.mask, record.mask) for branch in local_frontier if branch.canonical_key is None)
            }
            for source_index in selected_bridge_indices:
                if 0 <= source_index < novelty_targets.shape[0]:
                    novelty_targets[source_index] = max(float(novelty_targets[source_index].item()), 0.5)
            novelty_loss = torch.nn.functional.binary_cross_entropy_with_logits(novelty_logits, novelty_targets)
            step_loss = (
                energy_terms["total"].mean()
                + args.edit_band_weight * edit_band_loss.mean()
                + args.archive_repulsion_weight * repulsion.mean()
                + args.entropy_floor_weight * entropy_floor_loss.mean()
                - args.entropy_weight * entropy.mean()
                + args.novelty_head_weight * novelty_loss
            )
            loss_terms[-1] = step_loss
            if aligned_masks:
                epoch_new_archive += add_to_archive(
                    archive_items=archive_items,
                    candidate_masks=torch.stack(aligned_masks, dim=0),
                    analyzed=analyzed,
                    capacity=args.archive_capacity,
                )
                archive_tensor = torch.stack([item[0] for item in archive_items[-args.archive_capacity :]], dim=0).to(device)
            for branch in local_frontier:
                if branch.canonical_key is not None:
                    seen_canonical.add(branch.canonical_key)
                for class_id in branch.matched_classes:
                    seen_classes.add(int(class_id))

        loss = torch.stack(loss_terms).mean()
        if not torch.isfinite(loss):
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        frontier = local_frontier if local_frontier else frontier
        next_branch_id = local_next_branch_id
        diversity_summary = summarize_frontier_diversity(frontier)

        epoch_summary = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "mean_energy": float(torch.stack([item["energy"] for item in step_metrics]).mean().item()),
            "mean_expected_edit": float(torch.stack([item["expected_edit"] for item in step_metrics]).mean().item()),
            "mean_archive_repulsion": float(torch.stack([item["repulsion"] for item in step_metrics]).mean().item()),
            "mean_edit_entropy": float(torch.stack([item["entropy"] for item in step_metrics]).mean().item()),
            "mean_entropy_floor_loss": float(torch.stack([item["entropy_floor_loss"] for item in step_metrics]).mean().item()),
            "mean_novelty_logit": float(torch.stack([item["novelty_logit"] for item in step_metrics]).mean().item()),
            "archive_size": len(archive_items),
            "frontier_size": len(frontier),
            "seen_exact_classes": sorted(seen_classes),
            "epoch_exact_classes": sorted(epoch_exact_classes),
            "epoch_candidate_records": epoch_candidate_records,
            "epoch_new_archive_items": epoch_new_archive,
            "grad_norm": float(torch.as_tensor(grad_norm).item()),
            **diversity_summary,
        }

        if epoch_summary["loss"] < best_loss:
            best_loss = epoch_summary["loss"]
            best_payload = {
                "model_state": model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "history_dim": args.hidden_dim,
                    "history_layers": args.history_layers,
                    "max_history": args.history_length,
                },
                "training_args": serializable_args,
                "seed_summary": seed_summary,
                "best_epoch": epoch,
                "best_loss": best_loss,
            }
            torch.save(best_payload, checkpoint_path)

        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))

    if best_payload is not None:
        summary_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "best_loss": best_loss,
                    "seed_summary": seed_summary,
                    "history": history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
