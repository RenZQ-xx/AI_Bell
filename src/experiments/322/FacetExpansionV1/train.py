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
    from seed_bank import load_seed_bank, seed_tensor, summarize_seed_bank
    from symmetry_graph import build_symmetry_graph
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .editor_model import ConditionalEditModel
    from .seed_bank import load_seed_bank, seed_tensor, summarize_seed_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first conditional facet expander for 3-2-2.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--rollout-steps", type=int, default=2, help="Number of on-policy expansion steps per training epoch.")
    parser.add_argument("--frontier-sample-size", type=int, default=8, help="Maximum number of frontier masks expanded at each rollout step.")
    parser.add_argument("--edit-min", type=float, default=3.0, help="Minimum expected number of flips away from the seed.")
    parser.add_argument("--edit-max", type=float, default=18.0, help="Maximum expected number of flips away from the seed.")
    parser.add_argument("--edit-band-weight", type=float, default=0.8)
    parser.add_argument(
        "--edit-min-penalty-scale",
        type=float,
        default=3.0,
        help="Extra multiplier applied to under-editing violations to prevent collapse toward zero edits.",
    )
    parser.add_argument("--archive-repulsion-weight", type=float, default=0.6)
    parser.add_argument("--archive-capacity", type=int, default=256)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument(
        "--entropy-floor",
        type=float,
        default=8.0,
        help="Minimum total edit entropy per sample before an additional stability penalty is applied.",
    )
    parser.add_argument(
        "--entropy-floor-weight",
        type=float,
        default=0.05,
        help="Weight on the entropy-floor stability penalty.",
    )
    parser.add_argument("--logit-clip", type=float, default=8.0)
    parser.add_argument(
        "--train-stochastic-rollout",
        action="store_true",
        help="Use Bernoulli-sampled edit masks for train-time frontier/archive rollout instead of deterministic thresholding.",
    )
    parser.add_argument(
        "--train-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to edit logits before stochastic train-time rollout sampling.",
    )
    parser.add_argument(
        "--prefer-novel-frontier",
        action="store_true",
        help="Prefer new canonical/class exact matches when selecting the next rollout frontier.",
    )
    parser.add_argument(
        "--frontier-repeat-cap",
        type=int,
        default=1,
        help="Maximum number of repeated canonical/class items kept in each rollout frontier step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/experiments/322/FacetExpansionV1/outputs"),
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
    parent_ids: list[int] | None = None,
) -> tuple[list[dict[str, object]], list[torch.Tensor], list[int]]:
    analyzed = []
    aligned_masks = []
    aligned_parent_ids = []
    for index, hard_mask in enumerate(candidate_masks):
        validation = classify_hard_mask(hard_mask >= 0.5, energy=energy, reference=reference)
        if validation is None:
            continue
        validation["hard_energy"] = energy.energy_of_hard_mask(hard_mask >= 0.5)
        analyzed.append(validation)
        aligned_masks.append(hard_mask.detach().cpu())
        if parent_ids is not None:
            aligned_parent_ids.append(int(parent_ids[index]))
    return analyzed, aligned_masks, aligned_parent_ids


def select_frontier_candidates(
    analyzed: list[dict[str, object]],
    aligned_masks: list[torch.Tensor],
    archive_items: list[tuple[torch.Tensor, str]],
    seen_exact_class_counts: dict[int, int],
    prefer_novel: bool,
    frontier_sample_size: int,
    repeated_cap: int,
    parent_ids: list[int],
) -> tuple[list[torch.Tensor], int, int]:
    archive_keys = {canonical_key for _, canonical_key in archive_items}
    candidate_items = []
    for mask_tensor, item, parent_id in zip(aligned_masks, analyzed, parent_ids):
        if item.get("tier") != "exact_match":
            continue
        canonical_key = item.get("canonical_key")
        if canonical_key is None:
            continue
        matched_classes = [int(class_id) for class_id in item.get("matched_classes", [])]
        is_new_canonical = str(canonical_key) not in archive_keys
        is_new_class = any(class_id not in seen_exact_class_counts for class_id in matched_classes)
        candidate_items.append(
            {
                "mask": mask_tensor,
                "canonical_key": str(canonical_key),
                "matched_classes": matched_classes,
                "is_new_canonical": is_new_canonical,
                "is_new_class": is_new_class,
                "parent_id": int(parent_id),
            }
        )

    if prefer_novel:
        prioritized = [item for item in candidate_items if item["is_new_canonical"] or item["is_new_class"]]
        deferred = [item for item in candidate_items if not (item["is_new_canonical"] or item["is_new_class"])]
        ordered_items = prioritized + deferred
    else:
        ordered_items = candidate_items

    next_frontier = []
    added_canonical = set()
    used_parent_ids = set()
    novel_frontier_count = 0
    repeated_frontier_count = 0
    repeated_used = 0

    def try_add(item: dict[str, object]) -> bool:
        nonlocal repeated_used, novel_frontier_count, repeated_frontier_count
        if item["canonical_key"] in added_canonical:
            return False
        is_repeated = not (item["is_new_canonical"] or item["is_new_class"])
        if is_repeated and repeated_used >= max(repeated_cap, 0):
            return False
        added_canonical.add(item["canonical_key"])
        used_parent_ids.add(int(item["parent_id"]))
        next_frontier.append(item["mask"])
        if is_repeated:
            repeated_frontier_count += 1
            repeated_used += 1
        else:
            novel_frontier_count += 1
        for class_id in item["matched_classes"]:
            seen_exact_class_counts[class_id] = seen_exact_class_counts.get(class_id, 0) + 1
        return True

    for item in ordered_items:
        if int(item["parent_id"]) in used_parent_ids:
            continue
        if try_add(item) and len(next_frontier) >= frontier_sample_size:
            break

    if len(next_frontier) < frontier_sample_size:
        for item in ordered_items:
            if try_add(item) and len(next_frontier) >= frontier_sample_size:
                break

    if not next_frontier:
        fallback_limit = min(frontier_sample_size, len(candidate_items))
        for item in candidate_items[:fallback_limit]:
            if try_add(item) and len(next_frontier) >= frontier_sample_size:
                break
    return next_frontier, novel_frontier_count, repeated_frontier_count


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
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    archive_items: list[tuple[torch.Tensor, str]] = [(mask.cpu(), f"seed:{index}") for index, mask in enumerate(seed_masks)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "facet_expander_v1.pt"
    summary_path = args.output_dir / "training_summary.json"

    history = []
    best_loss = float("inf")
    best_payload = None
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    frontier_items: list[torch.Tensor] = [mask.cpu() for mask in seed_masks[: min(args.frontier_sample_size, seed_masks.shape[0])]]
    seen_exact_class_counts: dict[int, int] = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        archive_tensor = None
        if archive_items:
            archive_tensor = torch.stack([item[0] for item in archive_items[-args.archive_capacity :]], dim=0).to(device)

        loss_terms = []
        step_metrics = []
        local_frontier = frontier_items[:]
        epoch_novel_frontier = 0
        epoch_repeated_frontier = 0
        for step_index in range(args.rollout_steps):
            parent_ids: list[int] = []
            if local_frontier:
                choose = min(len(local_frontier), args.frontier_sample_size, args.batch_size)
                frontier_indices = torch.randperm(len(local_frontier))[:choose].tolist()
                frontier_seed = torch.stack([local_frontier[index] for index in frontier_indices], dim=0).to(device)
                parent_ids.extend(frontier_indices)
            else:
                choose = min(seed_masks.shape[0], args.batch_size)
                batch_indices = torch.randint(0, seed_masks.shape[0], (choose,), device=device)
                frontier_seed = seed_masks[batch_indices]
                parent_ids.extend([int(index) for index in batch_indices.tolist()])

            if choose < args.batch_size:
                random_count = args.batch_size - choose
                random_indices = torch.randint(0, seed_masks.shape[0], (random_count,), device=device)
                random_seed = seed_masks[random_indices]
                batch_seed = torch.cat([frontier_seed, random_seed], dim=0)
                offset = len(local_frontier) if local_frontier else seed_masks.shape[0]
                parent_ids.extend([offset + int(index) for index in range(random_count)])
            else:
                batch_seed = frontier_seed

            archive_context = build_archive_context(batch_seed, archive_masks=archive_tensor)
            edit_logits = model(
                point_features=points,
                seed_mask=batch_seed,
                archive_context=archive_context,
                edge_index=edge_index,
                edge_type=edge_type,
            ).clamp(-args.logit_clip, args.logit_clip)
            safe_train_temperature = max(float(args.train_temperature), 1e-6)
            edit_prob = torch.sigmoid(edit_logits / safe_train_temperature)
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
            step_loss = (
                energy_terms["total"].mean()
                + args.edit_band_weight * edit_band_loss.mean()
                + args.archive_repulsion_weight * repulsion.mean()
                + args.entropy_floor_weight * entropy_floor_loss.mean()
                - args.entropy_weight * entropy.mean()
            )
            loss_terms.append(step_loss)
            step_metrics.append(
                {
                    "energy": energy_terms["total"].mean().detach(),
                    "plane": energy_terms["plane"].mean().detach(),
                    "boundary": energy_terms["boundary"].mean().detach(),
                    "cardinality": energy_terms["cardinality"].mean().detach(),
                    "expected_edit": expected_edit.mean().detach(),
                    "low_edit_loss": low_edit_loss.mean().detach(),
                    "high_edit_loss": high_edit_loss.mean().detach(),
                    "edit_band_loss": edit_band_loss.mean().detach(),
                    "repulsion": repulsion.mean().detach(),
                    "entropy": entropy.mean().detach(),
                    "entropy_floor_loss": entropy_floor_loss.mean().detach(),
                }
            )

            if args.train_stochastic_rollout:
                rollout_edit = torch.bernoulli(edit_prob.detach())
                hard_candidates = candidate_from_edit(batch_seed.detach(), rollout_edit).float()
            else:
                hard_candidates = (candidate_prob.detach() >= 0.5).float()
            analyzed, aligned_masks, aligned_parent_ids = hard_exact_masks(
                hard_candidates,
                energy=energy,
                reference=reference,
                parent_ids=parent_ids,
            )
            next_frontier, novel_frontier_count, repeated_frontier_count = select_frontier_candidates(
                analyzed=analyzed,
                aligned_masks=aligned_masks,
                archive_items=archive_items,
                seen_exact_class_counts=seen_exact_class_counts,
                prefer_novel=args.prefer_novel_frontier,
                frontier_sample_size=args.frontier_sample_size,
                repeated_cap=args.frontier_repeat_cap,
                parent_ids=aligned_parent_ids,
            )
            epoch_novel_frontier += novel_frontier_count
            epoch_repeated_frontier += repeated_frontier_count
            if next_frontier:
                local_frontier = next_frontier[: args.frontier_sample_size]
            if aligned_masks:
                add_to_archive(
                    archive_items=archive_items,
                    candidate_masks=torch.stack(aligned_masks, dim=0),
                    analyzed=analyzed,
                    capacity=args.archive_capacity,
                )
                archive_tensor = torch.stack([item[0] for item in archive_items[-args.archive_capacity :]], dim=0).to(device)

        loss = torch.stack(loss_terms).mean()
        if not torch.isfinite(loss):
            epoch_summary = {
                "epoch": epoch,
                "loss": float("nan"),
                "status": "nonfinite_loss_before_backward",
                "archive_size": len(archive_items),
                "frontier_size": len(local_frontier),
                "novel_frontier_count": epoch_novel_frontier,
                "repeated_frontier_count": epoch_repeated_frontier,
                "seen_exact_classes": sorted(seen_exact_class_counts.keys()),
            }
            history.append(epoch_summary)
            print(json.dumps(epoch_summary, ensure_ascii=False))
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            epoch_summary = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "status": "nonfinite_grad_norm_after_backward",
                "archive_size": len(archive_items),
                "frontier_size": len(local_frontier),
                "novel_frontier_count": epoch_novel_frontier,
                "repeated_frontier_count": epoch_repeated_frontier,
                "seen_exact_classes": sorted(seen_exact_class_counts.keys()),
            }
            history.append(epoch_summary)
            print(json.dumps(epoch_summary, ensure_ascii=False))
            break
        optimizer.step()
        parameters_finite = all(torch.isfinite(parameter).all() for parameter in model.parameters())
        if not parameters_finite:
            epoch_summary = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "status": "nonfinite_parameters_after_step",
                "archive_size": len(archive_items),
                "frontier_size": len(local_frontier),
                "novel_frontier_count": epoch_novel_frontier,
                "repeated_frontier_count": epoch_repeated_frontier,
                "seen_exact_classes": sorted(seen_exact_class_counts.keys()),
            }
            history.append(epoch_summary)
            print(json.dumps(epoch_summary, ensure_ascii=False))
            break
        frontier_items = local_frontier if local_frontier else frontier_items

        mean_energy = torch.stack([item["energy"] for item in step_metrics]).mean()
        mean_plane = torch.stack([item["plane"] for item in step_metrics]).mean()
        mean_boundary = torch.stack([item["boundary"] for item in step_metrics]).mean()
        mean_cardinality = torch.stack([item["cardinality"] for item in step_metrics]).mean()
        mean_expected_edit = torch.stack([item["expected_edit"] for item in step_metrics]).mean()
        mean_low_edit_loss = torch.stack([item["low_edit_loss"] for item in step_metrics]).mean()
        mean_high_edit_loss = torch.stack([item["high_edit_loss"] for item in step_metrics]).mean()
        mean_edit_band_loss = torch.stack([item["edit_band_loss"] for item in step_metrics]).mean()
        mean_repulsion = torch.stack([item["repulsion"] for item in step_metrics]).mean()
        mean_entropy = torch.stack([item["entropy"] for item in step_metrics]).mean()
        mean_entropy_floor_loss = torch.stack([item["entropy_floor_loss"] for item in step_metrics]).mean()
        epoch_summary = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "mean_energy": float(mean_energy.item()),
            "mean_plane": float(mean_plane.item()),
            "mean_boundary": float(mean_boundary.item()),
            "mean_cardinality": float(mean_cardinality.item()),
            "mean_expected_edit": float(mean_expected_edit.item()),
            "mean_low_edit_loss": float(mean_low_edit_loss.item()),
            "mean_high_edit_loss": float(mean_high_edit_loss.item()),
            "mean_edit_band_loss": float(mean_edit_band_loss.item()),
            "mean_archive_repulsion": float(mean_repulsion.item()),
            "mean_edit_entropy": float(mean_entropy.item()),
            "mean_entropy_floor_loss": float(mean_entropy_floor_loss.item()),
            "archive_size": len(archive_items),
            "frontier_size": len(frontier_items),
            "novel_frontier_count": epoch_novel_frontier,
            "repeated_frontier_count": epoch_repeated_frontier,
            "seen_exact_classes": sorted(seen_exact_class_counts.keys()),
            "grad_norm": float(torch.as_tensor(grad_norm).item()),
        }

        if epoch_summary["loss"] < best_loss:
            best_loss = epoch_summary["loss"]
            best_payload = {
                "model_state": model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
                "training_args": serializable_args,
                "seed_summary": seed_summary,
                "best_epoch": epoch,
                "best_loss": best_loss,
            }
            torch.save(best_payload, checkpoint_path)

        should_sample = args.sample_every > 0 and (epoch % args.sample_every == 0 or epoch == args.epochs)
        if should_sample:
            model.eval()
            with torch.no_grad():
                sample_indices = torch.randint(0, seed_masks.shape[0], (args.batch_size,), device=device)
                sample_seed = seed_masks[sample_indices]
                sample_edit_prob = torch.sigmoid(
                    model(
                        point_features=points,
                        seed_mask=sample_seed,
                        archive_context=build_archive_context(sample_seed, archive_masks=archive_tensor),
                        edge_index=edge_index,
                        edge_type=edge_type,
                    ).clamp(-args.logit_clip, args.logit_clip)
                )
                sample_candidate_prob = candidate_from_edit(sample_seed, sample_edit_prob)
                hard_candidates = (sample_candidate_prob >= 0.5).float()
                analyzed = []
                aligned_masks = []
                for hard_mask in hard_candidates:
                    validation = classify_hard_mask(hard_mask >= 0.5, energy=energy, reference=reference)
                    if validation is None:
                        continue
                    validation["hard_energy"] = energy.energy_of_hard_mask(hard_mask >= 0.5)
                    analyzed.append(validation)
                    aligned_masks.append(hard_mask)
                exact_classes = sorted({class_id for item in analyzed for class_id in item.get("matched_classes", [])})
                new_archive_items = add_to_archive(
                    archive_items=archive_items,
                    candidate_masks=torch.stack(aligned_masks, dim=0) if aligned_masks else hard_candidates[:0],
                    analyzed=analyzed,
                    capacity=args.archive_capacity,
                )
                epoch_summary["sample_valid"] = len(analyzed)
                epoch_summary["sample_exact_classes"] = exact_classes
                epoch_summary["sample_new_archive_items"] = new_archive_items
                epoch_summary["sample_exact_count"] = sum(1 for item in analyzed if item.get("tier") == "exact_match")

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
