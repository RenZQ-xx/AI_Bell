from __future__ import annotations

import argparse
import json
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
    from inference import analyze_candidate_probabilities
    from editor_model import ConditionalEditModel
    from seed_bank import load_seed_bank, seed_tensor
    from symmetry_graph import build_symmetry_graph
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import analyze_candidate_probabilities
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .editor_model import ConditionalEditModel
    from .seed_bank import load_seed_bank, seed_tensor


def candidate_from_edit(seed_mask: torch.Tensor, edit_prob: torch.Tensor) -> torch.Tensor:
    return seed_mask * (1.0 - edit_prob) + (1.0 - seed_mask) * edit_prob


def build_archive_context(seed_mask: torch.Tensor, archive_masks: torch.Tensor | None) -> torch.Tensor:
    if archive_masks is None or archive_masks.numel() == 0:
        return torch.full_like(seed_mask, 0.5)
    mean_archive = archive_masks.float().mean(dim=0, keepdim=True)
    return mean_archive.expand(seed_mask.shape[0], -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conditional facet expansion inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--frontier-size", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true", help="Sample Bernoulli edit masks instead of using deterministic probabilities.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature applied to edit logits before sigmoid.")
    parser.add_argument("--prefer-novel", action="store_true", help="Filter repeated canonical orbits from the rollout frontier whenever possible.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


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
    archive_list = [seed_masks[index].detach().cpu() for index in range(min(args.archive_size, seed_masks.shape[0]))]
    frontier = [seed_masks[seed_index].detach().cpu()]

    model = ConditionalEditModel(
        point_dim=points.shape[1],
        edge_types=len(graph.generator_names),
        hidden_dim=payload["model_config"]["hidden_dim"],
        layers=payload["model_config"]["layers"],
        dropout=payload["model_config"]["dropout"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    all_results = []
    seen_canonical = set()
    round_summaries = []
    seen_exact_class_counts: dict[int, int] = {}

    with torch.no_grad():
        for round_index in range(args.rounds):
            archive_masks = torch.stack(archive_list[-args.archive_size :], dim=0).to(device)
            next_frontier = []
            round_results = []
            for frontier_mask in frontier[: args.frontier_size]:
                seed_mask = frontier_mask.to(device).unsqueeze(0).expand(args.num_samples, -1)
                archive_context = build_archive_context(seed_mask, archive_masks=archive_masks)
                logits = model(points, seed_mask, archive_context, edge_index, edge_type)
                safe_temperature = max(float(args.temperature), 1e-6)
                edit_prob = torch.sigmoid(logits / safe_temperature)
                if args.stochastic:
                    sampled_edit = torch.bernoulli(edit_prob)
                    candidate_prob = candidate_from_edit(seed_mask, sampled_edit)
                else:
                    candidate_prob = candidate_from_edit(seed_mask, edit_prob)
                analyzed = analyze_candidate_probabilities(
                    probabilities=candidate_prob.detach().cpu(),
                    energy=energy,
                    reference=reference,
                )
                for item in analyzed:
                    item["round"] = round_index + 1
                round_results.extend(analyzed)

                if args.prefer_novel:
                    prioritized = []
                    deferred = []
                    for item in analyzed:
                        canonical_key = item.get("canonical_key")
                        matched_classes = tuple(item.get("matched_classes", []))
                        is_new_canonical = canonical_key is not None and canonical_key not in seen_canonical
                        is_new_class = any(class_id not in seen_exact_class_counts for class_id in matched_classes)
                        if is_new_canonical or is_new_class:
                            prioritized.append(item)
                        else:
                            deferred.append(item)
                    analyzed_for_frontier = prioritized if prioritized else deferred
                else:
                    analyzed_for_frontier = analyzed

                for item in analyzed_for_frontier:
                    canonical_key = item.get("canonical_key")
                    if item.get("tier") != "exact_match" or canonical_key is None or canonical_key in seen_canonical:
                        continue
                    seen_canonical.add(canonical_key)
                    for class_id in item.get("matched_classes", []):
                        seen_exact_class_counts[int(class_id)] = seen_exact_class_counts.get(int(class_id), 0) + 1
                    new_mask = torch.zeros(seed_masks.shape[1], dtype=torch.float32)
                    new_mask[item["indices"]] = 1.0
                    archive_list.append(new_mask)
                    next_frontier.append(new_mask)

            if len(archive_list) > args.archive_size:
                archive_list = archive_list[-args.archive_size :]
            if not next_frontier:
                next_frontier = frontier[:1]
            frontier = next_frontier
            round_summaries.append(
                {
                    "round": round_index + 1,
                    "num_results": len(round_results),
                    "exact_count": sum(1 for item in round_results if item.get("tier") == "exact_match"),
                    "matched_classes": sorted({class_id for item in round_results for class_id in item.get("matched_classes", [])}),
                    "new_canonical_added": len(next_frontier),
                    "seen_exact_classes": sorted(seen_exact_class_counts.keys()),
                    "archive_size": len(archive_list),
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
