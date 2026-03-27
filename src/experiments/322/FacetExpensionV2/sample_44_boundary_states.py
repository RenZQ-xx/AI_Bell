from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    EXP322_DIR = CURRENT_DIR.parent
    DIFFUCO_DIR = EXP322_DIR / "DIffUCO"
    V1_DIR = EXP322_DIR / "FacetExpansionV1"
    sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR)]
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
    from ..FacetExpansionV1.editor_model import ConditionalEditModel
    from ..FacetExpansionV1.seed_bank import load_seed_bank, seed_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a 44-basin state cloud without using non-44 seed targets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--source-class-id", type=int, default=44)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--local-samples", type=int, default=128)
    parser.add_argument("--swap-samples", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--min-swap-frac", type=float, default=0.08)
    parser.add_argument("--max-swap-frac", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_archive_context(seed_mask: torch.Tensor, archive_masks: torch.Tensor | None) -> torch.Tensor:
    if archive_masks is None or archive_masks.numel() == 0:
        return torch.full_like(seed_mask, 0.5)
    mean_archive = archive_masks.float().mean(dim=0, keepdim=True)
    return mean_archive.expand(seed_mask.shape[0], -1)


def candidate_from_edit(seed_mask: torch.Tensor, edit_prob: torch.Tensor) -> torch.Tensor:
    return seed_mask * (1.0 - edit_prob) + (1.0 - seed_mask) * edit_prob


def jaccard_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_mask = left >= 0.5
    right_mask = right >= 0.5
    intersection = (left_mask & right_mask).sum().item()
    union = (left_mask | right_mask).sum().item()
    if union == 0:
        return 1.0
    return float(intersection / union)


def sample_swap_candidate(
    source_mask: torch.Tensor,
    rng: random.Random,
    min_swap_frac: float,
    max_swap_frac: float,
) -> torch.Tensor:
    source_bool = source_mask >= 0.5
    on_indices = torch.nonzero(source_bool, as_tuple=False).view(-1).cpu().tolist()
    off_indices = torch.nonzero(~source_bool, as_tuple=False).view(-1).cpu().tolist()
    cardinality = max(len(on_indices), 1)
    min_swaps = max(1, int(round(cardinality * min_swap_frac)))
    max_swaps = max(min_swaps, int(round(cardinality * max_swap_frac)))
    num_swaps = rng.randint(min_swaps, max_swaps)
    num_swaps = min(num_swaps, len(on_indices), len(off_indices))
    remove_indices = rng.sample(on_indices, num_swaps)
    add_indices = rng.sample(off_indices, num_swaps)
    candidate = source_mask.clone()
    candidate[remove_indices] = 0.0
    candidate[add_indices] = 1.0
    return candidate


def boundary_score(*, exact: bool, matched_classes: list[int], hard_energy: float, hamming_distance: int, jaccard_to_source: float) -> float:
    score = 0.0
    if exact and matched_classes == [44]:
        score -= 3.0
    elif exact and any(class_id != 44 for class_id in matched_classes):
        score += 3.0
    else:
        score += 1.0
    score += 0.04 * float(hamming_distance)
    score += 1.5 * float(max(0.0, 1.0 - jaccard_to_source))
    score -= 0.5 * float(hard_energy)
    return score


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)
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

    source_indices = [index for index, item in enumerate(seed_items) if int(item.class_id) == int(args.source_class_id)]
    source_indices = source_indices[args.seed_start : args.seed_start + args.seed_count]
    if not source_indices:
        raise ValueError(f"No seeds found for class {args.source_class_id}")

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

    source_archive = torch.stack([seed_masks[index] for index in source_indices], dim=0)

    records: list[dict[str, object]] = []
    for source_index in source_indices:
        source_item = seed_items[source_index]
        source_mask = seed_masks[source_index]
        archive_context = build_archive_context(
            source_mask.unsqueeze(0).expand(args.local_samples, -1),
            archive_masks=source_archive,
        )

        with torch.no_grad():
            edit_logits, _ = model(
                points,
                source_mask.unsqueeze(0).expand(args.local_samples, -1),
                archive_context,
                edge_index,
                edge_type,
            )
            edit_prob = torch.sigmoid(edit_logits / max(float(args.temperature), 1e-6))
            sampled_edit = torch.bernoulli(edit_prob)
            local_candidates = (candidate_from_edit(source_mask.unsqueeze(0).expand(args.local_samples, -1), sampled_edit) >= 0.5).float().cpu()

        swap_candidates = torch.stack(
            [
                sample_swap_candidate(
                    source_mask=source_mask.cpu(),
                    rng=rng,
                    min_swap_frac=args.min_swap_frac,
                    max_swap_frac=args.max_swap_frac,
                )
                for _ in range(args.swap_samples)
            ],
            dim=0,
        )

        for candidate_type, batch in (("local_edit", local_candidates), ("swap_jump", swap_candidates)):
            for sample_index, candidate in enumerate(batch):
                validation = classify_hard_mask(candidate >= 0.5, energy=energy, reference=reference)
                exact = validation is not None
                matched_classes = [] if validation is None else [int(class_id) for class_id in validation.get("matched_classes", [])]
                hard_energy = float(energy.energy_of_hard_mask(candidate >= 0.5))
                hamming = int((candidate >= 0.5).ne(source_mask.cpu() >= 0.5).sum().item())
                jaccard = jaccard_similarity(candidate, source_mask.cpu())
                records.append(
                    {
                        "source_seed_index": int(source_index),
                        "source_class_id": int(source_item.class_id),
                        "source_example_id": int(source_item.example_id),
                        "candidate_type": candidate_type,
                        "sample_index": int(sample_index),
                        "exact": exact,
                        "matched_classes": matched_classes,
                        "canonical_key": None if validation is None else validation.get("canonical_key"),
                        "cardinality": int(candidate.sum().item()),
                        "hard_energy": hard_energy,
                        "hamming_distance": hamming,
                        "jaccard_to_source": jaccard,
                        "boundary_score": boundary_score(
                            exact=exact,
                            matched_classes=matched_classes,
                            hard_energy=hard_energy,
                            hamming_distance=hamming,
                            jaccard_to_source=jaccard,
                        ),
                        "mask": candidate.int().tolist(),
                    }
                )

    summary = {
        "source_class_id": int(args.source_class_id),
        "source_seed_indices": source_indices,
        "num_source_seeds": len(source_indices),
        "num_records": len(records),
        "num_exact_records": sum(1 for record in records if record["exact"]),
        "num_non44_exact_records": sum(
            1 for record in records if record["exact"] and any(int(class_id) != 44 for class_id in record["matched_classes"])
        ),
        "top_boundary_scores": sorted(
            [
                {
                    "source_seed_index": record["source_seed_index"],
                    "candidate_type": record["candidate_type"],
                    "matched_classes": record["matched_classes"],
                    "exact": record["exact"],
                    "boundary_score": record["boundary_score"],
                    "hard_energy": record["hard_energy"],
                    "hamming_distance": record["hamming_distance"],
                }
                for record in records
            ],
            key=lambda item: item["boundary_score"],
            reverse=True,
        )[:20],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
