from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.append(str(CURRENT_DIR))
    from diffusion import BernoulliDiffusionProcess, DiffusionConfig
    from energy import EnergyConfig, GeometricHyperplaneEnergy
    from facet_reference import build_reference_database, classify_discovered_plane
    from geometry import generate_points_322
    from model import GeometricDiffUCOModel
    from symmetry_graph import build_symmetry_graph
else:
    from .diffusion import BernoulliDiffusionProcess, DiffusionConfig
    from .energy import EnergyConfig, GeometricHyperplaneEnergy
    from .facet_reference import build_reference_database, classify_discovered_plane
    from .geometry import generate_points_322
    from .model import GeometricDiffUCOModel
    from .symmetry_graph import build_symmetry_graph


def conditional_expectation_rounding(probabilities: torch.Tensor, energy: GeometricHyperplaneEnergy, min_cardinality: int) -> torch.Tensor:
    ranked = torch.argsort(probabilities, descending=True)
    mask = probabilities.clone()
    num_nodes = int(mask.shape[0])

    for position, candidate in enumerate(ranked.tolist()):
        remaining = num_nodes - position - 1
        assigned_true = int((mask[ranked[:position]] > 0.5).sum().item()) if position > 0 else 0
        required_true = max(min_cardinality - assigned_true, 0)

        if required_true > remaining:
            mask[candidate] = 1.0
            continue
        if assigned_true + remaining < min_cardinality:
            mask[candidate] = 1.0
            continue

        trial_zero = mask.clone()
        trial_zero[candidate] = 0.0
        zero_energy = energy(trial_zero.unsqueeze(0))["total"].item()

        trial_one = mask.clone()
        trial_one[candidate] = 1.0
        one_energy = energy(trial_one.unsqueeze(0))["total"].item()

        mask[candidate] = 0.0 if zero_energy <= one_energy else 1.0

    hard_mask = mask >= 0.5
    current_cardinality = int(hard_mask.sum().item())
    if current_cardinality < min_cardinality:
        hard_mask[ranked[:min_cardinality]] = True
    return hard_mask


def _plane_embedding(normal: torch.Tensor, offset: float) -> torch.Tensor:
    normal_tensor = torch.as_tensor(normal, dtype=torch.float32).flatten().cpu()
    vector = torch.cat([normal_tensor, torch.tensor([float(offset)], dtype=torch.float32)])
    norm = vector.norm().clamp_min(1e-6)
    return vector / norm


def _candidate_quality(validation: Dict[str, object]) -> float:
    tier_bonus = 1000.0 if validation.get("tier") == "exact_match" else 0.0
    hard_energy = float(validation.get("hard_energy", 0.0))
    cardinality = float(validation.get("cardinality", 0))
    direction_error = float(validation.get("direction_error") or 0.0)
    return tier_bonus - hard_energy + 0.25 * cardinality - 10.0 * direction_error


def _plane_similarity(left: Dict[str, object], right: Dict[str, object]) -> float:
    left_embedding = left.get("plane_embedding")
    right_embedding = right.get("plane_embedding")
    if left_embedding is None or right_embedding is None:
        return 0.0
    left_vector = torch.tensor(left_embedding, dtype=torch.float32)
    right_vector = torch.tensor(right_embedding, dtype=torch.float32)
    return float(torch.abs(torch.dot(left_vector, right_vector)).clamp(0.0, 1.0).item())


def rerank_diverse_candidates(
    candidates: List[Dict[str, object]],
    max_results: int,
    diversity_weight: float,
) -> List[Dict[str, object]]:
    if max_results <= 0 or len(candidates) <= max_results:
        return candidates

    remaining = [dict(item) for item in candidates]
    selected: List[Dict[str, object]] = []
    target = min(max_results, len(remaining))

    while remaining and len(selected) < target:
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(remaining):
            similarity_penalty = 0.0
            if selected:
                similarity_penalty = max(_plane_similarity(candidate, chosen) for chosen in selected)
            score = float(candidate.get("selection_score", 0.0)) - diversity_weight * similarity_penalty
            if score > best_score:
                best_score = score
                best_index = index
        chosen = remaining.pop(best_index)
        chosen["selection_score_after_diversity"] = best_score
        selected.append(chosen)

    return selected


def _priority_key(item: Dict[str, object]) -> tuple:
    tier = item.get("tier")
    if tier == "exact_match":
        return (0, -int(item["cardinality"]), item.get("direction_error") or 1e9, item["indices"])
    return (1, -int(item["cardinality"]), item.get("direction_error") or 1e9, item["indices"])


def classify_hard_mask(
    hard_mask: torch.Tensor,
    energy: GeometricHyperplaneEnergy,
    reference: Dict[str, object] | None = None,
) -> Dict[str, object] | None:
    validation = energy.validate_hard_mask(hard_mask)
    if not validation["valid"]:
        return None
    validation["validation_tier"] = "candidate_supporting_face"
    if reference is not None:
        validation.update(classify_discovered_plane(validation["normal"], validation["offset"], reference))
    else:
        validation.update(
            {
                "tier": "candidate_supporting_face",
                "matched_classes": [],
                "num_row_matches": 0,
                "recovered_integer_row": None,
                "canonical_integer_row": None,
                "canonical_key": None,
                "orbit_size": None,
                "orbit_unique_rows": None,
                "direction_error": None,
                "max_ratio_error": None,
            }
        )
    return validation


def analyze_candidate_probabilities(
    probabilities: torch.Tensor,
    energy: GeometricHyperplaneEnergy,
    reference: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    analyzed: List[Dict[str, object]] = []
    for row in probabilities:
        hard_mask = conditional_expectation_rounding(
            probabilities=row,
            energy=energy,
            min_cardinality=energy.config.min_cardinality,
        )
        validation = classify_hard_mask(hard_mask, energy=energy, reference=reference)
        if validation is not None:
            validation["hard_energy"] = energy.energy_of_hard_mask(hard_mask)
            validation["selection_score"] = _candidate_quality(validation)
            validation["plane_embedding"] = _plane_embedding(
                validation["normal"],
                validation["offset"],
            ).tolist()
            analyzed.append(validation)
    return analyzed


def cluster_unknown_faces(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in results:
        if item.get("tier") != "unknown_supporting_face":
            continue
        key = item.get("canonical_key")
        key = "null" if key is None else str(key)
        grouped[key].append(item)

    clusters: List[Dict[str, object]] = []
    for key, members in grouped.items():
        parsed_key = None if key == "null" else members[0].get("canonical_integer_row")
        clusters.append(
            {
                "canonical_integer_row": parsed_key,
                "count": len(members),
                "orbit_size": members[0].get("orbit_size"),
                "cardinalities": sorted([int(member["cardinality"]) for member in members], reverse=True),
                "best_direction_error": min(member.get("direction_error") or 1e9 for member in members),
                "best_max_ratio_error": min(member.get("max_ratio_error") or 1e9 for member in members),
                "example_indices": members[0]["indices"],
            }
        )
    clusters.sort(key=lambda item: (-item["count"], item["best_direction_error"]))
    return clusters


def sample_unique_facets(
    model: GeometricDiffUCOModel,
    diffusion: BernoulliDiffusionProcess,
    energy: GeometricHyperplaneEnergy,
    point_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_samples: int,
    reference: Dict[str, object] | None = None,
    max_results: int = 0,
    diversity_weight: float = 0.0,
) -> List[Dict[str, object]]:
    probabilities, _ = diffusion.rollout(
        model=model,
        energy=energy,
        point_features=point_features,
        edge_index=edge_index,
        edge_type=edge_type,
        batch_size=num_samples,
        stochastic=True,
    )

    analyzed = analyze_candidate_probabilities(probabilities, energy=energy, reference=reference)
    unique: Dict[tuple[int, ...], Dict[str, object]] = {}
    seen_classes: set[int] = set()
    seen_canonical_keys: set[str] = set()
    for validation in analyzed:
        matched_classes = validation.get("matched_classes", [])
        canonical_key = validation.get("canonical_key")
        validation["is_new_class"] = bool(matched_classes and any(class_id not in seen_classes for class_id in matched_classes))
        validation["is_new_canonical"] = bool(canonical_key and canonical_key not in seen_canonical_keys)
        seen_classes.update(matched_classes)
        if canonical_key is not None:
            seen_canonical_keys.add(str(canonical_key))
        key = tuple(validation["indices"])
        unique[key] = validation

    ordered = sorted(unique.values(), key=_priority_key)
    if diversity_weight > 0.0 and max_results > 0:
        return rerank_diverse_candidates(ordered, max_results=max_results, diversity_weight=diversity_weight)
    if max_results > 0:
        return ordered[:max_results]
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Geometric-DiffUCO inference for the 3-2-2 Bell polytope.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--num-samples", type=int, default=256, help="Parallel diffusion seeds.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--reference", type=Path, default=None, help="Optional path to known 3-2-2 facets_322.txt.")
    parser.add_argument("--filter-classes", type=int, nargs="*", default=[], help="Exact-match class ids to remove from final results.")
    parser.add_argument("--max-results", type=int, default=0, help="Optional limit on the number of returned candidates after ranking.")
    parser.add_argument("--diversity-weight", type=float, default=0.0, help="Greedy plane-space diversity penalty used when selecting a subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)

    model = GeometricDiffUCOModel(
        point_dim=points.shape[1],
        edge_types=len(graph.generator_names),
        hidden_dim=payload["model_config"]["hidden_dim"],
        layers=payload["model_config"]["layers"],
        dropout=payload["model_config"]["dropout"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    energy_kwargs = dict(payload["energy_config"])
    energy_kwargs.pop("size_reward", None)
    energy = GeometricHyperplaneEnergy(points=points, config=EnergyConfig(**energy_kwargs))
    diffusion = BernoulliDiffusionProcess(DiffusionConfig(**payload["diffusion_config"]), device=device)
    reference = build_reference_database(args.reference)

    raw_results = sample_unique_facets(
        model=model,
        diffusion=diffusion,
        energy=energy,
        point_features=points,
        edge_index=edge_index,
        edge_type=edge_type,
        num_samples=args.num_samples,
        reference=reference,
        max_results=args.max_results,
        diversity_weight=args.diversity_weight,
    )

    filtered_classes = set(args.filter_classes)
    filtered_out = [
        item for item in raw_results
        if item.get("tier") == "exact_match" and set(item.get("matched_classes", [])) & filtered_classes
    ]
    kept_results = [
        item for item in raw_results
        if not (item.get("tier") == "exact_match" and set(item.get("matched_classes", [])) & filtered_classes)
    ]

    unknown_clusters = cluster_unknown_faces(kept_results)
    canonical_counter = Counter(item.get("canonical_key") for item in kept_results if item.get("canonical_key") is not None)
    class_counter = Counter(class_id for item in kept_results for class_id in item.get("matched_classes", []))
    summary = {
        "raw_candidate_supporting_faces": len(raw_results),
        "filtered_candidate_supporting_faces": len(kept_results),
        "exact_matched_facets_after_filter": sum(1 for item in kept_results if item.get("tier") == "exact_match"),
        "matched_classes_after_filter": sorted({class_id for item in kept_results for class_id in item.get("matched_classes", [])}),
        "matched_class_hit_counter": dict(sorted(class_counter.items())),
        "canonical_orbit_count_after_filter": len(canonical_counter),
        "canonical_orbit_hit_counter": dict(sorted(canonical_counter.items())),
        "filtered_out_exact_matches": len(filtered_out),
        "filter_classes": sorted(filtered_classes),
        "unknown_cluster_count": len(unknown_clusters),
    }
    output = {
        "checkpoint": str(args.checkpoint),
        "summary": summary,
        "filtered_out": filtered_out,
        "unknown_clusters": unknown_clusters,
        "results": kept_results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
