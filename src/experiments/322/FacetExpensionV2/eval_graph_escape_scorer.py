from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    EXP322_DIR = CURRENT_DIR.parent
    DIFFUCO_DIR = EXP322_DIR / "DIffUCO"
    V1_DIR = EXP322_DIR / "FacetExpansionV1"
    sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR)]
    from energy import EnergyConfig, GeometricHyperplaneEnergy
    from geometry import generate_points_322
    from facet_reference import build_reference_database
    from seed_bank import load_seed_bank
    from symmetry_graph import build_symmetry_graph
    from graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer
    from train_graph_escape_scorer import build_escape_samples, split_samples, tensorize
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.facet_reference import build_reference_database
    from ..FacetExpansionV1.seed_bank import load_seed_bank
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer
    from .train_graph_escape_scorer import build_escape_samples, split_samples, tensorize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the graph-based 322 escape scorer.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-example-id", type=int, default=3)
    parser.add_argument("--negatives-per-source", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_cli_path(path: Path) -> Path:
    if os.name == "nt" and path.is_absolute() and str(path).startswith(("/home/", "/mnt/")):
        return Path("\\\\wsl$\\Ubuntu") / str(path).lstrip("/").replace("/", "\\")
    return path


def main() -> None:
    args = parse_args()
    args.checkpoint = normalize_cli_path(args.checkpoint)
    args.output = normalize_cli_path(args.output)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)
    energy = GeometricHyperplaneEnergy(points=points, config=EnergyConfig())
    reference = build_reference_database()
    seed_items = load_seed_bank(points=points.cpu())
    samples = build_escape_samples(
        seed_items,
        energy=energy,
        reference=reference,
        negatives_per_source=args.negatives_per_source,
        rng=__import__("random").Random(0),
    )
    _, val_samples = split_samples(samples, val_example_id=args.val_example_id)
    dataset = tensorize(val_samples, device=device)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    architecture = payload["model_config"].get("architecture", "graph")
    model_kwargs = dict(payload["model_config"])
    model_kwargs.pop("architecture", None)
    if architecture == "graph":
        model = GraphEscapeScorer(**model_kwargs).to(device)
    else:
        model = SetEscapeScorer(**model_kwargs).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    scores = []
    with torch.no_grad():
        for batch_index, (source_mask, candidate_mask, energy_value, labels, source_class, candidate_class) in enumerate(loader):
            logits = model(points, source_mask, candidate_mask, edge_index, edge_type, energy_value=energy_value)
            probs = torch.sigmoid(logits)
            for row_index in range(source_mask.shape[0]):
                sample = val_samples[batch_index * loader.batch_size + row_index]
                scores.append(
                    {
                        "source_index": sample.source_index,
                        "source_class_id": sample.source_class_id,
                        "source_example_id": sample.source_example_id,
                        "candidate_index": sample.candidate_index,
                        "candidate_class_id": sample.candidate_class_id,
                        "candidate_example_id": sample.candidate_example_id,
                        "candidate_type": sample.candidate_type,
                        "label": sample.label,
                        "score": float(probs[row_index].item()),
                    }
                )

    by_source = defaultdict(list)
    for row in scores:
        by_source[(row["source_index"], row["source_class_id"], row["source_example_id"])].append(row)

    ranking_stats = []
    source44_stats = []
    for key, rows in by_source.items():
        ordered = sorted(rows, key=lambda item: item["score"], reverse=True)
        positive_positions = [index for index, row in enumerate(ordered) if int(row["label"]) == 1]
        negative_positions = [index for index, row in enumerate(ordered) if int(row["label"]) == 0]
        top_positive_hit = int(bool(ordered and int(ordered[0]["label"]) == 1))
        source_record = {
            "source_index": key[0],
            "source_class_id": key[1],
            "source_example_id": key[2],
            "top1_is_escape": top_positive_hit,
            "top5_scores": ordered[:5],
            "best_positive_rank": min(positive_positions) if positive_positions else None,
            "best_negative_rank": min(negative_positions) if negative_positions else None,
        }
        ranking_stats.append(source_record)
        if int(key[1]) == 44:
            source44_stats.append(source_record)

    summary = {
        "num_val_samples": len(scores),
        "num_val_sources": len(ranking_stats),
        "top1_escape_rate": sum(item["top1_is_escape"] for item in ranking_stats) / max(len(ranking_stats), 1),
        "top1_escape_rate_source44": sum(item["top1_is_escape"] for item in source44_stats) / max(len(source44_stats), 1),
        "source44_examples": source44_stats,
    }
    output_payload = {
        "checkpoint": str(args.checkpoint),
        "summary": summary,
        "ranking_stats": ranking_stats[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
