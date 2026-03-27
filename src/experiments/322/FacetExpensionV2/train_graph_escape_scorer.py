from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    EXP322_DIR = CURRENT_DIR.parent
    DIFFUCO_DIR = EXP322_DIR / "DIffUCO"
    V1_DIR = EXP322_DIR / "FacetExpansionV1"
    sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR)]
    from energy import EnergyConfig, GeometricHyperplaneEnergy
    from geometry import generate_points_322
    from inference import classify_hard_mask
    from facet_reference import build_reference_database
    from seed_bank import load_seed_bank
    from symmetry_graph import build_symmetry_graph
    from graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.facet_reference import build_reference_database
    from ..FacetExpansionV1.seed_bank import load_seed_bank
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from .graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer


@dataclass
class EscapeSample:
    source_index: int
    source_class_id: int
    source_example_id: int
    candidate_index: int
    candidate_class_id: int | None
    candidate_example_id: int | None
    label: int
    candidate_type: str
    source_mask: list[int]
    candidate_mask: list[int]
    energy_value: float
    exact: bool
    hamming_distance: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a graph-based escape scorer on 322 candidate transitions.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--architecture", type=str, choices=["graph", "set"], default="set")
    parser.add_argument("--val-example-id", type=int, default=3)
    parser.add_argument("--negatives-per-source", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_cli_path(path: Path) -> Path:
    if os.name == "nt" and path.is_absolute() and str(path).startswith(("/home/", "/mnt/")):
        return Path("\\\\wsl$\\Ubuntu") / str(path).lstrip("/").replace("/", "\\")
    return path


def sample_invalid_candidate(source_mask: list[int], rng: random.Random, flip_count: int) -> list[int]:
    indices = list(range(len(source_mask)))
    chosen = rng.sample(indices, min(flip_count, len(indices)))
    candidate = list(source_mask)
    for index in chosen:
        candidate[index] = 1 - candidate[index]
    return candidate


def build_escape_samples(seed_items, energy: GeometricHyperplaneEnergy, reference, negatives_per_source: int, rng: random.Random) -> list[EscapeSample]:
    samples: list[EscapeSample] = []
    exact_seed_records = []
    for candidate_index, candidate_item in enumerate(seed_items):
        hard = torch.tensor(candidate_item.mask, dtype=torch.float32)
        exact_seed_records.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_class_id": int(candidate_item.class_id),
                "candidate_example_id": int(candidate_item.example_id),
                "candidate_mask": list(candidate_item.mask),
                "energy_value": float(energy.energy_of_hard_mask(hard >= 0.5)),
            }
        )
    for source_index, source_item in enumerate(seed_items):
        source_mask = list(source_item.mask)
        # Positive exact escapes: known exact masks from other classes.
        for candidate in exact_seed_records:
            label = int(candidate["candidate_class_id"] != source_item.class_id)
            samples.append(
                EscapeSample(
                    source_index=int(source_index),
                    source_class_id=int(source_item.class_id),
                    source_example_id=int(source_item.example_id),
                    candidate_index=int(candidate["candidate_index"]),
                    candidate_class_id=int(candidate["candidate_class_id"]),
                    candidate_example_id=int(candidate["candidate_example_id"]),
                    label=label,
                    candidate_type="exact_seed",
                    source_mask=source_mask,
                    candidate_mask=list(candidate["candidate_mask"]),
                    energy_value=float(candidate["energy_value"]),
                    exact=True,
                    hamming_distance=sum(int(left != right) for left, right in zip(source_mask, candidate["candidate_mask"])),
                )
            )

        # Hard negatives: perturbed infeasible/high-energy candidates.
        added = 0
        trials = 0
        while added < negatives_per_source and trials < negatives_per_source * 20:
            trials += 1
            flip_count = rng.choice([2, 4, 6, 8, 10, 12])
            candidate_mask = sample_invalid_candidate(source_mask, rng=rng, flip_count=flip_count)
            hard = torch.tensor(candidate_mask, dtype=torch.float32)
            validation = classify_hard_mask(hard >= 0.5, energy=energy, reference=reference)
            if validation is not None:
                continue
            samples.append(
                EscapeSample(
                    source_index=int(source_index),
                    source_class_id=int(source_item.class_id),
                    source_example_id=int(source_item.example_id),
                    candidate_index=-1,
                    candidate_class_id=None,
                    candidate_example_id=None,
                    label=0,
                    candidate_type="invalid_flip",
                    source_mask=source_mask,
                    candidate_mask=candidate_mask,
                    energy_value=float(energy.energy_of_hard_mask(hard >= 0.5)),
                    exact=False,
                    hamming_distance=sum(int(left != right) for left, right in zip(source_mask, candidate_mask)),
                )
            )
            added += 1
    return samples


def split_samples(samples: list[EscapeSample], val_example_id: int) -> tuple[list[EscapeSample], list[EscapeSample]]:
    train = []
    val = []
    for sample in samples:
        candidate_example_id = -1 if sample.candidate_example_id is None else sample.candidate_example_id
        if sample.source_example_id == val_example_id or candidate_example_id == val_example_id:
            val.append(sample)
        else:
            train.append(sample)
    return train, val


def tensorize(samples: list[EscapeSample], device: torch.device) -> TensorDataset:
    source_mask = torch.tensor([sample.source_mask for sample in samples], dtype=torch.float32, device=device)
    candidate_mask = torch.tensor([sample.candidate_mask for sample in samples], dtype=torch.float32, device=device)
    energy_value = torch.tensor([sample.energy_value for sample in samples], dtype=torch.float32, device=device)
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.float32, device=device)
    source_class = torch.tensor([sample.source_class_id for sample in samples], dtype=torch.long, device=device)
    candidate_class = torch.tensor([sample.candidate_class_id or -1 for sample in samples], dtype=torch.long, device=device)
    return TensorDataset(source_mask, candidate_mask, energy_value, labels, source_class, candidate_class)


def collect_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    accuracy = float((preds == labels).float().mean().item())
    positive_mask = labels >= 0.5
    negative_mask = labels < 0.5
    pos_count = int(positive_mask.sum().item())
    neg_count = int(negative_mask.sum().item())
    pos_acc = float((preds[positive_mask] == labels[positive_mask]).float().mean().item()) if pos_count else 0.0
    neg_acc = float((preds[negative_mask] == labels[negative_mask]).float().mean().item()) if neg_count else 0.0
    return {
        "accuracy": accuracy,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "mean_positive_score": float(probs[positive_mask].mean().item()) if pos_count else 0.0,
        "mean_negative_score": float(probs[negative_mask].mean().item()) if neg_count else 0.0,
    }


def evaluate(model, dataset: TensorDataset, point_features: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, batch_size: int) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for source_mask, candidate_mask, energy_value, labels, _, _ in loader:
            logits = model(point_features, source_mask, candidate_mask, edge_index, edge_type, energy_value=energy_value)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total += source_mask.shape[0]
            total_loss += float(loss.item()) * source_mask.shape[0]
            all_logits.append(logits)
            all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = collect_metrics(logits, labels)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir = normalize_cli_path(args.output_dir)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device(args.device)

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)
    energy = GeometricHyperplaneEnergy(points=points, config=EnergyConfig())
    reference = build_reference_database()
    seed_items = load_seed_bank(points=points.cpu())

    samples = build_escape_samples(seed_items, energy=energy, reference=reference, negatives_per_source=args.negatives_per_source, rng=random.Random(args.random_seed))
    train_samples, val_samples = split_samples(samples, val_example_id=args.val_example_id)
    train_dataset = tensorize(train_samples, device=device)
    val_dataset = tensorize(val_samples, device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.architecture == "graph":
        model = GraphEscapeScorer(
            point_dim=points.shape[1],
            edge_types=len(graph.generator_names),
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
        model_config = {
            "architecture": args.architecture,
            "point_dim": int(points.shape[1]),
            "edge_types": len(graph.generator_names),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
        }
    else:
        model = SetEscapeScorer(
            point_dim=points.shape[1],
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
        model_config = {
            "architecture": args.architecture,
            "point_dim": int(points.shape[1]),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_state = None
    best_metric = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for source_mask, candidate_mask, energy_value, labels, _, _ in train_loader:
            logits = model(points, source_mask, candidate_mask, edge_index, edge_type, energy_value=energy_value)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        should_eval = epoch == 1 or epoch == args.epochs or (args.eval_every > 0 and epoch % args.eval_every == 0)
        if should_eval:
            train_metrics = evaluate(model, train_dataset, points, edge_index, edge_type, batch_size=args.batch_size)
            val_metrics = evaluate(model, val_dataset, points, edge_index, edge_type, batch_size=args.batch_size)
        else:
            train_metrics = {"loss": float("nan"), "accuracy": float("nan"), "positive_accuracy": float("nan"), "negative_accuracy": float("nan")}
            val_metrics = {
                "loss": float("nan"),
                "accuracy": float("nan"),
                "positive_accuracy": float("nan"),
                "negative_accuracy": float("nan"),
                "mean_positive_score": float("nan"),
                "mean_negative_score": float("nan"),
            }
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_positive_accuracy": train_metrics["positive_accuracy"],
            "train_negative_accuracy": train_metrics["negative_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_positive_accuracy": val_metrics["positive_accuracy"],
            "val_negative_accuracy": val_metrics["negative_accuracy"],
            "val_mean_positive_score": val_metrics["mean_positive_score"],
            "val_mean_negative_score": val_metrics["mean_negative_score"],
        }
        history.append(record)
        metric = val_metrics["accuracy"] + 0.25 * val_metrics["positive_accuracy"] + 0.25 * val_metrics["negative_accuracy"] if should_eval else float("-inf")
        if should_eval and metric > best_metric:
            best_metric = metric
            best_state = {
                "model_state": model.state_dict(),
                "model_config": model_config,
                "train_config": vars(args),
                "best_epoch": epoch,
                "best_metric": best_metric,
            }
        if should_eval:
            print(json.dumps(record, ensure_ascii=False))

    assert best_state is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "graph_escape_scorer.pt"
    torch.save(best_state, checkpoint_path)
    summary = {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "best_epoch": best_state["best_epoch"],
        "best_metric": best_state["best_metric"],
        "history_tail": history[-10:],
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "sample_preview.json").write_text(
        json.dumps(
            {
                "train_preview": [asdict(sample) for sample in train_samples[:20]],
                "val_preview": [asdict(sample) for sample in val_samples[:20]],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
