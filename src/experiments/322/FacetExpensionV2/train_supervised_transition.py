from __future__ import annotations

import argparse
import json
import random
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
    from geometry import generate_points_322
    from seed_bank import load_seed_bank
    from supervised_transition import ClassConditionedTransitionModel, TransitionSample, build_class_index_map
else:
    from ..DIffUCO.geometry import generate_points_322
    from ..FacetExpansionV1.seed_bank import load_seed_bank
    from .supervised_transition import ClassConditionedTransitionModel, TransitionSample, build_class_index_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised class-conditioned transition model on the 322 seed bank.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--class-embed-dim", type=int, default=32)
    parser.add_argument("--val-example-id", type=int, default=3)
    parser.add_argument("--full-fit", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def nearest_target_index(source_mask: list[int], candidate_masks: list[list[int]]) -> tuple[int, int]:
    best_index = 0
    best_hamming = None
    for index, mask in enumerate(candidate_masks):
        hamming = sum(int(left != right) for left, right in zip(source_mask, mask))
        if best_hamming is None or hamming < best_hamming:
            best_hamming = hamming
            best_index = index
    return best_index, int(best_hamming if best_hamming is not None else 0)


def build_samples(seed_items) -> list[TransitionSample]:
    by_class: dict[int, list[tuple[int, object]]] = {}
    for index, item in enumerate(seed_items):
        by_class.setdefault(int(item.class_id), []).append((index, item))

    samples: list[TransitionSample] = []
    for source_index, source_item in enumerate(seed_items):
        for target_class_id, candidates in by_class.items():
            target_candidate_masks = [candidate.mask for _, candidate in candidates]
            candidate_index, hamming = nearest_target_index(source_item.mask, target_candidate_masks)
            target_index, target_item = candidates[candidate_index]
            samples.append(
                TransitionSample(
                    source_index=int(source_index),
                    source_class_id=int(source_item.class_id),
                    source_example_id=int(source_item.example_id),
                    target_class_id=int(target_class_id),
                    target_index=int(target_index),
                    target_example_id=int(target_item.example_id),
                    source_mask=list(source_item.mask),
                    target_mask=list(target_item.mask),
                    hamming_to_target=int(hamming),
                )
            )
    return samples


def split_samples(samples: list[TransitionSample], val_example_id: int, full_fit: bool) -> tuple[list[TransitionSample], list[TransitionSample]]:
    if full_fit:
        return samples, samples
    train_samples = []
    val_samples = []
    for sample in samples:
        if sample.source_example_id == val_example_id or sample.target_example_id == val_example_id:
            val_samples.append(sample)
        else:
            train_samples.append(sample)
    return train_samples, val_samples


def tensorize(samples: list[TransitionSample], class_index: dict[int, int], device: torch.device) -> TensorDataset:
    source_mask = torch.tensor([sample.source_mask for sample in samples], dtype=torch.float32, device=device)
    target_mask = torch.tensor([sample.target_mask for sample in samples], dtype=torch.float32, device=device)
    source_class = torch.tensor([class_index[sample.source_class_id] for sample in samples], dtype=torch.long, device=device)
    target_class = torch.tensor([class_index[sample.target_class_id] for sample in samples], dtype=torch.long, device=device)
    target_cardinality = target_mask.sum(dim=-1)
    return TensorDataset(source_mask, source_class, target_class, target_mask, target_cardinality)


def evaluate(model, dataset: TensorDataset, batch_size: int) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    total_loss = 0.0
    exact_mask = 0
    exact_bits = 0
    total_bits = 0
    with torch.no_grad():
        for source_mask, source_class, target_class, target_mask, target_cardinality in loader:
            logits, predicted_cardinality = model(source_mask, source_class, target_class)
            loss = F.binary_cross_entropy_with_logits(logits, target_mask) + 0.05 * F.l1_loss(predicted_cardinality, target_cardinality)
            total += source_mask.shape[0]
            total_loss += float(loss.item()) * source_mask.shape[0]
            hard = (torch.sigmoid(logits) >= 0.5).float()
            exact_mask += int((hard == target_mask).all(dim=-1).sum().item())
            exact_bits += int((hard == target_mask).sum().item())
            total_bits += int(target_mask.numel())
    return {
        "loss": total_loss / max(total, 1),
        "exact_mask_rate": exact_mask / max(total, 1),
        "bit_accuracy": exact_bits / max(total_bits, 1),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device(args.device)

    points = generate_points_322()
    seed_items = load_seed_bank(points=points)
    samples = build_samples(seed_items)
    class_index = build_class_index_map(sample.source_class_id for sample in samples)
    train_samples, val_samples = split_samples(samples, val_example_id=args.val_example_id, full_fit=args.full_fit)
    train_dataset = tensorize(train_samples, class_index=class_index, device=device)
    val_dataset = tensorize(val_samples, class_index=class_index, device=device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = ClassConditionedTransitionModel(
        num_classes=len(class_index),
        mask_dim=len(seed_items[0].mask),
        class_embed_dim=args.class_embed_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_metric = float("-inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        total_loss = 0.0
        for source_mask, source_class, target_class, target_mask, target_cardinality in train_loader:
            logits, predicted_cardinality = model(source_mask, source_class, target_class)
            loss = F.binary_cross_entropy_with_logits(logits, target_mask) + 0.05 * F.l1_loss(predicted_cardinality, target_cardinality)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += source_mask.shape[0]
            total_loss += float(loss.item()) * source_mask.shape[0]

        train_metrics = evaluate(model, train_dataset, batch_size=args.batch_size)
        val_metrics = evaluate(model, val_dataset, batch_size=args.batch_size)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_exact_mask_rate": train_metrics["exact_mask_rate"],
            "train_bit_accuracy": train_metrics["bit_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_exact_mask_rate": val_metrics["exact_mask_rate"],
            "val_bit_accuracy": val_metrics["bit_accuracy"],
        }
        history.append(record)
        metric = val_metrics["exact_mask_rate"] + 0.1 * val_metrics["bit_accuracy"]
        if metric > best_metric:
            best_metric = metric
            best_state = {
                "model_state": model.state_dict(),
                "model_config": {
                    "num_classes": len(class_index),
                    "mask_dim": len(seed_items[0].mask),
                    "class_embed_dim": args.class_embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
                "class_index": class_index,
                "train_config": vars(args),
                "best_epoch": epoch,
                "best_metric": best_metric,
            }
        if epoch % 25 == 0 or epoch == args.epochs:
            print(json.dumps(record, ensure_ascii=False))

    assert best_state is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "transition_model.pt"
    torch.save(best_state, checkpoint_path)
    summary = {
        "checkpoint": str(checkpoint_path),
        "num_samples": len(samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "split": "full_fit" if args.full_fit else f"holdout_example_{args.val_example_id}",
        "best_epoch": best_state["best_epoch"],
        "best_metric": best_state["best_metric"],
        "final_train": history[-1]["train_exact_mask_rate"],
        "final_val": history[-1]["val_exact_mask_rate"],
        "history_tail": history[-10:],
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
