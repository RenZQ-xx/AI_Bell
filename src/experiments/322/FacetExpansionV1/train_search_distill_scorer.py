from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_FEATURE_NAMES = [
    "round_index",
    "parent_depth",
    "history_length",
    "candidate_cardinality",
    "hard_energy",
    "novelty_score",
    "seed_jaccard",
    "archive_mean_absdiff",
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal candidate scorer from search-distillation JSONL.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--objective", type=str, choices=("mse", "pairwise"), default="mse")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--min-target-gap", type=float, default=0.25)
    parser.add_argument("--use-learned-embeddings", action="store_true")
    parser.add_argument("--rare-class-aware", action="store_true")
    parser.add_argument("--rare-class-power", type=float, default=0.5)
    parser.add_argument("--critical-pairs-only", action="store_true")
    parser.add_argument("--min-coverage-gap", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def target_return(record: dict[str, object]) -> float:
    if "future_total_return" in record:
        return float(record.get("future_total_return", 0.0))
    return (
        0.25 * float(record.get("teacher_selected", 0))
        + 2.5 * float(record.get("future_first_non44_hit", 0))
        + 2.0 * float(record.get("future_non44_hit", 0))
        + 2.0 * float(record.get("future_unique_class_gain", 0))
        + 1.5 * float(record.get("future_rare_class_gain", 0))
        + 1.5 * float(record.get("future_rare_class_count", 0))
        + 0.75 * float(record.get("future_new_class_count", 0))
        + 0.5 * float(record.get("future_new_canonical_count", 0))
        + 0.1 * float(record.get("future_survival_rounds", 0))
    )


def embedding_feature_names(record: dict[str, object]) -> list[str]:
    names: list[str] = []
    fields = [
        "history_embedding",
        "global_summary",
        "candidate_summary",
        "seed_summary",
        "archive_summary",
    ]
    for field in fields:
        values = record.get(field)
        if not isinstance(values, list):
            continue
        names.extend([f"{field}:{index}" for index in range(len(values))])
    for field in ("candidate_minus_seed", "candidate_minus_archive", "candidate_minus_history"):
        source = {
            "candidate_minus_seed": ("candidate_summary", "seed_summary"),
            "candidate_minus_archive": ("candidate_summary", "archive_summary"),
            "candidate_minus_history": ("candidate_summary", "history_embedding"),
        }[field]
        left = record.get(source[0])
        right = record.get(source[1])
        if isinstance(left, list) and isinstance(right, list):
            names.extend([f"{field}:{index}" for index in range(min(len(left), len(right)))])
    return names


def build_feature_names(records: list[dict[str, object]], use_learned_embeddings: bool) -> list[str]:
    names = list(BASE_FEATURE_NAMES)
    if use_learned_embeddings and records:
        names.extend(embedding_feature_names(records[0]))
    return names


def record_to_feature_row(record: dict[str, object], use_learned_embeddings: bool) -> list[float]:
    row = [float(record.get(name, 0.0)) for name in BASE_FEATURE_NAMES]
    if not use_learned_embeddings:
        return row

    history = record.get("history_embedding", [])
    global_summary = record.get("global_summary", [])
    candidate = record.get("candidate_summary", [])
    seed = record.get("seed_summary", [])
    archive = record.get("archive_summary", [])
    if isinstance(history, list):
        row.extend(float(value) for value in history)
    if isinstance(global_summary, list):
        row.extend(float(value) for value in global_summary)
    if isinstance(candidate, list):
        row.extend(float(value) for value in candidate)
    if isinstance(seed, list):
        row.extend(float(value) for value in seed)
    if isinstance(archive, list):
        row.extend(float(value) for value in archive)
    if isinstance(candidate, list) and isinstance(seed, list):
        row.extend(float(left) - float(right) for left, right in zip(candidate, seed))
    if isinstance(candidate, list) and isinstance(archive, list):
        row.extend(float(left) - float(right) for left, right in zip(candidate, archive))
    if isinstance(candidate, list) and isinstance(history, list):
        row.extend(float(left) - float(right) for left, right in zip(candidate, history))
    return row


def build_pairwise_dataset(
    normalized: torch.Tensor,
    targets: torch.Tensor,
    records: list[dict[str, object]],
    min_target_gap: float,
    critical_pairs_only: bool = False,
    min_coverage_gap: float = 1.0,
    rare_class_aware: bool = False,
    rare_class_power: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, record in enumerate(records):
        key = (
            int(record.get("seed_index", 0)),
            int(record.get("round_index", 0)),
            int(record.get("parent_branch_id", -1)),
        )
        groups.setdefault(key, []).append(index)

    class_counts: Counter[int] = Counter()
    if rare_class_aware:
        for record in records:
            if float(record.get("teacher_selected", 0)) > 0 or float(record.get("future_non44_hit", 0)) > 0:
                for class_id in record.get("matched_classes", []):
                    class_id = int(class_id)
                    if class_id != 44:
                        class_counts[class_id] += 1

    def rarity_bonus(record: dict[str, object]) -> float:
        if not rare_class_aware:
            return 1.0
        non44 = [int(class_id) for class_id in record.get("matched_classes", []) if int(class_id) != 44]
        if not non44 or not class_counts:
            return 1.0
        bonuses = [(1.0 / max(class_counts[class_id], 1)) ** rare_class_power for class_id in non44]
        return max(bonuses)

    def coverage_value(record: dict[str, object]) -> float:
        return (
            2.5 * float(record.get("future_first_non44_hit", 0))
            + 2.0 * float(record.get("future_non44_hit", 0))
            + 2.0 * float(record.get("future_unique_class_gain", 0))
            + 1.5 * float(record.get("future_rare_class_gain", 0))
            + 1.5 * float(record.get("future_rare_class_count", 0))
            + 0.75 * float(record.get("future_new_class_count", 0))
            + 0.5 * float(record.get("future_new_canonical_count", 0))
        )

    left_items: list[torch.Tensor] = []
    right_items: list[torch.Tensor] = []
    weights: list[float] = []
    for indices in groups.values():
        sorted_indices = sorted(indices, key=lambda item: float(targets[item].item()), reverse=True)
        for better_pos, better_index in enumerate(sorted_indices):
            better_target = float(targets[better_index].item())
            for worse_index in sorted_indices[better_pos + 1 :]:
                worse_target = float(targets[worse_index].item())
                gap = better_target - worse_target
                if gap < min_target_gap:
                    continue
                if critical_pairs_only:
                    better_record = records[better_index]
                    worse_record = records[worse_index]
                    better_coverage = coverage_value(better_record)
                    worse_coverage = coverage_value(worse_record)
                    coverage_gap = better_coverage - worse_coverage
                    exact_disagreement = bool(better_record.get("exact")) != bool(worse_record.get("exact"))
                    selected_disagreement = bool(better_record.get("teacher_selected")) != bool(
                        worse_record.get("teacher_selected")
                    )
                    if coverage_gap < min_coverage_gap and not exact_disagreement and not selected_disagreement:
                        continue
                left_items.append(normalized[better_index])
                right_items.append(normalized[worse_index])
                bonus = rarity_bonus(records[better_index])
                if critical_pairs_only:
                    bonus *= max(1.0, coverage_value(records[better_index]) - coverage_value(records[worse_index]))
                weights.append(gap * bonus)

    if not left_items:
        raise ValueError("No valid pairwise examples were constructed from the distillation dataset.")
    return torch.stack(left_items, dim=0), torch.stack(right_items, dim=0), torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {args.dataset}")

    feature_names = build_feature_names(records, use_learned_embeddings=args.use_learned_embeddings)
    features = torch.tensor(
        [record_to_feature_row(record, use_learned_embeddings=args.use_learned_embeddings) for record in records],
        dtype=torch.float32,
    )
    targets = torch.tensor([target_return(record) for record in records], dtype=torch.float32)
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0).clamp_min(1e-6)
    normalized = (features - feature_mean) / feature_std

    model = DistilledCandidateScorer(input_dim=len(feature_names), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.objective == "mse":
        dataset = TensorDataset(normalized, targets)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        loss_fn = nn.MSELoss()
        pair_count = 0
    else:
        left_features, right_features, pair_weights = build_pairwise_dataset(
            normalized=normalized,
            targets=targets,
            records=records,
            min_target_gap=args.min_target_gap,
            rare_class_aware=args.rare_class_aware,
            rare_class_power=args.rare_class_power,
            critical_pairs_only=args.critical_pairs_only,
            min_coverage_gap=args.min_coverage_gap,
        )
        dataset = TensorDataset(left_features, right_features, pair_weights)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        loss_fn = nn.MarginRankingLoss(margin=args.margin, reduction="none")
        pair_count = int(pair_weights.shape[0])

    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        if args.objective == "mse":
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch_features)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * batch_features.shape[0]
                total += batch_features.shape[0]
        else:
            target_direction = torch.ones(args.batch_size, device=device)
            for batch_left, batch_right, batch_weight in loader:
                batch_left = batch_left.to(device)
                batch_right = batch_right.to(device)
                batch_weight = batch_weight.to(device)
                optimizer.zero_grad(set_to_none=True)
                left_score = model(batch_left)
                right_score = model(batch_right)
                direction = target_direction[: batch_left.shape[0]]
                raw_loss = loss_fn(left_score, right_score, direction)
                weight_scale = batch_weight / batch_weight.mean().clamp_min(1e-6)
                loss = (raw_loss * weight_scale).mean()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * batch_left.shape[0]
                total += batch_left.shape[0]

        epoch_loss = running_loss / max(total, 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(json.dumps(history[-1], ensure_ascii=False))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                "model_state": model.state_dict(),
                "feature_names": feature_names,
                "feature_mean": feature_mean.tolist(),
                "feature_std": feature_std.tolist(),
                "best_loss": best_loss,
                "hidden_dim": args.hidden_dim,
                "objective": args.objective,
                "margin": args.margin,
                "min_target_gap": args.min_target_gap,
                "use_learned_embeddings": args.use_learned_embeddings,
                "rare_class_aware": args.rare_class_aware,
                "rare_class_power": args.rare_class_power,
                "critical_pairs_only": args.critical_pairs_only,
                "min_coverage_gap": args.min_coverage_gap,
                "pair_count": pair_count,
                "training_history": history,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, args.output)
        args.output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "num_records": len(records),
                    "best_loss": best_loss,
                    "feature_names": feature_names,
                    "hidden_dim": args.hidden_dim,
                    "objective": args.objective,
                    "margin": args.margin,
                    "min_target_gap": args.min_target_gap,
                    "use_learned_embeddings": args.use_learned_embeddings,
                    "rare_class_aware": args.rare_class_aware,
                    "rare_class_power": args.rare_class_power,
                    "critical_pairs_only": args.critical_pairs_only,
                    "min_coverage_gap": args.min_coverage_gap,
                    "pair_count": pair_count,
                    "training_history": history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
