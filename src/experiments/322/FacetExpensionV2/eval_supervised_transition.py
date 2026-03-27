from __future__ import annotations

import argparse
import json
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
    from seed_bank import load_seed_bank
    from supervised_transition import ClassConditionedTransitionModel, TransitionSample
    from train_supervised_transition import build_samples, split_samples
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..FacetExpansionV1.seed_bank import load_seed_bank
    from .supervised_transition import ClassConditionedTransitionModel, TransitionSample
    from .train_supervised_transition import build_samples, split_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the supervised class-conditioned transition model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-example-id", type=int, default=3)
    parser.add_argument("--full-fit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate_samples(
    model: ClassConditionedTransitionModel,
    samples: list[TransitionSample],
    class_index: dict[int, int],
    device: torch.device,
    energy,
    reference,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    exact_mask = 0
    exact_class = 0
    exact_non_target = 0
    for sample in samples:
        source_mask = torch.tensor(sample.source_mask, dtype=torch.float32, device=device).unsqueeze(0)
        source_class = torch.tensor([class_index[sample.source_class_id]], dtype=torch.long, device=device)
        target_class = torch.tensor([class_index[sample.target_class_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(source_mask, source_class, target_class)
        hard = (torch.sigmoid(logits).squeeze(0) >= 0.5).float().cpu()
        validation = classify_hard_mask(hard >= 0.5, energy=energy, reference=reference)
        matched_classes = [] if validation is None else [int(class_id) for class_id in validation.get("matched_classes", [])]
        mask_match = bool(torch.equal(hard, torch.tensor(sample.target_mask, dtype=torch.float32)))
        class_match = int(sample.target_class_id) in matched_classes
        exact_mask += int(mask_match)
        exact_class += int(class_match)
        exact_non_target += int(validation is not None and not class_match)
        results.append(
            {
                "source_index": int(sample.source_index),
                "source_class_id": int(sample.source_class_id),
                "source_example_id": int(sample.source_example_id),
                "target_class_id": int(sample.target_class_id),
                "target_index": int(sample.target_index),
                "target_example_id": int(sample.target_example_id),
                "mask_match": mask_match,
                "matched_classes": matched_classes,
                "class_match": class_match,
                "is_exact": validation is not None,
            }
        )
    summary = {
        "num_samples": len(samples),
        "exact_mask_rate": exact_mask / max(len(samples), 1),
        "target_class_hit_rate": exact_class / max(len(samples), 1),
        "exact_but_wrong_class_rate": exact_non_target / max(len(samples), 1),
    }
    return summary, results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ClassConditionedTransitionModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    points = generate_points_322()
    seed_items = load_seed_bank(points=points)
    samples = build_samples(seed_items)
    train_samples, val_samples = split_samples(samples, val_example_id=args.val_example_id, full_fit=args.full_fit)
    energy = GeometricHyperplaneEnergy(points=points.to(device), config=EnergyConfig())
    reference = build_reference_database()

    train_summary, _ = evaluate_samples(model, train_samples, payload["class_index"], device=device, energy=energy, reference=reference)
    val_summary, val_results = evaluate_samples(model, val_samples, payload["class_index"], device=device, energy=energy, reference=reference)

    output_payload = {
        "checkpoint": str(args.checkpoint),
        "split": "full_fit" if args.full_fit else f"holdout_example_{args.val_example_id}",
        "train_summary": train_summary,
        "val_summary": val_summary,
        "val_examples": val_results[:200],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
