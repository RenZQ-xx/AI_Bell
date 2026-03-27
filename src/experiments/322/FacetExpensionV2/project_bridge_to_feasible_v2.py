from __future__ import annotations

import argparse
import json
import os
import pathlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    EXP322_DIR = CURRENT_DIR.parent
    DIFFUCO_DIR = EXP322_DIR / "DIffUCO"
    V1_DIR = EXP322_DIR / "FacetExpansionV1"
    sys.path[:0] = [str(DIFFUCO_DIR), str(V1_DIR), str(CURRENT_DIR)]
    from energy import EnergyConfig, GeometricHyperplaneEnergy
    from facet_reference import build_reference_database
    from geometry import generate_points_322
    from inference import classify_hard_mask
    from seed_bank import load_seed_bank, seed_tensor
    from symmetry_graph import build_symmetry_graph
    from graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer
else:
    from ..DIffUCO.energy import EnergyConfig, GeometricHyperplaneEnergy
    from ..DIffUCO.facet_reference import build_reference_database
    from ..DIffUCO.geometry import generate_points_322
    from ..DIffUCO.inference import classify_hard_mask
    from ..DIffUCO.symmetry_graph import build_symmetry_graph
    from ..FacetExpansionV1.seed_bank import load_seed_bank, seed_tensor
    from .graph_escape_scorer import GraphEscapeScorer, SetEscapeScorer


def normalize_cli_path(path: Path) -> Path:
    if os.name == "nt" and path.is_absolute() and str(path).startswith(("/home/", "/mnt/")):
        return Path("\\\\wsl$\\Ubuntu") / str(path).lstrip("/").replace("/", "\\")
    return path


class EscapeScorerWrapper(nn.Module):
    def __init__(self, checkpoint: Path, device: torch.device):
        super().__init__()
        original_windows_path = pathlib.WindowsPath
        try:
            if os.name != "nt":
                pathlib.WindowsPath = pathlib.PosixPath
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
        finally:
            pathlib.WindowsPath = original_windows_path
        architecture = payload["model_config"].get("architecture", "set")
        model_kwargs = dict(payload["model_config"])
        model_kwargs.pop("architecture", None)
        if architecture == "graph":
            model = GraphEscapeScorer(**model_kwargs)
        else:
            model = SetEscapeScorer(**model_kwargs)
        model.load_state_dict(payload["model_state"])
        self.model = model.to(device)
        self.model.eval()

    def score(
        self,
        point_features: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        energy_value: float,
    ) -> float:
        with torch.no_grad():
            logits = self.model(
                point_features,
                source_mask.unsqueeze(0).float(),
                candidate_mask.unsqueeze(0).float(),
                edge_index,
                edge_type,
                energy_value=torch.tensor([energy_value], dtype=torch.float32, device=point_features.device),
            )
        return float(torch.sigmoid(logits).item())


@dataclass
class ProtectionProfile:
    remove_scores: torch.Tensor
    add_scores: torch.Tensor


@dataclass
class SearchState:
    mask: torch.Tensor
    step: int
    objective: float
    energy_value: float
    escape_score: float
    similarity_to_source: float
    support_size: int
    exact: bool
    matched_classes: list[int]
    canonical_key: str | None
    retention_ratio: float
    retained_mass: float
    reverted_mass: float
    exact44_penalty: float
    action: dict[str, object]
    parent_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anti-collapse bridge projection with protected edits.")
    parser.add_argument("--jump-bank", type=Path, required=True)
    parser.add_argument("--escape-scorer", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-class-id", type=int, default=44)
    parser.add_argument("--source-seed-indices", type=str, default="")
    parser.add_argument("--proposals-per-source", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--top-remove", type=int, default=3)
    parser.add_argument("--top-add", type=int, default=3)
    parser.add_argument("--top-swap", type=int, default=4)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--escape-weight", type=float, default=4.0)
    parser.add_argument("--source-similarity-weight", type=float, default=2.5)
    parser.add_argument("--support-drop-weight", type=float, default=1.5)
    parser.add_argument("--retention-weight", type=float, default=6.0)
    parser.add_argument("--revert-weight", type=float, default=4.0)
    parser.add_argument("--exact-bonus", type=float, default=1.0)
    parser.add_argument("--exact44-penalty", type=float, default=8.0)
    parser.add_argument("--min-retention-ratio", type=float, default=0.55)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_source_indices(spec: str) -> list[int]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def mask_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_mask = left >= 0.5
    right_mask = right >= 0.5
    intersection = int((left_mask & right_mask).sum().item())
    union = int((left_mask | right_mask).sum().item())
    if union == 0:
        return 1.0
    return float(intersection / union)


def build_protection_profile(seed_items, seed_masks: torch.Tensor, source_seed_index: int, source_class_id: int) -> ProtectionProfile:
    source_mask = seed_masks[source_seed_index].detach().cpu()
    source_bool = source_mask >= 0.5
    remove_scores = torch.zeros_like(source_mask)
    add_scores = torch.zeros_like(source_mask)

    for candidate_index, item in enumerate(seed_items):
        if int(item.class_id) == int(source_class_id):
            continue
        candidate_mask = seed_masks[candidate_index].detach().cpu()
        candidate_bool = candidate_mask >= 0.5
        remove_scores = remove_scores + (source_bool & (~candidate_bool)).float()
        add_scores = add_scores + ((~source_bool) & candidate_bool).float()

    remove_total = float(remove_scores.sum().item())
    add_total = float(add_scores.sum().item())
    if remove_total > 0.0:
        remove_scores = remove_scores / remove_total
    if add_total > 0.0:
        add_scores = add_scores / add_total
    return ProtectionProfile(remove_scores=remove_scores, add_scores=add_scores)


def bridge_protection_total(profile: ProtectionProfile, source_mask: torch.Tensor, bridge_mask: torch.Tensor) -> float:
    source_bool = source_mask >= 0.5
    bridge_bool = bridge_mask >= 0.5
    removed_mass = float((profile.remove_scores * (source_bool & (~bridge_bool)).float()).sum().item())
    added_mass = float((profile.add_scores * ((~source_bool) & bridge_bool).float()).sum().item())
    return removed_mass + added_mass


def compute_retention(profile: ProtectionProfile, source_mask: torch.Tensor, candidate_mask: torch.Tensor, bridge_total_mass: float) -> tuple[float, float, float]:
    source_bool = source_mask >= 0.5
    candidate_bool = candidate_mask >= 0.5
    retained_removed = float((profile.remove_scores * (source_bool & (~candidate_bool)).float()).sum().item())
    retained_added = float((profile.add_scores * ((~source_bool) & candidate_bool).float()).sum().item())
    retained_mass = retained_removed + retained_added
    if bridge_total_mass <= 1e-8:
        return 1.0, retained_mass, 0.0
    retention_ratio = retained_mass / bridge_total_mass
    reverted_mass = max(bridge_total_mass - retained_mass, 0.0)
    return float(retention_ratio), float(retained_mass), float(reverted_mass)


def evaluate_state(
    *,
    candidate_mask: torch.Tensor,
    source_mask: torch.Tensor,
    source_support_size: int,
    source_class_id: int,
    profile: ProtectionProfile,
    bridge_total_mass: float,
    energy: GeometricHyperplaneEnergy,
    reference,
    scorer: EscapeScorerWrapper,
    points: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    args: argparse.Namespace,
    step: int,
    action: dict[str, object],
    parent_index: int | None,
) -> SearchState:
    hard = (candidate_mask >= 0.5).float().detach().cpu()
    validation = classify_hard_mask(hard >= 0.5, energy=energy, reference=reference)
    energy_value = float(energy.energy_of_hard_mask(hard >= 0.5))
    escape_score = scorer.score(
        points,
        source_mask.to(points.device),
        hard.to(points.device),
        edge_index,
        edge_type,
        energy_value=energy_value,
    )
    similarity_to_source = mask_similarity(source_mask, hard)
    support_size = int((hard >= 0.5).sum().item())
    support_drop = max(source_support_size - support_size, 0) / max(source_support_size, 1)
    exact = validation is not None
    matched_classes = [] if validation is None else [int(class_id) for class_id in validation.get("matched_classes", [])]
    retention_ratio, retained_mass, reverted_mass = compute_retention(profile, source_mask, hard, bridge_total_mass)
    exact_bonus = args.exact_bonus if exact else 0.0
    exact44_penalty = args.exact44_penalty if exact and matched_classes == [int(source_class_id)] else 0.0
    objective = (
        -args.energy_weight * energy_value
        + args.escape_weight * escape_score
        - args.source_similarity_weight * similarity_to_source
        + args.support_drop_weight * support_drop
        + args.retention_weight * retention_ratio
        - args.revert_weight * reverted_mass
        + exact_bonus
        - exact44_penalty
    )
    return SearchState(
        mask=hard,
        step=step,
        objective=float(objective),
        energy_value=energy_value,
        escape_score=float(escape_score),
        similarity_to_source=float(similarity_to_source),
        support_size=support_size,
        exact=exact,
        matched_classes=matched_classes,
        canonical_key=None if validation is None else validation.get("canonical_key"),
        retention_ratio=float(retention_ratio),
        retained_mass=float(retained_mass),
        reverted_mass=float(reverted_mass),
        exact44_penalty=float(exact44_penalty),
        action=action,
        parent_index=parent_index,
    )


def propose_neighbors(
    state: SearchState,
    *,
    source_mask: torch.Tensor,
    source_support_size: int,
    source_class_id: int,
    profile: ProtectionProfile,
    bridge_total_mass: float,
    energy: GeometricHyperplaneEnergy,
    reference,
    scorer: EscapeScorerWrapper,
    points: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    args: argparse.Namespace,
    step: int,
    parent_index: int,
) -> list[SearchState]:
    mask = state.mask
    active = [int(index) for index in torch.nonzero(mask >= 0.5, as_tuple=False).flatten().tolist()]
    inactive = [int(index) for index in torch.nonzero(mask < 0.5, as_tuple=False).flatten().tolist()]

    def maybe_append(candidate_mask: torch.Tensor, action: dict[str, object], bucket: list[tuple[float, SearchState]]) -> None:
        evaluated = evaluate_state(
            candidate_mask=candidate_mask,
            source_mask=source_mask,
            source_support_size=source_support_size,
            source_class_id=source_class_id,
            profile=profile,
            bridge_total_mass=bridge_total_mass,
            energy=energy,
            reference=reference,
            scorer=scorer,
            points=points,
            edge_index=edge_index,
            edge_type=edge_type,
            args=args,
            step=step,
            action=action,
            parent_index=parent_index,
        )
        if evaluated.retention_ratio + 1e-8 < args.min_retention_ratio:
            return
        bucket.append((evaluated.objective, evaluated))

    remove_candidates: list[tuple[float, SearchState]] = []
    for index in active:
        candidate = mask.clone()
        candidate[index] = 0.0
        maybe_append(candidate, {"type": "remove", "index": index}, remove_candidates)
    remove_candidates.sort(key=lambda item: item[0], reverse=True)
    best_remove = [item[1] for item in remove_candidates[: args.top_remove]]

    add_candidates: list[tuple[float, SearchState]] = []
    for index in inactive:
        candidate = mask.clone()
        candidate[index] = 1.0
        maybe_append(candidate, {"type": "add", "index": index}, add_candidates)
    add_candidates.sort(key=lambda item: item[0], reverse=True)
    best_add = [item[1] for item in add_candidates[: args.top_add]]

    swap_candidates: list[tuple[float, SearchState]] = []
    remove_pool = [item.action["index"] for item in best_remove]
    add_pool = [item.action["index"] for item in best_add]
    for remove_index in remove_pool:
        for add_index in add_pool:
            candidate = mask.clone()
            candidate[remove_index] = 0.0
            candidate[add_index] = 1.0
            maybe_append(
                candidate,
                {"type": "swap", "remove_index": remove_index, "add_index": add_index},
                swap_candidates,
            )
    swap_candidates.sort(key=lambda item: item[0], reverse=True)
    best_swap = [item[1] for item in swap_candidates[: args.top_swap]]

    combined = best_remove + best_add + best_swap
    deduped: list[SearchState] = []
    seen_masks: set[tuple[int, ...]] = set()
    for candidate in combined:
        key = tuple(int(value) for value in candidate.mask.tolist())
        if key in seen_masks:
            continue
        seen_masks.add(key)
        deduped.append(candidate)
    return deduped


def reconstruct_path(states: list[SearchState], end_index: int) -> list[dict[str, object]]:
    path = []
    current = end_index
    while current is not None:
        state = states[current]
        path.append(
            {
                "step": int(state.step),
                "objective": float(state.objective),
                "energy_value": float(state.energy_value),
                "escape_score": float(state.escape_score),
                "similarity_to_source": float(state.similarity_to_source),
                "support_size": int(state.support_size),
                "exact": bool(state.exact),
                "matched_classes": list(state.matched_classes),
                "canonical_key": state.canonical_key,
                "retention_ratio": float(state.retention_ratio),
                "retained_mass": float(state.retained_mass),
                "reverted_mass": float(state.reverted_mass),
                "exact44_penalty": float(state.exact44_penalty),
                "action": state.action,
                "mask": [int(value) for value in state.mask.tolist()],
            }
        )
        current = state.parent_index
    path.reverse()
    return path


def project_single_bridge(
    *,
    bridge_mask: torch.Tensor,
    source_mask: torch.Tensor,
    source_class_id: int,
    profile: ProtectionProfile,
    energy: GeometricHyperplaneEnergy,
    reference,
    scorer: EscapeScorerWrapper,
    points: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    source_support_size = int((source_mask >= 0.5).sum().item())
    bridge_total_mass = bridge_protection_total(profile, source_mask, bridge_mask)
    initial = evaluate_state(
        candidate_mask=bridge_mask,
        source_mask=source_mask,
        source_support_size=source_support_size,
        source_class_id=source_class_id,
        profile=profile,
        bridge_total_mass=bridge_total_mass,
        energy=energy,
        reference=reference,
        scorer=scorer,
        points=points,
        edge_index=edge_index,
        edge_type=edge_type,
        args=args,
        step=0,
        action={"type": "start"},
        parent_index=None,
    )
    states = [initial]
    beam = [0]
    best_exact_non44_index = None
    visited = {tuple(int(value) for value in initial.mask.tolist())}

    for step in range(1, args.steps + 1):
        expanded: list[tuple[float, int]] = []
        for parent_index in beam:
            neighbors = propose_neighbors(
                states[parent_index],
                source_mask=source_mask,
                source_support_size=source_support_size,
                source_class_id=source_class_id,
                profile=profile,
                bridge_total_mass=bridge_total_mass,
                energy=energy,
                reference=reference,
                scorer=scorer,
                points=points,
                edge_index=edge_index,
                edge_type=edge_type,
                args=args,
                step=step,
                parent_index=parent_index,
            )
            for candidate in neighbors:
                key = tuple(int(value) for value in candidate.mask.tolist())
                if key in visited:
                    continue
                visited.add(key)
                states.append(candidate)
                state_index = len(states) - 1
                expanded.append((candidate.objective, state_index))
                if candidate.exact and any(class_id != source_class_id for class_id in candidate.matched_classes):
                    best_exact_non44_index = state_index
                    break
            if best_exact_non44_index is not None:
                break
        if best_exact_non44_index is not None:
            break
        expanded.sort(key=lambda item: item[0], reverse=True)
        beam = [item[1] for item in expanded[: args.beam_size]]
        if not beam:
            break

    best_state_index = max(range(len(states)), key=lambda index: states[index].objective)
    exact_states = [(states[index].objective, index) for index in range(len(states)) if states[index].exact]
    exact_states.sort(key=lambda item: item[0], reverse=True)
    best_exact_index = exact_states[0][1] if exact_states else None

    if best_exact_non44_index is None:
        exact_non44 = [
            (states[index].objective, index)
            for index in range(len(states))
            if states[index].exact and any(class_id != source_class_id for class_id in states[index].matched_classes)
        ]
        if exact_non44:
            exact_non44.sort(key=lambda item: item[0], reverse=True)
            best_exact_non44_index = exact_non44[0][1]

    return {
        "success": best_exact_non44_index is not None,
        "bridge_total_mass": float(bridge_total_mass),
        "num_states_visited": len(states),
        "best_state": reconstruct_path(states, best_state_index),
        "best_exact_path": None if best_exact_index is None else reconstruct_path(states, best_exact_index),
        "best_exact_non44_path": None if best_exact_non44_index is None else reconstruct_path(states, best_exact_non44_index),
    }


def main() -> None:
    args = parse_args()
    args.jump_bank = normalize_cli_path(args.jump_bank)
    args.escape_scorer = normalize_cli_path(args.escape_scorer)
    args.output = normalize_cli_path(args.output)

    device = torch.device(args.device)
    jump_bank = json.loads(args.jump_bank.read_text(encoding="utf-8"))

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)
    energy = GeometricHyperplaneEnergy(points=points, config=EnergyConfig())
    reference = build_reference_database()
    scorer = EscapeScorerWrapper(args.escape_scorer, device=device)

    seed_items = load_seed_bank(points=points)
    seed_masks = seed_tensor(seed_items, device=torch.device("cpu"))
    requested_source_indices = set(parse_source_indices(args.source_seed_indices))

    results = []
    success_count = 0
    exact_count = 0
    exact_non44_count = 0

    for source_record in jump_bank.get("jump_proposals", []):
        source_seed_index = int(source_record["source_seed_index"])
        source_item = seed_items[source_seed_index]
        if int(source_item.class_id) != int(args.seed_class_id):
            continue
        if requested_source_indices and source_seed_index not in requested_source_indices:
            continue
        source_mask = seed_masks[source_seed_index].detach().cpu()
        profile = build_protection_profile(
            seed_items=seed_items,
            seed_masks=seed_masks,
            source_seed_index=source_seed_index,
            source_class_id=int(source_item.class_id),
        )
        top_remove = torch.topk(profile.remove_scores, k=min(12, profile.remove_scores.numel())).indices.tolist()
        top_add = torch.topk(profile.add_scores, k=min(12, profile.add_scores.numel())).indices.tolist()
        for proposal in list(source_record.get("targets", []))[: args.proposals_per_source]:
            if "mask" not in proposal:
                continue
            bridge_mask = torch.tensor(proposal["mask"], dtype=torch.float32)
            projection = project_single_bridge(
                bridge_mask=bridge_mask,
                source_mask=source_mask,
                source_class_id=int(source_item.class_id),
                profile=profile,
                energy=energy,
                reference=reference,
                scorer=scorer,
                points=points,
                edge_index=edge_index,
                edge_type=edge_type,
                args=args,
            )
            if projection["best_exact_path"] is not None:
                exact_count += 1
            if projection["best_exact_non44_path"] is not None:
                exact_non44_count += 1
            if projection["success"]:
                success_count += 1
            results.append(
                {
                    "source_seed_index": source_seed_index,
                    "source_class_id": int(source_item.class_id),
                    "source_example_id": int(source_item.example_id),
                    "proposal_type": proposal.get("proposal_type", "unknown"),
                    "proposal_index": int(proposal.get("proposal_index", len(results))),
                    "jump_size": int(proposal.get("jump_size", 0)),
                    "bridge_support_size": int(sum(proposal["mask"])),
                    "top_protected_remove_indices": [int(index) for index in top_remove if float(profile.remove_scores[index]) > 0.0],
                    "top_protected_add_indices": [int(index) for index in top_add if float(profile.add_scores[index]) > 0.0],
                    "projection": projection,
                }
            )

    summary = {
        "jump_bank": str(args.jump_bank),
        "escape_scorer": str(args.escape_scorer),
        "seed_class_id": int(args.seed_class_id),
        "num_bridge_starts": len(results),
        "num_with_any_exact_projection": int(exact_count),
        "num_with_exact_non44_projection": int(exact_non44_count),
        "exact_non44_projection_rate": (exact_non44_count / len(results)) if results else 0.0,
        "success_count": int(success_count),
        "steps": int(args.steps),
        "beam_size": int(args.beam_size),
        "top_remove": int(args.top_remove),
        "top_add": int(args.top_add),
        "top_swap": int(args.top_swap),
        "retention_weight": float(args.retention_weight),
        "revert_weight": float(args.revert_weight),
        "exact44_penalty": float(args.exact44_penalty),
        "min_retention_ratio": float(args.min_retention_ratio),
    }
    payload = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
