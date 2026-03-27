from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, List, Sequence

import torch

if __package__ in (None, ""):
    from energy import GeometricHyperplaneEnergy
    from facet_reference import build_reference_database, classify_discovered_plane
    from orbit_tracker import OrbitCoverageTracker, OrbitRewardConfig
else:
    from .energy import GeometricHyperplaneEnergy
    from .facet_reference import build_reference_database, classify_discovered_plane
    from .orbit_tracker import OrbitCoverageTracker, OrbitRewardConfig


@dataclass
class OrbitTerminalConfig:
    invalid_energy: float = 250.0


def classify_terminal_mask(
    mask: torch.Tensor,
    energy: GeometricHyperplaneEnergy,
    reference: Dict[str, object],
) -> Dict[str, object] | None:
    validation = energy.validate_hard_mask(mask)
    if not validation["valid"]:
        return None
    validation["validation_tier"] = "candidate_supporting_face"
    validation.update(classify_discovered_plane(validation["normal"], validation["offset"], reference))
    return validation


def orbit_terminal_log_prob(
    item: Dict[str, object] | None,
    geometric_energy: float,
    tracker: OrbitCoverageTracker,
    reward_config: OrbitRewardConfig,
    terminal_temperature: float,
    terminal_config: OrbitTerminalConfig,
) -> float:
    safe_temperature = max(float(terminal_temperature), 1e-6)
    if item is None:
        return -terminal_config.invalid_energy / safe_temperature

    orbit_size = max(int(item.get("orbit_size") or 1), 1)
    novelty_bonus = tracker.bonus_for_item(item, reward_config)
    orbit_energy = geometric_energy - novelty_bonus
    return -orbit_energy / safe_temperature - log(float(orbit_size))


def build_orbit_terminal_terms(
    masks: Sequence[torch.Tensor],
    geometric_energy: torch.Tensor,
    energy: GeometricHyperplaneEnergy,
    tracker: OrbitCoverageTracker,
    reward_config: OrbitRewardConfig,
    terminal_temperature: float,
    terminal_config: OrbitTerminalConfig,
    reference: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if reference is None:
        reference = build_reference_database()

    items: List[Dict[str, object]] = []
    per_sample_items: List[Dict[str, object] | None] = []
    log_p_0_values: List[float] = []
    terminal_costs: List[float] = []
    novelty_bonuses: List[float] = []

    for index, mask in enumerate(masks):
        item = classify_terminal_mask(mask > 0.5, energy=energy, reference=reference)
        geometric_value = float(geometric_energy[index].item())
        if item is not None:
            items.append(item)
            novelty_bonus = tracker.bonus_for_item(item, reward_config)
        else:
            novelty_bonus = 0.0
        per_sample_items.append(item)
        log_p_0 = orbit_terminal_log_prob(
            item=item,
            geometric_energy=geometric_value,
            tracker=tracker,
            reward_config=reward_config,
            terminal_temperature=terminal_temperature,
            terminal_config=terminal_config,
        )
        log_p_0_values.append(log_p_0)
        novelty_bonuses.append(novelty_bonus)
        terminal_costs.append(-safe_temperature(terminal_temperature) * log_p_0)

    device = geometric_energy.device
    return {
        "items": items,
        "per_sample_items": per_sample_items,
        "log_p_0": torch.tensor(log_p_0_values, device=device, dtype=torch.float32),
        "terminal_cost": torch.tensor(terminal_costs, device=device, dtype=torch.float32),
        "novelty_bonus": torch.tensor(novelty_bonuses, device=device, dtype=torch.float32),
    }


def safe_temperature(value: float) -> float:
    return max(float(value), 1e-6)
