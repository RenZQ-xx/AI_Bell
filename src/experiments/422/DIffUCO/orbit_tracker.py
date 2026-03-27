from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List

import torch


@dataclass
class OrbitRewardConfig:
    exact_novelty_weight: float = 0.5
    unknown_novelty_weight: float = 0.2
    orbit_size_penalty: float = 0.15
    apply_orbit_reward: bool = True


@dataclass
class OrbitCoverageTracker:
    class_counts: Dict[int, int] = field(default_factory=dict)
    canonical_counts: Dict[str, int] = field(default_factory=dict)

    def _novelty(self, count: int) -> float:
        return 1.0 / sqrt(count + 1.0)

    def bonus_for_item(self, item: Dict[str, object], config: OrbitRewardConfig) -> float:
        if not config.apply_orbit_reward:
            return 0.0

        matched_classes = [int(value) for value in item.get("matched_classes", [])]
        canonical_key = item.get("canonical_key")
        orbit_size = max(int(item.get("orbit_size") or 1), 1)
        orbit_penalty = config.orbit_size_penalty * float(torch.log(torch.tensor(float(orbit_size))).item())

        if matched_classes:
            best_bonus = None
            for class_id in matched_classes:
                count = self.class_counts.get(class_id, 0)
                candidate = config.exact_novelty_weight * self._novelty(count) - orbit_penalty
                if best_bonus is None or candidate > best_bonus:
                    best_bonus = candidate
            return best_bonus or 0.0

        if canonical_key is None:
            return 0.0
        count = self.canonical_counts.get(str(canonical_key), 0)
        return config.unknown_novelty_weight * self._novelty(count) - orbit_penalty

    def update(self, items: List[Dict[str, object]]) -> None:
        for item in items:
            for class_id in item.get("matched_classes", []):
                self.class_counts[int(class_id)] = self.class_counts.get(int(class_id), 0) + 1
            canonical_key = item.get("canonical_key")
            if canonical_key is not None:
                key = str(canonical_key)
                self.canonical_counts[key] = self.canonical_counts.get(key, 0) + 1

