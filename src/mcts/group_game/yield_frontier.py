from __future__ import annotations

"""Sparse replay steering from deterministic online geometric novelty."""

from collections import Counter
from dataclasses import dataclass
import math


@dataclass
class WorkEstimate:
    observations: int = 0
    reward_ema: float = 0.0
    seconds_ema: float = 0.0
    seconds: float = 0.0
    new_targets: int = 0
    new_pairs: int = 0

    def update(self, seconds, targets, pairs):
        weight = 1.0 if not self.observations else 0.2
        reward = targets + 0.1 * pairs
        self.reward_ema += weight * (reward - self.reward_ema)
        self.seconds_ema += weight * (seconds - self.seconds_ema)
        self.observations += 1
        self.seconds += seconds
        self.new_targets += targets
        self.new_pairs += pairs


class YieldFrontier:
    lanes = ("ridge", "orbit", "completion", "delivery")
    replay_stride = 4
    probe_stride = 4
    cost_floor = 0.0005

    def __init__(self):
        self.estimates = {}
        self.lane_estimates = {}
        self.services = Counter()
        self.decisions = []
        self.choices = 0
        self.probe_cursor = 0

    @staticmethod
    def novelty(before_pairs, observed_sources, after_pairs):
        fresh = {(s, t) for s, t in after_pairs - before_pairs if s != t}
        known = {t for _, t in before_pairs} | set(observed_sources)
        return len({t for _, t in fresh} - known), len(fresh)

    def observe(self, source, lane, seconds, new_targets, new_pairs):
        if lane not in self.lanes or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid measured frontier work")
        if not 0 <= new_targets <= new_pairs:
            raise ValueError("invalid geometric novelty")
        self.estimates.setdefault((source, lane), WorkEstimate()).update(seconds, new_targets, new_pairs)
        self.lane_estimates.setdefault(lane, WorkEstimate()).update(seconds, new_targets, new_pairs)

    def score(self, source, lane):
        local = self.estimates.get((source, lane))
        prior = self.lane_estimates.get(lane, WorkEstimate())
        estimate = local or prior
        observations = local.observations if local else 0
        uncertainty = 0.05 / math.sqrt(1 + observations + self.services[(source, lane)])
        return estimate.reward_ema + uncertainty

    def choose(self, available, *, iteration):
        available = {lane: sorted(set(available.get(lane, ()))) for lane in self.lanes
                     if available.get(lane)}
        if not available:
            return None
        self.choices += 1
        probe = (self.choices - 1) % self.probe_stride == 0
        if probe:
            for _ in self.lanes:
                lane = self.lanes[self.probe_cursor % len(self.lanes)]
                self.probe_cursor += 1
                if lane in available:
                    break
            source = min(available[lane], key=lambda w: (self.services[(w, lane)], w))
        else:
            # Delivery makes no geometry. It retains probe slots and is always
            # used after legacy or adaptive geometry work by the caller.
            candidates = [(w, lane) for lane, words in available.items() if lane != "delivery" for w in words]
            if not candidates:
                candidates = [(w, "delivery") for w in available["delivery"]]
            source, lane = min(candidates, key=lambda x: (-self.score(*x), self.services[x], x))
        estimate = self.estimates.get((source, lane), WorkEstimate())
        self.decisions.append({"iteration": iteration, "canonical_source_hex": f"0x{source:016x}",
                               "lane": lane, "reason": "probe" if probe else "yield",
                               "score": self.score(source, lane), "observations": estimate.observations,
                               "reward_ema": estimate.reward_ema, "seconds_ema": estimate.seconds_ema})
        self.services[(source, lane)] += 1
        return lane, source

    def to_dict(self):
        return {"class_labels_used": False, "measured_time_affects_choices": False,
                "selection_normalization": "novelty_per_dispatch",
                "replay_stride": self.replay_stride, "probe_stride": self.probe_stride,
                "cost_floor_seconds": self.cost_floor, "choices": self.choices,
                "decisions": self.decisions,
                "work": [{"canonical_source_hex": f"0x{source:016x}", "lane": lane,
                          **vars(estimate)} for (source, lane), estimate in sorted(self.estimates.items())]}
