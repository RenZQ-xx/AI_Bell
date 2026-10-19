from __future__ import annotations

"""Work-type quotas over online geometry, independent of class labels."""

from collections import Counter


class FrontierScheduler:
    base_cycle = ("ridge", "completion", "orbit", "completion",
                  "delivery", "ridge", "completion", "orbit")
    pressure_cycle = ("ridge", "completion", "completion", "completion",
                      "delivery", "ridge", "completion", "orbit")

    def __init__(self, *, high_water: int = 24):
        if high_water <= 0:
            raise ValueError("frontier high water must be positive")
        self.high_water = high_water
        self.cursors = Counter()
        self.selections = Counter()
        self.source_counts = Counter()
        self.pressure_counts = Counter()
        self.max_backlog = 0

    def choose(self, available, remaining, *, scope):
        if not any(available.values()):
            return None
        backlog = max((remaining.get(w, 0) for w in available.get("completion", ())), default=0)
        self.max_backlog = max(self.max_backlog, backlog)
        pressure = backlog >= self.high_water
        cycle = self.pressure_cycle if pressure else self.base_cycle
        for _ in cycle:
            index = self.cursors[scope] % len(cycle)
            self.cursors[scope] += 1
            lane = cycle[index]
            sources = available.get(lane, ())
            if not sources:
                continue
            word = min(sources, key=lambda w: (self.source_counts[(scope[0], lane, w)], w))
            self.source_counts[(scope[0], lane, word)] += 1
            self.selections[(scope[0], lane)] += 1
            if pressure:
                self.pressure_counts[scope[0]] += 1
            return lane, word
        raise ValueError("unknown frontier lane")

    def to_dict(self):
        return {"class_labels_used": False, "high_water_lp_attempts": self.high_water,
                "base_cycle": self.base_cycle, "pressure_cycle": self.pressure_cycle,
                "max_observed_source_backlog": self.max_backlog,
                "selections": {f"{scope}:{lane}": n for (scope, lane), n in self.selections.items()},
                "pressure_selections": dict(self.pressure_counts),
                "source_selections": [{"scope": scope, "lane": lane,
                                       "canonical_source_hex": f"0x{word:016x}", "count": n}
                                      for (scope, lane, word), n in sorted(self.source_counts.items())]}
