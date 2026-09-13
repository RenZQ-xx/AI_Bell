from __future__ import annotations

"""Geometry-only completion endpoints; lookahead is not a discovery event."""

from collections import Counter, defaultdict


class EndpointGraph:
    def __init__(self):
        self.targets: dict[tuple[int, int, int], int] = {}
        self.pairs: set[tuple[int, int]] = set()
        self.pending: dict[int, set[int]] = defaultdict(set)
        self.executions: Counter[tuple[int, int]] = Counter()
        self.proposals: Counter[tuple[int, int]] = Counter()
        self.stats: Counter[str] = Counter()

    def register(self, key: tuple[int, int, int], target: int) -> None:
        if key in self.targets:
            if self.targets[key] != target:
                raise ValueError("inconsistent geometry endpoint for a verified completion")
            return
        self.targets[key] = target
        source = key[0]
        self.pairs.add((source, target))
        if target != source and not self.executions[(source, target)]:
            self.pending[source].add(target)

    def mark_used(self, key: tuple[int, int, int]) -> None:
        target = self.targets.get(key)
        if target is None:
            self.stats["unmapped_executions"] += 1
            return
        pair = key[0], target
        self.stats["mapped_executions"] += 1
        if self.executions[pair]:
            self.stats["repeat_pair_executions"] += 1
        if target == key[0]:
            self.stats["self_pair_executions"] += 1
        self.executions[pair] += 1
        self.pending[key[0]].discard(target)

    def to_dict(self) -> dict[str, object]:
        return {
            "class_labels_used": False,
            "lookahead_is_discovery": False,
            "verified_action_keys": len(self.targets),
            "unique_source_target_pairs": len(self.pairs),
            "lookahead_canonical_targets": len(set(self.targets.values())),
            "self_action_keys": sum(key[0] == target for key, target in self.targets.items()),
            "distinct_executed_pairs": len(self.executions),
            "pending_nonself_pairs": sum(map(len, self.pending.values())),
            "statistics": dict(self.stats),
            "pairs": [{"source_hex": f"0x{s:016x}", "target_hex": f"0x{t:016x}",
                       "executions": self.executions[(s, t)], "frontier_proposals": self.proposals[(s, t)]}
                      for s, t in sorted(self.pairs)],
        }
