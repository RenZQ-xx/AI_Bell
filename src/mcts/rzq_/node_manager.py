"""RZQ node closure, compatibility caching, and symmetry-aware deduplication."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from baseline.orbit_blocks import BlockKey, add_block, unselected_blocks
from mcts.decision_trace import emit as trace_emit

if TYPE_CHECKING:
    from mcts.search import MCTSNode
    from .rollout_scorer import RZQRolloutScorer


def apply_block_map(key: BlockKey, block_map: tuple[int, ...]) -> BlockKey:
    moved = [0] * len(key)
    for index, selected in enumerate(key):
        if int(selected):
            moved[int(block_map[index])] = 1
    return tuple(moved)


@dataclass(frozen=True)
class NormalizedNodeState:
    key: BlockKey
    closure_blocks: tuple[int, ...]
    canonical_key: BlockKey


class RZQNodeManager:
    """Own the canonical index while preserving each master's real coordinates."""

    def __init__(self, scorer: "RZQRolloutScorer", *, symmetry_dedup: bool = True) -> None:
        self.scorer = scorer
        self.symmetry_dedup = symmetry_dedup
        self.canonical_index: dict[BlockKey, MCTSNode] = {}
        self.aliases: dict[int, MCTSNode] = {}
        self.creation_serial = 0

    def resolve(self, node: "MCTSNode") -> "MCTSNode":
        while id(node) in self.aliases:
            node = self.aliases[id(node)]
        return node

    def flat_closure(self, key: BlockKey) -> tuple[BlockKey, tuple[int, ...]]:
        return self.scorer.flat_closure(key)

    def canonicalize(self, key: BlockKey) -> tuple[BlockKey, tuple[int, ...]]:
        if not self.symmetry_dedup:
            return key, tuple(range(len(key)))
        best: BlockKey | None = None
        best_map: tuple[int, ...] | None = None
        for block_map in self.scorer.block_symmetry_maps:
            moved = apply_block_map(key, block_map)
            if best is None or moved < best:
                best, best_map = moved, tuple(block_map)
        if best is None or best_map is None:
            raise RuntimeError("node symmetry group is empty")
        return best, best_map

    def symmetry_map(self, source: BlockKey, target: BlockKey) -> tuple[int, ...]:
        if not self.symmetry_dedup:
            if source != target:
                raise ValueError("symmetry deduplication is disabled")
            return tuple(range(len(source)))
        for block_map in self.scorer.block_symmetry_maps:
            if apply_block_map(source, block_map) == target:
                return tuple(block_map)
        raise ValueError("states do not belong to the same global symmetry class")

    def normalize(self, key: BlockKey) -> NormalizedNodeState:
        closed, closure_blocks = self.flat_closure(key)
        canonical, _mapping = self.canonicalize(closed)
        return NormalizedNodeState(closed, closure_blocks, canonical)

    def get_or_create(
        self,
        nodes: dict[BlockKey, "MCTSNode"],
        key: BlockKey,
        *,
        path: list[int],
        parent: "MCTSNode | None",
        action: int | None,
    ) -> "MCTSNode":
        from mcts.search import MCTSNode

        normalized = self.normalize(key)
        existing = self.canonical_index.get(normalized.canonical_key)
        arrival = tuple(int(value) for value in path)
        if existing is not None:
            master = self.resolve(existing)
            self.scorer.refresh_node_compatibility(master)
            if arrival not in master.arrival_paths:
                master.arrival_paths.append(arrival)
            if parent is not None and action is not None:
                parent.child_symmetry_maps[int(action)] = self.symmetry_map(
                    normalized.key,
                    master.key,
                )
                parent.child_closure_blocks[int(action)] = normalized.closure_blocks
            trace_emit(
                "symmetric_node_reuse" if self.symmetry_dedup else "exact_node_reuse",
                incoming_key=list(normalized.key),
                incoming_path=list(arrival),
                canonical_key=list(normalized.canonical_key),
                master_key=list(master.key),
                master_paths=[list(item) for item in master.arrival_paths],
                symmetry_map=list(self.symmetry_map(normalized.key, master.key)),
                closure_blocks=list(normalized.closure_blocks),
            )
            return master

        rank = self.scorer.affine_rank(normalized.key)
        terminal = self.scorer.terminal_label(normalized.key) if rank >= 25 else None
        self.creation_serial += 1
        node = MCTSNode(
            key=normalized.key,
            path=list(path),
            rank=rank,
            parent=parent,
            action_from_parent=action,
            terminal=terminal,
            canonical_key=normalized.canonical_key,
            creation_index=self.creation_serial,
            arrival_paths=[arrival],
        )
        nodes[node.key] = node
        self.canonical_index[normalized.canonical_key] = node
        self.scorer.refresh_node_compatibility(node)
        if parent is not None and action is not None:
            parent.child_symmetry_maps[int(action)] = tuple(range(len(node.key)))
            parent.child_closure_blocks[int(action)] = normalized.closure_blocks
        return node

    @staticmethod
    def _choose_master(left: "MCTSNode", right: "MCTSNode") -> tuple["MCTSNode", "MCTSNode", str]:
        left_has, right_has = bool(left.children), bool(right.children)
        if left_has != right_has:
            return (left, right, "has_children") if left_has else (right, left, "has_children")
        if not left_has:
            if left.visits != right.visits:
                return (left, right, "more_visits") if left.visits > right.visits else (right, left, "more_visits")
        return (left, right, "earlier_creation") if left.creation_index <= right.creation_index else (right, left, "earlier_creation")

    def merge_nodes(
        self,
        nodes: dict[BlockKey, "MCTSNode"],
        left: "MCTSNode",
        right: "MCTSNode",
    ) -> "MCTSNode":
        left, right = self.resolve(left), self.resolve(right)
        if left is right:
            return left
        if not self.symmetry_dedup:
            raise ValueError("node merging is disabled")
        if left.canonical_key != right.canonical_key:
            raise ValueError("only symmetry-equivalent nodes can be merged")
        master, secondary, reason = self._choose_master(left, right)
        mapping = self.symmetry_map(secondary.key, master.key)
        before = {
            "master_visits": master.visits,
            "secondary_visits": secondary.visits,
            "master_children": sorted(master.children),
            "secondary_children": sorted(secondary.children),
        }
        master.visits += secondary.visits
        master.value_visits += secondary.value_visits
        master.value_sum += secondary.value_sum
        master.escape_sum += secondary.escape_sum
        master.survival_sum += secondary.survival_sum
        master.novelty_sum += secondary.novelty_sum
        for path in secondary.arrival_paths:
            if path not in master.arrival_paths:
                master.arrival_paths.append(path)
        if secondary.compatibility_epoch > master.compatibility_epoch:
            master.compatibility_counts = dict(secondary.compatibility_counts)
            master.compatibility_epoch = secondary.compatibility_epoch
        elif secondary.compatibility_epoch == master.compatibility_epoch:
            master.compatibility_counts.update(secondary.compatibility_counts)

        for action, child in list(secondary.children.items()):
            mapped_action = int(mapping[int(action)])
            child = self.resolve(child)
            mapped_child_key = apply_block_map(child.key, mapping)
            canonical, _ = self.canonicalize(mapped_child_key)
            target = self.canonical_index.get(canonical)
            if target is not None:
                child = self.merge_nodes(nodes, target, child)
            master.children.setdefault(mapped_action, child)
            if int(action) in secondary.action_priors:
                master.action_priors.setdefault(mapped_action, secondary.action_priors[int(action)])
            if int(action) in secondary.expansion_scores:
                master.expansion_scores.setdefault(mapped_action, secondary.expansion_scores[int(action)])
            if int(action) in secondary.expansion_score_details:
                master.expansion_score_details.setdefault(
                    mapped_action,
                    dict(secondary.expansion_score_details[int(action)]),
                )
            master.edge_value_visits[mapped_action] = (
                master.edge_value_visits.get(mapped_action, 0.0)
                + secondary.edge_value_visits.get(int(action), 0.0)
            )
            master.edge_value_sums[mapped_action] = (
                master.edge_value_sums.get(mapped_action, 0.0)
                + secondary.edge_value_sums.get(int(action), 0.0)
            )
            master.edge_escape_sums[mapped_action] = (
                master.edge_escape_sums.get(mapped_action, 0.0)
                + secondary.edge_escape_sums.get(int(action), 0.0)
            )
            master.edge_survival_sums[mapped_action] = (
                master.edge_survival_sums.get(mapped_action, 0.0)
                + secondary.edge_survival_sums.get(int(action), 0.0)
            )
            master.edge_novelty_sums[mapped_action] = (
                master.edge_novelty_sums.get(mapped_action, 0.0)
                + secondary.edge_novelty_sums.get(int(action), 0.0)
            )
            master.child_symmetry_maps[mapped_action] = tuple(mapping)
            master.child_closure_blocks[mapped_action] = tuple(
                sorted(int(mapping[index]) for index in secondary.child_closure_blocks.get(action, ()))
            )

        self.aliases[id(secondary)] = master
        self.canonical_index[master.canonical_key] = master
        nodes.pop(secondary.key, None)
        nodes[master.key] = master
        trace_emit(
            "symmetric_node_merge",
            reason=reason,
            canonical_key=list(master.canonical_key or ()),
            master_key=list(master.key),
            secondary_key=list(secondary.key),
            symmetry_map=list(mapping),
            before=before,
            after={
                "visits": master.visits,
                "value_visits": master.value_visits,
                "children": sorted(master.children),
                "arrival_paths": [list(path) for path in master.arrival_paths],
            },
        )
        return master
