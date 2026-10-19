from __future__ import annotations

"""Local-budget continuation from actually visited reduced facets.

Anchors include geometrically valid reduced facets whose full lift fails. Their
historical paths prove reachability, but are never replayed or rewarded again.
This experiment deliberately does not enable counterfactual correction credit.
"""

from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .geometry import FacetOracle
from .reduced_subgroup_search import (
    ROLLOUT_CYCLE,
    ReducedAction,
    ReducedEdge,
    ReducedFacetResult,
    ReducedGeometry,
    ReducedNode,
    ReducedSearchConfig,
    ReducedState,
    ReducedSubgroupMCTS,
    ReducedSubgroupProblem,
    ReducedTransition,
    word_from_indices,
)


@dataclass(frozen=True)
class ReducedContinuationConfig(ReducedSearchConfig):
    enable_local_continuation: bool = True
    continuation_interval: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.continuation_interval < 2:
            raise ValueError("continuation_interval must preserve empty-root episodes")


@dataclass
class ObservedFacetAnchor:
    word: int
    witness: tuple[ReducedAction, ...]
    observed_iteration: int
    lifted: bool
    class_id: str | None
    productive: bool = False
    starts: int = 0

    @property
    def proof_cost(self) -> tuple[int, int]:
        return (sum(action.kind == "remove" for action in self.witness), len(self.witness))


class ReducedContinuationMCTS(ReducedSubgroupMCTS):
    def __init__(self, problem, config: ReducedContinuationConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.anchors: dict[int, ObservedFacetAnchor] = {}
        self.anchor_queue: deque[int] = deque()
        self.episode_origin: ObservedFacetAnchor | None = None
        self.episode_prefix: tuple[ReducedAction, ...] = ()

    def _register_anchor(
        self,
        terminal: ReducedFacetResult,
        witness: tuple[ReducedAction, ...],
        new_class: bool,
    ) -> None:
        if not terminal.valid:
            return
        word = terminal.closure_word
        full = terminal.full_result
        candidate = ObservedFacetAnchor(
            word, witness, self.current_iteration, terminal.lifted,
            full.class_id if terminal.lifted else None, productive=new_class,
        )
        anchor = self.anchors.get(word)
        if anchor is None:
            self.anchors[word] = candidate
            self.anchor_queue.append(word)
            self.stats["observed_continuation_anchors"] += 1
            self.stats["failed_lift_bridge_anchors"] += int(not candidate.lifted)
        else:
            if candidate.proof_cost < anchor.proof_cost:
                anchor.witness = witness
            anchor.productive |= new_class

    def _iteration_start(self, iteration: int) -> ReducedNode:
        self.episode_origin = None
        self.episode_prefix = ()
        if self.anchor_queue and iteration % self.config.continuation_interval == 0:
            word = self.anchor_queue.popleft()
            self.anchor_queue.append(word)
            anchor = self.anchors[word]
            self.episode_origin = anchor
            self.episode_prefix = anchor.witness
            anchor.starts += 1
            self.stats["continuation_episodes"] += 1
            self.stats["failed_lift_bridge_episodes"] += int(not anchor.lifted)
            self.stats["historical_actions_not_replayed"] += len(anchor.witness)
            self.stats["historical_corrections_not_replayed"] += anchor.proof_cost[0]
            state = ReducedState(word, "corrector", 0, word)
            return self._node(ReducedTransition(state, self.problem.facet_rank))
        self.stats["empty_root_episodes"] += 1
        return self.root

    def _correction_limit(self, path_actions: Sequence[ReducedAction]) -> int:
        if self.episode_origin is not None:
            return (
                self.config.productive_max_corrections
                if self.episode_origin.productive else self.config.max_corrections
            )
        return super()._correction_limit(path_actions)

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        if not self.config.enable_local_continuation:
            return super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)
        before = len(self.discovery_events)
        # Continuation has no LP entrance from the empty root. Project only the
        # reward's path metadata; the actual edge/action and its frequencies stay intact.
        reward_actions = (
            [replace(action, sampled_facet_word=0) for action in path_actions]
            if self.episode_origin is not None else path_actions
        )
        reward, new_class = super()._step_reward(
            node, edge, reward_actions, path_terminals, tree_edge
        )
        if new_class and self.episode_origin is not None:
            self.episode_origin.productive = True
        full_path = self.episode_prefix + tuple(path_actions)
        if edge.terminal is not None:
            self._register_anchor(edge.terminal, full_path, new_class)
        if len(self.discovery_events) != before:
            event = self.discovery_events[-1]
            origin = self.episode_origin
            prefix_length = len(self.episode_prefix)
            event.update({
                "path": [action.to_dict() for action in full_path],
                "depth": len(full_path),
                "local_path": [action.to_dict() for action in path_actions],
                "local_depth": len(path_actions),
                "historical_path_length": prefix_length,
                "historical_corrections": sum(a.kind == "remove" for a in self.episode_prefix),
                "local_corrections": sum(a.kind == "remove" for a in path_actions),
                "local_correction_limit": self._correction_limit(path_actions),
                "episode_kind": "continuation" if origin else "empty_root",
                "continuation_origin": ({
                    "word_hex": f"0x{origin.word:x}",
                    "lifted": origin.lifted,
                    "class_id": origin.class_id,
                    "observed_iteration": origin.observed_iteration,
                } if origin else None),
                "local_terminal_steps": [dict(item) for item in path_terminals],
                "terminal_steps": [
                    {**item, "action_depth": prefix_length + int(item["action_depth"])}
                    for item in path_terminals
                ],
                "terminal_steps_scope": "current_episode_with_absolute_depths",
            })
            self.stats[f"{event['episode_kind']}_discoveries"] += 1
            self.stats["failed_lift_origin_discoveries"] += int(bool(origin and not origin.lifted))
            if new_class:
                recorded = self.shared_discovery.discovered_classes[event["class_id"]]
                recorded.update({key: value for key, value in event.items() if key != "path"})
                recorded["path_action_ids"] = [action.action_id for action in full_path]
        return reward, new_class

    def _rollout_policy(self, iteration: int) -> str:
        return ROLLOUT_CYCLE[(iteration - 1) % len(ROLLOUT_CYCLE)]

    def run_iteration(self, iteration: int, *, global_iteration: int | None = None) -> None:
        if not self.config.enable_local_continuation:
            return super().run_iteration(iteration, global_iteration=global_iteration)
        # Keep the experimental episode orchestration isolated from the baseline.
        self.current_iteration = iteration
        self.current_global_iteration = int(global_iteration or iteration)
        start_node = node = self._iteration_start(iteration)
        policy = self._rollout_policy(iteration)
        self.stats[f"rollout_policy_{policy}"] += 1
        path: list[tuple[ReducedNode, ReducedEdge, float, bool]] = []
        path_actions: list[ReducedAction] = []
        path_terminals: list[dict[str, object]] = []
        persistent = True
        leaf_value = 0.0
        for _depth in range(self.config.max_depth):
            if (node.state.role == "corrector"
                    and node.state.corrections_used >= self._correction_limit(path_actions)):
                break
            if persistent:
                widened = self._widen(node)
                if widened is not None:
                    edge, child = widened
                    expanded = True
                else:
                    edge = self._select(node)
                    if edge is None:
                        leaf_value = self.config.dead_end_reward
                        self.stats["dead_ends"] += 1
                        break
                    child = self.nodes[edge.child_key]
                    expanded = False
                self.stats["tree_policy_steps"] += 1
            else:
                rollout = self._rollout_edge(node, policy)
                if rollout is None:
                    leaf_value = self.config.dead_end_reward
                    self.stats["dead_ends"] += 1
                    break
                edge, child = rollout
                expanded = False
                self.stats["rollout_steps"] += 1
            path_actions.append(edge.action)
            reward, new_class = self._step_reward(node, edge, path_actions, path_terminals, persistent)
            path.append((node, edge, reward, persistent))
            if new_class:
                edge.discovery_hits += 1
            node = child
            if persistent and expanded:
                persistent = False
        else:
            self.stats["depth_limit_hits"] += 1
        value = leaf_value
        for parent, edge, reward, tree_edge in reversed(path):
            value = reward + self.config.discount * value
            if tree_edge:
                edge.update(value, self.shared_discovery.epoch)
                parent.visits += 1
                parent.value_sum += value
        start_node.visits += int(not path or not path[0][3])
        self.stats["iterations"] += 1

    def result(self, *, target_ids: Sequence[str] = ()) -> dict[str, object]:
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_observed_reduced_facet_continuation_v1"
        result["features"].update({
            "one_persistent_mcts_tree": not self.config.enable_local_continuation,
            "one_shared_persistent_state_graph": True,
            "multiple_observed_episode_entries": self.config.enable_local_continuation,
            "observed_facet_continuation": self.config.enable_local_continuation,
            "failed_lift_bridge_continuation": self.config.enable_local_continuation,
            "local_correction_budget_reset": self.config.enable_local_continuation,
            "counterfactual_credit": False,
            "historical_path_reward_replay": False,
        })
        result["continuation"] = {
            "anchor_count": len(self.anchors),
            "started_anchor_count": sum(a.starts > 0 for a in self.anchors.values()),
            "failed_lift_anchor_count": sum(not a.lifted for a in self.anchors.values()),
            "anchor_visit_histogram": dict(sorted(Counter(a.starts for a in self.anchors.values()).items())),
            "max_anchor_witness_corrections": max((a.proof_cost[0] for a in self.anchors.values()), default=0),
            "selection": "FIFO round robin over actual visited concrete reduced facets",
            "historical_paths": "reachability proofs only; no execution or backpropagation",
        }
        return result


def audit_local_paths(
    problem: ReducedSubgroupProblem,
    events: Sequence[dict],
    config: ReducedContinuationConfig,
    handoff_root: Path,
    full_oracle: FacetOracle,
) -> dict:
    """Independent geometric replay, including the saved continuation boundary."""
    geometry = ReducedGeometry(problem, handoff_root=handoff_root, full_oracle=full_oracle)
    failures = []
    transitions = 0
    kinds: Counter[str] = Counter()
    max_local = max_historical = 0

    def replay(word: int, actions: Sequence[dict]) -> int:
        nonlocal transitions
        for action in actions:
            rank_before = geometry.rank_oracle.rank(word)
            if action["kind"] == "remove":
                retained = word_from_indices(action["retained_vertices"])
                if not retained or retained == word or retained & ~word:
                    raise ValueError("remove is not a nonempty strict subset")
                if not geometry.validate(word).valid:
                    raise ValueError("remove parent is not a reduced facet")
                word = retained
                if geometry.rank_oracle.rank(word) >= problem.facet_rank:
                    raise ValueError("remove did not leave the facet rank")
            elif action["kind"] == "add":
                word |= word_from_indices(action["add_vertices"])
                if geometry.rank_oracle.rank(word) <= rank_before:
                    raise ValueError("add did not increase rank")
            else:
                raise ValueError("unknown action kind")
            rank = geometry.rank_oracle.rank(word)
            if rank > problem.facet_rank:
                raise ValueError("rank exceeded reduced facet rank")
            if rank == problem.facet_rank:
                facet = geometry.validate(word)
                if not facet.valid:
                    raise ValueError("invalid reduced terminal")
                word = facet.closure_word
            transitions += 1
        return word

    for event in events:
        try:
            prefix_length = event["historical_path_length"]
            prefix = event["path"][:prefix_length]
            local = event["local_path"]
            if prefix_length < 0 or event["path"][prefix_length:] != local:
                raise ValueError("local path differs from full witness suffix")
            if event["depth"] != len(event["path"]) or event["local_depth"] != len(local):
                raise ValueError("path depth metadata mismatch")
            word = replay(0, prefix)
            origin = event["continuation_origin"]
            expected_kind = "continuation" if origin else "empty_root"
            if event["episode_kind"] != expected_kind:
                raise ValueError("episode kind mismatch")
            if origin:
                if word != int(origin["word_hex"], 16) or not geometry.validate(word).valid:
                    raise ValueError("historical proof does not reach continuation origin")
                source = geometry.validate(word)
                source_class = source.full_result.class_id if source.lifted else None
                if origin["lifted"] != source.lifted or origin["class_id"] != source_class:
                    raise ValueError("continuation origin lift metadata mismatch")
                if origin["observed_iteration"] >= event["iteration"]:
                    raise ValueError("continuation origin was not previously observed")
            elif prefix_length or word:
                raise ValueError("empty-root episode has a historical prefix")
            local_count = sum(a["kind"] == "remove" for a in local)
            historical_count = sum(a["kind"] == "remove" for a in prefix)
            if (local_count != event["local_corrections"]
                    or historical_count != event["historical_corrections"]):
                raise ValueError("correction metadata mismatch")
            if (event["local_correction_limit"] not in
                    (config.max_corrections, config.productive_max_corrections)
                    or local_count > event["local_correction_limit"]
                    or len(local) > config.max_depth):
                raise ValueError("local episode exceeded its budget")
            word = replay(word, local)
            if word != int(event["reduced_facet_word_hex"], 16):
                raise ValueError("local path reached a different facet")
            full = geometry.validate(word).full_result
            if not full or not full.valid or full.class_id != event["class_id"]:
                raise ValueError("local path lift/class mismatch")
            if list(full.canonical_h_row) != event["canonical_h_row"]:
                raise ValueError("local path canonical normal mismatch")
            kinds[expected_kind] += 1
            max_local = max(max_local, local_count)
            max_historical = max(max_historical, historical_count + local_count)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            failures.append({"class_id": event.get("class_id"), "error": str(error)})
    return {
        "paths_checked": len(events), "transitions_checked": transitions,
        "episode_discoveries": dict(kinds), "max_local_corrections": max_local,
        "max_full_witness_corrections": max_historical,
        "failure_count": len(failures), "failures": failures,
    }
