from __future__ import annotations

"""Reduced-facet novelty drives search; full-space lifting only records results."""

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

from .reduced_ridge_exit_search import ReducedRidgeExitConfig, ReducedRidgeExitMCTS, RidgeExitGeometry
from .reduced_subgroup_search import ReducedFacetResult, ReducedGeometry, SharedDiscoveryState, word_from_indices


CHANGED_REWARD_FIELDS = frozenset({
    "new_reduced_facet_reward", "new_class_reward", "known_class_reward",
    "new_lifted_facet_reward", "failed_lift_reward",
})


@dataclass(frozen=True)
class ReducedNoveltyConfig(ReducedRidgeExitConfig):
    new_reduced_facet_reward: float = 10.0
    new_class_reward: float = 0.0
    known_class_reward: float = 0.0
    new_lifted_facet_reward: float = 0.0
    failed_lift_reward: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.new_reduced_facet_reward) or self.new_reduced_facet_reward <= 0:
            raise ValueError("reduced novelty reward must be finite and positive")
        if any(value != 0 for value in (self.new_class_reward, self.known_class_reward,
                                       self.new_lifted_facet_reward, self.failed_lift_reward)):
            raise ValueError("lifting/classification cannot provide search rewards in this mode")


@dataclass
class ReducedNoveltyDiscovery(SharedDiscoveryState):
    reduced_keys: set[tuple[str, int]] = field(default_factory=set)

    def observe(self, class_id, event):
        # Class collection is reporting-only; it must not advance the UCB epoch.
        epoch = self.epoch
        result = super().observe(class_id, event)
        self.epoch = epoch
        return result

    def observe_reduced(self, node_id: str, closure_word: int) -> bool:
        key = (node_id, closure_word)
        if key in self.reduced_keys:
            return False
        self.reduced_keys.add(key)
        self.epoch += 1
        return True


class DeferredLiftGeometry(RidgeExitGeometry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lift_cache: dict[int, ReducedFacetResult] = {}

    def _lift_reduced_support(self, original_support):
        return None

    def lift_taken_terminal(self, terminal: ReducedFacetResult) -> ReducedFacetResult:
        if not terminal.valid:
            raise ValueError("cannot lift an invalid reduced facet")
        cached = self.lift_cache.get(terminal.closure_word)
        if cached is not None:
            self.stats["taken_lift_cache_hits"] += 1
            return cached
        self.stats["taken_lift_checks"] += 1
        full = self.full_oracle.validate(terminal.original_support_word)
        lifted = replace(terminal, full_result=full)
        self.lift_cache[terminal.closure_word] = lifted
        self.stats["successful_full_lifts" if full.valid else "failed_full_lifts"] += 1
        if not full.valid:
            self.stats[f"failed_lift_{full.reason}"] += 1
        return lifted

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["taken_terminal_lift_cache_entries"] = len(self.lift_cache)
        result["lift_count_scope"] = "unique actual visited reduced closures, not candidate validation misses"
        return result


class ReducedNoveltyMCTS(ReducedRidgeExitMCTS):
    def __init__(self, problem, config: ReducedNoveltyConfig, **kwargs):
        shared = kwargs.get("shared_discovery")
        if shared is not None and not isinstance(shared, ReducedNoveltyDiscovery):
            raise ValueError("reduced novelty requires a reduced-epoch shared discovery state")
        kwargs["shared_discovery"] = shared if shared is not None else ReducedNoveltyDiscovery()
        super().__init__(problem, config, **kwargs)
        original = self.geometry
        self.geometry = DeferredLiftGeometry(
            problem, handoff_root=kwargs["handoff_root"], full_oracle=original.full_oracle,
            slope_tolerance=config.ridge_slope_tolerance,
            interval_tolerance=config.ridge_interval_tolerance,
        )
        self.geometry.shared_full_oracle = original.shared_full_oracle
        self.reduced_productive_anchors: set[int] = set()
        self.reduced_productive_basins: set[int] = set()
        self.reduced_witnesses: dict[int, tuple] = {}

    def _register_anchor(self, terminal, witness, new_class):
        super()._register_anchor(terminal, witness, False)
        if terminal.valid:
            self.anchors[terminal.closure_word].productive = (
                terminal.closure_word in self.reduced_productive_anchors
            )

    @staticmethod
    def _root_basin(path_actions) -> int:
        return next((action.sampled_facet_word for action in path_actions
                     if action.kind == "add" and action.sampled_facet_word), 0)

    def _correction_limit(self, path_actions) -> int:
        productive = (
            self.episode_origin.word in self.reduced_productive_anchors
            if self.episode_origin is not None else
            self._root_basin(path_actions) in self.reduced_productive_basins
        )
        return self.config.productive_max_corrections if productive else self.config.max_corrections

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        reduced_terminal = edge.terminal
        locally_new = globally_new = False
        if reduced_terminal is not None:
            locally_new = reduced_terminal.closure_word not in self.discovered_reduced_facets
            globally_new = self.shared_discovery.observe_reduced(
                self.problem.node_id, reduced_terminal.closure_word
            )
            # Temporary result enrichment is only for collection and witness metadata.
            # Persistent/candidate edges retain reduced-only validation and equal priors.
            edge.terminal = self.geometry.lift_taken_terminal(reduced_terminal)
            if globally_new and node.state.source_facet_word:
                source = node.state.source_facet_word
                if source != reduced_terminal.closure_word:
                    self.reduced_productive_anchors.add(source)
                    if source in self.anchors:
                        self.anchors[source].productive = True
                    if self.episode_origin is not None:
                        self.reduced_productive_anchors.add(self.episode_origin.word)
                    else:
                        basin = self._root_basin(path_actions)
                        if basin:
                            self.reduced_productive_basins.add(basin)
        try:
            reward, new_class = super()._step_reward(
                node, edge, path_actions, path_terminals, tree_edge
            )
        finally:
            edge.terminal = reduced_terminal
        if locally_new and not globally_new:
            reward -= self.config.new_reduced_facet_reward
        if self.episode_origin is not None:
            self.episode_origin.productive = self.episode_origin.word in self.reduced_productive_anchors
        if reduced_terminal is not None:
            if globally_new:
                self.stats["new_rewarded_reduced_facets"] += 1
            if locally_new:
                proof = self.episode_prefix + tuple(path_actions)
                self.reduced_witnesses[reduced_terminal.closure_word] = proof
                event = self.reduced_discovery_events[-1]
                event.update({
                    "episode_kind": self.entry_kind,
                    "source_facet_word_hex": f"0x{node.state.source_facet_word:x}",
                    "local_depth": len(path_actions), "historical_path_length": len(self.episode_prefix),
                    "local_corrections": sum(action.kind == "remove" for action in path_actions),
                    "local_correction_limit": self._correction_limit(path_actions),
                    "new_rewarded_reduced_facet": globally_new,
                    "continuation_origin": ({
                        "word_hex": f"0x{self.episode_origin.word:x}",
                        "observed_iteration": self.episode_origin.observed_iteration,
                    } if self.episode_origin is not None else None),
                })
        return reward, new_class

    def _diagnostic_snapshot(self) -> dict[str, int]:
        snapshot = super()._diagnostic_snapshot()
        snapshot.update({
            "new_reduced_facets": len(self.discovered_reduced_facets),
            "new_rewarded_reduced_facets": self.stats.get("new_rewarded_reduced_facets", 0),
            "taken_lift_checks": self.geometry.stats.get("taken_lift_checks", 0),
            "taken_lift_cache_hits": self.geometry.stats.get("taken_lift_cache_hits", 0),
        })
        return snapshot

    def result(self, *, target_ids: Sequence[str] = ()) -> dict:
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_reduced_facet_novelty_validation_only_lifting_v4"
        result["features"].update({
            "dynamic_global_new_class_reward": False,
            "reduced_facet_primary_reward": True,
            "full_lift_candidate_prior": False,
            "lifting_is_reporting_only": True,
        })
        result["tree"]["productive_root_basin_count"] = len(self.reduced_productive_basins)
        result["entry_diagnostics"]["lift_count_scope"] = "unique actual visited reduced facet closures"
        result["reduced_novelty"] = {
            "reward": self.config.new_reduced_facet_reward,
            "key": "(subgroup node_id, closed irredundant reduced facet support word)",
            "symmetry_class_quotient": False,
            "selection_epoch": self.shared_discovery.epoch,
            "epoch_source": "actual reduced-facet discovery, not full class discovery",
            "productive_source": "a subsequent new reduced exit; anchor registration alone is insufficient",
            "productive_anchor_count": len(self.reduced_productive_anchors),
            "productive_basin_count": len(self.reduced_productive_basins),
            "class_discovery_hits_are_diagnostics_only": True,
            "lifting_timing": "once per actual visited closure, before result collection; no pending queue",
            "validated_closures": len(self.geometry.lift_cache),
            "candidate_lifting": False,
        }
        records = {int(event["reduced_facet_word_hex"], 16): event for event in self.reduced_discovery_events}
        result["reduced_facets"]["witnesses"] = []
        for word, proof in self.reduced_witnesses.items():
            record = records[word]
            path = [action.to_dict() for action in proof]
            result["reduced_facets"]["witnesses"].append({
                **record, "class_id": f"reduced:{word:x}", "path": path,
                "local_path": path[record["historical_path_length"]:],
            })
        return result


def audit_reduced_witnesses(problem, events, config, handoff_root) -> dict:
    class ReducedOnlyAuditGeometry(ReducedGeometry):
        def _lift_reduced_support(self, original_support):
            return None

    geometry = ReducedOnlyAuditGeometry(problem, handoff_root=handoff_root)
    failures, checked = [], set()
    transitions = 0

    def replay(word, actions):
        nonlocal transitions
        for action in actions:
            before = geometry.rank_oracle.rank(word)
            if action["kind"] == "remove":
                retained = word_from_indices(action["retained_vertices"])
                if not retained or retained == word or retained & ~word or not geometry.validate(word).valid:
                    raise ValueError("invalid reduced removal")
                word = retained
                if geometry.rank_oracle.rank(word) >= problem.facet_rank:
                    raise ValueError("remove did not leave facet rank")
            elif action["kind"] == "add":
                word |= word_from_indices(action["add_vertices"])
                if geometry.rank_oracle.rank(word) <= before:
                    raise ValueError("add did not increase rank")
            else:
                raise ValueError("unknown reduced action")
            rank = geometry.rank_oracle.rank(word)
            if rank > problem.facet_rank:
                raise ValueError("rank exceeded facet rank")
            if rank == problem.facet_rank:
                terminal = geometry.validate(word)
                if not terminal.valid:
                    raise ValueError("invalid reduced facet")
                word = terminal.closure_word
            transitions += 1
        return word

    for event in events:
        try:
            path, local = event["path"], event["local_path"]
            prefix_length = event["historical_path_length"]
            if prefix_length < 0 or path[prefix_length:] != local or len(local) != event["local_depth"]:
                raise ValueError("local proof boundary mismatch")
            word = replay(0, path[:prefix_length])
            origin = event["continuation_origin"]
            if origin:
                if (word != int(origin["word_hex"], 16) or not geometry.validate(word).valid
                        or origin["observed_iteration"] >= event["iteration"]):
                    raise ValueError("invalid/late continuation origin")
            elif prefix_length or word:
                raise ValueError("empty-root proof has a historical prefix")
            count = sum(action["kind"] == "remove" for action in local)
            if (count != event["local_corrections"] or len(local) > config.max_depth
                    or event["local_correction_limit"] not in (config.max_corrections, config.productive_max_corrections)
                    or count > event["local_correction_limit"]):
                raise ValueError("local reduced proof exceeded its budget")
            word = replay(word, local)
            if not geometry.validate(word).valid or word != int(event["reduced_facet_word_hex"], 16):
                raise ValueError("reduced terminal mismatch")
            if word in checked:
                raise ValueError("duplicate reduced closure")
            checked.add(word)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            failures.append({"class_id": event.get("class_id"), "error": str(error)})
    return {"paths_checked": len(events), "unique_closures_checked": len(checked),
            "transitions_checked": transitions, "failure_count": len(failures), "failures": failures,
            "uses_full_lifting": False, "uses_interval_solver": False}
