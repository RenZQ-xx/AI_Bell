from __future__ import annotations

import importlib
import random
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


schedule = importlib.import_module("mcts.422.reduced_frontier_schedule_search")
multi = importlib.import_module("mcts.422.reduced_multi_exit_search")
continuation = importlib.import_module("mcts.422.reduced_continuation_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
benchmark = importlib.import_module("mcts.422.difficult_node_benchmark")
comparison = importlib.import_module("mcts.422.compare_frontier_benchmarks")


class ReportingOracle:
    def __init__(self, valid=False):
        self.valid, self.calls = valid, []

    def validate(self, word):
        self.calls.append(word)
        return reduced.FacetResult(
            self.valid, "fixture", word, support_word=word,
            affine_rank=79 if self.valid else -1,
            canonical_h_row=(word, 0) if self.valid else None,
            class_id=f"fixture-{word:x}" if self.valid else None,
        )

    def to_dict(self):
        return {"validation_count": len(self.calls)}


def pyramid_problem():
    points = np.asarray([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=float)
    return reduced.ReducedSubgroupProblem(
        "fixture", "fixture", 3, points, points, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4),
        (1, 2, 4, 8, 16), 5, Path("fixture"),
    )


def engine(**options):
    return schedule.ReducedFrontierScheduleMCTS(
        pyramid_problem(), schedule.ReducedFrontierScheduleConfig(iterations=1, progress_interval=0, **options),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )


def remember_lower(search, source=19):
    retained = reduced.ReducedNode(reduced.ReducedState(16, "filler", 1, source), 0)
    search._filler_actions(retained, sample_count=2, rng=random.Random(1))


def test_defaults_preserve_v6_and_old_variants():
    old, new = asdict(multi.ReducedMultiExitConfig()), asdict(schedule.ReducedFrontierScheduleConfig())
    assert all(new[key] == value for key, value in old.items())
    assert new["continuation_interval"] == 4
    assert benchmark.VARIANTS["multi_exit"][1] is multi.ReducedMultiExitMCTS
    assert benchmark.VARIANTS["frontier_schedule"][1] is schedule.ReducedFrontierScheduleMCTS
    with pytest.raises(ValueError):
        schedule.ReducedFrontierScheduleConfig(frontier_anchor_candidate_window=0)
    with pytest.raises(ValueError):
        schedule.ReducedFrontierScheduleConfig(enable_shared_face_frontier=False)


def test_cached_face_index_is_lookahead_not_discovery_or_lifting():
    search = engine()
    remember_lower(search)
    assert search.cached_source_faces[19] == {16}
    assert search._source_frontier(19)[:2] == (3, 3)
    assert search.frontier_anchor_ring == deque([19])
    assert not search.discovered_reduced_facets and not search.anchors
    assert not search.shared_discovery.epoch and not search.geometry.full_oracle.calls
    remember_lower(search)
    assert search.stats["schedule_cached_faces_indexed"] == 1


def test_remaining_outlet_reservation_survives_generic_action_frequency_order(monkeypatch):
    search = engine()
    remember_lower(search)
    node = reduced.ReducedNode(reduced.ReducedState(19, "corrector"), 2)
    lower = reduced.ReducedAction("remove", retained_word=16, sampled_facet_word=19,
                                  source="reduced_facet_lower_face")
    ridge = reduced.ReducedAction("remove", retained_word=17, sampled_facet_word=19,
                                  source="reduced_facet_ridge")
    search.corrector_ridge_bank[19] = [17]
    search.corrector_lower_face_bank[19] = [16]
    search.frontier_attempts[(19, 16)] = 1
    search.action_frequencies[lower.action_id] = 100
    monkeypatch.setattr(multi.ReducedMultiExitMCTS, "_corrector_actions", lambda *args, **kwargs: [ridge, lower])
    actions = search._generate_actions(node, rollout=True, policy="guided", rng=random.Random(2))
    assert actions[0] == lower
    assert search._quota_kind(19, rollout=True) == "remaining_exit"
    assert not search.corrector_schedule_attempts
    search.shared_discovery.reduced_keys.update((search.problem.node_id, word) for word in (22, 25, 28))
    assert search._quota_faces(19, "remaining_exit") == [16]
    for word in (22, 25, 28):
        search.frontier_traversals[(19, 16, word)] = 1
    assert not search._quota_faces(19, "remaining_exit")


def test_quota_cycle_tracks_actual_removals_and_fresh_faces_do_not_force_extra_sampling(monkeypatch):
    search = engine()
    search.current_iteration = search.current_global_iteration = 1
    remember_lower(search)
    source = reduced.ReducedNode(reduced.ReducedState(19, "corrector"), 2)
    search.corrector_ridge_bank[19] = [17, 18]
    search.corrector_lower_face_bank[19] = [16]
    search.fresh_source_faces[19] = {18}
    search.frontier_attempts[(19, 16)] = 1
    for expected, retained in (("remaining_exit", 16), ("untried_face", 17), ("fresh_face", 18)):
        assert search._quota_kind(19, rollout=True) == expected
        if expected != "untried_face":
            assert search._reserved_action(source, rollout=True, rng=random.Random(2)).retained_word == retained
        action = reduced.ReducedAction("remove", retained_word=retained, sampled_facet_word=19,
                                      source="reduced_facet_lower_face" if retained == 16 else "reduced_facet_ridge")
        edge, _transition = search._edge(source, action)
        search._step_reward(source, edge, [action], [], False)
        assert search.stats[f"schedule_{expected}_fulfilled"] == 1
    assert search._quota_kind(19, rollout=True) == "remaining_exit"
    assert search._quota_kind(19, rollout=False) == "remaining_exit"
    assert search.source_removals[19] == 3
    assert not search.fresh_source_faces[19]
    assert not search.geometry.stats["ridge_lp_attempts"]


def test_existing_reserved_tree_edge_is_selected_without_false_expansion():
    search = engine()
    remember_lower(search)
    node = reduced.ReducedNode(reduced.ReducedState(19, "corrector"), 2, visits=100)
    for retained, value in ((16, -10), (17, 100)):
        action = reduced.ReducedAction("remove", retained_word=retained, sampled_facet_word=19,
                                      source="reduced_facet_lower_face" if retained == 16 else "reduced_facet_ridge")
        edge, transition = search._edge(node, action)
        edge.visits, edge.value_sum = 2, value
        node.edges[action.action_id] = edge
        search._node(transition)
    search.frontier_attempts[(19, 16)] = 1
    assert search._widen(node) is None
    selected = search._select(node)
    assert selected.action.retained_word == 16
    assert selected.value_sum == -10 and selected.visits == 2
    assert not search.stats["tree_expansions"]
    assert search.stats["schedule_existing_edge_reservations"] == 1


def test_new_reserved_action_obeys_progressive_widening_and_correction_limit():
    search = engine()
    remember_lower(search)
    node = reduced.ReducedNode(reduced.ReducedState(19, "corrector"), 2)
    expanded = search._widen(node)
    assert expanded[0].action.retained_word == 16
    assert len(node.edges) <= search._child_limit(node)
    exhausted = reduced.ReducedNode(reduced.ReducedState(19, "corrector", 7), 2)
    assert search._reserved_action(exhausted, rollout=True, rng=random.Random(1)) is None


def test_candidate_queries_do_not_invoke_uncached_geometry(monkeypatch):
    search = engine()
    remember_lower(search)
    def forbidden(*args, **kwargs):
        raise AssertionError("selection should only read cached geometry")
    monkeypatch.setattr(search.geometry, "certified_lower_face_exits", forbidden)
    monkeypatch.setattr(search.geometry, "certified_ridge_exits", forbidden)
    monkeypatch.setattr(search.geometry, "sample_ridges", forbidden)
    search._source_frontier(19)
    node = reduced.ReducedNode(reduced.ReducedState(19, "corrector"), 2)
    assert search._reserved_action(node, rollout=True, rng=random.Random(1))


def test_anchor_mix_preserves_episode_fraction_and_uses_no_lift_labels():
    search = engine(frontier_anchor_candidate_window=1)
    for word in (19, 22, 25):
        search.anchors[word] = continuation.ObservedFacetAnchor(word, (), 1, word == 19, "ignored")
        search.anchor_queue.append(word)
    remember_lower(search, source=25)
    origins = []
    for iteration in (4, 8, 12):
        search._iteration_start(iteration)
        origins.append(search.episode_origin.word)
        assert search.episode_origin.starts == 1
    assert origins == [19, 25, 22]
    assert search.stats["schedule_fair_entries"] == 1
    assert search.stats["schedule_frontier_entries"] == 1
    assert search.stats["schedule_underexplored_entries"] == 1
    assert search._iteration_start(13) is search.root and search.episode_origin is None
    assert search.stats["continuation_episodes"] == 3
    assert search.stats["empty_root_episodes"] == 1


def test_first_frontier_queries_and_attempts_keep_diagnostic_keys_stable():
    search = engine()
    before = search._diagnostic_snapshot()
    remember_lower(search)
    assert before.keys() == search._diagnostic_snapshot().keys()


def test_posthoc_gap_diagnostics_distinguish_recorded_bank_cache_and_execution():
    source, retained, destination = 2 ** 100 + 19, 2 ** 99 + 16, 2 ** 100 + 22
    key = (f"0x{source:x}", f"0x{retained:x}")
    data = {"bank_history": {}, "cached_faces": {}}
    visited = {source}
    def category():
        return comparison.recorded_frontier_gap(source, retained, destination, visited, data)["category"]
    assert comparison.recorded_frontier_gap(source, retained, destination, visited, None) is None
    assert comparison.recorded_frontier_gap(None, None, destination, visited, data)["category"] == "direct_root_path"
    assert comparison.recorded_frontier_gap(source, retained, destination, set(), data)["category"] == "source_not_visited"
    assert comparison.recorded_frontier_gap(source, None, destination, visited, data)["category"] == "reference_retained_not_recorded"
    assert category() == "retained_not_in_recorded_bank_or_cache"
    data["bank_history"][key] = {"first_generated_iteration": 3, "first_attempted_iteration": None,
                                "actual_removals": 0}
    assert category() == "retained_banked_not_taken"
    data["bank_history"][key].update(first_attempted_iteration=4, actual_removals=1)
    assert category() == "retained_taken_without_complete_indexed_cache"
    data["cached_faces"][key] = {"actual_removals": 1, "outlets": []}
    assert category() == "reference_destination_not_in_cached_outlets"
    outlet = {"word_hex": f"0x{destination:x}", "actual_traversals": 0}
    data["cached_faces"][key]["outlets"].append(outlet)
    assert category() == "cached_outlet_not_taken"
    outlet["actual_traversals"] = 1
    assert category() == "route_taken_without_same_full_class"
    report = comparison.recorded_frontier_gap(source, retained, destination, visited, data)
    assert report["first_banked_iteration"] == 3 and report["first_taken_iteration"] == 4
    assert report["reference_destination_actual_traversals"] == 1


@pytest.mark.parametrize("tree_edge", [False, True])
def test_reward_is_unchanged_for_identical_actual_paths(tree_edge):
    searches = [engine(), multi.ReducedMultiExitMCTS(
        pyramid_problem(), multi.ReducedMultiExitConfig(iterations=1, progress_interval=0),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )]
    actions = [reduced.ReducedAction("add", add_word=19, sampled_facet_word=19),
               reduced.ReducedAction("remove", retained_word=16, sampled_facet_word=19,
                                     source="reduced_facet_lower_face"),
               reduced.ReducedAction("add", add_word=6, sampled_facet_word=22,
                                     source="certified_lower_face_exit")]
    all_values = []
    for search in searches:
        search.current_iteration = search.current_global_iteration = 1
        node, path, values = search.root, [], []
        for action in actions:
            path.append(action)
            edge, transition = search._edge(node, action)
            values.append(search._step_reward(node, edge, path, [], tree_edge)[0])
            node = reduced.ReducedNode(transition.state, transition.rank)
        all_values.append(values)
    assert all_values[0] == all_values[1]


external = pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="external handoff")


@external
def test_disabled_scheduling_reproduces_v6_paths_values_and_rng():
    data = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    searches = [multi.ReducedMultiExitMCTS(
        data, multi.ReducedMultiExitConfig(iterations=24, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    ), schedule.ReducedFrontierScheduleMCTS(
        data, schedule.ReducedFrontierScheduleConfig(iterations=24, seed=17, progress_interval=0,
                                                    enable_corrector_outlet_quota=False,
                                                    enable_frontier_anchor_schedule=False),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    )]
    results = [search.run() for search in searches]
    assert results[0]["tree"] == results[1]["tree"]
    assert searches[0].reduced_witnesses == searches[1].reduced_witnesses
    assert searches[0].rng.getstate() == searches[1].rng.getstate()
    assert {key: rng.getstate() for key, rng in searches[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in searches[1].rollout_rngs.items()}


@external
def test_lift_outcomes_cannot_change_scheduling_paths_values_or_rng():
    data = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    searches = [schedule.ReducedFrontierScheduleMCTS(
        data, schedule.ReducedFrontierScheduleConfig(iterations=36, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(valid),
    ) for valid in (False, True)]
    results = [search.run() for search in searches]
    assert results[0]["tree"] == results[1]["tree"]
    assert results[0]["frontier_schedule"] == results[1]["frontier_schedule"]
    assert searches[0].reduced_witnesses == searches[1].reduced_witnesses
    assert searches[0].rng.getstate() == searches[1].rng.getstate()
    assert {key: rng.getstate() for key, rng in searches[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in searches[1].rollout_rngs.items()}
    for key, node in searches[0].nodes.items():
        other = searches[1].nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum, edge.prior) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum, alternative.prior)
