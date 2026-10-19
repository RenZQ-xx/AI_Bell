from __future__ import annotations

import importlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


fresh = importlib.import_module("mcts.422.reduced_fresh_geometry_search")
multi = importlib.import_module("mcts.422.reduced_multi_exit_search")
schedule = importlib.import_module("mcts.422.reduced_frontier_schedule_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
benchmark = importlib.import_module("mcts.422.difficult_node_benchmark")
comparison = importlib.import_module("mcts.422.compare_frontier_benchmarks")


class ReportingOracle:
    def __init__(self, valid=False):
        self.valid, self.calls = valid, []

    def validate(self, word):
        self.calls.append(word)
        return reduced.FacetResult(self.valid, "fixture", word, support_word=word,
                                   affine_rank=79 if self.valid else -1,
                                   canonical_h_row=(word, 0) if self.valid else None,
                                   class_id=f"fixture-{word:x}" if self.valid else None)

    def to_dict(self):
        return {"validation_count": len(self.calls)}


def pyramid_problem():
    points = np.asarray([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=float)
    return reduced.ReducedSubgroupProblem("fixture", "fixture", 3, points, points, (0, 1, 2, 3, 4),
                                          (0, 1, 2, 3, 4), (1, 2, 4, 8, 16), 5, Path("fixture"))


def engine(**options):
    return fresh.ReducedFreshGeometryMCTS(
        pyramid_problem(), fresh.ReducedFreshGeometryConfig(iterations=1, progress_interval=0, **options),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )


def source_node(word=19, corrections=0, visits=0):
    return reduced.ReducedNode(reduced.ReducedState(word, "corrector", corrections), 2, visits=visits)


def remember_lower(search):
    node = reduced.ReducedNode(reduced.ReducedState(16, "filler", 1, 19), 0)
    search._filler_actions(node, sample_count=2, rng=random.Random(1))


def test_default_config_isolated_and_same_v6_reward_budgets_sampling():
    config = fresh.ReducedFreshGeometryConfig()
    old = asdict(multi.ReducedMultiExitConfig())
    assert all(asdict(config)[key] == value for key, value in old.items())
    assert not config.enable_frontier_anchor_schedule
    assert config.fresh_probe_lp_limit == 3
    assert benchmark.VARIANTS["multi_exit"][1] is multi.ReducedMultiExitMCTS
    assert benchmark.VARIANTS["frontier_schedule"][1] is schedule.ReducedFrontierScheduleMCTS
    for variant in ("fresh_geometry", "fresh_geometry_probe_only", "fresh_geometry_quota_only"):
        assert benchmark.VARIANTS[variant][1] is fresh.ReducedFreshGeometryMCTS
    assert not fresh.ReducedFreshProbeOnlyConfig().enable_corrector_outlet_quota
    assert not fresh.ReducedSparseQuotaOnlyConfig().enable_fresh_geometry_probes
    with pytest.raises(ValueError):
        fresh.ReducedFreshGeometryConfig(fresh_probe_lp_limit=0)
    with pytest.raises(ValueError):
        fresh.ReducedFreshProbeOnlyConfig(enable_shared_face_frontier=False)


def test_global_actual_removal_clock_does_not_restart_for_each_source_or_context():
    search = engine()
    search.current_iteration = search.current_global_iteration = 1
    slots = []
    for index in range(8):
        slots.append(search._slot())
        source, retained = (19, 17) if index % 2 == 0 else (22, 18)
        node = source_node(source)
        action = reduced.ReducedAction("remove", retained_word=retained, sampled_facet_word=source,
                                       source="reduced_facet_ridge")
        edge, _transition = search._edge(node, action)
        search._step_reward(node, edge, [action], [], bool(index % 2))
    assert tuple(slots) == fresh.SPARSE_CYCLE
    assert search.removal_clock == 8 and search._slot() == "regular"
    assert search.stats["sparse_regular_removals"] == 6
    assert search.stats["sparse_remaining_exit_removals"] == 1
    assert search.stats["sparse_fresh_geometry_removals"] == 1
    assert search.source_removals[19] == search.source_removals[22] == 4


def test_remaining_slot_protects_unknown_closures_not_known_unused_routes(monkeypatch):
    search = engine()
    remember_lower(search)
    search.removal_clock = 6
    def forbidden(*args, **kwargs):
        raise AssertionError("cache service must not query new geometry")
    monkeypatch.setattr(search.geometry, "probe_fresh_ridge", forbidden)
    monkeypatch.setattr(search.geometry, "certified_lower_face_exits", forbidden)
    assert search._planned_action(source_node(), rollout=True, rng=random.Random(1)).retained_word == 16
    search.shared_discovery.reduced_keys.update((search.problem.node_id, word) for word in (22, 25, 28))
    assert search._outlet_frontier(19, 16)[1]
    assert search._planned_action(source_node(), rollout=True, rng=random.Random(1)) is None
    assert search.removal_clock == 6 and not search.geometry.full_oracle.calls


def test_regular_tree_select_uses_ucb_and_sparse_existing_edge_is_not_expansion():
    search = engine()
    remember_lower(search)
    node = source_node(visits=100)
    for retained, value in ((16, -10), (17, 100)):
        action = reduced.ReducedAction("remove", retained_word=retained, sampled_facet_word=19,
                                       source="reduced_facet_lower_face" if retained == 16 else "reduced_facet_ridge")
        edge, transition = search._edge(node, action)
        edge.visits, edge.value_sum = 2, value
        node.edges[action.action_id] = edge
        search._node(transition)
    for clock in range(6):
        search.removal_clock = clock
        assert search._select(node).action.retained_word == 17
    search.removal_clock = 6
    assert search._widen(node) is None
    assert search._select(node).action.retained_word == 16
    assert not search.stats["tree_expansions"]
    assert search.stats["sparse_existing_edge_reservations"] == 1


def test_bounded_probe_portfolio_returns_fresh_supporting_ridges_without_lifting():
    search = engine()
    seen, seeds, modes = set(), [], []
    for _ in range(8):
        before = search.geometry.stats["fresh_probe_lp_attempts"]
        words = search.geometry.probe_fresh_ridge(19, seen=seen, lp_limit=3, rng=search.fresh_probe_rng)
        modes.append(search.geometry.last_probe["mode"])
        seeds.append(search.geometry.last_probe["condition_vertices"])
        assert 1 <= search.geometry.stats["fresh_probe_lp_attempts"] - before <= 3
        for word in words:
            assert word in (3, 17, 18) and word not in seen
            assert search.geometry.rank_oracle.rank(word) == search.problem.ridge_rank
            assert all(word & (1 << vertex) for vertex in seeds[-1])
            seen.add(word)
    assert tuple(modes) == fresh.PROBE_MODES * 2
    assert seeds[1] != seeds[2]
    assert seen == {3, 17, 18}
    assert search.geometry.stats["fresh_probe_new_geometry"] == 3
    assert search.geometry.stats["fresh_probe_lp_attempts"] == search.geometry.stats["ridge_lp_attempts"]
    assert not search.geometry.full_oracle.calls and not search.shared_discovery.reduced_keys


def test_infeasible_pair_probe_stops_after_one_lp_and_falls_back():
    search = engine()
    search.geometry.stats["fresh_probe_requests"] = 3
    search.geometry.condition_vertex_attempts.update({(15, 1): 10, (15, 3): 10})
    assert search.geometry.probe_fresh_ridge(15, seen=set(), lp_limit=3, rng=random.Random(1)) == ()
    assert set(search.geometry.last_probe["condition_vertices"]) == {0, 2}
    assert search.geometry.last_probe["lp_attempts"] == 1
    assert search.geometry.stats["fresh_probe_lp_failures"] == 1


def test_probe_bypasses_backoff_but_resets_it_only_for_actual_new_geometry():
    search = engine()
    geometry = search.geometry
    geometry.ridge_next_probe[19] = 10000
    geometry.ridge_zero_streak[19] = 9
    assert geometry.sample_ridges(19, count=1, rng=random.Random(1)) == ()
    assert geometry.stats["ridge_backoff_skipped"] == 1
    assert geometry.probe_fresh_ridge(19, seen=set(), lp_limit=3, rng=random.Random(2))
    assert geometry.ridge_next_probe[19] == 2 and geometry.ridge_zero_streak[19] == 0
    geometry.ridge_next_probe[19] = 10000
    geometry.ridge_zero_streak[19] = 9
    assert not geometry.probe_fresh_ridge(19, seen={3, 17, 18}, lp_limit=3, rng=random.Random(3))
    assert geometry.ridge_next_probe[19] == 10000 and geometry.ridge_zero_streak[19] == 9


def test_seen_geometry_is_tracked_when_normal_backoff_is_disabled():
    search = engine(enable_ridge_sampling_backoff=False)
    sampled = search.geometry.sample_ridges(19, count=3, rng=random.Random(1))
    assert sampled and set(sampled) <= search.geometry.ridge_sample_seen[19]
    words = search.geometry.probe_fresh_ridge(19, seen=search.geometry.ridge_sample_seen[19],
                                             lp_limit=3, rng=random.Random(2))
    assert not set(words) & set(sampled)


def test_probe_ticket_is_reused_and_only_execution_consumes_slot():
    search = engine()
    search.current_iteration = search.current_global_iteration = 1
    search.removal_clock = 7
    node = source_node()
    action = search._planned_action(node, rollout=True, rng=random.Random(1))
    assert action is not None
    calls = search.geometry.stats["fresh_probe_lp_attempts"]
    assert search._planned_action(node, rollout=False, rng=random.Random(9)) == action
    assert search._generate_actions(node, rollout=True, policy="guided", rng=random.Random(3)) == [action]
    assert search.geometry.stats["fresh_probe_lp_attempts"] == calls
    assert search.removal_clock == 7 and not search.discovered_reduced_facets and not search.shared_discovery.epoch
    assert not search.geometry.full_oracle.calls
    edge, _transition = search._edge(node, action)
    search._step_reward(node, edge, [action], [], False)
    assert search.removal_clock == 8 and not search.probe_tickets
    assert search.stats["sparse_fresh_geometry_fulfilled"] == 1
    assert search.probe_events[0]["first_taken_iteration"] == 1
    assert search.stats["fresh_probe_executed_proposals"] == 1
    retained = reduced.ReducedNode(_transition.state, _transition.rank)
    completion = search._filler_actions(retained, sample_count=2, rng=random.Random(1))[0]
    exit_edge, _exit = search._edge(retained, completion)
    search._step_reward(retained, exit_edge, [action, completion], [], False)
    assert search.stats["fresh_probe_exit_arrivals"] == 1
    assert search.stats["fresh_probe_new_reduced_exit_arrivals"] == 1
    assert search.probe_events[0]["new_reduced_exit_arrivals"] == 1


def test_no_probe_when_tree_cannot_widen_or_correction_budget_exhausted(monkeypatch):
    search = engine()
    search.removal_clock = 7
    node = source_node()
    action = reduced.ReducedAction("remove", retained_word=17, sampled_facet_word=19)
    edge, _transition = search._edge(node, action)
    node.edges = {str(index): edge for index in range(search._child_limit(node))}
    def forbidden(*args, **kwargs):
        raise AssertionError("no LP should be spent on an unexpandable fresh edge")
    monkeypatch.setattr(search.geometry, "probe_fresh_ridge", forbidden)
    assert search._widen(node) is None and search._select(node) is edge
    assert search._planned_action(source_node(corrections=7), rollout=True, rng=random.Random(1)) is None
    assert search.stats["sparse_probe_pw_skipped"] == 1


def test_unbanked_fresh_proposal_does_not_evict_untried_bank_and_is_reported(monkeypatch):
    search = engine(corrector_face_bank_limit=1)
    search.current_iteration = search.current_global_iteration = 1
    search.removal_clock = 7
    search.corrector_ridge_bank[19] = [17]
    monkeypatch.setattr(search.geometry, "probe_fresh_ridge", lambda *args, **kwargs: (3,))
    action = search._planned_action(source_node(), rollout=True, rng=random.Random(1))
    assert action.retained_word == 3 and search.corrector_ridge_bank[19] == [17]
    assert search.stats["sparse_probe_bank_rejections"] == 1
    report = search.result()
    row = report["frontier_schedule"]["bank_history"][0]
    assert row["first_generated_iteration"] == 1 and row["first_banked_iteration"] is None
    data = {"bank_history": {("0x13", "0x3"): row}, "cached_faces": {}}
    assert comparison.recorded_frontier_gap(19, 3, 22, {19}, data)["category"] == "retained_proposed_not_taken"


def test_diagnostic_keys_stable_before_first_probe():
    search = engine()
    before = search._diagnostic_snapshot()
    search.removal_clock = 7
    search._planned_action(source_node(), rollout=True, rng=random.Random(1))
    assert before.keys() == search._diagnostic_snapshot().keys()


@pytest.mark.parametrize("tree_edge", [False, True])
def test_identical_path_reward_is_unchanged(tree_edge):
    searches = [engine(), multi.ReducedMultiExitMCTS(
        pyramid_problem(), multi.ReducedMultiExitConfig(iterations=1, progress_interval=0),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )]
    actions = [reduced.ReducedAction("add", add_word=19, sampled_facet_word=19),
               reduced.ReducedAction("remove", retained_word=16, sampled_facet_word=19, source="reduced_facet_lower_face"),
               reduced.ReducedAction("add", add_word=6, sampled_facet_word=22, source="certified_lower_face_exit")]
    rewards = []
    for search in searches:
        search.current_iteration = search.current_global_iteration = 1
        node, path, values = search.root, [], []
        for action in actions:
            path.append(action)
            edge, transition = search._edge(node, action)
            values.append(search._step_reward(node, edge, path, [], tree_edge)[0])
            node = reduced.ReducedNode(transition.state, transition.rank)
        rewards.append(values)
    assert rewards[0] == rewards[1]


def test_explicit_policy_switch_comparison_cannot_hide_seed_or_budget_changes(tmp_path):
    paths = []
    for name, config in (("old", schedule.ReducedFrontierScheduleConfig()),
                         ("new", fresh.ReducedFreshGeometryConfig())):
        output = tmp_path / f"{name}.json"
        output.write_text(json.dumps({
            "coverage": {"discoveries": []}, "reduced_facets": {"discovery_events": []},
            "geometry": {"statistics": {"ridge_lp_attempts": 1}},
            "runtime": {"iterations_completed": 1, "elapsed_seconds": 1},
            "entry_diagnostics": {"windows": []},
        }), encoding="utf-8")
        path = tmp_path / f"{name}.summary.json"
        path.write_text(json.dumps({"state": "COMPLETED", "selected_node_ids": ["fixture"],
                                   "config": asdict(config), "elapsed_seconds": 1,
                                   "runs": [{"node_id": "fixture", "output": str(output),
                                             "audit": {"failure_count": 0, "elapsed_seconds": 0,
                                                       "reduced_witness_audit": {"paths_checked": 0}}}]}), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="parameters"):
        comparison.compare(paths[:1], paths[1:])
    report = comparison.compare(paths[:1], paths[1:], allow_policy_differences=("enable_frontier_anchor_schedule",))
    assert report["intentional_policy_differences"][0]["switches"] == {
        "enable_frontier_anchor_schedule": {"reference": True, "variant": False}}
    new_output = tmp_path / "new.json"
    result = json.loads(new_output.read_text(encoding="utf-8"))
    result["fresh_geometry"] = {
        "geometry_statistics": {"fresh_probe_lp_attempts": 1},
        "probe_events": [{"mode": "vertex", "lp_attempts": 1, "retained_word_hex": "0x1",
                          "first_taken_iteration": 1, "exit_arrivals": 1,
                          "new_reduced_exit_arrivals": 1, "terminal_class_discoveries": 0}],
    }
    new_output.write_text(json.dumps(result), encoding="utf-8")
    report = comparison.compare(paths[:1], paths[1:], allow_policy_differences=("enable_frontier_anchor_schedule",))
    assert report["variant"]["totals"]["lp_attempts"] == 1
    assert report["variant"]["totals"]["fresh_probe_lp_attempts"] == 1
    assert report["variant"]["fresh_probe_mode_statistics"]["vertex"]["new_geometry"] == 1
    with pytest.raises(ValueError, match="policy switches"):
        comparison.compare(paths[:1], paths[1:], allow_policy_differences=("seed",))
    summary = json.loads(paths[1].read_text(encoding="utf-8"))
    summary["config"]["seed"] += 1
    paths[1].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="parameters"):
        comparison.compare(paths[:1], paths[1:], allow_policy_differences=("enable_frontier_anchor_schedule",))


external = pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="external handoff")


@external
def test_disabling_both_corrector_changes_replays_v6_with_fifo_and_exact_rng_values():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    searches = [multi.ReducedMultiExitMCTS(
        problem, multi.ReducedMultiExitConfig(iterations=32, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    ), fresh.ReducedFreshGeometryMCTS(
        problem, fresh.ReducedFreshGeometryConfig(iterations=32, seed=17, progress_interval=0,
                                                 enable_corrector_outlet_quota=False, enable_fresh_geometry_probes=False),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    )]
    results = [search.run() for search in searches]
    assert results[0]["tree"] == results[1]["tree"]
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


@external
def test_lifting_outcomes_cannot_change_fresh_probes_paths_values_or_rng():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    searches = [fresh.ReducedFreshGeometryMCTS(
        problem, fresh.ReducedFreshGeometryConfig(iterations=40, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(valid),
    ) for valid in (False, True)]
    results = [search.run() for search in searches]
    assert results[0]["tree"] == results[1]["tree"]
    assert searches[0].reduced_witnesses == searches[1].reduced_witnesses
    assert searches[0].probe_events
    # Full-class reporting may differ; proposals, actual geometry and scheduling must not.
    events = [[{key: value for key, value in event.items() if key != "terminal_class_discoveries"}
               for event in search.probe_events] for search in searches]
    assert events[0] == events[1]
    assert searches[0].rng.getstate() == searches[1].rng.getstate()
    assert searches[0].fresh_probe_rng.getstate() == searches[1].fresh_probe_rng.getstate()
    assert {key: rng.getstate() for key, rng in searches[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in searches[1].rollout_rngs.items()}
    for key, node in searches[0].nodes.items():
        other = searches[1].nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum, edge.prior) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum, alternative.prior)
