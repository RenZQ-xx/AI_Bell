from __future__ import annotations

import importlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


adaptive = importlib.import_module("mcts.422.reduced_adaptive_exploration_search")
fresh = importlib.import_module("mcts.422.reduced_fresh_geometry_search")
multi = importlib.import_module("mcts.422.reduced_multi_exit_search")
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


def problem():
    points = np.asarray([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=float)
    return reduced.ReducedSubgroupProblem("fixture", "fixture", 3, points, points, (0, 1, 2, 3, 4),
                                          (0, 1, 2, 3, 4), (1, 2, 4, 8, 16), 5, Path("fixture"))


def engine(**options):
    search = adaptive.ReducedAdaptiveExplorationMCTS(
        problem(), adaptive.ReducedAdaptiveExplorationConfig(iterations=1, progress_interval=0, **options),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )
    search.current_iteration = search.current_global_iteration = 1
    return search


def source_node(word=19):
    return reduced.ReducedNode(reduced.ReducedState(word, "corrector", 0), 2)


def take_fresh_remove(search, word=19, retained=17):
    search.removal_clock = 7
    action = reduced.ReducedAction("remove", retained_word=retained, sampled_facet_word=word)
    node = source_node(word)
    edge, transition = search._edge(node, action)
    search._step_reward(node, edge, [action], [], False)
    return action, transition


def register_sources(search, words=(19, 22, 28, 25, 15)):
    for iteration, word in enumerate(words, 1):
        search.current_iteration = iteration
        search._register_anchor(search.geometry.validate(word),
                                (reduced.ReducedAction("add", add_word=word, sampled_facet_word=word),), False)


def test_isolated_defaults_and_ablation_configs_preserve_old_parameters():
    config = asdict(adaptive.ReducedAdaptiveExplorationConfig())
    assert all(config[key] == value for key, value in asdict(fresh.ReducedFreshProbeOnlyConfig()).items())
    assert not adaptive.ReducedAdaptiveProbeOnlyConfig().enable_diverse_continuation
    entry = adaptive.ReducedDiverseEntryOnlyConfig()
    assert not entry.enable_fresh_geometry_probes and not entry.enable_adaptive_probe_cooldown
    assert benchmark.VARIANTS["multi_exit"][1] is multi.ReducedMultiExitMCTS
    assert benchmark.VARIANTS["fresh_geometry_probe_only"][1] is fresh.ReducedFreshGeometryMCTS
    for variant in ("adaptive_geometry", "adaptive_probe_only", "diverse_entry_only"):
        assert benchmark.VARIANTS[variant][1] is adaptive.ReducedAdaptiveExplorationMCTS
    for options in ({"probe_warmup_requests": -1}, {"probe_cooldown_max_interval": 0},
                    {"diverse_entry_candidate_window": 1}, {"diverse_entry_recent_count": 0},
                    {"enable_fresh_geometry_probes": False}, {"enable_frontier_anchor_schedule": True},
                    {"enable_local_continuation": False}):
        with pytest.raises(ValueError):
            adaptive.ReducedAdaptiveExplorationConfig(**options)


def test_query_tickets_do_not_advance_source_opportunities_or_repeat_feedback(monkeypatch):
    search = engine()
    search.removal_clock = 7
    monkeypatch.setattr(search.geometry, "probe_fresh_ridge", lambda *args, **kwargs: ())
    assert search._probe_action(source_node(), rollout=True) is None
    assert search._probe_action(source_node(), rollout=False) is None
    state = search.probe_cooldowns[19]
    assert state.requests == 1 and state.opportunities == 0 and state.empty_outcomes == 1
    assert state.zero_streak == 0 and state.next_probe_opportunity == 2
    assert search.stats["adaptive_probe_requests"] == 1
    assert not search.shared_discovery.epoch and not search.geometry.full_oracle.calls
    take_fresh_remove(search)
    assert state.opportunities == 1 and not search.probe_tickets


def test_exponential_cooldown_caps_and_periodic_escape_use_actual_source_removals(monkeypatch):
    search = engine(probe_warmup_requests=0, probe_cooldown_max_interval=4)
    monkeypatch.setattr(search.geometry, "probe_fresh_ridge", lambda *args, **kwargs: ())
    request_opportunities = []
    for opportunity in range(1, 13):
        search.removal_clock = 7
        before = search.stats["adaptive_probe_requests"]
        search._probe_action(source_node(), rollout=True)
        if search.stats["adaptive_probe_requests"] != before:
            request_opportunities.append(opportunity)
        take_fresh_remove(search)
    assert request_opportunities == [1, 3, 7, 11]
    assert search.probe_cooldowns[19].opportunities == 12
    assert search.stats["adaptive_probe_cooldown_skips"] == 8
    assert search.probe_cooldowns[19].next_probe_opportunity == 15
    assert search._probe_action(source_node(22), rollout=True) is None
    assert search.probe_cooldowns[22].requests == 1


def test_warmup_is_per_source_and_failed_requests_do_not_reset_normal_sampler(monkeypatch):
    search = engine(probe_warmup_requests=2)
    search.geometry.ridge_next_probe[19] = 9999
    search.geometry.ridge_zero_streak[19] = 9
    monkeypatch.setattr(search.geometry, "probe_fresh_ridge", lambda *args, **kwargs: ())
    for opportunity in range(1, 4):
        search.removal_clock = 7
        search._probe_action(source_node(), rollout=True)
        assert search.probe_cooldowns[19].requests == opportunity
        take_fresh_remove(search)
    assert search.probe_cooldowns[19].next_probe_opportunity == 5
    assert search.geometry.ridge_next_probe[19] == 9999 and search.geometry.ridge_zero_streak[19] == 9


def test_actual_completion_not_new_ridge_generation_decides_probe_feedback():
    search = engine(probe_warmup_requests=0)
    search.removal_clock = 7
    action = search._probe_action(source_node(), rollout=True)
    event = search.probe_events[-1]
    assert action and event["adaptive_outcome"] == "pending"
    assert not search.probe_cooldowns[19].novel_outcomes
    parent = source_node()
    edge, transition = search._edge(parent, action)
    search._step_reward(parent, edge, [action], [], False)
    node = reduced.ReducedNode(transition.state, transition.rank)
    completion = search._filler_actions(node, sample_count=2, rng=random.Random(1))[0]
    terminal_edge, _terminal = search._edge(node, completion)
    reward = search._step_reward(node, terminal_edge, [action, completion], [], False)
    assert event["adaptive_outcome"] == "novel_exit"
    assert search.probe_cooldowns[19].novel_outcomes == 1
    assert search.probe_cooldowns[19].new_reduced_exits == 1
    assert search.probe_cooldowns[19].next_probe_opportunity == 2
    assert reward[0] > 9 and not search.discovery_events


def test_known_completion_settles_failure_once_and_late_novelty_reopens_probe():
    search = engine(probe_warmup_requests=0)
    search.removal_clock = 7
    action = search._probe_action(source_node(), rollout=True)
    event = search.probe_events[-1]
    edge, transition = search._edge(source_node(), action)
    search._step_reward(source_node(), edge, [action], [], False)
    node = reduced.ReducedNode(transition.state, transition.rank)
    completion = search._filler_actions(node, sample_count=2, rng=random.Random(1))[0]
    terminal_edge, _terminal = search._edge(node, completion)
    search.shared_discovery.reduced_keys.add((search.problem.node_id, terminal_edge.terminal.closure_word))
    search._step_reward(node, terminal_edge, [action, completion], [], False)
    state = search.probe_cooldowns[19]
    assert event["adaptive_outcome"] == "zero_exit" and state.zero_streak == 1
    assert state.next_probe_opportunity == 3 and not state.new_reduced_exits
    search._step_reward(node, terminal_edge, [action, completion], [], True)
    assert state.zero_exit_outcomes == 1 and state.zero_streak == 1
    search._settle_probe_outcome(event, novel=True)
    assert state.zero_streak == 0 and state.next_probe_opportunity == 2
    assert state.novel_outcomes == 1 and search.stats["adaptive_probe_late_novel_outcomes"] == 1


def test_cooldown_skips_lp_but_retains_regular_actions_and_ticket(monkeypatch):
    search = engine()
    search.removal_clock = 7
    search._probe_state(19).next_probe_opportunity = 3
    def forbidden(*args, **kwargs):
        raise AssertionError("cooldown must not run a probe LP")
    monkeypatch.setattr(search.geometry, "probe_fresh_ridge", forbidden)
    assert search._generate_actions(source_node(), rollout=True, policy="guided", rng=random.Random(1))
    assert search.stats["adaptive_probe_cooldown_skips"] == 1
    assert search._planned_action(source_node(), rollout=False, rng=random.Random(1)) is None
    assert search.stats["adaptive_probe_cooldown_skips"] == 1 and not search.probe_events


def test_only_actual_fresh_slot_source_removals_advance_cooldown_clock():
    search = engine()
    node = source_node()
    action = reduced.ReducedAction("remove", retained_word=17, sampled_facet_word=19)
    edge, _transition = search._edge(node, action)
    for clock in range(8):
        search.removal_clock = clock
        search._step_reward(node, edge, [action], [], True)
    assert search.probe_cooldowns[19].opportunities == 1
    assert search.stats["adaptive_probe_opportunities"] == 1


def test_diverse_entrance_distance_is_label_independent_and_candidate_window_bounded():
    search = engine(diverse_entry_candidate_window=4)
    register_sources(search)
    search.recent_entry_words.append(19)
    candidates = search._diverse_candidates()
    assert len(candidates) <= 4 and set(candidates) == {19, 22, 15, 25}
    word, count = search._choose_diverse_entry()
    assert count == 4 and word == 15
    for anchor in search.anchors.values():
        anchor.lifted = not anchor.lifted
        anchor.class_id = "different-class"
        anchor.productive = not anchor.productive
    assert search._choose_diverse_entry()[0] == word
    search.anchors[15].starts = 10
    assert 15 not in search._diverse_candidates()


def test_half_fair_continuation_preserves_parent_proof_budget_and_cadence():
    search = engine()
    register_sources(search)
    first = search._iteration_start(4)
    assert first.state.support_word == 19 and first.state.corrections_used == 0
    assert search.episode_prefix == search.anchors[19].witness
    assert search._iteration_start(5) is search.root
    assert search._iteration_start(8).state.support_word == 28
    assert search._iteration_start(12).state.support_word == 22
    assert [row["kind"] for row in search.diverse_entry_events] == ["fair", "diverse", "fair"]
    assert search.stats["continuation_episodes"] == 3 and search.stats["empty_root_episodes"] == 1
    assert search.stats["diversity_first_starts"] == 3
    assert search._correction_limit([]) == search.config.max_corrections


def test_anchor_registration_is_actual_only_and_queue_members_are_unique():
    search = engine()
    register_sources(search, (19, 22))
    register_sources(search, (19, 22))
    assert list(search.unstarted_anchor_queue) == [19, 22]
    assert not search.geometry.full_oracle.calls
    search.anchors[19].starts = search.anchors[22].starts = 1
    assert search._diverse_candidates() == list(search.anchor_queue)


def test_reporting_has_stable_diagnostic_keys_and_keeps_probe_lp_subset():
    search = engine()
    before = search._diagnostic_snapshot()
    search.removal_clock = 7
    search._probe_action(source_node(), rollout=True)
    assert before.keys() == search._diagnostic_snapshot().keys()
    report = search.result()
    assert report["adaptive_exploration"]["selection_uses_lifting_or_catalog_labels"] is False
    assert report["adaptive_exploration"]["sources"][0]["requests"] == 1
    stats = report["geometry"]["statistics"]
    assert stats["ridge_lp_attempts"] == stats["fresh_probe_lp_attempts"]


def test_new_policy_switches_and_report_aggregation_do_not_hide_seed_differences(tmp_path):
    paths = []
    configs = [adaptive.ReducedAdaptiveExplorationConfig(), adaptive.ReducedAdaptiveProbeOnlyConfig()]
    for name, config in zip(("old", "new"), configs):
        output = tmp_path / f"{name}.json"
        output.write_text(json.dumps({
            "coverage": {"discoveries": []}, "reduced_facets": {"discovery_events": []},
            "geometry": {"statistics": {"ridge_lp_attempts": 2}},
            "runtime": {"iterations_completed": 1, "elapsed_seconds": 1},
            "entry_diagnostics": {"windows": []},
            "adaptive_exploration": {"statistics": {"adaptive_probe_cooldown_skips": 3}},
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
    report = comparison.compare(paths[:1], paths[1:], allow_policy_differences=("enable_diverse_continuation",))
    assert report["variant"]["adaptive_exploration_statistics"]["adaptive_probe_cooldown_skips"] == 3
    assert report["variant"]["totals"]["lp_attempts"] == 2
    summary = json.loads(paths[1].read_text(encoding="utf-8"))
    summary["config"]["seed"] += 1
    paths[1].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="parameters"):
        comparison.compare(paths[:1], paths[1:], allow_policy_differences=("enable_diverse_continuation",))


def assert_same_search(left, right):
    assert left.reduced_witnesses == right.reduced_witnesses
    assert left.rng.getstate() == right.rng.getstate()
    assert {key: rng.getstate() for key, rng in left.rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in right.rollout_rngs.items()}
    assert left.nodes.keys() == right.nodes.keys()
    for key, node in left.nodes.items():
        other = right.nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        assert node.edges.keys() == other.edges.keys()
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum, edge.prior) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum, alternative.prior)


external = pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="external handoff")


@external
@pytest.mark.parametrize("probes", [False, True])
def test_disabled_new_switches_replay_v6_or_v8_exact_paths_values_and_rng(probes):
    actual = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    old_config = fresh.ReducedFreshProbeOnlyConfig(iterations=32, seed=17, progress_interval=0,
                                                   enable_fresh_geometry_probes=probes)
    new_config = adaptive.ReducedAdaptiveExplorationConfig(
        iterations=32, seed=17, progress_interval=0, enable_fresh_geometry_probes=probes,
        enable_adaptive_probe_cooldown=False, enable_diverse_continuation=False,
    )
    searches = [cls(actual, config, handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle())
                for cls, config in ((fresh.ReducedFreshGeometryMCTS, old_config),
                                    (adaptive.ReducedAdaptiveExplorationMCTS, new_config))]
    results = [search.run() for search in searches]
    assert results[0]["tree"] == results[1]["tree"]
    assert_same_search(*searches)
    assert searches[0].fresh_probe_rng.getstate() == searches[1].fresh_probe_rng.getstate()


@external
def test_lifting_labels_cannot_change_cooldown_entries_rewards_or_rng():
    actual = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    searches = [adaptive.ReducedAdaptiveExplorationMCTS(
        actual, adaptive.ReducedAdaptiveExplorationConfig(iterations=48, seed=17, progress_interval=0,
                                                         probe_warmup_requests=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(valid),
    ) for valid in (False, True)]
    results = [search.run() for search in searches]
    assert results[0]["tree"] == results[1]["tree"]
    assert_same_search(*searches)
    assert searches[0].probe_cooldowns == searches[1].probe_cooldowns
    assert searches[0].diverse_entry_events == searches[1].diverse_entry_events
    assert searches[0].fresh_probe_rng.getstate() == searches[1].fresh_probe_rng.getstate()
    assert searches[0].stats["adaptive_probe_cooldown_skips"]
    assert searches[0].stats["diversity_selected_entries"]
