from __future__ import annotations

import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest


balanced = importlib.import_module("mcts.422.reduced_balanced_continuation_search")
continuation = importlib.import_module("mcts.422.reduced_continuation_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
catalog = importlib.import_module("mcts.422.catalog_reproduction")


def test_quarter_budget_is_explicit_and_invalid_windows_are_rejected():
    assert balanced.ReducedBalancedContinuationConfig().continuation_interval == 4
    with pytest.raises(ValueError):
        balanced.ReducedBalancedContinuationConfig(diagnostic_window=0)


def test_each_entrance_traverses_the_full_rollout_cycle_without_an_rng():
    engine = object.__new__(balanced.ReducedBalancedContinuationMCTS)
    engine.config = balanced.ReducedBalancedContinuationConfig()
    engine.entry_episodes = Counter()
    engine.entry_policy_counts = defaultdict(Counter)
    actual = defaultdict(list)
    for iteration in range(1, 25):
        engine.episode_origin = SimpleNamespace() if iteration % 4 == 0 else None
        actual[engine.entry_kind].append(engine._rollout_policy(iteration))
    assert actual["empty_root"] == list(reduced.ROLLOUT_CYCLE) * 3
    assert actual["continuation"] == list(reduced.ROLLOUT_CYCLE)
    assert engine.entry_policy_counts["continuation"] == {"guided": 3, "novel": 2, "random": 1}
    assert engine.entry_episodes == {"empty_root": 18, "continuation": 6}


def assert_same_search(first, second, first_result, second_result):
    for key in ("coverage", "tree", "search_stats"):
        assert first_result[key] == second_result[key]
    assert first_result["geometry"] == second_result["geometry"]
    assert first.rng.getstate() == second.rng.getstate()
    assert {key: rng.getstate() for key, rng in first.rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in second.rollout_rngs.items()
    }
    for key, node in first.nodes.items():
        other = second.nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum
            )


external = pytest.mark.skipif(
    not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="4-2-2 handoff is external"
)


@external
def test_global_cycle_and_half_budget_reproduce_previous_continuation_exactly():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    legacy = continuation.ReducedContinuationMCTS(
        problem, continuation.ReducedContinuationConfig(iterations=12, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    alternative = balanced.ReducedBalancedContinuationMCTS(
        problem, balanced.ReducedBalancedContinuationConfig(
            iterations=12, seed=17, progress_interval=0, continuation_interval=2,
            entry_local_rollout_cycle=False,
        ), handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    assert_same_search(legacy, alternative, legacy.run(), alternative.run())


@external
def test_disabled_continuation_still_reproduces_the_original_baseline():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    legacy = reduced.ReducedSubgroupMCTS(
        problem, reduced.ReducedSearchConfig(iterations=12, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    alternative = balanced.ReducedBalancedContinuationMCTS(
        problem, balanced.ReducedBalancedContinuationConfig(
            iterations=12, seed=17, progress_interval=0, enable_local_continuation=False,
        ), handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    assert_same_search(legacy, alternative, legacy.run(), alternative.run())
    assert alternative.entry_episodes == {"empty_root": 12}


@external
def test_observational_diagnostics_do_not_change_paths_values_or_rng():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    engines = [
        balanced.ReducedBalancedContinuationMCTS(
            problem, balanced.ReducedBalancedContinuationConfig(
                iterations=12, seed=17, progress_interval=0, record_entry_diagnostics=enabled,
            ), handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
        ) for enabled in (False, True)
    ]
    results = [engine.run() for engine in engines]
    assert_same_search(*engines, *results)
    assert results[0]["entry_diagnostics"]["entries"] == {}
    assert results[1]["entry_diagnostics"]["entries"]


@external
def test_legacy_policy_hook_preserves_saved_v1_discovery_paths():
    path = Path(reduced.__file__).parent / "runs" / "difficult_nodes" / "continuation_i300_s4222028" / "bfs422-h-1171.json"
    if not path.exists():
        pytest.skip("saved v1 experiment is unavailable")
    saved = json.loads(path.read_text(encoding="utf-8"))
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-1171")
    config = continuation.ReducedContinuationConfig(**{
        **saved["config"], "iterations": 12, "progress_interval": 0,
    })
    engine = continuation.ReducedContinuationMCTS(problem, config, handoff_root=reduced.DEFAULT_HANDOFF_ROOT)
    expected = [
        {key: value for key, value in event.items() if key != "canonical_h_row"}
        for event in saved["coverage"]["discoveries"] if event["iteration"] <= 12
    ]
    assert engine.run()["coverage"]["discoveries"] == expected


@pytest.fixture(scope="module")
def balanced_run():
    if not reduced.DEFAULT_HANDOFF_ROOT.exists():
        pytest.skip("4-2-2 handoff is external")
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-1171")
    engine = balanced.ReducedBalancedContinuationMCTS(
        problem, balanced.ReducedBalancedContinuationConfig(iterations=48, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    result = engine.run()
    for event in result["coverage"]["discoveries"]:
        terminal = engine.geometry.validate(int(event["reduced_facet_word_hex"], 16))
        event["canonical_h_row"] = list(terminal.full_result.canonical_h_row)
    return engine, result


def test_balanced_policy_counts_and_diagnostics_reconcile_with_search(balanced_run):
    engine, result = balanced_run
    assert result["entry_rollout"]["episode_counts"] == {"empty_root": 36, "continuation": 12}
    assert result["entry_rollout"]["policy_episode_counts"] == {
        "empty_root": {"guided": 18, "novel": 12, "random": 6},
        "continuation": {"guided": 6, "novel": 4, "random": 2},
    }
    entries = result["entry_diagnostics"]["entries"]
    assert sum(item["iterations"] for item in entries.values()) == 48
    assert sum(item["new_local_classes"] for item in entries.values()) == len(engine.discovery_events)
    assert sum(item["source_returns"] for item in entries.values()) == engine.stats["corrector_source_returns"]
    assert sum(item["support_lp_attempts"] for item in entries.values()) == engine.geometry.stats["support_lp_attempts"]
    assert sum(item["ridge_lp_attempts"] for item in entries.values()) == engine.geometry.stats["ridge_lp_attempts"]
    assert sum(item["rank_cache_misses"] for item in entries.values()) == engine.geometry.rank_oracle.misses
    assert sum(item["elapsed_seconds"] for item in entries.values()) <= result["runtime"]["elapsed_seconds"]
    assert engine.root.visits == 36
    assert result["entry_diagnostics"]["selection_reads_diagnostics"] is False


def test_balanced_paths_pass_full_and_local_independent_audits(balanced_run):
    engine, result = balanced_run
    events = result["coverage"]["discoveries"]
    assert events
    oracle = continuation.FacetOracle(handoff_root=reduced.DEFAULT_HANDOFF_ROOT)
    full = catalog.audit_node(engine.problem, events, reduced.DEFAULT_HANDOFF_ROOT, oracle)
    local = continuation.audit_local_paths(engine.problem, events, engine.config, reduced.DEFAULT_HANDOFF_ROOT, oracle)
    assert full["failure_count"] == 0, full
    assert local["failure_count"] == 0, local
    for event in events:
        assert event["rollout_policy"] == reduced.ROLLOUT_CYCLE[
            (event["entry_episode_ordinal"] - 1) % len(reduced.ROLLOUT_CYCLE)
        ]
        shared = engine.shared_discovery.discovered_classes[event["class_id"]]
        assert shared["rollout_policy"] == event["rollout_policy"]


def test_diagnostic_window_counts_are_bounded_and_reconcile(balanced_run):
    _engine, result = balanced_run
    windows = result["entry_diagnostics"]["windows"]
    assert len(windows) == 1
    assert windows[0]["start_iteration_exclusive"] == 0
    assert windows[0]["end_iteration_inclusive"] == 48
    assert windows[0]["entries"] == result["entry_diagnostics"]["entries"]
