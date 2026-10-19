from __future__ import annotations

import importlib
import copy
import itertools
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ridge = importlib.import_module("mcts.422.reduced_ridge_exit_search")
balanced = importlib.import_module("mcts.422.reduced_balanced_continuation_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
continuation = importlib.import_module("mcts.422.reduced_continuation_search")
catalog = importlib.import_module("mcts.422.catalog_reproduction")


class FailedLiftOracle:
    def __init__(self):
        self.calls = []

    def validate(self, word):
        self.calls.append(word)
        return reduced.FacetResult(False, "rank", word)


def synthetic_geometry(points, redundant=()):
    points = np.asarray(points, dtype=float)
    all_points = np.concatenate((points, np.asarray(redundant).reshape(-1, points.shape[1])))
    problem = reduced.ReducedSubgroupProblem(
        "synthetic", "fixture", points.shape[1], all_points, points,
        tuple(range(len(points))), tuple(range(len(points))) + (None,) * len(redundant),
        tuple(1 << i for i in range(len(all_points))), len(all_points), Path("fixture"),
    )
    oracle = FailedLiftOracle()
    return ridge.RidgeExitGeometry(problem, handoff_root=Path("fixture"), full_oracle=oracle), oracle


@pytest.mark.parametrize("word,expected", [(1, {3, 9}), (4, {6, 12})])
def test_square_has_two_certified_exits_without_requiring_a_successful_lift(word, expected):
    geometry, oracle = synthetic_geometry([[0, 0], [1, 0], [1, 1], [0, 1]])
    result = geometry.certified_ridge_exits(word)
    assert result.complete
    assert set(result.facet_words) == expected
    assert result.closed_ridge_word == word
    assert result.parameter_interval[0] < result.parameter_interval[1]
    assert len(oracle.calls) == 2
    assert all(not geometry.validate(facet).lifted for facet in result.facet_words)
    assert geometry.stats["support_lp_attempts"] == 0


def test_cube_edge_has_two_exits_but_diagonals_are_not_ridges():
    geometry, _oracle = synthetic_geometry(list(itertools.product((0, 1), repeat=3)))
    result = geometry.certified_ridge_exits(3)
    assert result.complete
    assert set(result.facet_words) == {15, 51}
    assert result.closed_ridge_word == 3
    assert not geometry.certified_ridge_exits(65).complete  # Face diagonal.
    assert not geometry.certified_ridge_exits(129).complete  # Body diagonal.
    assert geometry.certified_ridge_exits(15).reason == "not_ridge_rank"


def test_translated_anisotropic_rectangle_has_the_same_exits():
    points = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]]) * [100, 0.01] + [15, -3]
    geometry, _oracle = synthetic_geometry(points)
    assert set(geometry.certified_ridge_exits(1).facet_words) == {3, 9}


def test_redundant_tight_points_are_included_in_lifting():
    geometry, oracle = synthetic_geometry([[0, 0], [1, 0], [1, 1], [0, 1]], [[0.5, 0]])
    assert geometry.certified_ridge_exits(1).complete
    bottom = geometry.validate(3)
    assert bottom.tight_reduced_ids == (0, 1, 4)
    assert bottom.original_support_word == 19
    assert 19 in oracle.calls


def test_independent_adjacency_audit_rejects_a_false_opposite_exit():
    geometry, oracle = synthetic_geometry([[0, 0], [1, 0], [1, 1], [0, 1]])
    path = [
        reduced.ReducedAction("add", add_word=3).to_dict(),
        reduced.ReducedAction("remove", retained_word=1).to_dict(),
        reduced.ReducedAction("add", add_word=8, sampled_facet_word=9,
                              source="deterministic_ridge_exit").to_dict(),
    ]
    event = {"class_id": "fixture", "path": path}
    audit = ridge.audit_ridge_exits(geometry.problem, [event], Path("fixture"), oracle)
    assert audit["failure_count"] == 0
    assert audit["unique_ridge_actions_checked"] == 1
    corrupted = copy.deepcopy(event)
    corrupted["path"][-1]["sampled_facet_word_hex"] = "0x3"
    assert ridge.audit_ridge_exits(geometry.problem, [corrupted], Path("fixture"), oracle)["failure_count"] == 1


def test_complete_and_failed_certificates_are_cached_without_more_lifting():
    geometry, oracle = synthetic_geometry(list(itertools.product((0, 1), repeat=3)))
    for word in (3, 65):
        first = geometry.certified_ridge_exits(word)
        calls = len(oracle.calls)
        assert geometry.certified_ridge_exits(word) is first
        assert len(oracle.calls) == calls
    assert geometry.stats["deterministic_ridge_computations"] == 2
    assert geometry.stats["deterministic_ridge_cache_hits"] == 2
    summary = geometry.exit_cache_summary()
    assert summary["complete_pairs"] == 1
    assert summary["failed_results_are_pruning"] is False


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_tolerances_are_rejected(value):
    with pytest.raises(ValueError):
        ridge.ReducedRidgeExitConfig(ridge_slope_tolerance=value)
    with pytest.raises(ValueError):
        ridge.ReducedRidgeExitConfig(ridge_interval_tolerance=value)


@pytest.mark.parametrize("case", ["complete", "incomplete", "source_mismatch", "lower_face", "disabled"])
def test_only_a_complete_source_matched_ridge_replaces_the_lp(monkeypatch, case):
    engine = object.__new__(ridge.ReducedRidgeExitMCTS)
    engine.config = ridge.ReducedRidgeExitConfig(enable_deterministic_ridge_exits=case != "disabled")
    engine.problem = SimpleNamespace(ridge_rank=1)
    engine.stats = Counter()
    engine._independent_bundle = lambda *_args, **_kwargs: 8
    queries, fallback = [], []

    def certificate(word):
        queries.append(word)
        return ridge.CertifiedRidgeExits("collapsed_interval") if case == "incomplete" else (
            ridge.CertifiedRidgeExits("complete", (7, 11), 3, (0, 1))
        )

    engine.geometry = SimpleNamespace(certified_ridge_exits=certificate)

    def random_lp(*_args, **_kwargs):
        fallback.append(True)
        return ["fallback"]

    monkeypatch.setattr(balanced.ReducedBalancedContinuationMCTS, "_filler_actions", random_lp)
    node = reduced.ReducedNode(
        reduced.ReducedState(3, "filler", 1, 15 if case == "source_mismatch" else 7),
        0 if case == "lower_face" else 1,
    )
    actions = engine._filler_actions(node, sample_count=6, rng=random.Random(0))
    if case == "complete":
        assert not fallback
        assert actions == [reduced.ReducedAction("add", add_word=8, sampled_facet_word=11,
                                                source="deterministic_ridge_exit")]
    else:
        assert actions == ["fallback"]
        assert len(fallback) == 1
    assert bool(queries) == (case not in {"lower_face", "disabled"})


external = pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="4-2-2 handoff is external")


@external
def test_disabled_ridge_solver_reproduces_balanced_search_and_rng():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    engines = [
        cls(problem, config(iterations=12, seed=17, progress_interval=0, **extra),
            handoff_root=reduced.DEFAULT_HANDOFF_ROOT)
        for cls, config, extra in (
            (balanced.ReducedBalancedContinuationMCTS, balanced.ReducedBalancedContinuationConfig, {}),
            (ridge.ReducedRidgeExitMCTS, ridge.ReducedRidgeExitConfig, {"enable_deterministic_ridge_exits": False}),
        )
    ]
    results = [engine.run() for engine in engines]
    for key in ("coverage", "tree", "geometry", "search_stats"):
        assert results[0][key] == results[1][key]
    assert engines[0].rng.getstate() == engines[1].rng.getstate()
    assert {key: rng.getstate() for key, rng in engines[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in engines[1].rollout_rngs.items()
    }
    for key, node in engines[0].nodes.items():
        other = engines[1].nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)


@external
def test_real_ridge_search_counts_only_taken_terminals_and_passes_path_audits():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-1171")
    engine = ridge.ReducedRidgeExitMCTS(
        problem, ridge.ReducedRidgeExitConfig(iterations=36, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    assert engine.geometry.shared_full_oracle is False
    result = engine.run()
    assert engine.stats["ridge_exit_actions_generated"] > 0
    assert set(engine.anchors) == engine.discovered_reduced_facets
    events = result["coverage"]["discoveries"]
    assert events
    for event in events:
        terminal = engine.geometry.validate(int(event["reduced_facet_word_hex"], 16))
        event["canonical_h_row"] = list(terminal.full_result.canonical_h_row)
        local = event.get("local_path", event["path"])
        assert event["deterministic_ridge_actions_in_local_path"] == sum(
            action["source"] == "deterministic_ridge_exit" for action in local
        )
        assert event["terminal_completion_source"] == local[-1]["source"]
        recorded = engine.shared_discovery.discovered_classes[event["class_id"]]
        assert recorded["terminal_completion_source"] == event["terminal_completion_source"]
    oracle = continuation.FacetOracle(handoff_root=reduced.DEFAULT_HANDOFF_ROOT)
    assert catalog.audit_node(problem, events, reduced.DEFAULT_HANDOFF_ROOT, oracle)["failure_count"] == 0
    assert continuation.audit_local_paths(problem, events, engine.config,
                                         reduced.DEFAULT_HANDOFF_ROOT, oracle)["failure_count"] == 0
    audit = ridge.audit_ridge_exits(problem, events, reduced.DEFAULT_HANDOFF_ROOT, oracle)
    assert audit["failure_count"] == 0, audit
    assert audit["ridge_action_occurrences"] > 0
    entries = result["entry_diagnostics"]["entries"]
    assert sum(entry["deterministic_ridge_queries"] for entry in entries.values()) == (
        engine.geometry.stats["deterministic_ridge_queries"]
    )
    assert result["ridge_exits"]["candidate_validation_is_discovery"] is False
