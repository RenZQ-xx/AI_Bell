from __future__ import annotations

import copy
import importlib
from collections import Counter, deque
from types import SimpleNamespace

import pytest


continuation = importlib.import_module("mcts.422.reduced_continuation_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
catalog = importlib.import_module("mcts.422.catalog_reproduction")


def bare_engine():
    engine = object.__new__(continuation.ReducedContinuationMCTS)
    engine.config = continuation.ReducedContinuationConfig()
    engine.anchors = {}
    engine.anchor_queue = deque()
    engine.episode_origin = None
    engine.episode_prefix = ()
    engine.current_iteration = 1
    engine.stats = Counter()
    engine.problem = SimpleNamespace(facet_rank=4)
    engine.root = reduced.ReducedNode(reduced.ReducedState(0), -1)
    engine.nodes = {engine.root.state.key: engine.root}
    return engine


def test_interval_preserves_root_exploration():
    with pytest.raises(ValueError):
        continuation.ReducedContinuationConfig(continuation_interval=1)


def test_failed_lift_anchor_is_retained_and_reset_without_replay():
    engine = bare_engine()
    proof = (reduced.ReducedAction("add", add_word=31),)
    facet = reduced.ReducedFacetResult(True, "ok", 31, 31)
    engine._register_anchor(facet, proof, False)
    assert engine._iteration_start(1) is engine.root
    node = engine._iteration_start(2)
    assert node.state == reduced.ReducedState(31, "corrector", 0, 31)
    assert node.visits == 0
    assert engine.root.visits == 0
    assert engine.episode_prefix == proof
    assert engine._correction_limit([]) == 5
    assert engine.stats["failed_lift_bridge_episodes"] == 1
    assert engine._iteration_start(4) is node
    engine.episode_origin.productive = True
    assert engine._correction_limit([]) == 7


def test_anchor_round_robin_preserves_concrete_facets_and_shortest_proof():
    engine = bare_engine()
    long_proof = (reduced.ReducedAction("remove", retained_word=3),) * 3
    short_proof = (reduced.ReducedAction("add", add_word=31),)
    for word in (31, 63, 127):
        engine._register_anchor(reduced.ReducedFacetResult(True, "ok", word, word), long_proof, False)
    engine._register_anchor(reduced.ReducedFacetResult(True, "ok", 31, 31), short_proof, True)
    assert list(engine.anchor_queue) == [31, 63, 127]
    assert [engine._iteration_start(i).state.support_word for i in (2, 4, 6, 8)] == [31, 63, 127, 31]
    assert engine.anchors[31].witness == short_proof
    assert engine.anchors[31].productive is True


@pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="4-2-2 handoff is external")
def test_disabled_continuation_matches_baseline_paths_values_and_rng():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    base = reduced.ReducedSubgroupMCTS(
        problem, reduced.ReducedSearchConfig(iterations=12, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    variant = continuation.ReducedContinuationMCTS(
        problem, continuation.ReducedContinuationConfig(
            iterations=12, seed=17, progress_interval=0, enable_local_continuation=False
        ), handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    first, second = base.run(), variant.run()
    for key in ("coverage", "tree", "search_stats"):
        assert first[key] == second[key]
    assert base.rng.getstate() == variant.rng.getstate()
    assert {key: rng.getstate() for key, rng in base.rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in variant.rollout_rngs.items()
    }
    for key, node in base.nodes.items():
        other = variant.nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum
            )
    assert not variant.anchors


@pytest.fixture(scope="module")
def observed_run():
    if not reduced.DEFAULT_HANDOFF_ROOT.exists():
        pytest.skip("4-2-2 handoff is external")
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-1171")
    engine = continuation.ReducedContinuationMCTS(
        problem, continuation.ReducedContinuationConfig(iterations=60, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    result = engine.run()
    for event in result["coverage"]["discoveries"]:
        terminal = engine.geometry.validate(int(event["reduced_facet_word_hex"], 16))
        event["canonical_h_row"] = list(terminal.full_result.canonical_h_row)
    return engine, result


def test_only_taken_terminals_register_anchors_and_history_is_not_rewarded(observed_run):
    engine, result = observed_run
    assert set(engine.anchors) == engine.discovered_reduced_facets
    assert result["search_stats"]["continuation_episodes"] == 30
    assert engine.root.visits == 30
    assert result["search_stats"]["failed_lift_bridge_episodes"] > 0
    continued = [e for e in result["coverage"]["discoveries"] if e["episode_kind"] == "continuation"]
    assert continued
    for event in continued:
        assert event["root_basin_word_hex"] == "0x0"
        assert event["path"][event["historical_path_length"]:] == event["local_path"]
        assert event["local_corrections"] <= 7
        shared = engine.shared_discovery.discovered_classes[event["class_id"]]
        assert shared["depth"] == event["depth"]
        assert len(shared["path_action_ids"]) == event["depth"]


def test_full_witness_and_independent_local_audits_pass(observed_run):
    engine, result = observed_run
    events = result["coverage"]["discoveries"]
    oracle = continuation.FacetOracle(handoff_root=reduced.DEFAULT_HANDOFF_ROOT)
    full_audit = catalog.audit_node(engine.problem, events, reduced.DEFAULT_HANDOFF_ROOT, oracle)
    assert full_audit["failure_count"] == 0, full_audit
    local_audit = continuation.audit_local_paths(
        engine.problem, events, engine.config, reduced.DEFAULT_HANDOFF_ROOT, oracle
    )
    assert local_audit["failure_count"] == 0, local_audit
    assert local_audit["episode_discoveries"]["continuation"] > 0


@pytest.mark.parametrize("corruption", ["origin", "local_path", "local_budget"])
def test_local_audit_rejects_corrupted_boundary_and_budget(observed_run, corruption):
    engine, result = observed_run
    event = copy.deepcopy(next(
        e for e in result["coverage"]["discoveries"] if e["episode_kind"] == "continuation"
    ))
    if corruption == "origin":
        event["continuation_origin"]["word_hex"] = "0x1"
    elif corruption == "local_path":
        event["local_path"] = []
    else:
        event["local_correction_limit"] = 0
    audit = continuation.audit_local_paths(
        engine.problem, [event], engine.config, reduced.DEFAULT_HANDOFF_ROOT,
        continuation.FacetOracle(handoff_root=reduced.DEFAULT_HANDOFF_ROOT),
    )
    assert audit["failure_count"] == 1
