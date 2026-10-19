from __future__ import annotations

import copy
import importlib
import random
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest


multi = importlib.import_module("mcts.422.reduced_multi_exit_search")
frontier = importlib.import_module("mcts.422.reduced_frontier_search")
novelty = importlib.import_module("mcts.422.reduced_novelty_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
benchmark = importlib.import_module("mcts.422.difficult_node_benchmark")


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


def problem(points):
    points = np.asarray(points, dtype=float)
    count = len(points)
    return reduced.ReducedSubgroupProblem(
        "fixture", "fixture", points.shape[1], points, points,
        tuple(range(count)), tuple(range(count)), tuple(1 << i for i in range(count)),
        count, Path("fixture"),
    )


def pyramid_engine(**options):
    return multi.ReducedMultiExitMCTS(
        problem([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]]),
        multi.ReducedMultiExitConfig(iterations=1, progress_interval=0, **options),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )


def test_defaults_preserve_prior_parameters_and_variant_is_isolated():
    old, new = asdict(frontier.ReducedFrontierConfig()), asdict(multi.ReducedMultiExitConfig())
    assert all(new[key] == value for key, value in old.items())
    assert benchmark.VARIANTS["baseline"][1] is reduced.ReducedSubgroupMCTS
    assert benchmark.VARIANTS["shared_frontier"][1] is frontier.ReducedFrontierMCTS
    assert benchmark.VARIANTS["multi_exit"][1] is multi.ReducedMultiExitMCTS
    with pytest.raises(ValueError):
        multi.ReducedMultiExitConfig(ridge_backoff_max_interval=0)


@pytest.mark.parametrize("points,retained,expected", [
    (list(product([0, 1], repeat=3)), 1, {15, 51, 85}),
    ([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], 16, {19, 22, 25, 28}),
])
def test_polygon_enumerates_all_incident_facets_without_lifting(points, retained, expected):
    oracle = ReportingOracle()
    geometry = multi.MultiExitGeometry(problem(points), handoff_root=Path("fixture"), full_oracle=oracle)
    result = geometry.certified_lower_face_exits(retained)
    assert result.complete
    assert set(result.facet_words) == expected
    assert result.closed_face_word == retained
    assert geometry.certified_lower_face_exits(retained) is result
    assert geometry.stats["lower_exit_computations"] == 1
    assert geometry.stats["lower_exit_cache_hits"] == 1
    assert not oracle.calls


def test_rank_or_svd_failure_is_cached_but_not_a_pruning_claim(monkeypatch):
    engine = pyramid_engine()
    assert not engine.geometry.certified_lower_face_exits(19).complete
    monkeypatch.setattr(np.linalg, "svd", lambda *_args, **_kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError()))
    result = engine.geometry.certified_lower_face_exits(16)
    assert result.reason == "svd_failure"
    assert engine.geometry.certified_lower_face_exits(16) is result
    assert not engine.result()["lower_face_exits"]["exhausted_exits_are_facet_completeness"]


def test_exit_selection_uses_actual_arrivals_not_query_counts_and_keeps_known_routes():
    engine = pyramid_engine()
    node = reduced.ReducedNode(reduced.ReducedState(16, "filler", 1, 19), 0)
    rng = random.Random(8)
    selected = []
    for _ in range(3):
        action = engine._filler_actions(node, sample_count=2, rng=rng)[0]
        assert action.source == "certified_lower_face_exit" and action.add_word.bit_count() == 2
        selected.append(action.sampled_facet_word)
        engine.shared_discovery.reduced_keys.add((engine.problem.node_id, action.sampled_facet_word))
        engine.frontier_traversals[(19, 16, action.sampled_facet_word)] += 1
    assert set(selected) == {22, 25, 28}
    engine.frontier_traversals[(19, 16, 22)] = 10
    engine.frontier_traversals[(19, 16, 25)] = 7
    assert engine._filler_actions(node, sample_count=2, rng=rng)[0].sampled_facet_word == 28
    assert engine.shared_discovery.epoch == 0
    assert not engine.discovered_reduced_facets
    assert not engine.geometry.full_oracle.calls


def test_uncertified_or_wrong_source_uses_original_lp(monkeypatch):
    engine = pyramid_engine()
    calls = []
    monkeypatch.setattr(frontier.ReducedFrontierMCTS, "_filler_actions",
                        lambda self, node, **kwargs: calls.append(node.state.support_word) or [])
    node = reduced.ReducedNode(reduced.ReducedState(16, "filler", 1, 15), 0)
    assert engine._filler_actions(node, sample_count=2, rng=random.Random(1)) == []
    assert calls == [16]
    assert engine.stats["lower_exit_source_mismatches"] == 1


def test_zero_increment_backoff_preserves_periodic_probes_and_resets_on_new_geometry(monkeypatch):
    geometry = pyramid_engine(ridge_backoff_max_interval=8).geometry
    calls = []
    def sample(self, facet_word, **kwargs):
        calls.append(facet_word)
        return (1,) if len(calls) < 6 else (1, 2)
    monkeypatch.setattr(novelty.DeferredLiftGeometry, "sample_ridges", sample)
    for _ in range(30):
        geometry.sample_ridges(19, count=4, rng=random.Random(1))
    assert 6 <= len(calls) < 15
    assert geometry.stats["ridge_backoff_skipped"] > 0
    assert geometry.ridge_sample_seen[19] == {1, 2}
    assert geometry.stats["ridge_backoff_new_ridges"] == 2
    assert geometry.ridge_next_probe[19] <= 38


@pytest.mark.parametrize("tree_edge", [False, True])
def test_actual_exit_reward_matches_v4_and_independent_audit_rejects_corruption(tree_edge):
    engine = pyramid_engine()
    old = novelty.ReducedNoveltyMCTS(
        engine.problem, novelty.ReducedNoveltyConfig(iterations=1, progress_interval=0),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )
    root_add = reduced.ReducedAction("add", add_word=19, sampled_facet_word=19)
    remove = reduced.ReducedAction("remove", retained_word=16, sampled_facet_word=19)
    add = reduced.ReducedAction("add", add_word=6, sampled_facet_word=22, source="certified_lower_face_exit")
    rewards = []
    for search in (engine, old):
        search.current_iteration = search.current_global_iteration = 1
        current, path, values = search.root, [], []
        for action in (root_add, remove, add):
            path.append(action)
            edge, transition = search._edge(current, action)
            values.append(search._step_reward(current, edge, path, [], tree_edge)[0])
            current = reduced.ReducedNode(transition.state, transition.rank)
        rewards.append(values)
    assert rewards[0] == rewards[1]
    assert engine.frontier_traversals[(19, 16, 22)] == 1
    assert engine.stats["lower_exit_new_reduced_arrivals"] == 1
    events = [{"path": [action.to_dict() for action in (root_add, remove, add)]}]
    audit = multi.audit_lower_face_exits(engine.problem, events, Path("fixture"), ReportingOracle())
    assert audit["failure_count"] == 0 and audit["unique_lower_actions_checked"] == 1
    corrupted = copy.deepcopy(events)
    corrupted[0]["path"][-1]["sampled_facet_word_hex"] = "0x19"
    assert multi.audit_lower_face_exits(engine.problem, corrupted, Path("fixture"), ReportingOracle())["failure_count"] == 1


def test_diagnostic_keys_are_stable():
    engine = pyramid_engine()
    before = engine._diagnostic_snapshot()
    node = reduced.ReducedNode(reduced.ReducedState(16, "filler", 1, 19), 0)
    engine._filler_actions(node, sample_count=2, rng=random.Random(1))
    assert before.keys() == engine._diagnostic_snapshot().keys()


external = pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="external handoff")


@external
def test_disabled_features_reproduce_v5_exactly():
    data = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    engines = [frontier.ReducedFrontierMCTS(
        data, frontier.ReducedFrontierConfig(iterations=18, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    ), multi.ReducedMultiExitMCTS(
        data, multi.ReducedMultiExitConfig(iterations=18, seed=17, progress_interval=0,
                                         enable_lower_face_exits=False, enable_ridge_sampling_backoff=False),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    )]
    results = [search.run() for search in engines]
    assert results[0]["tree"] == results[1]["tree"]
    assert engines[0].reduced_witnesses == engines[1].reduced_witnesses
    assert engines[0].rng.getstate() == engines[1].rng.getstate()
    assert {key: rng.getstate() for key, rng in engines[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in engines[1].rollout_rngs.items()}


@external
def test_lift_outcomes_do_not_change_v6_paths_rewards_or_rng():
    data = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    engines = [multi.ReducedMultiExitMCTS(
        data, multi.ReducedMultiExitConfig(iterations=24, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(valid),
    ) for valid in (False, True)]
    results = [search.run() for search in engines]
    assert results[0]["tree"] == results[1]["tree"]
    for key, value in results[0]["lower_face_exits"].items():
        if key != "statistics":
            assert value == results[1]["lower_face_exits"][key]
    for key, value in results[0]["lower_face_exits"]["statistics"].items():
        if "class_discoveries" not in key:
            assert value == results[1]["lower_face_exits"]["statistics"][key]
    assert engines[0].reduced_witnesses == engines[1].reduced_witnesses
    assert engines[0].rng.getstate() == engines[1].rng.getstate()
    assert {key: rng.getstate() for key, rng in engines[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in engines[1].rollout_rngs.items()}
    for key, node in engines[0].nodes.items():
        assert (node.visits, node.value_sum) == (engines[1].nodes[key].visits, engines[1].nodes[key].value_sum)
        for action_id, edge in node.edges.items():
            other = engines[1].nodes[key].edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum, edge.prior) == (
                other.visits, other.value_sum, other.epoch_value_sum, other.prior)
