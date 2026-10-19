from __future__ import annotations

import copy
import importlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


novelty = importlib.import_module("mcts.422.reduced_novelty_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
ridge = importlib.import_module("mcts.422.reduced_ridge_exit_search")
continuation = importlib.import_module("mcts.422.reduced_continuation_search")
catalog = importlib.import_module("mcts.422.catalog_reproduction")
benchmark = importlib.import_module("mcts.422.difficult_node_benchmark")


class ReportingOracle:
    def __init__(self, valid, *, unique_ids=False):
        self.valid = valid
        self.unique_ids = unique_ids
        self.calls = []

    def validate(self, word):
        self.calls.append(word)
        return reduced.FacetResult(
            self.valid, "fixture" if self.valid else "rank", word,
            support_word=word, affine_rank=79 if self.valid else -1,
            canonical_h_row=((word, 0) if self.unique_ids else (1, 0)) if self.valid else None,
            class_id=(f"fixture-{word:x}" if self.unique_ids else "fixture-class") if self.valid else None,
        )

    def to_dict(self):
        return {"validation_count": len(self.calls)}


def square_problem():
    points = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return reduced.ReducedSubgroupProblem(
        "square", "fixture", 2, points, points, (0, 1, 2, 3), (0, 1, 2, 3),
        (1, 2, 4, 8), 4, Path("fixture"),
    )


@pytest.mark.parametrize("field", ["new_class_reward", "known_class_reward", "new_lifted_facet_reward", "failed_lift_reward"])
def test_full_space_reward_fields_must_be_zero(field):
    with pytest.raises(ValueError):
        novelty.ReducedNoveltyConfig(**{field: 1})


def test_reduced_discoveries_not_class_labels_advance_epoch():
    state = novelty.ReducedNoveltyDiscovery()
    assert state.observe("one", {"class_id": "one"}) == (True, 0)
    assert state.epoch == 0
    assert state.observe_reduced("first-node", 3)
    assert state.epoch == 1
    assert not state.observe_reduced("first-node", 3)
    assert state.observe_reduced("second-node", 3)
    assert state.epoch == 2
    assert state.observe("two", {"class_id": "two"}) == (True, 0)
    assert state.observe("one", None) == (False, 1)
    assert state.epoch == 2


def test_candidate_exit_validation_never_calls_full_oracle_and_taken_lifting_is_cached():
    oracle = ReportingOracle(True)
    geometry = novelty.DeferredLiftGeometry(square_problem(), handoff_root=Path("fixture"), full_oracle=oracle)
    assert geometry.certified_ridge_exits(1).complete
    assert not oracle.calls
    terminal = geometry.validate(3)
    assert terminal.full_result is None
    assert geometry.lift_taken_terminal(terminal).lifted
    assert geometry.lift_taken_terminal(geometry.validate(3)).lifted
    assert oracle.calls == [3]
    assert geometry.stats["taken_lift_checks"] == 1
    assert geometry.stats["taken_lift_cache_hits"] == 1
    assert geometry.validate(3).full_result is None


def test_actual_reduced_novelty_reward_and_prior_do_not_depend_on_full_validity():
    values = []
    for valid in (False, True):
        engine = novelty.ReducedNoveltyMCTS(
            square_problem(), novelty.ReducedNoveltyConfig(iterations=1, seed=7, progress_interval=0),
            handoff_root=Path("fixture"), full_oracle=ReportingOracle(valid),
        )
        engine.current_iteration = 1
        engine.current_global_iteration = 1
        action = reduced.ReducedAction("add", add_word=3, sampled_facet_word=3)
        edge, _transition = engine._edge(engine.root, action)
        first, _new_class = engine._step_reward(engine.root, edge, [action], [], True)
        second, _new_class = engine._step_reward(engine.root, edge, [action], [], True)
        values.append((first, second, edge.prior, engine.shared_discovery.epoch))
        assert first == pytest.approx(10 + 2 * 0.045 + 0.18)
        assert second == pytest.approx(2 * 0.045 - 0.02 / np.sqrt(2))
        assert edge.terminal.full_result is None
        assert not engine.anchors[3].productive
        assert engine._correction_limit([action]) == 5
    assert values[0] == values[1]


def test_repeated_global_reduced_closure_is_not_rewarded_again_in_another_tree():
    state = novelty.ReducedNoveltyDiscovery()
    first_rewards = []
    for _ in range(2):
        engine = novelty.ReducedNoveltyMCTS(
            square_problem(), novelty.ReducedNoveltyConfig(iterations=1, seed=7, progress_interval=0),
            handoff_root=Path("fixture"), full_oracle=ReportingOracle(False), shared_discovery=state,
        )
        action = reduced.ReducedAction("add", add_word=3)
        edge, _transition = engine._edge(engine.root, action)
        first_rewards.append(engine._step_reward(engine.root, edge, [action], [], True)[0])
    assert first_rewards[0] - first_rewards[1] == 10
    assert state.epoch == 1


external = pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="4-2-2 handoff is external")


@external
@pytest.mark.parametrize("unique_ids", [False, True])
def test_changing_all_lifts_and_class_labels_does_not_change_search_paths_values_or_rng(unique_ids):
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    engines = [novelty.ReducedNoveltyMCTS(
        problem, novelty.ReducedNoveltyConfig(iterations=18, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(valid, unique_ids=unique_ids),
    ) for valid in (False, True)]
    results = [engine.run() for engine in engines]
    assert results[0]["tree"] == results[1]["tree"]
    assert engines[0].rng.getstate() == engines[1].rng.getstate()
    assert {key: rng.getstate() for key, rng in engines[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in engines[1].rollout_rngs.items()
    }
    assert engines[0].reduced_witnesses == engines[1].reduced_witnesses
    assert engines[0].reduced_productive_anchors == engines[1].reduced_productive_anchors
    assert engines[0].reduced_productive_basins == engines[1].reduced_productive_basins
    assert engines[0].shared_discovery.epoch == engines[1].shared_discovery.epoch
    for key, node in engines[0].nodes.items():
        other = engines[1].nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum, edge.prior) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum, alternative.prior
            )


@pytest.fixture(scope="module")
def real_run():
    if not reduced.DEFAULT_HANDOFF_ROOT.exists():
        pytest.skip("4-2-2 handoff is external")
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-1171")
    engine = novelty.ReducedNoveltyMCTS(
        problem, novelty.ReducedNoveltyConfig(iterations=36, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    result = engine.run()
    for event in result["coverage"]["discoveries"]:
        event["canonical_h_row"] = list(engine.geometry.lift_cache[
            int(event["reduced_facet_word_hex"], 16)
        ].full_result.canonical_h_row)
    return engine, result


def test_real_reduced_mode_lifts_every_visited_facet_once_and_audits_every_witness(real_run):
    engine, result = real_run
    assert set(engine.anchors) == set(engine.geometry.lift_cache) == engine.discovered_reduced_facets
    assert engine.geometry.stats["taken_lift_checks"] == len(engine.discovered_reduced_facets)
    assert engine.shared_discovery.epoch == len(engine.discovered_reduced_facets)
    assert engine.stats["new_rewarded_reduced_facets"] == len(engine.discovered_reduced_facets)
    assert result["reduced_novelty"]["candidate_lifting"] is False
    assert result["features"]["dynamic_global_new_class_reward"] is False
    oracle = continuation.FacetOracle(handoff_root=reduced.DEFAULT_HANDOFF_ROOT)
    events = result["coverage"]["discoveries"]
    assert events
    assert catalog.audit_node(engine.problem, events, reduced.DEFAULT_HANDOFF_ROOT, oracle)["failure_count"] == 0
    assert continuation.audit_local_paths(engine.problem, events, engine.config,
                                         reduced.DEFAULT_HANDOFF_ROOT, oracle)["failure_count"] == 0
    witnesses = result["reduced_facets"]["witnesses"]
    audit = novelty.audit_reduced_witnesses(engine.problem, witnesses, engine.config, reduced.DEFAULT_HANDOFF_ROOT)
    assert audit["failure_count"] == 0, audit
    assert audit["unique_closures_checked"] == len(engine.discovered_reduced_facets)
    assert ridge.audit_ridge_exits(engine.problem, witnesses, reduced.DEFAULT_HANDOFF_ROOT, oracle)["failure_count"] == 0
    entries = result["entry_diagnostics"]["entries"]
    assert sum(entry["new_reduced_facets"] for entry in entries.values()) == len(engine.discovered_reduced_facets)


def test_reduced_witness_audit_rejects_corrupted_terminal_and_budget(real_run):
    engine, result = real_run
    for field, value in (("reduced_facet_word_hex", "0x1"), ("local_correction_limit", -1)):
        event = copy.deepcopy(result["reduced_facets"]["witnesses"][0])
        event[field] = value
        assert novelty.audit_reduced_witnesses(engine.problem, [event], engine.config,
                                              reduced.DEFAULT_HANDOFF_ROOT)["failure_count"] == 1


def test_reference_accepts_only_the_intended_reward_changes(tmp_path):
    import json
    reference = {"state": "COMPLETED", "selected_node_ids": list(benchmark.DIFFICULT_NODES),
                 "config": asdict(ridge.ReducedRidgeExitConfig(iterations=300, seed=7))}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    config = novelty.ReducedNoveltyConfig(iterations=300, seed=7)
    assert benchmark.load_matched_reference(path, config) == reference
    reference["config"]["new_class_reward"] = 99
    path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.load_matched_reference(path, config)


def test_shared_class_epoch_state_is_rejected():
    with pytest.raises(ValueError):
        novelty.ReducedNoveltyMCTS(square_problem(), novelty.ReducedNoveltyConfig(),
                                 handoff_root=Path("fixture"), shared_discovery=reduced.SharedDiscoveryState())
