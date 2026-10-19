from __future__ import annotations

import importlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


frontier = importlib.import_module("mcts.422.reduced_frontier_search")
novelty = importlib.import_module("mcts.422.reduced_novelty_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
benchmark = importlib.import_module("mcts.422.difficult_node_benchmark")
comparison = importlib.import_module("mcts.422.compare_frontier_benchmarks")


class ReportingOracle:
    def __init__(self, valid=False):
        self.valid = valid
        self.calls = []

    def validate(self, word):
        self.calls.append(word)
        return reduced.FacetResult(
            self.valid, "fixture" if self.valid else "rank", word, support_word=word,
            affine_rank=79 if self.valid else -1,
            canonical_h_row=(word, 0) if self.valid else None,
            class_id=f"fixture-{word:x}" if self.valid else None,
        )

    def to_dict(self):
        return {"validation_count": len(self.calls)}


def square_problem():
    points = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return reduced.ReducedSubgroupProblem(
        "square", "fixture", 2, points, points, (0, 1, 2, 3), (0, 1, 2, 3),
        (1, 2, 4, 8), 4, Path("fixture"),
    )


def square_engine(**options):
    return frontier.ReducedFrontierMCTS(
        square_problem(), frontier.ReducedFrontierConfig(iterations=1, progress_interval=0, **options),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )


def test_refresh_interval_is_positive_and_default_baseline_is_unchanged():
    with pytest.raises(ValueError):
        frontier.ReducedFrontierConfig(frontier_refresh_interval=0)
    assert benchmark.VARIANTS["baseline"][1] is reduced.ReducedSubgroupMCTS
    assert benchmark.VARIANTS["shared_frontier"][1] is frontier.ReducedFrontierMCTS
    old = asdict(novelty.ReducedNoveltyConfig())
    new = asdict(frontier.ReducedFrontierConfig())
    assert all(new[key] == value for key, value in old.items())


def test_diagnostic_keys_are_stable_before_and_after_first_request(monkeypatch):
    engine = square_engine(corrector_face_bank_limit=2)
    before = engine._diagnostic_snapshot()
    monkeypatch.setattr(engine.geometry, "sample_ridges", lambda *_args, **_kwargs: (1, 2))
    node = reduced.ReducedNode(reduced.ReducedState(3, "corrector"), 1)
    engine._corrector_actions(node, count=4, rng=random.Random(1), persistent=False)
    after = engine._diagnostic_snapshot()
    assert before.keys() == after.keys()
    assert after["frontier_rollout_requests"] - before["frontier_rollout_requests"] == 1


def test_tree_and_rollout_share_bounded_banks_with_periodic_fresh_sampling(monkeypatch):
    engine = square_engine(corrector_face_bank_limit=2)
    node = reduced.ReducedNode(reduced.ReducedState(3, "corrector"), 1)
    calls = []

    def sample(word, *, count, rng):
        calls.append((word, count))
        return (1, 2)

    monkeypatch.setattr(engine.geometry, "sample_ridges", sample)
    for request in range(1, 9):
        actions = engine._corrector_actions(node, count=4, rng=random.Random(request),
                                            persistent=request == 1)
        assert actions
        assert len(engine.corrector_ridge_bank[3]) <= 2
    assert calls == [(3, 2), (3, 4)]
    assert engine.stats["frontier_ridge_sampling_skipped"] == 6
    assert engine.stats["frontier_tree_requests"] == 1
    assert engine.stats["frontier_rollout_requests"] == 7
    assert not engine.discovered_reduced_facets
    assert not engine.geometry.full_oracle.calls


def test_refresh_never_evicts_untried_faces_and_known_routes_are_not_pruned():
    engine = square_engine()
    bank = [1, 2]
    assert not engine._bank_face(bank, 4, 3, 2, refresh=True)
    assert bank == [1, 2]
    engine.frontier_attempts[(3, 2)] = 8
    assert engine._bank_face(bank, 4, 3, 2, refresh=True)
    assert bank == [1, 4]
    engine.frontier_attempts[(3, 1)] = 100
    engine.frontier_attempts[(3, 4)] = 100
    assert all(np.isfinite(engine._frontier_key(3, word, random.Random(1))[1]) for word in bank)
    assert engine.result()["face_frontier"]["known_exits_remain_traversable"]
    assert not engine.result()["face_frontier"]["exhausted_bank_is_facet_completeness"]


def test_cached_opposite_exit_remains_lookahead_until_the_action_is_taken():
    engine = square_engine()
    assert engine.geometry.certified_ridge_exits(1).complete
    node = reduced.ReducedNode(reduced.ReducedState(1, "filler", 1, 3), 0)
    actions = engine._filler_actions(node, sample_count=2, rng=random.Random(7))
    assert len(actions) == 1
    assert actions[0].source == "deterministic_ridge_exit"
    assert actions[0].sampled_facet_word == 9
    assert engine.stats["frontier_certified_exit_reuses"] == 1
    assert not engine.frontier_traversals
    assert not engine.discovered_reduced_facets
    assert not engine.shared_discovery.epoch
    assert not engine.geometry.full_oracle.calls


@pytest.mark.parametrize("tree_edge", [False, True])
def test_actual_attempts_and_exit_repeats_are_recorded_without_changing_reward(tree_edge):
    engines = [square_engine(), novelty.ReducedNoveltyMCTS(
        square_problem(), novelty.ReducedNoveltyConfig(iterations=1, progress_interval=0),
        handoff_root=Path("fixture"), full_oracle=ReportingOracle(),
    )]
    outcomes = []
    for engine in engines:
        engine.current_iteration = engine.current_global_iteration = 1
        root_add = reduced.ReducedAction("add", add_word=3, sampled_facet_word=3)
        edge, transition = engine._edge(engine.root, root_add)
        engine._step_reward(engine.root, edge, [root_add], [], tree_edge)
        source = reduced.ReducedNode(transition.state, transition.rank)
        remove = reduced.ReducedAction("remove", retained_word=1, sampled_facet_word=3,
                                       source="reduced_facet_ridge")
        edge, transition = engine._edge(source, remove)
        values = [engine._step_reward(source, edge, [root_add, remove], [], tree_edge)[0]]
        retained = reduced.ReducedNode(transition.state, transition.rank)
        add = reduced.ReducedAction("add", add_word=8, sampled_facet_word=9,
                                   source="deterministic_ridge_exit")
        edge, _transition = engine._edge(retained, add)
        for _ in range(2):
            values.append(engine._step_reward(retained, edge, [root_add, remove, add], [], tree_edge)[0])
        outcomes.append(values)
    assert outcomes[0] == outcomes[1]
    engine = engines[0]
    assert engine.frontier_attempts[(3, 1)] == 1
    assert engine.frontier_exits[(3, 1)] == {9}
    assert engine.frontier_novel_exits[(3, 1)] == {9}
    assert engine.frontier_traversals[(3, 1, 9)] == 2
    assert engine.stats["frontier_repeated_exit_arrivals"] == 1
    assert engine.result()["face_frontier"]["repeat_exit_arrival_fraction"] == 0.5


def test_reference_preserves_all_v4_parameters(tmp_path):
    reference = {"state": "COMPLETED", "selected_node_ids": list(benchmark.DIFFICULT_NODES),
                 "config": asdict(novelty.ReducedNoveltyConfig(iterations=300, seed=7))}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    config = frontier.ReducedFrontierConfig(iterations=300, seed=7)
    assert benchmark.load_matched_reference(path, config) == reference
    reference["config"]["corrector_candidates"] += 1
    path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.load_matched_reference(path, config)


def test_comparison_preserves_exact_normals_and_rejects_parameter_changes(tmp_path):
    big = 2 ** 64 + 1
    paths = []
    for variant, config in (("reduced_novelty", novelty.ReducedNoveltyConfig()),
                            ("shared_frontier", frontier.ReducedFrontierConfig())):
        output = tmp_path / f"{variant}.json"
        output.write_text(json.dumps({
            "coverage": {"discoveries": [{"iteration": 1, "canonical_h_row": [big, 1],
                                           "catalog_index": 0}]},
            "reduced_facets": {"discovery_events": [{"reduced_facet_word_hex": "0x3"}]},
            "geometry": {"statistics": {"ridge_lp_attempts": 5, "support_lp_attempts": 5}},
            "runtime": {"iterations_completed": 1, "elapsed_seconds": 1},
            "entry_diagnostics": {"windows": []},
        }), encoding="utf-8")
        summary_path = tmp_path / f"{variant}.summary.json"
        summary_path.write_text(json.dumps({
            "state": "COMPLETED", "selected_node_ids": ["square"], "config": asdict(config),
            "elapsed_seconds": 2, "runs": [{"node_id": "square", "output": str(output),
                "audit": {"failure_count": 0, "elapsed_seconds": 1,
                          "reduced_witness_audit": {"paths_checked": 1}}}],
        }), encoding="utf-8")
        paths.append(summary_path)
    result = comparison.compare(paths[:1], paths[1:])
    assert result["canonical_overlap_count"] == 1
    assert result["variant"]["canonical_h_rows"] == [[big, 1]]
    assert result["variant"]["totals"]["catalog_matched_union"] == 1
    assert result["missing_reference_path_diagnostics"][0]["missing_class_records"] == 0
    old_output = tmp_path / "reduced_novelty.json"
    old_result = json.loads(old_output.read_text(encoding="utf-8"))
    old_result["coverage"]["discoveries"][0].update({
        "class_id": "old-only", "reduced_facet_word_hex": "0x6",
        "continuation_origin": {"word_hex": "0x3"},
        "path": [{"kind": "remove", "sampled_facet_word_hex": "0x3", "retained_vertices": [0]},
                 {"kind": "add", "source": "deterministic_ridge_exit"}],
    })
    old_output.write_text(json.dumps(old_result), encoding="utf-8")
    new_output = tmp_path / "shared_frontier.json"
    new_result = json.loads(new_output.read_text(encoding="utf-8"))
    new_result["coverage"]["discoveries"][0]["canonical_h_row"] = [big + 1, 1]
    new_output.write_text(json.dumps(new_result), encoding="utf-8")
    diagnostic = comparison.compare(paths[:1], paths[1:])["missing_reference_path_diagnostics"][0]
    assert diagnostic["visited_source_exit_gaps"] == 1
    assert diagnostic["records"][0]["continuation_origin_was_visited"] is True
    assert diagnostic["records"][0]["reference_terminal_was_visited"] is False
    assert diagnostic["records"][0]["reference_terminal_completion_source"] == "deterministic_ridge_exit"
    assert diagnostic["records"][0]["reference_retained_word_hex"] == "0x1"
    new_result["frontier_schedule"] = {
        "bank_history": [{"source_word_hex": "0x3", "retained_word_hex": "0x1",
                          "first_generated_iteration": 1, "first_attempted_iteration": None,
                          "actual_removals": 0}], "cached_faces": [],
    }
    new_output.write_text(json.dumps(new_result), encoding="utf-8")
    diagnostic = comparison.compare(paths[:1], paths[1:])["missing_reference_path_diagnostics"][0]
    assert diagnostic["recorded_frontier_gap_counts"] == {"retained_banked_not_taken": 1}
    old_result["coverage"]["discoveries"][0]["path"][0]["sampled_facet_word_hex"] = "0x9"
    old_output.write_text(json.dumps(old_result), encoding="utf-8")
    assert comparison.compare(paths[:1], paths[1:])["missing_reference_path_diagnostics"][0]["unvisited_source_gaps"] == 1
    summary = json.loads(paths[1].read_text(encoding="utf-8"))
    summary["config"]["seed"] += 1
    paths[1].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="parameters"):
        comparison.compare(paths[:1], paths[1:])


external = pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="4-2-2 handoff is external")


@external
def test_disabled_frontier_reproduces_v4_search_exactly():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    engines = [novelty.ReducedNoveltyMCTS(
        problem, novelty.ReducedNoveltyConfig(iterations=18, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    ), frontier.ReducedFrontierMCTS(
        problem, frontier.ReducedFrontierConfig(iterations=18, seed=17, progress_interval=0,
                                              enable_shared_face_frontier=False),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(),
    )]
    results = [engine.run() for engine in engines]
    assert results[0]["tree"] == results[1]["tree"]
    assert engines[0].reduced_witnesses == engines[1].reduced_witnesses
    assert engines[0].rng.getstate() == engines[1].rng.getstate()
    assert {key: rng.getstate() for key, rng in engines[0].rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in engines[1].rollout_rngs.items()}


@external
def test_lifting_outcomes_do_not_change_frontier_paths_values_or_rng():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    engines = [frontier.ReducedFrontierMCTS(
        problem, frontier.ReducedFrontierConfig(iterations=18, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT, full_oracle=ReportingOracle(valid),
    ) for valid in (False, True)]
    results = [engine.run() for engine in engines]
    assert results[0]["tree"] == results[1]["tree"]
    assert results[0]["face_frontier"] == results[1]["face_frontier"]
    assert engines[0].reduced_witnesses == engines[1].reduced_witnesses
    assert engines[0].frontier_attempts == engines[1].frontier_attempts
    assert engines[0].frontier_exits == engines[1].frontier_exits
    assert engines[0].rng.getstate() == engines[1].rng.getstate()
    for key, node in engines[0].nodes.items():
        other = engines[1].nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum, edge.prior) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum, alternative.prior)
