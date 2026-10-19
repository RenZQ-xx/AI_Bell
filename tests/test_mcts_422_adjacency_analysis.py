from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


analysis = importlib.import_module("mcts.422.analyze_adjacency_benchmark")


@pytest.mark.parametrize("baseline,expected", [
    (None, {"intersection_with_v6": None, "v6_only": None, "variant_only": None}),
    (set(), {"intersection_with_v6": 0, "v6_only": 0, "variant_only": 2}),
    ({2, 3}, {"intersection_with_v6": 1, "v6_only": 1, "variant_only": 1}),
])
def test_absent_reference_is_not_reported_as_zero_overlap(baseline, expected):
    assert analysis.reference_counts({1, 2}, baseline) == expected


def saved_fixture(monkeypatch, state="COMPLETED"):
    summary = {"state": state, "config": {"nodes": ["node"], "seeds": [1]}, "runs": [
        {"node_id": "node", "seed": 1, "mode": mode, "iterations": 10, "result": mode}
        for mode in ("v6", "v6_as")]}
    payloads = {"summary": summary}
    for mode, times in (("v6", (2, 9)), ("v6_as", (1, 3))):
        payloads[mode] = {
            "coverage": {"discoveries": [{"iteration": i, "canonical_h_row": [i, 1]} for i in times]},
            "reduced_facets": {"discovery_events": [
                {"iteration": i, "reduced_facet_word_hex": hex(i)} for i in times]}}
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: json.dumps(payloads[str(self)]))


def test_common_prefix_filters_classes_and_reduced_closures_without_rerunning(monkeypatch):
    saved_fixture(monkeypatch)
    result = analysis.compare_mcts_prefix(Path("summary"), 4)
    case = result["cases"][0]
    assert case["v6"]["canonical_classes"] == case["v6"]["reduced_facets"] == 1
    assert case["v6_as"]["canonical_classes"] == case["v6_as"]["reduced_facets"] == 2
    assert case["intersection"] == 0 and case["v6_only"] == 1 and case["v6_as_only"] == 2
    assert result["higher_class_count"] == 1
    assert not result["same_iterations_mean_equal_compute"]
    assert not result["prefix_elapsed_times_available"]
    assert "as_only" in result["excluded_modes"]


def test_prefix_requires_complete_saved_coverage_of_iteration_budget(monkeypatch):
    saved_fixture(monkeypatch)
    for iterations in (0, 11):
        with pytest.raises(ValueError):
            analysis.compare_mcts_prefix(Path("summary"), iterations)


def test_unfinished_experiment_is_not_analyzed(monkeypatch):
    saved_fixture(monkeypatch, state="RUNNING")
    with pytest.raises(ValueError):
        analysis.compare_mcts_prefix(Path("summary"), 4)


def test_terminal_proof_matches_source_and_ridge_not_just_destination():
    correct, other = {"chain_word_hex": ["0xff", "0xf"]}, {"chain_word_hex": ["0xff", "0x3c"]}
    proofs = {(15, 3, 51): correct, (60, 48, 51): other}
    event = {"reduced_facet_word_hex": "0x33", "path": [
        {"kind": "remove", "source": "recursive_as_ridge",
         "sampled_facet_word_hex": "0xf", "retained_vertices": [0, 1]},
        {"kind": "add", "source": "deterministic_ridge_exit"}]}
    assert analysis.terminal_as_proof(event, proofs) is correct
    event["path"][0]["sampled_facet_word_hex"] = "0x3c"
    assert analysis.terminal_as_proof(event, proofs) is None


def test_historical_as_move_does_not_attribute_a_later_non_as_terminal():
    event = {"path": [{"kind": "remove", "source": "recursive_as_ridge"},
                      {"kind": "remove", "source": "reduced_facet_ridge"}]}
    assert analysis.terminal_as_proof(event, {}) is None
