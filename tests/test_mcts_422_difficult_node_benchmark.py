import importlib
import json
from types import SimpleNamespace

import pytest


benchmark = importlib.import_module("mcts.422.difficult_node_benchmark")


def test_discovery_metrics_separates_path_contribution_and_windows():
    result = {
        "runtime": {"iterations_completed": 250},
        "coverage": {
            "discoveries": [
                {"iteration": 1, "path": [{"kind": "add"}]},
                {"iteration": 101, "path": [{"kind": "add"}, {"kind": "remove"}, {"kind": "add"}]},
            ],
            "class_frequencies": {"one": 7, "two": 3},
        },
        "geometry": {"statistics": {"successful_full_lifts": 3, "failed_full_lifts": 9}},
    }
    metrics = benchmark.discovery_metrics(result)
    assert [row["new_classes"] for row in metrics["growth_windows"]] == [1, 1, 0]
    assert metrics["growth_windows"][-1]["end_iteration_inclusive"] == 250
    assert metrics["repeat_class_hit_fraction"] == pytest.approx(0.8)
    assert metrics["validation_miss_lift_success_fraction"] == pytest.approx(0.25)
    assert metrics["direct_filler_discoveries"] == 1
    assert metrics["corrector_assisted_discoveries"] == 1
    assert metrics["iterations_since_last_discovery"] == 149


def test_discovery_metrics_does_not_report_zero_hit_rates_as_zero():
    result = {
        "runtime": {"iterations_completed": 20},
        "coverage": {"discoveries": [], "class_frequencies": {}},
        "geometry": {"statistics": {}},
    }
    metrics = benchmark.discovery_metrics(result)
    assert metrics["repeat_class_hit_fraction"] is None
    assert metrics["validation_miss_lift_success_fraction"] is None
    assert metrics["iterations_since_last_discovery"] == 20


def test_benchmark_refuses_existing_outputs(tmp_path):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(
            handoff_root=tmp_path,
            output_dir=tmp_path,
            config=benchmark.ReducedSearchConfig(iterations=1),
            baseline_manifest=tmp_path / "missing.json",
        )


def test_continuation_variant_keeps_baseline_default_and_requires_its_config(tmp_path):
    assert benchmark.VARIANTS["baseline"][0] is benchmark.ReducedSearchConfig
    assert benchmark.VARIANTS["continuation"][1] is benchmark.ReducedContinuationMCTS
    assert benchmark.VARIANTS["balanced_continuation"][0]().continuation_interval == 4
    assert benchmark.VARIANTS["balanced_continuation"][1] is benchmark.ReducedBalancedContinuationMCTS
    assert benchmark.VARIANTS["ridge_exit"][1] is benchmark.ReducedRidgeExitMCTS
    with pytest.raises(ValueError, match="configuration"):
        benchmark.run_benchmark(
            handoff_root=tmp_path, output_dir=tmp_path / "new",
            config=benchmark.ReducedSearchConfig(), baseline_manifest=tmp_path / "missing.json",
            variant="continuation",
        )


@pytest.mark.parametrize("variant", ["continuation", "balanced_continuation", "ridge_exit"])
def test_disabled_continuation_does_not_audit_nonexistent_local_metadata(tmp_path, monkeypatch, variant):
    node_id = "fixture-node"
    monkeypatch.setattr(benchmark, "DIFFICULT_NODES", {node_id: "fixture"})
    monkeypatch.setattr(benchmark, "baseline_snapshot", lambda *_args: {
        "nodes": [{"node_id": node_id, "same_budget_prefix_covered_count": 0}]
    })
    node = SimpleNamespace(class_ids=(), class_count=0, to_dict=lambda: {"node_id": node_id})
    monkeypatch.setattr(benchmark, "load_inventory", lambda _root: SimpleNamespace(by_id={node_id: node}))
    monkeypatch.setattr(benchmark, "ReducedSubgroupProblem", SimpleNamespace(load=lambda *_args: None))
    monkeypatch.setattr(benchmark, "FacetOracle", lambda **_kwargs: None)

    class FakeEngine:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, **_kwargs):
            return {
                "runtime": {"iterations_completed": 1, "elapsed_seconds": 0.0},
                "coverage": {"discoveries": [], "covered_target_count": 0,
                             "missing_target_ids": [], "class_frequencies": {}},
                "geometry": {"statistics": {}},
            }

    config_class = benchmark.VARIANTS[variant][0]
    monkeypatch.setitem(benchmark.VARIANTS, variant, (config_class, FakeEngine))
    monkeypatch.setattr(benchmark, "audit_node", lambda *_args: {"failure_count": 0})
    monkeypatch.setattr(benchmark, "audit_ridge_exits", lambda *_args: {"failure_count": 0})

    def unexpected_audit(*_args):
        raise AssertionError("disabled continuation has no local-path metadata")

    monkeypatch.setattr(benchmark, "audit_local_paths", unexpected_audit)
    result = benchmark.run_benchmark(
        handoff_root=tmp_path, output_dir=tmp_path / "result",
        config=config_class(iterations=1, enable_local_continuation=False),
        baseline_manifest=tmp_path / "missing.json", variant=variant,
    )
    assert result["state"] == "COMPLETED"


def test_discovery_metrics_distinguishes_local_actions_from_historical_proof():
    result = {
        "runtime": {"iterations_completed": 10},
        "coverage": {
            "discoveries": [{
                "iteration": 2, "episode_kind": "continuation",
                "path": [{"kind": "remove"}] * 4,
                "local_path": [{"kind": "remove"}],
                "continuation_origin": {"lifted": False},
            }], "class_frequencies": {"one": 1},
        }, "geometry": {"statistics": {}},
    }
    metrics = benchmark.discovery_metrics(result)
    assert metrics["remove_count_histogram"] == {4: 1}
    assert metrics["local_remove_count_histogram"] == {1: 1}
    assert metrics["episode_discoveries"] == {"continuation": 1}
    assert metrics["failed_lift_origin_discoveries"] == 1


@pytest.mark.parametrize("mismatch", ["state", "nodes", "seed"])
def test_benchmark_rejects_unmatched_reference_before_search(tmp_path, mismatch):
    reference = {
        "state": "COMPLETED",
        "selected_node_ids": list(benchmark.DIFFICULT_NODES),
        "config": {"seed": 4222028},
    }
    if mismatch == "state":
        reference["state"] = "RUNNING"
    elif mismatch == "nodes":
        reference["selected_node_ids"] = ["BFS422-H-0000"]
    else:
        reference["config"]["seed"] = 123
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    output = tmp_path / "new-run"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(
            handoff_root=tmp_path,
            output_dir=output,
            config=benchmark.ReducedCreditConfig(iterations=300, seed=4222028),
            baseline_manifest=tmp_path / "missing.json",
            variant="credit",
            reference_summary=path,
        )
    assert not output.exists()


def test_additional_reference_is_checked_before_creating_outputs(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"state": "RUNNING", "selected_node_ids": []}), encoding="utf-8")
    output = tmp_path / "new-run"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(
            handoff_root=tmp_path, output_dir=output,
            config=benchmark.ReducedRidgeExitConfig(),
            baseline_manifest=tmp_path / "missing.json", variant="ridge_exit",
            additional_reference_summaries=(path,),
        )
    assert not output.exists()


def test_canonical_comparison_preserves_large_integer_rows(tmp_path):
    big = 2 ** 64
    previous_path = tmp_path / "previous.json"
    previous_path.write_text(json.dumps({"coverage": {"discoveries": [
        {"canonical_h_row": [big, 1]}, {"canonical_h_row": [big + 1, 1]},
    ]}}), encoding="utf-8")
    reference = {"runs": [{"node_id": "fixture", "output": str(previous_path),
                           "covered_target_count": 2, "runtime": {"elapsed_seconds": 1}}]}
    result = {"coverage": {"covered_target_count": 2, "discoveries": [
        {"canonical_h_row": [big, 1]}, {"canonical_h_row": [big + 2, 1]},
    ]}}
    comparison = benchmark.canonical_comparison(reference, "fixture", result)
    assert comparison["overlap_canonical_count"] == 1
    assert comparison["reference_only_canonical_h_rows"] == [[big + 1, 1]]
    assert comparison["variant_only_canonical_h_rows"] == [[big + 2, 1]]
