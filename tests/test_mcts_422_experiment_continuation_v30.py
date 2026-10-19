from __future__ import annotations

import copy
import importlib

experiment = importlib.import_module("mcts.422.experiment_continuation_v30")


def results():
    old = {
        "coverage": {"discoveries": [{"canonical_h_row": [1, 2], "iteration": 1,
                                       "path": [{"kind": "add"}], "elapsed_seconds": 1.0}]},
        "config": {"iterations": 1000, "seed": 7, "progress_interval": 100},
        "runtime": {"elapsed_seconds": 100.0, "truncated": False, "iterations_completed": 1000},
        "tree": {"node_count": 3},
    }
    new = copy.deepcopy(old)
    new["config"]["enable_early_noop_guard"] = True
    new["runtime"]["elapsed_seconds"] = 50.0
    new["coverage"]["discoveries"][0]["elapsed_seconds"] = 0.5
    return old, new


def test_gate_accepts_faster_same_trajectory():
    old, new = results()
    comparison = experiment.compare_runs(old, new)
    assert comparison["eligible_for_long_run"]
    assert comparison["speedup"] == 2.0
    assert not comparison["shared_config_differences"]


def test_gate_rejects_changed_path_even_if_coverage_is_same():
    old, new = results()
    new["coverage"]["discoveries"][0]["path"] = [{"kind": "rewrite"}]
    comparison = experiment.compare_runs(old, new)
    assert comparison["identical_canonical_class_set"]
    assert not comparison["eligible_for_long_run"]


def test_gate_rejects_changed_config_partial_run_and_no_speed_gain():
    for mode in ("config", "partial", "time"):
        old, new = results()
        if mode == "config":
            new["config"]["seed"] = 8
        elif mode == "partial":
            new["runtime"]["truncated"] = True
        else:
            new["runtime"]["elapsed_seconds"] = 99.0
        assert not experiment.compare_runs(old, new)["eligible_for_long_run"]
