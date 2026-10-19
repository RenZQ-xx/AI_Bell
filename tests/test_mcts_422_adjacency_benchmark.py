from __future__ import annotations

import importlib
import math

import pytest


benchmark = importlib.import_module("mcts.422.adjacency_benchmark")


class FakeEngine:
    def __init__(self):
        self.discovery_events = []
        self.reduced_discovery_events = []
        self.search_deadline = 0.0
        self.calls = []

    def run_iteration(self, iteration):
        self.calls.append(iteration)
        self.discovery_events.append({"iteration": iteration})
        self.reduced_discovery_events.append({"iteration": iteration})


@pytest.mark.parametrize("seconds,iterations", [(None, None), (1.0, 3), (0, None),
                                               (math.inf, None), (None, 0), (None, 1.5)])
def test_invalid_budget_is_rejected(seconds, iterations):
    with pytest.raises(ValueError):
        benchmark.validate_budget(seconds, iterations)


def test_fixed_iteration_search_has_no_global_wall_deadline_or_coverage_stop(monkeypatch):
    clock = iter(range(0, 100_000, 10))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    engine = FakeEngine()
    result = benchmark.search_loop(engine, "v6_as", iterations=3)
    assert engine.calls == [1, 2, 3]
    assert math.isinf(engine.search_deadline)
    assert engine.termination_reason == "iteration_budget"
    assert result["iterations"] == len(result["iteration_trace"]) == 3
    assert result["search_seconds"] > 30
    assert all("elapsed_seconds" in event for event in engine.discovery_events)


def test_time_budget_keeps_wall_stopping_and_no_fake_iteration_trace(monkeypatch):
    clock = iter(range(1000))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    engine = FakeEngine()
    result = benchmark.search_loop(engine, "v6", seconds=5)
    assert engine.search_deadline == 5
    assert engine.termination_reason == "wall_time_limit"
    assert result["iterations"] == 2
    assert not result["iteration_trace"]


def test_as_only_is_not_treated_as_mcts_iterations():
    with pytest.raises(ValueError):
        benchmark.search_loop(FakeEngine(), "as_only", iterations=3)
