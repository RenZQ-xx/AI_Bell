from __future__ import annotations

import importlib
import copy
import random
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest

adj = importlib.import_module("mcts.422.recursive_adjacency")
resume = importlib.import_module("mcts.422.resumable_adjacency")
lazy = importlib.import_module("mcts.422.reduced_lazy_adjacency_search")
multi = importlib.import_module("mcts.422.reduced_multi_exit_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")
bench = importlib.import_module("mcts.422.adjacency_benchmark")


class ReportingOracle:
    def __init__(self):
        self.calls = []

    def validate(self, word):
        self.calls.append(word)
        return reduced.FacetResult(False, "fixture", word)

    def to_dict(self):
        return {"calls": len(self.calls)}


def make_engine(**kwargs):
    points = np.asarray(list(product((0, 1), repeat=3)), dtype=float)
    p = reduced.ReducedSubgroupProblem(
        "fixture", "fixture", 3, points, points, tuple(range(8)), tuple(range(8)),
        tuple(1 << i for i in range(8)), 8, Path("fixture"))
    config = lazy.ReducedLazyAdjacencyConfig(iterations=20, seed=32, progress_interval=0, **kwargs)
    return lazy.ReducedLazyAdjacencyMCTS(p, config, handoff_root=Path("fixture"),
                                       full_oracle=ReportingOracle())


def test_resuming_each_atomic_step_preserves_batch_and_rng():
    points = list(product((0, 1), repeat=4))
    config = adj.AdjacencyConfig(cutoff_vertices=4, max_macro_seconds=5)
    old = adj.RecursiveAdjacency(points, config)
    expected = old.sample(255, random.Random(4))
    geometry = resume.ResumableAdjacency(points, config)
    job, rng = geometry.new_job(255), random.Random(4)
    outputs = []
    for _ in range(300):
        outputs.extend(geometry.advance(job, rng, max_steps=1, seconds=5))
        if job.done:
            break
    assert job.done and job.status == "chain_finished"
    assert outputs == expected.outputs
    assert geometry.stats["resumed_advances"] > 5
    assert geometry.stats["descent_lp_attempts"] == old.stats["descent_lp_attempts"]
    assert geometry.stats["leaf_enumerations_finished"] == 1
    assert all(adj.audit_adjacency_proof(points, proof) for proof in outputs)


def test_partial_leaf_cursor_and_rotation_survive_deadline():
    geometry = resume.ResumableAdjacency(list(product((0, 1), repeat=4)),
                                       adj.AdjacencyConfig(cutoff_vertices=4))
    job, rng = geometry.new_job(255), random.Random(4)
    for _ in range(100):
        geometry.advance(job, rng, max_steps=1)
        if job.simplex_cursor:
            break
    assert job.simplex_cursor > 0 and job.leaves is None
    saved = (list(job.chain), job.simplex_cursor, job.leaf_cursor, job.level)
    assert not geometry.advance(job, rng, deadline=time.perf_counter()-1)
    assert (job.chain, job.simplex_cursor, job.leaf_cursor, job.level) == saved
    outputs = []
    for _ in range(200):
        outputs.extend(geometry.advance(job, rng, max_steps=1))
        if job.done:
            break
    assert len(outputs) == 4 and job.done


def test_output_batch_limit_preserves_remaining_chain_work():
    geometry = resume.ResumableAdjacency(list(product((0, 1), repeat=3)))
    job, rng = geometry.new_job(15), random.Random(1)
    first = geometry.advance(job, rng, max_outputs=1)
    assert len(first) == 1 and not job.done
    chain = tuple(job.chain)
    rest = geometry.advance(job, rng, max_outputs=8)
    assert len(rest) == 3 and tuple(job.chain) == chain and job.done
    assert not geometry.advance(job, rng)


def test_original_benchmark_defaults_do_not_enable_new_mode():
    assert "v6_as_lazy" in bench.AVAILABLE_MODES and "v6_as_lazy" not in bench.MODES


@pytest.mark.parametrize("kwargs", [
    {"as_operator_probability": -0.1}, {"as_time_fraction": 1},
    {"as_slice_seconds": 0}, {"as_bootstrap_seconds": float("inf")},
    {"as_outputs_per_advance": 0}, {"as_max_live_jobs": 0},
    {"as_min_slice_seconds": 1},
])
def test_invalid_lazy_config(kwargs):
    with pytest.raises(ValueError):
        lazy.ReducedLazyAdjacencyConfig(**kwargs)


def test_candidate_generation_is_lazy_and_never_lifts(monkeypatch):
    engine = make_engine(as_operator_probability=1)
    node = reduced.ReducedNode(reduced.ReducedState(15, "corrector"), 2)
    monkeypatch.setattr(engine.adjacency, "advance", lambda *a, **k: pytest.fail("eager AS"))
    engine._corrector_actions(node, count=3, rng=random.Random(1), persistent=True)
    assert not engine.as_jobs and not engine.as_bank and not engine.geometry.full_oracle.calls
    assert not engine.discovered_reduced_facets


def test_selected_operator_uses_cached_exits_without_refill_or_lifting(monkeypatch):
    engine = make_engine(as_operator_probability=1, as_bootstrap_seconds=2, as_slice_seconds=1)
    node = reduced.ReducedNode(reduced.ReducedState(15, "corrector"), 2)
    action = engine._as_action(node, tree=True)
    assert action is not None
    assert len(engine.as_bank[15]) == 2
    assert not engine.geometry.full_oracle.calls and not engine.discovered_reduced_facets
    monkeypatch.setattr(engine.adjacency, "advance", lambda *a, **k: pytest.fail("unnecessary refill"))
    assert engine._as_action(node, tree=True) is not None
    filler = reduced.ReducedNode(reduced.ReducedState(action.retained_word, "filler", 1, 15), 1)
    monkeypatch.setattr(engine.geometry, "_compute_ridge_exits", lambda *a: pytest.fail("duplicate solve"))
    completion = engine._filler_actions(filler, sample_count=2, rng=random.Random(2))
    assert len(completion) == 1 and completion[0].source == "deterministic_ridge_exit"
    assert not engine.geometry.full_oracle.calls


def test_time_debt_defers_computation_but_not_cached_consumption(monkeypatch):
    engine = make_engine(as_operator_probability=1, as_bootstrap_seconds=2, as_slice_seconds=1)
    node = reduced.ReducedNode(reduced.ReducedState(15, "corrector"), 2)
    assert engine._as_action(node, tree=True) is not None
    engine.as_compute_seconds = 1e6
    monkeypatch.setattr(engine.adjacency, "advance", lambda *a, **k: pytest.fail("budget exceeded"))
    assert engine._as_action(node, tree=True) is not None
    for retained in engine.as_bank[15]:
        engine.as_consumed[(15, retained)] += 1
    assert engine._as_action(node, tree=True) is None
    assert engine.stats["lazy_as_budget_deferrals"] == 1


def test_tiny_credit_is_saved_instead_of_spent_on_micro_slices(monkeypatch):
    engine = make_engine(as_operator_probability=1)
    node = reduced.ReducedNode(reduced.ReducedState(15, "corrector"), 2)
    monkeypatch.setattr(engine, "_allowance", lambda: 0.001)
    monkeypatch.setattr(engine.adjacency, "advance", lambda *a, **k: pytest.fail("tiny slice"))
    assert engine._as_action(node, tree=True) is None
    assert not engine.as_jobs


def test_proof_cache_rejects_invalid_neighbors_without_lifting():
    engine = make_engine()
    proof = adj.RecursiveAdjacency(engine.problem.irredundant_points).sample(15, random.Random(1)).outputs[0]
    invalid = copy.deepcopy(proof)
    invalid['destination_word_hex'] = hex(15)
    with pytest.raises(ValueError):
        engine._install_proof(15, invalid)
    assert not engine.as_bank and not engine.geometry.full_oracle.calls


def test_unfinished_job_lru_is_bounded(monkeypatch):
    engine = make_engine(as_operator_probability=1, as_max_live_jobs=1)
    monkeypatch.setattr(engine, '_allowance', lambda: 0.04)
    monkeypatch.setattr(engine.adjacency, 'advance', lambda *a, **k: [])
    for word in (15, 240):
        node = reduced.ReducedNode(reduced.ReducedState(word, 'corrector'), 2)
        assert engine._as_action(node, tree=False) is None
    assert list(engine.as_jobs) == [240]
    assert engine.stats['lazy_as_unfinished_jobs_evicted'] == 1


def test_full_tree_does_not_generate_as_work(monkeypatch):
    engine = make_engine(as_operator_probability=1)
    node = reduced.ReducedNode(reduced.ReducedState(15, "corrector"), 2)
    monkeypatch.setattr(engine, "_child_limit", lambda node: 0)
    monkeypatch.setattr(engine.adjacency, "advance", lambda *a, **k: pytest.fail("tree is full"))
    assert engine._widen(node) is None and not engine.as_jobs


def test_disabled_lazy_search_is_exact_v6_trajectory():
    new = make_engine(enable_recursive_adjacency=False)
    old = multi.ReducedMultiExitMCTS(new.problem, multi.ReducedMultiExitConfig(
        iterations=20, seed=32, progress_interval=0), handoff_root=Path("fixture"),
        full_oracle=ReportingOracle())
    assert all(asdict(new.config)[key] == value for key, value in asdict(old.config).items())
    for i in range(1, 21):
        old.run_iteration(i)
        new.run_iteration(i)
    assert old.rng.getstate() == new.rng.getstate()
    assert old.stats == new.stats and old.reduced_witnesses == new.reduced_witnesses


def test_actual_paths_and_as_proofs_pass_independent_audits():
    engine = make_engine(as_operator_probability=1, as_bootstrap_seconds=5, as_slice_seconds=1)
    engine.started_at = time.perf_counter()
    for i in range(1, 21):
        engine.run_iteration(i)
    result = engine.result()
    assert engine.as_executed_proofs
    assert bench.audit_reduced_paths(engine.problem, result['reduced_facets']['witnesses'], Path('fixture'))['failure_count'] == 0
    assert all(adj.audit_adjacency_proof(engine.problem.irredundant_points, proof)
               for proof in engine.as_executed_proofs.values())
    assert all(e['local_corrections'] <= e['local_correction_limit']
               for e in result['reduced_facets']['discovery_events'])
    assert result['features']['lifting_is_reporting_only']
