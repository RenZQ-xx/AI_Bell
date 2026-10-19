from __future__ import annotations

import importlib
import random
import time
from dataclasses import asdict, replace

import pytest

atlas = importlib.import_module('mcts.422.atlas')
model = importlib.import_module('mcts.422.model')
v30 = importlib.import_module('mcts.422.continuation_search_v30')
v31 = importlib.import_module('mcts.422.continuation_search_v31')
adj = importlib.import_module('mcts.422.recursive_adjacency')
HANDOFF = atlas.DEFAULT_HANDOFF_ROOT
SOURCE = int('ffffffccffccffccccfffcfccfcfffccccffcfcffcfcffccccffccffccffffff', 16)
NEIGHBOR = int('ffffa3c4a3c4a3c4c4a3fafa8585a3c4c4a38585fafaa3c4c4a3c4a3c4a3ffff', 16)
pytestmark = pytest.mark.skipif(not HANDOFF.exists(), reason='4-2-2 handoff is external')


@pytest.fixture
def engine():
    e = v31.Bell422LazyASSearch(v31.FullspaceASConfig(iterations=4, progress_interval=0,
        use_catalog=False, canonicalize_classes=False, stabilizer_probes=4), handoff_root=HANDOFF)
    e.started_at = time.perf_counter()
    assert e.facet_oracle.validate(SOURCE).valid
    return e


def certified_pair(e):
    e.as_observed_supports.add(SOURCE)
    assert e._seed_source_certificate(SOURCE)
    ridge = SOURCE & NEIGHBOR
    target, rotation = e.adjacency.rotate(e.adjacency.full_word, SOURCE, ridge)
    assert target == NEIGHBOR
    return {'chain_word_hex':[hex(e.adjacency.full_word), hex(SOURCE)],
        'leaf_boundary_word_hex':hex(ridge), 'rotations':[rotation],
        'destination_word_hex':hex(target), 'retained_word_hex':hex(ridge)}


def test_fullspace_geometry_is_not_a_reduced_subproblem(engine):
    assert engine.adjacency.points.shape == (256, 80)
    assert engine.adjacency.frame(engine.adjacency.full_word).dimension == 80
    assert not engine.result()['features']['reduced_to_full_lift']
    assert engine.config.new_class_reward == 10 and engine.config.extended_max_corrections == 8


def test_only_actually_observed_facets_may_start_jobs(engine, monkeypatch):
    monkeypatch.setattr(engine.adjacency, 'new_job', lambda *a: pytest.fail('unvisited source'))
    node = engine._node(model.SearchState(SOURCE, 'corrector', 0, atlas.IDENTITY_PATTERN_ID, SOURCE), 79)
    assert engine._as_remove_action(node, tree=False) is None
    assert not engine.as_jobs and not engine.as_bank


def test_cache_import_does_not_classify_unvisited_targets(engine, monkeypatch):
    proof = certified_pair(engine)
    monkeypatch.setattr(engine.facet_oracle, 'validate', lambda *a: pytest.fail('candidate classification'))
    engine._as_cache_output(SOURCE, proof)
    assert NEIGHBOR not in engine.facet_oracle.cache
    assert not engine.discovered and not engine.discovery_events
    ridge = SOURCE & NEIGHBOR
    assert set(engine.actions._ridge_cache[(SOURCE, ridge)].facet_words) == {SOURCE, NEIGHBOR}
    assert adj.audit_adjacency_proof(engine.facet_oracle.points, proof)


def test_cached_pair_reaches_original_rank79_without_another_pivot(engine, monkeypatch):
    proof = certified_pair(engine)
    engine._as_cache_output(SOURCE, proof)
    node = engine._node(model.SearchState(SOURCE, 'corrector', 0, atlas.IDENTITY_PATTERN_ID, SOURCE), 79)
    monkeypatch.setattr(engine.adjacency, 'advance', lambda *a, **k: pytest.fail('unneeded refill'))
    action = engine._as_remove_action(node, tree=False)
    moved = engine._transition(node.state, action)
    assert moved.rank == 78 and moved.state.corrections_used == 1
    assert moved.state.source_facet_word == SOURCE
    filler = engine._node(moved.state, moved.rank)
    edge, end = engine._as_completion(filler)
    assert edge.action.add_word.bit_count() == 1
    assert end.rank == 79 and end.state.support_word == NEIGHBOR and end.facet.valid
    assert end.state.corrections_used == 1 and not engine.discovered


def test_fullspace_resumable_job_returns_auditable_original_neighbor(engine):
    engine.as_observed_supports.add(SOURCE)
    engine._seed_source_certificate(SOURCE)
    job = engine.adjacency.new_job(SOURCE)
    outputs = []
    deadline = time.perf_counter()+20
    while not outputs and not job.done and time.perf_counter() < deadline:
        outputs.extend(engine.adjacency.advance(job, engine.as_rng, max_outputs=1, max_steps=4, seconds=1))
    assert outputs, (job.status, len(job.chain), engine.adjacency.diagnostics())
    for proof in outputs:
        assert adj.audit_adjacency_proof(engine.facet_oracle.points, proof)
        engine._as_cache_output(SOURCE, proof)
        assert engine.rank_oracle.rank(int(proof['retained_word_hex'], 16)) == 78
    assert not engine.discovery_events


def test_budget_debt_defers_new_geometry(engine, monkeypatch):
    engine.as_observed_supports.add(SOURCE)
    engine.as_compute_seconds = 1e6
    monkeypatch.setattr(engine.adjacency, 'advance', lambda *a, **k: pytest.fail('time debt'))
    node = engine._node(model.SearchState(SOURCE, 'corrector', 0, atlas.IDENTITY_PATTERN_ID, SOURCE), 79)
    assert engine._as_remove_action(node, tree=False) is None and not engine.as_jobs


def test_disabled_as_preserves_v30_trajectory():
    config = v31.FullspaceASConfig(iterations=16, max_depth=6, tail_checks=2,
        stabilizer_probes=4, quotient_candidate_limit=4, quotient_secondary_limit=1,
        partial_rank_checks=2, ridge_rank_checks=4, ridge_basis_rank_checks=8,
        progress_interval=0, use_catalog=False, canonicalize_classes=False, enable_lazy_as=False)
    old_config = v30.EarlyNoopSearchConfig(**{k:v for k,v in asdict(config).items()
                                           if k in v30.EarlyNoopSearchConfig.__dataclass_fields__})
    old = v30.Bell422EarlyNoopSearch(old_config, handoff_root=HANDOFF)
    new = v31.Bell422LazyASSearch(config, handoff_root=HANDOFF)
    a,b = old.run(),new.run()
    assert a['tree'] == b['tree'] and a['search_stats'] == b['search_stats']
    clean = lambda events:[{k:v for k,v in e.items() if k!='elapsed_seconds'} for e in events]
    assert clean(a['coverage']['discoveries']) == clean(b['coverage']['discoveries'])
    assert old.rng.getstate() == new.rng.getstate()
    assert all(old.rollout_rngs[p].getstate() == new.rollout_rngs[p].getstate() for p in old.rollout_rngs)
    assert not new.as_jobs and not new.as_observed_supports


def test_as_fallback_preserves_native_rollout_random_draws(engine, monkeypatch):
    engine.config = replace(engine.config, as_operator_probability=1)
    native_config = v30.EarlyNoopSearchConfig(**{k:v for k,v in asdict(engine.config).items()
                                               if k in v30.EarlyNoopSearchConfig.__dataclass_fields__})
    old = v30.Bell422EarlyNoopSearch(native_config, handoff_root=HANDOFF)
    assert old.facet_oracle.validate(SOURCE).valid
    state = model.SearchState(SOURCE, 'corrector', 0, atlas.IDENTITY_PATTERN_ID, SOURCE)
    old_node, new_node = old._node(state, 79), engine._node(state, 79)
    old.rollout_rngs['mixed'] = random.Random(1)
    engine.rollout_rngs['mixed'] = random.Random(1)
    monkeypatch.setattr(engine, '_as_remove_action', lambda *a, **k: None)
    first, second = old._rollout_edge(old_node, 'mixed'), engine._rollout_edge(new_node, 'mixed')
    assert first[0].action.action_id == second[0].action.action_id
    assert old.rollout_rngs['mixed'].getstate() == engine.rollout_rngs['mixed'].getstate()


def test_as_operator_does_not_bypass_native_rollout_stop_probability(engine, monkeypatch):
    engine.config = replace(engine.config, as_operator_probability=1)
    engine.rollout_rngs['mixed'] = random.Random(2)
    node = engine._node(model.SearchState(SOURCE, 'corrector', 0, atlas.IDENTITY_PATTERN_ID, SOURCE), 79)
    monkeypatch.setattr(engine, '_as_remove_action', lambda *a, **k: pytest.fail('native rollout stops'))
    assert engine._rollout_edge(node, 'mixed')[0].action.kind == 'stop'


def test_as_operator_respects_original_correction_limit(engine):
    node = engine._node(model.SearchState(SOURCE, 'corrector', 8, atlas.IDENTITY_PATTERN_ID, SOURCE), 79)
    assert not engine._as_selected(node)


def test_cooperative_service_limits_fragmentation_without_redirecting_state(engine):
    engine.config = replace(engine.config, as_max_active_jobs=1)
    engine.as_jobs[SOURCE] = engine.adjacency.new_job(SOURCE)
    assert engine._as_work_source(NEIGHBOR) == SOURCE
    assert len(engine.as_jobs) == 1
    engine.as_bank[SOURCE] = {3:{}}
    assert engine._as_work_source(NEIGHBOR) == NEIGHBOR


def test_ready_exits_are_selected_without_more_geometry_or_rng(engine, monkeypatch):
    engine.config = replace(engine.config, as_operator_probability=0)
    engine._as_cache_output(SOURCE, certified_pair(engine))
    node = engine._node(model.SearchState(SOURCE, 'corrector', 0, atlas.IDENTITY_PATTERN_ID, SOURCE), 79)
    monkeypatch.setattr(engine.as_policy_rng, 'random', lambda: pytest.fail('ready exit does not need admission'))
    assert engine._as_selected(node)


def test_cooperative_outputs_stay_with_their_actual_source(engine, monkeypatch):
    proof = certified_pair(engine)
    engine.as_observed_supports.add(NEIGHBOR)
    assert engine.facet_oracle.validate(NEIGHBOR).valid
    engine.config = replace(engine.config, as_max_active_jobs=1)
    job = engine.adjacency.new_job(SOURCE)
    engine.as_jobs[SOURCE] = job
    monkeypatch.setattr(engine.adjacency, 'advance', lambda selected, *a, **k:
                        [proof] if selected is job else pytest.fail('wrong source job'))
    node = engine._node(model.SearchState(NEIGHBOR, 'corrector', 0,
                        atlas.IDENTITY_PATTERN_ID, NEIGHBOR), 79)
    assert engine._as_remove_action(node, tree=False) is None
    assert SOURCE in engine.as_bank and NEIGHBOR not in engine.as_bank
    assert not engine.discovered and not engine.as_executed_proofs


def test_actual_as_arrival_preserves_global_reward_and_counterfactual_credit(engine):
    origin_node = engine._node(model.SearchState(SOURCE, 'corrector', 0,
                              atlas.IDENTITY_PATTERN_ID, SOURCE), 79)
    origin = engine._terminal_outcome(engine.facet_oracle.validate(SOURCE),
                                     path=[], depth=0, source='test')
    engine._as_cache_output(SOURCE, certified_pair(engine))
    action = engine._as_remove_action(origin_node, tree=False)
    remove_edge, retained = engine._edge_for(origin_node, action)
    remove_step = engine._make_step(origin_node, remove_edge, retained.state,
                                   current_terminal=origin, is_tree_edge=False)
    filler = engine._node(retained.state, retained.rank)
    add_edge, arrived = engine._as_completion(filler)
    add_step = engine._make_step(filler, add_edge, arrived.state,
                                current_terminal=None, is_tree_edge=False)
    path = [remove_step, add_step]
    outcome = engine._terminal_outcome(arrived.facet, path=path, depth=2, source='test')
    assert outcome.new_class and outcome.reward >= engine.config.new_class_reward
    engine._resolve_corrector_credit(remove_step, outcome)
    expected = (outcome.counterfactual_utility-origin.counterfactual_utility
                + engine.config.corrector_new_basin_bonus+engine._transition_reward(remove_edge))
    assert remove_step.corrector_credit == pytest.approx(expected)
    assert remove_step.corrector_novelty_credit == engine.config.new_class_reward
    assert len(engine.as_executed_proofs) == 1
    assert engine.discovered[arrived.facet.class_id]['as_actions_in_local_path'] == 2
    repeated = engine._terminal_outcome(arrived.facet, path=path, depth=2, source='test')
    assert not repeated.new_class and repeated.reward < engine.config.known_class_reward
    assert len(engine.discovered) == 2 and len(engine.as_executed_proofs) == 1


@pytest.mark.parametrize('changes', [{'as_time_fraction':1}, {'as_operator_probability':-1},
    {'as_min_slice_seconds':1}, {'as_slice_seconds':0}, {'as_max_live_jobs':0},
    {'as_max_active_jobs':0}, {'as_max_active_jobs':65},
    {'as_cutoff_vertices':100}])
def test_invalid_as_configuration(changes):
    with pytest.raises(ValueError):
        v31.FullspaceASConfig(**changes)
