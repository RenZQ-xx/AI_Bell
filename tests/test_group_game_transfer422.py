from dataclasses import asdict, replace
import random
import time

import pytest

from mcts.group_game.model import GameNode, MacroAction, SymmetryState
from mcts.group_game.search import GroupGameConfig, SingleTreeGroupGame
from mcts.group_game.transfer422.audit import audit_result
from mcts.group_game.transfer422.search import (
    TransferConfig, TransferredGroupGame, adjacency_module,
)
from mcts.subgroup_patterns import IDENTITY_PATTERN_ID


def config(**kwargs):
    return TransferConfig(iterations=16, include_level2_analysis=False,
        max_level2_pair_checks=4, max_pattern_analyses_per_class=1,
        ridge_max_candidates=64, ridge_batch_size=8, ridge_focus_batches=1, **kwargs)


def test_all_disabled_preserves_native_v25_trajectory():
    cfg = config(enable_local_continuation=False, enable_early_noop_guard=False, enable_lazy_as=False)
    base = GroupGameConfig(**{k: v for k, v in asdict(cfg).items() if k in GroupGameConfig.__dataclass_fields__})
    old, new = SingleTreeGroupGame(base), TransferredGroupGame(cfg)
    a, b = old.run(), new.run()
    assert a['tree'] == b['tree']
    assert a['summary']['discovered_class_ids'] == b['summary']['discovered_class_ids']
    clean = lambda es: [{k: v for k, v in e.items() if k not in (
        'elapsed_seconds', 'continuation_origin', 'local_path')} for e in es]
    assert clean(a['discoveries']) == clean(b['discoveries'])
    assert old.rng.getstate() == new.rng.getstate()
    assert all(old.rollout_rngs[p].getstate() == new.rollout_rngs[p].getstate() for p in old.rollout_rngs)
    assert not new.as_jobs and not new.as_bank


@pytest.fixture
def observed():
    e = TransferredGroupGame(config())
    e.started_at = time.perf_counter()
    word = e.adjacency.sample_facet(e.adjacency.full_word, random.Random(29)).word
    node = e._node(SymmetryState(word, 'corrector', 0, IDENTITY_PATTERN_ID), 25)
    e.current_iteration = 1
    e._observe_terminal(node, path=[])
    return e, node


def test_geometry_is_322_and_start_has_no_default_class():
    e = TransferredGroupGame(config())
    assert e.adjacency.points.shape == (64, 26)
    assert e.adjacency.frame(e.adjacency.full_word).dimension == 26
    assert not e.discovered_classes and not e.discovery_events


def test_only_observed_facet_may_request_as(observed, monkeypatch):
    e, node = observed
    e.facet_bank.anchors.clear()
    monkeypatch.setattr(e.adjacency, 'new_job', lambda *a: pytest.fail('unobserved source'))
    assert e._as_remove(node, persistent=False) is None


def test_continuation_resets_local_budget_without_losing_history(observed):
    e, node = observed
    anchor = next(iter(next(iter(e.continuation_anchors.values())).values()))
    dummy = MacroAction('add', IDENTITY_PATTERN_ID, (), anchor.word, anchor.word,
                        'fixture', 'fixture', anchor.word.bit_count(), 1)
    anchor.witness = (dummy,)
    start, replay = e._iteration_start(4)
    assert not replay and start.state.corrections_used == 0
    assert start.state.support_word == node.facet_word
    assert e.episode_prefix == (dummy,) and e.episode_resets == (1,)
    assert e._validation_witness([]) == (dummy,)


def test_lazy_as_outputs_are_not_discoveries_and_actual_edge_is_certified(observed, monkeypatch):
    e, node = observed
    count = len(e.discovery_events)
    deadline = time.perf_counter()+10
    candidate = None
    while candidate is None and time.perf_counter() < deadline:
        candidate = e._as_remove(node, persistent=False)
    assert candidate is not None
    edge, state, rank = candidate
    assert rank == 24 and state.corrections_used == 1
    assert state.repair_source_word == node.facet_word
    monkeypatch.setattr(e.scorer, 'terminal_label', lambda *a: pytest.fail('early class lookup'))
    completion = e._as_completion(GameNode(state, rank))
    assert completion is not None and completion[2] == 25
    assert len(e.discovery_events) == count
    ridge = state.support_word
    assert completion[1].support_word == e.as_pairs[node.facet_word, ridge]
    assert adjacency_module.audit_adjacency_proof(e.scorer.points, e.as_bank[node.facet_word][ridge])


def test_budget_debt_does_not_run_geometry(observed, monkeypatch):
    e, node = observed
    e.as_compute_seconds = 1e6
    monkeypatch.setattr(e.adjacency, 'advance', lambda *a, **k: pytest.fail('budget debt'))
    assert e._as_remove(node, persistent=False) is None


def test_early_noop_requires_validated_parent_and_retained_full_rank(observed):
    e, node = observed
    word = node.state.support_word
    redundant = [1 << i for i in range(64) if word & (1 << i)
                 and e._rank(word & ~(1 << i)) == 25]
    assert redundant
    removed = redundant[0]
    action = MacroAction('remove', IDENTITY_PATTERN_ID, (), removed, removed,
                         'fixture', 'fixture', 1, 1, remove_word=removed)
    assert e._materialize_action(node, action) is None
    assert node.state.corrections_used == 0
    assert e.transfer_stats['certified_noop_skips'] == 1
    e.config = replace(e.config, enable_early_noop_guard=False)
    moved = e._materialize_action(node, action)
    assert moved[1] == 25 and moved[0].corrections_used == 1


def test_as_fallback_does_not_repeat_native_stop_draw(observed, monkeypatch):
    e, node = observed
    e.config = replace(e.config, rollout_corrector_probability=1)
    monkeypatch.setattr(e, '_as_selected', lambda node: True)
    monkeypatch.setattr(e, '_as_remove', lambda *a, **k: None)
    monkeypatch.setattr(SingleTreeGroupGame, '_rollout_edge',
                        lambda self, node, policy: self.rollout_rngs[policy].random())
    rng = e.rollout_rngs['mixed']
    state = rng.getstate()
    expected = rng.random()
    rng.setstate(state)
    assert e._rollout_edge(node, 'mixed') == expected


def test_full_short_run_passes_history_and_terminal_audit():
    e = TransferredGroupGame(config())
    payload = e.run()
    result = audit_result(payload)
    assert result['passed'], result
    assert result['terminals_and_paths_checked'] == len(payload['discoveries'])


@pytest.mark.parametrize('changes', [{'as_time_fraction': 0}, {'as_operator_probability': 2},
    {'as_min_slice_seconds': 1}, {'as_max_active_jobs': 100}, {'as_cutoff_vertices': 2}])
def test_bad_as_limits_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(TransferConfig(), **changes)
