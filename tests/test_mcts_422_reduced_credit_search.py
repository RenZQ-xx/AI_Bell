from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


credit = importlib.import_module("mcts.422.reduced_credit_search")
reduced = importlib.import_module("mcts.422.reduced_subgroup_search")


def make_credit(index=0, *, utility=0.25, persistent=True):
    return credit.CorrectionCredit(index, 31, utility, -0.06, persistent)


def test_novelty_pool_is_shared_once_and_recency_weighted():
    steps = [make_credit(1), make_credit(3)]
    shares = credit.distribute_novelty(steps, terminal_index=4, reward=10, discount=0.985)
    assert sum(shares) == pytest.approx(10)
    assert shares[1] > shares[0]
    assert [step.novelty_credit for step in steps] == shares


def test_credit_edge_does_not_learn_large_refill_return():
    record = make_credit()
    record.outcome_utility = 0.25
    action = reduced.ReducedAction("remove", retained_word=3)
    edge = credit.CreditEdge(action, (3, "filler", 1, 31), 4, 1, None, 0.25, credit=record)
    edge.update(100, 2)
    assert edge.mean(2) == pytest.approx(-0.06)
    assert edge.visits == 1


def test_missing_endpoint_cannot_improve_source_utility():
    record = make_credit()
    assert record.value == pytest.approx(-0.31)
    record.outcome_utility = -0.5
    assert record.value == pytest.approx(-0.81)


def test_failed_lift_has_zero_utility_without_being_pruned():
    engine = object.__new__(credit.ReducedCreditMCTS)
    engine.config = credit.ReducedCreditConfig()
    terminal = reduced.ReducedFacetResult(True, "ok", 31, 31, full_result=None)
    assert engine._utility(terminal) == 0


def test_dead_end_resolves_pending_credit():
    engine = object.__new__(credit.ReducedCreditMCTS)
    engine.config = credit.ReducedCreditConfig()
    record = make_credit()
    engine.pending_credit = record
    engine._fail_pending()
    assert record.outcome_utility == engine.config.dead_end_reward
    assert engine.pending_credit is None


def test_filler_edge_still_learns_the_ordinary_return():
    action = reduced.ReducedAction("add", add_word=3)
    edge = credit.CreditEdge(action, (3, "filler", 0, 0), -1, 1, None, 0.25)
    edge.update(12, 1)
    assert edge.mean(1) == 12


def test_utility_frequency_penalty_uses_observed_hits_only():
    engine = object.__new__(credit.ReducedCreditMCTS)
    engine.config = credit.ReducedCreditConfig()
    engine.shared_discovery = reduced.SharedDiscoveryState()
    terminal = reduced.ReducedFacetResult(
        True, "ok", 31, 31,
        full_result=SimpleNamespace(valid=True, class_id="observed"),
    )
    fresh = engine._utility(terminal)
    engine.shared_discovery.class_frequencies["observed"] = 10000
    assert fresh == 0.25
    assert engine._utility(terminal) == pytest.approx(0.05)


@pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="4-2-2 handoff is external")
def test_disabled_credit_matches_baseline_paths_values_and_rng():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-0000")
    base = reduced.ReducedSubgroupMCTS(
        problem, reduced.ReducedSearchConfig(iterations=12, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    variant = credit.ReducedCreditMCTS(
        problem, credit.ReducedCreditConfig(
            iterations=12, seed=17, progress_interval=0, enable_counterfactual_credit=False
        ),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    first, second = base.run(), variant.run()
    assert first["coverage"] == second["coverage"]
    assert first["tree"] == second["tree"]
    assert first["search_stats"] == second["search_stats"]
    assert base.rng.getstate() == variant.rng.getstate()
    assert {key: rng.getstate() for key, rng in base.rollout_rngs.items()} == {
        key: rng.getstate() for key, rng in variant.rollout_rngs.items()
    }
    for key, node in base.nodes.items():
        other = variant.nodes[key]
        assert (node.visits, node.value_sum) == (other.visits, other.value_sum)
        for action_id, edge in node.edges.items():
            alternative = other.edges[action_id]
            assert (edge.visits, edge.value_sum, edge.epoch_value_sum) == (
                alternative.visits, alternative.value_sum, alternative.epoch_value_sum
            )


@pytest.mark.skipif(not reduced.DEFAULT_HANDOFF_ROOT.exists(), reason="4-2-2 handoff is external")
def test_enabled_credit_records_exact_conserved_pools_and_preserves_bridges():
    problem = reduced.ReducedSubgroupProblem.load(reduced.DEFAULT_HANDOFF_ROOT, "BFS422-H-1171")
    engine = credit.ReducedCreditMCTS(
        problem, credit.ReducedCreditConfig(iterations=12, seed=17, progress_interval=0),
        handoff_root=reduced.DEFAULT_HANDOFF_ROOT,
    )
    result = engine.run()
    assert result["search_stats"]["corrector_credit_observations"] > 0
    assert result["corrector_credit"]["failed_lifts_remain_searchable"] is True
    for event in result["corrector_credit"]["discovery_events"]:
        assert sum(item["share"] for item in event["contributors"]) == pytest.approx(10)


@pytest.mark.parametrize("kwargs", [
    {"corrector_exact_utility": float("nan")},
    {"corrector_utility_frequency_penalty": -0.1},
    {"corrector_utility_penalty_cap": 0.5},
])
def test_invalid_utility_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        credit.ReducedCreditConfig(**kwargs)
