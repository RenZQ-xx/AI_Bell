from __future__ import annotations

import importlib

import pytest


atlas_module = importlib.import_module("mcts.422.atlas")
geometry_module = importlib.import_module("mcts.422.geometry")
model_module = importlib.import_module("mcts.422.model")
search_module = importlib.import_module("mcts.422.quotient_search_v27")

HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT


def test_rank_oracle_reports_lru_evictions() -> None:
    oracle = geometry_module.RankOracle(max_entries=2)
    oracle.rank(1)
    oracle.rank(3)
    oracle.rank(7)

    assert oracle.to_dict()["max_entries"] == 2
    assert oracle.to_dict()["evictions"] >= 1


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v27_preserves_embedding_bridges_and_uses_larger_caches() -> None:
    config = search_module.AdaptiveQuotientSearchConfig(
        iterations=1,
        progress_interval=0,
        use_catalog=False,
    )
    engine = search_module.Bell422AdaptiveQuotientSearch(
        config,
        handoff_root=HANDOFF,
    )

    assert engine.rank_oracle.max_entries == config.rank_cache_size
    assert engine.actions.candidate_cache_size == config.candidate_cache_size
    assert engine.actions.quotient_cache_size == config.quotient_cache_size
    assert engine.actions.unseen_exit_first is False
    assert engine.actions.canonical_frontier_dedup is False
    assert engine.actions.ridge_exit_scan == config.ridge_exit_scan


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v27_corrector_dominance_keeps_periodic_reseed() -> None:
    config = search_module.AdaptiveQuotientSearchConfig(
        iterations=1,
        progress_interval=0,
        use_catalog=False,
        filler_warmup_iterations=10,
        adaptive_schedule_min_iteration=20,
        filler_discovery_stagnation=1,
        recent_yield_window=100,
        recent_yield_min_discoveries=2,
        reseed_interval=200,
        reseed_span=20,
        enable_adaptive_schedule=True,
    )
    engine = search_module.Bell422AdaptiveQuotientSearch(
        config,
        handoff_root=HANDOFF,
    )
    engine.current_iteration = 50
    engine.discovery_events.extend(
        [
            {"iteration": 40, "path": [{"kind": "remove"}, {"kind": "add"}]},
            {"iteration": 45, "path": [{"kind": "rewrite"}, {"kind": "add"}]},
        ]
    )
    engine.discovered.update({"left": {}, "right": {}})

    assert engine._free_filler_ratio() == config.productive_corrector_filler_floor
    engine.current_iteration = 205
    assert engine._free_filler_ratio() >= config.reseed_filler_ratio


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v27_smoke_records_adaptive_features_and_canonical_rows() -> None:
    config = search_module.AdaptiveQuotientSearchConfig(
        iterations=2,
        max_depth=4,
        tail_checks=2,
        quotient_candidate_limit=4,
        quotient_secondary_limit=1,
        partial_rank_checks=2,
        ridge_rank_checks=4,
        ridge_basis_rank_checks=8,
        progress_interval=0,
        use_catalog=False,
    )
    result = search_module.Bell422AdaptiveQuotientSearch(
        config,
        handoff_root=HANDOFF,
    ).run()

    assert result["algorithm"] == "bell422_adaptive_quotient_ridge_mcts_v27"
    assert result["features"]["canonical_ridge_endpoint_pair_tracking"] is False
    assert result["features"]["concrete_embedding_bridges_preserved"] is True
    assert result["problem"]["total_class_count_known"] is False
    for event in result["coverage"]["discoveries"]:
        assert len(event["canonical_h_row"]) == 81
