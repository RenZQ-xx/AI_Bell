from __future__ import annotations

import importlib

import pytest


global_module = importlib.import_module("mcts.422.global_reduced_search")
reduced_module = importlib.import_module("mcts.422.reduced_subgroup_search")
HANDOFF = global_module.DEFAULT_HANDOFF_ROOT


def test_shared_discovery_state_deduplicates_classes_globally() -> None:
    state = reduced_module.SharedDiscoveryState()
    event = {"class_id": "F-1", "node_id": "H-1"}

    assert state.observe("F-1", event) == (True, 0)
    assert state.observe("F-1", None) == (False, 1)
    assert state.epoch == 1
    assert state.class_frequencies["F-1"] == 2
    assert len(state.discovery_events) == 1


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_global_inventory_contains_all_facet_blind_subgroup_lanes() -> None:
    engine = global_module.GlobalReducedSearch(
        global_module.GlobalReducedConfig(iterations=1, progress_interval=0),
        handoff_root=HANDOFF,
    )

    assert len(engine.lanes) == 721
    assert all(not lane.activated for lane in engine.lanes.values())


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_global_racing_shares_oracle_and_discovers_across_lanes() -> None:
    config = global_module.GlobalReducedConfig(
        iterations=8,
        seed=11,
        warmup_iterations=1,
        race_chunk=2,
        progress_interval=0,
    )
    lane_config = reduced_module.ReducedSearchConfig(
        iterations=8,
        seed=11,
        progress_interval=0,
    )
    engine = global_module.GlobalReducedSearch(
        config,
        handoff_root=HANDOFF,
        node_ids=("BFS422-H-0000", "BFS422-H-0022"),
        lane_config=lane_config,
    )

    result = engine.run()

    assert result["runtime"]["iterations_completed"] == 8
    assert result["scheduler"]["activated_lane_count"] == 2
    assert result["coverage"]["discovered_class_count"] >= 1
    assert all(
        lane.engine is not None
        and lane.engine.geometry.full_oracle is engine.full_oracle
        for lane in engine.lanes.values()
    )
    assert result["prior_boundary"]["target_labels_affect_search"] is False
    assert "nodes/*/facets.csv" in result["prior_boundary"]["never_used"]
    assert result["algorithm"].endswith("_v2")
    assert result["features"]["stage_aware_lane_dormancy"] is True
    assert "revive" in result["config"]["race_cycle"]
