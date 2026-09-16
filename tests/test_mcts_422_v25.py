from __future__ import annotations

import importlib

import pytest


atlas_module = importlib.import_module("mcts.422.atlas")
fc_module = importlib.import_module("mcts.422.filler_corrector")
geometry_module = importlib.import_module("mcts.422.geometry")
model_module = importlib.import_module("mcts.422.model")
search_module = importlib.import_module("mcts.422.v25_search")

HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v25_preserves_independent_embedding_and_widening_design() -> None:
    config = search_module.V25SearchConfig(
        iterations=1,
        max_depth=3,
        progress_interval=0,
        use_catalog=False,
        canonicalize_classes=False,
    )
    engine = search_module.Bell422V25Search(config, handoff_root=HANDOFF)

    assert engine.actions.filler_max_orbits_per_action == 2
    assert engine.actions.corrector_max_orbits_per_action == 4
    assert engine.actions.max_pattern_embeddings_per_family == 4
    assert engine.knowledge.to_dict()["share_embedding_family_values"] is False
    assert engine._pattern_limit(engine.root) >= 1
    assert engine._block_limit(engine.root) >= 1
    assert engine._child_limit(engine.root) >= 1
    assert search_module.ROLLOUT_POLICY_CYCLE.count("mixed") == 6


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v25_broad_corrector_reaches_four_orbit_edits() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    ranks = geometry_module.RankOracle()
    facets = geometry_module.FacetOracle(
        rank_oracle=ranks,
        handoff_root=HANDOFF,
        use_catalog=False,
        canonicalize=False,
    )
    generator = fc_module.FillerCorrector(
        atlas,
        ranks,
        facets,
        seed=25,
        corrector_candidate_limit=24,
        corrector_max_orbits_per_action=4,
    )
    state = model_module.SearchState(
        (1 << 256) - 1,
        role="corrector",
        context_pattern_id=atlas_module.IDENTITY_PATTERN_ID,
    )

    actions = generator.corrector_actions(
        state, atlas_module.IDENTITY_PATTERN_ID, lane="remove_broad"
    )

    assert any(action.arity == 4 for action in actions)
    assert all(action.kind == "remove" for action in actions)


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v25_embedding_family_is_structural_not_value_shared() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    first = atlas.register_online_pattern(
        "TEST422-FAMILY-E0",
        ((index,) for index in range(256)),
        structure="test",
        family_id="TEST422-FAMILY",
    )
    second = atlas.register_online_pattern(
        "TEST422-FAMILY-E1",
        (tuple(range(256)),),
        structure="test",
        family_id="TEST422-FAMILY",
    )

    assert first.statistical_family_id == second.statistical_family_id
    assert atlas.family_embeddings("TEST422-FAMILY") == (
        "TEST422-FAMILY-E0",
        "TEST422-FAMILY-E1",
    )

    knowledge = importlib.import_module("mcts.422.v25_knowledge").V25OnlineKnowledge(
        atlas
    )
    action = model_module.MacroAction(
        "add",
        first.pattern_id,
        (0,),
        1,
        "test",
        "test",
        add_word=1,
    )
    knowledge.update_action(
        "filler",
        action,
        rank_before=0,
        rank_after=1,
        reward=3.0,
        exact_hit=False,
        new_class=False,
        context_pattern_id=atlas_module.IDENTITY_PATTERN_ID,
        relation="reembed",
    )

    assert knowledge.pattern_score("filler", first.pattern_id) != knowledge.pattern_score(
        "filler", second.pattern_id
    )


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v25_smoke_builds_one_tree_without_seeding_a_class(tmp_path) -> None:
    config = search_module.V25SearchConfig(
        iterations=2,
        max_depth=4,
        tail_checks=2,
        progress_interval=0,
        use_catalog=False,
        canonicalize_classes=False,
    )
    checkpoint = tmp_path / "i1.json"
    result = search_module.Bell422V25Search(
        config, handoff_root=HANDOFF
    ).run(checkpoint_paths={1: checkpoint})

    assert result["runtime"]["iterations_completed"] == 2
    assert result["tree"]["root_visits"] == 2
    assert result["tree"]["node_count"] >= 3
    assert result["tree"]["edge_count"] <= 2
    assert result["prior_boundary"]["initial_discovered_classes"] == []
    assert result["features"]["persistent_expansions_per_simulation"] == 1
    assert result["features"]["nonterminal_leaf_value"] == 0.0
    snapshot = __import__("json").loads(checkpoint.read_text(encoding="utf-8"))
    assert snapshot["config"]["iterations"] == 1
    assert snapshot["runtime"]["iterations_completed"] == 1
    assert snapshot["runtime"]["planned_iterations"] == 2


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v25_wall_limit_returns_a_complete_truncated_result() -> None:
    config = search_module.V25SearchConfig(
        iterations=5,
        max_depth=2,
        tail_checks=1,
        progress_interval=0,
        use_catalog=False,
        canonicalize_classes=False,
        max_wall_seconds=1e-9,
    )

    result = search_module.Bell422V25Search(
        config, handoff_root=HANDOFF
    ).run()

    assert result["runtime"]["iterations_completed"] == 1
    assert result["runtime"]["planned_iterations"] == 5
    assert result["runtime"]["termination_reason"] == "wall_time_limit"
    assert result["runtime"]["truncated"] is True
