from __future__ import annotations

import importlib

import pytest


atlas_module = importlib.import_module("mcts.422.atlas")
geometry_module = importlib.import_module("mcts.422.geometry")
model_module = importlib.import_module("mcts.422.model")
actions_module = importlib.import_module("mcts.422.quotient_actions")
search_module = importlib.import_module("mcts.422.quotient_search")

HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_quotient_search_merges_pattern_contexts_at_one_geometry_node() -> None:
    config = search_module.QuotientSearchConfig(
        iterations=1,
        max_depth=2,
        progress_interval=0,
        use_catalog=False,
        canonicalize_classes=False,
    )
    engine = search_module.Bell422QuotientSearch(config, handoff_root=HANDOFF)
    left = model_module.SearchState(3, "filler", 1, "BFS422-H-0000", 7)
    right = model_module.SearchState(3, "filler", 1, "BFS422-H-0001", 7)

    first = engine._node(left, engine.rank_oracle.rank(3))
    second = engine._node(right, engine.rank_oracle.rank(3))

    assert first is second
    assert first.context_pattern_ids == {"BFS422-H-0000", "BFS422-H-0001"}
    assert engine.stats["merged_pattern_contexts"] == 1
    assert engine._pattern_limit(engine.root) > search_module.Bell422V25Search._pattern_limit(
        engine, engine.root
    )
    assert engine._child_limit(engine.root) > search_module.Bell422V25Search._child_limit(
        engine, engine.root
    )


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_common_refinement_proposes_full_space_filler_actions() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    ranks = geometry_module.RankOracle()
    facets = geometry_module.FacetOracle(
        rank_oracle=ranks,
        handoff_root=HANDOFF,
        use_catalog=False,
        canonicalize=False,
    )
    generator = actions_module.QuotientActionGenerator(
        atlas,
        ranks,
        facets,
        seed=26,
        quotient_candidate_limit=12,
    )
    state = model_module.SearchState(0, context_pattern_id="BFS422-H-0000")

    actions = generator.quotient_filler_actions(
        state,
        "BFS422-H-0000",
        ("BFS422-H-0001",),
        lane="single",
    )

    assert actions
    assert all(action.pattern_id == "BFS422-H-0000" for action in actions)
    assert any(action.structure == "quotient_refinement" for action in actions)
    assert all(action.add_word and not action.remove_word for action in actions)


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_supporting_face_lp_returns_a_full_rank_candidate_without_lift() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    ranks = geometry_module.RankOracle()
    facets = geometry_module.FacetOracle(
        rank_oracle=ranks,
        handoff_root=HANDOFF,
        use_catalog=False,
        canonicalize=False,
    )
    generator = actions_module.QuotientActionGenerator(
        atlas,
        ranks,
        facets,
        seed=26,
    )
    state = model_module.SearchState(0, context_pattern_id="BFS422-H-1195")

    actions = generator.supporting_face_actions(
        state,
        "BFS422-H-1195",
        nonce=0,
        count=1,
    )

    assert len(actions) == 1
    assert actions[0].source == "fullspace_supporting_face_lp"
    assert ranks.rank(actions[0].add_word) == 79
    assert facets.validate(actions[0].add_word).valid


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_ridge_frontier_consumes_each_full_space_exit_once() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    ranks = geometry_module.RankOracle()
    facets = geometry_module.FacetOracle(
        rank_oracle=ranks,
        handoff_root=HANDOFF,
        use_catalog=False,
        canonicalize=False,
    )
    generator = actions_module.QuotientActionGenerator(
        atlas,
        ranks,
        facets,
        seed=26,
        ridge_pivot=True,
    )
    source = int(
        "ffffffccffccffccccfffcfccfcfffccccffcfcffcfcffccccffccffccffffff",
        16,
    )
    neighbor = int(
        "ffffa3c4a3c4a3c4c4a3fafa8585a3c4c4a38585fafaa3c4c4a3c4a3c4a3ffff",
        16,
    )
    state = model_module.SearchState(
        source & neighbor,
        role="filler",
        corrections_used=1,
        context_pattern_id=atlas_module.IDENTITY_PATTERN_ID,
        source_facet_word=source,
    )

    first = generator._ridge_complete(state)
    second = generator._ridge_complete(state)

    assert first is not None
    assert first.facet.support_word == neighbor
    assert first.action.source == "fullspace_ridge_frontier"
    assert second is None
    assert generator.quotient_stats["ridge_exits_consumed"] == 1


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_quotient_search_smoke_records_v26_features() -> None:
    config = search_module.QuotientSearchConfig(
        iterations=2,
        max_depth=4,
        tail_checks=2,
        quotient_candidate_limit=4,
        quotient_secondary_limit=1,
        partial_rank_checks=2,
        ridge_rank_checks=4,
        progress_interval=0,
        use_catalog=False,
        canonicalize_classes=False,
    )

    result = search_module.Bell422QuotientSearch(
        config, handoff_root=HANDOFF
    ).run()

    assert result["algorithm"] == "bell422_fullspace_quotient_ridge_mcts_v26"
    assert result["runtime"]["iterations_completed"] == 2
    assert result["tree"]["root_visits"] == 2
    assert result["features"]["fullspace_geometric_transpositions"] is True
    assert result["features"]["reduced_to_full_lift"] is False


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_corrector_share_only_grows_after_corrector_discovery() -> None:
    config = search_module.QuotientSearchConfig(
        iterations=1,
        progress_interval=0,
        use_catalog=False,
        canonicalize_classes=False,
    )
    engine = search_module.Bell422QuotientSearch(config, handoff_root=HANDOFF)
    engine.discovered["filler-class"] = {}
    engine.discovery_events.append(
        {"path": [{"kind": "add"}]}
    )
    filler_only_ratio = engine._free_filler_ratio()
    engine.discovered["corrector-class"] = {}
    engine.discovery_events.append(
        {"path": [{"kind": "remove"}, {"kind": "add"}]}
    )

    assert filler_only_ratio == config.unproductive_corrector_filler_floor
    assert engine._free_filler_ratio() < filler_only_ratio
