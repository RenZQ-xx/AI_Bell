from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


atlas_module = importlib.import_module("mcts.422.atlas")
fc_module = importlib.import_module("mcts.422.filler_corrector")
geometry_module = importlib.import_module("mcts.422.geometry")
model_module = importlib.import_module("mcts.422.model")
problem_module = importlib.import_module("mcts.422.problem")
search_module = importlib.import_module("mcts.422.search")


HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT


def test_problem_definition_matches_frozen_handoff() -> None:
    points = problem_module.deterministic_vertices()
    assert points.shape == (256, 80)
    assert np.all(np.isin(points, (-1, 1)))
    generators = problem_module.vertex_symmetry_generators()
    assert len(generators) == 15
    assert all(sorted(permutation) == list(range(256)) for _name, permutation in generators)


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_atlas_reads_structures_and_lazy_orbit_partition_only() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF, cache_size=4)
    assert len(atlas) == 722
    assert len(atlas.patterns) == 722
    blocks = atlas.blocks("BFS422-H-0000")
    assert len(blocks) == 80
    assert sorted(vertex for block in blocks for vertex in block) == list(range(256))
    assert atlas.to_dict()["answer_files_read_by_atlas"] == []


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_known_handoff_facet_round_trips_through_terminal_oracle() -> None:
    facet_path = HANDOFF / "02_complete_nodes" / "nodes" / "BFS422-H-0000" / "facets.csv"
    with facet_path.open(newline="", encoding="utf-8") as handle:
        source = next(csv.DictReader(handle))
    h_row = tuple(map(int, json.loads(source["h_row"])))
    augmented = np.column_stack(
        (np.ones(256, dtype=int), problem_module.deterministic_vertices())
    )
    values = augmented @ np.asarray(h_row)
    support = sum(1 << index for index, value in enumerate(values) if value == 0)
    oracle = geometry_module.FacetOracle(handoff_root=HANDOFF)
    result = oracle.validate(support)
    assert result.valid
    assert result.h_row == h_row
    assert result.class_id == source["facet_class_id"]
    assert result.support_word == support


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_filler_supports_single_and_union_orbit_actions() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    ranks = geometry_module.RankOracle()
    facets = geometry_module.FacetOracle(
        rank_oracle=ranks, handoff_root=HANDOFF, use_catalog=False, canonicalize=False
    )
    actions = fc_module.FillerCorrector(atlas, ranks, facets, seed=7)
    state = model_module.SearchState(0, context_pattern_id="BFS422-H-0000")
    singles = actions.filler_actions(state, "BFS422-H-0000", lane="single")
    unions = actions.filler_actions(state, "BFS422-H-0000", lane="union")
    assert singles and all(action.kind == "add" for action in singles)
    assert any(action.arity == 2 for action in unions)


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_minimal_search_builds_one_persistent_tree_without_prior_classes() -> None:
    config = search_module.SearchConfig(
        iterations=2,
        max_depth=4,
        rollout_depth=4,
        tail_checks=2,
        progress_interval=0,
        use_catalog=False,
        canonicalize_classes=False,
    )
    result = search_module.Bell422Search(config, handoff_root=HANDOFF).run()
    assert result["runtime"]["iterations_completed"] == 2
    assert result["tree"]["node_count"] >= 2
    assert result["tree"]["root_visits"] == 2
    assert result["prior_boundary"]["initial_discovered_classes"] == []
    assert result["prior_boundary"]["never_read"] == ["nodes/*/facets.csv"]


def test_geometric_state_key_merges_pattern_only_context() -> None:
    left = model_module.SearchState(3, "filler", 1, "left", 7)
    right = model_module.SearchState(3, "filler", 1, "right", 7)

    assert left.key != right.key
    assert left.geometry_key == right.geometry_key


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_context_free_action_cache_reuses_geometric_state() -> None:
    atlas = atlas_module.HandoffPatternAtlas(HANDOFF)
    ranks = geometry_module.RankOracle()
    facets = geometry_module.FacetOracle(
        rank_oracle=ranks, handoff_root=HANDOFF, use_catalog=False, canonicalize=False
    )
    actions = fc_module.FillerCorrector(
        atlas,
        ranks,
        facets,
        seed=7,
        context_free_cache_keys=True,
    )
    left = model_module.SearchState(0, context_pattern_id="BFS422-H-0000")
    right = model_module.SearchState(0, context_pattern_id="BFS422-H-0001")

    actions.filler_actions(left, "BFS422-H-0000", lane="single")
    actions.filler_actions(right, "BFS422-H-0000", lane="single")

    assert actions.stats["candidate_cache_misses"] == 1
    assert actions.stats["candidate_cache_hits"] == 1


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_dimension_independent_pivot_finds_a_422_facet_neighbor() -> None:
    ridge_module = importlib.import_module("mcts.symmetry_game.ridge")
    source = int(
        "ffffffccffccffccccfffcfccfcfffccccffcfcffcfcffccccffccffccffffff",
        16,
    )
    neighbor = int(
        "ffffa3c4a3c4a3c4c4a3fafa8585a3c4c4a38585fafaa3c4c4a3c4a3c4a3ffff",
        16,
    )

    result = ridge_module.adjacent_facet_words(
        problem_module.deterministic_vertices(),
        source,
        source & neighbor,
    )

    assert set(result.facet_words) == {source, neighbor}
    assert result.candidate_rays == 2
