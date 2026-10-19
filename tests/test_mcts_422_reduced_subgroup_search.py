from __future__ import annotations

import importlib
import random

import pytest


module = importlib.import_module("mcts.422.reduced_subgroup_search")
HANDOFF = module.DEFAULT_HANDOFF_ROOT


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_h0000_uses_intrinsic_facet_and_ridge_ranks() -> None:
    problem = module.ReducedSubgroupProblem.load(HANDOFF, "BFS422-H-0000")

    assert problem.intrinsic_dimension == 23
    assert problem.vertex_count == 48
    assert problem.facet_rank == 22
    assert problem.ridge_rank == 21
    assert problem.reduced_point_count == 80


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_supporting_face_sampler_produces_valid_reduced_facets() -> None:
    problem = module.ReducedSubgroupProblem.load(HANDOFF, "BFS422-H-0000")
    geometry = module.ReducedGeometry(problem, handoff_root=HANDOFF)

    facets = geometry.sample_supporting_facets(
        0,
        count=3,
        rng=random.Random(1),
    )

    assert len(facets) == 3
    assert all(geometry.rank_oracle.rank(word) == 22 for word in facets)
    assert all(geometry.validate(word).valid for word in facets)
    assert any(geometry.validate(word).lifted for word in facets)


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_reduced_search_smoke_discovers_a_lifted_class() -> None:
    problem = module.ReducedSubgroupProblem.load(HANDOFF, "BFS422-H-0000")
    config = module.ReducedSearchConfig(
        iterations=4,
        seed=1,
        progress_interval=0,
    )
    engine = module.ReducedSubgroupMCTS(
        problem,
        config,
        handoff_root=HANDOFF,
    )

    result = engine.run()

    assert result["problem"]["reduced_facet_affine_rank"] == 22
    assert result["reduced_facets"]["unique_count"] >= 1
    assert result["coverage"]["discovered_class_count"] >= 1
    assert result["features"]["persistent_corrector_face_banks"] is True
    assert result["tree"]["root_basin_count"] >= 1


def test_productive_correction_limit_extends_only_known_basins() -> None:
    config = module.ReducedSearchConfig(
        max_corrections=5,
        productive_max_corrections=7,
    )

    assert config.max_corrections == 5
    assert config.productive_max_corrections == 7
    with pytest.raises(ValueError):
        module.ReducedSearchConfig(
            max_corrections=5,
            productive_max_corrections=4,
        )
