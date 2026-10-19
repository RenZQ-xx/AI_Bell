from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy.optimize import linprog


actions = importlib.import_module("mcts.422.symmetric_actions")
search = importlib.import_module("mcts.422.symmetric_search_v28")
geometry = importlib.import_module("mcts.422.geometry")
model = importlib.import_module("mcts.422.model")
problem = importlib.import_module("mcts.422.problem")
atlas_module = importlib.import_module("mcts.422.atlas")
HANDOFF = atlas_module.DEFAULT_HANDOFF_ROOT


@pytest.mark.parametrize("classes,seconds,selected", [(114,400,"v27"),(116,900,"v28"),(115,700,"v28"),(115,760,None)])
def test_longrun_gate_preserves_coverage_priority(classes, seconds, selected) -> None:
    benchmark = importlib.import_module("mcts.422.symmetric_benchmark")
    reference = {"coverage": {"discovered_class_count": 115}, "runtime": {"elapsed_seconds": 759}}
    candidate = {"coverage": {"discovered_class_count": classes}, "runtime": {"elapsed_seconds": seconds}}
    assert benchmark.screening_decision(candidate, reference)["selected_version"] == selected


def test_coefficient_parameterization_preserves_lp_optimum() -> None:
    augmented = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=float)
    blocks = ((0, 1), (2, 3))
    chart = actions.CoefficientChart.build(augmented, blocks)
    objective = np.array([0.2, -1.3, 0.8])
    differences = np.array([augmented[1] - augmented[0], augmented[3] - augmented[2]])
    reference = linprog(
        objective, A_ub=-augmented, b_ub=np.zeros(4),
        A_eq=np.vstack((differences, augmented.sum(axis=0))),
        b_eq=np.array([0, 0, 1]), bounds=[(None, None)] * 3,
    )
    reduced = linprog(
        objective @ chart.basis,
        A_ub=-chart.full_slacks[list(chart.representatives)], b_ub=np.zeros(2),
        A_eq=chart.normalization[np.newaxis], b_eq=np.ones(1),
        bounds=[(None, None)] * chart.basis.shape[1],
    )
    assert reference.success and reduced.success
    assert reduced.fun == pytest.approx(reference.fun, abs=1e-9)
    assert chart.basis.shape == (3, 2)
    assert np.max(np.abs(differences @ chart.basis)) < 1e-9


def test_containing_chart_preserves_selected_equalities_and_rejects_full_support() -> None:
    augmented = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=float)
    chart = actions.CoefficientChart.build(augmented, ((0, 1), (2, 3)))
    containing = chart.containing(1)
    assert containing is not None
    assert containing.basis.shape[1] == 1
    assert np.max(np.abs(containing.full_slacks[0])) < 1e-9
    assert chart.containing(15) is None


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_relaxations_are_true_proper_subgroups_and_refine_orbits() -> None:
    atlas = actions.RelaxationPatternAtlas(HANDOFF)
    parent_id = "BFS422-H-0000"
    parent = set(atlas.subgroup(parent_id))
    children = atlas.proper_subgroups(parent_id)
    assert children[-1] == atlas_module.IDENTITY_PATTERN_ID
    assert len(children) == 4
    parent_blocks = [set(block) for block in atlas.blocks(parent_id)]
    for child_id in children:
        assert set(atlas.subgroup(child_id)) < parent
        for block in atlas.blocks(child_id):
            assert any(set(block) <= parent_block for parent_block in parent_blocks)


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_low_dimensional_proposal_is_full_space_facet_and_conditional_is_source_safe() -> None:
    atlas = actions.RelaxationPatternAtlas(HANDOFF)
    ranks = geometry.RankOracle()
    facets = geometry.FacetOracle(rank_oracle=ranks, use_catalog=False, canonicalize=False)
    generator = actions.SymmetricActionGenerator(atlas, ranks, facets, seed=26)
    state = model.SearchState(0)
    proposed = generator.supporting_face_actions(state, "BFS422-H-1195", nonce=0)
    assert proposed
    facet = facets.validate(proposed[0].add_word)
    assert facet.valid and facet.affine_rank == 79
    assert generator.coefficient_chart("BFS422-H-1195").basis.shape[1] < 81
    support = 0
    for vertex in problem.word_indices(facet.support_word):
        support |= 1 << vertex
        if ranks.rank(support) >= 72:
            break
    retained = model.SearchState(support, "filler", 1, "BFS422-H-1195", facet.support_word)
    alternatives = generator.conditional_completion_actions(
        retained, "BFS422-H-1195", nonce=0, relaxation_trials=2
    )
    for action in alternatives:
        terminal = facets.validate(support | action.add_word)
        assert terminal.valid
        assert terminal.support_word != facet.support_word
        assert support & ~terminal.support_word == 0


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_v28_smoke_keeps_empty_discovery_start_and_full_space_validation() -> None:
    engine = search.Bell422SymmetricSearch(search.SymmetricSearchConfig(
        iterations=2, max_depth=4, progress_interval=0,
        use_catalog=False, canonicalize_classes=False,
    ))
    assert not engine.discovered
    result = engine.run()
    assert result["features"]["verified_proper_subgroup_relaxation"]
    assert result["problem"]["total_class_count_known"] is False
    assert result["quotient_game"]["symmetric_lp"]["stats"]["lp_attempts"] > 0


@pytest.mark.skipif(not HANDOFF.exists(), reason="4-2-2 handoff is external")
def test_progress_recorder_preserves_baseline_actions_and_random_state() -> None:
    benchmark = importlib.import_module("mcts.422.symmetric_benchmark")
    baseline_module = importlib.import_module("mcts.422.quotient_search_v27")
    config = baseline_module.AdaptiveQuotientSearchConfig(
        iterations=2, seed=4222025, max_depth=4, progress_interval=0,
        use_catalog=False, canonicalize_classes=False,
    )
    baseline = baseline_module.Bell422AdaptiveQuotientSearch(config)
    recorded = benchmark.RecordedBaselineSearch(config)
    left, right = baseline.run(), recorded.run()
    assert baseline.rng.getstate() == recorded.rng.getstate()
    assert left["search_stats"] == right["search_stats"]
    assert left["coverage"]["discovered_class_ids"] == right["coverage"]["discovered_class_ids"]
    assert [event["path"] for event in left["coverage"]["discoveries"]] == [
        event["path"] for event in right["coverage"]["discoveries"]
    ]
