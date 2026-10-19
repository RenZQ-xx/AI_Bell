import numpy as np
import pytest
from math import comb
from scipy.optimize import linprog

from baseline.bell322 import generate_bell322_points
from baseline.geometry import affine_rank, validate_facet_support
from baseline.orbit_blocks import build_state_group_permutations
from mcts.group_game.facet_corrector import FacetCorrectorBank
from mcts.group_game.knowledge import PatternSpec
from mcts.group_game.model import block_word
from mcts.group_game.orbit_faces import OrbitFaceEnumerator
from mcts.group_game.repair_cache import transported_pattern
from mcts.group_game.search import GroupGameConfig, build_parser
from mcts.subgroup_patterns import move_support_word


def _online_facet():
    points = generate_bell322_points(dtype=float)
    centered = points - points.mean(axis=0)
    lp = linprog(-np.random.default_rng(11).normal(size=26), A_ub=centered,
                 b_ub=np.ones(64), bounds=[(None, None)] * 26, method="highs")
    assert lp.success
    indices = list(map(int, np.flatnonzero(np.abs(centered @ lp.x - 1) < 1e-7)))
    check = validate_facet_support(points, indices)
    assert check.valid
    return points, block_word(tuple(indices)), check


def _verify_face(points, source, retained):
    inside = [i for i in range(64) if retained >> i & 1]
    removed = [i for i in range(64) if (source & ~retained) >> i & 1]
    affine = np.column_stack((points, np.ones(64)))
    certificate = linprog(np.zeros(27), A_eq=affine[inside], b_eq=np.zeros(len(inside)),
                          A_ub=-affine[removed], b_ub=-np.ones(len(removed)),
                          bounds=[(None, None)] * 27, method="highs")
    assert certificate.success


def test_orbit_candidates_are_bounded_supported_and_pattern_invariant():
    points, source, validation = _online_facet()
    bank = FacetCorrectorBank(points, seed=123)
    bank.observe(source, validation.normal, validation.offset)
    canonical, to_canonical, _ = bank.shared.coordinates[source]
    patterns = [transported_pattern(p, to_canonical) for p in bank.anchors[source].patterns]
    enumerator = OrbitFaceEnumerator(points, canonical, patterns, min_rank=17, max_candidates=256)
    other = OrbitFaceEnumerator(points, canonical, patterns, min_rank=17, max_candidates=256)
    found = []
    while not enumerator.finished:
        before = enumerator.examined
        result = enumerator.advance(8)
        assert 0 < enumerator.examined - before <= 8
        if result is not None:
            retained, pattern, rank = result
            assert not retained & ~canonical
            assert 17 <= rank <= 24
            assert rank == affine_rank(points[bank.indices(retained)])
            for block in pattern.blocks:
                mask = block_word(block)
                assert retained & mask in (0, mask)
            _verify_face(points, canonical, retained)
            found.append((retained, pattern.pattern_id, rank))
    assert found
    other_found = []
    while not other.finished:
        result = other.advance(1)
        if result is not None:
            other_found.append((result[0], result[1].pattern_id, result[2]))
    assert found == other_found
    assert enumerator.examined == min(enumerator.total, 256)
    assert enumerator.advance(8) is None
    stats = enumerator.stats
    assert sum(stats[k] for k in ("accepted", "rank_low", "rank_high", "not_supporting")) == enumerator.examined


@pytest.mark.parametrize("endpoint_dedup", [False, True])
def test_orbit_frontier_shared_progress_and_verified_publication(endpoint_dedup):
    points, source, validation = _online_facet()
    bank = FacetCorrectorBank(points, seed=123, max_attempts=0,
                             orbit_face_max_candidates=256, orbit_face_batch_size=8,
                             endpoint_dedup=endpoint_dedup, lower_face_max_attempts=6)
    bank.current_iteration = 1
    bank.observe(source, validation.normal, validation.offset)
    permutation = next(g for g in build_state_group_permutations()
                       if move_support_word(source, g) != source)
    moved = move_support_word(source, permutation)
    check = validate_facet_support(points, bank.indices(moved))
    bank.observe(moved, check.normal, check.offset)
    assert len(bank.orbit_faces) == 1
    enumerator = next(iter(bank.orbit_faces.values()))
    assert bank.has_delivery_frontier(source) and bank.has_delivery_frontier(moved)
    plans = []
    index = 0
    while not enumerator.finished:
        word = (source, moved)[index % 2]
        index += 1
        before = enumerator.examined
        plan = bank._grow_orbit_faces(bank.anchors[word])
        assert 0 < enumerator.examined - before <= 8
        if plan is not None:
            assert plan.source_word == word
            plans.append(plan)
            for g in bank.anchors[word].generators[plan.pattern_id]:
                assert move_support_word(word, g) == word
                assert move_support_word(plan.retained_word, g) == plan.retained_word
            for outside in plan.exit_words:
                assert outside & ~word
                assert validate_facet_support(points, bank.indices(plan.retained_word | outside)).valid
                assert bank.shared.uses(word, plan.retained_word, outside) == 0
    assert plans
    assert bank._grow_orbit_faces(bank.anchors[moved]) is None
    assert bank.to_dict()["publication_audit"]["unpublished_candidate_action_keys"] == 0


def test_orbit_config_disabled_and_bad_partition():
    assert build_parser().parse_args([]).orbit_face_max_candidates == 256
    assert GroupGameConfig(orbit_face_max_candidates=0).orbit_face_max_candidates == 0
    with pytest.raises(ValueError, match="orbit face budget"):
        GroupGameConfig(orbit_face_batch_size=0)
    with pytest.raises(ValueError, match="orbit face budget"):
        GroupGameConfig(orbit_face_max_candidates=-1)
    points, source, validation = _online_facet()
    bank = FacetCorrectorBank(points, seed=123)
    bank.observe(source, validation.normal, validation.offset)
    assert not bank.orbit_faces
    small = FacetCorrectorBank(points, seed=123, ridge_max_candidates=comb(source.bit_count(), 25),
                              orbit_face_max_candidates=256)
    small.observe(source, validation.normal, validation.offset)
    assert next(iter(small.ridge_enumerators.values())).eligible
    assert not small.orbit_faces
    bad = PatternSpec("bad", 0, "bad", (tuple(range(64)),), "test")
    with pytest.raises(ValueError, match="preserve"):
        OrbitFaceEnumerator(points, source, [bad], min_rank=17, max_candidates=8)
