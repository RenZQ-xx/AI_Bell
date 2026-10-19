from itertools import product
from math import comb

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from mcts.group_game.ridge_enumeration import RidgeEnumerator
from mcts.group_game.search import GroupGameConfig
from mcts.group_game.model import block_word


def enumerate_all(points, *, batch_size=7):
    enumerator = RidgeEnumerator(points, list(range(len(points))), max_candidates=4096)
    words = []
    while not enumerator.finished:
        before = enumerator.examined
        word = enumerator.advance(batch_size)
        assert 0 < enumerator.examined - before <= batch_size
        if word is not None:
            words.append(word)
    assert enumerator.advance(batch_size) is None
    return words, enumerator


def test_simplex_all_26_ridges_are_enumerated_once():
    points = np.zeros((26, 26))
    points[1:, :25] = np.eye(25)
    words, enumerator = enumerate_all(points)
    whole = (1 << 26) - 1
    assert set(words) == {whole ^ (1 << i) for i in range(26)}
    assert len(words) == enumerator.examined == 26
    assert enumerator.rejected == enumerator.duplicates == 0


def test_degenerate_ridges_include_all_coplanar_vertices():
    cube = np.asarray(list(product((-1., 1.), repeat=3)))
    points = np.column_stack((cube, np.zeros(8)))
    words, enumerator = enumerate_all(points, batch_size=2)
    expected = {block_word(tuple(map(int, np.flatnonzero(cube[:, axis] == side))))
                for axis in range(3) for side in (-1, 1)}
    assert set(words) == expected
    assert len(words) == 6
    assert enumerator.examined == comb(8, 3)
    assert enumerator.duplicates > 0
    assert enumerator.rejected > 0


def test_ridges_match_independent_convex_hull():
    local = np.random.default_rng(736).normal(size=(8, 4))
    points = np.column_stack((local, np.zeros(8)))
    words, _ = enumerate_all(points)
    expected = {block_word(tuple(map(int, face))) for face in ConvexHull(local).simplices}
    assert set(words) == expected


def test_candidate_cap_disables_work_without_truncating_coverage_claim():
    points = np.zeros((29, 26))
    enumerator = RidgeEnumerator(points, list(range(29)), max_candidates=4096)
    assert enumerator.total == comb(29, 25)
    assert not enumerator.eligible
    assert enumerator.finished
    assert enumerator.advance(32) is None
    assert enumerator.examined == 0


def test_double_pyramid_zero_gale_rows_do_not_expose_lower_rank_faces():
    local = np.asarray([[-1, -1, 0, 0], [-1, 1, 0, 0], [1, -1, 0, 0], [1, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    points = np.column_stack((local, np.zeros(6)))
    words, _ = enumerate_all(points)
    hull = ConvexHull(local)
    expected = {block_word(tuple(map(int, np.flatnonzero(abs(local @ eq[:-1] + eq[-1]) < 1e-8))))
                for eq in hull.equations}
    assert set(words) == expected
    assert 15 not in words


@pytest.mark.parametrize("kwargs", [{"ridge_max_candidates": -1}, {"ridge_batch_size": 0}])
def test_invalid_ridge_config(kwargs):
    with pytest.raises(ValueError, match="ridge budget"):
        GroupGameConfig(**kwargs)
