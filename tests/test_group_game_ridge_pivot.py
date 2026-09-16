from itertools import product

import numpy as np

from mcts.group_game.model import block_word
from mcts.group_game.ridge_pivot import adjacent_facet_words


def test_cube_edge_pivots_to_exactly_two_incident_facets() -> None:
    points = np.asarray(list(product((-1.0, 1.0), repeat=3)))
    source = block_word(tuple(i for i, p in enumerate(points) if p[0] == 1.0))
    ridge = block_word(tuple(i for i, p in enumerate(points) if p[0] == p[1] == 1.0))
    neighbor = block_word(tuple(i for i, p in enumerate(points) if p[1] == 1.0))

    result = adjacent_facet_words(points, source, ridge)

    assert set(result.facet_words) == {source, neighbor}
    assert result.reason is None
    assert result.candidate_rays == 2


def test_simplex_ridge_returns_source_and_other_facet() -> None:
    points = np.vstack((np.zeros(4), np.eye(4)))
    source = block_word((1, 2, 3, 4))
    ridge = block_word((2, 3, 4))
    neighbor = block_word((0, 2, 3, 4))

    result = adjacent_facet_words(points, source, ridge)

    assert set(result.facet_words) == {source, neighbor}


def test_non_ridge_is_rejected_before_pivoting() -> None:
    points = np.asarray(list(product((-1.0, 1.0), repeat=3)))
    source = block_word(tuple(i for i, p in enumerate(points) if p[0] == 1.0))
    vertex = block_word((int(np.flatnonzero(np.all(points == (1.0, 1.0, 1.0), axis=1))[0]),))

    result = adjacent_facet_words(points, source, vertex)

    assert result.facet_words == ()
    assert result.reason == "insufficient_vertices"
