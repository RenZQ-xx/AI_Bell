from itertools import product

import numpy as np

from mcts.symmetry_game.patterns import PatternRelationIndex, geometric_state_key
from mcts.symmetry_game.ridge import adjacent_facet_words


def _word(indices) -> int:
    return sum(1 << int(index) for index in indices)


def test_pattern_relations_distinguish_refinement_and_intersection() -> None:
    index = PatternRelationIndex(4)
    index.register("coarse", ((0, 1), (2, 3)))
    index.register("fine", ((0,), (1,), (2,), (3,)))
    index.register("cross", ((0, 2), (1, 3)))

    assert index.relation("coarse", "coarse") == "continue"
    assert index.relation("coarse", "fine") == "refine"
    assert index.relation("coarse", "cross") == "intersection"
    assert index.directed_relation("coarse", "fine") == "refine"
    assert index.directed_relation("fine", "coarse") == "coarsen"
    assert index.directed_relation("coarse", "cross") == "intersection"
    assert index.relation("fine", "coarse") == "intersection"
    assert index.coherence(_word((0, 1)), "coarse") == 1.0
    assert index.coherence(_word((0, 1)), "cross") == 0.0
    assert index.common_refinement_size(("coarse", "cross")) == 4


def test_geometric_state_key_does_not_depend_on_pattern_history() -> None:
    assert geometric_state_key(17, "filler", 2, 99) == (17, "filler", 2, 99)


def test_ridge_pivot_supports_words_wider_than_64_bits() -> None:
    cube = list(product((-1.0, 1.0), repeat=3))
    points = np.asarray([cube[index % len(cube)] for index in range(70)])
    source = _word(i for i, point in enumerate(points) if point[0] == 1.0)
    ridge = _word(i for i, point in enumerate(points) if point[0] == point[1] == 1.0)
    neighbor = _word(i for i, point in enumerate(points) if point[1] == 1.0)

    result = adjacent_facet_words(points, source, ridge)

    assert source.bit_length() > 64
    assert set(result.facet_words) == {source, neighbor}
