from mcts.group_game.model import SymmetryState
from tests.test_group_game_search import _engine


def test_geometric_transposition_merges_pattern_histories() -> None:
    engine = _engine(
        geometric_transpositions=True,
        hierarchical_pattern_policy=True,
    )
    pattern_ids = sorted(engine.knowledge.active_pattern_ids)
    assert len(pattern_ids) >= 2
    first, second = pattern_ids[:2]
    state_a = SymmetryState(1, "filler", 0, first)
    state_b = SymmetryState(1, "filler", 0, second)

    node_a = engine._node(state_a, 0)
    node_b = engine._node(state_b, 0)

    assert node_a is node_b
    assert {first, second} <= node_a.context_pattern_ids


def test_legacy_nodes_keep_pattern_in_identity() -> None:
    engine = _engine(geometric_transpositions=False)
    pattern_ids = sorted(engine.knowledge.active_pattern_ids)
    first, second = pattern_ids[:2]

    node_a = engine._node(SymmetryState(1, "filler", 0, first), 0)
    node_b = engine._node(SymmetryState(1, "filler", 0, second), 0)

    assert node_a is not node_b


def test_hierarchical_pattern_relation_uses_partition_certificates() -> None:
    engine = _engine(
        geometric_transpositions=True,
        hierarchical_pattern_policy=True,
    )
    node = engine.root
    candidate = next(
        pattern_id
        for pattern_id in sorted(engine.knowledge.active_pattern_ids)
        if pattern_id not in node.context_pattern_ids
    )

    assert engine._node_pattern_relation(node, candidate) in {
        "refine",
        "intersection",
        "identity",
    }


def test_v22_switches_remain_opt_in() -> None:
    engine = _engine()

    assert not engine.config.geometric_transpositions
    assert not engine.config.hierarchical_pattern_policy
