from types import SimpleNamespace

import mcts.group_game.hierarchical_search as hierarchical_search
from mcts.group_game.search import GroupGameConfig
from tests.test_group_game_search import _engine


def test_directed_hierarchy_recognizes_coarsening_from_identity() -> None:
    engine = _engine(
        hierarchical_pattern_policy=True,
        directed_pattern_relations=True,
    )

    assert engine._node_pattern_relation(engine.root, "PAIR") == "coarsen"


def test_relation_value_updates_are_isolated_behind_switch() -> None:
    engine = _engine(
        hierarchical_pattern_policy=True,
        directed_pattern_relations=True,
        learn_relation_values=True,
    )
    action = engine.candidate_actions_for_pattern(engine.root, "PAIR")[0]

    engine.knowledge.update_action(
        "filler",
        action,
        rank_before=0,
        rank_after=2,
        reward=1.5,
        exact_hit=False,
        new_class=False,
        context_pattern_id=engine.root.state.context_pattern_id,
        relation="coarsen",
        update_relation=True,
    )

    stats = engine.knowledge.relation_stats[("filler", "coarsen")]
    assert stats.visits == 1
    assert stats.mean_reward == 1.5


def test_filler_reheat_requires_class_and_frontier_stagnation() -> None:
    engine = _engine(
        adaptive_corrector_replay=True,
        adaptive_filler_reheat=True,
        target_classes=(1, 2, 3, 4),
        adaptive_replay_min_coverage=0.50,
        filler_reheat_class_patience=512,
        filler_reheat_frontier_patience=256,
        filler_reheat_ratio=0.35,
    )
    engine.class_terminal_counts.update({1: 1, 2: 1, 3: 1})
    engine.discovery_events.append(SimpleNamespace(iteration=400))
    engine.current_iteration = 1000
    engine.last_corrector_frontier_progress_iteration = 700

    assert engine._filler_reheat_active()
    assert engine._free_filler_ratio() == 0.35

    engine.last_corrector_frontier_progress_iteration = 900
    assert not engine._filler_reheat_active()
    assert engine._free_filler_ratio() < 0.15


def test_v23_entry_keeps_geometric_node_merging_disabled(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr(hierarchical_search, "run_main", fake_main)
    hierarchical_search.main(["--iterations", "2", "--output", "test.json"])

    arguments = captured["arguments"]
    assert "--hierarchical-pattern-policy" in arguments
    assert "--directed-pattern-relations" in arguments
    assert "--learn-relation-values" in arguments
    assert "--adaptive-filler-reheat" in arguments
    assert "--geometric-transpositions" not in arguments


def test_v23_switches_remain_opt_in() -> None:
    config = GroupGameConfig()

    assert not config.directed_pattern_relations
    assert not config.learn_relation_values
    assert not config.adaptive_filler_reheat
