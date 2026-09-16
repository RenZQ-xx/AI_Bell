from types import SimpleNamespace

import mcts.group_game.policy_reward_ablation_search as ablation_search
from mcts.group_game.model import EdgeStats, GameNode, MacroAction, SymmetryState
from mcts.group_game.search import GroupGameConfig, PathStep
from tests.test_group_game_search import _engine


def _add_edge() -> EdgeStats:
    action = MacroAction(
        kind="add",
        pattern_id="PAIR",
        orbit_indices=(0,),
        block_word=0b11,
        effective_word=0b11,
        source="atlas",
        structure="C2",
        block_size=2,
        arity=1,
    )
    return EdgeStats(
        action=action,
        child_key=(),
        prior=1.0,
        rank_before=0,
        rank_after=2,
    )


def test_policy_reward_ablation_switches_are_opt_out() -> None:
    config = GroupGameConfig()
    assert config.multilevel_progressive_widening
    assert config.size_bucket_widening
    assert config.action_lane_widening
    assert config.weighted_relation_widening
    assert config.transition_reward_shaping
    assert config.filler_transition_reward_shaping
    assert config.corrector_transition_reward_shaping
    assert config.terminal_support_reward_shaping
    assert config.terminal_frequency_reward_shaping
    assert config.corrector_novelty_credit
    assert config.endpoint_novelty_credit
    assert config.leaf_value_shaping


def test_child_only_widening_removes_pattern_and_block_formulas() -> None:
    engine = _engine(multilevel_progressive_widening=False)
    node = engine.root
    node.visits = 25
    node.expanded_patterns.add("PAIR")

    assert engine._pattern_limit(node) == len(engine.knowledge.active_pattern_ids)
    assert engine._block_limit(node) == engine._child_limit(node)


def test_transition_and_leaf_shaping_can_be_disabled() -> None:
    engine = _engine(
        transition_reward_shaping=False,
        leaf_value_shaping=False,
    )
    assert engine._transition_reward(_add_edge()) == 0.0
    node = GameNode(SymmetryState(0b11, "filler", 0), rank=2)
    assert engine._leaf_value(node) == 0.0


def test_filler_and_corrector_transition_shaping_are_independent() -> None:
    filler_off = _engine(filler_transition_reward_shaping=False)
    assert filler_off._transition_reward(_add_edge()) == 0.0

    corrector_off = _engine(corrector_transition_reward_shaping=False)
    remove_edge = _add_edge()
    remove_edge.action = MacroAction(
        kind="remove",
        pattern_id="PAIR",
        orbit_indices=(0,),
        block_word=0b11,
        effective_word=0b11,
        source="atlas",
        structure="C2",
        block_size=2,
        arity=1,
    )
    remove_edge.rank_before = 25
    remove_edge.rank_after = 23
    assert corrector_off._transition_reward(remove_edge) == 0.0


def test_corrector_novelty_credit_can_be_disabled() -> None:
    engine = _engine(corrector_novelty_credit=False)
    edge = _add_edge()
    edge.action = MacroAction(
        kind="remove",
        pattern_id="PAIR",
        orbit_indices=(0,),
        block_word=0b11,
        effective_word=0b11,
        source="atlas",
        structure="C2",
        block_size=2,
        arity=1,
    )
    node = GameNode(SymmetryState((1 << 25) - 1, "corrector", 0), rank=25)
    step = PathStep(
        node=node,
        edge=SimpleNamespace(action=edge.action),
        reward=0.0,
        corrector_origin_utility=-1.5,
    )

    engine._assign_corrector_novelty([step], 2)

    assert step.corrector_novelty_credit == 0.0
    assert not engine.novelty_credit_events


def test_ablation_entry_builds_lean_single_factor_run(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr(ablation_search, "run_main", fake_main)
    ablation_search.main([
        "--variant",
        "no_transition_reward",
        "--iterations",
        "2",
        "--output",
        "test.json",
    ])

    arguments = captured["arguments"]
    assert "--retain-pattern-embeddings" in arguments
    assert "--embedding-relation-policy" in arguments
    assert "--learn-relation-values" in arguments
    assert "--share-embedding-family-values" not in arguments
    assert "--adaptive-filler-reheat" not in arguments
    assert "--no-transition-reward-shaping" in arguments


def test_epoch_ablation_zeroes_both_child_budgets(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return {}

    monkeypatch.setattr(ablation_search, "run_main", fake_main)
    ablation_search.main([
        "--variant",
        "no_epoch_widening",
        "--iterations",
        "2",
        "--output",
        "test.json",
    ])

    arguments = captured["arguments"]
    for option in (
        "--discovery-epoch-child-reserve",
        "--discovery-epoch-child-recycle",
    ):
        assert arguments[arguments.index(option) + 1] == "0"
