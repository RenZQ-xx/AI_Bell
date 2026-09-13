from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import mcts.group_game.search as search_module
from mcts.group_game.knowledge import GroupKnowledgeBase, PatternSpec
from mcts.group_game.model import EdgeStats, GameNode, MacroAction, SymmetryState
from mcts.group_game.search import (
    GeometryGrowthEstimate,
    GroupGameConfig,
    PathStep,
    SingleTreeGroupGame,
    TerminalOutcome,
)
from mcts.subgroup_patterns import SubgroupPattern, SupportPatternAnalysis


def _partition(width: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(start, min(64, start + width)))
        for start in range(0, 64, width)
    )


def _pattern(pattern_id: str, level: int, structure: str, width: int) -> SubgroupPattern:
    return SubgroupPattern(
        pattern_id=pattern_id,
        level=level,
        structure=structure,
        order=max(1, width),
        generators=(),
        generator_labels=(),
        blocks=_partition(width),
        parent_root_ids=() if level <= 1 else ("PAIR",),
    )


class FakeAtlas:
    source_path = Path("fake_atlas.json")
    identity = _pattern("BFS322-C1-00", 0, "C1", 1)
    roots = (
        _pattern("PAIR", 1, "C2", 2),
        _pattern("QUAD", 1, "C4", 4),
    )
    level2 = (_pattern("LEVEL2", 2, "C2xC2", 4),)

    def children(self, root_id: str):
        if root_id == "PAIR":
            return ((self.level2[0], SimpleNamespace(edge_id="edge")),)
        return ()

    def analyze_support(self, support: int, **_kwargs) -> SupportPatternAnalysis:
        return SupportPatternAnalysis(
            support_word=int(support),
            support_size=int(support).bit_count(),
            stabilizer_order=2,
            stabilizer_blocks=_partition(2),
            stabilizer_generators=(),
            root_pattern_ids=("PAIR",),
            root_subgroup_multiplicities=(("PAIR", 1),),
            level2_pattern_ids=("LEVEL2",),
        )


class FakeScorer:
    def __init__(self) -> None:
        self.points = np.zeros((64, 1), dtype=float)
        self.config = SimpleNamespace(support_tol=1e-6)

    @staticmethod
    def affine_rank(key: tuple[int, ...]) -> int:
        return min(25, sum(int(value) for value in key))

    @staticmethod
    def terminal_label(_key: tuple[int, ...]):
        validation = SimpleNamespace(valid=True, normal=None, offset=None)
        return SimpleNamespace(
            label="exact:class2",
            is_exact=True,
            validation=validation,
        )


def _config(**overrides) -> GroupGameConfig:
    values = {
        "iterations": 2,
        "known_classes": (1,),
        "target_classes": (1, 2),
        "max_depth": 40,
        "max_corrections": 1,
        "min_corrector_rank": 10,
        "include_level2_analysis": False,
        "growth_children_per_root": 1,
    }
    values.update(overrides)
    return GroupGameConfig(**values)


def _engine(**overrides) -> SingleTreeGroupGame:
    return SingleTreeGroupGame(
        _config(**overrides),
        atlas=FakeAtlas(),
        scorer=FakeScorer(),
    )


def test_dynamic_filler_exposes_multiple_pattern_block_forms() -> None:
    engine = _engine()
    pair_actions = engine.candidate_actions_for_pattern(engine.root, "PAIR")
    quad_actions = engine.candidate_actions_for_pattern(engine.root, "QUAD")

    assert {action.block_size for action in pair_actions} >= {2, 4}
    assert {action.block_size for action in quad_actions} >= {4, 8}
    assert all(action.kind == "add" for action in pair_actions + quad_actions)


def test_generation_caches_preserve_candidates_and_within_pattern_order() -> None:
    engine = _engine()
    first = engine.candidate_actions_for_pattern(engine.root, "PAIR")
    second = engine.candidate_actions_for_pattern(engine.root, "PAIR")
    assert first is not second
    assert [action.action_id for action in first] == [action.action_id for action in second]
    assert engine.generation_cache_stats["candidate_misses"] == 1
    assert engine.generation_cache_stats["candidate_hits"] == 1

    coherence = engine._coherence(engine.root.state.support_word, "PAIR")
    assert engine._coherence(engine.root.state.support_word, "PAIR") == coherence
    assert engine.generation_cache_stats["coherence_hits"] >= 1
    full = sorted(first, key=lambda action: (
        engine._approximate_action_score(engine.root, action), -action.block_size, action.action_id),
        reverse=True)
    shape = sorted(first, key=lambda action: (
        engine._action_shape_score(engine.root, action), -action.block_size, action.action_id),
        reverse=True)
    assert [action.action_id for action in full] == [action.action_id for action in shape]
    assert "action_id" in first[0].__dict__


def test_lazy_lane_candidates_equal_filtering_the_full_pool() -> None:
    engine = _engine()
    cases = [(engine.root, "PAIR", lane) for lane in ("fine", "single", "union", "broad")]
    corrector = GameNode(
        SymmetryState((1 << 25) - 1, "corrector", 1, "PAIR"),
        25,
    )
    cases.extend((corrector, "PAIR", lane)
                 for lane in ("remove_small", "remove_broad", "rewrite"))
    for node, pattern_id, lane in cases:
        full = engine.candidate_actions_for_pattern(node, pattern_id)
        expected = [action for action in full if engine._matches_action_lane(action, lane)]
        lazy = engine.candidate_actions_for_pattern(
            node, pattern_id, lane=lane, fallback=False)
        assert [action.action_id for action in lazy] == [action.action_id for action in expected]

    assert not engine.candidate_actions_for_pattern(
        engine.root, "QUAD", lane="fine", fallback=False)
    assert engine.candidate_actions_for_pattern(
        engine.root, "QUAD", lane="fine", fallback=True)
    assert engine.generation_cache_stats["lane_fallbacks"] == 1


def test_widening_falls_back_after_lane_actions_are_excluded(monkeypatch) -> None:
    engine = _engine()
    node = engine.root
    node.expanded_patterns.add("PAIR")
    lane_action = MacroAction(
        "add", "PAIR", (0,), 1, 1, "atlas", "C2", 1, 1,
    )
    fallback_action = MacroAction(
        "add", "PAIR", (0, 1, 2), 0b111111, 0b111111,
        "atlas", "C2", 6, 3,
    )
    node.rejected_action_ids.add(lane_action.action_id)
    calls = []

    def candidates(_node, _pattern_id, *, lane=None, fallback=True):
        calls.append((lane, fallback))
        return [lane_action] if lane == "fine" else [lane_action, fallback_action]

    monkeypatch.setattr(engine, "_choose_new_pattern", lambda _node: None)
    monkeypatch.setattr(engine, "_next_action_lane", lambda _node: "fine")
    monkeypatch.setattr(engine, "candidate_actions_for_pattern", candidates)

    edge = engine._widen_once(node)
    assert edge is not None
    assert edge.action.action_id == fallback_action.action_id
    assert calls == [("fine", False), (None, True)]
    assert "PAIR" not in node.exhausted_patterns


def test_saturated_facet_source_requires_executed_geometry_without_pending_exit() -> None:
    engine = _engine()
    raw, canonical = 100, 1
    engine.facet_bank.shared.coordinates[raw] = (canonical, (), ())
    graph = engine.facet_bank.shared.endpoints
    assert not engine._facet_source_saturated(raw)
    graph.register((canonical, 10, 20), 2)
    assert not engine._facet_source_saturated(raw)
    graph.mark_used((canonical, 10, 20))
    assert engine._facet_source_saturated(raw)
    graph.register((canonical, 11, 21), 3)
    assert not engine._facet_source_saturated(raw)


def test_saturation_gate_skips_repair_before_materialization(monkeypatch) -> None:
    engine = _engine(facet_repair_probability=1.0,
                     saturated_facet_repair_probability=0.0,
                     endpoint_saturation_gate=True)
    node = GameNode(SymmetryState((1 << 25) - 1, "corrector", 0, "PAIR"), 25,
                    facet_word=100)
    engine.facet_bank.anchors[100] = SimpleNamespace()
    action = MacroAction("rewrite", "PAIR", (), 1, 1, "facet_repartition", "C2", 1, 1)
    monkeypatch.setattr(engine, "_facet_actions", lambda _node: [action])
    monkeypatch.setattr(engine, "_facet_action_context_uses", lambda _node, _action: 1)
    monkeypatch.setattr(engine, "_ephemeral_edge",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    assert engine._facet_edge(node, rng=engine.rng, persistent=False) is None
    assert engine.replay_stats["saturated_facet_repair_offers"] == 1
    assert engine.replay_stats["saturated_facet_repair_skips"] == 1


def test_new_context_keeps_globally_saturated_bridge_open(monkeypatch) -> None:
    engine = _engine(facet_repair_probability=1.0,
                     saturated_facet_repair_probability=0.0,
                     endpoint_saturation_gate=True)
    node = GameNode(SymmetryState((1 << 25) - 1, "corrector", 4, "PAIR"), 25,
                    facet_word=100)
    engine.facet_bank.anchors[100] = SimpleNamespace()
    action = MacroAction("rewrite", "PAIR", (), 1, 1, "facet_repartition", "C2", 1, 1)
    result = (object(), SymmetryState(1, "filler", 5, "PAIR", 100), 1)
    monkeypatch.setattr(engine, "_facet_actions", lambda _node: [action])
    monkeypatch.setattr(engine, "_facet_action_context_uses", lambda _node, _action: 0)
    monkeypatch.setattr(engine, "_facet_source_saturated", lambda _word: True)
    monkeypatch.setattr(engine, "_ephemeral_edge", lambda *_args, **_kwargs: result)
    assert engine._facet_edge(node, rng=engine.rng, persistent=False) is result
    assert engine.replay_stats["global_saturated_context_open_offers"] == 1
    assert engine.replay_stats["context_open_facet_repair_offers"] == 1
    assert not engine.replay_stats["saturated_facet_repair_skips"]


def test_endpoint_reuse_is_conditioned_on_pattern_and_correction_depth() -> None:
    engine = _engine()
    raw, canonical, retained, outside, target = 100, 1, 10, 20, 2
    identity = tuple(range(64))
    engine.facet_bank.shared.coordinates[raw] = (canonical, identity, identity)
    engine.facet_bank.shared.endpoints.register((canonical, retained, outside), target)
    engine.facet_bank.exits[(raw, retained, "PAIR")] = (outside,)
    completion_node = GameNode(
        SymmetryState(retained, "filler", 3, "PAIR", raw),
        20,
    )
    completion = MacroAction(
        "add", "PAIR", (), outside, outside, "facet_external_exit", "C2", 1, 1,
        source_facet_word=raw,
    )
    novelty = engine._record_endpoint_context(completion_node, completion)
    key = canonical, target, "PAIR", 3
    assert novelty == 1.0
    assert engine.context_endpoint_executions[key] == 1
    assert engine._external_action_context_uses(completion_node, completion) == 1
    engine.facet_bank.shared.endpoints.mark_used((canonical, retained, outside))
    assert engine._record_endpoint_context(completion_node, completion) == 0.0

    repair_node = GameNode(
        SymmetryState(retained | 1, "corrector", 2, "PAIR"),
        25,
        facet_word=raw,
    )
    repair = MacroAction(
        "rewrite", "PAIR", (), retained, 1, "facet_repartition", "C2", 1, 1,
        remove_word=1,
        source_facet_word=raw,
    )
    assert engine._facet_action_context_uses(repair_node, repair) == 2
    deeper = GameNode(
        SymmetryState(retained | 1, "corrector", 3, "PAIR"),
        25,
        facet_word=raw,
    )
    assert engine._facet_action_context_uses(deeper, repair) == 0


def test_global_repair_lane_can_throttle_a_context_open_saturated_source(monkeypatch) -> None:
    engine = _engine(
        facet_repair_probability=1.0,
        saturated_facet_repair_probability=0.0,
        endpoint_saturation_gate=True,
    )
    engine.current_repair_policy = "global"
    node = GameNode(SymmetryState((1 << 25) - 1, "corrector", 4, "PAIR"), 25,
                    facet_word=100)
    engine.facet_bank.anchors[100] = SimpleNamespace()
    action = MacroAction("rewrite", "PAIR", (), 1, 1, "facet_repartition", "C2", 1, 1)
    monkeypatch.setattr(engine, "_facet_actions", lambda _node: [action])
    monkeypatch.setattr(engine, "_facet_action_context_uses", lambda _node, _action: 0)
    monkeypatch.setattr(engine, "_facet_source_saturated", lambda _word: True)
    monkeypatch.setattr(engine, "_ephemeral_edge",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    assert engine._facet_edge(node, rng=engine.rng, persistent=False) is None
    assert engine.replay_stats["global_policy_saturated_offers"] == 1
    assert engine.replay_stats["global_policy_saturated_skips"] == 1


def test_context_geometry_growth_races_on_online_endpoint_yield(monkeypatch) -> None:
    engine = _engine(context_frontier_stagnation_patience=0)
    raw, canonical, retained, outside, target = 100, 1, 10, 20, 2
    identity = tuple(range(64))
    engine.facet_bank.shared.coordinates[raw] = (canonical, identity, identity)
    engine.facet_bank.shared.pools[canonical] = []
    engine.facet_bank.anchors[raw] = SimpleNamespace(
        requests=0,
        attempts=0,
        plans=[],
    )
    monkeypatch.setattr(
        engine,
        "_depth_witness_entries",
        lambda: [((canonical, 0), (), raw)],
    )
    monkeypatch.setattr(engine.facet_bank, "has_growth_frontier", lambda _word: True)

    def grow(_word):
        engine.facet_bank.shared.endpoints.register(
            (canonical, retained, outside),
            target,
        )

    monkeypatch.setattr(engine.facet_bank, "grow", grow)

    assert engine._run_context_geometry_growth() == canonical
    assert engine.geometry_growth_stats["calls"] == 1
    assert engine.geometry_growth_stats["new_targets"] == 1
    assert engine.geometry_growth_stats["new_pairs"] == 1
    assert not engine.class_terminal_counts


def test_growth_score_does_not_promote_cheap_empty_work() -> None:
    empty = GeometryGrowthEstimate()
    for _ in range(20):
        empty.observe(seconds=0.00001, targets=0, pairs=0, plans=0)
    productive = GeometryGrowthEstimate()
    productive.observe(seconds=0.1, targets=1, pairs=1, plans=1)

    assert productive.score(1) > empty.score(20)
    assert empty.consecutive_empty == 20


def test_growth_plans_without_endpoints_count_as_empty_work() -> None:
    estimate = GeometryGrowthEstimate()
    estimate.observe(seconds=0.01, targets=0, pairs=0, plans=1)

    assert estimate.consecutive_empty == 1
    assert estimate.score(1) == GeometryGrowthEstimate(
        observations=1,
        consecutive_empty=1,
    ).score(1)


def test_geometry_growth_hard_limits_productive_source_service_lead(
    monkeypatch,
) -> None:
    engine = _engine(
        context_frontier_stagnation_patience=0,
        geometry_growth_warmup=1,
        geometry_growth_probe_stride=100,
        geometry_growth_max_service_lead=2,
        geometry_growth_target_lead_bonus=0,
    )
    identity = tuple(range(64))
    for raw, canonical in ((100, 1), (200, 2)):
        engine.facet_bank.shared.coordinates[raw] = (
            canonical,
            identity,
            identity,
        )
        engine.facet_bank.shared.pools[canonical] = []
        engine.facet_bank.anchors[raw] = SimpleNamespace(
            requests=0,
            attempts=0,
            plans=[],
        )
    monkeypatch.setattr(
        engine,
        "_depth_witness_entries",
        lambda: [((1, 0), (), 100), ((2, 0), (), 200)],
    )
    monkeypatch.setattr(
        engine.facet_bank,
        "has_growth_frontier",
        lambda _word: True,
    )
    target = 10

    def grow(raw_source):
        nonlocal target
        if raw_source != 100:
            return
        target += 1
        engine.facet_bank.shared.endpoints.register(
            (1, target << 1, target << 2),
            target,
        )

    monkeypatch.setattr(engine.facet_bank, "grow", grow)

    for _ in range(20):
        engine._run_context_geometry_growth()

    services = engine.geometry_growth_services
    assert services[1] - services[2] <= 2
    assert engine.geometry_growth_stats["service_lead_limited_choices"] > 0


def test_filler_and_corrector_transitions_share_one_tree() -> None:
    engine = _engine()
    support = (1 << 24) - 1
    filler = engine._node(SymmetryState(support, "filler", 0), rank=24)
    add = MacroAction(
        kind="add",
        pattern_id="BFS322-C1-00",
        orbit_indices=(24,),
        block_word=1 << 24,
        effective_word=1 << 24,
        source="atlas",
        structure="C1",
        block_size=1,
        arity=1,
    )
    corrector_state, corrector_rank = engine._materialize_action(filler, add)
    corrector = engine._node(corrector_state, corrector_rank)
    remove = MacroAction(
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
    repaired_state, repaired_rank = engine._materialize_action(corrector, remove)
    repaired = engine._node(repaired_state, repaired_rank)

    assert corrector.state.role == "corrector"
    assert repaired.state.role == "filler"
    assert repaired.state.corrections_used == 1
    assert filler.state.key in engine.nodes
    assert corrector.state.key in engine.nodes
    assert repaired.state.key in engine.nodes


def test_hierarchical_widening_adds_pattern_and_block_diversity() -> None:
    engine = _engine()
    engine.root.visits = 100
    for _ in range(20):
        engine._widen_once(engine.root)

    pattern_ids = {
        edge.action.pattern_id
        for edge in engine.root.edges.values()
        if edge.action.kind == "add"
    }
    block_sizes = {
        edge.action.block_size
        for edge in engine.root.edges.values()
        if edge.action.kind == "add"
    }
    arities = {
        edge.action.arity
        for edge in engine.root.edges.values()
        if edge.action.kind == "add"
    }
    assert len(pattern_ids) >= 2
    assert len(block_sizes) >= 2
    assert arities >= {1, 2}


def test_root_progressive_limit_remains_open_after_discovery_epoch() -> None:
    engine = _engine()
    engine.root.visits = 10_000
    engine.knowledge.begin_discovery_epoch()

    assert engine._child_limit(engine.root) > 48


def test_corrector_can_delete_wide_orbit_unions_and_rewrite() -> None:
    engine = _engine(max_corrections=3, min_corrector_rank=0)
    support = (1 << 32) - 1
    node = GameNode(
        SymmetryState(support, "corrector", 0, "PAIR"),
        rank=25,
    )

    actions = engine.candidate_actions_for_pattern(node, "PAIR")

    removes = [action for action in actions if action.kind == "remove"]
    rewrites = [action for action in actions if action.kind == "rewrite"]
    assert any(action.arity >= 4 and action.block_size >= 8 for action in removes)
    assert rewrites
    assert all(action.remove_word and action.add_word for action in rewrites)


def test_one_simulation_persists_only_one_new_tree_edge(monkeypatch) -> None:
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    engine = _engine(iterations=1, target_classes=(1, 99))

    engine.run_iteration(1)

    assert engine.tree_expansions == 1
    assert len(engine.nodes) == 2
    assert engine.rollout_steps > 1


def test_online_knowledge_learns_productive_pattern() -> None:
    patterns = (
        PatternSpec("A", 1, "C2", _partition(2), "atlas"),
        PatternSpec("B", 1, "C2", _partition(2), "atlas"),
    )
    knowledge = GroupKnowledgeBase(patterns)
    action = MacroAction(
        kind="add",
        pattern_id="A",
        orbit_indices=(0,),
        block_word=0b11,
        effective_word=0b11,
        source="atlas",
        structure="C2",
        block_size=2,
        arity=1,
    )
    for _ in range(4):
        knowledge.update_action(
            "filler",
            action,
            rank_before=20,
            rank_after=22,
            reward=4.0,
            exact_hit=True,
            new_class=True,
        )

    assert knowledge.pattern_score("filler", "A") > knowledge.pattern_score("filler", "B")


def test_terminal_hit_activates_observed_stabilizer_and_decomposition(monkeypatch) -> None:
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    engine = _engine()
    support = (1 << 25) - 1
    node = GameNode(SymmetryState(support, "corrector", 0), rank=25)
    engine.started_at = 1.0
    engine.current_iteration = 7

    outcome = engine._observe_terminal(node, path=[])

    assert outcome.reward > 10.0
    assert outcome.exact_hit is True
    assert outcome.new_class is True
    assert 2 in engine.discovered_classes
    assert "LEVEL2" in engine.knowledge.active_pattern_ids
    assert f"OBS-{support:016x}" in engine.knowledge.active_pattern_ids
    assert engine.knowledge.observed_supports[support].root_pattern_ids == ("PAIR",)


def test_known_terminal_becomes_negative_after_global_discovery(monkeypatch) -> None:
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    engine = _engine()
    support = (1 << 25) - 1
    node = GameNode(SymmetryState(support, "corrector", 0), rank=25)
    engine.started_at = 1.0

    first = engine._observe_terminal(node, path=[])
    second = engine._observe_terminal(node, path=[])

    assert first.new_class is True
    assert first.reward > 10.0
    assert second.new_class is False
    assert second.reward < 0.0


def test_corrector_does_not_receive_credit_for_discovery_before_removal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    engine = _engine(iterations=1, max_corrections=1)

    engine.run()

    corrector_stats = [
        stats
        for (role, _pattern_id), stats in engine.knowledge.pattern_stats.items()
        if role == "corrector"
    ]
    assert corrector_stats
    assert all(stats.new_classes == 0 for stats in corrector_stats)
    assert sum(engine.corrector_outcome_counts.values()) > 0


def test_counterfactual_credit_counts_terminal_outcome_once() -> None:
    engine = _engine()
    action = MacroAction(
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
    edge = SimpleNamespace(action=action)
    step = PathStep(
        node=node,
        edge=edge,
        reward=4.9,
        corrector_origin_label="invalid:boundary",
        corrector_origin_utility=-1.5,
        corrector_transition_cost=-0.1,
    )
    outcome = TerminalOutcome(
        reward=5.0,
        counterfactual_utility=5.0,
        label="exact:class2",
        class_id=2,
        exact_hit=True,
        new_class=True,
        valid=True,
        canonical_support_word=1,
    )

    engine._resolve_corrector_credit(step, outcome=outcome)

    assert step.corrector_credit == 6.4
    engine._assign_corrector_novelty([step], 2)
    assert step.corrector_novelty_credit == 10.0
    assert engine.corrector_diagnostics[0]["transition_cost"] == -0.1


def test_invalid_to_invalid_is_not_a_positive_corrector_improvement() -> None:
    engine = _engine()
    action = MacroAction(
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
        edge=SimpleNamespace(action=action),
        reward=-0.02,
        corrector_origin_label="invalid:boundary",
        corrector_origin_utility=-1.5,
        corrector_origin_support=1,
        corrector_transition_cost=-0.02,
    )
    outcome = TerminalOutcome(
        reward=-1.15,
        counterfactual_utility=-1.5,
        label="invalid:boundary",
        class_id=None,
        exact_hit=False,
        new_class=False,
        valid=False,
        canonical_support_word=2,
    )

    engine._resolve_corrector_credit(step, outcome=outcome)

    assert step.corrector_credit == 0.0
    assert engine.corrector_outcome_counts["neutral"] == 1


def test_small_run_reports_single_tree_and_no_facet_guided_actions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    output = tmp_path / "single_tree.json"
    engine = _engine(iterations=2, max_corrections=0, target_classes=(2,))

    payload = engine.run(output_path=output)

    assert output.exists()
    assert payload["metadata"]["one_mcts_tree"] is True
    assert payload["metadata"]["facet_examples_used_for_action_generation"] is False
    assert payload["tree"]["node_count"] == len(engine.nodes)
    assert payload["summary"]["iterations_completed"] == 1
    assert payload["summary"]["new_class_ids"] == [2]


def test_initial_known_classes_do_not_count_as_observed_or_finish_search(monkeypatch):
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    engine = _engine(iterations=2, max_corrections=0)
    payload = engine.run()
    summary = payload["summary"]
    assert summary["iterations_completed"] == 2
    assert summary["discovered_class_ids"] == [2]
    assert summary["discovered_class_count"] == 1
    assert summary["reward_known_class_ids"] == [1, 2]
    assert summary["missing_target_class_ids"] == [1]
    engine.class_terminal_counts[1] += 1
    assert engine.result(0)["summary"]["discovered_class_ids"] == [1, 2]
    assert engine.result(0)["summary"]["new_class_ids"] == [2]


def test_singleton_start_does_not_imply_known_class_one():
    assert GroupGameConfig().known_classes == ()
    assert search_module.build_parser().parse_args([]).known_classes == []


def test_report_replace_retries_transient_windows_lock(monkeypatch, tmp_path):
    original = Path.replace
    attempts = []
    def replace(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError("reader holds destination")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(search_module.time, "sleep", lambda seconds: None)
    output = tmp_path / "report.json"
    search_module._write_json(output, {"complete": True})
    assert len(attempts) == 3
    assert output.exists()
    assert not output.with_suffix(".json.tmp").exists()


def test_snapshot_failure_does_not_abort_search(monkeypatch, tmp_path):
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: word)
    original = search_module._write_json
    calls = []
    def write(path, payload):
        calls.append(payload["summary"]["iterations_completed"])
        if len(calls) <= 2:
            raise PermissionError("temporary reader lock")
        return original(path, payload)
    monkeypatch.setattr(search_module, "_write_json", write)
    output = tmp_path / "report.json"
    result = _engine(iterations=2, max_corrections=0, snapshot_interval=1).run(output_path=output)
    assert calls == [1, 2, 2]
    assert result["summary"]["iterations_completed"] == 2
    assert output.exists()


def test_cross_rewrite_uses_context_removal_and_neighbor_addition() -> None:
    engine = _engine(min_corrector_rank=0)
    node = GameNode(SymmetryState((1 << 32) - 1, "corrector", 0, "PAIR"), 25)
    actions = engine.candidate_actions_for_pattern(node, "BFS322-C1-00")
    cross = [a for a in actions if a.kind == "rewrite" and a.remove_pattern_id == "PAIR"]
    assert cross
    for action in cross:
        assert action.add_pattern_id == "BFS322-C1-00"
        assert action.remove_word & node.state.support_word == action.remove_word
        assert action.add_word & node.state.support_word == 0
        assert action.remove_word.bit_count() % 2 == 0


def test_fine_rollout_preserves_small_action_path() -> None:
    engine = _engine()
    node = engine.root
    for _ in range(12):
        edge, state, rank = engine._rollout_edge(node, "fine")
        assert edge.action.block_size <= 2
        assert edge.action.arity == 1
        node = GameNode(state, rank)


def test_epoch_recycles_redundant_negative_children_for_unseen_patterns() -> None:
    engine = _engine(max_children_per_node=2, discovery_epoch_child_recycle=1)
    node = engine.root
    node.visits = 100
    node.knowledge_epoch_seen = 0
    node.expanded_patterns = {"PAIR"}
    actions = engine.candidate_actions_for_pattern(node, "PAIR")[:2]
    for action in actions:
        state, rank = engine._materialize_action(node, action)
        edge = EdgeStats(action, state.key, 1.0, 0, rank)
        for _ in range(10):
            edge.update(-2.0, 0)
        node.edges[action.action_id] = edge
        node.child_state_keys.add(state.key)
    node.pattern_action_counts["PAIR"] = 2
    engine.knowledge.begin_discovery_epoch()

    engine._refresh_node_epoch(node)

    assert len(node.edges) == 1
    assert len(node.retired_action_ids) == 1
    assert engine.epoch_recycled_edges == 1
    assert node.pattern_action_counts["PAIR"] == 1
    assert engine._widen_once(node) is not None


def test_corrector_new_class_reward_is_used_once(monkeypatch) -> None:
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    engine = _engine()
    node = GameNode(SymmetryState((1 << 25) - 1, "corrector", 0), 25)
    action = engine.candidate_actions_for_pattern(node, "PAIR")[0]
    credits = []
    for _ in range(2):
        step = PathStep(node, SimpleNamespace(action=action), 0.0,
                        corrector_origin_utility=-1.5)
        outcome = engine._observe_terminal(node, path=[step])
        engine._resolve_corrector_credit(step, outcome=outcome)
        credits.append(step.corrector_credit + step.corrector_novelty_credit)
    assert 11.0 < credits[0] < 12.0
    assert credits[1] < 2.0
    assert engine.corrector_outcome_counts["new_class"] == 1
    assert len(engine.novelty_credit_events) == 1


def _saturated_root(engine: SingleTreeGroupGame) -> list[EdgeStats]:
    node = engine.root
    node.visits = 200
    node.knowledge_epoch_seen = engine.knowledge.epoch
    node.expanded_patterns = {"PAIR"}
    actions = engine.candidate_actions_for_pattern(node, "PAIR")[:4]
    for action in actions:
        state, rank = engine._materialize_action(node, action)
        edge = EdgeStats(action, state.key, 1.0, 0, rank)
        for _ in range(12):
            edge.update(-2.0, engine.knowledge.epoch)
        node.edges[action.action_id] = edge
        node.child_state_keys.add(state.key)
    node.pattern_action_counts["PAIR"] = len(actions)
    return list(node.edges.values())


def test_stagnation_reopens_without_new_epoch_and_protects_discovery_path() -> None:
    engine = _engine(max_children_per_node=4, discovery_epoch_child_recycle=2)
    edges = _saturated_root(engine)
    edges[0].discovery_hits = 1
    engine.current_iteration = 600

    engine._refresh_node_epoch(engine.root)

    assert engine.knowledge.epoch == 0
    assert engine.stagnation_recycled_edges == 2
    assert engine.epoch_recycled_edges == 0
    assert edges[0].action.action_id in engine.root.edges
    assert len(engine.root.expanded_patterns) == 3
    assert engine.recycle_events[0]["reason"] == "stagnation"
    assert engine.recycle_events[0]["fresh_pattern_ids"]
    assert engine._widen_once(engine.root).action.pattern_id != "PAIR"


def test_recycling_waits_for_stagnation_and_local_cooldown() -> None:
    engine = _engine(max_children_per_node=4)
    _saturated_root(engine)
    engine.current_iteration = 499
    engine._refresh_node_epoch(engine.root)
    assert not engine.recycle_events

    engine.current_iteration = 600
    engine.root.last_recycle_visit = 100
    engine._refresh_node_epoch(engine.root)
    assert not engine.recycle_events

    engine.root.visits = 228
    engine._refresh_node_epoch(engine.root)
    assert engine.stagnation_recycled_edges > 0


def test_tree_edges_record_discovery_credit_for_recycling_protection(monkeypatch) -> None:
    monkeypatch.setattr(search_module, "canonical_support_word", lambda word: int(word))
    engine = _engine(iterations=1)
    engine.run()
    assert sum(edge.discovery_hits for edge in engine.root.edges.values()) == 1
