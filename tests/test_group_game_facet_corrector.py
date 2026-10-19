from types import SimpleNamespace
from math import comb

import numpy as np
import pytest
from scipy.optimize import linprog

from baseline.bell322 import generate_bell322_points
from baseline.geometry import affine_rank, validate_facet_support
from mcts.group_game.facet_corrector import FacetCorrectorBank, RepairPlan
from mcts.group_game.model import GameNode, SymmetryState, MacroAction, block_word
from baseline.orbit_blocks import build_state_group_permutations
from mcts.group_game.search import PathStep
from mcts.subgroup_patterns import move_support_word
from tests.test_group_game_search import _engine


def test_online_facet_subgroups_and_external_completions():
    points = generate_bell322_points(dtype=float)
    centered = points - points.mean(axis=0)
    lp = linprog(-np.random.default_rng(11).normal(size=26), A_ub=centered,
                 b_ub=np.ones(64), bounds=[(None, None)] * 26, method="highs")
    assert lp.success
    indices = np.flatnonzero(np.abs(centered @ lp.x - 1) < 1e-7)
    validation = validate_facet_support(points, indices)
    assert validation.valid
    word = block_word(tuple(map(int, indices)))
    bank = FacetCorrectorBank(points, seed=123, max_attempts=16)
    bank.observe(word, validation.normal, validation.offset)
    anchor = bank.anchors[word]
    for generators in anchor.generators.values():
        assert all(move_support_word(word, g) == word for g in generators)
    for _ in range(16):
        bank.grow(word)
    assert anchor.plans
    patterns = {p.pattern_id: p for p in anchor.patterns}
    for plan in anchor.plans:
        assert plan.retained_word & ~word == 0
        assert 17 <= affine_rank(points[bank.indices(plan.retained_word)]) <= 24
        for block in patterns[plan.pattern_id].blocks:
            mask = block_word(block)
            assert plan.retained_word & mask in (0, mask)
        for exit_word in plan.exit_words:
            assert exit_word & ~word
            assert validate_facet_support(points, bank.indices(plan.retained_word | exit_word)).valid
    attempts = anchor.attempts
    assert bank.grow(word) is None
    assert anchor.attempts == attempts
    bank.observe(word, validation.normal, validation.offset)
    assert len(bank.anchors) == 1
    permutation = next(g for g in build_state_group_permutations() if move_support_word(word, g) != word)
    moved = move_support_word(word, permutation)
    moved_validation = validate_facet_support(points, bank.indices(moved))
    bank.observe(moved, moved_validation.normal, moved_validation.offset)
    checks_before = bank.stats["exit_geometry_checks"]
    borrowed = bank._borrow(bank.anchors[moved])
    assert borrowed is not None
    assert bank.stats["exit_geometry_checks"] == checks_before
    assert len(bank.shared.pools) == 1
    for g in bank.anchors[moved].generators[borrowed.pattern_id]:
        assert move_support_word(moved, g) == moved
        assert move_support_word(borrowed.retained_word, g) == borrowed.retained_word
    for addition in borrowed.exit_words:
        assert addition & ~moved
        assert validate_facet_support(points, bank.indices(borrowed.retained_word | addition)).valid
    addition = borrowed.exit_words[0]
    bank.shared.mark_used(moved, borrowed.retained_word, addition)
    assert bank.shared.uses(moved, borrowed.retained_word, addition) == 1
    canonical, retained_key, added_key = bank.shared.exit_key(moved, borrowed.retained_word, addition)
    assert (retained_key, added_key) not in bank.shared.untried_exits[canonical]
    before = set(bank.shared.untried_exits[canonical])
    proposal = bank.next_frontier_exit(moved)
    assert proposal is not None
    assert bank.shared.untried_exits[canonical] == before
    proposed_plan, proposed_exit = proposal
    assert proposed_exit in bank.exits[(moved, proposed_plan.retained_word, proposed_plan.pattern_id)]
    assert bank.shared.uses(moved, proposed_plan.retained_word, proposed_exit) == 0
    assert validate_facet_support(points, bank.indices(proposed_plan.retained_word | proposed_exit)).valid

    systematic = FacetCorrectorBank(points, seed=123, max_attempts=0,
                                    ridge_max_candidates=comb(len(indices), 25), ridge_batch_size=1)
    systematic.observe(word, validation.normal, validation.offset)
    systematic.observe(moved, moved_validation.normal, moved_validation.offset)
    assert len(systematic.ridge_enumerators) == 1
    assert systematic.has_frontier(word)
    assert systematic.has_frontier(moved)
    enumerator = next(iter(systematic.ridge_enumerators.values()))
    for source in (word, moved):
        before = enumerator.examined
        plan = systematic.grow(source)
        assert enumerator.examined == before + 1
        assert systematic.anchors[source].attempts == 0
        if plan is not None:
            assert affine_rank(points[systematic.indices(plan.retained_word)]) == 24
            for addition in plan.exit_words:
                assert validate_facet_support(points, systematic.indices(plan.retained_word | addition)).valid


def test_repair_source_participates_in_transposition_key():
    assert SymmetryState(7, repair_source_word=15).key != SymmetryState(7, repair_source_word=31).key


def test_filler_cannot_close_repair_inside_source_facet():
    engine = _engine()
    source = (1 << 30) - 1
    node = GameNode(SymmetryState((1 << 24) - 1, repair_source_word=source), 24)
    actions = engine.candidate_actions_for_pattern(node, "BFS322-C1-00")
    inside = next(a for a in actions if a.effective_word == 1 << 24)
    outside = next(a for a in actions if a.effective_word == 1 << 31)
    assert engine._materialize_action(node, inside) is None
    state, rank = engine._materialize_action(node, outside)
    assert rank == 25
    assert state.repair_source_word == 0


def test_delayed_credit_is_normalized_and_can_reach_earlier_corrections():
    engine = _engine()
    node = engine.root
    action = SimpleNamespace(source="facet_repartition")
    first = PathStep(node, SimpleNamespace(action=action), 0.0, corrector_origin_utility=0.2)
    filler = PathStep(node, SimpleNamespace(action=action), 0.0)
    last = PathStep(node, SimpleNamespace(action=action), 0.0, corrector_origin_utility=0.2)
    engine._assign_corrector_novelty([first, filler, last], 2)
    assert 0 < first.corrector_novelty_credit < last.corrector_novelty_credit
    assert first.corrector_novelty_credit + last.corrector_novelty_credit == pytest.approx(10)
    assert filler.corrector_novelty_credit == 0
    assert engine.novelty_credit_events[0]["total_credit"] == pytest.approx(10)


def test_known_valid_reward_stays_above_invalid_after_many_repeats(monkeypatch):
    engine = _engine()
    monkeypatch.setattr("mcts.group_game.search.canonical_support_word", lambda word: word)
    node = GameNode(SymmetryState((1 << 25) - 1, "corrector"), 25)
    engine.discovered_classes.add(2)
    engine.class_terminal_counts[2] = 10**12
    engine.terminal_support_counts[node.state.support_word] = 10**12
    known = engine._observe_terminal(node, path=[])
    assert known.reward >= -0.4 - 1e-9
    invalid_label = SimpleNamespace(label="invalid:boundary", is_exact=False,
                                    validation=SimpleNamespace(valid=False))
    monkeypatch.setattr(engine.scorer, "terminal_label", lambda key: invalid_label)
    invalid = engine._observe_terminal(node, path=[])
    assert known.reward > invalid.reward
    assert invalid.reward == engine.config.invalid_terminal_reward


def test_corrector_extension_requires_unrepeated_valid_frontier(monkeypatch):
    engine = _engine(max_corrections=1, extended_max_corrections=3)
    node = GameNode(SymmetryState(31, "corrector", 1), 25, facet_word=63)
    engine.facet_bank.anchors[63] = SimpleNamespace()
    engine.facet_bank.shared.coordinates[63] = (7, (), ())
    monkeypatch.setattr(engine.facet_bank, "has_frontier", lambda word: True)
    engine.iteration_facet_counts[7] = 1
    assert not engine._correction_limit_reached(node)
    engine.iteration_facet_counts[7] = 2
    assert engine._correction_limit_reached(node)
    engine.iteration_facet_counts[7] = 1
    monkeypatch.setattr(engine.facet_bank, "has_frontier", lambda word: False)
    assert engine._correction_limit_reached(node)
    node.state = SymmetryState(31, "corrector", 3)
    assert engine._correction_limit_reached(node)
    stop = engine._add_stop_edge(node)
    node.edges["tempting"] = SimpleNamespace(action=SimpleNamespace(kind="rewrite"), mean_value=100)
    assert engine._select_edge(node) is stop


def test_open_context_can_use_extended_correction_budget(monkeypatch):
    engine = _engine(
        max_corrections=1,
        extended_max_corrections=3,
        context_frontier_stagnation_patience=0,
    )
    node = GameNode(SymmetryState(31, "corrector", 1), 25, facet_word=63)
    engine.facet_bank.anchors[63] = SimpleNamespace()
    engine.facet_bank.shared.coordinates[63] = (7, (), ())
    engine.iteration_facet_counts[7] = 3
    monkeypatch.setattr(engine, "_has_open_context_exit", lambda _node: True)
    monkeypatch.setattr(engine.facet_bank, "has_frontier", lambda _word: False)
    assert not engine._correction_limit_reached(node)
    node.state = SymmetryState(31, "corrector", 3)
    assert engine._correction_limit_reached(node)


def test_terminal_observation_preserves_one_witness_per_correction_depth(monkeypatch):
    engine = _engine()
    monkeypatch.setattr("mcts.group_game.search.canonical_support_word", lambda word: word)
    support = (1 << 25) - 1
    add = MacroAction("add", "PAIR", (), support, support, "atlas", "C2", 25, 1)
    rewrite = MacroAction(
        "rewrite", "PAIR", (), support, 1, "atlas", "C2", 1, 1,
        remove_word=1,
    )
    short = [PathStep(engine.root, SimpleNamespace(action=add), 0.0)]
    deep = short + [PathStep(engine.root, SimpleNamespace(action=rewrite), 0.0)]
    shallow_node = GameNode(SymmetryState(support, "corrector", 0, "PAIR"), 25,
                            facet_word=support)
    deep_node = GameNode(SymmetryState(support, "corrector", 2, "PAIR"), 25,
                         facet_word=support)

    engine._observe_terminal(shallow_node, path=short)
    engine._observe_terminal(deep_node, path=deep)

    assert engine.facet_witnesses[support] == (add,)
    assert engine.facet_depth_witnesses[(support, 0)] == (add,)
    assert engine.facet_depth_witnesses[(support, 2)] == (add, rewrite)


def test_replay_is_fair_and_keeps_one_persistent_expansion(monkeypatch):
    engine = _engine(iterations=12, facet_replay_interval=2)
    monkeypatch.setattr("mcts.group_game.search.canonical_support_word", lambda word: word)
    monkeypatch.setattr(engine.facet_bank, "has_frontier", lambda word: True)
    node = engine.root
    path = []
    for _ in range(4):
        action = max(engine.candidate_actions_for_pattern(node, "QUAD"), key=lambda a: a.block_size)
        path.append(action)
        state, rank = engine._materialize_action(node, action)
        node = GameNode(state, rank)
    for word in (1, 2):
        engine.facet_witnesses[word] = tuple(path)
        engine.facet_witness_sources[word] = 123
    for iteration in range(1, 13):
        before = engine.tree_expansions
        engine.run_iteration(iteration)
        assert engine.tree_expansions - before <= 1
    assert engine.replay_stats["persistent_witness_edges"] > 0
    assert engine.facet_replay_counts[1] == engine.facet_replay_counts[2] == 3
    assert engine.root.visits == 12


def test_systematic_batch_continues_after_rejected_candidates(monkeypatch):
    bank = FacetCorrectorBank(np.zeros((64, 26)), seed=1, ridge_batch_size=3)
    identity = tuple(range(64))
    bank.shared.coordinates[1] = (1, identity, identity)
    enumerator = SimpleNamespace(examined=0, finished=False)
    def advance(limit):
        enumerator.examined += 1
        return enumerator.examined
    enumerator.advance = advance
    bank.ridge_enumerators[1] = enumerator
    marker = object()
    monkeypatch.setattr(bank, "_check_systematic_candidate", lambda a, w, g: marker if w == 3 else None)
    assert bank._grow_systematic(SimpleNamespace(word=1)) is marker
    assert enumerator.examined == bank.stats["systematic_candidates"] == 3
    monkeypatch.setattr(bank, "_check_systematic_candidate", lambda *args: None)
    assert bank._grow_systematic(SimpleNamespace(word=1)) is None
    assert enumerator.examined == 6


@pytest.mark.parametrize("fair_frontier", [False, True])
def test_frontier_witness_planning_does_not_discover_or_consume_exit(monkeypatch, fair_frontier):
    engine = _engine(iterations=4, facet_replay_interval=1, frontier_replay_stride=1,
                     fair_frontier=fair_frontier)
    monkeypatch.setattr("mcts.group_game.search.canonical_support_word", lambda w: w)
    source, selected, retained, outside = (1 << 26) - 1, (1 << 25) - 1, (1 << 24) - 1, 1 << 30
    pattern = engine.knowledge.patterns["BFS322-C1-00"]
    action = MacroAction(kind="add", pattern_id=pattern.pattern_id, orbit_indices=(),
                         block_word=selected, effective_word=selected, source="atlas",
                         structure=pattern.structure, block_size=25, arity=25)
    plan = RepairPlan(source, retained, pattern.pattern_id, (outside,))
    engine.facet_witnesses[source] = (action,)
    engine.facet_witness_sources[source] = source
    engine.facet_bank.anchors[source] = SimpleNamespace(word=source, patterns=(pattern,), plans=[plan])
    engine.facet_bank.exits[(source, retained, pattern.pattern_id)] = (outside,)
    identity = tuple(range(64))
    engine.facet_bank.shared.coordinates[source] = (source, identity, identity)
    engine.facet_bank.shared.untried_exits[source] = {(retained, outside)}
    monkeypatch.setattr(engine.facet_bank, "has_frontier", lambda w: True)
    monkeypatch.setattr(engine.facet_bank, "has_delivery_frontier", lambda w: True)
    monkeypatch.setattr(engine.facet_bank, "frontier_lanes", lambda w: ("delivery",))
    monkeypatch.setattr(engine.facet_bank, "next_frontier_exit", lambda w, **kwargs: (plan, outside))
    replay = engine._choose_replay(1)
    assert [a.kind for a in replay] == ["add", "rewrite", "add"]
    assert not engine.class_terminal_counts
    assert not engine.facet_bank.shared.exit_uses
    assert engine.tree_expansions == 0
    for i in range(1, 5):
        before = engine.tree_expansions
        engine.run_iteration(i)
        assert engine.tree_expansions - before <= 1
    assert engine.replay_stats["frontier_executed_rewrite"] > 0
    assert engine.replay_stats["frontier_executed_add"] > 0
    assert (retained, outside) not in engine.facet_bank.shared.untried_exits[source]


def test_context_frontier_replays_globally_used_exit_from_new_depth(monkeypatch):
    engine = _engine(
        context_frontier_grow_stride=0,
        context_frontier_stagnation_patience=0,
        context_frontier_quota_stride=1,
        context_geometry_growth=False,
    )
    source = (1 << 26) - 1
    selected = (1 << 25) - 1
    retained = (1 << 24) - 1
    outside = 1 << 30
    target = source | outside
    pattern = engine.knowledge.patterns["BFS322-C1-00"]
    action = MacroAction(
        kind="add", pattern_id=pattern.pattern_id, orbit_indices=(),
        block_word=selected, effective_word=selected, source="atlas",
        structure=pattern.structure, block_size=25, arity=25,
    )
    plan = RepairPlan(source, retained, pattern.pattern_id, (outside,))
    engine.facet_depth_witnesses[(source, 0)] = (action,)
    engine.facet_depth_witness_sources[(source, 0)] = source
    engine.facet_bank.anchors[source] = SimpleNamespace(
        word=source, patterns=(pattern,), plans=[plan],
    )
    engine.facet_bank.exits[(source, retained, pattern.pattern_id)] = (outside,)
    identity = tuple(range(64))
    engine.facet_bank.shared.coordinates[source] = (source, identity, identity)
    graph_key = source, retained, outside
    engine.facet_bank.shared.endpoints.register(graph_key, target)
    engine.facet_bank.shared.endpoints.mark_used(graph_key)
    monkeypatch.setattr(engine.facet_bank, "has_frontier", lambda _word: False)
    engine.replay_stats["context_frontier_lane_requests"] = 3

    replay = engine._choose_replay(8)

    assert [item.kind for item in replay] == ["add", "rewrite", "add"]
    assert engine.frontier_replay_start == 1
    assert engine.replay_stats["context_frontier_planned_pairs"] == 1
    assert engine.replay_stats["context_frontier_bridge_planned_pairs"] == 1
    assert not engine.context_endpoint_executions
    assert engine.facet_bank.shared.endpoints.executions[(source, target)] == 1


def test_frontier_extension_respects_dynamic_correction_stop(monkeypatch):
    engine = _engine()
    engine.frontier_replay_start = 3
    engine.frontier_replay_keys = {(1, "action")}
    node = GameNode(SymmetryState(31, "corrector"), 25)
    monkeypatch.setattr(engine, "_correction_limit_reached", lambda n: True)
    assert not engine._frontier_replay_blocked(node, 2)
    assert not engine._frontier_replay_blocked(node, 4)
    assert engine._frontier_replay_blocked(node, 3)
    assert engine.frontier_replay_start is None
    assert not engine.frontier_replay_keys
