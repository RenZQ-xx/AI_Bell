import numpy as np
import pytest
from scipy.optimize import linprog

from baseline.bell322 import generate_bell322_points
from baseline.geometry import validate_facet_support
from baseline.orbit_blocks import build_state_group_permutations
from mcts.group_game.endpoint_graph import EndpointGraph
from mcts.group_game.facet_corrector import FacetCorrectorBank
from mcts.group_game.model import block_word
from mcts.subgroup_patterns import canonical_support_word, move_support_word


def test_endpoint_aliases_are_consumed_only_by_execution():
    graph = EndpointGraph()
    graph.register((1, 10, 20), 2)
    graph.register((1, 11, 21), 2)
    graph.register((1, 12, 22), 1)
    graph.register((3, 13, 23), 2)
    assert graph.pending[1] == {2}
    assert not graph.executions
    graph.proposals[(1, 2)] += 1
    assert graph.pending[1] == {2}
    graph.mark_used((1, 10, 20))
    assert not graph.pending[1]
    assert graph.pending[3] == {2}
    graph.register((1, 14, 24), 2)
    assert not graph.pending[1]
    graph.mark_used((1, 11, 21))
    assert graph.stats["repeat_pair_executions"] == 1
    assert graph.to_dict()["unique_source_target_pairs"] == 3
    assert graph.to_dict()["distinct_executed_pairs"] == 1


def test_inconsistent_endpoint_is_rejected():
    graph = EndpointGraph()
    graph.register((1, 2, 3), 4)
    with pytest.raises(ValueError, match="inconsistent"):
        graph.register((1, 2, 3), 5)


def test_online_endpoints_match_full_geometry_and_symmetry():
    points = generate_bell322_points(dtype=float)
    centered = points - points.mean(axis=0)
    lp = linprog(-np.random.default_rng(11).normal(size=26), A_ub=centered,
                 b_ub=np.ones(64), bounds=[(None, None)] * 26, method="highs")
    assert lp.success
    indices = np.flatnonzero(abs(centered @ lp.x - 1) < 1e-7)
    validation = validate_facet_support(points, indices)
    source = block_word(tuple(map(int, indices)))
    bank = FacetCorrectorBank(points, seed=123, max_attempts=8, endpoint_dedup=True)
    bank.observe(source, validation.normal, validation.offset)
    for _ in range(8):
        bank.grow(source)
    graph = bank.shared.endpoints
    assert graph.targets
    assert len(graph.targets) > len(graph.pairs)
    assert not graph.executions
    for plan in bank.anchors[source].plans:
        for outside in plan.exit_words:
            check = validate_facet_support(points, bank.indices(plan.retained_word | outside))
            assert check.valid
            tight = block_word(tuple(map(int, np.flatnonzero(abs(points @ check.normal + check.offset) < 1e-6))))
            assert graph.targets[bank.shared.exit_key(source, plan.retained_word, outside)] == canonical_support_word(tight)
    permutation = next(g for g in build_state_group_permutations() if move_support_word(source, g) != source)
    moved = move_support_word(source, permutation)
    check = validate_facet_support(points, bank.indices(moved))
    bank.observe(moved, check.normal, check.offset)
    plan = bank._borrow(bank.anchors[moved])
    assert plan is not None
    key = bank.shared.exit_key(moved, plan.retained_word, plan.exit_words[0])
    assert key in graph.targets
    bank.shared.mark_used(moved, plan.retained_word, plan.exit_words[0])
    assert graph.executions[(key[0], graph.targets[key])] == 1
    assert graph.stats["unmapped_executions"] == 0
    # Only published plans may create pending targets; failed/duplicate plans
    # must not leave an undeliverable target at the front of the scheduler.
    published = {(s, p.retained, w) for s, pool in bank.shared.pools.items() for p in pool for w in p.exits}
    assert set(graph.targets) == published


@pytest.mark.parametrize("endpoint_dedup", [False, True])
def test_local_merge_reaches_existing_remote_plan_without_requeueing_used_exit(endpoint_dedup):
    points = generate_bell322_points(dtype=float)
    centered = points - points.mean(axis=0)
    lp = linprog(-np.random.default_rng(11).normal(size=26), A_ub=centered,
                 b_ub=np.ones(64), bounds=[(None, None)] * 26, method="highs")
    assert lp.success
    indices = np.flatnonzero(abs(centered @ lp.x - 1) < 1e-7)
    validation = validate_facet_support(points, indices)
    source = block_word(tuple(map(int, indices)))
    prototype = FacetCorrectorBank(points, seed=123, max_attempts=8)
    prototype.observe(source, validation.normal, validation.offset)
    for _ in range(8):
        prototype.grow(source)
    original = next(p for p in prototype.anchors[source].plans if len(p.exit_words) >= 2)
    retained = original.retained_word
    first, second = original.exit_words[:2]
    bank = FacetCorrectorBank(points, seed=123, endpoint_dedup=endpoint_dedup)
    bank.observe(source, validation.normal, validation.offset)
    anchor = bank.anchors[source]
    pattern = next(p for p in anchor.patterns if p.pattern_id == original.pattern_id)
    def remember(outside):
        check = validate_facet_support(points, bank.indices(retained | outside))
        assert check.valid
        bank._remember_endpoint(source, retained, outside, check)
    remember(first)
    initial = bank._publish_plan(anchor, retained, pattern, (first, first))
    assert initial.exit_words == (first,)
    permutation = next(g for g in build_state_group_permutations() if move_support_word(source, g) != source)
    moved = move_support_word(source, permutation)
    check = validate_facet_support(points, bank.indices(moved))
    bank.observe(moved, check.normal, check.offset)
    remote = bank._borrow(bank.anchors[moved])
    assert remote is not None and len(remote.exit_words) == 1
    bank.shared.mark_used(source, retained, first)
    remember(second)
    assert bank.to_dict()["publication_audit"]["unpublished_candidate_action_keys"] == int(endpoint_dedup)
    merged = bank._publish_plan(anchor, retained, pattern, (first, second, second))
    assert merged.exit_words == (first, second)
    assert len(anchor.plans) == 1
    assert bank.exits[(source, retained, pattern.pattern_id)] == merged.exit_words
    assert bank.stats["verified_plans"] == 1
    assert bank.stats["verified_exits"] == 2
    assert bank.stats["merged_local_plans"] == bank.stats["merged_local_exits"] == 1
    updated = bank._borrow(bank.anchors[moved])
    assert updated is not None and updated.pattern_id == remote.pattern_id
    assert len(bank.anchors[moved].plans) == 1
    assert len(updated.exit_words) == 2
    assert set(updated.exit_words) > set(remote.exit_words)
    assert bank.shared.uses(moved, updated.retained_word, remote.exit_words[0]) == 1
    key = bank.shared.exit_key(source, retained, first)
    assert key[1:] not in bank.shared.untried_exits[key[0]]
    new_key = bank.shared.exit_key(source, retained, second)
    assert new_key[1:] in bank.shared.untried_exits[new_key[0]]
    pool_size = len(bank.shared.pools[key[0]])
    assert bank._publish_plan(anchor, retained, pattern, (second, first)) is None
    assert len(bank.shared.pools[key[0]]) == pool_size
    assert bank.stats["duplicate_plan_unchanged"] == 1
    assert bank._borrow(bank.anchors[moved]) is None
    assert bank.to_dict()["publication_audit"]["unpublished_candidate_action_keys"] == 0
    assert len(bank.local_merge_events) == 1
