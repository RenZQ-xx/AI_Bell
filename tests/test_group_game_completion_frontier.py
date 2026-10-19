import numpy as np
import pytest
from scipy.optimize import linprog

from baseline.bell322 import generate_bell322_points
from baseline.geometry import affine_rank, validate_facet_support
from baseline.orbit_blocks import build_state_group_permutations
from mcts.group_game.completion_frontier import CompletionFrontier
from mcts.group_game.facet_corrector import FacetCorrectorBank
from mcts.group_game.model import block_word
from mcts.group_game.search import GroupGameConfig, build_parser
from mcts.subgroup_patterns import move_support_word


def test_completion_jobs_are_shared_bounded_and_round_robin():
    frontier = CompletionFrontier(seed=123, max_attempts=6)
    first = frontier.register((1, 2, "p"), source=1, retained=2, pattern_id="p")
    first.attempts = 2
    again = frontier.register((1, 2, "p"), source=9, retained=8, pattern_id="q")
    assert again is first and again.attempts == 2
    second = frontier.register((1, 3, "p"), source=1, retained=3, pattern_id="p")
    assert frontier.next_job(1) is first
    assert frontier.next_job(1) is second
    first.attempts = 6
    assert frontier.next_job(1) is second
    second.attempts = 6
    assert not frontier.pending(1)
    assert frontier.next_job(1) is None
    assert not frontier.pending(99)


def test_completion_rng_is_independent_of_registration_order():
    a = CompletionFrontier(seed=123, max_attempts=6)
    b = CompletionFrontier(seed=123, max_attempts=6)
    b.register((1, 7, "q"), source=1, retained=7, pattern_id="q")
    left = a.register((1, 2, "p"), source=1, retained=2, pattern_id="p")
    right = b.register((1, 2, "p"), source=1, retained=2, pattern_id="p")
    assert np.array_equal(left.rng.normal(size=10), right.rng.normal(size=10))


@pytest.mark.parametrize("endpoint_dedup", [False, True])
def test_lower_face_continuation_shares_budget_and_publishes_geometry(endpoint_dedup):
    points = generate_bell322_points(dtype=float)
    centered = points - points.mean(axis=0)
    lp = linprog(-np.random.default_rng(11).normal(size=26), A_ub=centered,
                 b_ub=np.ones(64), bounds=[(None, None)] * 26, method="highs")
    assert lp.success
    indices = list(map(int, np.flatnonzero(np.abs(centered @ lp.x - 1) < 1e-7)))
    validation = validate_facet_support(points, indices)
    assert validation.valid
    source = block_word(tuple(indices))
    bank = FacetCorrectorBank(points, seed=123, max_attempts=0,
                             endpoint_dedup=endpoint_dedup, lower_face_max_attempts=6)
    bank.current_iteration = 1
    bank.observe(source, validation.normal, validation.offset)
    retained_indices = []
    for vertex in indices:
        candidate = retained_indices + [vertex]
        if affine_rank(points[candidate]) == len(retained_indices):
            retained_indices = candidate
        if len(retained_indices) == 24:
            break
    retained = block_word(tuple(retained_indices))
    assert affine_rank(points[retained_indices]) == 23
    anchor = bank.anchors[source]
    pattern = anchor.patterns[0]
    exits = bank._lower_face_exits(anchor, pattern, retained)
    if exits:
        bank._publish_plan(anchor, retained, pattern, tuple(exits))
    job = next(iter(bank.completions.jobs.values()))
    assert job.attempts == 2
    permutation = next(g for g in build_state_group_permutations()
                       if move_support_word(source, g) != source)
    moved = move_support_word(source, permutation)
    check = validate_facet_support(points, bank.indices(moved))
    bank.observe(moved, check.normal, check.offset)
    moved_retained = move_support_word(job.key[1], bank.shared.coordinates[moved][2])
    other = bank.anchors[moved]
    other_pattern = other.patterns[0]
    exits = bank._lower_face_exits(other, other_pattern, moved_retained, attempts=1)
    if exits:
        bank._publish_plan(other, moved_retained, other_pattern, tuple(exits))
    assert len(bank.completions.jobs) == 1
    assert job.attempts == 3
    assert bank.has_delivery_frontier(source) and bank.has_delivery_frontier(moved)
    state = repr(anchor.rng.bit_generator.state)
    before = bank.stats["completion_lp_attempts"]
    for word in (moved, source, moved):
        plan = bank._grow_lower_faces(bank.anchors[word])
        if plan is not None:
            assert plan.source_word == word
    assert repr(anchor.rng.bit_generator.state) == state
    assert job.attempts == 6
    assert bank.stats["completion_lp_attempts"] == before + 3
    assert job.continuation_batches == 2
    assert bank._lower_face_exits(anchor, pattern, retained) == []
    assert bank.stats["completion_lp_attempts"] == before + 3
    assert not bank.completions.pending(job.key[0])
    assert bank.stats["completion_continuation_lp_attempts"] == 3
    assert job.exit_keys
    for owner in bank.anchors.values():
        for plan in owner.plans:
            for outside in plan.exit_words:
                assert outside & ~owner.word
                assert validate_facet_support(points, bank.indices(plan.retained_word | outside)).valid
                assert bank.shared.uses(owner.word, plan.retained_word, outside) == 0
    assert not bank.to_dict()["publication_audit"]["unpublished_candidate_action_keys"]


def test_completion_config_and_disable_switch():
    assert build_parser().parse_args([]).lower_face_max_attempts == 6
    assert GroupGameConfig(lower_face_max_attempts=0).lower_face_max_attempts == 0
    with pytest.raises(ValueError, match="completion budget"):
        GroupGameConfig(lower_face_max_attempts=-1)
    with pytest.raises(ValueError, match="completion budget"):
        CompletionFrontier(seed=1, max_attempts=-1)
