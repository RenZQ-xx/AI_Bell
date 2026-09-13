from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baseline.bell322 import generate_bell322_points
from baseline.orbit_blocks import (
    add_block,
    build_state_group_permutations,
    canonical_partition_key,
    empty_key,
)
from baseline.reference_classes import (
    parse_example_rows,
    support_mask_from_row,
    support_word_from_indices,
)
from baseline.scorer import ExpansionScorer
from mcts.interrupt_search import (
    InterruptGlobalDiscoveryState,
    InterruptSearchState,
    InterruptSearchTask,
    _is_identity_singleton_partition,
    _rank23_rank24_actions,
    _rank23_terminal_completion_batch,
    _rank23_terminal_completions,
    _run_one_iteration,
)
from mcts.search import MCTSConfig, MCTSNode
from mcts.pair_involution_search import build_pair_involution_patterns
from mcts.subgroup_interrupt_search import (
    DEFAULT_ATOMIC_TAIL_CANDIDATES,
    PatternFrontier,
    PatternJob,
    _support_word_for_discovery,
    parse_args,
)
from mcts.subgroup_patterns import (
    canonical_support_word,
    load_subgroup_pattern_atlas,
    move_support_word,
    permutation_group_closure,
)


def _class_support(class_id: int) -> object:
    rows = parse_example_rows()
    return support_mask_from_row(rows[int(class_id)][1])


def test_subgroup_cli_defaults_to_larger_atomic_tail_batch() -> None:
    default_args = parse_args([])
    override_args = parse_args(["--rank23-tail-candidates-per-step", "128"])

    assert DEFAULT_ATOMIC_TAIL_CANDIDATES == 512
    assert default_args.rank23_tail_candidates_per_step == 512
    assert override_args.rank23_tail_candidates_per_step == 128


def test_subgroup_atlas_reconstructs_two_levels_and_pair_roots() -> None:
    atlas = load_subgroup_pattern_atlas()

    assert atlas.source_metadata["facet_answers_used"] is False
    assert atlas.identity.order == 1
    assert len(atlas.identity.blocks) == 64
    assert len(atlas.roots) == 24
    assert len(atlas.level2) == 239
    assert len(atlas.edges) == 550
    assert len(atlas.pair_roots) == 18
    assert all(
        sorted(vertex for block in pattern.blocks for vertex in block)
        == list(range(64))
        for pattern in atlas.patterns
    )


def test_atlas_pair_roots_are_exactly_the_old_pair_involution_patterns() -> None:
    atlas = load_subgroup_pattern_atlas()
    atlas_keys = {
        canonical_partition_key(pattern.blocks)
        for pattern in atlas.pair_roots
    }
    old_pair_keys = {
        canonical_partition_key(pattern.blocks)
        for pattern in build_pair_involution_patterns()
    }

    assert atlas_keys == old_pair_keys


def test_canonical_support_word_is_invariant_under_bell_group_action() -> None:
    support = _class_support(1)
    support_word = sum(
        1 << index for index, selected in enumerate(support) if int(selected)
    )
    permutation = build_state_group_permutations()[-1]

    assert canonical_support_word(support_word) == canonical_support_word(
        move_support_word(support_word, permutation)
    )
    assert canonical_support_word(canonical_support_word(support_word)) == (
        canonical_support_word(support_word)
    )


def test_observed_class18_support_infers_one_pair_root_without_target_bank() -> None:
    atlas = load_subgroup_pattern_atlas()
    analysis = atlas.analyze_support(
        _class_support(18),
        include_level2=False,
    )

    assert analysis.support_size == 26
    assert analysis.stabilizer_order == 2
    assert len(analysis.root_pattern_ids) == 1
    assert atlas.by_id[analysis.root_pattern_ids[0]].is_pair_pattern
    assert analysis.root_subgroup_multiplicities == (
        (analysis.root_pattern_ids[0], 1),
    )


def test_high_order_class44_stabilizer_is_decomposed_into_minimal_roots() -> None:
    atlas = load_subgroup_pattern_atlas()
    analysis = atlas.analyze_support(
        _class_support(44),
        include_level2=False,
    )

    assert analysis.stabilizer_order == 48
    assert sorted(len(block) for block in analysis.stabilizer_blocks) == [8, 8, 24, 24]
    assert analysis.root_pattern_ids
    assert sum(count for _pattern_id, count in analysis.root_subgroup_multiplicities) > 1
    reconstructed = permutation_group_closure(analysis.stabilizer_generators)
    assert reconstructed is not None
    assert len(reconstructed) == 48


def test_order4_observed_stabilizer_maps_to_atlas_level2_pattern() -> None:
    atlas = load_subgroup_pattern_atlas()
    analysis = atlas.analyze_support(
        _class_support(25),
        include_level2=True,
        max_level2_pair_checks=32,
    )

    assert analysis.stabilizer_order == 4
    assert analysis.level2_pattern_ids
    assert all(atlas.by_id[pattern_id].level == 2 for pattern_id in analysis.level2_pattern_ids)


def test_discovery_pattern_uses_complete_tight_support_not_selected_basis() -> None:
    full_support = _class_support(44)
    support_indices = [
        index for index, selected in enumerate(full_support) if int(selected)
    ]
    points = generate_bell322_points(dtype=np.float64)
    chosen: list[int] = []
    current_rank = -1
    for vertex in support_indices:
        candidate = [*chosen, vertex]
        rank = (
            0
            if len(candidate) <= 1
            else int(
                np.linalg.matrix_rank(
                    points[candidate[1:]] - points[candidate[0]],
                    tol=1e-8,
                )
            )
        )
        if not chosen or rank > current_rank:
            chosen.append(vertex)
            current_rank = rank
        if current_rank == 25:
            break
    assert len(chosen) == 26

    scorer = ExpansionScorer(blocks=tuple((index,) for index in range(64)))
    frame = SimpleNamespace(
        task=SimpleNamespace(state=SimpleNamespace(scorer=scorer))
    )
    terminal_word = _support_word_for_discovery(frame, chosen)

    assert terminal_word == support_word_from_indices(support_indices)
    assert terminal_word.bit_count() == 56


def test_pattern_frontier_deduplicates_conjugate_partition_aliases() -> None:
    atlas = load_subgroup_pattern_atlas()
    frontier = PatternFrontier(atlas, max_tasks=10)
    first = PatternJob(
        pattern_id="BFS322-H-0023",
        anchor_class_id=44,
        source_class_id=44,
        reason="test",
    )
    alias = PatternJob(
        pattern_id="BFS322-H-0040",
        anchor_class_id=44,
        source_class_id=44,
        reason="test",
    )

    assert frontier.enqueue(first)
    assert not frontier.enqueue(alias)
    assert frontier.skipped[-1]["skip_reason"] == "canonical_partition_duplicate"


def test_rank23_tail_enumerates_two_vertex_full_rank_completion() -> None:
    scorer = ExpansionScorer(blocks=tuple((index,) for index in range(64)))
    support = _class_support(2)
    support_indices = [
        index for index, selected in enumerate(support) if int(selected)
    ]

    basis: list[int] = []
    key = empty_key(64)
    current_rank = scorer.affine_rank(key)
    for vertex in support_indices:
        candidate = add_block(key, vertex)
        candidate_rank = scorer.affine_rank(candidate)
        if basis and candidate_rank <= current_rank:
            continue
        basis.append(vertex)
        key = candidate
        current_rank = candidate_rank
        if current_rank == 25:
            break

    assert _is_identity_singleton_partition(scorer)
    assert len(basis) == 26

    prefix_key = empty_key(64)
    for vertex in basis[:24]:
        prefix_key = add_block(prefix_key, vertex)
    expected_key = add_block(add_block(prefix_key, basis[24]), basis[25])

    assert scorer.affine_rank(prefix_key) == 23
    assert any(
        terminal_key == expected_key
        for terminal_key, _suffix in _rank23_terminal_completions(
            scorer,
            prefix_key,
        )
    )
    assert scorer.terminal_label(expected_key).label == "exact:class2"


def test_rank23_tail_batches_resume_without_changing_enumeration() -> None:
    scorer = ExpansionScorer(blocks=tuple((index,) for index in range(64)))
    support_indices = [
        index for index, selected in enumerate(_class_support(2)) if int(selected)
    ]
    prefix_key = empty_key(64)
    current_rank = scorer.affine_rank(prefix_key)
    basis: list[int] = []
    for vertex in support_indices:
        candidate = add_block(prefix_key, vertex)
        candidate_rank = scorer.affine_rank(candidate)
        if basis and candidate_rank <= current_rank:
            continue
        basis.append(vertex)
        prefix_key = candidate
        current_rank = candidate_rank
        if current_rank == 23:
            break

    actions = _rank23_rank24_actions(scorer, prefix_key)
    cursor = 0
    batched = []
    pair_count = len(actions) * max(0, len(actions) - 1) // 2
    while cursor < pair_count:
        values, cursor = _rank23_terminal_completion_batch(
            scorer,
            prefix_key,
            actions,
            start_index=cursor,
            candidate_limit=17,
        )
        batched.extend(values)

    assert batched == list(_rank23_terminal_completions(scorer, prefix_key))


def test_rank23_atomic_continuation_matches_unbounded_tail() -> None:
    scorer = ExpansionScorer(blocks=tuple((index,) for index in range(64)))
    support_indices = [
        index for index, selected in enumerate(_class_support(2)) if int(selected)
    ]
    prefix_key = empty_key(64)
    current_rank = scorer.affine_rank(prefix_key)
    basis: list[int] = []
    for vertex in support_indices:
        candidate = add_block(prefix_key, vertex)
        candidate_rank = scorer.affine_rank(candidate)
        if basis and candidate_rank <= current_rank:
            continue
        basis.append(vertex)
        prefix_key = candidate
        current_rank = candidate_rank
        if current_rank == 23:
            break

    actions = _rank23_rank24_actions(scorer, prefix_key)
    pair_count = len(actions) * max(0, len(actions) - 1) // 2
    assert pair_count > 17

    root = MCTSNode(
        key=prefix_key,
        path=list(basis),
        rank=23,
        parent=None,
        action_from_parent=None,
    )
    state = InterruptSearchState(
        scorer=scorer,
        config=MCTSConfig(
            iterations=1,
            rank23_tail_enabled=True,
            rank23_tail_candidates_per_step=17,
            rank23_tail_active_service=True,
        ),
        global_discovery=InterruptGlobalDiscoveryState(),
        nodes={prefix_key: root},
        root=root,
    )
    task = InterruptSearchTask(
        class_id=2,
        search_index=1,
        parent_search_index=None,
        state=state,
        threshold=1_000_000.0,
        base_iteration_limit=1,
        max_iteration_limit=1,
        adaptive_min_iterations=1,
        adaptive_min_terminal_hits=1,
        adaptive_sink_ratio=1.0,
        adaptive_extra_iterations=0,
        productive_patience_iterations=1,
        iteration_limit=1,
    )
    discoveries = task.step()
    assert discoveries == []
    assert state.rank23_pending_iteration is not None
    assert state.iterations_completed == 0
    assert root.visits == 0

    continuation_calls = 0
    while state.rank23_pending_iteration is not None:
        batch_discoveries = task.step()
        continuation_calls += 1
        if state.rank23_pending_iteration is not None:
            assert batch_discoveries == []
        discoveries.extend(batch_discoveries)

    reference_scorer = ExpansionScorer(
        blocks=tuple((index,) for index in range(64))
    )
    reference_root = MCTSNode(
        key=prefix_key,
        path=list(basis),
        rank=23,
        parent=None,
        action_from_parent=None,
    )
    reference_state = InterruptSearchState(
        scorer=reference_scorer,
        config=MCTSConfig(
            iterations=1,
            rank23_tail_enabled=True,
            rank23_tail_candidates_per_step=0,
        ),
        global_discovery=InterruptGlobalDiscoveryState(),
        nodes={prefix_key: reference_root},
        root=reference_root,
    )
    reference_discoveries = _run_one_iteration(
        reference_state,
        iteration_index=1,
    )

    assert [item.to_dict() for item in discoveries] == [
        item.to_dict() for item in reference_discoveries
    ]
    assert root.visits == 1
    assert state.iterations_completed == 1
    assert state.rank23_tail_deferred_iterations == 1
    assert state.rank23_tail_active_service_batches == continuation_calls
    assert state.rank23_tail_max_batch_candidates == 17
