from __future__ import annotations

import inspect
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

import mcts.pair_involution_search as pair_search
from baseline.bell322 import generate_bell322_points
from baseline.reference_classes import build_support_class_index
from mcts.pair_involution_search import (
    GlobalDiscoveryState,
    PairPatternTree,
    PairSearchConfig,
    PairTargetBank,
    SharedReferenceTerminalCache,
    _basis_for_mask,
    _extend_pair_basis,
    build_pair_involution_patterns,
    corrected_prefix_masks,
    run_pair_involution_search,
    tail_completion_masks,
)


def _augmented_points() -> np.ndarray:
    points = np.asarray(generate_bell322_points(), dtype=np.float64)
    return np.concatenate(
        [np.ones((len(points), 1), dtype=np.float64), points],
        axis=1,
    )


def test_pair_involutions_form_18_partitions_covering_all_261_elements() -> None:
    patterns = build_pair_involution_patterns()

    assert len(patterns) == 18
    assert sum(pattern.conjugacy_size for pattern in patterns) == 261
    assert sorted(pattern.conjugacy_size for pattern in patterns) == [
        1,
        3,
        3,
        6,
        6,
        6,
        8,
        12,
        12,
        12,
        12,
        12,
        24,
        24,
        24,
        24,
        24,
        48,
    ]
    for pattern in patterns:
        assert len(pattern.blocks) == 32
        assert sorted(vertex for block in pattern.blocks for vertex in block) == list(range(64))
        assert all(pattern.involution[pattern.involution[index]] == index for index in range(64))
        assert all(pattern.involution[index] != index for index in range(64))


def test_class18_has_one_pair_pattern_and_64_representable_supports() -> None:
    support_index = build_support_class_index()
    matches = []
    for pattern in build_pair_involution_patterns():
        bank = PairTargetBank(pattern, support_index, range(1, 47))
        if 18 in bank.masks_by_class:
            matches.append((pattern, bank))

    assert len(matches) == 1
    pattern, bank = matches[0]
    assert pattern.pattern_id == 3
    assert len(bank.masks_by_class[18]) == 64
    assert all(mask.bit_count() == 13 for mask in bank.masks_by_class[18])


def test_every_class18_pair_prefix_has_the_maximum_affine_rank() -> None:
    support_index = build_support_class_index()
    pattern = build_pair_involution_patterns()[2]
    bank = PairTargetBank(pattern, support_index, range(1, 47))
    target_mask = bank.masks_by_class[18][0]
    augmented = _augmented_points()
    basis = np.zeros((0, augmented.shape[1]), dtype=np.float64)

    for depth, block_index in enumerate(
        [index for index in range(32) if target_mask & (1 << index)],
        start=1,
    ):
        basis, added = _extend_pair_basis(
            basis,
            augmented,
            pattern.blocks[block_index],
            1e-9,
        )
        assert added == 2
        assert basis.shape[0] - 1 == 2 * depth - 1


def test_rank21_rank23_tail_and_corrector_cardinalities() -> None:
    terminal_mask = sum(1 << index for index in range(13))
    rank21_prefix = sum(1 << index for index in range(11))
    rank23_prefix = sum(1 << index for index in range(12))

    assert len(tail_completion_masks(rank21_prefix)) == 210
    assert len(set(tail_completion_masks(rank21_prefix))) == 210
    assert len(tail_completion_masks(rank23_prefix)) == 20
    assert len(corrected_prefix_masks(terminal_mask, 1)) == 13
    assert len(corrected_prefix_masks(terminal_mask, 2)) == 78


def test_rank19_terminal_lookahead_finds_class18_without_target_bank() -> None:
    support_index = build_support_class_index()
    pattern = build_pair_involution_patterns()[2]
    diagnostics = PairTargetBank(pattern, support_index, range(1, 47))
    target_mask = diagnostics.masks_by_class[18][0]
    selected = [index for index in range(32) if target_mask & (1 << index)]
    rank19_mask = sum(1 << index for index in selected[:10])
    config = PairSearchConfig(
        iterations=1,
        rank19_lookahead_candidate_pool=32,
        corrector_interval=1,
        corrector_max_events_per_tree=2,
        checkpoint_interval=0,
        pattern_ids=(3,),
    )
    started = time.perf_counter()
    global_discovery = GlobalDiscoveryState.from_config(config, started_at=started)
    terminal_cache = SharedReferenceTerminalCache(support_index)
    tree = PairPatternTree(
        pattern=pattern,
        augmented_points=_augmented_points(),
        terminal_cache=terminal_cache,
        global_discovery=global_discovery,
        config=config,
    )
    basis, full_rank = _basis_for_mask(
        rank19_mask,
        pattern.blocks,
        tree.augmented_points,
        config.full_rank_tol,
    )
    assert full_rank
    support_word = 0
    for action in selected[:10]:
        support_word = tree._support_word_after(support_word, action)
    extensions = tree._extensions(
        rank19_mask,
        basis,
        stage="test_rank19",
    )

    tree._rank19_lookahead_choice(
        extensions,
        selected_mask=rank19_mask,
        support_word=support_word,
        local_iteration=1,
        global_iteration=1,
    )
    tree._run_corrector(local_iteration=1, global_iteration=1)

    assert 18 in global_discovery.discovered_classes
    assert tree.local_new_classes == [18]
    assert [event.delete_count for event in tree.corrector_events] == [1, 2]
    assert all(event.corrected_rank in {21, 23} for event in tree.corrector_events)
    assert Counter(event.delete_count for event in tree.corrector_events) == Counter({1: 1, 2: 1})


def test_search_tree_has_no_target_bank_dependency() -> None:
    assert "target_bank" not in inspect.signature(PairPatternTree).parameters
    support_index = build_support_class_index()
    pattern = build_pair_involution_patterns()[0]
    config = PairSearchConfig(
        iterations=3,
        rank19_lookahead_candidate_pool=1,
        corrector_enabled=False,
        checkpoint_interval=0,
        pattern_ids=(1,),
    )
    global_discovery = GlobalDiscoveryState.from_config(
        config,
        started_at=time.perf_counter(),
    )
    tree = PairPatternTree(
        pattern=pattern,
        augmented_points=_augmented_points(),
        terminal_cache=SharedReferenceTerminalCache(support_index),
        global_discovery=global_discovery,
        config=config,
    )

    for iteration in range(1, 4):
        tree.run_iteration(iteration, iteration)

    assert tree.stats["iterations"] == 3
    assert "uniform_full_rank" in tree.policy_bucket_counts


def test_pair_target_bank_is_constructed_only_after_search_iterations() -> None:
    events: list[str] = []
    original_bank = pair_search.PairTargetBank
    original_iteration = pair_search.PairPatternTree.run_iteration

    class TrackingBank(original_bank):
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert "iteration" in events
            events.append("post_search_bank")
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def tracked_iteration(
        tree: PairPatternTree,
        local_iteration: int,
        global_iteration: int,
    ) -> None:
        events.append("iteration")
        original_iteration(tree, local_iteration, global_iteration)

    pair_search.PairTargetBank = TrackingBank
    pair_search.PairPatternTree.run_iteration = tracked_iteration
    try:
        with tempfile.TemporaryDirectory() as directory:
            payload = run_pair_involution_search(
                PairSearchConfig(
                    iterations=1,
                    pattern_ids=(1,),
                    rank19_lookahead_candidate_pool=0,
                    corrector_enabled=False,
                    checkpoint_interval=0,
                ),
                output_path=Path(directory) / "pair_post_search_bank.json",
            )
    finally:
        pair_search.PairTargetBank = original_bank
        pair_search.PairPatternTree.run_iteration = original_iteration

    assert events == ["iteration", "post_search_bank"]
    assert payload["meta"]["target_bank_usage"] == "post_search_diagnostics_only"
    assert payload["runs"][0]["representability_analysis_phase"] == "post_search"
