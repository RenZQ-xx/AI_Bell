from __future__ import annotations

import random
from collections import Counter
from types import SimpleNamespace

import mcts.filler_corrector_search as filler_corrector_search
from mcts.filler_corrector_search import (
    BlockCorrector,
    CORRECTOR_FEATURE_NAMES,
    CorrectorConfig,
    FillerCorrectorTask,
    SharedCorrectorPolicy,
    remove_blocks,
)
from mcts.interrupt_search import (
    InterruptSearchState,
    InterruptSearchTask,
)
from mcts.search import ExactClassDiscovery, MCTSConfig, MCTSNode, TerminalHit


class _FakeScorer:
    blocks = [(0,), (1,), (2,), (3,)]
    target_classes = {1, 2, 3}

    @staticmethod
    def affine_rank(key: tuple[int, ...]) -> int:
        return sum(int(value) for value in key)

    @staticmethod
    def support_key(key: tuple[int, ...]) -> int:
        return sum(1 << index for index, value in enumerate(key) if int(value))

    @staticmethod
    def rank24_entrance_metrics(key: tuple[int, ...]) -> dict[str, int]:
        return {
            "rare": 1 if key[3] == 0 else 0,
            "class44": 0,
            "other_valid": 1,
            "invalid": 0,
        }

    @staticmethod
    def rank24_entrance_class_counts(
        key: tuple[int, ...],
    ) -> tuple[dict[int, int], int]:
        return ({1: 1, 2: 2} if key[3] == 0 else {1: 1}), 0


class _FakeCompatibilityBank:
    @staticmethod
    def active_signature(
        _key: tuple[int, ...],
        class_ids: list[int],
    ) -> tuple[int, ...]:
        del class_ids
        raise AssertionError("Corrector must not query undiscovered-class compatibility")


def test_remove_blocks_does_not_mutate_source_key() -> None:
    source = (1, 1, 0, 1)

    corrected = remove_blocks(source, [1, 3])

    assert source == (1, 1, 0, 1)
    assert corrected == (1, 0, 0, 0)


def test_corrector_protects_core_and_returns_nonterminal_high_rank_prefix() -> None:
    source = TerminalHit(
        label="exact:class1",
        key=(1, 1, 1, 1),
        path=[0, 1, 2, 3],
        score=10.0,
        rank=4,
    )
    state = SimpleNamespace(
        scorer=_FakeScorer(),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovered_label_counts=Counter({"exact:class1": 5}),
            discovery_epoch=0,
        ),
        terminal_bests={source.label: source},
        compatibility_bank=_FakeCompatibilityBank(),
    )
    corrector = BlockCorrector(
        CorrectorConfig(
            protected_rank=1,
            min_corrected_rank=3,
            max_corrected_rank=3,
            terminal_rank=4,
            max_remove_blocks=1,
            candidate_pool=4,
        )
    )

    proposal = corrector.propose(state, rng=random.Random(7))

    assert proposal is not None
    assert proposal.protected_blocks == (0,)
    assert 0 not in proposal.removed_blocks
    assert proposal.removed_blocks == (3,)
    assert proposal.corrected_rank == 3
    assert proposal.corrected_rank < corrector.config.terminal_rank
    assert proposal.entrance_rare_count == 1


def test_corrector_does_not_repeat_the_same_corrected_prefix() -> None:
    source = TerminalHit(
        label="exact:class1",
        key=(1, 1, 1, 1),
        path=[0, 1, 2, 3],
        score=10.0,
        rank=4,
    )
    state = SimpleNamespace(
        scorer=_FakeScorer(),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovered_label_counts=Counter(),
            discovery_epoch=0,
        ),
        terminal_bests={source.label: source},
        compatibility_bank=None,
    )
    corrector = BlockCorrector(
        CorrectorConfig(
            protected_rank=1,
            min_corrected_rank=3,
            max_corrected_rank=3,
            terminal_rank=4,
            candidate_pool=4,
        )
    )

    first = corrector.propose(state, rng=random.Random(3))
    second = corrector.propose(state, rng=random.Random(3))

    assert first is not None
    assert second is not None
    assert first.corrected_key != second.corrected_key


def test_corrector_can_offer_one_and_two_block_deletions() -> None:
    source = TerminalHit(
        label="exact:class1",
        key=(1, 1, 1, 1),
        path=[0, 1, 2, 3],
        score=10.0,
        rank=4,
    )
    state = SimpleNamespace(
        scorer=_FakeScorer(),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovered_label_counts=Counter(),
            discovery_epoch=0,
        ),
        terminal_bests={source.label: source},
        compatibility_bank=_FakeCompatibilityBank(),
    )
    corrector = BlockCorrector(
        CorrectorConfig(
            protected_rank=0,
            min_corrected_rank=2,
            max_corrected_rank=3,
            terminal_rank=4,
            max_remove_blocks=2,
            candidate_pool=10,
            tournament_candidates=10,
            tournament_finalists=2,
        )
    )

    proposals = corrector.propose_candidates(
        state,
        rng=random.Random(11),
        limit=10,
    )

    assert {len(proposal.removed_blocks) for proposal in proposals} == {1, 2}


def test_shared_corrector_policy_learns_across_task_local_correctors() -> None:
    config = CorrectorConfig(window_iterations=1)
    policy = SharedCorrectorPolicy(config)
    first = BlockCorrector(config, policy)
    second = BlockCorrector(config, policy)
    features = (1.0,) + (0.0,) * (len(CORRECTOR_FEATURE_NAMES) - 1)

    before = policy.evaluate(features, discovery_epoch=0)[0]
    first.observe(features, 4.0, discovery_epoch=0)
    after = second.shared_policy.evaluate(features, discovery_epoch=0)[0]

    assert before == 0.0
    assert after > before
    assert policy.summary()["observations"] == 1


def test_shared_corrector_policy_bounds_concurrency_and_global_budget() -> None:
    config = CorrectorConfig(
        window_iterations=20,
        global_iteration_budget=40,
        max_concurrent_tournaments=1,
    )
    policy = SharedCorrectorPolicy(config)

    assert policy.try_begin_tournament(20)
    assert not policy.try_begin_tournament(20)
    policy.finish_tournament(20, 20)
    assert policy.try_begin_tournament(20)
    policy.finish_tournament(20, 20)
    assert not policy.try_begin_tournament(20)
    summary = policy.summary()
    assert summary["completed_corrector_iterations"] == 40
    assert summary["active_tournament_peak"] == 1
    assert summary["denied_tournaments"] == 2


def test_corrector_scores_unseen_rank24_terminal_exits_without_examples() -> None:
    source = TerminalHit(
        label="exact:class1",
        key=(1, 1, 1, 1),
        path=[0, 1, 2, 3],
        score=10.0,
        rank=4,
    )
    state = SimpleNamespace(
        scorer=_FakeScorer(),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovered_label_counts=Counter({"exact:class1": 1}),
            discovery_epoch=0,
        ),
        terminal_bests={source.label: source},
        compatibility_bank=None,
    )
    corrector = BlockCorrector(
        CorrectorConfig(
            protected_rank=1,
            min_corrected_rank=3,
            max_corrected_rank=3,
            terminal_rank=4,
            max_remove_blocks=1,
            lookahead_new_class_reward=10.0,
        )
    )

    proposal = corrector.propose(state, rng=random.Random(5))

    assert proposal is not None
    assert proposal.compatible_undiscovered_classes == (2,)
    assert proposal.entrance_class_entropy > 0.0


def test_basin_stagnation_can_trigger_before_fixed_start_iteration() -> None:
    scorer = _FakeScorer()
    root = MCTSNode(
        key=(0, 0, 0, 0),
        path=[],
        rank=0,
        parent=None,
        action_from_parent=None,
    )
    state = InterruptSearchState(
        scorer=scorer,  # type: ignore[arg-type]
        config=MCTSConfig(iterations=20),
        nodes={root.key: root},
        root=root,
        terminal_bests={
            "exact:class1": TerminalHit(
                label="exact:class1",
                key=(1, 1, 1, 1),
                path=[0, 1, 2, 3],
                score=10.0,
                rank=25,
            )
        },
        productive=True,
        iterations_completed=8,
    )
    base_task = InterruptSearchTask(
        class_id=1,
        search_index=1,
        parent_search_index=None,
        state=state,
        threshold=10.0,
        base_iteration_limit=20,
        max_iteration_limit=20,
        adaptive_min_iterations=1,
        adaptive_min_terminal_hits=1,
        adaptive_sink_ratio=0.9,
        adaptive_extra_iterations=0,
        productive_patience_iterations=20,
        iteration_limit=20,
    )
    task = FillerCorrectorTask(
        base_task,
        corrector_config=CorrectorConfig(
            start_iteration=100,
            basin_trigger_min_exact_hits=4,
            basin_stagnation_iterations=4,
            basin_trigger_duplicate_ratio=0.75,
        ),
        event_sink=[],
    )
    task._basin_counts.update({3: 4})
    task._last_new_basin_iteration = 2

    assert task._basin_trigger_ready(8)
    assert task._eligible_for_correction()


def test_structural_basin_utility_rewards_novelty_and_penalizes_repetition() -> None:
    config = CorrectorConfig(
        new_class_reward=0.0,
        distinct_exact_reward=0.0,
        size_delta_weight=0.0,
        wall_time_penalty=0.0,
        new_basin_reward=2.0,
        basin_novelty_weight=0.5,
        repeated_basin_penalty=0.25,
    )
    node = MCTSNode(
        key=(0, 0, 0, 0),
        path=[],
        rank=0,
        parent=None,
        action_from_parent=None,
    )
    novel = filler_corrector_search._CorrectionArm(
        name="novel",
        proposal=None,
        nodes={node.key: node},
        root=node,
        resume_root=node,
        rng=random.Random(1),
        reference_vertex_count=0,
        completed_iterations=1,
        new_basin_count=1,
        basin_novelty_sum=1.0,
    )
    repeated = filler_corrector_search._CorrectionArm(
        name="repeated",
        proposal=None,
        nodes={node.key: node},
        root=node,
        resume_root=node,
        rng=random.Random(1),
        reference_vertex_count=0,
        completed_iterations=1,
        repeated_basin_count=1,
        basin_novelty_sum=0.5,
    )

    assert novel.utility_rate(scorer=_FakeScorer(), config=config) > repeated.utility_rate(
        scorer=_FakeScorer(),
        config=config,
    )


def test_corrector_tournament_isolates_arms_and_keeps_winner(monkeypatch) -> None:
    scorer = _FakeScorer()
    root = MCTSNode(
        key=(0, 0, 0, 0),
        path=[],
        rank=0,
        parent=None,
        action_from_parent=None,
    )
    source = TerminalHit(
        label="exact:class1",
        key=(1, 1, 1, 1),
        path=[0, 1, 2, 3],
        score=10.0,
        rank=4,
    )
    state = InterruptSearchState(
        scorer=scorer,  # type: ignore[arg-type]
        config=MCTSConfig(iterations=6, max_depth=1, seed=17),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            remaining_rare_target_classes=set(),
            discovered_label_counts=Counter({"exact:class1": 1}),
            discovery_epoch=0,
        ),
        nodes={root.key: root},
        root=root,
        terminal_bests={source.label: source},
        productive=True,
    )
    base_task = InterruptSearchTask(
        class_id=1,
        search_index=1,
        parent_search_index=None,
        state=state,
        threshold=10.0,
        base_iteration_limit=6,
        max_iteration_limit=6,
        adaptive_min_iterations=1,
        adaptive_min_terminal_hits=1,
        adaptive_sink_ratio=0.9,
        adaptive_extra_iterations=0,
        productive_patience_iterations=20,
        iteration_limit=6,
    )
    events = []
    task = FillerCorrectorTask(
        base_task,
        corrector_config=CorrectorConfig(
            protected_rank=1,
            min_corrected_rank=3,
            max_corrected_rank=3,
            terminal_rank=4,
            max_remove_blocks=1,
            start_iteration=0,
            stagnation_iterations=0,
            window_iterations=4,
            cooldown_iterations=0,
            max_corrections_per_task=1,
            tournament_candidates=1,
            tournament_pilot_iterations=1,
            tournament_finalists=1,
            tournament_finalist_iterations=0,
        ),
        event_sink=events,
    )

    def fake_iteration(
        active_state: InterruptSearchState,
        iteration_index: int,
    ) -> list[ExactClassDiscovery]:
        assert active_state.root is not None
        active_state.root.visits += 1
        if active_state.root.rank != 3:
            return []
        hit = TerminalHit(
            label="exact:class2",
            key=(1, 1, 1, 1),
            path=[0, 1, 2, 3],
            score=100.0,
            rank=4,
        )
        active_state.last_exact_hit = hit
        active_state.exact_hit_serial += 1
        return [
            ExactClassDiscovery(
                class_id=2,
                label=hit.label,
                iteration=iteration_index,
                depth=4,
                score=hit.score,
                rank=hit.rank,
                path=list(hit.path),
                chosen_blocks=[0, 1, 2, 3],
            )
        ]

    monkeypatch.setattr(filler_corrector_search, "_run_one_iteration", fake_iteration)

    task.step()
    while task._active_event is not None:
        task.step()

    assert events[0].selected_arm == "delete_1"
    assert events[0].counterfactual_advantage > 0.0
    assert state.root is not None
    assert state.root.key == (0, 0, 0, 0)
    assert state.root.visits == 1
    assert any(node.rank == 3 for node in state.nodes.values())
