from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import random
import threading
from types import SimpleNamespace

import numpy as np

from baseline.facet_validator import FacetValidator
from baseline.reference_classes import parse_example_rows, support_mask_from_row
from baseline.scorer import (
    ExpansionScore,
    ExpansionScorer,
    ExpansionStructure,
    ScorerConfig,
    SharedScorerStructureCache,
    SharedTerminalValidationCache,
)
from mcts.interrupt_search import (
    InterruptGlobalDiscoveryState,
    InterruptSearchConfig,
    InterruptSearchState,
    InterruptSearchTask,
    Rank23TailProgress,
    _canonical_pattern_signature,
    _compatibility_frontier_class_ids,
    _compatibility_frontier_snapshot,
    _enqueue_rank23_tail_work,
    _finish_rank23_tail_work,
    _pop_rank23_tail_work,
    _productive_iteration_limit,
    _rollout_score_batch_for_rollout,
    _run_one_iteration,
    _search_task_identity,
    _should_stop_for_exact_frequency,
    _should_stop_as_terminal_sink,
    _should_stop_productive_for_no_novelty,
    _synchronize_discovery_epoch,
    run_interruptible_search,
)
from mcts.search import (
    ExactClassDiscovery,
    MCTSConfig,
    MCTSNode,
    _prepare_actions,
    _progressive_child_limit,
    _refresh_discovery_dependent_action_scores,
    _select_progressive_action,
    _widening_score_batch,
)


def test_epoch_refresh_does_not_compare_numpy_payloads() -> None:
    key = (0, 0)
    candidate = (1, 0)
    payload = np.asarray([1.0, 2.0])
    original = ExpansionScore(
        action=0,
        key=candidate,
        score=3.0,
        phase="B2",
        old_rank=23,
        new_rank=24,
        rank_gain=1,
        flat_capacity=0,
        supportability=payload,  # type: ignore[arg-type]
        terminal=None,
    )
    node = MCTSNode(
        key=key,
        path=[],
        rank=23,
        parent=None,
        action_from_parent=None,
        action_scores={0: original},
    )

    class ArrayPayloadScorer:
        def score_action(self, _key, action, *, terminal_score_fn=None):
            del terminal_score_fn
            return ExpansionScore(
                action=int(action),
                key=candidate,
                score=3.0,
                phase="B2",
                old_rank=23,
                new_rank=24,
                rank_gain=1,
                flat_capacity=0,
                supportability=payload.copy(),  # type: ignore[arg-type]
                terminal=None,
            )

    _refresh_discovery_dependent_action_scores(
        node,
        ArrayPayloadScorer(),  # type: ignore[arg-type]
        MCTSConfig(),
        terminal_score_fn=None,
    )

    assert node.action_scores[0] is original


def test_global_discovery_state_is_atomic_across_queue_workers() -> None:
    state = InterruptGlobalDiscoveryState(
        initial_rare_target_classes=set(range(1, 65)),
        remaining_rare_target_classes=set(range(1, 65)),
    )

    def worker() -> None:
        for class_id in range(1, 65):
            state.mark_discovered(class_id)
            increment = getattr(state.discovered_label_counts, "increment")
            increment("exact:shared")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker) for _ in range(4)]
        for future in futures:
            future.result()

    discovered, remaining_rare, epoch = state.search_snapshot()
    assert discovered == set(range(1, 65))
    assert remaining_rare == set()
    assert epoch == 64
    assert state.discovered_label_counts_snapshot()["exact:shared"] == 256


@dataclass
class FakeResult:
    best: object | None = None
    terminal_bests: dict[str, object] = field(default_factory=dict)
    exact_discoveries: list[ExactClassDiscovery] = field(default_factory=list)
    encountered_label_counts: Counter[str] = field(default_factory=Counter)
    iterations_completed: int = 0
    nodes_created: int = 0
    root_visits: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "best": None,
            "terminal_bests": {},
            "exact_discoveries": [discovery.to_dict() for discovery in self.exact_discoveries],
            "encountered_label_counts": dict(self.encountered_label_counts),
            "exact_labels": [],
            "exact_class_ids": [discovery.class_id for discovery in self.exact_discoveries],
            "iterations_completed": self.iterations_completed,
            "nodes_created": self.nodes_created,
            "root_visits": self.root_visits,
        }


@dataclass
class FakeTask:
    class_id: int
    search_index: int
    parent_search_index: int | None
    script: list[list[ExactClassDiscovery]]
    finish_after: int
    started_interrupts: list[int] = field(default_factory=list)
    resumed_interrupts: list[int] = field(default_factory=list)
    state: object = field(
        default_factory=lambda: SimpleNamespace(
            config=SimpleNamespace(seed=0),
            stop_reason="fake_finished",
        )
    )

    def __post_init__(self) -> None:
        self._step_count = 0
        self._finished = False
        self.iteration_limit = self.finish_after

    @property
    def finished(self) -> bool:
        return self._finished

    def step(self) -> list[ExactClassDiscovery]:
        if self._finished:
            return []
        if self._step_count >= len(self.script):
            self._finished = True
            return []
        discoveries = list(self.script[self._step_count])
        self._step_count += 1
        if self._step_count >= self.finish_after:
            self._finished = True
        return discoveries

    def result(self) -> FakeResult:
        flat_discoveries = [discovery for bucket in self.script for discovery in bucket]
        return FakeResult(exact_discoveries=flat_discoveries)


def test_interruptible_search_pauses_and_resumes_parent() -> None:
    config = InterruptSearchConfig(
        initial_class_id=1,
        target_classes=(1, 2),
        rare_target_classes=(),
        iterations=3,
        max_depth=1,
        seed=7,
    )

    class2_hit = ExactClassDiscovery(
        class_id=2,
        label="exact:class2",
        iteration=1,
        depth=1,
        score=10.0,
        rank=25,
        path=[1],
        chosen_blocks=[1],
    )

    def task_factory(class_id: int, search_index: int, parent_search_index: int | None) -> FakeTask:
        if class_id == 1:
            return FakeTask(
                class_id=1,
                search_index=search_index,
                parent_search_index=parent_search_index,
                script=[[class2_hit], [], []],
                finish_after=3,
            )
        if class_id == 2:
            return FakeTask(
                class_id=2,
                search_index=search_index,
                parent_search_index=parent_search_index,
                script=[[], []],
                finish_after=2,
            )
        raise AssertionError(f"unexpected class_id {class_id}")

    report = run_interruptible_search(config, task_factory=task_factory)

    assert report.started_class_ids == [1, 2]
    assert [run.start_class_id for run in report.runs] == [2, 1]
    assert report.runs[1].resumed_after_class_ids == [2]
    assert [event.class_id for event in report.interrupt_events] == [2]


def test_exact_frequency_threshold_helper() -> None:
    fake_state = SimpleNamespace(
        productive=False,
        local_discovered_exact_classes={5},
        scorer=SimpleNamespace(discovered_label_counts=Counter({"exact:class5": 5})),
    )
    fake_task = SimpleNamespace(class_id=5, state=fake_state, threshold=4.0)

    assert _should_stop_for_exact_frequency(fake_task)
    fake_task.threshold = 5.0
    assert not _should_stop_for_exact_frequency(fake_task)

    fake_task.threshold = 4.0
    fake_state.scorer.discovered_label_counts["exact:class6"] = 6
    assert not _should_stop_for_exact_frequency(fake_task)
    fake_state.scorer.discovered_label_counts["exact:class6"] = 1
    fake_state.local_discovered_exact_classes.add(6)
    assert not _should_stop_for_exact_frequency(fake_task)
    fake_state.productive = True
    assert not _should_stop_for_exact_frequency(fake_task)


class FakeTerminalScorer:
    def __init__(self) -> None:
        self.blocks = [(0,)]
        self.target_classes = {9}
        self.rare_target_classes = {9}
        self.discovered_label_counts: Counter[str] = Counter()
        self.rank24_entrance_cache: dict[object, object] = {}
        self.terminal = SimpleNamespace(label="exact:class9", is_exact=True)
        self.terminal_score_calls = 0
        self.config = SimpleNamespace(
            terminal_scoring_mode="static",
            dynamic_new_class_score=100.0,
            dynamic_known_class_score=10.0,
            dynamic_frequent_class_score=-5.0,
            dynamic_frequent_class_threshold=16,
            class44_terminal_score=-5.0,
            target_terminal_score=10.0,
            valid_terminal_score=1.0,
        )

    def affine_rank(self, key: tuple[int, ...]) -> int:
        return 25 if key == (1,) else 24

    def terminal_label(self, _key: tuple[int, ...]) -> object:
        return self.terminal

    def terminal_score(
        self,
        _key: tuple[int, ...],
        *,
        rare_target_classes: set[int] | None = None,
        discovered_label_counts: Counter[str] | None = None,
    ) -> float:
        del discovered_label_counts
        self.terminal_score_calls += 1
        active_rare = self.rare_target_classes if rare_target_classes is None else rare_target_classes
        return 100.0 if 9 in active_rare else 10.0

    def score_action(
        self,
        key: tuple[int, ...],
        action: int,
        *,
        terminal_score_fn=None,
    ) -> ExpansionScore:
        score = (
            100.0
            if terminal_score_fn is None
            else float(terminal_score_fn((1,), self.terminal))
        )
        return ExpansionScore(
            action=action,
            key=(1,),
            score=score,
            phase="C",
            old_rank=self.affine_rank(key),
            new_rank=25,
            rank_gain=1,
            flat_capacity=0,
            supportability=None,
            terminal=self.terminal,
        )


def test_cross_class_discovery_permanently_marks_task_productive() -> None:
    shared = InterruptGlobalDiscoveryState(
        initial_rare_target_classes={9},
        remaining_rare_target_classes={9},
    )
    scorer = FakeTerminalScorer()
    root = MCTSNode(
        key=(1,),
        path=[0],
        rank=25,
        parent=None,
        action_from_parent=0,
        terminal=scorer.terminal,  # type: ignore[arg-type]
    )
    state = InterruptSearchState(
        scorer=scorer,  # type: ignore[arg-type]
        config=MCTSConfig(iterations=1, max_depth=1),
        global_discovery=shared,
        nodes={(1,): root},
        root=root,
    )
    task = InterruptSearchTask(
        class_id=7,
        search_index=1,
        parent_search_index=None,
        state=state,
        threshold=0.0,
        base_iteration_limit=1,
        max_iteration_limit=5,
        adaptive_min_iterations=1,
        adaptive_min_terminal_hits=1,
        adaptive_sink_ratio=0.0,
        adaptive_extra_iterations=1,
        productive_patience_iterations=10,
        iteration_limit=1,
    )

    discoveries = task.step()

    assert [item.class_id for item in discoveries] == [9]
    assert state.productive
    assert task.iteration_limit == 2
    assert task.last_new_class_iteration == 1
    assert task.last_global_new_iteration == 1
    assert state.stop_reason is None


def test_productive_budget_scales_with_distinct_cross_classes() -> None:
    class35 = SimpleNamespace(
        class_id=35,
        state=SimpleNamespace(local_discovered_exact_classes={35, 45, 46}),
        base_iteration_limit=200,
        max_iteration_limit=1000,
        adaptive_extra_iterations=100,
    )
    class7 = SimpleNamespace(
        class_id=7,
        state=SimpleNamespace(local_discovered_exact_classes={7, 8, 10, 11, 12, 15, 20, 28, 29}),
        base_iteration_limit=200,
        max_iteration_limit=1000,
        adaptive_extra_iterations=100,
    )

    assert _productive_iteration_limit(class35) == 400
    assert _productive_iteration_limit(class7) == 1000


def test_sink_and_productive_patience_gates() -> None:
    state = SimpleNamespace(
        productive=False,
        iterations_completed=50,
        local_discovered_exact_classes={5},
        encountered=Counter({"exact:class5": 45, "invalid:test": 5}),
    )
    task = SimpleNamespace(
        class_id=5,
        state=state,
        adaptive_min_iterations=50,
        adaptive_min_terminal_hits=20,
        adaptive_sink_ratio=0.9,
        productive_patience_iterations=400,
        last_new_class_iteration=200,
        last_global_new_iteration=200,
        base_iteration_limit=200,
    )

    assert _should_stop_as_terminal_sink(task)
    state.productive = True
    assert not _should_stop_as_terminal_sink(task)
    state.iterations_completed = 599
    assert not _should_stop_productive_for_no_novelty(task)
    state.iterations_completed = 600
    assert _should_stop_productive_for_no_novelty(task)


def test_productive_patience_ignores_local_only_novelty() -> None:
    state = SimpleNamespace(productive=True, iterations_completed=399)
    task = SimpleNamespace(
        state=state,
        productive_patience_iterations=200,
        base_iteration_limit=200,
        last_new_class_iteration=399,
        last_global_new_iteration=200,
    )

    assert not _should_stop_productive_for_no_novelty(task)
    state.iterations_completed = 400
    assert _should_stop_productive_for_no_novelty(task)

    task.last_global_new_iteration = None
    state.iterations_completed = 200
    assert _should_stop_productive_for_no_novelty(task)


def test_productive_patience_renews_for_actionable_compatibility_frontier() -> None:
    class ToggleCompatibilityBank:
        enabled = True

        def active_signature(self, key, class_ids):
            if not self.enabled or key != (1, 1, 1):
                return ()
            return tuple(class_id for class_id in class_ids if int(class_id) == 9)

    bank = ToggleCompatibilityBank()
    frontier_node = MCTSNode(
        key=(1, 1, 0),
        path=[0, 1],
        rank=24,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
        unexpanded_actions=[2],
    )
    state = SimpleNamespace(
        productive=True,
        iterations_completed=200,
        scorer=SimpleNamespace(target_classes={1, 9, 10}),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1, 10},
            discovery_epoch=0,
        ),
        compatibility_bank=bank,
        nodes={frontier_node.key: frontier_node},
    )
    task = SimpleNamespace(
        state=state,
        productive_patience_iterations=200,
        base_iteration_limit=200,
        last_global_new_iteration=None,
        iteration_limit=600,
        compatibility_frontier_extension_iterations=50,
        compatibility_frontier_extension_deadline=0,
        compatibility_frontier_extensions=0,
        last_compatibility_frontier_class_ids=(),
    )

    assert _compatibility_frontier_class_ids(task) == (9,)
    assert not _should_stop_productive_for_no_novelty(task)
    assert task.compatibility_frontier_extension_deadline == 250
    assert task.compatibility_frontier_extensions == 1
    assert task.last_compatibility_frontier_class_ids == (9,)

    bank.enabled = False
    state.iterations_completed = 249
    assert not _should_stop_productive_for_no_novelty(task)
    state.iterations_completed = 250
    assert _should_stop_productive_for_no_novelty(task)


def test_rank22_frontier_renews_only_while_edges_progress() -> None:
    class CompatibilityBank:
        def active_signature(self, _key, class_ids):
            return tuple(class_id for class_id in class_ids if int(class_id) == 9)

    rank22_node = MCTSNode(
        key=(1, 1, 0, 0),
        path=[0, 1],
        rank=22,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
        unexpanded_actions=[2],
    )
    state = SimpleNamespace(
        productive=True,
        iterations_completed=200,
        scorer=SimpleNamespace(target_classes={1, 9}),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovery_epoch=1,
        ),
        compatibility_bank=CompatibilityBank(),
        nodes={rank22_node.key: rank22_node},
    )
    task = SimpleNamespace(
        state=state,
        productive_patience_iterations=200,
        base_iteration_limit=200,
        last_global_new_iteration=None,
        iteration_limit=600,
        compatibility_frontier_extension_iterations=50,
        compatibility_frontier_min_rank=22,
        compatibility_frontier_extension_deadline=0,
        compatibility_frontier_extensions=0,
    )

    assert not _should_stop_productive_for_no_novelty(task)
    assert task.compatibility_frontier_extension_deadline == 250
    assert task.last_compatibility_frontier_max_rank == 22
    assert task.last_compatibility_frontier_new_edge_count == 1

    rank23_node = MCTSNode(
        key=(1, 0, 1, 0),
        path=[0, 2],
        rank=23,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
        unexpanded_actions=[3],
    )
    state.nodes[rank23_node.key] = rank23_node
    state.iterations_completed = 250
    assert not _should_stop_productive_for_no_novelty(task)
    assert task.compatibility_frontier_extension_deadline == 300
    assert task.compatibility_frontier_extensions == 2
    assert task.last_compatibility_frontier_max_rank == 23
    assert task.last_compatibility_frontier_new_edge_count == 1

    state.iterations_completed = 300
    assert _should_stop_productive_for_no_novelty(task)
    assert task.compatibility_frontier_observations == 3
    assert task.last_compatibility_frontier_new_edge_count == 0


def test_rank20_rank21_shadow_diagnostics_never_extend_budget() -> None:
    class CompatibilityBank:
        def active_signature(self, key, class_ids):
            compatible_by_key = {
                (1, 0, 1, 0): {9},
                (0, 1, 0, 1): {11},
            }
            compatible = compatible_by_key.get(key, set())
            return tuple(
                class_id
                for class_id in class_ids
                if int(class_id) in compatible
            )

    rank20_node = MCTSNode(
        key=(1, 0, 0, 0),
        path=[0],
        rank=20,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
        unexpanded_actions=[2],
    )
    rank21_node = MCTSNode(
        key=(0, 1, 0, 0),
        path=[1],
        rank=21,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
        unexpanded_actions=[3],
    )
    state = SimpleNamespace(
        productive=True,
        iterations_completed=200,
        scorer=SimpleNamespace(target_classes={1, 9, 11}),
        global_discovery=SimpleNamespace(
            discovered_exact_classes={1},
            discovery_epoch=1,
        ),
        compatibility_bank=CompatibilityBank(),
        nodes={
            rank20_node.key: rank20_node,
            rank21_node.key: rank21_node,
        },
    )
    task = SimpleNamespace(
        state=state,
        productive_patience_iterations=200,
        base_iteration_limit=200,
        last_global_new_iteration=None,
        iteration_limit=600,
        compatibility_frontier_extension_iterations=50,
        compatibility_frontier_min_rank=22,
        compatibility_frontier_shadow_ranks=(20, 21),
        compatibility_frontier_extension_deadline=0,
        compatibility_frontier_extensions=0,
    )

    assert _should_stop_productive_for_no_novelty(task)
    assert task.compatibility_frontier_extensions == 0
    assert task.compatibility_frontier_extension_deadline == 0
    assert task.last_compatibility_frontier_class_ids == ()
    assert task.last_compatibility_frontier_edge_count == 0

    details_by_rank = {
        detail["shadow_rank"]: detail
        for detail in task.compatibility_frontier_shadow_observation_details
    }
    assert details_by_rank[20]["retained_class_ids"] == [9]
    assert details_by_rank[20]["retained_edge_count"] == 1
    assert details_by_rank[21]["retained_class_ids"] == [11]
    assert details_by_rank[21]["retained_edge_count"] == 1
    assert set(task.compatibility_frontier_shadow_outcome_counts) == {20, 21}


def test_frontier_diagnostics_classify_every_candidate_outcome() -> None:
    class CompatibilityBank:
        def active_signature(self, key, class_ids):
            if key[2]:
                compatible = {9}
            elif key[3]:
                compatible = {1, 10}
            else:
                compatible = set()
            return tuple(
                class_id
                for class_id in class_ids
                if int(class_id) in compatible
            )

    actionable = MCTSNode(
        key=(1, 1, 0, 0, 0, 0),
        path=[0, 1],
        rank=22,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
        unexpanded_actions=[2, 3, 4],
    )
    actionable.children[5] = MCTSNode(
        key=(1, 1, 0, 0, 0, 1),
        path=[0, 1, 5],
        rank=23,
        parent=actionable,
        action_from_parent=5,
    )
    low_rank = MCTSNode(
        key=(1, 0, 0, 0, 0, 0),
        path=[0],
        rank=21,
        parent=None,
        action_from_parent=None,
    )
    terminal = MCTSNode(
        key=(0, 1, 0, 0, 0, 0),
        path=[1],
        rank=22,
        parent=None,
        action_from_parent=None,
        terminal=SimpleNamespace(),
    )
    facet_rank = MCTSNode(
        key=(0, 0, 1, 0, 0, 0),
        path=[2],
        rank=25,
        parent=None,
        action_from_parent=None,
    )
    exhausted = MCTSNode(
        key=(1, 1, 1, 1, 1, 1),
        path=[0, 1, 2, 3, 4, 5],
        rank=22,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
    )
    nodes = {
        node.key: node
        for node in (actionable, low_rank, terminal, facet_rank, exhausted)
    }
    task = SimpleNamespace(
        state=SimpleNamespace(
            scorer=SimpleNamespace(target_classes={1, 9, 10}),
            global_discovery=SimpleNamespace(discovered_exact_classes={1, 10}),
            compatibility_bank=CompatibilityBank(),
            nodes=nodes,
        ),
        compatibility_frontier_min_rank=22,
    )

    snapshot = _compatibility_frontier_snapshot(task)
    outcomes = dict(snapshot.outcome_counts)

    assert snapshot.class_ids == (9,)
    assert snapshot.scanned_node_count == 5
    assert snapshot.candidate_action_count == 3
    assert snapshot.edge_count == 1
    assert outcomes["node_rank_below_min"] == 1
    assert outcomes["node_terminal"] == 1
    assert outcomes["node_rank_at_or_above_facet"] == 1
    assert outcomes["node_no_unexpanded_actions"] == 1
    assert outcomes["node_actionable"] == 1
    assert outcomes["action_already_expanded"] == 1
    assert outcomes["action_retained_missing_compatibility"] == 1
    assert outcomes["action_only_discovered_compatibility"] == 1
    assert outcomes["action_no_target_compatibility"] == 1
    assert dict(snapshot.missing_class_compatible_action_counts) == {9: 1}
    action_outcomes = sum(
        count
        for outcome, count in outcomes.items()
        if outcome.startswith("action_") and outcome != "action_already_expanded"
    )
    assert action_outcomes == snapshot.candidate_action_count


def test_productive_batch_schedule_broadens_search() -> None:
    config = MCTSConfig(
        widening_score_batch=4,
        widening_score_batch_max=16,
        widening_score_batch_scale=1.0,
        widening_score_batch_beta=0.5,
        rollout_score_batch=8,
        productive_rollout_score_batch=16,
        broad_rollout_score_batch=24,
        broad_rollout_interval=8,
    )
    node = MCTSNode(
        key=(0,),
        path=[],
        rank=0,
        parent=None,
        action_from_parent=None,
        visits=25,
    )
    state = SimpleNamespace(config=config, productive=False)

    assert _widening_score_batch(node, config, adaptive=False) == 4
    assert _widening_score_batch(node, config, adaptive=True) == 9
    assert _rollout_score_batch_for_rollout(state, 8) == 8
    state.productive = True
    assert _rollout_score_batch_for_rollout(state, 7) == 16
    assert _rollout_score_batch_for_rollout(state, 8) == 24


def test_global_discovery_reward_overrides_stale_terminal_action_score() -> None:
    shared = InterruptGlobalDiscoveryState(
        initial_rare_target_classes={9},
        remaining_rare_target_classes=set(),
        discovered_exact_classes={9},
        discovered_label_counts=Counter({"exact:class9": 1}),
    )
    scorer = FakeTerminalScorer()
    root = MCTSNode(
        key=(0,),
        path=[],
        rank=24,
        parent=None,
        action_from_parent=None,
        actions_initialized=True,
        unexpanded_actions=[0],
        action_priors={0: 1.0},
        action_scores={0: scorer.score_action((0,), 0)},
    )
    state = InterruptSearchState(
        scorer=scorer,  # type: ignore[arg-type]
        config=MCTSConfig(iterations=1, max_depth=1),
        global_discovery=shared,
        nodes={(0,): root},
        root=root,
    )

    discoveries = _run_one_iteration(state, iteration_index=1)

    assert discoveries == []
    assert state.exact_discoveries[0].score == 10.0
    assert root.survival_sum == 10.0
    assert root.action_scores[0].score == 10.0
    assert shared.discovered_label_counts["exact:class9"] == 2


def test_parallel_terminal_hits_award_global_novelty_once() -> None:
    barrier = threading.Barrier(2)
    shared = InterruptGlobalDiscoveryState(
        initial_rare_target_classes={9},
        remaining_rare_target_classes={9},
    )

    class ConcurrentTerminalScorer(FakeTerminalScorer):
        def __init__(self) -> None:
            super().__init__()
            self.config.terminal_scoring_mode = "dynamic"

        def terminal_score(self, *args, **kwargs) -> float:
            del args, kwargs
            barrier.wait(timeout=5.0)
            return 100.0

    def terminal_state() -> InterruptSearchState:
        scorer = ConcurrentTerminalScorer()
        root = MCTSNode(
            key=(1,),
            path=[0],
            rank=25,
            parent=None,
            action_from_parent=0,
            terminal=scorer.terminal,  # type: ignore[arg-type]
        )
        return InterruptSearchState(
            scorer=scorer,  # type: ignore[arg-type]
            config=MCTSConfig(iterations=1, max_depth=1),
            global_discovery=shared,
            nodes={(1,): root},
            root=root,
        )

    states = [terminal_state(), terminal_state()]
    with ThreadPoolExecutor(max_workers=2) as executor:
        discoveries = list(
            executor.map(
                lambda state: _run_one_iteration(state, iteration_index=1),
                states,
            )
        )

    rewards = sorted(
        float(state.root.survival_sum)
        for state in states
        if state.root is not None
    )
    assert rewards == [10.0, 100.0]
    assert sum(len(items) for items in discoveries) == 1
    assert shared.discovered_label_counts_snapshot()["exact:class9"] == 2


class EpochSensitiveScorer:
    def __init__(self) -> None:
        self.blocks = [(0,)]
        self.rare_target_classes = {9}

    def score_action(
        self,
        key: tuple[int, ...],
        action: int,
        *,
        terminal_score_fn=None,
    ) -> ExpansionScore:
        del terminal_score_fn
        return ExpansionScore(
            action=int(action),
            key=(1,),
            score=100.0 if 9 in self.rare_target_classes else 10.0,
            phase="B",
            old_rank=23,
            new_rank=24,
            rank_gain=1,
            flat_capacity=0,
            supportability=None,
            terminal=None,
        )


def test_discovery_epoch_rescores_rank24_actions_and_ages_values() -> None:
    shared = InterruptGlobalDiscoveryState(
        initial_rare_target_classes={9},
        remaining_rare_target_classes={9},
    )
    scorer = EpochSensitiveScorer()
    root = MCTSNode(
        key=(0,),
        path=[],
        rank=23,
        parent=None,
        action_from_parent=None,
        visits=8,
        value_visits=8.0,
        value_sum=800.0,
        survival_sum=800.0,
        actions_initialized=True,
        unexpanded_actions=[0],
        action_priors={0: 1.0},
        action_scores={0: scorer.score_action((0,), 0)},
    )
    state = InterruptSearchState(
        scorer=scorer,  # type: ignore[arg-type]
        config=MCTSConfig(discovery_epoch_value_decay=0.25),
        global_discovery=shared,
        nodes={(0,): root},
        root=root,
        synchronized_discovery_epoch=0,
    )

    assert shared.mark_discovered(9)
    _synchronize_discovery_epoch(state, terminal_score_fn=lambda _key, _terminal: 10.0)

    assert state.synchronized_discovery_epoch == 1
    assert scorer.rare_target_classes == set()
    assert root.action_scores[0].score == 10.0
    assert root.value_visits == 2.0
    assert root.value_sum == 200.0
    assert root.survival_sum == 200.0

    _synchronize_discovery_epoch(state, terminal_score_fn=lambda _key, _terminal: 10.0)
    assert root.value_visits == 2.0


def test_all_interrupt_tasks_observe_the_same_global_discovery_set() -> None:
    shared = InterruptGlobalDiscoveryState(
        initial_rare_target_classes={9},
        remaining_rare_target_classes={9},
    )

    def terminal_state() -> InterruptSearchState:
        scorer = FakeTerminalScorer()
        root = MCTSNode(
            key=(1,),
            path=[0],
            rank=25,
            parent=None,
            action_from_parent=0,
            terminal=scorer.terminal,  # type: ignore[arg-type]
        )
        return InterruptSearchState(
            scorer=scorer,  # type: ignore[arg-type]
            config=MCTSConfig(iterations=1, max_depth=1),
            global_discovery=shared,
            nodes={(1,): root},
            root=root,
        )

    first_state = terminal_state()
    second_state = terminal_state()
    first_discoveries = _run_one_iteration(first_state, iteration_index=1)
    second_discoveries = _run_one_iteration(second_state, iteration_index=1)

    assert [item.class_id for item in first_discoveries] == [9]
    assert second_discoveries == []
    assert first_state.exact_discoveries[0].score == 100.0
    assert second_state.exact_discoveries[0].score == 10.0
    assert first_state.scorer.terminal_score_calls == 1
    assert second_state.scorer.terminal_score_calls == 0
    assert first_state.global_discovery is second_state.global_discovery


def test_rank24_entrance_cache_reclassifies_without_revalidating() -> None:
    scorer = object.__new__(ExpansionScorer)
    scorer.rank24_entrance_cache = {}
    scorer.rare_target_classes = {9, 44}
    scorer.config = SimpleNamespace(
        cache_heavy_max_entries=0,
        cache_light_max_entries=0,
    )
    labels = {
        0: SimpleNamespace(label="exact:class9", is_exact=True),
        1: SimpleNamespace(label="exact:class44", is_exact=True),
        2: SimpleNamespace(label="invalid:test", is_exact=False),
    }
    validation_calls = Counter()
    scorer.affine_rank = lambda _key: 25  # type: ignore[method-assign]

    def terminal_label(key: tuple[int, ...]) -> object:
        action = key.index(1)
        validation_calls[action] += 1
        return labels[action]

    scorer.terminal_label = terminal_label  # type: ignore[method-assign]

    first = scorer.rank24_entrance_metrics((0, 0, 0))
    scorer.rare_target_classes = {9}
    second = scorer.rank24_entrance_metrics((0, 0, 0))

    assert first == {"rare": 2, "class44": 0, "invalid": 1, "other_valid": 0}
    assert second == {"rare": 1, "class44": 1, "invalid": 1, "other_valid": 0}
    assert sum(validation_calls.values()) == 3


def test_dynamic_terminal_reward_updates_at_frequent_threshold() -> None:
    shared = InterruptGlobalDiscoveryState(
        initial_rare_target_classes={9},
        remaining_rare_target_classes=set(),
        discovered_exact_classes={9},
        discovered_label_counts=Counter({"exact:class9": 15}),
    )

    def terminal_state() -> InterruptSearchState:
        scorer = FakeTerminalScorer()
        scorer.config.terminal_scoring_mode = "dynamic"
        root = MCTSNode(
            key=(1,),
            path=[0],
            rank=25,
            parent=None,
            action_from_parent=0,
            terminal=scorer.terminal,  # type: ignore[arg-type]
        )
        return InterruptSearchState(
            scorer=scorer,  # type: ignore[arg-type]
            config=MCTSConfig(iterations=1, max_depth=1),
            global_discovery=shared,
            nodes={(1,): root},
            root=root,
        )

    threshold_state = terminal_state()
    _run_one_iteration(threshold_state, iteration_index=1)
    frequent_state = terminal_state()
    _run_one_iteration(frequent_state, iteration_index=1)

    assert threshold_state.root is not None
    assert frequent_state.root is not None
    assert threshold_state.root.survival_sum == 10.0
    assert frequent_state.root.survival_sum == -5.0
    assert shared.discovered_label_counts["exact:class9"] == 17


class CountingActionScorer:
    def __init__(self, block_count: int = 12) -> None:
        self.blocks = [(index,) for index in range(block_count)]
        self.calls: list[int] = []

    def score_action(self, key: tuple[int, ...], action: int, *, terminal_score_fn=None) -> ExpansionScore:
        del terminal_score_fn
        self.calls.append(int(action))
        candidate = tuple(1 if index == action else value for index, value in enumerate(key))
        return ExpansionScore(
            action=int(action),
            key=candidate,
            score=float(action),
            phase="A",
            old_rank=0,
            new_rank=1,
            rank_gain=1,
            flat_capacity=0,
            supportability=None,
            terminal=None,
        )


def test_progressive_widening_initializes_actions_without_eager_scoring() -> None:
    scorer = CountingActionScorer()
    node = MCTSNode(
        key=tuple(0 for _ in scorer.blocks),
        path=[],
        rank=0,
        parent=None,
        action_from_parent=None,
    )
    config = MCTSConfig(widening_score_batch=3)

    _prepare_actions(node, scorer, config)  # type: ignore[arg-type]

    assert scorer.calls == []
    assert len(node.unexpanded_actions) == len(scorer.blocks)
    action = _select_progressive_action(
        node,
        random.Random(7),
        scorer=scorer,  # type: ignore[arg-type]
        cfg=config,
        compatibility_bank=None,
        discovered_exact_classes=set(),
        seen_signatures=Counter(),
    )

    assert len(scorer.calls) == 3
    assert action in scorer.calls
    assert len(node.action_scores) == 3


def test_progressive_bucket_quota_reaches_all_six_buckets_early() -> None:
    scorer = CountingActionScorer()
    node = MCTSNode(
        key=tuple(0 for _ in scorer.blocks),
        path=[],
        rank=0,
        parent=None,
        action_from_parent=None,
    )
    config = MCTSConfig(widening_score_batch=1, progressive_bucket_quota=1)
    _prepare_actions(node, scorer, config)  # type: ignore[arg-type]

    node.visits = 2
    assert _progressive_child_limit(node, config) == 3
    node.visits = 5
    assert _progressive_child_limit(node, config) == 6

    rng = random.Random(11)
    for _ in range(6):
        action = _select_progressive_action(
            node,
            rng,
            scorer=scorer,  # type: ignore[arg-type]
            cfg=config,
            compatibility_bank=None,
            discovered_exact_classes=set(),
            seen_signatures=Counter(),
        )
        node.unexpanded_actions.remove(action)

    assert node.bucket_expansion_counts == [1, 1, 1, 1, 1, 1]


def test_action_structure_cache_reuses_geometry_not_reward() -> None:
    scorer = object.__new__(ExpansionScorer)
    scorer.action_structure_cache = {}
    scorer.config = SimpleNamespace(
        cache_heavy_max_entries=0,
        cache_light_max_entries=0,
        rank_gain_weight_scale=1.0,
        flat_penalty_weight_scale=1.0,
        nonpositive_rank_gain_penalty=8.0,
        supportability_start_rank=99,
        phase_b_start_rank=22,
        rank24_entrance_weight=1.0,
    )
    calls: Counter[str] = Counter()

    def affine_rank(key: tuple[int, ...]) -> int:
        calls["rank"] += 1
        return sum(int(value) for value in key)

    def flat_capacity(_key: tuple[int, ...]) -> int:
        calls["flat"] += 1
        return 2

    scorer.affine_rank = affine_rank  # type: ignore[method-assign]
    scorer.flat_capacity = flat_capacity  # type: ignore[method-assign]

    first = scorer.score_action((0, 0), 1)
    second = scorer.score_action((0, 0), 1)

    assert first.score == second.score
    assert calls == Counter({"rank": 2, "flat": 1})
    assert len(scorer.action_structure_cache) == 1


def test_terminal_reward_is_recomputed_from_cached_structure() -> None:
    scorer = object.__new__(ExpansionScorer)
    terminal = SimpleNamespace(label="exact:class9", is_exact=True)
    structure = ExpansionStructure(
        action=0,
        key=(1,),
        phase="C",
        old_rank=24,
        new_rank=25,
        rank_gain=1,
        flat_capacity=0,
        supportability=None,
        terminal=terminal,
        base_score=None,
    )
    scorer.action_structure = lambda _key, _action: structure  # type: ignore[method-assign]

    first = scorer.score_action((0,), 0, terminal_score_fn=lambda _key, _terminal: 100.0)
    second = scorer.score_action((0,), 0, terminal_score_fn=lambda _key, _terminal: 10.0)

    assert first.score == 100.0
    assert second.score == 10.0


def test_terminal_validation_cache_is_shared_across_scorers() -> None:
    shared = SharedTerminalValidationCache()
    validation_calls: Counter[str] = Counter()
    label = SimpleNamespace(label="exact:class9", is_exact=True)

    class Validator:
        def validate_mask(self, _mask: object) -> object:
            validation_calls["validate"] += 1
            return label

    def make_scorer() -> ExpansionScorer:
        scorer = object.__new__(ExpansionScorer)
        scorer.shared_terminal_validation_cache = shared
        scorer.terminal_cache = {}
        scorer.config = SimpleNamespace(cache_heavy_max_entries=0, cache_light_max_entries=0)
        scorer.validator = Validator()
        scorer.support_key = lambda _key: 12345  # type: ignore[method-assign]
        scorer.vertex_mask = lambda _key: object()  # type: ignore[method-assign]
        return scorer

    first = make_scorer().terminal_label((1, 0))
    second = make_scorer().terminal_label((0, 1))

    assert first is second
    assert validation_calls == Counter({"validate": 1})
    assert shared.hits == 1
    assert shared.misses == 1


def test_class1_and_class6_share_partition_but_not_basin() -> None:
    config = InterruptSearchConfig(rep_index=1, pattern_index=0)
    singleton_partition = _canonical_pattern_signature(
        tuple((vertex,) for vertex in range(64))
    )

    class1 = _search_task_identity(config, 1, singleton_partition)
    class6 = _search_task_identity(config, 6, singleton_partition)

    assert class1.partition_key == class6.partition_key
    assert class1.basin_key != class6.basin_key
    assert class1 != class6


def test_structure_cache_shares_geometry_without_sharing_reward_counts() -> None:
    shared = SharedScorerStructureCache()
    config = ScorerConfig()
    first = ExpansionScorer(
        blocks=[(0,), (1,)],
        config=config,
        structure_cache=shared,
    )
    second = ExpansionScorer(
        blocks=[(0,), (1,)],
        config=config,
        structure_cache=shared,
    )

    first.rank_cache[(0, 0)] = 0
    first.discovered_label_counts["exact:class1"] = 3

    assert second.rank_cache is first.rank_cache
    assert second._structure_lock is first._structure_lock
    assert second.rank_cache[(0, 0)] == 0
    assert second.discovered_label_counts is not first.discovered_label_counts
    assert second.discovered_label_counts["exact:class1"] == 0


def test_shared_structure_cache_eviction_is_thread_safe() -> None:
    shared = SharedScorerStructureCache()
    config = ScorerConfig(
        cache_heavy_max_entries=4,
        cache_light_max_entries=4,
    )
    scorers = [
        ExpansionScorer(
            blocks=[(0,), (1,)],
            config=config,
            structure_cache=shared,
        )
        for _ in range(4)
    ]

    def churn(worker_index: int) -> None:
        scorer = scorers[worker_index]
        for value in range(2000):
            key = (worker_index, value)
            scorer._cache_put(
                scorer.terminal_cache,
                key,  # type: ignore[arg-type]
                value,
                heavy=False,
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(churn, index) for index in range(4)]
        for future in futures:
            future.result()

    assert len(scorers[0].terminal_cache) <= 4


def test_affine_hull_flat_capacity_matches_child_rank() -> None:
    blocks = [tuple(range(index * 4, (index + 1) * 4)) for index in range(16)]
    child_rank = ExpansionScorer(
        blocks=blocks,
        config=ScorerConfig(flat_capacity_method="child_rank"),
    )
    affine_hull = ExpansionScorer(
        blocks=blocks,
        config=ScorerConfig(flat_capacity_method="affine_hull"),
    )
    rng = random.Random(7322)

    for depth in range(2, 15):
        chosen = set(rng.sample(range(len(blocks)), depth))
        key = tuple(1 if index in chosen else 0 for index in range(len(blocks)))
        assert affine_hull.flat_capacity(key) == child_rank.flat_capacity(key)


def test_facet_validator_uses_support_index_for_known_facet() -> None:
    row = parse_example_rows()[10][1]
    support = support_mask_from_row(row)
    validator = FacetValidator()

    label = validator.validate_mask(support)

    assert label.label == "exact:class10"
    assert label.match is not None
    assert label.match["match_source"] == "support_index"


def test_rank24_entrance_skips_validation_above_facet_rank() -> None:
    scorer = object.__new__(ExpansionScorer)
    scorer.rank24_entrance_cache = {}
    scorer.rare_target_classes = set()
    scorer.config = SimpleNamespace(cache_heavy_max_entries=0, cache_light_max_entries=0)
    scorer.affine_rank = lambda key: 26 if key[0] else 25  # type: ignore[method-assign]
    validations: list[tuple[int, ...]] = []

    def terminal_label(key: tuple[int, ...]) -> object:
        validations.append(key)
        return SimpleNamespace(label="invalid:test", is_exact=False)

    scorer.terminal_label = terminal_label  # type: ignore[method-assign]

    metrics = scorer.rank24_entrance_metrics((0, 0))

    assert metrics["invalid"] == 2
    assert validations == [(0, 1)]


def test_compatibility_bank_compacts_state_cache() -> None:
    from mcts.search import ClassCompatibilityBank

    blocks = [(index,) for index in range(64)]
    bank = ClassCompatibilityBank(
        blocks=blocks,
        examples_path=InterruptSearchConfig().examples_path,
    )
    key = tuple(1 if index < 3 else 0 for index in range(64))

    first = bank.total_compat_count(key, [1, 2, 3])
    signature = bank.active_signature(key, [1, 2, 3])
    second = sum(bank.compat_count(key, class_id) for class_id in [1, 2, 3])

    assert first == second
    assert signature == tuple(class_id for class_id in [1, 2, 3] if bank.compat_count(key, class_id) > 0)
    assert len(bank._compatibility_vectors) == 1


def test_interrupt_state_release_tree_breaks_references() -> None:
    root = MCTSNode(key=(0,), path=[], rank=0, parent=None, action_from_parent=None)
    child = MCTSNode(key=(1,), path=[0], rank=1, parent=root, action_from_parent=0)
    root.children[0] = child
    bank = SimpleNamespace(cleared=False)
    bank.clear_cache = lambda: setattr(bank, "cleared", True)
    state = InterruptSearchState(
        scorer=SimpleNamespace(),  # type: ignore[arg-type]
        config=MCTSConfig(rank23_tail_active_service=True),
        compatibility_bank=bank,  # type: ignore[arg-type]
        nodes={(0,): root, (1,): child},
        root=root,
    )
    state.rank23_tail_progress[7] = Rank23TailProgress(
        rank24_actions=(1, 2),
        pair_candidate_count=1,
    )
    _enqueue_rank23_tail_work(state, 7, (0,), [3])

    state.release_tree()

    assert state.root is None
    assert state.nodes == {}
    assert root.children == {}
    assert child.parent is None
    assert bank.cleared
    assert not state.rank23_tail_pending
    assert not state.rank23_tail_pending_set
    assert not state.rank23_tail_work


def test_rank23_active_service_queue_is_unique_and_round_robin() -> None:
    state = InterruptSearchState(
        scorer=SimpleNamespace(),  # type: ignore[arg-type]
        config=MCTSConfig(rank23_tail_active_service=True),
    )
    for prefix_word in (11, 22):
        state.rank23_tail_progress[prefix_word] = Rank23TailProgress(
            rank24_actions=(1, 2),
            pair_candidate_count=1,
        )

    _enqueue_rank23_tail_work(state, 11, (1,), [1])
    _enqueue_rank23_tail_work(state, 11, (1,), [1])
    _enqueue_rank23_tail_work(state, 22, (2,), [2])

    assert list(state.rank23_tail_pending) == [11, 22]
    assert state.rank23_tail_pending_peak == 2
    first = _pop_rank23_tail_work(state)
    assert first is not None and first[0] == 11
    _enqueue_rank23_tail_work(state, 11, (1,), [1])
    second = _pop_rank23_tail_work(state)
    third = _pop_rank23_tail_work(state)
    assert second is not None and second[0] == 22
    assert third is not None and third[0] == 11

    _finish_rank23_tail_work(state, 11)
    assert _pop_rank23_tail_work(state) is None


def test_shared_structure_cache_evicts_cold_partitions() -> None:
    shared = SharedScorerStructureCache(max_partitions=2)
    config = ScorerConfig()

    first = shared.for_partition([(0,)], config)
    shared.for_partition([(1,)], config)
    shared.for_partition([(2,)], config)

    assert len(shared.partitions) == 2
    assert shared.evictions == 1
    assert all(value is not first for value in shared.partitions.values())
    assert shared.for_partition([(0,)], config) is first
