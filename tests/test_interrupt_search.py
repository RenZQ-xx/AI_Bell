from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

from mcts.interrupt_search import InterruptSearchConfig, run_interruptible_search, _should_stop_for_exact_frequency
from mcts.search import ExactClassDiscovery


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
    state: object = field(default_factory=lambda: SimpleNamespace(config=SimpleNamespace(seed=0)))

    def __post_init__(self) -> None:
        self._step_count = 0
        self._finished = False

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
    fake_state = SimpleNamespace(scorer=SimpleNamespace(discovered_label_counts=Counter({"exact:class5": 5})))
    assert _should_stop_for_exact_frequency(fake_state, threshold=4.0)
    assert not _should_stop_for_exact_frequency(fake_state, threshold=5.0)