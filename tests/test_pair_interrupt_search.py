from __future__ import annotations

import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from mcts.pair_interrupt_search import (
    PairBridgeSeed,
    PairBridgeTaskFactory,
    PairInterruptSearchConfig,
    TimedInterruptGlobalDiscoveryState,
    _InterruptProgress,
    _bridge_seeds_from_pair_payload,
    _seed_interrupt_global_state,
    run_pair_interrupt_search,
)


def _pair_payload() -> dict[str, object]:
    return {
        "meta": {"status": "complete"},
        "summary": {
            "coverage_class_ids": [1, 18],
            "class_hit_counts": {"1": 1, "18": 3},
        },
        "discovery_timeline": [
            {
                "event_index": 0,
                "class_id": 1,
                "source": "initial_seed",
                "pattern_id": None,
                "local_iteration": 0,
                "global_iteration": 0,
                "round": 0,
                "wall_seconds": 0.0,
                "selected_blocks": [],
                "support_word_hex": None,
            },
            {
                "event_index": 1,
                "class_id": 18,
                "source": "rank19_terminal_lookahead:rank23_tail",
                "pattern_id": 3,
                "local_iteration": 63,
                "global_iteration": 1119,
                "round": 63,
                "wall_seconds": 41.9,
                "selected_blocks": list(range(13)),
                "support_word_hex": "0xca534c1182c235ac",
            },
        ],
        "terminal_cache": {},
    }


def test_bridge_uses_only_observed_noninitial_pair_discoveries() -> None:
    seeds = _bridge_seeds_from_pair_payload(
        _pair_payload(),
        initial_class_id=1,
    )

    assert len(seeds) == 1
    seed = seeds[0]
    assert seed.class_id == 18
    assert seed.pattern_id == 3
    assert seed.pair_round == 63
    assert seed.selected_blocks == tuple(range(13))
    assert seed.support_word == 0xCA534C1182C235AC


def test_pair_discoveries_preseed_global_novelty_without_timed_duplicates() -> None:
    config = PairInterruptSearchConfig(iterations=1)
    seeds = _bridge_seeds_from_pair_payload(
        _pair_payload(),
        initial_class_id=1,
    )
    state = _seed_interrupt_global_state(
        config,
        _pair_payload(),
        seeds,
        hybrid_started_at=time.perf_counter(),
    )

    assert state.discovered_exact_classes == {1, 18}
    assert 1 not in state.remaining_rare_target_classes
    assert 18 not in state.remaining_rare_target_classes
    assert state.discovered_label_counts["exact:class18"] == 3
    assert state.timed_discoveries == []
    assert state.record_timeline


@dataclass
class _FakeState:
    iterations_completed: int = 0
    stop_reason: str | None = None


@dataclass
class _FakeTask:
    class_id: int
    search_index: int
    parent_search_index: int | None
    state: _FakeState = field(default_factory=_FakeState)
    started_interrupts: list[int] = field(default_factory=list)
    resumed_interrupts: list[int] = field(default_factory=list)
    delegate_steps: int = 0

    @property
    def finished(self) -> bool:
        return self.state.stop_reason is not None

    def step(self) -> list[object]:
        self.delegate_steps += 1
        self.state.iterations_completed += 1
        self.state.stop_reason = "test_complete"
        return []


def test_initial_interrupt_task_emits_bridge_before_consuming_budget() -> None:
    global_state = TimedInterruptGlobalDiscoveryState(
        hybrid_started_at=time.perf_counter()
    )
    progress = _InterruptProgress(
        started_at=time.perf_counter(),
        global_discovery=global_state,
    )
    seed = PairBridgeSeed(
        class_id=18,
        pattern_id=3,
        pair_round=63,
        pair_local_iteration=63,
        pair_global_iteration=1119,
        pair_wall_seconds=41.9,
        selected_blocks=tuple(range(13)),
        support_word=None,
        source="rank19_terminal_lookahead:rank23_tail",
    )
    delegates: list[_FakeTask] = []

    def base_factory(
        class_id: int,
        search_index: int,
        parent_search_index: int | None,
    ) -> _FakeTask:
        task = _FakeTask(class_id, search_index, parent_search_index)
        delegates.append(task)
        return task

    factory = PairBridgeTaskFactory(
        base_factory,  # type: ignore[arg-type]
        initial_class_id=1,
        seeds=(seed,),
        progress=progress,
        bridge_score=100.0,
    )
    initial = factory(1, 1, None)

    discoveries = initial.step()
    assert [item.class_id for item in discoveries] == [18]
    assert discoveries[0].iteration == 0
    assert discoveries[0].rank == 25
    assert delegates[0].delegate_steps == 0

    assert initial.step() == []
    assert delegates[0].delegate_steps == 1
    child = factory(18, 2, 1)
    assert child.pending_seeds == deque()


def test_hybrid_runner_opens_pair_discovery_as_interrupt_task() -> None:
    def fake_pair_runner(
        _config: object,
        *,
        output_path: Path,
    ) -> dict[str, object]:
        return _pair_payload()

    with tempfile.TemporaryDirectory() as directory:
        payload = run_pair_interrupt_search(
            PairInterruptSearchConfig(iterations=1),
            output_path=Path(directory) / "hybrid.json",
            pair_runner=fake_pair_runner,
        )

    assert payload["summary"]["bridge_seeded_class_ids"] == [18]
    assert payload["summary"]["class18_started_by_interrupt_phase"]
    assert {1, 18}.issubset(payload["summary"]["coverage_class_ids"])
    assert payload["meta"]["bridge_payload"] == (
        "observed_complete_terminal_supports_only"
    )
