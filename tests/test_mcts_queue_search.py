from __future__ import annotations

from mcts.queue_search import QueueDiscoveryEvent, QueueSearchConfig, run_queue_supervisor
from mcts.search import ExactClassDiscovery, MCTSResult


def test_queue_supervisor_enqueues_new_exact_classes_once() -> None:
    config = QueueSearchConfig(
        initial_class_id=1,
        target_classes=(2, 3, 4),
        rare_target_classes=(),
        iterations=1,
        max_depth=1,
        seed=7,
    )

    responses: dict[int, list[ExactClassDiscovery]] = {
        1: [
            ExactClassDiscovery(
                class_id=2,
                label="exact:class2",
                iteration=3,
                depth=5,
                score=100.0,
                rank=25,
                path=[1, 2, 3],
                chosen_blocks=[1, 3, 5],
            ),
            ExactClassDiscovery(
                class_id=9,
                label="exact:class9",
                iteration=4,
                depth=6,
                score=100.0,
                rank=25,
                path=[2, 4, 6],
                chosen_blocks=[2, 4, 6],
            ),
        ],
        2: [
            ExactClassDiscovery(
                class_id=3,
                label="exact:class3",
                iteration=2,
                depth=4,
                score=100.0,
                rank=25,
                path=[7, 8],
                chosen_blocks=[7, 8],
            )
        ],
        9: [
            ExactClassDiscovery(
                class_id=4,
                label="exact:class4",
                iteration=1,
                depth=2,
                score=100.0,
                rank=25,
                path=[9],
                chosen_blocks=[9],
            )
        ],
    }

    def runner(start_class_id: int, seed: int) -> MCTSResult:
        del seed
        return MCTSResult(best=None, exact_discoveries=list(responses.get(start_class_id, [])))

    report = run_queue_supervisor(config, search_runner=runner)

    assert report.started_class_ids == [1, 2, 9]
    assert report.discovered_class_ids == [2, 3, 4]
    assert report.stop_reason == "all_target_classes_found"
    assert [event.class_id for event in report.discovery_events] == [2, 3, 4]
    assert report.discovery_events[0].summary == "第1次搜索中第3次iteration第5深度时找到 class 2"
    assert [run.start_class_id for run in report.runs] == [1, 2, 9]
    assert report.runs[0].newly_queued_class_ids == [2, 9]
    assert report.runs[1].newly_queued_class_ids == [3]
    assert report.runs[2].newly_queued_class_ids == [4]
