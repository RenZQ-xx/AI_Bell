from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from mcts.decision_trace import set_trace_sink
from mcts.interrupt_search import InterruptSearchReport, run_interruptible_search
from mcts.pair_interrupt_search import PairInterruptSearchConfig, run_pair_interrupt_search


ORIGINAL_PAIR = Path("src/mcts/runs/pair_interrupt_search_class1_i300.pair.json")
OUTPUT = Path("src/mcts/runs/replay_pair_interrupt_class9_trace.json")
TRACE = Path("src/mcts/runs/class9_iteration66_full_replay_trace.jsonl")
REPORT = Path("src/mcts/runs/class9_iteration66_full_replay_trace.md")
EXPECTED_PREFIX = [(44, 1), (43, 2), (46, 5), (29, 7), (15, 53), (42, 60), (11, 66)]
EXPECTED_PATH = [9, 17, 12, 22, 23, 16, 5, 2, 14, 35, 29, 36, 33, 38, 26, 27, 28]


def pair_runner(*_args: object, **_kwargs: object) -> dict[str, object]:
    return json.loads(ORIGINAL_PAIR.read_text(encoding="utf-8"))


def tracing_interrupt_runner(config: object, **kwargs: object) -> InterruptSearchReport:
    original_factory = kwargs["task_factory"]
    records: list[dict[str, object]] = []
    class9_task: object | None = None

    def factory(class_id: int, search_index: int, parent_search_index: int | None) -> object:
        nonlocal class9_task
        task = original_factory(class_id, search_index, parent_search_index)  # type: ignore[operator]
        if class_id == 9:
            class9_task = task
            original_step: Callable[[], object] = task.step  # type: ignore[attr-defined]

            def traced_step() -> object:
                next_iteration = int(task.state.iterations_completed) + 1  # type: ignore[attr-defined]
                set_trace_sink(records.append if next_iteration == 66 else None)
                try:
                    return original_step()
                finally:
                    set_trace_sink(None)

            task.step = traced_step  # type: ignore[attr-defined]
        return task

    kwargs["task_factory"] = factory
    report = run_interruptible_search(config, **kwargs)  # type: ignore[arg-type]

    with TRACE.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    actual: list[tuple[int, int]] = []
    actual_path: list[int] | None = None
    if class9_task is not None:
        discoveries = class9_task.state.exact_discoveries  # type: ignore[attr-defined]
        actual = [(item.class_id, item.iteration) for item in discoveries]
        class11 = next((item for item in discoveries if item.class_id == 11), None)
        actual_path = None if class11 is None else list(class11.path)
    matched = actual[: len(EXPECTED_PREFIX)] == EXPECTED_PREFIX and actual_path == EXPECTED_PATH
    REPORT.write_text(
        "# Full hybrid replay: class-9 iteration 66\n\n"
        f"- Exact replay match: **{matched}**\n"
        f"- Discovery prefix: `{actual[:len(EXPECTED_PREFIX)]}`\n"
        f"- Class-11 path: `{actual_path}`\n"
        f"- Trace records: `{len(records)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"class9_replay_matches": matched, "trace_records": len(records)}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    payload = run_pair_interrupt_search(
        PairInterruptSearchConfig(initial_class_id=1, iterations=300),
        output_path=OUTPUT,
        pair_runner=pair_runner,
        interrupt_runner=tracing_interrupt_runner,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
