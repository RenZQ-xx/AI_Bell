from __future__ import annotations

import json
from pathlib import Path

from mcts.decision_trace import set_trace_sink
from mcts.interrupt_search import (
    InterruptGlobalDiscoveryState,
    InterruptSearchConfig,
    _default_task_factory,
)


PRE_CLASS9_DISCOVERED = {
    1, 7, 8, 9, 12, 15, 18, 19, 20, 22, 23, 25, 26, 28, 29, 30,
    31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
}
EXPECTED_DISCOVERIES = [(44, 1), (43, 2), (46, 5), (29, 7), (15, 53), (42, 60), (11, 66)]
EXPECTED_CLASS11_PATH = [9, 17, 12, 22, 23, 16, 5, 2, 14, 35, 29, 36, 33, 38, 26, 27, 28]


def main() -> None:
    output_dir = Path("src/mcts/runs")
    trace_path = output_dir / "class9_iteration66_trace.jsonl"
    report_path = output_dir / "class9_iteration66_trace.md"

    config = InterruptSearchConfig(
        initial_class_id=1,
        iterations=300,
        seed=20260502,
        rare_target_classes=tuple(range(1, 47)),
    )
    global_discovery = InterruptGlobalDiscoveryState.from_config(config)
    for class_id in sorted(PRE_CLASS9_DISCOVERED):
        global_discovery.mark_discovered(class_id)

    task = _default_task_factory(config, global_discovery)(9, 31, 30)
    records: list[dict[str, object]] = []
    for iteration in range(1, 67):
        set_trace_sink(records.append if iteration == 66 else None)
        task.step()
    set_trace_sink(None)

    actual = [(item.class_id, item.iteration) for item in task.state.exact_discoveries]
    class11 = next((item for item in task.state.exact_discoveries if item.class_id == 11), None)
    path_matches = class11 is not None and class11.path == EXPECTED_CLASS11_PATH
    replay_matches = actual[: len(EXPECTED_DISCOVERIES)] == EXPECTED_DISCOVERIES and path_matches

    with trace_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# Class-9 MCTS iteration 66 trace",
        "",
        f"- Replay matches original through iteration 66: **{replay_matches}**",
        f"- Seed: `{task.state.config.seed}`",
        f"- Exact discoveries: `{actual}`",
        f"- Class-11 path matches: **{path_matches}**",
        "",
        "## Iteration 66 decision sequence",
        "",
    ]
    for record in records:
        event = record["event"]
        if event == "node_visit":
            lines.append(
                f"- Visit rank {record['rank']} node, path `{record['path']}`, "
                f"visits={record['visits']}, children={record['child_count']}."
            )
        elif event == "ucb_choice":
            lines.append(f"- UCB selects block `{record['chosen_action']}` from `{len(record['candidates'])}` children.")
        elif event == "progressive_choice":
            lines.append(
                f"- Progressive widening bucket `{record['bucket']}` "
                f"(`{record['decision_kind']}`) selects block `{record['chosen_action']}`."
            )
        elif event == "expand_action":
            lines.append(f"- Expand block `{record['action']}` after path `{record['parent_path']}`.")
        elif event == "rollout_choice":
            lines.append(
                f"- Rollout at rank {record['current_rank']} selects block `{record['chosen_action']}` "
                f"from top pool `{record['pool']}` with weights `{record['weights']}`."
            )
        elif event == "backpropagation":
            lines.append(f"- Backpropagate components `{record['value']}`.")
    lines.extend([
        "",
        "The JSONL file contains complete candidate scores, priors, UCB components,",
        "rollout pools, softmax weights, and before/after backpropagation values.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "replay_matches": replay_matches,
        "actual_discoveries": actual,
        "class11_path_matches": path_matches,
        "trace_records": len(records),
        "trace_path": str(trace_path),
        "report_path": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
