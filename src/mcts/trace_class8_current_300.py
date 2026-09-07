from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from mcts.decision_trace import set_trace_sink
from mcts.interrupt_search import InterruptGlobalDiscoveryState, InterruptSearchConfig, _default_task_factory


PRE_CLASS8_DISCOVERED = {
    1, 7, 8, 12, 15, 18, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
}
OUTPUT_DIR = Path("src/mcts/runs/class8_current_300_trace")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = OUTPUT_DIR / "decisions.jsonl"
    result_path = OUTPUT_DIR / "result.json"
    report_path = OUTPUT_DIR / "README.md"

    config = InterruptSearchConfig(
        initial_class_id=1,
        iterations=300,
        seed=20260502,
        rare_target_classes=tuple(range(1, 47)),
    )
    global_state = InterruptGlobalDiscoveryState.from_config(config)
    for class_id in sorted(PRE_CLASS8_DISCOVERED):
        global_state.mark_discovered(class_id)
    task = _default_task_factory(config, global_state)(8, 29, 27)

    event_counts: Counter[str] = Counter()
    iteration_events: dict[int, list[dict[str, object]]] = defaultdict(list)
    with trace_path.open("w", encoding="utf-8") as stream:
        def sink(record: dict[str, object]) -> None:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            event_counts[str(record["event"])] += 1
            iteration_events[int(record.get("iteration", 0))].append(record)

        set_trace_sink(sink)
        try:
            for _ in range(300):
                task.step()
        finally:
            set_trace_sink(None)

    result = task.result().to_dict()
    payload = {
        "meta": {
            "class_id": 8,
            "search_index": 29,
            "parent_search_index": 27,
            "seed": task.state.config.seed,
            "iterations": 300,
            "pre_discovered_class_ids": sorted(PRE_CLASS8_DISCOVERED),
            "trace_event_counts": dict(sorted(event_counts.items())),
        },
        "result": result,
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Class-8 current-environment MCTS trace (300 iterations)", "",
        "This is the accepted current-environment rerun, not a replay of the original Windows process.", "",
        "## Files", "",
        "- `decisions.jsonl`: complete decision-level trace.",
        "- `result.json`: run configuration and final MCTS result.", "",
        "## Run", "",
        f"- Seed: `{task.state.config.seed}`",
        f"- Iterations: `{result['iterations_completed']}`",
        f"- Nodes created: `{result['nodes_created']}`",
        f"- Event counts: `{dict(sorted(event_counts.items()))}`", "",
        "## Per-iteration index", "",
        "| Iteration | Tree path before expansion | Expansion | Rollout blocks | Terminal |",
        "|---:|---|---:|---|---|",
    ]
    for iteration in range(1, 301):
        events = iteration_events[iteration]
        visits = [event for event in events if event["event"] == "node_visit"]
        expansions = [event for event in events if event["event"] == "expand_action"]
        rollouts = [event for event in events if event["event"] == "rollout_choice"]
        terminals = [event for event in events if event["event"] == "terminal_observation"]
        tree_path = visits[-1]["path"] if visits else []
        expansion = expansions[-1]["action"] if expansions else "—"
        rollout_actions = [event["chosen_action"] for event in rollouts]
        terminal = terminals[-1]["label"] if terminals else "—"
        lines.append(f"| {iteration} | `{tree_path}` | {expansion} | `{rollout_actions}` | `{terminal}` |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "trace": str(trace_path),
        "result": str(result_path),
        "report": str(report_path),
        "event_counts": dict(sorted(event_counts.items())),
        "encountered_label_counts": result["encountered_label_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
