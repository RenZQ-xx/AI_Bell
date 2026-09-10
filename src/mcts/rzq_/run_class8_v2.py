"""Run the class-8 v2 experiment with symmetry-aware tree expansion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Callable

from audit_environment import HERE, snapshot
from mcts import trace_class8_current_300 as baseline
from mcts.decision_trace import set_trace_sink
from mcts.interrupt_search import InterruptGlobalDiscoveryState, InterruptSearchConfig
from mcts.rzq_.run_class8_round1 import rzq_task_factory


DEFAULT_OUTPUT = HERE / "runs" / "class8_v2_300_trace"


def run(
    output: Path,
    *,
    iterations: int = 300,
    task_factory: Callable = rzq_task_factory,
    experiment_version: str = "class8_v2",
    based_on: str = "class8_v1",
    ucb_metadata: dict[str, object] | None = None,
    feature_description: str | None = None,
) -> dict[str, object]:
    """Execute v2 without overwriting an existing result directory."""

    if output.exists():
        raise FileExistsError(f"Output already exists; preserve or move it before rerunning: {output}")
    output.mkdir(parents=True)
    (output / "environment.json").write_text(
        json.dumps(snapshot(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    config = InterruptSearchConfig(
        initial_class_id=1,
        iterations=int(iterations),
        seed=20260502,
        rare_target_classes=tuple(range(1, 47)),
    )
    global_state = InterruptGlobalDiscoveryState.from_config(config)
    for class_id in sorted(baseline.PRE_CLASS8_DISCOVERED):
        global_state.mark_discovered(class_id)
    task = task_factory(config, global_state)(8, 29, 27)

    event_counts: Counter[str] = Counter()
    iteration_events: dict[int, list[dict[str, object]]] = defaultdict(list)
    trace_path = output / "decisions.jsonl"
    with trace_path.open("w", encoding="utf-8") as stream:
        def sink(record: dict[str, object]) -> None:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            event_counts[str(record["event"])] += 1
            iteration_events[int(record.get("iteration", 0))].append(record)

        set_trace_sink(sink)
        try:
            for _ in range(iterations):
                task.step()
        finally:
            set_trace_sink(None)

    result = task.result().to_dict()
    payload: dict[str, object] = {
        "meta": {
            "experiment_version": experiment_version,
            "based_on": based_on,
            "class_id": 8,
            "search_index": 29,
            "parent_search_index": 27,
            "seed": task.state.config.seed,
            "iterations": int(iterations),
            "pre_discovered_class_ids": sorted(baseline.PRE_CLASS8_DISCOVERED),
            "trace_event_counts": dict(sorted(event_counts.items())),
            "rzq_scoring": {
                "process_score": "rank_gain + flat + supportability + decline",
                "terminal_mode": "baseline_static_without_class44_special_case",
                "invalid_terminal_score": -10.0,
                "escape_enabled": False,
                "novelty_enabled": False,
                "backpropagated_component": "survival",
                "expansion_symmetry_quotient": "parent_stabilizer",
                "root_uses_full_partition_group": True,
                "representative_action_rule": "minimum_action_id",
                "flat_closure": True,
                "global_symmetric_node_dedup": True,
                "canonical_key_role": "lookup_only",
                "dynamic_node_compatibility_cache": True,
                "expansion_score": "2*log(1+compatible_classes)+mean(log(1+compat_masks_per_class))",
                "expansion_prior": "softmax(expansion_score/prior_temperature)",
                "expansion_scores_all_canonical_actions": True,
                "expansion_bucket_cycle": {
                    "0": "compatibility_richness_prior",
                    "1": "uniform_random",
                },
                "ucb_visit_counts": "parent.visits_and_child.visits",
                "ucb_value": "edge_survival_q",
                **(ucb_metadata or {}),
            },
        },
        "result": result,
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {experiment_version} MCTS trace",
        "",
        f"{experiment_version} 基于 {based_on}，保留相同的 class8 配置、预发现 class、rollout scorer、terminal 规则和 survival 回传。",
        feature_description or "共同功能：action 对称分类与最小编号代表、flat closure、动态 compat 缓存、全局对称节点合并、compat-richness expansion prior、prior/随机双 bucket，以及使用真实父子节点 visits 的 UCB 探索项。",
        *( [f"UCB 覆盖参数：`{json.dumps(ucb_metadata, ensure_ascii=False, sort_keys=True)}`"] if ucb_metadata else [] ),
        "",
        "## 配置",
        "",
        f"- Iterations: `{iterations}`",
        f"- Seed: `{task.state.config.seed}`",
        f"- Pre-discovered classes: `{sorted(baseline.PRE_CLASS8_DISCOVERED)}`",
        f"- Nodes created: `{result['nodes_created']}`",
        f"- Event counts: `{dict(sorted(event_counts.items()))}`",
        "",
        "## 文件",
        "",
        "- `decisions.jsonl`: 完整决策 trace。",
        f"- `result.json`: 配置、{experiment_version}功能标记和最终结果。",
        "- `environment.json`: 运行环境及源码指纹。",
        "",
        "## 每轮索引",
        "",
        "| Iteration | Tree path before expansion | Expansion | Rollout blocks | Terminal |",
        "|---:|---|---:|---|---|",
    ]
    for iteration in range(1, iterations + 1):
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
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.output.resolve(), iterations=args.iterations)
    print(json.dumps({"output": str(args.output), "result": payload["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
