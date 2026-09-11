"""Run v3 plus unbiased rollout cutoff sampling as class-8 v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_v2 import run
from mcts.rzq_.run_class8_v3 import v3_task_factory


DEFAULT_OUTPUT = HERE / "runs" / "class8_v5_300_trace"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(),
        iterations=args.iterations,
        task_factory=v3_task_factory,
        experiment_version="class8_v5",
        based_on="class8_v3",
        ucb_metadata={
            "ucb_selection": "minimum_visits_then_normalized_ucb",
            "min_action_visits": 5,
            "normalized_q": "(edge_survival_q-parent_q_min)/(parent_q_max-parent_q_min+1e-12)",
            "equal_q_normalized_value": 0.5,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
            "rollout_tie_rng": "experiment_seed_plus_2147483647",
        },
        feature_description="v5 严格保留 v3 的全部树、UCB、scorer、terminal、回传、对称 action、节点合并、flat closure、compatibility prior 和双 bucket；唯一算法变化是 rollout top-k 边界同分组的均匀随机截断。",
    )
    print(json.dumps({"output": str(args.output), "result": payload["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
