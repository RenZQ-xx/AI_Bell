"""Run class-8 v3 with minimum action visits and normalized UCB values."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_round1 import rzq_task_factory
from mcts.rzq_.run_class8_v2 import run


DEFAULT_OUTPUT = HERE / "runs" / "class8_v3_300_trace"


def v3_task_factory(config, global_discovery):
    base_factory = rzq_task_factory(config, global_discovery)

    def factory(class_id: int, search_index: int, parent_search_index: int | None):
        task = base_factory(class_id, search_index, parent_search_index)
        task.state.config = replace(
            task.state.config,
            ucb_min_action_visits=5,
            ucb_normalize_edge_survival_q=True,
            ucb_normalization_epsilon=1e-12,
        )
        return task

    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(),
        iterations=args.iterations,
        task_factory=v3_task_factory,
        experiment_version="class8_v3",
        based_on="class8_v2",
        ucb_metadata={
            "ucb_selection": "minimum_visits_then_normalized_ucb",
            "min_action_visits": 5,
            "normalized_q": "(edge_survival_q-parent_q_min)/(parent_q_max-parent_q_min+1e-12)",
            "equal_q_normalized_value": 0.5,
            "exploration_constant": 1.4,
        },
    )
    print(json.dumps({"output": str(args.output), "result": payload["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
