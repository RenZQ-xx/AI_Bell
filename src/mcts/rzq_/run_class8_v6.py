"""Run v5 with the v1 structural-score prior as class-8 v6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_v2 import run
from mcts.rzq_.run_class8_v3 import v3_task_factory


DEFAULT_OUTPUT = HERE / "runs" / "class8_v6_300_trace"


def v6_task_factory(config, global_discovery):
    base_factory = v3_task_factory(config, global_discovery)

    def factory(class_id: int, search_index: int, parent_search_index: int | None):
        task = base_factory(class_id, search_index, parent_search_index)
        task.state.scorer.expansion_prior_mode = "structural_score"
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
        task_factory=v6_task_factory,
        experiment_version="class8_v6",
        based_on="class8_v5",
        ucb_metadata={
            "ucb_selection": "minimum_visits_then_normalized_ucb",
            "min_action_visits": 5,
            "normalized_q": "(edge_survival_q-parent_q_min)/(parent_q_max-parent_q_min+1e-12)",
            "equal_q_normalized_value": 0.5,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
            "rollout_tie_rng": "experiment_seed_plus_2147483647",
            "expansion_prior": "softmax(v1_structural_score/prior_temperature)",
        },
        feature_description="v6严格保留v5的全部功能和配置；唯一变化是将canonical action的compatibility-richness prior替换为v1 structural scorer的softmax prior。该prior同时用于bucket 0 expansion和后续UCB。",
    )
    print(json.dumps({"output": str(args.output), "result": payload["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
