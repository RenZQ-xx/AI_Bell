"""Run v4 with compatibility-richness priors as class-8 v7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.rollout_scorer import RZQRolloutScorer
from mcts.rzq_.run_class8_v2 import run
from mcts.rzq_.run_class8_v4 import v4_task_factory


DEFAULT_OUTPUT = HERE / "runs" / "class8_v7_300_trace"


def v7_task_factory(config, global_discovery):
    base_factory = v4_task_factory(config, global_discovery)

    def factory(class_id: int, search_index: int, parent_search_index: int | None):
        task = base_factory(class_id, search_index, parent_search_index)
        # v4 shadows all post-v1 hooks. Restore only the noncanonical prior hook.
        task.state.scorer.prepare_legacy_action_priors = (
            RZQRolloutScorer.prepare_legacy_action_priors.__get__(task.state.scorer)
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
        task_factory=v7_task_factory,
        experiment_version="class8_v7",
        based_on="class8_v4",
        ucb_metadata={
            "ucb_visit_counts": "parent.value_visits_and_child.value_visits",
            "ucb_value": "survival_q",
            "expansion_prior": "softmax(compatibility_richness/prior_temperature)",
            "rollout_tied_cutoff": "uniform_without_replacement",
        },
        feature_description="v7保留v4的40-action普通树、六bucket、value_visits UCB、scorer、terminal、回传和rollout并列修复；唯一变化是将structural prior替换为逐真实action的compatibility-richness prior。",
    )
    print(json.dumps({"output": str(args.output), "result": payload["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
