"""Run v1 plus unbiased rollout cutoff sampling as class-8 v4."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from audit_environment import HERE
from baseline.orbit_blocks import empty_key
from mcts.search import _get_or_create_node
from mcts.rzq_.run_class8_round1 import rzq_task_factory
from mcts.rzq_.run_class8_v2 import run


DEFAULT_OUTPUT = HERE / "runs" / "class8_v4_300_trace"


def v4_task_factory(config, global_discovery):
    """Recreate the v1 tree policy while retaining the corrected rollout pool."""

    base_factory = rzq_task_factory(config, global_discovery)

    def factory(class_id: int, search_index: int, parent_search_index: int | None):
        task = base_factory(class_id, search_index, parent_search_index)
        scorer = task.state.scorer
        # These three optional hooks are precisely the post-v1 tree features.
        scorer.node_manager = None
        scorer.representative_action_families = None
        scorer.prepare_expansion_priors = None
        scorer.prepare_legacy_action_priors = None
        task.state.config = replace(
            task.state.config,
            ucb_use_real_node_visits=False,
            ucb_min_action_visits=0,
            ucb_normalize_edge_survival_q=False,
        )
        task.state.nodes.clear()
        root_key = empty_key(len(scorer.blocks))
        task.state.root = _get_or_create_node(task.state.nodes, scorer, root_key, path=[])
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
        task_factory=v4_task_factory,
        experiment_version="class8_v4",
        based_on="class8_v1",
        ucb_metadata={
            "ucb_visit_counts": "parent.value_visits_and_child.value_visits",
            "ucb_value": "survival_q",
            "rollout_tied_cutoff": "uniform_without_replacement",
            "rollout_tie_rng": "experiment_seed_plus_2147483647",
        },
        feature_description="v4 严格保留 v1 的普通 action 树、六 bucket progressive widening、value_visits UCB、scorer、terminal 和 survival 回传；唯一算法变化是 rollout top-k 边界同分组的均匀随机截断。",
    )
    print(json.dumps({"output": str(args.output), "result": payload["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
