"""V4.3: v4.2 with raw parent-edge Q instead of normalized Q."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_v4_2 import v4_2_task_factory
from mcts.rzq_.run_class8_v2 import run

DEFAULT_OUTPUT = HERE / "runs" / "class8_v4_3_300_trace"


def v4_3_task_factory(config, global_discovery):
    base_factory = v4_2_task_factory(config, global_discovery)

    def factory(class_id, search_index, parent_search_index):
        task = base_factory(class_id, search_index, parent_search_index)
        task.state.config = replace(task.state.config, ucb_normalize_edge_survival_q=False)
        return task

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(), iterations=args.iterations,
        task_factory=v4_3_task_factory,
        experiment_version="class8_v4.3", based_on="class8_v4.2",
        ucb_metadata={
            "expansion_symmetry_quotient": None,
            "root_uses_full_partition_group": False,
            "representative_action_rule": "all_unselected_blocks",
            "global_symmetric_node_dedup": False,
            "canonical_key_role": "exact_closed_state_lookup_only",
            "flat_closure": True,
            "exact_closed_state_reuse": True,
            "expansion_scores_all_canonical_actions": False,
            "expansion_scores_all_actions": True,
            "ucb_selection": "minimum_visits_then_raw_edge_q_ucb",
            "min_action_visits": 1,
            "normalized_q": None,
            "equal_q_normalized_value": None,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
        },
        feature_description="v4.3 相对 v4.2 唯一算法改动：selection 使用原始父边 survival Q，不做 min-max 归一化。最低 child.visits=1；其他设置不变。",
    )
    print(json.dumps({"output": str(args.output), "counts": payload["result"]["encountered_label_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

