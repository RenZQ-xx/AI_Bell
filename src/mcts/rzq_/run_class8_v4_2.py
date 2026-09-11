"""V4.2: v4.1 with minimum child visits reduced from five to one."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_v4_1 import v4_1_task_factory
from mcts.rzq_.run_class8_v2 import run

DEFAULT_OUTPUT = HERE / "runs" / "class8_v4_2_300_trace"


def v4_2_task_factory(config, global_discovery):
    base_factory = v4_1_task_factory(config, global_discovery)

    def factory(class_id, search_index, parent_search_index):
        task = base_factory(class_id, search_index, parent_search_index)
        task.state.config = replace(task.state.config, ucb_min_action_visits=1)
        return task

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(), iterations=args.iterations,
        task_factory=v4_2_task_factory,
        experiment_version="class8_v4.2", based_on="class8_v4.1",
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
            "ucb_selection": "minimum_visits_then_normalized_ucb",
            "min_action_visits": 1,
            "normalized_q": "(edge_survival_q-parent_q_min)/(parent_q_max-parent_q_min+1e-12)",
            "equal_q_normalized_value": 0.5,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
        },
        feature_description="v4.2 相对 v4.1 唯一算法改动：ucb_min_action_visits 从5降到1。其余 flat closure、全部普通 block 动作、无对称合并、compatibility-richness prior、双 bucket、归一化父边Q、真实节点 visits 与 rollout 同分修复均不变。",
    )
    print(json.dumps({"output": str(args.output), "counts": payload["result"]["encountered_label_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
