"""V4.5: v4.4 with all six expansion buckets sampled uniformly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_v4_4 import v4_4_task_factory
from mcts.rzq_.run_class8_v2 import run

DEFAULT_OUTPUT = HERE / "runs" / "class8_v4_5_300_trace"


def v4_5_task_factory(config, global_discovery):
    base_factory = v4_4_task_factory(config, global_discovery)

    def factory(class_id, search_index, parent_search_index):
        task = base_factory(class_id, search_index, parent_search_index)
        task.state.scorer.uniform_expansion_buckets_only = True
        return task

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(), iterations=args.iterations,
        task_factory=v4_5_task_factory,
        experiment_version="class8_v4.5", based_on="class8_v4.4",
        ucb_metadata={
            "expansion_symmetry_quotient": None,
            "representative_action_rule": "all_unselected_blocks",
            "global_symmetric_node_dedup": False,
            "flat_closure": True,
            "exact_closed_state_reuse": True,
            "expansion_prior": None,
            "expansion_bucket_count": 6,
            "expansion_bucket_cycle": {"0-5": "uniform_random"},
            "ucb_selection": "minimum_visits_then_raw_edge_q_ucb",
            "min_action_visits": 1,
            "normalized_q": None,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
        },
        feature_description="v4.5 相对 v4.4 唯一算法改动：六个 expansion bucket 全部在未展开动作中均匀随机抽样；六 bucket 渐进扩展容量及其他设置不变。",
    )
    print(json.dumps({"output": str(args.output), "counts": payload["result"]["encountered_label_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
