"""V4.6: v4.5 with six alternating prior/uniform expansion buckets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_v4_5 import v4_5_task_factory
from mcts.rzq_.run_class8_v2 import run

DEFAULT_OUTPUT = HERE / "runs" / "class8_v4_6_300_trace"


def v4_6_task_factory(config, global_discovery):
    base_factory = v4_5_task_factory(config, global_discovery)

    def factory(class_id, search_index, parent_search_index):
        task = base_factory(class_id, search_index, parent_search_index)
        task.state.scorer.uniform_expansion_buckets_only = False
        task.state.scorer.prior_expansion_buckets = (0, 2, 4)
        return task

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(), iterations=args.iterations,
        task_factory=v4_6_task_factory,
        experiment_version="class8_v4.6", based_on="class8_v4.5",
        ucb_metadata={
            "expansion_symmetry_quotient": None,
            "representative_action_rule": "all_unselected_blocks",
            "global_symmetric_node_dedup": False,
            "flat_closure": True,
            "exact_closed_state_reuse": True,
            "expansion_prior": "compatibility_richness",
            "expansion_bucket_count": 6,
            "expansion_bucket_cycle": {
                "0,2,4": "compatibility_richness_prior",
                "1,3,5": "uniform_random",
            },
            "ucb_selection": "minimum_visits_then_raw_edge_q_ucb",
            "min_action_visits": 1,
            "normalized_q": None,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
        },
        feature_description="v4.6 相对 v4.5 唯一算法改动：六个 expansion bucket 按 prior/random 交替；bucket 0、2、4使用compatibility-richness prior，1、3、5均匀随机。",
    )
    print(json.dumps({"output": str(args.output), "counts": payload["result"]["encountered_label_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
