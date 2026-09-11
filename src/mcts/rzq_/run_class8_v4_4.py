"""V4.4: v4.3 with six compatibility-prior expansion buckets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_environment import HERE
from mcts.rzq_.run_class8_v4_3 import v4_3_task_factory
from mcts.rzq_.run_class8_v2 import run

DEFAULT_OUTPUT = HERE / "runs" / "class8_v4_4_300_trace"


def v4_4_task_factory(config, global_discovery):
    base_factory = v4_3_task_factory(config, global_discovery)

    def factory(class_id, search_index, parent_search_index):
        task = base_factory(class_id, search_index, parent_search_index)
        task.state.scorer.expansion_bucket_count = 6
        return task

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(), iterations=args.iterations,
        task_factory=v4_4_task_factory,
        experiment_version="class8_v4.4", based_on="class8_v4.3",
        ucb_metadata={
            "expansion_symmetry_quotient": None,
            "representative_action_rule": "all_unselected_blocks",
            "global_symmetric_node_dedup": False,
            "flat_closure": True,
            "exact_closed_state_reuse": True,
            "expansion_prior": "compatibility_richness",
            "expansion_bucket_count": 6,
            "expansion_bucket_cycle": {
                "0": "compatibility_richness_prior",
                "1-5": "uniform_random",
            },
            "ucb_selection": "minimum_visits_then_raw_edge_q_ucb",
            "min_action_visits": 1,
            "normalized_q": None,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
        },
        feature_description="v4.4 相对 v4.3 唯一算法改动：compatibility-richness expansion 的 bucket 数从2恢复为6；bucket0按prior抽样，bucket1至5均匀随机。其他设置不变。",
    )
    print(json.dumps({"output": str(args.output), "counts": payload["result"]["encountered_label_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
