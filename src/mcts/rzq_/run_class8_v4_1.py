"""V4.1: v3 tree policy without action quotient or symmetry deduplication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_environment import HERE
from baseline.orbit_blocks import empty_key
from mcts.search import _get_or_create_node
from mcts.rzq_.node_manager import RZQNodeManager
from mcts.rzq_.run_class8_v3 import v3_task_factory
from mcts.rzq_.run_class8_v2 import run

DEFAULT_OUTPUT = HERE / "runs" / "class8_v4_1_300_trace"


def v4_1_task_factory(config, global_discovery):
    base_factory = v3_task_factory(config, global_discovery)

    def factory(class_id, search_index, parent_search_index):
        task = base_factory(class_id, search_index, parent_search_index)
        scorer = task.state.scorer
        scorer.representative_action_families = None
        scorer.node_manager = RZQNodeManager(scorer, symmetry_dedup=False)
        task.state.nodes.clear()
        task.state.root = _get_or_create_node(
            task.state.nodes, scorer, empty_key(len(scorer.blocks)), path=[],
        )
        return task

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.output.resolve(), iterations=args.iterations,
        task_factory=v4_1_task_factory,
        experiment_version="class8_v4.1", based_on="class8_v3",
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
            "min_action_visits": 5,
            "normalized_q": "(edge_survival_q-parent_q_min)/(parent_q_max-parent_q_min+1e-12)",
            "equal_q_normalized_value": 0.5,
            "exploration_constant": 1.4,
            "rollout_tied_cutoff": "uniform_without_replacement",
        },
        feature_description="v4.1 保留全部普通 block 动作，不做对称节点合并；完全相同的闭包状态仍复用。开启 flat closure，使用 compatibility-richness prior、prior/随机双 bucket、child.visits>=5 最低访问保障、父边 Q min-max 归一化、真实父子节点 visits 探索项。沿用当前公共 rollout 同分修复；历史 v3 trace 尚无此修复。",
    )
    print(json.dumps({"output": str(args.output), "counts": payload["result"]["encountered_label_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
