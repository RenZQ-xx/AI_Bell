"""Run the first class-8 scorer experiment in the isolated RZQ environment."""

from __future__ import annotations

from dataclasses import replace
import json

from audit_environment import HERE, snapshot

from mcts import trace_class8_current_300 as experiment
from mcts.interrupt_search import _default_task_factory as baseline_task_factory
from mcts.search import _get_or_create_node
from baseline.orbit_blocks import empty_key
from mcts.rzq_.compatibility import ClassCompatibilityIndex
from mcts.rzq_.rollout_scorer import RZQRolloutScorer, RolloutScoreConfig


def rzq_task_factory(config, global_discovery):
    """Build class tasks using only the new survival score for MCTS values."""

    baseline_factory = baseline_task_factory(config, global_discovery)

    def factory(class_id: int, search_index: int, parent_search_index: int | None):
        task = baseline_factory(class_id, search_index, parent_search_index)
        base = task.state.scorer

        # The interrupt runner has a separate fast path for already-known exact
        # classes.  Align its config with the RZQ terminal rule as well.
        base.config = replace(
            base.config,
            class44_terminal_score=base.config.target_terminal_score,
            invalid_terminal_score=-10.0,
            invalid_rank_terminal_score=-10.0,
            boundary_invalid_base_score=-10.0,
            boundary_invalid_closer_weight=0.0,
        )
        compatibility = ClassCompatibilityIndex.build(base.blocks)
        task.state.scorer = RZQRolloutScorer(
            base,
            compatibility,
            discovered_class_ids=global_discovery.discovered_exact_classes,
            config=RolloutScoreConfig(invalid_terminal_score=-10.0),
        )
        task.state.scorer.configure_discovery_cache(
            lambda: global_discovery.discovered_exact_classes,
            lambda: global_discovery.discovery_epoch,
        )
        task.state.nodes.clear()
        root_key = empty_key(len(base.blocks))
        task.state.root = _get_or_create_node(
            task.state.nodes,
            task.state.scorer,
            root_key,
            path=[],
        )

        # Escape and novelty are disabled at their source.  The generic MCTS
        # value carrier retains zero-valued fields for API compatibility; only
        # survival is accumulated and affects selection/backpropagation.
        task.state.compatibility_bank = None
        task.state.seen_signatures.clear()
        task.state.config = replace(
            task.state.config,
            selection_survival_weight=1.0,
            selection_novelty_weight=0.0,
            compatibility_examples_path=None,
            ucb_use_real_node_visits=True,
        )
        return task

    return factory


def main() -> None:
    output = HERE / "runs" / "class8_v1_300_trace"
    if output.exists():
        raise SystemExit(f"Output already exists; preserve or move it before rerunning: {output}")
    output.mkdir(parents=True)
    (output / "environment.json").write_text(
        json.dumps(snapshot(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    experiment.OUTPUT_DIR = output
    experiment._default_task_factory = rzq_task_factory
    experiment.main()
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["meta"]["rzq_scoring"] = {
        "process_score": "rank_gain + flat + supportability + decline",
        "terminal_mode": "baseline_static_without_class44_special_case",
        "invalid_terminal_score": -10.0,
        "escape_enabled": False,
        "novelty_enabled": False,
        "backpropagated_component": "survival",
        "expansion_symmetry_quotient": "parent_stabilizer",
        "root_uses_full_partition_group": True,
        "representative_action_rule": "minimum_action_id",
        "flat_closure": True,
        "global_symmetric_node_dedup": True,
        "canonical_key_role": "lookup_only",
        "dynamic_node_compatibility_cache": True,
        "expansion_score": "2*log(1+compatible_classes)+mean(log(1+compat_masks_per_class))",
        "expansion_prior": "softmax(expansion_score/prior_temperature)",
        "expansion_scores_all_canonical_actions": True,
        "expansion_bucket_cycle": {
            "0": "compatibility_richness_prior",
            "1": "uniform_random",
        },
        "ucb_visit_counts": "parent.visits_and_child.visits",
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
