from __future__ import annotations

"""Single-factor progressive-widening and reward ablations for v25."""

import argparse
import sys
from pathlib import Path
from typing import Iterable

from .embedding_search import _argument_value, _has_option
from .search import DEFAULT_RUNS_DIR, main as run_main


VARIANT_OPTIONS = {
    "full": (),
    "child_only_widening": ("--no-multilevel-progressive-widening",),
    "no_size_buckets": ("--no-size-bucket-widening",),
    "no_action_lanes": ("--no-action-lane-widening",),
    "uniform_relation_lanes": ("--no-weighted-relation-widening",),
    "no_epoch_widening": (),
    "no_child_recycle": (),
    "no_transition_reward": ("--no-transition-reward-shaping",),
    "no_filler_transition_reward": ("--no-filler-transition-reward-shaping",),
    "no_corrector_transition_reward": ("--no-corrector-transition-reward-shaping",),
    "no_support_reward": ("--no-terminal-support-reward-shaping",),
    "no_frequency_reward": ("--no-terminal-frequency-reward-shaping",),
    "no_corrector_novelty_credit": ("--no-corrector-novelty-credit",),
    "no_endpoint_novelty_credit": ("--no-endpoint-novelty-credit",),
    "no_leaf_value": ("--no-leaf-value-shaping",),
    "sqrt_pattern_widening": (),
    "no_endpoint_context_credit": (),
    "no_endpoint_pair_credit": (),
    "no_endpoint_target_credit": (),
    "no_novel_support_bonus": (),
    "no_repeated_support_penalty": (),
    "no_known_class_frequency_penalty": (),
    "no_corrector_exact_frequency_penalty": (),
}

VARIANT_VALUES = {
    "no_epoch_widening": {
        "--discovery-epoch-child-reserve": 0,
        "--discovery-epoch-child-recycle": 0,
    },
    "no_child_recycle": {
        "--discovery-epoch-child-recycle": 0,
    },
    "sqrt_pattern_widening": {
        "--pattern-widening-beta": 0.5,
    },
    "no_endpoint_context_credit": {
        "--endpoint-context-novelty-reward": 0.0,
    },
    "no_endpoint_pair_credit": {
        "--endpoint-pair-novelty-reward": 0.0,
    },
    "no_endpoint_target_credit": {
        "--endpoint-target-novelty-reward": 0.0,
    },
    "no_novel_support_bonus": {
        "--novel-terminal-support-bonus": 0.0,
    },
    "no_repeated_support_penalty": {
        "--repeated-terminal-penalty": 0.0,
    },
    "no_known_class_frequency_penalty": {
        "--known-class-frequency-penalty": 0.0,
    },
    "no_corrector_exact_frequency_penalty": {
        "--corrector-exact-frequency-penalty": 0.0,
    },
}


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=tuple(VARIANT_OPTIONS), required=True)
    parsed, arguments = parser.parse_known_args(raw_arguments)

    enabled = (
        "--adaptive-corrector-replay",
        "--ridge-pivot",
        "--eager-ridge-focus",
        "--hierarchical-pattern-policy",
        "--directed-pattern-relations",
        "--learn-relation-values",
        "--retain-pattern-embeddings",
        "--embedding-relation-policy",
    )
    for option in enabled:
        if not _has_option(arguments, option):
            arguments.append(option)
    for option in VARIANT_OPTIONS[parsed.variant]:
        if not _has_option(arguments, option):
            arguments.append(option)

    defaults = {
        "--ridge-max-candidates": 32768,
        "--ridge-batch-size": 256,
        "--ridge-focus-batches": 48,
        "--max-pattern-embeddings-per-family": 4,
        "--max-pattern-analyses-per-class": 4,
        **VARIANT_VALUES.get(parsed.variant, {}),
    }
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            arguments.extend((option, str(value)))
    if not _has_option(arguments, "--output"):
        iterations = _argument_value(arguments, "--iterations", 3000)
        output = (
            DEFAULT_RUNS_DIR
            / f"single_tree_group_game_i{iterations}_v25_{parsed.variant}.json"
        )
        arguments.extend(("--output", str(Path(output))))
    return run_main(arguments)


if __name__ == "__main__":
    main()
