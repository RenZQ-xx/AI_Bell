from __future__ import annotations

"""Reproducible v24 ablations with one mechanism disabled per run."""

import argparse
import sys
from pathlib import Path
from typing import Iterable

from .embedding_search import _argument_value, _has_option
from .search import DEFAULT_RUNS_DIR, main as run_main


VARIANT_DISABLED_OPTIONS = {
    "full": (),
    "no_family_value": ("--share-embedding-family-values",),
    "no_embedding_policy": ("--embedding-relation-policy",),
    "no_relation_ucb": ("--learn-relation-values",),
    "no_reheat": ("--adaptive-filler-reheat",),
    "no_family_value_no_embedding_policy": (
        "--share-embedding-family-values",
        "--embedding-relation-policy",
    ),
    "no_family_value_no_reheat": (
        "--share-embedding-family-values",
        "--adaptive-filler-reheat",
    ),
    "simplified_no_relation_ucb": (
        "--share-embedding-family-values",
        "--adaptive-filler-reheat",
        "--learn-relation-values",
    ),
    "simplified_cap2": (
        "--share-embedding-family-values",
        "--adaptive-filler-reheat",
    ),
}


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_DISABLED_OPTIONS),
        required=True,
    )
    parsed, arguments = parser.parse_known_args(raw_arguments)
    disabled = set(VARIANT_DISABLED_OPTIONS[parsed.variant])
    enabled = (
        "--adaptive-corrector-replay",
        "--ridge-pivot",
        "--eager-ridge-focus",
        "--hierarchical-pattern-policy",
        "--directed-pattern-relations",
        "--learn-relation-values",
        "--adaptive-filler-reheat",
        "--retain-pattern-embeddings",
        "--share-embedding-family-values",
        "--embedding-relation-policy",
    )
    for option in enabled:
        if option not in disabled and not _has_option(arguments, option):
            arguments.append(option)
    defaults = {
        "--ridge-max-candidates": 32768,
        "--ridge-batch-size": 256,
        "--ridge-focus-batches": 48,
        "--filler-reheat-class-patience": 512,
        "--filler-reheat-frontier-patience": 256,
        "--filler-reheat-ratio": 0.35,
        "--max-pattern-embeddings-per-family": 4,
        "--max-pattern-analyses-per-class": 4,
    }
    if parsed.variant == "simplified_cap2":
        defaults["--max-pattern-embeddings-per-family"] = 2
        defaults["--max-pattern-analyses-per-class"] = 2
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            arguments.extend((option, str(value)))
    if not _has_option(arguments, "--output"):
        iterations = _argument_value(arguments, "--iterations", 6000)
        output = (
            DEFAULT_RUNS_DIR
            / f"single_tree_group_game_i{iterations}_v24_{parsed.variant}.json"
        )
        arguments.extend(("--output", str(Path(output))))
    return run_main(arguments)


if __name__ == "__main__":
    main()
