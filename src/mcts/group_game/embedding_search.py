from __future__ import annotations

"""v24 entry point for canonical pattern families and concrete embeddings."""

import sys
from pathlib import Path
from typing import Iterable

from .search import DEFAULT_RUNS_DIR, main as run_main


def _argument_value(arguments: list[str], option: str, default: int) -> int:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return int(arguments[index + 1])
        if argument.startswith(option + "="):
            return int(argument.split("=", 1)[1])
    return int(default)


def _has_option(arguments: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in arguments
    )


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    arguments = list(sys.argv[1:] if argv is None else argv)
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
        if not _has_option(arguments, option):
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
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            arguments.extend((option, str(value)))
    if not _has_option(arguments, "--output"):
        iterations = _argument_value(arguments, "--iterations", 300)
        output = DEFAULT_RUNS_DIR / f"single_tree_group_game_i{iterations}_v24.json"
        arguments.extend(("--output", str(Path(output))))
    return run_main(arguments)


if __name__ == "__main__":
    main()
