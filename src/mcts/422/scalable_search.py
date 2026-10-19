from __future__ import annotations

"""Scalable 4-2-2 entry point with geometric transpositions and ridge pivots."""

import sys
from pathlib import Path
from typing import Sequence

from .search import DEFAULT_RUNS_DIR, main as run_main


def _has_option(arguments: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(option + "=") for argument in arguments)


def _argument_value(arguments: list[str], option: str, default: int) -> int:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return int(arguments[index + 1])
        if argument.startswith(option + "="):
            return int(argument.split("=", 1)[1])
    return int(default)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    for option in (
        "--geometric-transpositions",
        "--hierarchical-pattern-policy",
        "--ridge-pivot",
        "--context-free-action-cache",
    ):
        if not _has_option(arguments, option):
            arguments.append(option)
    defaults = {
        "--rollout-pattern-candidates": 2,
        "--rollout-action-candidates": 6,
        "--filler-candidate-limit": 16,
        "--corrector-candidate-limit": 24,
        "--tail-checks": 8,
        "--structural-novelty-reward": 0.05,
        "--repeated-basin-penalty-cap": 0.15,
    }
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            arguments.extend((option, str(value)))
    if not _has_option(arguments, "--output"):
        iterations = _argument_value(arguments, "--iterations", 300)
        output = DEFAULT_RUNS_DIR / f"bell422_scalable_i{iterations}_v2.json"
        arguments.extend(("--output", str(Path(output))))
    return run_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
