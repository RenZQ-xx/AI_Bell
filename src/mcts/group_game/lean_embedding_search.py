from __future__ import annotations

"""v25 entry point selected by the v24 mechanism ablation matrix."""

import sys
from pathlib import Path
from typing import Iterable

from .embedding_search import _argument_value, _has_option
from .search import DEFAULT_RUNS_DIR, main as run_main


def main(argv: Iterable[str] | None = None) -> dict[str, object]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    enabled = (
        "--adaptive-corrector-replay",
        "--ridge-pivot",
        "--eager-ridge-focus",
        "--hierarchical-pattern-policy",
        "--directed-pattern-relations",
        "--learn-relation-values",
        "--retain-pattern-embeddings",
        "--embedding-relation-policy",
        "--no-leaf-value-shaping",
    )
    for option in enabled:
        if not _has_option(arguments, option):
            arguments.append(option)
    defaults = {
        "--ridge-max-candidates": 32768,
        "--ridge-batch-size": 256,
        "--ridge-focus-batches": 48,
        "--max-pattern-embeddings-per-family": 4,
        "--max-pattern-analyses-per-class": 4,
    }
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            arguments.extend((option, str(value)))
    if not _has_option(arguments, "--output"):
        iterations = _argument_value(arguments, "--iterations", 300)
        output = DEFAULT_RUNS_DIR / f"single_tree_group_game_i{iterations}_v25.json"
        arguments.extend(("--output", str(Path(output))))
    return run_main(arguments)


if __name__ == "__main__":
    main()
