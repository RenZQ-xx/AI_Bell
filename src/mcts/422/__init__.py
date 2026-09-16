"""Single-tree symmetry Filler-Corrector search for Bell (4,2,2)."""

from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    if name in {"Bell422Search", "SearchConfig", "run_search"}:
        module = import_module("mcts.422.search")
        return getattr(module, name)
    if name in {"Bell422V25Search", "V25SearchConfig", "run_v25_search"}:
        module = import_module("mcts.422.v25_search")
        if name == "run_v25_search":
            return module.run_search
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "Bell422Search",
    "SearchConfig",
    "run_search",
    "Bell422V25Search",
    "V25SearchConfig",
    "run_v25_search",
]
