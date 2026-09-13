from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .search import (
    ExactClassDiscovery,
    MCTSConfig,
    MCTSResult,
    MCTSSearchSession,
    TerminalHit,
    run_mcts_search,
)

if TYPE_CHECKING:
    from .subgroup_interrupt_search import (
        SubgroupInterruptSearchConfig,
        run_subgroup_interrupt_search,
    )
    from .subgroup_patterns import (
        SubgroupPatternAtlas,
        load_subgroup_pattern_atlas,
    )


_SUBGROUP_SEARCH_EXPORTS = {
    "SubgroupInterruptSearchConfig",
    "run_subgroup_interrupt_search",
}

_SUBGROUP_PATTERN_EXPORTS = {
    "SubgroupPatternAtlas",
    "load_subgroup_pattern_atlas",
}


def __getattr__(name: str):
    if name in _SUBGROUP_SEARCH_EXPORTS:
        module_name = ".subgroup_interrupt_search"
    elif name in _SUBGROUP_PATTERN_EXPORTS:
        module_name = ".subgroup_patterns"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "ExactClassDiscovery",
    "MCTSConfig",
    "MCTSResult",
    "MCTSSearchSession",
    "SubgroupInterruptSearchConfig",
    "SubgroupPatternAtlas",
    "TerminalHit",
    "run_mcts_search",
    "run_subgroup_interrupt_search",
    "load_subgroup_pattern_atlas",
]
