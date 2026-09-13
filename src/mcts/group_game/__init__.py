from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .search import GroupGameConfig, SingleTreeGroupGame, run_group_game_search

__all__ = ["GroupGameConfig", "SingleTreeGroupGame", "run_group_game_search"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".search", __name__), name)
    globals()[name] = value
    return value
