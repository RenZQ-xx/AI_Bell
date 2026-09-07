from __future__ import annotations

"""Opt-in, side-effect-free decision tracing for MCTS diagnostics."""

from contextvars import ContextVar
from typing import Callable


TraceSink = Callable[[dict[str, object]], None]

_sink: ContextVar[TraceSink | None] = ContextVar("mcts_trace_sink", default=None)
_iteration: ContextVar[int | None] = ContextVar("mcts_trace_iteration", default=None)


def set_trace_sink(sink: TraceSink | None) -> None:
    _sink.set(sink)


def set_trace_iteration(iteration: int | None) -> None:
    _iteration.set(iteration)


def emit(event: str, **payload: object) -> None:
    sink = _sink.get()
    if sink is None:
        return
    record: dict[str, object] = {"event": event}
    iteration = _iteration.get()
    if iteration is not None:
        record["iteration"] = iteration
    record.update(payload)
    sink(record)
