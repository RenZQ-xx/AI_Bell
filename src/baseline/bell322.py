from __future__ import annotations

from typing import Iterator

import numpy as np


def iter_deterministic_states_322() -> Iterator[tuple[int, int, int, int, int, int]]:
    """Yield all deterministic assignments for the Bell 3-2-2 scenario.

    A state is `(a0, a1, b0, b1, c0, c1)`, where each entry is either `+1` or
    `-1`.  There are `2 ** 6 = 64` such deterministic vertices.
    """
    for a0 in (1, -1):
        for a1 in (1, -1):
            for b0 in (1, -1):
                for b1 in (1, -1):
                    for c0 in (1, -1):
                        for c1 in (1, -1):
                            yield (a0, a1, b0, b1, c0, c1)


def bell322_coordinates(state: tuple[int, int, int, int, int, int]) -> list[float]:
    """Convert one deterministic state into the 26 Bell-coordinate entries."""
    a0, a1, b0, b1, c0, c1 = state
    return [
        a0,
        a1,
        b0,
        b1,
        c0,
        c1,
        a0 * b0,
        a0 * b1,
        a1 * b0,
        a1 * b1,
        a0 * c0,
        a0 * c1,
        a1 * c0,
        a1 * c1,
        b0 * c0,
        b0 * c1,
        b1 * c0,
        b1 * c1,
        a0 * b0 * c0,
        a0 * b1 * c0,
        a1 * b0 * c0,
        a1 * b1 * c0,
        a0 * b0 * c1,
        a0 * b1 * c1,
        a1 * b0 * c1,
        a1 * b1 * c1,
    ]


def generate_bell322_points(*, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Return the 64 x 26 deterministic Bell 3-2-2 vertex matrix."""
    rows = [bell322_coordinates(state) for state in iter_deterministic_states_322()]
    return np.asarray(rows, dtype=dtype)


def generate_points_322() -> np.ndarray:
    """Compatibility alias for the old exploratory function name."""
    return generate_bell322_points()
