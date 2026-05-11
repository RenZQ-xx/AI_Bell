"""
Backward-compatible hyperplane helper.

New code should import `check_points_form_hyperplane` from `aibell.geometry`.
"""

from typing import Tuple

import numpy as np

from .geometry import check_points_form_hyperplane


def format_vector(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.3f}" if abs(x) > 1e-9 else "0.000" for x in vec) + "]"


def solve_hyperplane_from_index(index: list, points: np.ndarray) -> Tuple[bool, np.ndarray]:
    return check_points_form_hyperplane(points[index])
