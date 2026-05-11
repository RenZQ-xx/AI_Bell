"""
Backward-compatible exports for older experiment scripts.

New code should import point generators from `aibell.points` and geometry
helpers from `aibell.geometry`.
"""

from .geometry import check_points_form_hyperplane
from .points import Points_222, Points_223, Points_232, Points_322

__all__ = [
    "Points_222",
    "Points_223",
    "Points_232",
    "Points_322",
    "check_points_form_hyperplane",
]
