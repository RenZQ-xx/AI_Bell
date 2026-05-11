"""
Backward-compatible import path for the n-2-2 SDP solver.

New code should import from `aibell.solvers.n22`.
"""

from .solvers.n22 import BellInequalitySolver

__all__ = ["BellInequalitySolver"]
