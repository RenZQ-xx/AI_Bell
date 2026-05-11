"""
Backward-compatible import path for the 2-2-2 solvers.

New code should import from `aibell.solvers.scenario_222`.
"""

from .solvers.scenario_222 import get_classical_bound_batch, get_quantum_bound_batch

__all__ = ["get_classical_bound_batch", "get_quantum_bound_batch"]
