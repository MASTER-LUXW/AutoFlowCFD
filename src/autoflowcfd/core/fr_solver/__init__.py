"""
AutoFlowCFD Core - FR Solver Module

通量重构（FR）求解器主模块，包含求解器类、状态管理、时间步推进等核心功能。
"""

from autoflowcfd.core.fr_solver.solver import FRSolver
from autoflowcfd.core.fr_solver.state import FRState, SolverResult

__all__ = ['FRSolver', 'FRState', 'SolverResult']
