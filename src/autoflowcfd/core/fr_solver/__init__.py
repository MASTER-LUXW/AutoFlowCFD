"""
AutoFlowCFD Core - FR Solver Module

通量重构（FR）求解器主模块，包含求解器类、状态管理、时间步推进等核心功能。
"""

from typing import TYPE_CHECKING

from autoflowcfd.core.fr_solver.state import FRState, SolverResult

if TYPE_CHECKING:
    from autoflowcfd.core.fr_solver.solver import FRSolver

__all__ = ['FRSolver', 'FRState', 'SolverResult']


def __getattr__(name: str):
    """`FRSolver` 在首次访问时才导入（PEP 562）：求解器类依赖时间积分、隐式步等几乎全部数值模块，而它们又要用到本包
    的叶子模块（如 `state.N_MEAN_FLOW_VARS`）；包导入时预导入求解器类会让这些模块先导入时循环导入失败（2026-10-22
    I-123 以前被顶层包预导入 API 的加载顺序掩盖）。"""
    if name == "FRSolver":
        from autoflowcfd.core.fr_solver.solver import FRSolver

        return FRSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
