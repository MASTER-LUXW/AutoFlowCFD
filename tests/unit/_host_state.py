# -*- coding: utf-8 -*-
"""求解器替身的主机视图接口（2026-10-04 起 checkpoint 恢复与落盘经 `host_view()` / `edit_host_state()`）。

CPU `FRSolver` 的约定：状态本来就在主机上，视图即自身、编辑后无需写回（见
`core/fr_solver/solver/solve_loop.py`；单 GPU 的视图见 `core/gpu/solver/host_view.py`）。
"""

from contextlib import contextmanager


def with_host_state(solver):
    """给 SimpleNamespace 求解器替身补上 CPU 求解器的主机视图接口与产生项渐变计数器（真实求解器在构造时由
    同一个 `init_production_ramp` 设置，checkpoint 写入/恢复读它们），返回替身本身。"""
    from autoflowcfd.core.fr_solver.turbulence.init import init_production_ramp

    if not hasattr(solver, "_turb_ramp_step"):
        init_production_ramp(solver, solver.time_integrator.scheme)
    solver.host_view = lambda: solver

    @contextmanager
    def edit_host_state():
        yield solver

    solver.edit_host_state = edit_host_state
    return solver
