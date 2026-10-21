"""双时间步（BDF2 + 隐式内迭代）的时间精度阶数：等熵涡，时间自收敛。

等熵涡的特征时间（涡核切向速度 / 涡核尺度）约 2.5e-5 s。取三个物理时间步长推进到同一时刻，以最小步长的
解为参考（空间离散相同，差值只含时间离散误差）。BDF2 是二阶：步长减半误差降到 1/4。每个物理步的 BDF 方程由
带物理时间项的 Newton-Krylov 步求解（`time_integration/implicit/newton/physical_time.py`），内迭代解到 1e-7。

实测（2026-10-10，P2，20x20x1）：dt = 2e-5 / 1e-5 / 5e-6 对 1.25e-6 的参考解，x 速度平均误差
1.56e-3 / 3.75e-4 / 8.75e-5，阶数 2.05 / 2.10；每个物理步 2~3 次内迭代。此前的显式内迭代在这个算例上
给 200 次内迭代伪残差仍在增长（`test_isentropic_vortex.py` 模块文档第 2 条的旧记录）。

本测试取 dt = 2e-5 / 1e-5，参考 2.5e-6（控制耗时）：参考解自身的时间误差使观测比值为
(400 - 6.25) / (100 - 6.25) = 4.2，即阶数 2.07。
"""

import numpy as np

from autoflowcfd.core.fr_solver import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.validation import test_isentropic_vortex as V
from tests.validation._isentropic_vortex import build_vortex_farfield_ghost_provider, build_vortex_mesh
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR

T_END = 1.0e-4


def _advance(dt):
    mesh = build_vortex_mesh(V.ORDER, V.NX, V.NY, V.NZ, V.LX, V.H, V.LZ)
    solver = FRSolver(mesh=mesh, order=V.ORDER, turb_model_name="NONE", time_scheme=TimeIntegrationScheme.DUAL_TIME,
                      rho_inf=V.RHO_INF, vel_inf=V.U_INF, p_inf=V.P_INF, dual_time_inner_iter=30,
                      mu_molecular=MU_MOLECULAR)
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_vortex_farfield_ghost_provider(
        mesh, V.LX, V.H, V.LZ, V.RHO_INF, V.P_INF, V.U_INF)
    V._set_vortex_ic(solver, mesh, t=0.0)
    solver.time_integrator.dual_time_tolerance = 1.0e-7
    inner = []
    for _ in range(int(round(T_END / dt))):
        solver.step(dt)
        assert solver._dual_time_inner["converged"], solver._dual_time_inner
        inner.append(solver._dual_time_inner["iterations"])
    return solver.state.Q.copy(), inner


def test_bdf2_with_implicit_inner_iterations_is_second_order_in_time():
    reference, _ = _advance(2.5e-6)
    coarse, inner_coarse = _advance(2.0e-5)
    fine, inner_fine = _advance(1.0e-5)
    assert max(inner_coarse + inner_fine) <= 6, (inner_coarse, inner_fine)

    orders = []
    for column in (0, 1, 2, 4):                               # 密度、两个面内速度分量、压力
        e_coarse = np.abs(coarse[..., column] - reference[..., column]).mean()
        e_fine = np.abs(fine[..., column] - reference[..., column]).mean()
        assert e_fine > 0.0
        orders.append(float(np.log2(e_coarse / e_fine)))
    assert all(1.8 <= o <= 2.4 for o in orders), orders
