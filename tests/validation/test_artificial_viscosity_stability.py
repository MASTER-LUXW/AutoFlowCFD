"""AutoFlowCFD V2.0 - Persson-Peraire 人工粘性（opt-in）稳定性回归测试。

见 core/fr_operators/artificial_viscosity.py 模块文档、ProjectFiles/V2.0/
7_重大问题修复-求解稳定性.md。这个新增能力默认关闭，本文件复用项目已有
的两个"真正有效的稳定性回归测试"（test_couette.py 的从错误初场出发的
800 步压力测试、test_tgv.py 的 150 步真实粘性动能衰减定量测试）验证：
启用人工粘性后，这两个测试仍然通过——不引入新的失稳，也不严重破坏
TGV 的粘性耗散物理签名。

2026-08-29 调查记录（一次虚惊，避免重蹈覆辙）：最初用一个手写的、
固定 dt=0.001 的简化脚本复现 TGV 设置测试人工粘性，任何 alpha_av
（甚至 0.001）都在第 3 步发散——一度怀疑是传感器/人工粘性本身有 bug。
追查后发现是复现脚本本身的问题：真实 test_tgv.py 用
`solver._compute_local_time_step()`（外加针对 TGV 特定 MU 值的
monkeypatch）动态算出的自适应步长远小于手写脚本瞎猜的 0.001，手写
脚本用的固定步长本身就不满足这个偏大物理粘度（TGV 故意调大 MU 到
1.8375 以在粗网格上看到有意义的粘性衰减）下的显式粘性稳定性条件，
与人工粘性是否启用无关——改用真实 fixture 的动态步长后，人工粘性
启用/关闭两种情况都稳定。这个教训被记录进
core/fr_operators/artificial_viscosity.py 与 solver.py 的 CFL/
`_get_turbulent_viscosity_field` 文档（同一次调查确实也发现并修复了
一个真实 bug：CFL 步长计算必须与粘性残差看到同一个 `mu_t_field`，
否则会有由粗心引入的真实不一致——但这不是本次"发散"的根因，根因是
复现脚本本身的 dt）。
"""

import numpy as np

from tests.validation._channel_mesh import build_face_exact_ghost_provider, build_channel_mesh_prism
from tests.validation.test_couette import _build_couette_solver
from tests.validation.test_tgv import N_STEPS, _build_tgv_solver, _kinetic_energy, _set_tgv_ic

from autoflowcfd.core.fr_solver import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme


def test_couette_stable_from_wrong_ic_with_artificial_viscosity():
    """复用 test_couette.py::test_couette_prism_stable_from_wrong_ic 的
    设置（明确错误的半速初场，800 步），只是启用人工粘性——必须仍然
    全程数值稳定。
    """
    solver, mesh, H, U_wall, Lx, rho_inf, p_inf = _build_couette_solver()
    solver.artificial_viscosity_enabled = True
    solver.artificial_viscosity_alpha = 1.0

    gamma = 1.4
    y = mesh.sps_coords[:, :, 1]
    u0_wrong = 0.5 * U_wall * y / H
    e0 = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * u0_wrong**2
    solver.state.U[:, :, 1] = rho_inf * u0_wrong
    solver.state.U[:, :, 4] = rho_inf * e0
    solver.state._update_primitives()

    n_iter = 800
    for i in range(n_iter):
        solver.step(1e-6)
        assert np.all(np.isfinite(solver.state.U)), f"solution diverged (NaN/Inf) at iter {i}"


def test_tgv_stable_and_still_dissipative_with_artificial_viscosity():
    """复用 test_tgv.py 的真实 fixture（含针对 TGV 特定 MU 的 monkeypatch
    与动态 CFL 步长），启用人工粘性后：(a) 150 步全程数值稳定；(b) 动能
    仍然呈现真实的粘性衰减签名（不要求与关闭时的衰减比例完全一致——
    人工粘性额外耗散一些能量是预期行为，只要求量级合理、仍然衰减而不是
    异常增长或几乎不变）。
    """
    solver, mesh = _build_tgv_solver()
    _set_tgv_ic(solver, mesh)
    solver.artificial_viscosity_enabled = True
    solver.artificial_viscosity_alpha = 1.0

    ke0 = _kinetic_energy(solver)
    ke_history = [ke0]
    for i in range(N_STEPS):
        dt_this = float(solver._compute_local_time_step()[0, 0])
        solver.step(dt_this)
        assert np.all(np.isfinite(solver.state.U)), f"solution diverged (NaN/Inf) at global step {i}"
        ke_history.append(_kinetic_energy(solver))

    ke_ratio = ke_history[-1] / ke0
    # 关闭人工粘性时实测 KE/KE0≈0.678（见 test_tgv.py 模块文档）；启用后
    # 额外耗散预期让比值更小，但不应该崩溃到接近 0（那意味着人工粘性
    # 强度过大，把真实流动物理也一并抹掉）或反而增长（意味着传感器/
    # 耗散方向有 bug）。留足安全边际：[0.05, 0.75] 区间。
    assert 0.05 < ke_ratio < 0.75, f"unexpected KE ratio with artificial viscosity: {ke_ratio:.4f}"


def test_artificial_viscosity_stays_off_by_default():
    """回归防线：不显式启用时，`artificial_viscosity_enabled` 必须默认
    为 False，且完全不影响粘性残差数值——确保这个新增能力真正是
    opt-in，不会意外改变任何现有求解路径的行为。
    """
    solver, mesh, H, U_wall, Lx, rho_inf, p_inf = _build_couette_solver()
    assert solver.artificial_viscosity_enabled is False

    y = mesh.sps_coords[:, :, 1]
    gamma = 1.4
    u0 = U_wall * y / H
    e0 = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * u0**2
    solver.state.U[:, :, 1] = rho_inf * u0
    solver.state.U[:, :, 4] = rho_inf * e0
    solver.state._update_primitives()

    res_before = solver.compute_viscous_residual()

    # 显式启用但传感器在这个光滑解析场上应当输出零人工粘性——两次
    # 结果必须逐位相同,进一步确认"关闭"与"启用但传感器不触发"是
    # 等价的（而不是关闭状态本身有特殊代码路径导致行为分叉）。
    solver.artificial_viscosity_enabled = True
    res_after = solver.compute_viscous_residual()
    np.testing.assert_array_equal(res_before, res_after)
