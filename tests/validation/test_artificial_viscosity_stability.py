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
from tests.validation.test_tgv import (
    N_STEPS, RHO_INF, _analytic_tgv_dissipation, _build_tgv_solver,
    _kinetic_energy, _set_tgv_ic,
)

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
    衰减仍与**解析耗散率**同量级、且不增长。

    ## 判据换过一次（2026-09-19）

    原判据是 `0.05 < KE/KE0 < 0.75`，一个**校准值**，而它的前提是
    "启用人工粘性后会额外耗散、比值应当更小"。这个前提在**光滑**场上是
    错的：人工粘性的传感器（Persson-Peraire 探守恒密度）在光滑解上掩码
    为空，所以它**正确地什么都不做** —— 实测启用 `alpha=1.0` 之后
    `K/K0 = 0.98919717`，与基线**逐位相同**（差 0.000e+00）。

    那个区间此前能通过，是因为基线本身当时是 0.678，而那个 0.678 来自
    两个已经不成立的前提（`FILTER_MODE=legacy` 的人工耗散、以及 P2 四面体
    残差被机制3 整体清零 —— 两条见 `test_tgv.py` 里那段说明）。

    所以本测试现在断言三件**物理必须**的事：
      1. 全程有限（这是"stable"的本义，也是本测试最初的目的）；
      2. 动能不增长；
      3. 净衰减与解析耗散率同量级（人工粘性只能额外耗散，所以上限放宽）。
    并显式钉住"光滑场上它是无操作"这条设计行为 —— 那是可验证的事实，
    不是遗憾。人工粘性真正生效的场景由
    `test_couette_stable_from_wrong_ic_with_artificial_viscosity`（错误
    初场、有真实间断）与单元测试覆盖。
    """
    solver, mesh = _build_tgv_solver()
    _set_tgv_ic(solver, mesh)
    solver.artificial_viscosity_enabled = True
    solver.artificial_viscosity_alpha = 1.0

    ke0 = _kinetic_energy(solver)
    ke_history = [ke0]
    dt_history = []
    for i in range(N_STEPS):
        dt_this = float(solver._compute_local_time_step()[0, 0])
        dt_history.append(dt_this)
        solver.step(dt_this)
        assert np.all(np.isfinite(solver.state.U)), f"solution diverged (NaN/Inf) at global step {i}"
        ke_history.append(_kinetic_energy(solver))

    # 解析判据（见本函数文档）：与 test_tgv.py 同一套，基准取 step 1 以
    # 排除初场在 SPs 上的一次性离散适应。
    eps_ana, k_ana = _analytic_tgv_dissipation()
    ke_base = ke_history[1]
    t_total = sum(dt_history[1:])
    expect_drop = (RHO_INF * eps_ana / k_ana) * t_total
    actual_drop = 1.0 - ke_history[-1] / ke_base
    assert actual_drop > 0.0, (
        f"启用人工粘性后动能净增长 {-actual_drop * 100:+.3f}% —— 人工粘性"
        f"只可能耗散能量，增长意味着传感器/耗散方向有 bug")
    assert 0.1 < actual_drop / expect_drop < 10.0, (
        f"净衰减 {actual_drop * 100:.3f}% 与解析耗散率给出的 "
        f"{expect_drop * 100:.3f}% 相差 {actual_drop / expect_drop:.2f} 倍"
        f"（允许 0.1~10 倍；上限比 test_tgv 宽，因为人工粘性可以额外耗散）")

    # **光滑场上它必须是无操作**（传感器掩码为空）：这是设计行为，
    # 实测与基线逐位相同。钉住它，将来若哪天在光滑场上开始耗散能量，
    # 说明传感器误触发了。
    solver_ref, mesh_ref = _build_tgv_solver()
    _set_tgv_ic(solver_ref, mesh_ref)
    ke0_ref = _kinetic_energy(solver_ref)
    for _ in range(N_STEPS):
        solver_ref.step(float(solver_ref._compute_local_time_step()[0, 0]))
    ratio_ref = _kinetic_energy(solver_ref) / ke0_ref
    ke_ratio = ke_history[-1] / ke0
    assert abs(ke_ratio - ratio_ref) / ratio_ref < 1e-6, (
        f"人工粘性在**光滑** TGV 上不该有可测影响（传感器掩码为空），"
        f"实测启用 {ke_ratio:.8f} vs 关闭 {ratio_ref:.8f}")


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
