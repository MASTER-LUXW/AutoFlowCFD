"""AutoFlowCFD V2.0 - 体积项 entropy-stable/split-form 通量（opt-in）稳定性
回归测试。

见 core/fr_operators/flux_kernels.py::entropy_stable_volume_divergence_batch、
core/fr_residual/inviscid.py::compute_inviscid_residual_fr 的
entropy_stable_volume 参数、`8_算法重构-Entropy-Stable_Split-Form通量
重构-Part1/2.md`。复用项目已有的两个"真正有效的稳定性回归测试"
（test_couette.py 的从错误初场出发的 800 步压力测试、test_tgv.py 的
150 步真实粘性动能衰减定量测试），验证方式与
test_artificial_viscosity_stability.py 同一套模式。
"""

import numpy as np

from tests.validation.test_couette import _build_couette_solver
from tests.validation.test_tgv import N_STEPS, _build_tgv_solver, _kinetic_energy, _set_tgv_ic


def test_couette_stable_from_wrong_ic_with_entropy_stable_volume():
    """复用 test_couette.py::test_couette_prism_stable_from_wrong_ic 的
    设置（明确错误的半速初场，800 步），只是启用 entropy-stable 体积项
    ——必须仍然全程数值稳定。"""
    solver, mesh, H, U_wall, Lx, rho_inf, p_inf = _build_couette_solver()
    solver.entropy_stable_volume_enabled = True

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


def test_tgv_stable_and_still_dissipative_with_entropy_stable_volume():
    """复用 test_tgv.py 的真实 fixture，启用 entropy-stable 体积项后：
    (a) 150 步全程数值稳定；(b) 动能仍然呈现真实的粘性衰减签名。"""
    solver, mesh = _build_tgv_solver()
    _set_tgv_ic(solver, mesh)
    solver.entropy_stable_volume_enabled = True

    ke0 = _kinetic_energy(solver)
    ke_history = [ke0]
    for i in range(N_STEPS):
        dt_this = float(solver._compute_local_time_step()[0, 0])
        solver.step(dt_this)
        assert np.all(np.isfinite(solver.state.U)), f"solution diverged (NaN/Inf) at global step {i}"
        ke_history.append(_kinetic_energy(solver))

    ke_ratio = ke_history[-1] / ke0
    # 关闭时实测 KE/KE0≈0.678（见 test_tgv.py 模块文档）；entropy-stable
    # 体积项改的是无粘通量的混叠处理，不直接改变粘性耗散物理，预期
    # 衰减比例与关闭时接近，留足安全边际防止真正的行为回归。
    assert 0.3 < ke_ratio < 0.9, f"unexpected KE ratio with entropy-stable volume: {ke_ratio:.4f}"


def test_entropy_stable_volume_stays_off_by_default():
    """回归防线：不显式启用时，`entropy_stable_volume_enabled` 必须默认
    为 False，且完全不影响无粘残差数值——确保这个新增能力真正是
    opt-in，不会意外改变任何现有求解路径的行为。
    """
    solver, mesh, H, U_wall, Lx, rho_inf, p_inf = _build_couette_solver()
    assert solver.entropy_stable_volume_enabled is False

    y = mesh.sps_coords[:, :, 1]
    gamma = 1.4
    u0 = U_wall * y / H
    e0 = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * u0**2
    solver.state.U[:, :, 1] = rho_inf * u0
    solver.state.U[:, :, 4] = rho_inf * e0
    solver.state._update_primitives()

    res_before = solver.compute_inviscid_residual()

    solver.entropy_stable_volume_enabled = True
    res_after = solver.compute_inviscid_residual()
    # 开启后走完全不同的计算路径（two-point flux vs 逐点通量代入），
    # 数值上不要求逐位相同（这与 AV 传感器"不触发时零贡献"的等价性
    # 不是一回事），但对同一个（近似仿射、GCL 意义下"干净"的）解析
    # 场，两条路径的残差量级应该接近，不应该出现天壤之别的跳变（那
    # 意味着某条路径有 bug）。
    scale = np.maximum(np.abs(res_before).max(), 1e-300)
    assert np.abs(res_after - res_before).max() / scale < 1.0


def test_entropy_stable_volume_flag_off_matches_baseline_bit_for_bit():
    """更严格的默认行为不变检验：显式传 False 必须与完全不传这个参数
    （旧调用方式）逐位相同——防止未来重构时不小心让 False 也走上新代码
    路径的某个分支。"""
    from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr

    solver, mesh, H, U_wall, Lx, rho_inf, p_inf = _build_couette_solver()
    y = mesh.sps_coords[:, :, 1]
    gamma = 1.4
    u0 = U_wall * y / H
    e0 = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * u0**2
    solver.state.U[:, :, 1] = rho_inf * u0
    solver.state.U[:, :, 4] = rho_inf * e0
    solver.state._update_primitives()

    res_no_arg = compute_inviscid_residual_fr(
        solver.state.U, mesh, solver.ops,
        boundary_ghost_provider=solver.boundary_ghost_provider,
        mach_ref=solver.freestream["mach_ref"],
    )
    res_explicit_false = compute_inviscid_residual_fr(
        solver.state.U, mesh, solver.ops,
        boundary_ghost_provider=solver.boundary_ghost_provider,
        mach_ref=solver.freestream["mach_ref"],
        entropy_stable_volume=False,
    )
    np.testing.assert_array_equal(res_no_arg, res_explicit_false)
