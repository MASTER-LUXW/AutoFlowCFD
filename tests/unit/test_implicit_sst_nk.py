# -*- coding: utf-8 -*-
"""隐式稳态（Newton-Krylov）下的 SST：来流条件、隐式湍流残差、耦合收敛。

2026-09-25 同一批里查出并修掉的三件事，各有判据：

1. **k/omega 没有来流条件**（`transport/convection.py` 的
   `open_boundary_face`）：此前所有非壁面边界一律零梯度，来流湍流只存在
   于初场。判据：均匀流、全场 `k = 0.5*k_inf` 时，来流边界单元必须收到
   正的 `dk/dt`（被来流值补给），且 `k = k_inf` 时来流单元的输运残差为零。
2. **隐式湍流残差**（`fr_solver/turbulence/implicit.py::TurbulenceResidual`）：
   壁面单元 omega 行是强约束 `beta1*omega_t*(omega - omega_t)`；求值不得
   把试探场泄漏进模型状态。
3. **耦合收敛**：棱柱通道冲击启动，NK + SER + 块 Jacobi + 隐式 k-omega，
   平均流与湍流残差都必须降多个量级（修复前湍流残差停在 1.3e4）。
"""

import numpy as np
import pytest

from tests.validation._channel_mesh import (
    build_channel_mesh_prism,
    build_face_exact_ghost_provider,
)

RHO, U, P, GAMMA = 1.225, 30.0, 101325.0, 1.4
LX, H, LZ = 0.4, 0.1, 0.08


def _channel_solver(scheme):
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(1, nx=3, ny=4, nz=2, Lx=LX, H=H, Lz=LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    bc["wall_bottom"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    bc["wall_top"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U, 0.0, 0.0, P]}
    solver = FRSolver(mesh=mesh, order=1, turb_model_name="SST", n_vars=7,
                      time_scheme=scheme, rho_inf=RHO, vel_inf=U, p_inf=P,
                      mu_molecular=1.8e-5, bc_overrides=bc)
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    Ua = np.asarray(solver.state.U).copy()
    Ua[..., 0] = RHO
    Ua[..., 1] = RHO * U
    Ua[..., 2:4] = 0.0
    Ua[..., 4] = P / (GAMMA - 1.0) + 0.5 * RHO * U ** 2
    solver.state.U = np.ascontiguousarray(Ua)
    solver.state._update_primitives()
    y = np.asarray(mesh.sps_coords)[..., 1]
    solver.wall_distance = np.ascontiguousarray(np.maximum(np.minimum(y, H - y), 1e-12))
    return solver


@pytest.fixture(scope="module")
def nk_solver():
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    return _channel_solver(TimeIntegrationScheme.NEWTON_KRYLOV)


def _inflow_cells(solver):
    """拥有来流边界面（x_min，外法向 -x）的核心单元。贴壁单元另有 k=0 的
    壁面条件、那里的输运本来就不为零，不能用来判来流条件；棱柱是三角化的，
    按形心取"第一列"会混进不碰 x_min 的单元，所以按边界面本身取。"""
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

    ffg = get_flat_face_geometry(solver.mesh, solver.ops)
    bnd = np.asarray(ffg.is_boundary) & (np.asarray(ffg.true_normal)[:, 0, 0] < -0.99)
    cells = np.unique(np.asarray(ffg.owner_cell)[bnd])
    c = np.asarray(solver.mesh.sps_coords).mean(axis=1)
    core = np.minimum(c[cells, 1], H - c[cells, 1]) > H / 4
    return cells[core]


class TestInflowCondition:
    def test_depleted_inflow_cells_are_replenished(self, nk_solver):
        from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual

        t = nk_solver.turb_model
        saved = t.k_field.copy()
        try:
            t.k_field = np.full_like(saved, 0.5 * t.k_inf)
            dk, _ = compute_turbulence_transport_residual(nk_solver)
        finally:
            t.k_field = saved
        inflow = _inflow_cells(nk_solver)
        assert np.all(dk[inflow].mean(axis=1) > 0.0), "来流单元必须被来流值补给（此前零梯度下恒为 0）"

    def test_freestream_state_is_preserved_at_inflow(self, nk_solver):
        from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual

        t = nk_solver.turb_model
        saved = (t.k_field.copy(), t.omega_field.copy())
        try:
            t.k_field = np.full_like(saved[0], t.k_inf)
            t.omega_field = np.full_like(saved[1], t.omega_inf)
            dk_ref, dw_ref = compute_turbulence_transport_residual(nk_solver)
            t.k_field = np.full_like(saved[0], 0.5 * t.k_inf)
            dk_low, _ = compute_turbulence_transport_residual(nk_solver)
        finally:
            t.k_field, t.omega_field = saved
        inflow = _inflow_cells(nk_solver)
        # 来流值 = 场值时来流面不产生跳变：来流单元的输运远小于"场值偏离来流"时
        assert np.abs(dk_ref[inflow]).max() < 1e-6 * np.abs(dk_low[inflow]).max()


class TestTurbulenceResidual:
    def test_wall_rows_are_strong_constraint_and_state_is_restored(self, nk_solver):
        from autoflowcfd.core.fr_solver.turbulence.implicit import (
            CpuTurbulenceBackend,
            TurbulenceResidual,
        )

        t = nk_solver.turb_model
        backend = CpuTurbulenceBackend(nk_solver)
        backend.prepare()
        before = {a: np.array(getattr(t, a), copy=True) for a in ("k_field", "omega_field", "nu_t")}
        res = TurbulenceResidual(backend)
        rng = np.random.default_rng(0)
        kw = np.stack([t.k_field.ravel(), t.omega_field.ravel()], axis=1)
        kw = kw * (1.0 + 0.1 * rng.standard_normal(kw.shape))
        r = res(kw)

        assert res._wall_rows.size > 0
        want = t.beta1 * res._wall_target * (kw[res._wall_rows, 1] - res._wall_target)
        np.testing.assert_allclose(r[res._wall_rows, 1], want, rtol=1e-14)
        assert np.all(r[~res._real_rows] == 0.0), "零填充槽位不参与 Newton"
        for a, v in before.items():
            np.testing.assert_array_equal(getattr(t, a), v, err_msg=f"试探求值泄漏进了 {a}")


def test_nk_sst_channel_converges_coupled():
    """冲击启动 140 步：平均流与湍流残差都降多个量级，k/omega 不贴任何下限。

    修复前（同一算例）：平均流降 5 个量级之后湍流残差停在 1.3e4、每步
    Newton 被拒绝，核心区 omega 贴在 0.1*omega_inf 的下限上。

    步数预算 120 -> 140（2026-09-25）：物理性限幅改为逐单元、并且对增长也做
    约束（单步最多加倍，与 SU2 `MAX_UPDATE_SST` 同一量级）之后，湍流发展暂态
    （k 从来流值长到剪切层值）在本算例上实测从第 78 步推迟到第 96 步收尾，
    其后平均流每步降一个量级，与此前相同。
    """
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    s = _channel_solver(TimeIntegrationScheme.NEWTON_KRYLOV)
    mean, turb = [], []
    for _ in range(140):
        s.step(2.0e-7)
        mean.append(s._newton_last_info["res_norm"])
        turb.append(s._newton_turb_state["last_info"]["res_norm"])
    assert mean[-1] < 1e-6 * max(mean), (max(mean), mean[-1])
    assert turb[-1] < 1e-6 * max(turb), (max(turb), turb[-1])
    t = s.turb_model
    assert t.k_field.min() > 10.0 * 1e-3 * t.k_inf, "k 贴在正性下限上"
    assert t.omega_field.min() > 1.5 * 0.1 * t.omega_inf, "omega 贴在 realizability 下限上"
