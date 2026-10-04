# -*- coding: utf-8 -*-
"""隐式稳态（Newton-Krylov）下的 SST：来流条件、隐式湍流残差、耦合收敛。

2026-09-25 同一批里查出并修掉的三件事，各有判据：

1. **k/omega 没有来流条件**（`transport/convection.py` 的
   `open_boundary_face`）：此前所有非壁面边界一律零梯度，来流湍流只存在
   于初场。判据：均匀流、全场 `k = 0.5*k_inf` 时，来流边界单元必须收到
   正的 `dk/dt`（被来流值补给），且 `k = k_inf` 时来流单元的输运残差为零。
2. **耦合残差**（`time_integration/implicit/coupled_step.py::CoupledResidual`）：
   湍流列在真实行上就是输运方程本身（壁面 omega 只由扩散残差的面 Dirichlet 施加，
   2026-09-30 删除了壁面单元全部解点的强约束行，见 `fr_solver/turbulence/implicit.py`
   "omega 壁面条件"），平均流列就是 `Gamma R`；求值不得把试探场泄漏进求解器状态。
3. **耦合收敛**：棱柱通道冲击启动，NK + SER + 块预处理 + 平均流与 k-omega 紧耦合，
   平均流与湍流残差都必须降多个量级（修复前湍流残差停在 1.3e4）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence import apply_wall_distance_source
from tests.validation._channel_mesh import (
    channel_wall_source,
    build_channel_mesh_prism,
    build_face_exact_ghost_provider,
)

RHO, U, P, GAMMA = 1.225, 30.0, 101325.0, 1.4
LX, H, LZ = 0.4, 0.1, 0.08


def _channel_solver(scheme, order=1):
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(order, nx=3, ny=4, nz=2, Lx=LX, H=H, Lz=LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    bc["wall_bottom"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    bc["wall_top"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U, 0.0, 0.0, P]}
    solver = FRSolver(mesh=mesh, order=order, turb_model_name="SST", n_vars=7,
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
    apply_wall_distance_source(solver, channel_wall_source(LX, H, LZ))
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


def _transport(solver):
    """生产入口（先在当前 k/omega 上求源项、再求输运）给出的输运部分 `(dk, dw)`；
    源项刷新的模型缓存事后恢复，不影响共用夹具的其它测试。"""
    from autoflowcfd.core.fr_solver.turbulence.source import (
        evaluate_turbulence_rates, prepare_turbulence_inputs,
    )

    t = solver.turb_model
    saved = {a: getattr(t, a) for a in t.CACHED_ATTRS if hasattr(t, a)}
    try:
        tk, tw = evaluate_turbulence_rates(solver, *prepare_turbulence_inputs(solver), apply_des=False).transport
    finally:
        for a, v in saved.items():
            setattr(t, a, v)
    return tk, tw


class TestInflowCondition:
    def test_depleted_inflow_cells_are_replenished(self, nk_solver):
        t = nk_solver.turb_model
        saved = t.k_field.copy()
        try:
            t.k_field = np.full_like(saved, 0.5 * t.k_inf)
            dk, _ = _transport(nk_solver)
        finally:
            t.k_field = saved
        inflow = _inflow_cells(nk_solver)
        assert np.all(dk[inflow].mean(axis=1) > 0.0), "来流单元必须被来流值补给（此前零梯度下恒为 0）"

    def test_freestream_state_is_preserved_at_inflow(self, nk_solver):
        t = nk_solver.turb_model
        saved = (t.k_field.copy(), t.omega_field.copy())
        try:
            t.k_field = np.full_like(saved[0], t.k_inf)
            t.omega_field = np.full_like(saved[1], t.omega_inf)
            dk_ref, dw_ref = _transport(nk_solver)
            t.k_field = np.full_like(saved[0], 0.5 * t.k_inf)
            dk_low, _ = _transport(nk_solver)
        finally:
            t.k_field, t.omega_field = saved
        inflow = _inflow_cells(nk_solver)
        # 来流值 = 场值时来流面不产生跳变：来流单元的输运远小于"场值偏离来流"时
        assert np.abs(dk_ref[inflow]).max() < 1e-6 * np.abs(dk_low[inflow]).max()


class TestCoupledResidual:
    def test_residual_is_both_equations_and_state_is_restored(self, nk_solver):
        """真实行上湍流列 `= -(dk/dt, dw/dt)`（含壁面单元，没有被改写的行）、平均流列 `= Gamma R`；
        零填充槽位的湍流列为零；试探求值不泄漏进求解器状态。"""
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.fr_solver.turbulence.implicit import CpuCoupledBackend
        from autoflowcfd.core.time_integration.implicit.coupled_step import CoupledResidual
        from autoflowcfd.core.turbulence.transport import _compute_wall_dirichlet_face_mask
        from autoflowcfd.fr.native_padding import real_row_mask

        s, t = nk_solver, nk_solver.turb_model
        be = CpuCoupledBackend(s)
        before = {"U": np.array(s.state.U, copy=True)}
        before.update({a: np.array(getattr(t, a), copy=True) for a in ("k_field", "omega_field", "nu_t")})
        real = real_row_mask(be.cell_is_prism, be.n_sps, be.order)
        x0 = be.state()
        rng = np.random.default_rng(0)
        x = x0.copy()
        x[:, 0] *= 1.0 + 1e-3 * rng.standard_normal(x.shape[0])
        x[:, 5] *= 1.0 + 0.1 * rng.standard_normal(x.shape[0])
        x[:, 6] += 0.1 * rng.standard_normal(x.shape[0])
        r = CoupledResidual(be, real, x0)(x)
        np.testing.assert_array_equal(s.state.U, before["U"], err_msg="试探求值泄漏进了平均流状态")
        for a, v in ((a, before[a]) for a in ("k_field", "omega_field", "nu_t")):
            np.testing.assert_array_equal(getattr(t, a), v, err_msg=f"试探求值泄漏进了 {a}")

        # 参考：同一试探状态上直接求两个子系统
        snap = be.snapshot()
        try:
            be.set_trial(x)
            be.turb.prepare_inputs()
            rate_k, rate_w = be.turb.rates(apply_des=False)
            r_mean = be.mean_residual(be.trial_mu_t())
        finally:
            be.restore(snap)
        np.testing.assert_allclose(r[real, 5], -rate_k.ravel()[real], rtol=1e-13, atol=0)
        np.testing.assert_allclose(r[real, 6], -rate_w.ravel()[real], rtol=1e-13, atol=0)
        np.testing.assert_allclose(r[:, :5], r_mean, rtol=1e-13, atol=0)
        assert np.all(r[~real, 5:] == 0.0), "零填充槽位不参与 Newton"
        flat = get_flat_face_geometry(s.mesh, s.ops)
        wall_cells = np.unique(flat.owner_cell[_compute_wall_dirichlet_face_mask(s)])
        assert wall_cells.size > 0, "算例里应当有壁面单元（否则上面的判据对壁面行是空的）"


def test_nk_sst_channel_converges_coupled():
    """冲击启动 200 步：平均流与湍流残差都降多个量级，k/omega 不贴任何下限、不触发松弛。

    修复前（同一算例）：平均流降 5 个量级之后湍流残差停在 1.3e4、每步
    Newton 被拒绝，核心区 omega 贴在 0.1*omega_inf 的下限上。

    步数预算 120 -> 140（2026-09-25）：物理性限幅改为逐单元、并且对增长也做
    约束（单步最多加倍，与 SU2 `MAX_UPDATE_SST` 同一量级）之后，湍流发展暂态
    （k 从来流值长到剪切层值）在本算例上实测从第 78 步推迟到第 96 步收尾，
    其后平均流每步降一个量级，与此前相同。

    步数预算 140 -> 200（2026-09-26）：湍流标量输运的两处离散缺陷修掉之后
    （`turbulence/transport/face_frames.py`：neighbor 侧用错通量点顺序、扩散在
    neighbor 侧反扩散），全程逐单元松弛**一次都不触发**（此前第 60~100 步最多
    75% 的单元被松弛），湍流发展暂态平滑单调地走完、第 ~160 步收尾，其后平均流
    与湍流每步降一个量级（第 180 步 4.3e-6 / 1.9e-5）。暂态期平均流残差停在
    50~70、SER 律把 CFL 保持在 130~200，步数由这段物理暂态决定。
    0.1 omega_inf 的安全网在收敛解上处处不激活——下面最后一条断言钉住这一点。
    （2026-09-30 删除壁面单元强约束后收敛更快：55 步，omega 最小 0.88 omega_inf；
    此前 0.229 omega_inf 是强约束钉住的壁面单元目标值。）
    """
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    s = _channel_solver(TimeIntegrationScheme.NEWTON_KRYLOV)
    mean, turb, limited = [], [], []
    for _ in range(200):
        s.step(2.0e-7)
        info = s._newton_last_info
        mean.append(info["res_norm_mean"])
        turb.append(info["res_norm_turbulence"])
        limited.append(info["limited_fraction"])
    assert mean[-1] < 1e-6 * max(mean), (max(mean), mean[-1])
    assert max(limited) == 0.0, f"耦合 Newton 步触发了物理性松弛（最多 {100 * max(limited):.1f}% 的行）"
    assert turb[-1] < 1e-6 * max(turb), (max(turb), turb[-1])
    t = s.turb_model
    assert t.k_field.min() > 10.0 * 1e-3 * t.k_inf, "k 贴在正性下限上"
    assert t.omega_field.min() > 1.5 * 0.1 * t.omega_inf, "omega 贴在 realizability 下限上"


def test_nk_sst_channel_converges_coupled_p3():
    """同一算例 P3：平均流与湍流残差都降多个量级、k 没有失控的负欠冲、涡粘处处非负。

    2026-09-30 之前壁面 owner 单元全部解点的 w 行被强约束到壁面目标值：P3 贴壁
    单元的第一排解点 omega 被钉在 ~3900、生成/耗散比 1.9，k 在第一排长成尖峰并与
    平均流正反馈，在任何 CFL（固定 20 亦然）下都发散（200 步残差降不到 1 个量级、
    k 最小 -139 k_inf）。删除强约束后实测 92 步收敛、残差降 1.7e10。

    2026-10-02：湍流对流体积项改成一致的对流形式后，分离式在这里进入 4 步一周期的
    极限环（块 Gauss-Seidel 外迭代失稳），改为平均流与湍流紧耦合 Newton
    （`time_integration/implicit/coupled_step.py`）。收敛解在出口与壁面交界的贴壁解点上
    有 k 的负欠冲（实测最小 -0.27 = -2 k_inf，约为场内峰值 7.1 的 3.8%）：远场出口的
    幽灵态是均匀来流剖面，与无滑移壁面在角点处不相容。此前 k 全场为正只是因为守恒形式
    的伪源 `-k div_vol(rho u)` 恰好在那里为正。按约定被输运的 k 可以欠冲、realizability
    只作用于模型项求值（`turbulence/sst/bounds.py`），所以这里断言的是"没有失控"（欠冲
    不超过峰值的 5%，-139 k_inf 那类失控远超于此）与"涡粘处处非负"。
    """
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    s = _channel_solver(TimeIntegrationScheme.NEWTON_KRYLOV, order=3)
    mean, turb = [], []
    for _ in range(140):
        s.step(2.0e-7)
        mean.append(s._newton_last_info["res_norm_mean"])
        turb.append(s._newton_last_info["res_norm_turbulence"])
        if mean[-1] < 1e-8 * max(mean) and turb[-1] < 1e-8 * max(turb):
            break
    assert mean[-1] < 1e-6 * max(mean), (max(mean), mean[-1])
    assert turb[-1] < 1e-6 * max(turb), (max(turb), turb[-1])
    k = s.turb_model.k_field
    assert k.min() > -0.05 * k.max(), f"P3 下 k 负欠冲失控：最小 {k.min():.3e}、峰值 {k.max():.3e}"
    assert np.asarray(s.turb_model.nu_t).min() >= 0.0, "涡粘出现负值（realizability 未生效）"


def test_explicit_steps_keep_the_converged_steady_state():
    """隐式 NK 收敛的定常解，拿到显式 SSP-RK3 上推进几步应保持不变（它满足同一个离散方程）。

    2026-10-01 之前显式路径每步把壁面 owner 单元全部解点的 ln(omega) 往壁面目标值
    拉一半：同一个收敛解只做一次就把壁面单元 omega 改动 2.13 倍（P1），即正确的
    离散定常解不是显式路径的不动点（显式槽道 40000 步残差停在 1e4）。删除后壁面
    omega 只由扩散残差的面 Dirichlet 施加，显式与隐式是同一个离散问题。
    """
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.time_integration import TimeIntegrationScheme
    from autoflowcfd.core.turbulence.transport import _compute_wall_dirichlet_face_mask

    nk = _channel_solver(TimeIntegrationScheme.NEWTON_KRYLOV)
    mean = []
    for _ in range(120):
        nk.step(2.0e-7)
        mean.append(nk._newton_last_info["res_norm_mean"])
        if mean[-1] < 1e-9 * max(mean) and nk._newton_last_info["res_norm_turbulence"] < 1e-9:
            break
    assert mean[-1] < 1e-8 * max(mean)

    ex = _channel_solver(TimeIntegrationScheme.SSP_RK3)
    ex.state.U = np.array(nk.state.U, copy=True)
    ex.state._update_primitives()
    ex.turb_model.k_field = np.array(nk.turb_model.k_field, copy=True)
    ex.turb_model.omega_field = np.array(nk.turb_model.omega_field, copy=True)
    flat = get_flat_face_geometry(ex.mesh, ex.ops)
    wall_cells = np.unique(flat.owner_cell[_compute_wall_dirichlet_face_mask(ex)])
    om0 = ex.turb_model.omega_field[wall_cells].copy()
    for _ in range(5):
        ex.step(1e-6)
    change = np.abs(np.log(ex.turb_model.omega_field[wall_cells] / om0)).max()
    assert change < 1e-3, f"显式推进把收敛解的壁面单元 omega 改动了 {np.exp(change):.3f} 倍"
