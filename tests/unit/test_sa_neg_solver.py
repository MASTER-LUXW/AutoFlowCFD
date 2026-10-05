# -*- coding: utf-8 -*-
"""SA-neg 在 CPU 单机求解器上的集成：壁面解点、隐式 NK 收敛（P1~P3、棱柱与四面体）、显式与双时间步、
Order Continuation、checkpoint 往返。

算例与 SST 的隐式测试同一个槽道（`test_implicit_sst_nk.py`），壁距经生产入口
`apply_wall_distance_source` 由精确来源施加（`channel_wall_source`）。原生四面体的解点含顶点、
棱点与面点，会落在壁面上（`d == 0`）：SA 在那里施加强 Dirichlet `nu_tilde = 0`
（`core/turbulence/sa/model.py` 模块文档）。此前这些解点上耗散项 `cw1 fw (nu_tilde / d)^2` 按
截断后的壁距求值，残差 3e15、比其余解点大 12 个数量级，差分 Jacobian 被舍入淹没。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence import apply_wall_distance_source
from autoflowcfd.fr.native_padding import real_row_mask
from tests.validation._channel_mesh import (
    build_channel_mesh, build_channel_mesh_prism, build_face_exact_ghost_provider, channel_wall_source,
)

RHO, U, P = 1.225, 30.0, 101325.0
LX, H, LZ = 0.4, 0.1, 0.08


def _sa_channel(kind, order, scheme=None, order_continuation=False):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    mesh = (build_channel_mesh if kind == "tet" else build_channel_mesh_prism)(order, 3, 4, 2, LX, H, LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    for n in ("wall_bottom", "wall_top"):
        bc[n] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U, 0.0, 0.0, P]}
    s = FRSolver(mesh=mesh, order=order, turb_model_name="SA",
                 time_scheme=scheme if scheme is not None else TimeIntegrationScheme.NEWTON_KRYLOV,
                 rho_inf=RHO, vel_inf=U, p_inf=P, mu_molecular=1.8e-5, bc_overrides=bc)
    s.order_continuation_enabled = order_continuation
    provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    s.boundary_ghost_provider = provider
    s._build_boundary_ghost_provider = lambda _bc: build_face_exact_ghost_provider(s.mesh, LX, H, LZ, bc)
    apply_wall_distance_source(s, channel_wall_source(LX, H, LZ))
    return s


def _real_wall_points(s):
    """真实解点中位于壁面上的（零填充槽位不计）。"""
    real = real_row_mask(np.arange(s.mesh.n_cells) < s.mesh.n_prism_cells, s.mesh.n_sps_per_cell,
                         int(s.mesh.order)).reshape(s.mesh.n_cells, -1)
    return real & (np.asarray(s.wall_distance) == 0.0), real


def _rates(s):
    from autoflowcfd.core.fr_solver.turbulence.source import (
        evaluate_turbulence_rates, prepare_turbulence_inputs,
    )

    return evaluate_turbulence_rates(s, *prepare_turbulence_inputs(s), apply_des=False).total()[0]


def _run_nk(s, n_max, drop=1e-9):
    mean, turb = [], []
    for _ in range(n_max):
        s.step(2.0e-7)
        mean.append(s._newton_last_info["res_norm_mean"])
        turb.append(s._newton_last_info["res_norm_turbulence"])
        if mean[-1] < drop * max(mean) and turb[-1] < drop * max(turb):
            break
    return mean, turb


class TestWallPoints:
    def test_wall_points_pinned_and_rates_regular(self):
        s = _sa_channel("tet", 2)
        wall, real = _real_wall_points(s)
        assert wall.sum() > 0, "原生四面体 P2 的顶点/棱点应有落在壁面上的"
        nt = s.turb_model.nu_tilde_field
        assert np.all(nt[wall] == 0.0)
        rate = _rates(s)
        assert np.all(rate[wall] == 0.0)
        off = real & ~wall
        # 修复前壁面解点的速率 ~3e15，比其余解点大 12 个数量级；现在全场同一量级
        assert np.abs(rate[off]).max() < 1e4 * np.median(np.abs(rate[off]))

    def test_prism_solution_points_never_lie_on_the_wall(self):
        s = _sa_channel("prism", 2)
        wall, _ = _real_wall_points(s)
        assert wall.sum() == 0


@pytest.mark.parametrize("kind,order,n_max", [("prism", 1, 150), ("prism", 3, 100), ("tet", 3, 130)])
def test_nk_converges(kind, order, n_max):
    """冲击启动：平均流与 SA 残差都降 6 个量级以上（实测 P1 棱柱 91 步、P3 棱柱 58 步、P3 四面体
    82 步降 1e9），壁面解点始终为 0，涡粘处处非负。"""
    s = _sa_channel(kind, order)
    mean, turb = _run_nk(s, n_max)
    assert mean[-1] < 1e-6 * max(mean), (max(mean), mean[-1])
    assert turb[-1] < 1e-6 * max(turb), (max(turb), turb[-1])
    wall, _ = _real_wall_points(s)
    assert np.all(s.turb_model.nu_tilde_field[wall] == 0.0)
    assert np.asarray(s.turb_model.nu_t).min() >= 0.0


def test_explicit_steps_keep_the_converged_steady_state():
    """NK 收敛的定常解在显式 SSP-RK3 上推进应保持不变（同一个离散方程的不动点）。"""
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    nk = _sa_channel("tet", 1)
    mean, _ = _run_nk(nk, 150, drop=1e-10)
    assert mean[-1] < 1e-8 * max(mean)
    ex = _sa_channel("tet", 1, scheme=TimeIntegrationScheme.SSP_RK3)
    # 产生项斜坡（前 50 步 production_factor 从 0 线性增到 1）在 NK 一侧早已走完；新构造的显式
    # 求解器从 0 起步，第一步的离散方程不同，收敛解不是它的不动点。对齐到斜坡完成后再比较。
    ex._turb_ramp_step = ex._turb_production_ramp_steps
    ex.state.U = np.array(nk.state.U, copy=True)
    ex.state._update_primitives()
    ex.turb_model.nu_tilde_field = np.array(nk.turb_model.nu_tilde_field, copy=True)
    nt0 = ex.turb_model.nu_tilde_field.copy()
    for _ in range(5):
        ex.step(1e-6)
    _, real = _real_wall_points(ex)
    change = np.abs(ex.turb_model.nu_tilde_field - nt0)[real].max() / np.abs(nt0[real]).max()
    assert change < 1e-6, f"显式推进改动了收敛解的 nu_tilde（相对 {change:.2e}）"
    wall, _ = _real_wall_points(ex)
    assert np.all(ex.turb_model.nu_tilde_field[wall] == 0.0)


def test_dual_time_steps_are_finite_and_keep_wall_points():
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    s = _sa_channel("tet", 1, scheme=TimeIntegrationScheme.DUAL_TIME)
    nt0 = s.turb_model.nu_tilde_field.copy()
    for _ in range(3):
        s.step(1e-5)
    nt = s.turb_model.nu_tilde_field
    assert np.all(np.isfinite(nt)) and np.all(np.isfinite(s.state.U))
    assert not np.array_equal(nt, nt0), "双时间步应推进 SA 场"
    wall, _ = _real_wall_points(s)
    assert np.all(nt[wall] == 0.0)


def test_order_continuation_climbs_to_target_and_repins_wall_points():
    """生产 Order Continuation（P0 -> P1 -> P2）：每次换阶由来源重查壁距、壁面解点重新置 0，最终
    阶数上收敛。"""
    s = _sa_channel("tet", 2, order_continuation=True)
    result = s.solve(max_iter=400, dt=2.0e-7, tol=1e-6)
    assert int(s.current_order) == 2
    assert s.turb_model.nu_tilde_field.shape == (s.mesh.n_cells, s.mesh.n_sps_per_cell)
    wall, _ = _real_wall_points(s)
    assert wall.sum() > 0 and np.all(s.turb_model.nu_tilde_field[wall] == 0.0)
    assert result.converged, f"P2 阶段未收敛（{result.iterations} 步，残差 {result.residual_history[-1]:.3e}）"


class TestCheckpoint:
    def test_roundtrip_restores_nu_tilde_and_pins_wall_points(self, tmp_path):
        import h5py

        from autoflowcfd.cli.solve.checkpoint_io import write_checkpoint
        from autoflowcfd.cli.solve.checkpoint_io.restore import restore_solver_state_from_fields

        src = _sa_channel("tet", 1)
        _run_nk(src, 20)
        write_checkpoint(src, str(tmp_path), 20, "volume.nas", order=1, turbulence_model="sa",
                         backend="cpu", quiet=True)
        with h5py.File(tmp_path / "checkpoints" / "checkpoint_iter_000020.h5", "r") as f:
            fields = {k: f["solution"][k][:] for k in f["solution"]}
            metadata = dict(f.attrs)
        assert "nu_tilde_field" in fields and "k_field" not in fields
        np.testing.assert_array_equal(fields["nu_tilde_field"], src.turb_model.nu_tilde_field)

        dst = _sa_channel("tet", 1)
        wall, _ = _real_wall_points(dst)
        tampered = dict(fields)
        tampered["nu_tilde_field"] = np.where(wall, 1.0, fields["nu_tilde_field"])  # 壁面解点被写成非零
        restore_solver_state_from_fields(dst, tampered, metadata)
        np.testing.assert_array_equal(dst.turb_model.nu_tilde_field[~wall], fields["nu_tilde_field"][~wall])
        assert np.all(dst.turb_model.nu_tilde_field[wall] == 0.0)
