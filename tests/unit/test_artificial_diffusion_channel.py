"""问题单元人工粘性的施加通道：全部守恒变量的拉普拉斯 `div(nu grad U_k)`。

施加形式与判据的来由见 `core/fr_operators/artificial_viscosity/entropy_viscosity.py`
模块文档（2026-10-01 从"质量扩散 + 把系数叠进 mu_t"改为五个守恒量同一个 nu 的
拉普拉斯：旧形式质量扩散不带走动量与焓，plate_demo P1 上造出冷点并扩散）。

判据（每条都要有判别力）：

1. 启用 + 非均匀熵场时，粘性残差的质量分量非零（旧的 mu_t 通道质量分量恒为 0）；
2. 均匀流场下该项到机器精度为零（自由流保持）；
3. 结构性：第 k 个分量恰为 `compute_scalar_diffusion_residual(U_k, nu)`（复用湍流
   输运已验证的标量扩散装配）；
4. 关闭时粘性残差与"没有这段代码"逐位相同；
5. 耗散性：`<d_rho, div(nu grad d_rho)>_detJ < 0`，两档去混叠开关下都成立。
   全域守恒（质量矩阵下积分变化为零）见 `test_troubled_cell_artificial_viscosity.py`。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.solver import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from tests.validation._channel_mesh import (
    build_channel_mesh_prism, build_face_exact_ghost_provider)

H, LX, LZ = 1.0e-2, 2.0e-2, 2.5e-3
NX, NY, NZ = 4, 4, 1
RHO, P, U_INF = 1.225, 101325.0, 30.0
GAMMA = 1.4


def _build(av_enabled, alpha=1.0):
    mesh = build_channel_mesh_prism(2, nx=NX, ny=NY, nz=NZ, Lx=LX, H=H, Lz=LZ)
    bc = {
        "wall_bottom": {"type": "SYMMETRY"}, "wall_top": {"type": "SYMMETRY"},
        "z_min": {"type": "SYMMETRY"}, "z_max": {"type": "SYMMETRY"},
        "x_min": {"type": "OUTLET", "p_outlet": P},
        "x_max": {"type": "OUTLET", "p_outlet": P},
    }
    solver = FRSolver(
        mesh=mesh, order=2, turb_model_name="NONE",
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        rho_inf=RHO, vel_inf=U_INF, p_inf=P, mu_molecular=1.8e-5,
        bc_overrides=bc, n_threads=1,
        artificial_viscosity_enabled=av_enabled,
        artificial_viscosity_alpha=alpha,
    )
    solver.order_continuation_enabled = False
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    return solver, mesh


def _set_uniform(solver):
    U = solver.state.U
    U[..., 0] = RHO
    U[..., 1] = RHO * U_INF
    U[..., 2] = 0.0
    U[..., 3] = 0.0
    U[..., 4] = P / (GAMMA - 1.0) + 0.5 * RHO * U_INF ** 2
    solver.state._update_primitives()


def _set_density_bump(solver, mesh, amp=0.1, seed=3):
    """自由来流 + **SP 级高频**密度扰动（动量/压力保持来流值不变）。

    等压下的密度扰动就是熵扰动（s = ln p - gamma ln rho），沿流向的单元内熵梯度
    让熵残差判据触发；速度均匀，粘性应力为零，质量分量只来自人工扩散。
    用固定种子保证可复现。
    """
    _set_uniform(solver)
    rng = np.random.default_rng(seed)
    pert = amp * RHO * rng.standard_normal(solver.state.U.shape[:2])
    U = solver.state.U
    U[..., 0] = RHO + pert
    # 保持速度与压力不变（动量/能量随密度同步调整）
    U[..., 1] = U[..., 0] * U_INF
    U[..., 4] = P / (GAMMA - 1.0) + 0.5 * U[..., 0] * U_INF ** 2
    solver.state._update_primitives()
    return pert


def _diff(solver):
    """当前状态上的人工扩散项（系数按当前状态求，与 step() 按步冻结的是同一个函数）。"""
    return solver._artificial_diffusion_residual(solver.compute_artificial_diffusivity_field())


class TestChannelIsActive:
    def test_mass_component_nonzero_with_av_and_entropy_bump(self):
        solver, mesh = _build(av_enabled=True)
        _set_density_bump(solver, mesh)
        res = solver.compute_viscous_residual()
        assert np.abs(res[..., 0]).max() > 0.0, "启用且熵场非均匀，质量分量仍恒为 0——通道没有生效"

    def test_mass_component_zero_without_av(self):
        solver, mesh = _build(av_enabled=False)
        _set_density_bump(solver, mesh)
        res = solver.compute_viscous_residual()
        np.testing.assert_array_equal(res[..., 0], np.zeros_like(res[..., 0]))

    def test_av_off_residual_bitwise_unchanged_by_this_feature(self):
        solver_a, mesh = _build(av_enabled=False)
        _set_density_bump(solver_a, mesh)
        res_a = solver_a.compute_viscous_residual()

        solver_b, mesh_b = _build(av_enabled=False)
        _set_density_bump(solver_b, mesh_b)
        solver_b._artificial_diffusion_residual = lambda nu: np.zeros_like(solver_b.state.U)
        res_b = solver_b.compute_viscous_residual()
        np.testing.assert_array_equal(res_a, res_b)


class TestFreeStreamAndStructure:
    def test_uniform_flow_term_is_zero(self):
        solver, mesh = _build(av_enabled=True)
        _set_uniform(solver)
        nu = solver.compute_artificial_diffusivity_field()
        nu_test = np.full_like(nu, 1e-3)    # 判据在均匀流上为零；强行给系数检验算子本身
        res = solver._artificial_diffusion_residual(nu_test)
        # 自然量级 nu |U| / h_min^2；常数场的离散梯度只剩舍入（实测相对 3e-13）
        h_min = min(LX / NX, H / NY, LZ / NZ)
        scale = 1e-3 * np.abs(solver.state.U).max(axis=(0, 1)) / h_min ** 2
        assert np.all(np.abs(res).max(axis=(0, 1)) <= 1e-10 * scale), "均匀流场下人工扩散不为零——自由流保持被破坏"
        assert np.all(nu == 0.0), "均匀流上判据应为零"

    def test_each_component_is_the_established_scalar_diffusion(self):
        from autoflowcfd.core.turbulence.transport import compute_scalar_diffusion_residual

        solver, mesh = _build(av_enabled=True)
        _set_density_bump(solver, mesh)
        nu = solver.compute_artificial_diffusivity_field()
        assert np.abs(nu).max() > 0.0, "本用例需要判据真的触发"
        got = solver._artificial_diffusion_residual(nu)
        for k in range(5):
            expect = compute_scalar_diffusion_residual(
                np.ascontiguousarray(solver.state.U[..., k]), np.ascontiguousarray(nu), mesh, solver.ops)
            np.testing.assert_array_equal(got[..., k], expect, err_msg=f"分量 {k}")


class TestDiffusionIsDissipative:
    @pytest.mark.parametrize("overint", ["off", "on"])
    def test_sign_is_negative_under_both_dealiasing_modes(self, overint, monkeypatch):
        monkeypatch.setenv("AFCFD_TURB_OVERINT", overint)
        solver, mesh = _build(av_enabled=True)
        pert = _set_density_bump(solver, mesh)
        mass_res = _diff(solver)[..., 0]
        n_cells, n_sps = mass_res.shape
        det_jacs = np.abs(mesh.jacobians["det_jacs"].reshape(n_cells, n_sps))
        quad_form = float(np.sum(pert * mass_res * det_jacs))
        norm = float(np.sum(np.abs(pert) * np.abs(mass_res) * det_jacs))
        assert norm > 0.0, "本用例需要该项确实非零才有判别力"
        assert quad_form < 0.0, (
            f"AFCFD_TURB_OVERINT={overint} 下二次型 {quad_form:.6e} >= 0——"
            f"算子不是耗散的（反扩散会指数放大高频模态）")
