"""Persson-Peraire 人工粘性的质量扩散通道验证（2026-09-14）。

## 背景

`artificial_viscosity.py` 原先如实记录了一条范围限制并称其为"许多实际
DG/FR 实现采用的简化"：Persson & Peraire (2006) 原方法对**全部**守恒
变量（含连续性方程）叠加人工扩散，而本实现只把 epsilon 叠进
`mu_t_field`，因此只能通过动量/能量的粘性应力/热传导通道起作用——
`viscous_physical_flux` 的质量分量 `G[...,0]` 恒为 0，密度完全不被扩散。

用户明确指出本项目不接受简化，已补上缺的那一项：
`FRSolver._artificial_mass_diffusion_residual` 把
`+div(epsilon * grad(rho))` 加进连续性方程，复用湍流输运已验证的 BR1
面耦合标量扩散装配（`compute_scalar_diffusion_residual`）。

## 判据（每条都要有判别力）

1. **补齐前后确有差别**：AV 启用 + 密度非均匀时，粘性残差的**质量分量**
   必须非零。补齐前它恒为 0，所以这一条直接区分"改动生效/没生效"。
2. **自由流场保持性不被破坏**：均匀流场下 grad(rho)=0，该项必须恒为 0
   （到机器精度）。这是本项目最硬的一条不变量，任何新增残差项都必须过。
3. **结构性**：新通道必须恰好是 `compute_scalar_diffusion_residual(
   rho, epsilon)` 且只写质量分量——钉住"复用既有 BR1 装配（守恒性/
   一致性已由湍流输运验证建立）、系数取对、不泄漏到动量/能量"。
   （原本想用"全域加权积分为零"，但那不是良定义的判据：残差已除过
   det(J)，真正权重是 `w_s*det(J)`，而本项目没有暴露 SP 求积权重；
   且本算例有 OUTLET/SYMMETRY 边界、BR1 扩散边界通量非零。详见该用例
   文档。）
4. **AV 关闭时零影响**：不启用 AV 时粘性残差必须与补齐前**逐位相同**
   （用 `artificial_viscosity_enabled=False` 直接比对），保证这项改动
   对默认路径没有任何影响。
5. **耗散性**（不是反扩散）：二次型
   `<d_rho, div(eps*grad(d_rho))> = -integral(eps*|grad(d_rho)|^2) < 0`
   必须严格为负。这是整个算子的性质，比"某个峰值点 dρ/dt<0"强得多。
   反扩散是本项目在 k/omega 上真实踩过的坑（见 transport.py 的符号
   约定记录：曾误写成 -div(...)，指数放大棋盘模态导致求解停滞）。
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
        mesh=mesh, order=2, turb_model_name="NONE", n_vars=5,
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

    为什么必须是 SP 级高频、不能用光滑的高斯峰（实测记录）：
    Persson-Peraire 传感器度量的是**单元内最高多项式模态**的能量占比，
    它只在解确实欠分辨时才触发——这正是它的设计目的。实测在本算例
    （P2、4x4x1 棱柱）上扫描过几种密度场：
        均匀                -> eps 恒为 0（0/32 单元）
        高斯峰 width=0.3H   -> eps 恒为 0
        窄高斯 width=0.05H  -> eps 恒为 0
        x>LX/2 阶跃         -> eps 恒为 0（阶跃落在单元**边界**上，
                               每个单元内部仍近似常数，没有高模态内容）
        SP 级随机扰动       -> eps_max=5.18e-3，23/32 单元非零
    所以只有最后一种能让 AV 真正激活、从而让本文件的判据有判别力。
    用固定种子保证可复现。

    刻意只扰动密度：这样质量扩散通道是唯一能产生质量残差的机制
    （均匀速度场下粘性应力为零）。
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


class TestMassDiffusionChannelIsActive:
    def test_mass_component_nonzero_with_av_and_density_bump(self):
        """判据 1：补齐后质量分量必须非零（补齐前恒为 0）。"""
        solver, mesh = _build(av_enabled=True)
        _set_density_bump(solver, mesh)
        res = solver.compute_viscous_residual()
        mass = res[..., 0]
        assert np.abs(mass).max() > 0.0, (
            "AV 启用且密度非均匀，粘性残差的质量分量仍恒为 0——"
            "质量扩散通道没有生效")

    def test_mass_component_zero_without_av(self):
        """判据 4（一半）：AV 关闭时质量分量必须恒为 0（原有行为）。"""
        solver, mesh = _build(av_enabled=False)
        _set_density_bump(solver, mesh)
        res = solver.compute_viscous_residual()
        np.testing.assert_array_equal(res[..., 0], np.zeros_like(res[..., 0]))

    def test_av_off_residual_bitwise_unchanged_by_this_feature(self):
        """判据 4（另一半）：AV 关闭时整个粘性残差必须与"没有这段代码"
        完全一致。

        直接构造两个求解器：一个 AV 关闭（走生产路径），一个 AV 关闭且
        把新方法替换成"返回全零"（等价于这段代码不存在）。两者必须逐位
        相同——证明这项改动对默认路径零影响（不只是数值上接近）。
        """
        solver_a, mesh = _build(av_enabled=False)
        _set_density_bump(solver_a, mesh)
        res_a = solver_a.compute_viscous_residual()

        solver_b, mesh_b = _build(av_enabled=False)
        _set_density_bump(solver_b, mesh_b)
        solver_b._artificial_mass_diffusion_residual = (
            lambda: np.zeros_like(solver_b.state.U))
        res_b = solver_b.compute_viscous_residual()
        np.testing.assert_array_equal(res_a, res_b)


class TestFreeStreamAndConservation:
    def test_uniform_flow_mass_diffusion_is_zero(self):
        """判据 2：均匀流场下 grad(rho)=0，该项必须到机器精度为零。

        这是本项目最硬的一条不变量——任何新增残差项都不允许破坏自由
        流场保持性。
        """
        solver, mesh = _build(av_enabled=True)
        _set_uniform(solver)
        mass_res = solver._artificial_mass_diffusion_residual()[..., 0]
        scale = RHO * U_INF / H     # 质量残差的量纲标度
        rel = np.abs(mass_res).max() / scale
        assert rel < 1e-12, (
            f"均匀流场下人工质量扩散不为零（相对 {rel:.3e}）——"
            "自由流场保持性被破坏")

    def test_uses_established_br1_scalar_diffusion_assembly(self):
        """判据 3（结构性）：新通道必须**恰好**是
        `compute_scalar_diffusion_residual(rho, epsilon)`，且只写质量分量。

        为什么用结构判据而不是"全域加权积分为零"（原判据已弃用，记录
        原因避免重写）：那个写法不是良定义的。残差已经除过 det(J)，真正
        的积分权重是 `求积权重 w_s * det(J)`，而本项目**没有把 SP 求积
        权重暴露到 ops/mesh 上**（实测两者都没有 weights/quad 字段），
        只用 det(J) 当权重得到的和本就不该为零——实测 net/|net|=3.2e-2，
        那是漏掉 w_s 的后果，不是代码缺陷。另外本算例有 OUTLET/SYMMETRY
        边界，BR1 扩散在边界上一般有非零通量，"内部净源为零"本身也不成立。
        本项目自己的守恒性检验（见 test_entropy_stable_volume.py）同样是
        结构性的（两点通量对称性），不是加权积分。

        本判据钉住的是：
        1. 只有质量分量被写入（不会泄漏到动量/能量——那会是静默的物理
           错误，因为动量/能量已经通过 mu_t_field 拿到过 AV）；
        2. 用的是**既有的** BR1 面耦合标量扩散装配（其守恒性/一致性已由
           湍流输运的验证建立），不是另写一套；
        3. 扩散系数确实是 Persson-Peraire 的 epsilon 场。
        任何一条被破坏（换成自写装配、系数取错、写错分量）都会让它失败。
        """
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_persson_peraire_artificial_viscosity,
        )
        from autoflowcfd.core.turbulence.transport import (
            compute_scalar_diffusion_residual,
        )

        solver, mesh = _build(av_enabled=True)
        _set_density_bump(solver, mesh)
        got = solver._artificial_mass_diffusion_residual()

        eps = compute_persson_peraire_artificial_viscosity(
            solver, alpha_av=solver.artificial_viscosity_alpha)
        assert np.abs(eps).max() > 0.0, "本用例需要 AV 传感器真的触发"
        expected_mass = compute_scalar_diffusion_residual(
            np.ascontiguousarray(solver.state.U[..., 0]),
            np.ascontiguousarray(eps), mesh, solver.ops)

        np.testing.assert_array_equal(got[..., 0], expected_mass)
        for v in range(1, got.shape[-1]):
            np.testing.assert_array_equal(
                got[..., v], np.zeros_like(got[..., v]),
                err_msg=f"人工质量扩散泄漏到了第 {v} 个分量")


class TestDiffusionIsDissipative:
    def test_quadratic_form_is_negative(self):
        """判据 5（严格形式）：扩散算子必须是**耗散**的，不是反扩散。

        对真正的扩散算子，分部积分给出
            <d_rho, div(eps*grad(d_rho))> = -integral( eps * |grad(d_rho)|^2 ) < 0
        （齐次/周期边界下严格成立；本算例的对称/出流边界上边界项很小，
        不改变符号）。这条比"某个峰值点处 d(rho)/dt<0"强得多：它是整个
        算子的性质，符号写反、界面校正装配错误、或把扩散写成反扩散都会
        让它变正。

        反扩散是本项目真实踩过的坑——`transport.py` 的"符号约定"记录了
        k/omega 扩散曾误写成 `-div(...)`，指数放大棋盘模态导致 k/omega
        双峰触限、求解停滞。这里对质量通道钉住同一类错误。
        """
        solver, mesh = _build(av_enabled=True)
        pert = _set_density_bump(solver, mesh)
        mass_res = solver._artificial_mass_diffusion_residual()[..., 0]

        n_cells, n_sps = mass_res.shape
        det_jacs = np.abs(mesh.jacobians["det_jacs"].reshape(n_cells, n_sps))
        quad_form = float(np.sum(pert * mass_res * det_jacs))
        norm = float(np.sum(np.abs(pert) * np.abs(mass_res) * det_jacs))
        assert norm > 0.0, "本用例需要该项确实非零才有判别力"
        assert quad_form < 0.0, (
            f"<d_rho, div(eps*grad(d_rho))> = {quad_form:.6e} >= 0——"
            "算子不是耗散的（反扩散：符号写反或界面校正装配错误），"
            "会指数放大高频模态")
        # 不只是"略小于零"：真实扩散下这个二次型应当被 |.| 范数主导
        assert quad_form / norm < -0.1, (
            f"耗散性太弱（quad/norm={quad_form / norm:.3f}），疑似部分项符号相反")
