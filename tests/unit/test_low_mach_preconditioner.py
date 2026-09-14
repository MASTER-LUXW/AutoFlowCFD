"""Weiss-Smith 伪时间预处理矩阵 Gamma 的数学正确性验证。

为什么这个测试是整项优化的地基：Gamma 是为了把"数万步才收敛"这个
问题从根上改善而引入的（低马赫数下 dt 被声速而非对流速度限制，
M=0.1 时白付约 10 倍步数）。它成立的前提有三条，每一条都必须是可
数值验证的硬事实，而不是"看起来对"：

  1. **谱等价**：eig(Gamma @ A_n) 必须精确等于
     (un, un, un, lam_plus, lam_minus)，后两个是
     `preconditioned_acoustic_eigs` 给出的闭式预处理声学特征值。
     这一条同时验证了 Gamma 的整个秩一推导（phi、psi、系数）——
     推导里任何一处写错都会让特征值对不上。
  2. **可逆 / 不动点不变**：det(Gamma) = beta^2 > 0，因此
     `Gamma R = 0 <=> R = 0`，收敛解与不加预处理时**完全同一个解**。
     这是"计算结果准确"这条工业级要求的数学保证。
  3. **高速区自动退化**：beta^2 = 1（局部 M >= mach_ref，例如跨声速/
     驻点以外的主流区）时 Gamma 必须恰好是单位矩阵，不对原格式产生
     任何改变。

另外验证：n_vars>5（SST 把 k/omega 挂在同一数组）时湍流分量必须原样
保留——湍流标量是被动输运量、不含声学模态。
"""
import numpy as np
import pytest

from autoflowcfd.core.utils.preconditioning import (
    apply_low_mach_preconditioner,
    preconditioned_acoustic_eigs,
    preconditioned_sound_speed,
    _precond_beta2,
)

GAMMA = 1.4


def _euler_flux_normal(U, n):
    """法向欧拉通量 F_n(U)（用于数值 Jacobian）。"""
    rho = U[0]
    u, v, w = U[1] / rho, U[2] / rho, U[3] / rho
    q2 = u * u + v * v + w * w
    p = (GAMMA - 1.0) * (U[4] - 0.5 * rho * q2)
    un = u * n[0] + v * n[1] + w * n[2]
    return np.array([
        rho * un,
        rho * u * un + p * n[0],
        rho * v * un + p * n[1],
        rho * w * un + p * n[2],
        (U[4] + p) * un,
    ])


def _numeric_flux_jacobian(U, n, eps_rel=1e-7):
    """A_n = dF_n/dU 的中心差分数值 Jacobian。"""
    A = np.zeros((5, 5))
    for j in range(5):
        h = eps_rel * max(abs(U[j]), 1.0)
        Up = U.copy(); Up[j] += h
        Um = U.copy(); Um[j] -= h
        A[:, j] = (_euler_flux_normal(Up, n) - _euler_flux_normal(Um, n)) / (2.0 * h)
    return A


def _gamma_matrix(rho, u, v, w, p, beta2):
    """按实现中的秩一公式显式构造 Gamma（供矩阵级验证用）。"""
    a2 = GAMMA * p / rho
    q2 = u * u + v * v + w * w
    H = a2 / (GAMMA - 1.0) + 0.5 * q2
    phi = np.array([1.0, u, v, w, H])
    psi = (GAMMA - 1.0) * np.array([0.5 * q2, -u, -v, -w, 1.0])
    return np.eye(5) - ((1.0 - beta2) / a2) * np.outer(phi, psi)


def _apply_via_public_api(r5, rho, u, v, w, p, mach_ref, k=1.1):
    """走生产入口（numba kernel）施加一次 Gamma，返回 (5,) 结果。"""
    R = np.zeros((1, 1, 5)); R[0, 0] = r5
    Q = np.zeros((1, 1, 5)); Q[0, 0] = [rho, u, v, w, p]
    return apply_low_mach_preconditioner(R, Q, mach_ref, k)[0, 0]


class TestGammaMatchesKernel:
    """生产 kernel 与显式矩阵构造必须一致（钉住 kernel 里的手工展开）。"""

    @pytest.mark.parametrize("mach", [0.01, 0.05, 0.1, 0.3, 0.9, 1.5])
    def test_kernel_equals_explicit_matrix(self, mach):
        rng = np.random.default_rng(0)
        rho, p = 1.225, 101325.0
        a = np.sqrt(GAMMA * p / rho)
        u, v, w = mach * a, 0.13 * mach * a, -0.07 * mach * a
        q2 = u * u + v * v + w * w
        mach_ref = 0.1
        beta2 = float(_precond_beta2(np.array(q2), np.array(GAMMA * p / rho), mach_ref, 1.1))
        G = _gamma_matrix(rho, u, v, w, p, beta2)
        for _ in range(5):
            r = rng.standard_normal(5) * np.array([1.0, 30.0, 30.0, 30.0, 1e4])
            got = _apply_via_public_api(r, rho, u, v, w, p, mach_ref)
            ref = G @ r
            denom = max(np.abs(ref).max(), 1e-300)
            assert np.abs(ref - got).max() / denom <= 1e-13


class TestSpectralEquivalence:
    """**最强的一条**：eig(Gamma @ A_n) 必须等于闭式预处理特征值。"""

    @pytest.mark.parametrize("mach", [0.02, 0.05, 0.1, 0.4])
    @pytest.mark.parametrize("mach_ref", [0.05, 0.1])
    def test_eigenvalues_match_closed_form(self, mach, mach_ref):
        rho, p = 1.15, 98000.0
        a = np.sqrt(GAMMA * p / rho)
        # 速度与法向刻意不共线，确保验证的是一般情形
        u, v, w = mach * a, 0.2 * mach * a, -0.1 * mach * a
        n = np.array([0.6, -0.48, 0.64])
        n = n / np.linalg.norm(n)

        q2 = u * u + v * v + w * w
        a2 = GAMMA * p / rho
        beta2 = float(_precond_beta2(np.array(q2), np.array(a2), mach_ref, 1.1))
        G = _gamma_matrix(rho, u, v, w, p, beta2)

        U = np.array([rho, rho * u, rho * v, rho * w,
                      p / (GAMMA - 1.0) + 0.5 * rho * q2])
        A_n = _numeric_flux_jacobian(U, n)
        eig_got = np.sort_complex(np.linalg.eigvals(G @ A_n)).real

        un = u * n[0] + v * n[1] + w * n[2]
        # 闭式：预处理后的两个声学特征值 + 三重对流特征值 un
        lam_center = un * (1.0 + beta2) / 2.0
        radius = np.sqrt(((1.0 - beta2) * un / 2.0) ** 2 + beta2 * a2)
        eig_ref = np.sort(np.array([un, un, un,
                                    lam_center + radius, lam_center - radius]))

        scale = max(np.abs(eig_ref).max(), 1e-300)
        assert np.abs(eig_got - eig_ref).max() / scale <= 2e-5, (
            f"预处理后谱与闭式不符 (mach={mach}, mach_ref={mach_ref}):\n"
            f"  数值 {eig_got}\n  闭式 {eig_ref}"
        )

    def test_closed_form_helper_agrees(self):
        """`preconditioned_acoustic_eigs`（已有、用于通量与 CFL）与本测试
        用的闭式表达式必须是同一个公式——否则"预处理残差"与"预处理
        步长"两半会基于不同的 beta 定义，成对关系被破坏。"""
        un = np.array([12.0, -35.0, 0.0])
        a = np.array([340.0, 340.0, 340.0])
        lam_p, lam_m, c_pre = preconditioned_acoustic_eigs(un, a, mach_ref=0.1, k=1.1)
        beta2 = np.clip(np.maximum((un / a) ** 2, 1.1 * 0.01), 1e-10, 1.0)
        lam_center = un * (1.0 + beta2) / 2.0
        radius = np.sqrt(((1.0 - beta2) * un / 2.0) ** 2 + beta2 * a ** 2)
        np.testing.assert_allclose(lam_p, lam_center + radius, rtol=1e-14)
        np.testing.assert_allclose(lam_m, lam_center - radius, rtol=1e-14)
        np.testing.assert_allclose(c_pre, np.sqrt(beta2) * a, rtol=1e-14)


class TestInvertibilityAndFixedPoint:
    """不动点不变 —— "计算结果准确"的数学保证。"""

    @pytest.mark.parametrize("mach", [0.001, 0.01, 0.1, 0.5, 2.0])
    def test_determinant_equals_beta2(self, mach):
        rho, p = 1.225, 101325.0
        a2 = GAMMA * p / rho
        a = np.sqrt(a2)
        u, v, w = mach * a, 0.0, 0.0
        q2 = u * u
        beta2 = float(_precond_beta2(np.array(q2), np.array(a2), 0.1, 1.1))
        G = _gamma_matrix(rho, u, v, w, p, beta2)
        # psi . phi = a^2  =>  det = 1-(1-beta2) = beta2
        assert np.isclose(np.linalg.det(G), beta2, rtol=1e-10)
        assert np.linalg.det(G) > 0.0

    def test_zero_residual_stays_zero_and_nonzero_stays_nonzero(self):
        """Gamma R = 0 <=> R = 0：收敛判据与收敛解都不被改变。"""
        rho, u, v, w, p = 1.2, 30.0, 1.0, -2.0, 100000.0
        zero = _apply_via_public_api(np.zeros(5), rho, u, v, w, p, 0.1)
        assert np.all(zero == 0.0)
        rng = np.random.default_rng(7)
        for _ in range(20):
            r = rng.standard_normal(5) * np.array([1e-3, 1.0, 1.0, 1.0, 1e3])
            out = _apply_via_public_api(r, rho, u, v, w, p, 0.1)
            assert np.abs(out).max() > 0.0, "非零残差被预处理成了恒零（会伪造收敛）"


class TestDegeneratesAtHighMach:
    def test_gamma_is_identity_when_beta2_is_one(self):
        """局部 M >= mach_ref 的区域必须精确退化为不加预处理。"""
        rho, p = 1.0, 100000.0
        a = np.sqrt(GAMMA * p / rho)
        u = 1.2 * a   # 超声速 -> beta2 = 1
        rng = np.random.default_rng(11)
        for _ in range(5):
            r = rng.standard_normal(5) * np.array([1.0, 50.0, 50.0, 50.0, 1e4])
            out = _apply_via_public_api(r, rho, u, 0.0, 0.0, p, 0.1)
            np.testing.assert_allclose(out, r, rtol=0.0, atol=1e-12)


class TestTurbulenceComponentsUntouched:
    def test_extra_vars_pass_through_unchanged(self):
        """SST 下残差数组是 7 变量（k/omega 挂在 5:7）；湍流标量不含声学
        模态，必须原样保留。"""
        rng = np.random.default_rng(13)
        R = rng.standard_normal((4, 3, 7))
        Q = np.zeros((4, 3, 5))
        Q[..., 0] = 1.2; Q[..., 1] = 30.0; Q[..., 4] = 1e5
        out = apply_low_mach_preconditioner(R, Q, 0.1)
        assert out.shape == R.shape
        np.testing.assert_array_equal(out[:, :, 5:], R[:, :, 5:])
        assert not np.allclose(out[:, :, :5], R[:, :, :5])

    def test_nonphysical_state_passes_through(self):
        """密度/压力非物理（正性限制器尚未介入的瞬态）时原样透传，
        不在预处理里制造 NaN 掩盖真正的问题。"""
        R = np.ones((1, 1, 5))
        Q = np.zeros((1, 1, 5))
        Q[0, 0] = [0.0, 0.0, 0.0, 0.0, -1.0]
        out = apply_low_mach_preconditioner(R, Q, 0.1)
        np.testing.assert_array_equal(out[0, 0], np.ones(5))


class TestTrialStatePrimitivesContract:
    """钉住 `step.py::mean_flow_residual` 依赖的那个隐式契约。

    `mean_flow_residual_raw` 把 `solver.state.U` 临时换成 RK 子迭代的
    试探态 `U_trial` 求残差，在 `finally` 里恢复 `state.U`，但**故意不**
    恢复 `state.Q`——于是它返回后 `state.Q` 仍是 `U_trial` 对应的原始
    变量，`mean_flow_residual` 正是靠这一点把 `Gamma(Q_trial)` 作用在
    `R(U_trial)` 上，不需要再做一次全场原始变量转换（P2 规模下那是
    1.2GiB 的额外分配 + 一次完整转换）。

    为什么必须有这个测试：这个依赖只写在注释里。如果以后有人出于
    "对称清理"的直觉在 `finally` 里补上 `state.Q` 的恢复，预处理就会
    静默地用**基态**的 Q 去构造 Gamma——Gamma 仍可逆、残差仍趋零、
    所有现有测试仍通过，但每个 RK 子步的预处理矩阵都对应错误的状态，
    只会表现为收敛变慢/失稳这种极难溯源的症状。这里让那个改动直接
    在单元测试里失败。

    同时也是"Gamma 必须用 Q_trial 而不是基态 Q"这条要求的可执行文档。
    """

    def test_state_q_follows_trial_u_after_residual_eval(self):
        from tests.validation._channel_mesh import (
            build_channel_mesh_prism, build_face_exact_ghost_provider)
        from autoflowcfd.core.fr_solver import FRSolver
        from autoflowcfd.core.time_integration import TimeIntegrationScheme

        H, Lx, Lz = 1.0e-2, 2.0e-2, 2.5e-3
        bc = {
            "wall_bottom": {"type": "WALL", "is_no_slip": True,
                            "wall_velocity": [0.0, 0.0, 0.0]},
            "wall_top": {"type": "WALL", "is_no_slip": True,
                         "wall_velocity": [5.0, 0.0, 0.0]},
            "z_min": {"type": "SYMMETRY"}, "z_max": {"type": "SYMMETRY"},
            "x_min": {"type": "OUTLET", "p_outlet": 101325.0},
            "x_max": {"type": "OUTLET", "p_outlet": 101325.0},
        }
        mesh = build_channel_mesh_prism(1, nx=4, ny=4, nz=1, Lx=Lx, H=H, Lz=Lz)
        solver = FRSolver(
            mesh=mesh, order=1, turb_model_name="NONE", n_vars=5,
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            rho_inf=1.225, vel_inf=5.0, p_inf=101325.0, mu_molecular=1.8e-5,
            bc_overrides=bc, n_threads=1,
        )
        solver.order_continuation_enabled = False
        solver.boundary_ghost_provider = build_face_exact_ghost_provider(
            mesh, Lx, H, Lz, bc)

        base_U = solver.state.U.copy()
        # 试探态：与基态明确不同（动量整体抬 30%，能量随之改变）
        U_trial = base_U.copy()
        U_trial[:, :, 1] += 0.3 * abs(base_U[:, :, 0]).max() * 5.0

        saved = solver.state.U
        solver.state.U = U_trial
        try:
            solver.compute_inviscid_residual()
        finally:
            solver.state.U = saved

        # 契约：U 已恢复为基态，而 Q 必须仍是试探态的原始变量
        np.testing.assert_array_equal(solver.state.U, base_U)
        q_after = solver.state.Q.copy()

        # 参照值：把 U_trial 真正装进 state 再转换一次，取得"正确的 Q_trial"
        solver.state.U = U_trial
        solver.state._update_primitives()
        q_trial_ref = solver.state.Q.copy()
        solver.state.U = base_U
        solver.state._update_primitives()
        q_base_ref = solver.state.Q.copy()

        np.testing.assert_allclose(
            q_after, q_trial_ref, rtol=0.0, atol=0.0,
            err_msg="残差求值后 state.Q 不再对应试探态——step.py 里的低马赫数"
                    "预处理会用基态 Q 构造 Gamma（静默错误），见本类文档")
        assert not np.allclose(q_trial_ref[:, :, 1], q_base_ref[:, :, 1]), (
            "试探态与基态的速度场没有区别，本测试失去判别力（请调整扰动幅度）")


class TestPseudoTimeStepPairsWithGamma:
    """伪时间步长用的有效声速必须与 Gamma 共用同一个 beta^2。

    这是"dt 与被积分的算子必须成对"这条前提的可执行形式。显式推进的是
    `dU/dtau = -Gamma R`，稳定性由 `Gamma @ A_n` 的谱半径决定；cfl.py 用
    `|un| + c_precond` 作为该谱半径的上界估计。上界成立的前提是 c_precond
    里的 beta^2 就是 Gamma 里那一个（按**速度模**取）。

    历史（本测试正是为它而写）：cfl.py 最初把面法向速度 `un` 传给
    `preconditioned_acoustic_eigs`，于是 beta^2 按 un 取。因为 |un| <= |u|，
    这个 beta^2 偏小、有效声速偏小、dt 被高估——下面第二个测试用数值
    特征值证明那样取根本**不是**上界，也就是会给出超过稳定极限的步长。
    两者只在都落到下限 k*mach_ref^2 时才相等，所以钝体绕流的局部加速区
    （M 超过 mach_ref）才暴露问题，短算例容易漏掉。
    """

    # 面法向几乎垂直于流动（un = 0.1*|u|）。这个选择是刻意的：beta^2(un)
    # 与 beta^2(|u|) 的差距随夹角增大，实测 un=0.9|u| 时旧的（un）取法碰巧
    # 仍是上界，只有在强斜置下才暴露出不是上界——而真实网格上"速度与面
    # 法向接近垂直"的面到处都是（任何与流向大致平行的壁面/内部面）。
    OBLIQUE_N = np.array([0.1, 0.9, -0.4242640687119285])

    def _oblique_state(self, mach=0.5):
        """速度与面法向明显不共线、且局部 M 高于 mach_ref 的状态。"""
        rho, p = 1.15, 98000.0
        a = np.sqrt(GAMMA * p / rho)
        u, v, w = mach * a, 0.0, 0.0      # 流动沿 x
        n = self.OBLIQUE_N / np.linalg.norm(self.OBLIQUE_N)
        return rho, p, a, u, v, w, n

    def test_uses_velocity_magnitude_beta2(self):
        rho, p, a, u, v, w, n = self._oblique_state()
        q2 = u * u + v * v + w * w
        a2 = GAMMA * p / rho
        mach_ref = 0.1
        beta2_gamma = float(_precond_beta2(np.array(q2), np.array(a2), mach_ref, 1.1))
        c_ps = float(preconditioned_sound_speed(np.sqrt(q2), np.array(a), mach_ref))
        np.testing.assert_allclose(c_ps, np.sqrt(beta2_gamma) * a, rtol=1e-14)

        # 旧的（错误的）取法：beta^2 按 un 取，必定偏小
        un = u * n[0] + v * n[1] + w * n[2]
        _, _, c_un = preconditioned_acoustic_eigs(np.array(un), np.array(a), mach_ref)
        assert float(c_un) < c_ps, (
            "本测试失去判别力：所选状态下两种 beta^2 取法给出相同的有效声速"
            "（说明都落在 k*mach_ref^2 下限上，请调大 mach 或加大流动与法向的夹角）")

    @pytest.mark.parametrize("mach", [0.2, 0.35, 0.5, 0.8])
    def test_velocity_magnitude_bounds_spectrum_and_normal_component_does_not(self, mach):
        """数值特征值判定：|un| + c_precond 是不是 eig(Gamma A_n) 的上界。"""
        rho, p, a, u, v, w, n = self._oblique_state(mach)
        q2 = u * u + v * v + w * w
        a2 = GAMMA * p / rho
        mach_ref = 0.1
        beta2 = float(_precond_beta2(np.array(q2), np.array(a2), mach_ref, 1.1))
        G = _gamma_matrix(rho, u, v, w, p, beta2)
        U = np.array([rho, rho * u, rho * v, rho * w,
                      p / (GAMMA - 1.0) + 0.5 * rho * q2])
        rho_spec = float(np.abs(np.linalg.eigvals(G @ _numeric_flux_jacobian(U, n))).max())

        un = abs(u * n[0] + v * n[1] + w * n[2])
        bound_ok = un + float(preconditioned_sound_speed(
            np.sqrt(q2), np.array(a), mach_ref))
        _, _, c_un = preconditioned_acoustic_eigs(np.array(un), np.array(a), mach_ref)
        bound_bad = un + float(c_un)

        assert rho_spec <= bound_ok * (1.0 + 1e-9), (
            f"按速度模取 beta^2 竟不是谱半径上界 (mach={mach}): "
            f"谱半径={rho_spec:.6e} > 估计={bound_ok:.6e}")
        assert bound_bad < rho_spec, (
            f"按 un 取 beta^2 在本状态下恰好也是上界 (mach={mach})，本测试"
            f"失去判别力：谱半径={rho_spec:.6e} 估计={bound_bad:.6e}")
