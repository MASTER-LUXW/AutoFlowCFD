"""
AutoFlowCFD V2.0 - AUSM+up 数值通量（`core/fr_operators/kernels.py::compute_ausm_up_flux`）专项测试。

通量内部全部用物理声速（Liou 2006 原文 / SU2 `CUpwAUSMPLUSUP_Flow`）。2026-10-18 以前另有两档把 Weiss-Smith
预处理声速放进 M4±/P5± 的马赫数归一（`AFCFD_AUSM_PRECOND_MODE=legacy/pressure_physical`）：亚声速外流下预处理
声速 ≈ 来流速度，P5± 进入 |M|>=1 的饱和分支，固壁镜像面上 p_half ≈ 2p、dp_half/du_n = 0，不可穿透条件失效；
两档随环境变量删除。

覆盖：相容性/反对称性、与 Liou/SU2 独立转写逐位一致、运动接触间断、固壁镜像面的低马赫标度（核心判据：与精确
声学响应 rho*a*u_n 对照）、通量只经 fa 依赖 mach_ref、三份实现（CPU numba、GPU CuPy、GPU CUDA 源串）一致。
"""

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import wall_ghost_state
from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux

from tests.unit._numpy_as_cupy import NumpyAsCupy

GAMMA = 1.4
RHO_INF = 1.225
P_INF = 101325.0
U_INF = 33.33
A_INF = np.sqrt(GAMMA * P_INF / RHO_INF)
MACH_REF = U_INF / A_INF
Q_INF = 0.5 * RHO_INF * U_INF ** 2



def _flux(qL, qR, n, mach_ref):
    """按 njit 逐通量点函数的契约调用：一维连续数组。"""
    return np.asarray(compute_ausm_up_flux(
        np.ascontiguousarray(np.asarray(qL, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(qR, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(n, dtype=np.float64)),
        float(mach_ref),
    ))


def _wall_pair(u_n, normal=(1.0, 0.0, 0.0)):
    """静止绝热固壁的 (内部态, 镜像幽灵态) 对。

    用生产代码里真正参与残差组装的 `wall_ghost_state`，而不是在测试里
    自己重写一遍镜像公式——否则测的就不是求解器实际走的那条路。
    """
    n = np.asarray(normal, dtype=np.float64)
    Q_int = np.array([[RHO_INF, u_n * n[0], u_n * n[1], u_n * n[2], P_INF]])
    Q_gho = wall_ghost_state(Q_int, n[None, :], is_no_slip=True)
    return Q_int[0], Q_gho[0], n


class TestAusmUpFluxIdentities:
    """相容性与反对称性。"""

    def _random_states(self, n=200, seed=20260916):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            yield (
                np.array([RHO_INF * (0.5 + rng.random()),
                          *((rng.random(3) - 0.5) * 2.0 * U_INF),
                          P_INF * (0.5 + rng.random())]),
                np.array([RHO_INF * (0.5 + rng.random()),
                          *((rng.random(3) - 0.5) * 2.0 * U_INF),
                          P_INF * (0.5 + rng.random())]),
                rng.normal(size=3) / np.linalg.norm(rng.normal(size=3) + 1e-12),
            )

    def test_consistency(self):
        """F(U,U,n) 必须精确等于物理通量 F(U)·n。"""
        rng = np.random.default_rng(1234)
        worst = 0.0
        for _ in range(200):
            rho = RHO_INF * (0.5 + rng.random())
            vel = (rng.random(3) - 0.5) * 2.0 * U_INF
            p = P_INF * (0.5 + rng.random())
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            q = np.array([rho, vel[0], vel[1], vel[2], p])

            un = float(vel @ n)
            H = GAMMA / (GAMMA - 1.0) * p / rho + 0.5 * float(vel @ vel)
            F_exact = np.array([
                rho * un,
                rho * un * vel[0] + p * n[0],
                rho * un * vel[1] + p * n[1],
                rho * un * vel[2] + p * n[2],
                rho * un * H,
            ])
            F = _flux(q, q, n, MACH_REF)
            worst = max(worst, float(np.max(
                np.abs(F - F_exact) / np.maximum(np.abs(F_exact), 1e-30))))
        assert worst < 1e-11, (
            f"相容性最大相对误差 {worst:.3e}"
        )

    def test_antisymmetry(self):
        """F(A,B,n) 必须精确等于 -F(B,A,-n)（守恒性的离散前提）。"""
        rng = np.random.default_rng(4321)
        worst = 0.0
        for _ in range(200):
            qA = np.array([RHO_INF * (0.5 + rng.random()),
                           *((rng.random(3) - 0.5) * 2.0 * U_INF),
                           P_INF * (0.5 + rng.random())])
            qB = np.array([RHO_INF * (0.5 + rng.random()),
                           *((rng.random(3) - 0.5) * 2.0 * U_INF),
                           P_INF * (0.5 + rng.random())])
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            Fab = _flux(qA, qB, n, MACH_REF)
            Fba = _flux(qB, qA, -n, MACH_REF)
            worst = max(worst, float(np.max(
                np.abs(Fab + Fba) / np.maximum(np.abs(Fab), 1e-30))))
        assert worst < 1e-12, (
            f"反对称性最大相对误差 {worst:.3e}"
        )


def _liou_reference_flux(qL, qR, n, mach_ref):
    """Liou (2006) AUSM+up 的独立转写（逐项对照 SU2 `CUpwAUSMPLUSUP_Flow::
    ComputeMassAndPressureFluxes`）。

    刻意与生产实现分开写：两者只有同时出错才会一起通过——2026-10-17 以前这份参考与生产实现
    一样，用各自声速归一两侧马赫数、质量通量取中心化的 0.5*(rhoL*aL + rhoR*aR)，两者一起偏离 SU2。P5± 的 α 项
    **不**乘 1/4（Liou 式 (24) 展开、SU2 `pLP = 0.25*(mL+1)^2*(2-mL) +
    alpha*mL*(mL^2-1)^2`）—— 生产实现 2026-09-25 之前在这里多乘了 1/4。
    """
    g = GAMMA
    rL, rR = qL[0], qR[0]
    pL, pR = qL[4], qR[4]
    unL, unR = float(qL[1:4] @ n), float(qR[1:4] @ n)
    aL, aR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    ah = 0.5 * (aL + aR)
    rh = 0.5 * (rL + rR)
    Mbar2 = (unL ** 2 + unR ** 2) / (2 * ah ** 2)
    M0 = np.sqrt(min(1.0, max(Mbar2, mach_ref ** 2)))
    fa = max(M0 * (2 - M0), 1e-6)
    beta, alpha = 1 / 8, 3 / 16 * (-4 + 5 * fa * fa)
    sm = sp = 1.0
    # 两侧马赫数都用界面公共声速归一（SU2 `mL = ProjVelL/aF`）
    ML, MR = unL / (sm * ah), unR / (sm * ah)
    MLp, MRp = unL / (sp * ah), unR / (sp * ah)

    def m4(M, s):
        if abs(M) >= 1:
            return 0.5 * (M + s * abs(M))
        return s * 0.25 * (M + s) ** 2 + s * beta * (M * M - 1) ** 2

    def p5(M, s):
        if abs(M) >= 1:
            return 0.5 * (1 + s * np.sign(M))
        return 0.25 * (M + s) ** 2 * (2 - s * M) + s * alpha * M * (M * M - 1) ** 2

    Mp = -(0.25 / fa) * max(1 - Mbar2, 0.0) * (pR - pL) / (rh * (sm * ah) ** 2)
    mF = m4(ML, 1) + m4(MR, -1) + Mp
    mdot = sm * ah * (max(mF, 0.0) * rL + min(mF, 0.0) * rR)    # SU2 `mdot = aF*(max(mF,0)*rhoL+min(mF,0)*rhoR)`
    PL, PR = p5(MLp, 1), p5(MRp, -1)
    ph = PL * pL + PR * pR - 0.75 * PL * PR * (rL + rR) * fa * (sp * ah) * (unR - unL)
    up = qL if mdot >= 0 else qR
    H = g / (g - 1) * up[4] / up[0] + 0.5 * float(up[1:4] @ up[1:4])
    return np.array([mdot, mdot * up[1] + ph * n[0], mdot * up[2] + ph * n[1],
                     mdot * up[3] + ph * n[2], mdot * H])


class TestAusmUpMatchesLiouReference:
    """生产通量与 Liou/SU2 参考转写逐位（到舍入）一致。"""

    def test_random_states(self):
        rng = np.random.default_rng(20260925)
        worst = 0.0
        for _ in range(400):
            qL = np.array([RHO_INF * (0.5 + rng.random()), *((rng.random(3) - 0.5) * 2 * U_INF),
                           P_INF * (0.9 + 0.2 * rng.random())])
            qR = np.array([RHO_INF * (0.5 + rng.random()), *((rng.random(3) - 0.5) * 2 * U_INF),
                           P_INF * (0.9 + 0.2 * rng.random())])
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            F = _flux(qL, qR, n, MACH_REF)
            R = _liou_reference_flux(qL, qR, n, MACH_REF)
            worst = max(worst, float(np.max(np.abs(F - R) / (np.abs(R) + 1e-9 * P_INF))))
        assert worst < 1e-10, f"与 Liou/SU2 参考最大相对差 {worst:.3e}"


class TestAusmUpMovingContact:
    """运动接触间断（压力、速度相同，密度跳变）：标准 AUSM+up 的质量通量恰为迎风密度 x 速度，
    动量 = 质量通量 x 速度 + p n，能量 = 质量通量 x 迎风总焓（Liou 2006 的接触间断保持性质）。
    2026-10-17 以前质量通量取 0.5*(rhoL*aL + rhoR*aR)*M、马赫数用各自声速归一：密度不迎风，
    这条性质不成立。"""

    @pytest.mark.parametrize('u', [12.0, -12.0])
    def test_mass_flux_is_upwind(self, u):
        n = np.array([1.0, 0.0, 0.0])
        qL = np.array([1.40, u, 3.0, -2.0, P_INF])
        qR = np.array([0.95, u, 3.0, -2.0, P_INF])
        F = _flux(qL, qR, n, MACH_REF)
        up = qL if u > 0 else qR
        H = GAMMA / (GAMMA - 1.0) * up[4] / up[0] + 0.5 * float(up[1:4] @ up[1:4])
        mdot = up[0] * u
        np.testing.assert_allclose(F[0], mdot, rtol=1e-12)
        np.testing.assert_allclose(F[1:4], mdot * up[1:4] + P_INF * n, rtol=1e-12)
        np.testing.assert_allclose(F[4], mdot * H, rtol=1e-12)

    def test_stationary_contact_has_no_mass_flux(self):
        n = np.array([0.0, 1.0, 0.0])
        F = _flux(np.array([1.40, 0.0, 0.0, 0.0, P_INF]), np.array([0.95, 0.0, 0.0, 0.0, P_INF]), n, MACH_REF)
        assert F[0] == 0.0
        np.testing.assert_allclose(F[1:4], P_INF * n, rtol=1e-14)


class TestAusmUpLowMachWallScaling:
    """**本文件的核心**：固壁镜像面上界面压力的低马赫标度。

    镜像状态 (rho, u_n, p) | (rho, -u_n, p) 的**可压缩精确**黎曼解给出声学
    超压 `p* - p = rho*a*u_n`（O(M) 标度）。那正是 Godunov 型格式在低马赫下
    失效的机制（Guillard & Viozat 1999）：稳态驻点附近 u_n ~ U 时 Cp ~ 2/M。
    AUSM+up 引入 fa 缩放的目的就是在低马赫下**偏离**这个声学响应，使压力
    扰动回到 O(M^2)。2026-09-25 之前本类以"与 rho*a*u_n 吻合到 10%"为判据
    —— 那条判据与 P5± 的 α 项多乘 1/4 的实现缺陷互相掩护：plate_demo
    （M=0.098）P0 稳态驻点平台 Cp≈21.5 ≈ 2/M，Cd≈4.2（实验≈1.2）。
    """

    def _dp(self, u_n, mach_ref, p=P_INF):
        q_int, q_gho, n = _wall_pair(u_n)
        q_int[4] = q_gho[4] = p
        F = _flux(q_int, q_gho, n, mach_ref)
        assert abs(F[0]) < 1e-10 * RHO_INF * A_INF, '壁面质量通量必须为零'
        return F[1] / n[0] - p

    @pytest.mark.parametrize('frac', [1.0, 0.5, 0.2, 0.05])
    def test_physical_mode_is_far_below_acoustic_response(self, frac):
        """壁面超压必须远小于声学响应 rho*a*u_n（M=0.098 下 < 25%）。"""
        u_n = frac * U_INF
        ratio = self._dp(u_n, MACH_REF) / (RHO_INF * A_INF * u_n)
        assert 0.0 < ratio < 0.25, f"u_n/U={frac}: 超压/声学响应 = {ratio:.3f}"

    def test_physical_mode_scales_like_mach(self):
        """超压/声学响应 ∝ M：来流马赫数减半（u_n 按同一 u_n/U 取）时比值约减半。

        这是低马赫相容性的直接判据；α 项多乘 1/4 时该比值 ≈0.95 且与 M 无关。
        """
        r1 = self._dp(0.5 * U_INF, MACH_REF) / (RHO_INF * A_INF * 0.5 * U_INF)
        r2 = self._dp(0.25 * U_INF, 0.5 * MACH_REF) / (RHO_INF * A_INF * 0.25 * U_INF)
        assert 0.35 < r2 / r1 < 0.65, f"M 减半后比值之比 {r2 / r1:.3f}（应≈0.5）"

    def test_p5_split_slope_at_zero_is_order_fa_squared(self):
        """P5+(M)+P5-(-M) 在 M=0 处的斜率 = 2(0.75+α)，低马赫下 = O(fa^2)。"""
        eps = 1e-6
        slope = self._dp(eps * A_INF, MACH_REF) / (eps * P_INF)
        # 只取 P5 那一半：减去 Pu 项在同一 u_n 上的贡献（线性于 u_n）
        m0 = MACH_REF
        fa = m0 * (2 - m0)
        pu = 0.75 * 0.25 * 2 * RHO_INF * fa * A_INF * 2 * eps * A_INF / (eps * P_INF)
        alpha = 3 / 16 * (-4 + 5 * fa * fa)
        assert abs((slope - pu) - 2 * (0.75 + alpha)) < 1e-4, (slope - pu, 2 * (0.75 + alpha))
        assert abs(2 * (0.75 + alpha)) < 2.0 * fa * fa

    def test_zero_normal_velocity_gives_exactly_p(self):
        """u_n = 0 时（收敛态的壁面）p_half 精确等于 p。"""
        q_int, q_gho, n = _wall_pair(0.0)
        dp = _flux(q_int, q_gho, n, MACH_REF)[1] / n[0] - P_INF
        assert abs(dp) < 1e-9 * P_INF, f"u_n=0 时 p_half-p={dp:.3e}"

    def test_restoring_gradient_does_not_saturate_in_physical_mode(self):
        """dp_half/du_n 在整个工况范围内必须保持正值。

        这是比"超压大小"更本质的判据：壁面不可穿透条件是靠"法向速度增大 => 界面压力增大 => 把流体推回去"
        这个恢复梯度实现的。已删除的 legacy 档在 P5± 饱和后该梯度归零（实测斜率从 ~3323 塌到 ~86，极差 39 倍）。
        """
        n = np.array([1.0, 0.0, 0.0])
        u_list = np.linspace(0.02 * U_INF, 2.0 * U_INF, 25)
        dp = []
        for u_n in u_list:
            q_int, q_gho, _ = _wall_pair(u_n)
            dp.append(_flux(q_int, q_gho, n, MACH_REF)[1] - P_INF)
        slopes = np.diff(dp) / np.diff(u_list)
        # 低马赫相容的量级是 rho*U（不是声阻抗 rho*a —— 那是 O(M) 标度，
        # 见本类文档）；恢复梯度必须处处为正、不塌陷、且不低于 rho*U。
        assert np.all(slopes > RHO_INF * U_INF), (
            f"恢复梯度出现饱和/反向或过软：min slope={slopes.min():.1f}，"
            f"应处处 > rho*U={RHO_INF*U_INF:.1f}"
        )
        assert np.all(np.diff(slopes) > 0), "恢复梯度应随 u_n 单调增强（不饱和）"


class TestFluxDependsOnMachRefOnlyThroughFa:
    """通量只经 AUSM+up 自带的 fa 依赖 mach_ref。"""

    def test_physical_mode_flux_independent_of_mach_ref(self):
        """通量必须与 mach_ref 无关——除了 AUSM+up 自带的 fa（它本来就该依赖 mach_ref）。

        因此这里取 Mbar2 明显大于 mach_ref^2 的状态，使 M0_sq 由 Mbar2
        决定、fa 与 mach_ref 解耦，此时不同 mach_ref 必须给出逐位相同
        的通量。
        """
        qL = np.array([1.2, 200.0, 0.0, 0.0, 101300.0])
        qR = np.array([1.2, 180.0, 0.0, 0.0, 101500.0])
        n = np.array([1.0, 0.0, 0.0])
        ref = _flux(qL, qR, n, 0.05)
        for mach_ref in (0.1, 0.2, 0.3, 0.5):
            F = _flux(qL, qR, n, mach_ref)
            assert np.allclose(F, ref, rtol=1e-15, atol=0), (
                f"mach_ref={mach_ref} 改变了通量（fa 之外的依赖）"
            )


class TestEveryImplementationUsesTheSameP5:
    """AUSM+up 有三份实现（CPU numba 核、GPU CuPy 向量化版、GPU CUDA P0 源串）。2026-09-25 修正的
    α 项系数缺陷在当时的四份里同时存在（第四份 numba.cuda 版 `backend/fr_gpu_p0.py` 生产零调用，
    2026-10-04 删除），这里把"各份一致"钉成判据。"""

    def test_cupy_batch_version_matches_cpu_kernel(self, monkeypatch):
        import autoflowcfd.core.gpu.residual.gpu_inviscid.flux as gflux
        from tests.unit._gpu_cupy_shim import patch_module_get_cupy


        patch_module_get_cupy(monkeypatch, gflux, NumpyAsCupy())
        rng = np.random.default_rng(99)
        m = 300
        QL = np.column_stack([RHO_INF * (0.5 + rng.random(m)), (rng.random((m, 3)) - 0.5) * 2 * U_INF,
                              P_INF * (0.9 + 0.2 * rng.random(m))])
        QR = np.column_stack([RHO_INF * (0.5 + rng.random(m)), (rng.random((m, 3)) - 0.5) * 2 * U_INF,
                              P_INF * (0.9 + 0.2 * rng.random(m))])
        N = rng.normal(size=(m, 3))
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        Fg = np.asarray(gflux._ausm_up_flux_batch_gpu(QL, QR, N, MACH_REF))
        Fc = np.array([_flux(QL[i], QR[i], N[i], MACH_REF) for i in range(m)])
        rel = np.abs(Fg - Fc) / (np.abs(Fc) + 1e-9 * P_INF)
        assert rel.max() < 1e-10, f"GPU 批量版与 CPU 核最大相对差 {rel.max():.3e}"

    @pytest.mark.parametrize('module', [
        'autoflowcfd.core.gpu.residual.gpu_p0_inviscid',
        'autoflowcfd.core.gpu.residual.gpu_inviscid.flux',
        'autoflowcfd.core.fr_operators.kernels',
    ])
    def test_alpha_term_is_not_wrapped_in_the_quarter(self, module):
        """本机无法执行的 CUDA 实现只能做结构判据：P5 表达式里 α 项不得被
        包进 `0.25 * ((...` 的同一对括号里。"""
        import importlib
        import inspect
        import re

        src = inspect.getsource(importlib.import_module(module))
        code = "\n".join(ln.split("#", 1)[0].split("//", 1)[0] for ln in src.splitlines())
        bad = re.findall(r"0\.25\s*\*\s*\(\s*\(\s*M\w*\s*[+-]\s*1(?:\.0)?\s*\)[^;]*?alpha_pressure",
                         code, flags=re.S)
        assert not bad, f"{module} 仍把 α 项包进了 0.25*(...)：{bad[:1]}"
