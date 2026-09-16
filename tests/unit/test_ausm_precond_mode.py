"""
AutoFlowCFD V2.0 - AUSM+up 预处理声速作用域（precond_mode）专项测试。

## 背景

`core/fr_operators/kernels.py::compute_ausm_up_flux` 从 2026-08-23 起把
Weiss-Smith 预处理声速用在了"喂给 M4± / P5± 分裂函数的马赫数"上。
2026-09-16 定位到这是 `stagnation_face_overpressure_localized` 记录的
那个"与 CFL 无关的缓慢退化机制"的根因：

亚声速外流下 `beta2` 被压到下限 `1.1*mach_ref^2` 附近，于是预处理声速
`a_p = sqrt(beta2)*a ≈ u_inf`，任何法向速度接近来流量级的面上"预处理
马赫数"≈1。而 `P5±` 在 `|M| >= 1` 时**恒等于 1 / 0、导数精确为零**，
压力分裂进入饱和分支：

  * 固壁镜像幽灵态满足 `M_R = -M_L`，故 `P5+(M_L)+P5-(M_R) = 2*P5+(M_L)`；
    `M_L ≈ 0.95` 时该和 ≈ 1.994，`p_half ≈ p_L + p_R = 2p`。
  * 更本质：饱和分支上 `dp_half/du_n = 0`，壁面压力对法向速度的**恢复
    梯度消失**，不可穿透条件不再被强制执行。

本文件把这件事钉成回归测试。核心判据不是"p_half == p"（镜像黎曼问题
本来就有真实的压力升高），而是**与精确解对照**：镜像状态的精确黎曼解
是两道对称压缩波，声学近似下 `p* - p ≈ rho * a * u_n`。

## 覆盖内容

1. `TestPrecondModeResolution`         —— 环境变量/显式取值解析与默认值
2. `TestPrecondModeFluxIdentities`     —— 三档都必须满足相容性/反对称性
3. `TestAusmUpWallPressureVsExactRiemann` —— 与精确声学解的定量对照
                                            （本文件的核心）
4. `TestPrecondModeScopeIsolation`     —— s_mass / s_pres 各自只影响自己
                                            负责的那一半
"""

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import wall_ghost_state
from autoflowcfd.core.fr_operators.kernels import (
    DEFAULT_PRECOND_MODE,
    PRECOND_LEGACY,
    PRECOND_PHYSICAL,
    PRECOND_PRESSURE_PHYSICAL,
    ausm_precond_mode_label,
    compute_ausm_up_flux,
    resolve_ausm_precond_mode,
)

GAMMA = 1.4
RHO_INF = 1.225
P_INF = 101325.0
U_INF = 33.33
A_INF = np.sqrt(GAMMA * P_INF / RHO_INF)
MACH_REF = U_INF / A_INF
Q_INF = 0.5 * RHO_INF * U_INF ** 2

ALL_MODES = (PRECOND_PHYSICAL, PRECOND_PRESSURE_PHYSICAL, PRECOND_LEGACY)


def _flux(qL, qR, n, mach_ref, mode):
    """按 njit 逐通量点函数的契约调用：一维连续数组。"""
    return np.asarray(compute_ausm_up_flux(
        np.ascontiguousarray(np.asarray(qL, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(qR, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(n, dtype=np.float64)),
        float(mach_ref), int(mode),
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


class TestPrecondModeResolution:
    """`resolve_ausm_precond_mode` 的解析规则。"""

    def test_default_is_physical(self):
        """默认档必须是 physical（标准 AUSM+up）。

        这不是口味选择：legacy 档在固壁上有 7.3 倍虚假超压（见
        TestAusmUpWallPressureVsExactRiemann），把它留作默认等于让
        每个算例都带着这个误差跑。
        """
        assert DEFAULT_PRECOND_MODE == PRECOND_PHYSICAL

    def test_env_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv('AFCFD_AUSM_PRECOND_MODE', raising=False)
        assert resolve_ausm_precond_mode() == DEFAULT_PRECOND_MODE

    def test_env_empty_uses_default(self, monkeypatch):
        """空串/纯空白按"未设置"处理——CI/脚本里 `set VAR=` 很常见。"""
        for raw in ('', '   '):
            monkeypatch.setenv('AFCFD_AUSM_PRECOND_MODE', raw)
            assert resolve_ausm_precond_mode() == DEFAULT_PRECOND_MODE

    @pytest.mark.parametrize('name,expected', [
        ('physical', PRECOND_PHYSICAL),
        ('pressure_physical', PRECOND_PRESSURE_PHYSICAL),
        ('legacy', PRECOND_LEGACY),
        ('LEGACY', PRECOND_LEGACY),
        ('  Physical  ', PRECOND_PHYSICAL),
    ])
    def test_env_names(self, monkeypatch, name, expected):
        monkeypatch.setenv('AFCFD_AUSM_PRECOND_MODE', name)
        assert resolve_ausm_precond_mode() == expected

    def test_explicit_value_overrides_env(self, monkeypatch):
        monkeypatch.setenv('AFCFD_AUSM_PRECOND_MODE', 'legacy')
        assert resolve_ausm_precond_mode('physical') == PRECOND_PHYSICAL
        assert resolve_ausm_precond_mode(PRECOND_PRESSURE_PHYSICAL) == \
            PRECOND_PRESSURE_PHYSICAL

    def test_invalid_raises_not_silently_defaults(self, monkeypatch):
        """非法取值必须报错，不能静默回退到默认值。

        静默回退会让一次拼写错误伪装成"默认行为"，把 A/B 对照的两条
        运行悄悄变成同一档——这正是本项目要求"任何遗漏都在调用点直接
        报错"的原因。
        """
        monkeypatch.setenv('AFCFD_AUSM_PRECOND_MODE', 'phsyical')
        with pytest.raises(ValueError, match='AFCFD_AUSM_PRECOND_MODE'):
            resolve_ausm_precond_mode()
        with pytest.raises(ValueError):
            resolve_ausm_precond_mode(7)

    def test_labels_round_trip(self):
        for mode in ALL_MODES:
            assert resolve_ausm_precond_mode(ausm_precond_mode_label(mode)) == mode


class TestPrecondModeFluxIdentities:
    """三档都必须满足的两条恒等式——它们与声速归一的选择无关是设计要求。"""

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

    @pytest.mark.parametrize('mode', ALL_MODES)
    def test_consistency(self, mode):
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
            F = _flux(q, q, n, MACH_REF, mode)
            worst = max(worst, float(np.max(
                np.abs(F - F_exact) / np.maximum(np.abs(F_exact), 1e-30))))
        assert worst < 1e-11, (
            f"mode={ausm_precond_mode_label(mode)} 相容性最大相对误差 {worst:.3e}"
        )

    @pytest.mark.parametrize('mode', ALL_MODES)
    def test_antisymmetry(self, mode):
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
            Fab = _flux(qA, qB, n, MACH_REF, mode)
            Fba = _flux(qB, qA, -n, MACH_REF, mode)
            worst = max(worst, float(np.max(
                np.abs(Fab + Fba) / np.maximum(np.abs(Fab), 1e-30))))
        assert worst < 1e-12, (
            f"mode={ausm_precond_mode_label(mode)} 反对称性最大相对误差 {worst:.3e}"
        )


class TestAusmUpWallPressureVsExactRiemann:
    """**本文件的核心**：固壁镜像面上的界面压力与精确声学解的定量对照。

    判据来源：镜像状态 (rho, u_n, p) | (rho, -u_n, p) 的精确黎曼解是两道
    对称压缩波，声学（线性化）近似下中间压力

        p* - p = rho * a * u_n

    这是"用镜像黎曼解做固壁"这一标准做法的固有压力升高，不是缺陷；
    但它必须是 `rho*a*u_n` 这个量级，而不是 `p` 本身的量级。
    """

    @pytest.mark.parametrize('frac', [1.0, 0.5, 0.2, 0.05])
    def test_physical_mode_matches_acoustic_exact(self, frac):
        """physical 档的壁面超压必须与 rho*a*u_n 吻合到 10% 以内。"""
        u_n = frac * U_INF
        q_int, q_gho, n = _wall_pair(u_n)
        F = _flux(q_int, q_gho, n, MACH_REF, PRECOND_PHYSICAL)
        # 壁面上质量通量恒为零，故 F[1] = p_half * n_x
        assert abs(F[0]) < 1e-10 * RHO_INF * A_INF, '壁面质量通量必须为零'
        dp = F[1] / n[0] - P_INF
        dp_exact = RHO_INF * A_INF * u_n
        assert abs(dp - dp_exact) < 0.10 * dp_exact, (
            f"u_n/U_inf={frac}: physical 档超压 {dp:.1f} Pa 与精确声学解 "
            f"{dp_exact:.1f} Pa 偏差 {100*abs(dp-dp_exact)/dp_exact:.1f}% > 10%"
        )

    def test_legacy_mode_overpredicts_by_more_than_5x(self):
        """legacy 档的回归钉：它必须仍然表现出那个 >5 倍的过预测。

        这条断言的方向是刻意的——它不是"期望 legacy 正确"，而是把
        已确认的缺陷量级钉住，以免有人"顺手修一下 legacy" 后
        legacy/physical 的 A/B 基线悄悄失效、历史对照数据无法复现。
        """
        q_int, q_gho, n = _wall_pair(U_INF)
        dp_legacy = _flux(q_int, q_gho, n, MACH_REF, PRECOND_LEGACY)[1] / n[0] - P_INF
        dp_exact = RHO_INF * A_INF * U_INF
        assert dp_legacy / dp_exact > 5.0, (
            f"legacy 档过预测倍数 {dp_legacy/dp_exact:.2f} <= 5，"
            f"与 2026-09-16 实测的 7.3 倍不符，基线已变，请重新标定 A/B 对照"
        )

    def test_zero_normal_velocity_gives_exactly_p(self):
        """u_n = 0 时（收敛态的壁面）三档都必须给出 p_half 精确等于 p。"""
        for mode in ALL_MODES:
            q_int, q_gho, n = _wall_pair(0.0)
            dp = _flux(q_int, q_gho, n, MACH_REF, mode)[1] / n[0] - P_INF
            assert abs(dp) < 1e-9 * P_INF, (
                f"mode={ausm_precond_mode_label(mode)}: u_n=0 时 p_half-p={dp:.3e}"
            )

    def test_restoring_gradient_does_not_saturate_in_physical_mode(self):
        """physical 档的 dp_half/du_n 在整个工况范围内必须保持正值。

        这是比"超压大小"更本质的判据：壁面不可穿透条件是靠"法向速度
        增大 => 界面压力增大 => 把流体推回去"这个恢复梯度实现的。
        legacy 档在 P5± 饱和后该梯度归零，不可穿透条件就不再被强制。
        """
        n = np.array([1.0, 0.0, 0.0])
        u_list = np.linspace(0.02 * U_INF, 2.0 * U_INF, 25)
        dp = []
        for u_n in u_list:
            q_int, q_gho, _ = _wall_pair(u_n)
            dp.append(_flux(q_int, q_gho, n, MACH_REF, PRECOND_PHYSICAL)[1] - P_INF)
        slopes = np.diff(dp) / np.diff(u_list)
        assert np.all(slopes > 0.5 * RHO_INF * A_INF), (
            f"physical 档恢复梯度出现饱和/反向：min slope={slopes.min():.1f}，"
            f"应处处 > 0.5*rho*a={0.5*RHO_INF*A_INF:.1f}"
        )

    def test_legacy_restoring_gradient_saturates(self):
        """反向对照：legacy 档在同一区间内必须出现恢复梯度塌陷。

        没有这条，上一条就只是"physical 档没问题"，无法说明
        legacy 档失效的机制确实是饱和。
        """
        n = np.array([1.0, 0.0, 0.0])
        u_list = np.linspace(0.02 * U_INF, 2.0 * U_INF, 25)
        dp = []
        for u_n in u_list:
            q_int, q_gho, _ = _wall_pair(u_n)
            dp.append(_flux(q_int, q_gho, n, MACH_REF, PRECOND_LEGACY)[1] - P_INF)
        slopes = np.diff(dp) / np.diff(u_list)
        # 实测（2026-09-16）：legacy 档的斜率从 u_n 小时的 ~3323（比物理
        # 声阻抗 rho*a=417 还硬 8 倍，因为 1/sqrt(beta2)~10）一路塌到
        # ~86（只有 rho*a 的 0.21 倍），极差比 39 倍。两条判据一起断言，
        # 单看最小值不足以区分"整体偏软"和"随 u_n 塌陷"。
        rho_a = RHO_INF * A_INF
        assert slopes.min() < 0.3 * rho_a, (
            f"legacy 档恢复梯度未塌到物理声阻抗以下："
            f"min slope={slopes.min():.1f}，rho*a={rho_a:.1f}"
        )
        assert slopes.max() / slopes.min() > 20.0, (
            f"legacy 档恢复梯度未出现预期的随 u_n 塌陷："
            f"max/min={slopes.max()/slopes.min():.1f} <= 20"
        )


class TestPrecondModeScopeIsolation:
    """s_mass / s_pres 各自只影响自己负责的那一半。"""

    def test_wall_pressure_identical_between_physical_and_pressure_physical(self):
        """固壁上质量通量恒为零，所以 M4± 那一半用哪个声速都不影响壁面
        压力——physical 与 pressure_physical 必须逐位相同。

        这条同时是"壁面病理 100% 归因于 P5±"这个结论的机器可验证形式。
        """
        for frac in (1.0, 0.5, 0.2, 0.05):
            q_int, q_gho, n = _wall_pair(frac * U_INF)
            F_a = _flux(q_int, q_gho, n, MACH_REF, PRECOND_PHYSICAL)
            F_b = _flux(q_int, q_gho, n, MACH_REF, PRECOND_PRESSURE_PHYSICAL)
            assert np.allclose(F_a, F_b, rtol=0, atol=1e-9 * P_INF), (
                f"u_n/U_inf={frac}: physical 与 pressure_physical 在壁面上不相同"
            )

    def test_pressure_physical_differs_from_legacy_in_pressure_only(self):
        """一般内部面上：pressure_physical 与 legacy 的质量通量必须逐位
        相同（s_mass 相同），压力相关分量必须不同（s_pres 不同）。"""
        qL = np.array([1.2, 20.0, 3.0, -2.0, 101300.0])
        qR = np.array([1.25, 15.0, -1.0, 4.0, 101500.0])
        n = np.array([1.0, 0.0, 0.0])
        F_pp = _flux(qL, qR, n, MACH_REF, PRECOND_PRESSURE_PHYSICAL)
        F_lg = _flux(qL, qR, n, MACH_REF, PRECOND_LEGACY)
        assert F_pp[0] == pytest.approx(F_lg[0], rel=1e-14), \
            '质量通量应与 legacy 逐位相同（s_mass 未改）'
        assert abs(F_pp[1] - F_lg[1]) > 1e-3 * abs(F_lg[1]), \
            '动量通量应因 p_half 不同而可观测地不同'

    def test_physical_mode_flux_independent_of_mach_ref(self):
        """physical 档下 beta2 不再进入通量，故通量必须与 mach_ref 无关
        ——除了 AUSM+up 自带的 fa（它本来就该依赖 mach_ref）。

        因此这里取 Mbar2 明显大于 mach_ref^2 的状态，使 M0_sq 由 Mbar2
        决定、fa 与 mach_ref 解耦，此时不同 mach_ref 必须给出逐位相同
        的通量。
        """
        qL = np.array([1.2, 200.0, 0.0, 0.0, 101300.0])
        qR = np.array([1.2, 180.0, 0.0, 0.0, 101500.0])
        n = np.array([1.0, 0.0, 0.0])
        ref = _flux(qL, qR, n, 0.05, PRECOND_PHYSICAL)
        for mach_ref in (0.1, 0.2, 0.3, 0.5):
            F = _flux(qL, qR, n, mach_ref, PRECOND_PHYSICAL)
            assert np.allclose(F, ref, rtol=1e-15, atol=0), (
                f"physical 档下 mach_ref={mach_ref} 改变了通量，"
                f"说明 beta2 仍在通量内部生效"
            )
