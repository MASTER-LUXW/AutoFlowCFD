"""矩阵自由 Newton-Krylov（JFNK）稳态求解器的数值正确性。

用**有解析解**的合成非线性系统单独验证 JFNK 本身，与 FR 残差链路解耦：
这样一旦真实算例上隐式路径出问题，可以立刻分清是 JFNK 的数学错了还是
残差/几何那一侧的问题。

合成系统的变量量级刻意按本项目真实守恒变量构造
（`[rho, rho_u, rho_v, rho_w, rho_E] ~ [1.225, 40, 40, 40, 2.5e5]`），
因为"量级横跨 5 个数量级"正是 Fréchet 差分步长取法要处理的情形。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit import (
    EisenstatWalkerForcing,
    step_newton_krylov,
)
from autoflowcfd.core.time_integration.implicit import jacobian_vector as JV

#: 与本项目真实来流条件同量级的参考量级
_SCALES = np.array([1.225, 40.0, 40.0, 40.0, 2.5e5])
_N = 200
_NV = 5

#: 伪瞬态项趋零（纯 Newton）用的 dtau。取 1e12 而不是 inf：`I/dtau` 在
#: float64 下是 1e-12，相对 `J` 的量级（O(1)~O(1e4)）可忽略，同时避免
#: 除零。
_DTAU_PURE_NEWTON = 1e12


def _linear_system(seed=7):
    """`R(U) = A (U - U_star)`，`A` 良态且变量间耦合、行列按量级缩放。"""
    rng = np.random.default_rng(seed)
    a_var = rng.normal(size=(_NV, _NV)) + _NV * np.eye(_NV)
    a = a_var * _SCALES[:, None] * (1.0 / _SCALES)[None, :]
    u_star = _SCALES[None, :] * (1.0 + 0.1 * rng.normal(size=(_N, _NV)))

    def residual(u_flat):
        return (u_flat - u_star) @ a.T

    return residual, u_star, a


def _rms(x):
    return float(np.linalg.norm(x) / np.sqrt(x.size))


class TestLinearProblem:
    """线性 `R` 上，Newton 步的精度应当由线性求解容差 `eta` 决定。"""

    @pytest.mark.parametrize("eta_target,min_drop", [
        (1.0e-10, 1.0e6),   # 解准 -> 一步基本到解
        (1.0e-1, 5.0),      # 只解一位 -> 残差掉一个量级左右
    ])
    def test_residual_drop_tracks_linear_tolerance(self, eta_target,
                                                   min_drop):
        """**这条就是 `eta` 的契约**：线性问题上一个 Newton 步把残差降低
        的倍数应当约为 `1/eta`。

        （方法论记录：第一版把这条写成"线性问题必须一步到机器精度"，
        那是**判据写错了** —— 当时没传 `forcing`，`eta` 走的是固定的
        0.1，于是只解了一位有效数字、残差只掉 13 倍，测试"失败"其实是
        代码行为正确、期望错误。`eta` 是 inexact Newton 刻意留的自由度，
        不是缺陷。）
        """
        residual, _u_star, _a = _linear_system()
        u = _SCALES[None, :] * np.ones((_N, _NV))
        dtau = np.full(_N, _DTAU_PURE_NEWTON)

        class _FixedEta:
            def next_eta(self, res_norm, tol_nonlinear):
                return eta_target

        r0 = _rms(residual(u))
        u1, info = step_newton_krylov(
            residual, u, dtau, _SCALES, forcing=_FixedEta(),
            gmres_max_iter=400, gmres_restart=60)
        r1 = _rms(residual(u1))
        assert info["theta"] == pytest.approx(1.0), (
            f"线性良态问题上不该触发物理性限幅，theta={info['theta']}")
        drop = r0 / max(r1, 1e-300)
        assert drop > min_drop, (
            f"eta={eta_target:.1e} 下残差只降了 {drop:.3e} 倍，"
            f"低于期望的 {min_drop:.1e} 倍")


class TestQuadraticConvergence:
    """非线性问题上，连续 Newton 步必须体现超线性/二次收敛。"""

    def test_residual_drops_superlinearly(self):
        """判据是"最后一步的下降比值远小于前几步" —— 这正是二次收敛的
        可观测特征，而且它对 inexact Newton 同样成立（`eta` 随残差下降
        自动收紧，见 `forcing.py`）。

        实测（seed 固定）：`||R||` 走
        1.18e4 -> 9.82e2 -> 3.92e0 -> 2.52e-06 -> **恰好 0.0**，
        比值序列 8.3e-02 / 4.0e-03 / 6.4e-07 —— 四步到机器零。

        （这组数字是 `_ETA_MAX` 从 0.9 压到 0.1 之后测的。0.9 那一档同一
        算例要 6 步、且只到 2.8e-07；把上界压紧的理由与真实运行证据见
        `forcing.py` 里 `_ETA_MAX` 上方那节。）
        """
        rng = np.random.default_rng(7)
        a_var = rng.normal(size=(_NV, _NV)) + _NV * np.eye(_NV)
        a = a_var * _SCALES[:, None] * (1.0 / _SCALES)[None, :]
        u_star = _SCALES[None, :] * (1.0 + 0.1 * rng.normal(size=(_N, _NV)))

        def residual(u_flat):
            d = u_flat - u_star
            dn = d / _SCALES[None, :]
            return d @ a.T + 0.3 * _SCALES[None, :] * dn ** 2

        u = u_star * (1.0 + 0.02 * rng.normal(size=(_N, _NV)))
        forcing = EisenstatWalkerForcing()
        dtau = np.full(_N, _DTAU_PURE_NEWTON)

        prev = _rms(residual(u))
        assert prev > 0.0
        ratios = []
        n_steps = 0
        for _ in range(6):
            u, info = step_newton_krylov(
                residual, u, dtau, _SCALES, forcing=forcing,
                gmres_max_iter=400, gmres_restart=60)
            n_steps += 1
            # 良态问题上 Newton 方向不该被物理性限幅或残差接受判据收紧
            assert info["theta"] == pytest.approx(1.0), (
                f"第 {n_steps} 步 theta={info['theta']}（限幅前 "
                f"{info['theta_physicality']}）—— 良态合成问题上不该收紧")
            cur = _rms(residual(u))
            ratios.append(cur / max(prev, 1e-300))
            prev = cur
            if prev == 0.0:
                break        # 已到机器零，再迭代是无操作（下面单独断言）

        assert prev == 0.0 or prev < 1.0e-5, (
            f"{n_steps} 步之后残差仍有 {prev:.3e}")
        assert ratios[-1] < 1.0e-4, (
            f"最后一步的下降比值 {ratios[-1]:.3e} 不够小 —— 二次收敛"
            f"应当在接近解时出现数量级式的下降。完整比值序列 "
            f"{['%.2e' % r for r in ratios]}")
        assert ratios[-1] < 1.0e-3 * ratios[0], (
            f"下降比值没有随迭代显著收紧（首 {ratios[0]:.3e}、"
            f"末 {ratios[-1]:.3e}），不是超线性收敛的形态")
        assert n_steps <= 5, (
            f"良态合成问题上用了 {n_steps} 个 Newton 步才收敛 —— "
            f"实测应当是 4 步，超出说明线性容差或 forcing term 退化了")

    def test_converged_state_is_a_noop(self):
        """残差已到机器零之后，再调用一步必须是**无操作**。

        这条是上面那个循环的补充：`theta=0` 在"已收敛"与"这一步解不出来"
        两种情形下同值，所以必须分别钉住。这里钉的是前者 —— 判据是
        `n_matvec == 0`（连一次 J.v 都不做），而不只是 `theta == 0`。
        """
        u = _SCALES[None, :] * np.ones((_N, _NV))

        def residual(u_flat):
            return np.zeros_like(u_flat)

        u1, info = step_newton_krylov(
            residual, u, np.full(_N, _DTAU_PURE_NEWTON), _SCALES)
        assert info["theta"] == 0.0 and info["n_matvec"] == 0
        assert np.array_equal(u1, u)


class TestPhysicalityLimiter:
    """物理性限幅必须在 Newton 方向指向非物理态时真正收紧步长。"""

    def test_theta_shrinks_when_direction_kills_density(self):
        """构造一个"把密度打到负值"的残差，逐单元松弛因子必须 < 1 且
        推进后密度仍为正。

        这不是假想情形：Newton 方向在远离解时完全可以指向 `rho<0`，
        而 `state._update_primitives()` 的 `rho>=1e-10` 钳制只保护原始
        变量的计算、不阻止 `U` 本身变成非物理（那会让下一次残差求值
        算在垃圾态上并污染整个 Krylov 基）。
        """
        u0 = np.tile(_SCALES, (_N, 1)).copy()

        def residual(u_flat):
            # R = +k * (U - U_target)，其中 U_target 的密度是负的 ->
            # Newton 方向 dU = -J^{-1} R 指向 U_target
            target = u0.copy()
            target[:, 0] = -2.0 * _SCALES[0]
            return (u_flat - target) * 1.0

        dtau = np.full(_N, _DTAU_PURE_NEWTON)
        u1, info = step_newton_krylov(
            residual, u0, dtau, _SCALES, gmres_max_iter=200,
            gmres_restart=50)
        assert 0.0 < info["theta_physicality"] < 1.0, (
            f"Newton 方向把密度打到负值，松弛因子应当被收紧到 (0,1)，"
            f"实际 {info['theta_physicality']}")
        assert info["limited_fraction"] == 1.0, "每个点的方向都指向负密度，应全部被松弛"
        assert np.all(u1[:, 0] > 0.0), "限幅之后密度仍然出现非正值"
        # 相对变化不超过设定上限（留一点浮点余量）
        rel = np.max(np.abs(u1[:, 0] - u0[:, 0]) / u0[:, 0])
        from autoflowcfd.core.time_integration.implicit import (
            PHYSICALITY_MAX_RELATIVE_CHANGE,
        )
        assert rel <= PHYSICALITY_MAX_RELATIVE_CHANGE * (1.0 + 1e-9)


class TestFrechetEpsScaling:
    """`J v` 的差分步长取法：逐变量无量纲化 vs 单标量 eps 的**实测**对照。

    这里把 `jacobian_vector.py` 文档里那张表钉成可执行判据。它同时是
    一条**如实记录**：无量纲化的收益是 4~13 倍（在集中于小量级变量的
    方向上），不是"另一种不可用"——`rho_E` 主导的方向上它反而差约 1.6
    倍。两者都在 1e-9 量级、都远低于 inexact-Newton 的容差下限。
    """

    @staticmethod
    def _nonlinear_with_analytic_jacobian(seed=11):
        """非线性 `R` + 它在给定基态处的**解析** `J v`。

        必须用非线性 `R`：线性 `R` 没有截断误差，`eps` 越大舍入越小，
        于是"单标量 eps 取了个更大的 eps"反而赢 —— 第一版对照就是这么
        得出相反结论的，是测试方法论错误。
        """
        rng = np.random.default_rng(seed)
        a_var = rng.normal(size=(_NV, _NV)) + _NV * np.eye(_NV)
        a = a_var * _SCALES[:, None] * (1.0 / _SCALES)[None, :]
        u_star = _SCALES[None, :] * (
            1.0 + 0.05 * rng.normal(size=(_N, _NV)))

        def residual(u_flat):
            d = (u_flat - u_star) / _SCALES[None, :]
            return (u_flat - u_star) @ a.T + 0.5 * _SCALES[None, :] * d ** 2

        u_base = u_star * (1.0 + 0.03 * rng.normal(size=(_N, _NV)))

        def jv_exact(v):
            d = (u_base - u_star) / _SCALES[None, :]
            return v @ a.T + d * v

        return residual, u_base, jv_exact

    class _ScalarEpsJacobian(JV.MatrixFreeJacobian):
        """对照实现：所有变量共用一个**物理**标量 eps（不无量纲化）。"""

        def matvec(self, v_flat):
            v_flat = np.ascontiguousarray(v_flat, dtype=np.float64)
            vn = float(np.linalg.norm(v_flat))
            if vn == 0.0:
                return np.zeros_like(v_flat)
            u0n = float(np.linalg.norm(self._u0))
            eps = JV._SQRT_EPS * (1.0 + u0n) / vn
            r_pert = self._residual(self._u0 + eps * v_flat)
            return (np.asarray(r_pert) - self._r0) / eps

    def test_both_are_accurate_enough_but_scaled_wins_on_small_variables(self):
        residual, u_base, jv_exact = self._nonlinear_with_analytic_jacobian()
        r_base = residual(u_base)
        scaled = JV.MatrixFreeJacobian(residual, u_base, r_base, _SCALES)
        plain = self._ScalarEpsJacobian(residual, u_base, r_base, _SCALES)

        def rel_err(j, v):
            ex = jv_exact(v)
            return float(np.max(np.abs(j.matvec(v) - ex))
                         / max(np.max(np.abs(ex)), 1e-300))

        # 集中在 rho / 动量（小量级变量）的方向上，无量纲化必须更好
        for k in range(4):
            v = np.zeros((_N, _NV))
            v[:, k] = 1.0
            e_scaled, e_plain = rel_err(scaled, v), rel_err(plain, v)
            assert e_scaled < e_plain, (
                f"变量 {k} 方向上逐变量无量纲化 {e_scaled:.3e} 没有优于"
                f"单标量 eps {e_plain:.3e}")

        # 两者都必须远低于 inexact-Newton 的容差下限（否则差分误差就会
        # 成为收敛的限制因素）
        from autoflowcfd.core.time_integration.implicit.forcing import (
            _ETA_MIN,
        )
        v = _SCALES[None, :] * np.random.default_rng(3).normal(
            size=(_N, _NV))
        for name, j in (("scaled", scaled), ("plain", plain)):
            e = rel_err(j, v)
            assert e < 0.01 * _ETA_MIN, (
                f"{name} 的 J.v 相对误差 {e:.3e} 已经接近 eta 下限 "
                f"{_ETA_MIN:.1e}，差分步长取法需要重新标定")


class TestInputValidation:
    """非法输入必须硬失败，不静默钳制（项目一贯要求）。"""

    def test_nonpositive_dtau_raises(self):
        residual, _u_star, _a = _linear_system()
        u = _SCALES[None, :] * np.ones((_N, _NV))
        dtau = np.full(_N, 1.0e-6)
        dtau[3] = 0.0
        with pytest.raises(ValueError, match="非正值"):
            step_newton_krylov(residual, u, dtau, _SCALES)

    def test_scales_length_mismatch_raises(self):
        residual, _u_star, _a = _linear_system()
        u = _SCALES[None, :] * np.ones((_N, _NV))
        dtau = np.full(_N, 1.0e-6)
        with pytest.raises(ValueError, match="参考量级长度"):
            step_newton_krylov(residual, u, dtau, _SCALES[:3])

    def test_zero_residual_is_a_noop(self):
        u = _SCALES[None, :] * np.ones((_N, _NV))

        def residual(_u_flat):
            return np.zeros((_N, _NV))

        u1, info = step_newton_krylov(
            residual, u, np.full(_N, 1.0e-6), _SCALES)
        assert info["res_norm"] == 0.0
        assert info["n_matvec"] == 0, "残差已为零时不该再做任何 J.v"
        assert np.array_equal(u1, u)


def _step_limited_system(delta, seed=7):
    """`R` 在"离基态超过 `delta`（无量纲 RMS 位移）"之外整体放大 1e6 倍。

    用途：造出一个**方向不可信、但缩小 `dtau` 就可信**的情形，这正是
    PTC 缩 `dtau` 存在的理由（`implicit/dtau_control.py`）。

    为什么用一个不连续的"远支"而不是某个光滑强非线性：要钉住的是
    "一步被残差判据拒绝之后会不会缩 `dtau` 重试"这条**控制逻辑**，
    它需要的是"大步一定被拒、小步一定被接受"这个确定性，而光滑非线性
    要靠调参数去凑这个分界（调出来的分界还会随 seed 漂）。这里不连续
    是刻意的：它模拟"越过这一步残差求值就是垃圾"，而 Fréchet 差分用的
    `eps ~ 1e-8` 始终落在近支，所以 `J` 仍是近支的真实 Jacobian。
    """
    rng = np.random.default_rng(seed)
    a_var = rng.normal(size=(_NV, _NV)) + _NV * np.eye(_NV)
    a = a_var * _SCALES[:, None] * (1.0 / _SCALES)[None, :]
    u_star = _SCALES[None, :] * (1.0 + 0.1 * rng.normal(size=(_N, _NV)))
    u0 = _SCALES[None, :] * np.ones((_N, _NV))

    def residual(u_flat):
        d = (u_flat - u0) / _SCALES[None, :]
        base = (u_flat - u_star) @ a.T
        if float(np.sqrt(np.mean(d ** 2))) > delta:
            return base * 1.0e6
        return base

    return residual, u0


class TestPtcDtauScaleStateMachine:
    """`dtau_control.PtcDtauScale` 的策略本身（与残差链路无关）。"""

    def test_cut_shrinks_by_one_notch_until_floor(self):
        from autoflowcfd.core.time_integration.implicit import dtau_control as DC

        ctrl = DC.PtcDtauScale()
        assert ctrl.scale == 1.0
        assert ctrl.cut() is True
        assert ctrl.scale == pytest.approx(DC.FAIL_SHRINK)
        for _ in range(200):
            if not ctrl.cut():
                break
        assert ctrl.scale == pytest.approx(DC.MIN_SCALE)
        assert ctrl.cut() is False, "已到下限时必须如实返回 False"

    def test_reward_only_on_a_fully_accepted_step(self):
        from autoflowcfd.core.time_integration.implicit import dtau_control as DC

        ctrl = DC.PtcDtauScale(0.25)
        ctrl.reward(theta=0.5)   # 被回溯过
        assert ctrl.scale == pytest.approx(0.25)
        ctrl.reward(theta=1.0)   # 完整接受（逐单元物理性松弛不参与，见 reward 文档）
        assert ctrl.scale == pytest.approx(0.25 * DC.OK_GROW)

    def test_scale_never_exceeds_one(self):
        """`dtau` 的天花板由外层自适应 CFL 给（唯一事实来源），本层只在
        它之下缩放 —— 所以放大永远封顶在 1.0。"""
        from autoflowcfd.core.time_integration.implicit import dtau_control as DC

        ctrl = DC.PtcDtauScale(0.5)
        for _ in range(10):
            ctrl.reward(theta=1.0)
        assert ctrl.scale == 1.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_initial_scale_raises(self, bad):
        from autoflowcfd.core.time_integration.implicit import PtcDtauScale

        with pytest.raises(ValueError, match="正有限值"):
            PtcDtauScale(bad)


class TestDtauCutOnRejectedStep:
    """**这组钉住的是一个真实运行抓到的停滞**（完整记录见
    `implicit/dtau_control.py` 模块文档）：固定 CFL=10 下 Blasius 从
    step 48 起每步 `theta=0`、残差逐位不变、每步照烧 38 次残差求值，
    一直到第 200 步 —— 因为"交给外层自适应 CFL"这条交接在残差不变时
    原理上传不到控制器那里。"""

    def test_rejected_direction_cuts_dtau_and_still_advances(self):
        """实测（seed 固定）：`dtau=1` + `delta=1e-3` 下缩 3 档、
        `theta=0.125`，残差 5.9057e4 -> 5.8420e4 —— 也就是从"原地不动"
        变成"真的走了一步且残差下降"。"""
        residual, u0 = _step_limited_system(1.0e-3)
        u1, info = step_newton_krylov(
            residual, u0, np.full(_N, 1.0), _SCALES)

        assert info["n_dtau_cuts"] >= 1, "大步被拒时必须缩 dtau 重试"
        assert info["theta"] > 0.0, "缩过 dtau 之后这一步必须真的前进"
        assert info["dtau_scale"] < 1.0
        assert not np.array_equal(u1, u0)
        assert info["res_norm_new"] < info["res_norm"]

    def test_unused_notches_carry_over_to_the_next_call(self):
        """单步内的档数上限（`DTAU_MAX_CUTS_PER_STEP`）限制的是**单步
        成本**，不是总档数：用不完的档由下一次调用带着 `dtau_scale`
        继续缩。"""
        from autoflowcfd.core.time_integration.implicit import (
            DTAU_MAX_CUTS_PER_STEP,
        )
        from autoflowcfd.core.time_integration.implicit import dtau_control as DC

        residual, u0 = _step_limited_system(1.0e-3)
        dtau = np.full(_N, 1.0e12)   # 纯 Newton 档：缩 3 档远远不够

        _u1, info1 = step_newton_krylov(residual, u0, dtau, _SCALES)
        assert info1["theta"] == 0.0
        assert info1["n_dtau_cuts"] == DTAU_MAX_CUTS_PER_STEP
        assert info1["dtau_scale"] == pytest.approx(
            DC.FAIL_SHRINK ** DTAU_MAX_CUTS_PER_STEP)

        _u2, info2 = step_newton_krylov(
            residual, u0, dtau, _SCALES, dtau_scale=info1["dtau_scale"])
        assert info2["dtau_scale"] < info1["dtau_scale"], (
            "第二次调用必须从上一次缩到的档位继续，而不是回到 1.0")

    def test_floor_reached_raises_instead_of_spinning(self):
        """缩到下限还拿不到被接受的步 = 不是步长问题。此时**硬失败**，
        不每步照烧一次完整 GMRES 求解原地打转（那就是换个位置的停滞）。
        """
        residual, u0 = _step_limited_system(1.0e-3)
        dtau = np.full(_N, 1.0e12)
        scale = 1.0
        n_calls = 0
        with pytest.raises(RuntimeError, match="缩到下限"):
            for _ in range(20):
                _u, info = step_newton_krylov(
                    residual, u0, dtau, _SCALES, dtau_scale=scale)
                scale = info["dtau_scale"]
                n_calls += 1
        # 抛错发生在"缩到下限"的那一次调用里（它不返回），所以这里拿到的
        # 是上一次返回的档位 —— 实测 5 次调用、最后一次返回 5.96e-08。
        assert n_calls == 4
        assert scale < 1.0e-6

    def test_clean_step_grows_the_scale_back(self):
        """一次失败不能把 `dtau` 永久压低（否则隐式路径退化成显式）：
        良态问题上一步被完整接受就涨一档。"""
        residual, _u_star, _a = _linear_system()
        u = _SCALES[None, :] * np.ones((_N, _NV))
        dtau = np.full(_N, _DTAU_PURE_NEWTON)

        _u1, info = step_newton_krylov(
            residual, u, dtau, _SCALES, dtau_scale=0.0625)
        assert info["theta"] == pytest.approx(1.0)
        assert info["n_dtau_cuts"] == 0
        assert info["dtau_scale"] == pytest.approx(0.125)


def test_physicality_relaxation_is_cellwise_not_global():
    """一个单元想把正值场降掉 99%，只有它自己被松弛；其余单元照常走完整的
    Newton 步。2026-09-25 以前是全场取最小的一个 theta：plate_demo P0+SST 上
    一个单元让湍流 Newton 的 theta 掉到 7.9e-5，整个湍流场随之冻结。"""
    from autoflowcfd.core.time_integration.implicit.jfnk import (
        PHYSICALITY_MAX_RELATIVE_CHANGE as C, positive_fields_row_limits, step_newton_krylov,
    )

    n_cells, rows_per_cell = 6, 3
    n = n_cells * rows_per_cell
    target = np.full((n, 2), 1.2)
    target[:rows_per_cell] = 0.01            # 单元 0 的目标：下降 99%
    u0 = np.ones((n, 2))

    def residual(u):                          # 线性、逐点解耦：Newton 一步到位
        return u - target

    u1, info = step_newton_krylov(
        residual, u0, np.full(n, 1e12), np.ones(2), physicality=positive_fields_row_limits,
        rows_per_cell=rows_per_cell, gmres_max_iter=50)

    alpha0 = C / 0.99
    np.testing.assert_allclose(u1[:rows_per_cell], 1.0 - alpha0 * 0.99, rtol=1e-8)
    np.testing.assert_allclose(u1[rows_per_cell:], 1.2, rtol=1e-8)
    assert info["theta"] == 1.0
    assert info["theta_physicality"] == pytest.approx(alpha0)
    assert info["limited_fraction"] == pytest.approx(1.0 / n_cells)


def test_positive_field_relaxation_bounds_increase_too():
    """单步变化是双向（对数对称）约束：想把 k 放大 1000 倍的单元只被放大
    1/(1-c) 倍。"""
    from autoflowcfd.core.time_integration.implicit.jfnk import (
        PHYSICALITY_MAX_RELATIVE_CHANGE as C, positive_fields_row_limits, step_newton_krylov,
    )

    n = 4
    target = np.ones((n, 2))
    target[0] = 1000.0
    u1, info = step_newton_krylov(
        lambda u: u - target, np.ones((n, 2)), np.full(n, 1e12), np.ones(2),
        physicality=positive_fields_row_limits, rows_per_cell=1, gmres_max_iter=50)
    np.testing.assert_allclose(u1[0], 1.0 / (1.0 - C), rtol=1e-10)
    np.testing.assert_allclose(u1[1:], 1.0, rtol=1e-12)
    assert info["limited_fraction"] == pytest.approx(0.25)
