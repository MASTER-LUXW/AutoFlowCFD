"""JFNK：伪时间步长状态机、拒步后的 dtau 缩减、逐单元物理性松弛与局部 dtau 缩放（从 test_implicit_jfnk.py 拆出）。"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit import step_newton_krylov

from tests.unit._implicit_jfnk_common import (
    _DTAU_PURE_NEWTON,
    _N,
    _NV,
    _SCALES,
    _linear_system,
    _step_limited_system,
)


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
    from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov
    from autoflowcfd.core.time_integration.implicit.physicality import (
        PHYSICALITY_MAX_RELATIVE_CHANGE as C, ScaledFieldRowLimits,
    )

    n_cells, rows_per_cell = 6, 3
    n = n_cells * rows_per_cell
    target = np.full((n, 2), 1.2)
    target[:rows_per_cell] = 0.01            # 单元 0 的目标：下降 99%
    u0 = np.ones((n, 2))

    def residual(u):                          # 线性、逐点解耦：Newton 一步到位
        return u - target

    u1, info = step_newton_krylov(
        residual, u0, np.full(n, 1e12), np.ones(2), physicality=ScaledFieldRowLimits([1e-12, 1e-12]),
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
    from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov
    from autoflowcfd.core.time_integration.implicit.physicality import (
        PHYSICALITY_MAX_RELATIVE_CHANGE as C, ScaledFieldRowLimits,
    )

    n = 4
    target = np.ones((n, 2))
    target[0] = 1000.0
    u1, info = step_newton_krylov(
        lambda u: u - target, np.ones((n, 2)), np.full(n, 1e12), np.ones(2),
        physicality=ScaledFieldRowLimits([1e-12, 1e-12]), rows_per_cell=1, gmres_max_iter=50)
    np.testing.assert_allclose(u1[0], 1.0 / (1.0 - C), rtol=1e-10)
    np.testing.assert_allclose(u1[1:], 1.0, rtol=1e-12)
    assert info["limited_fraction"] == pytest.approx(0.25)


def test_scaled_field_limits_cross_the_floor_gradually():
    """尺度下限附近变成绝对限幅：值远小于尺度下限的解点每步可以移动 c*scale，
    可以越过零（被输运的 k/omega 不裁剪，见 `turbulence/sst/bounds.py`）——纯相对
    限幅在这里会冻结整个单元。"""
    from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov
    from autoflowcfd.core.time_integration.implicit.physicality import (
        PHYSICALITY_MAX_RELATIVE_CHANGE as C, ScaledFieldRowLimits,
    )

    scale = 1e-3
    target = np.array([[-1e-2, 1.0], [1.0, 1.0]])
    u0 = np.array([[1e-6, 1.0], [1.0, 1.0]])
    u1, info = step_newton_krylov(
        lambda u: u - target, u0, np.full(2, 1e12), np.ones(2),
        physicality=ScaledFieldRowLimits([scale, scale]), rows_per_cell=1, gmres_max_iter=50)
    np.testing.assert_allclose(u1[0, 0], 1e-6 - C * scale, rtol=1e-10)
    assert u1[0, 0] < 0.0
    np.testing.assert_allclose(u1[1], 1.0, rtol=1e-12)


def test_krylov_on_real_rows_matches_full_system():
    """Krylov 向量只存真实行（`real_rows`）与全尺寸求解给出同一个 Newton 步；
    零填充行残差不为零时拒绝（紧凑化的等价前提）。"""
    from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov

    rng = np.random.default_rng(11)
    n_cells, n_sps, n_var = 30, 4, 2
    real = np.tile([True, True, True, False], n_cells)          # 每单元 1 个零填充槽位
    n = n_cells * n_sps
    A = np.eye(n) * 4.0 + 0.3 * rng.standard_normal((n, n))
    A[~real, :] = 0.0
    A[:, ~real] = 0.0                                           # 零填充行列 Jacobian 为零
    t = rng.standard_normal((n, n_var))

    def residual(u):
        return np.einsum("ij,jv->iv", A, u - t)

    u0 = np.zeros((n, n_var))
    kw = dict(gmres_max_iter=200, rows_per_cell=1, physicality=lambda u, du, red: np.ones(u.shape[0]))
    u_full, info_full = step_newton_krylov(residual, u0, np.full(n, 1e3), np.ones(n_var), **kw)
    u_real, info_real = step_newton_krylov(residual, u0, np.full(n, 1e3), np.ones(n_var), real_rows=real, **kw)
    np.testing.assert_allclose(u_real, u_full, rtol=1e-10, atol=1e-12)
    assert np.all(u_real[~real] == 0.0)
    assert info_real["gmres_iters"] == info_full["gmres_iters"]

    with pytest.raises(ValueError, match="零填充"):
        step_newton_krylov(lambda u: residual(u) + 1.0, u0, np.full(n, 1e3), np.ones(n_var),
                           real_rows=real, **kw)


def test_scaled_field_limits_use_cell_magnitude():
    """限幅基准与单元真实解点的 |u| 均值取大：落在单元多项式振荡低谷的解点（值近零）
    按单元量级移动，不被自己的点值冻结；零填充槽位不计入均值。"""
    from autoflowcfd.core.time_integration.implicit.physicality import (
        PHYSICALITY_MAX_RELATIVE_CHANGE as C, ScaledFieldRowLimits,
    )
    from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions

    # 单元 0：解点 [1e-4, 10, 20]，槽位 3 为零填充（值 1e6 不得计入）；单元 1：[1, 1, 1, pad]
    u0 = np.array([[1e-4], [10.0], [20.0], [1e6], [1.0], [1.0], [1.0], [0.0]])
    real = np.array([True, True, True, False] * 2)
    du = np.full((8, 1), -10.0)
    lim = ScaledFieldRowLimits([1e-3], rows_per_cell=4, real_rows=real)(u0, du, LocalReductions(np))
    cell0 = (1e-4 + 10.0 + 20.0) / 3.0
    # 解点 0、1 按单元均值（均大于点值），解点 2 按自身 20 不受限
    np.testing.assert_allclose(lim[:3], [C * cell0 / 10.0, C * cell0 / 10.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(lim[4:7], C * 1.0 / 10.0, rtol=1e-12)
    # 不给单元结构时退回逐点基准
    lim_pt = ScaledFieldRowLimits([1e-3])(u0, du, LocalReductions(np))
    np.testing.assert_allclose(lim_pt[0], C * 1e-3 / 10.0, rtol=1e-12)


def test_local_dtau_scale_cuts_relaxed_rows_and_recovers_the_rest():
    """被物理性松弛的行下一步降局部 dtau（乘以 max(alpha, 下限)），未松弛的行按固定
    倍数恢复到 1（见 physicality.py 的 LOCAL_DTAU_* 说明）。"""
    from autoflowcfd.core.time_integration.implicit.physicality import (
        LOCAL_DTAU_CUT_MIN, LOCAL_DTAU_FLOOR, LOCAL_DTAU_GROW, update_local_dtau_scale,
    )

    scale = np.array([1.0, 1.0, 0.5, 0.01, 1e-8])
    alpha = np.array([1.0, 0.3, 1.0, 1e-6, 1e-3])
    new = update_local_dtau_scale(scale, alpha, np)
    np.testing.assert_allclose(new, [1.0, 0.3, min(0.5 * LOCAL_DTAU_GROW, 1.0),
                                     0.01 * LOCAL_DTAU_CUT_MIN, LOCAL_DTAU_FLOOR])


def test_newton_step_returns_and_applies_local_dtau_scale():
    """局部缩放随 info 返回，下一次调用传回时逐行作用在 dtau 上：缩放为 0 附近的行
    几乎不动（显式极限），其余行照常收敛。"""
    from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov

    n = 40
    target = np.tile([1.2, 0.3, 0.0, 0.0, 2.6], (n, 1))

    def R(u):
        return u - target

    u0 = np.tile([1.0, 0.0, 0.0, 0.0, 2.5], (n, 1))
    scale = np.ones(n)
    scale[:5] = 1e-8
    u1, info = step_newton_krylov(R, u0, np.full(n, 1e6), np.ones(5), local_dtau_scale=scale)
    assert info["local_dtau_scale"].shape == (n,)
    assert np.abs(u1[:5] - u0[:5]).max() < 1e-1 * np.abs(target[:5] - u0[:5]).max()
    # 线性求解按 inexact Newton 的默认容差（eta=0.1）停止：其余行一步收缩到初差的 1/10 以内
    assert np.abs(u1[5:] - target[5:]).max() <= 0.1 * np.abs(u0[5:] - target[5:]).max()
