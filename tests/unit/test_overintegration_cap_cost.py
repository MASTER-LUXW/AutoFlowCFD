"""
过积分上限 `OVERINTEGRATION_MAX_ORDER = 3` 在 P2/P3 上到底截断了多少？

## 为什么需要量这个

"P2 理想 over_order=4 被压到 3、P3 理想 6 被压到 3（= order，过积分完全
失效）"——这条结论此前只是按经验法则 `rule*order` 推出来的**定性**说法，
没有人量过它的**代价**。而放开这个上限看起来要改 `jacobians_fine` 的布局
与全部消费点，是个跨多文件的改动——**先量收益再付代价**。

## 后续（2026-09-17）：本文件量出的收益已经被兑现

四面体的上限已按单元类型独立出去（`fr/native_tet_overintegration.py::
NATIVE_TET_OVERINTEGRATION_MAX_ORDER = 6`，env `AFCFD_TET_OVERINT_MAX_ORDER`），
实际取到 P1 oo=2 / P2 oo=4（**完整达到理想**）/ P3 oo=5。"要改 jacobians_fine
布局"那条前提被推翻了：直边四面体的细点度量是一个逐单元常数的原样广播，
取前 `n_fine_tet` 列与在真实细点上求值恒等，不需要加宽任何数组。完整说明与
真实网格实测见 `test_tet_overintegration_cap_raised.py`。

所以下面表格里 P2 的 3->4 与 P3 的 3->6 现在是**已实现的收益**（P3 受布局
夹到 5，差理想约 4 倍），不再是"若放开则可得"。本文件的作用相应变成：钉住
各档去混叠误差的量级，让任何人改动过积分算子实现后立刻看到精度变化。
**`OVERINTEGRATION_MAX_ORDER` 现在只约束棱柱**——下面几处用它来表示"当前
上限"的地方保留原样，是为了记录量这些数时的历史状态。

## 算例设计（含一处必须避开的陷阱）

`test_native_tet_overintegration_aliasing_reduction.py` 用线性 Couette
剖面 + "解析残差恒为 0"做判据。那个算例量**不出**上限的代价：速度是线性
的，能量通量 `u*(E+p)` 只有三次，`over_order=3` 已经精确捕捉；实测 P2 的
over_order 3 与 4 在**舍入水平上不可分辨**（两者中位误差都报 1.1101e3，
逐点相对差 ~5e-10），P3 的 4/5/6 与不过积分完全一样（1.00x）。

（措辞更正：先前把这条记成"逐位相同"。严格说不是——它们差在 1e-10 相对
量级，那是舍入，不是零。本文件的阈值按 1e-8 设，落在舍入区间内。）

要量出代价，通量的非线性次数必须真正超过 3。而该文件文档记录过一个真实
陷阱：**不能**把 Q 的能量分量按 `0.5*rho*u^2` 从 u 反推——那样能量分量
次数是 `2*order`，在粗节点采样时**自身**就已丢信息，与"F_phys 非线性导致
的次数提升"混在一起，测出的数完全没有意义。

本文件的做法：Q 的**每个守恒分量**各自取次数恰为 `order` 的多项式（粗
空间精确可表示，零采样损失），非线性只来自 `euler_physical_flux_batch`
里的 `u = rho_u/rho`、`p = f(rho, rho_u, rho_E)` 这些**有理**运算——那
才是过积分真正要处理的对象。因为解析散度不再恒为零，判据改为"以
`over_order=8` 的结果为参照，量各档的相对误差"。

## 实测（2026-09-17，随机四面体中位数）

    order  over_order        n_fine   相对误差      相对不过积分
    P1     不过积分               4   2.2368e-01        1.0x
    P1     2  <- 当前上限        10   4.3949e-05     5089.5x
    P1     4                    35   1.0961e-11    2.0e+10x
    P2     不过积分              10   8.5983e-02        1.0x
    P2     3  <- 当前上限        20   2.3841e-02        3.6x
    P2     4（理想）             35   7.0102e-06    12265.5x
    P2     6                    84   2.7232e-07   315740.4x
    P3     不过积分（=当前上限） 20   9.1669e-02        1.0x
    P3     4                    35   2.7421e-02        3.3x
    P3     6（理想）             84   4.9181e-06    18639.2x

**上限的代价**：
  * P2：3 -> 4 是 **3400 倍**（2.38e-2 -> 7.01e-6）
  * P3：完全无操作 -> 6 是 **18600 倍**（9.17e-2 -> 4.92e-6）
  * P1：上限不约束（2*1=2 <= 3），但继续提到 4 还有 4 个数量级
    ——那是 `AFCFD_OVERINT_ORDER_RULE` 倍率的问题，与上限无关

本文件把这些数钉住，使"放开上限值不值得"这个判断有据可查，并在任何人
改动过积分算子实现后立刻发现精度变化。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_batch
from autoflowcfd.fr.collapsed_basis import OVERINTEGRATION_MAX_ORDER
from autoflowcfd.fr.native_simplex_basis import (
    build_native_tet_operators,
    compute_native_tet_jacobian,
    map_native_tet_to_physical,
)
from autoflowcfd.fr.native_tet_overintegration import (
    build_native_tet_overintegration_operators,
)

GAMMA, RHO, P0, U = 1.4, 1.225, 101325.0, 30.0


def _resolve_rule():
    from autoflowcfd.fr.collapsed_basis import resolve_overintegration_order_rule

    return resolve_overintegration_order_rule()

#: 参照档：8 次已远高于任何被测档，用它当"精确"解
_REF_OVER_ORDER = 8

#: 实测中位相对误差的上界（2026-09-17，留约 3 倍余量）。
#: (order, over_order) -> 误差上界
_MEASURED = {
    (1, 1): 7.0e-1,      # 不过积分
    (1, 2): 1.5e-4,      # 当前上限（= 理想 2*1）
    (1, 4): 5.0e-11,
    (2, 2): 3.0e-1,      # 不过积分
    (2, 3): 8.0e-2,      # 当前上限
    (2, 4): 2.5e-5,      # 理想 2*2
    (2, 6): 1.0e-6,
    (3, 3): 3.0e-1,      # 不过积分 == 当前上限（over_order == order）
    (3, 4): 9.0e-2,
    (3, 6): 2.0e-5,      # 理想 2*3
}


def _poly_Q(ref, order, seed):
    """各守恒分量取次数恰为 `order` 的多项式 —— 粗空间精确可表示。

    **不**从 u 反推能量：那会让能量分量次数变成 2*order、在粗节点采样时
    自身就丢信息（见模块文档"必须避开的陷阱"）。
    """
    r = np.random.default_rng(seed)
    x, y, z = ref[:, 0], ref[:, 1], ref[:, 2]

    def p(scale):
        v = np.zeros_like(x)
        for i in range(order + 1):
            for j in range(order + 1 - i):
                for k in range(order + 1 - i - j):
                    v += r.normal() * x ** i * y ** j * z ** k
        return scale * (1.0 + 0.15 * v / max(np.abs(v).max(), 1e-30))

    Q = np.zeros((ref.shape[0], 5))
    Q[:, 0] = p(RHO)
    Q[:, 1] = p(RHO * U) - RHO * U
    Q[:, 2] = p(RHO * U) - RHO * U
    Q[:, 3] = p(RHO * U) - RHO * U
    Q[:, 4] = p(P0 / (GAMMA - 1.0))
    return Q


def _div(nodes, order, over_order, Qc, ref_c, D_c):
    """coarse SPs 上的体积项散度（over_order == order 表示不过积分）。"""
    det_j, adj_j = compute_native_tet_jacobian(nodes)
    if over_order == order:
        F = euler_physical_flux_batch(Qc)
        Ft = np.einsum("ij,pjv->piv", adj_j, F)
        d = np.zeros((ref_c.shape[0], 5))
        for m in range(3):
            d += D_c[:, :, m] @ Ft[:, m, :]
        return -d / det_j
    ref_f, c2f, D_f, f2c = build_native_tet_overintegration_operators(
        order, over_order)
    Qf = c2f @ Qc
    F = euler_physical_flux_batch(Qf)
    Ft = np.einsum("ij,pjv->piv", adj_j, F)
    d = np.zeros((ref_f.shape[0], 5))
    for m in range(3):
        d += D_f[:, :, m] @ Ft[:, m, :]
    return -(f2c @ d) / det_j


def _median_rel_error(order, over_order, n_tets=8, seed=20260917):
    rng = np.random.default_rng(seed)
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    ref_c, D_c = build_native_tet_operators(order)
    errs = []
    for i in range(n_tets):
        nodes = base + 0.3 * rng.standard_normal((4, 3))
        Qc = _poly_Q(map_native_tet_to_physical(ref_c, nodes), order, 100 + i)
        ref = _div(nodes, order, _REF_OVER_ORDER, Qc, ref_c, D_c)
        got = _div(nodes, order, over_order, Qc, ref_c, D_c)
        errs.append(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
    return float(np.median(errs))


@pytest.mark.parametrize("order,over_order", sorted(_MEASURED))
def test_dealiasing_error_matches_measurement(order, over_order):
    got = _median_rel_error(order, over_order)
    assert got <= _MEASURED[(order, over_order)], (
        f"P{order} over_order={over_order}: 去混叠相对误差 {got:.4e} 超出"
        f"实测上界 {_MEASURED[(order, over_order)]:.1e}"
    )


class TestCapCostIsLarge:
    """上限的代价必须仍然是"大"的 —— 这是"值得放开"这个判断的依据。

    方向是刻意的：若哪天这些比值变小了（例如换了过积分算子的构造方式），
    放开上限的性价比就变了，那时应当连同 `OVERINTEGRATION_MAX_ORDER`
    上方的注释与项目记忆 `overintegration_cap_is_collapsed_only` 一起
    重新评估，而不是照旧引用"3400 倍"这个数。
    """

    def test_p2_cap_costs_at_least_100x(self):
        """P2：当前上限 3 vs 理想 4。实测 3400 倍。"""
        capped = _median_rel_error(2, min(2 * 2, OVERINTEGRATION_MAX_ORDER))
        ideal = _median_rel_error(2, 2 * 2)
        assert capped / ideal > 100.0, (
            f"P2 上限代价只有 {capped/ideal:.1f} 倍（实测 3400 倍），"
            f"放开上限的性价比已变，请重新评估"
        )

    def test_p3_overintegration_is_currently_a_noop(self):
        """P3：`min(2*3, 3) == 3 == order`，过积分退化为恒等，完全无操作。"""
        assert min(2 * 3, OVERINTEGRATION_MAX_ORDER) == 3
        noop = _median_rel_error(3, 3)
        ideal = _median_rel_error(3, 2 * 3)
        assert noop / ideal > 100.0, (
            f"P3 上限代价只有 {noop/ideal:.1f} 倍（实测 18600 倍）"
        )

    def test_p1_cap_is_not_binding(self):
        """P1：`min(2*1, 3) == 2`，上限不约束（理想阶数就是 2）。

        这条说明"上限只在 P2/P3 上造成截断"——P1（当前生产阶数）不受
        影响，所以放开上限不改变已验证的 P1 的过积分阶数。2026-09-17
        放开四面体上限后实测确实如此（四面体仍取 oo=2），见
        `test_tet_overintegration_cap_raised.py::
        TestProductionOrders::test_p1_unchanged_from_before_the_raise`。
        """
        assert min(2 * 1, OVERINTEGRATION_MAX_ORDER) == 2 * 1

    def test_the_raised_tet_cap_actually_delivers_the_measured_gain(self):
        """放开后四面体**实际**取到的阶数，其误差就是本文件量出的那一档。

        把"量到的收益"和"实际生效的配置"钉在同一个文件里，避免出现
        "收益量过了但默认值没改"这种状态（本项目此前真实发生过）。
        """
        from autoflowcfd.fr.native_tet_overintegration import (
            resolve_tet_overintegration_order,
        )

        if _resolve_rule() != 2:
            pytest.skip("AFCFD_OVERINT_ORDER_RULE 非默认 2x")
        # 布局宽度不再约束（四面体段的细点度量改成"第 0 列广播"），
        # 所以 P2/P3 都取到理想的 2*order
        assert resolve_tet_overintegration_order(2) == 4
        err_old = _median_rel_error(2, 3)
        err_new = _median_rel_error(2, 4)
        assert err_new < err_old / 100.0, (
            f"P2 实际生效的 oo=4 误差 {err_new:.3e} 相对旧上限 oo=3 的 "
            f"{err_old:.3e} 只改善了 {err_old/err_new:.1f} 倍（实测 3400 倍）")
        assert resolve_tet_overintegration_order(3) == 6
        err_p3_old = _median_rel_error(3, 3)
        err_p3_new = _median_rel_error(3, 6)
        assert err_p3_new < err_p3_old / 100.0, (
            f"P3 实际生效的 oo=6 误差 {err_p3_new:.3e} 相对 oo=3（完全无操作）"
            f"的 {err_p3_old:.3e} 只改善了 {err_p3_old/err_p3_new:.1f} 倍"
            f"（实测 13000 倍）")

    def test_the_cliff_is_at_twice_the_order(self):
        """去混叠误差在 `oo = 2*order` 处**断崖式**下降，低于它基本无收益。

        这条是"为什么必须取到 2*order、取 2*order-1 不够"的依据——P3 一度
        被布局夹到 5，看起来"只差一阶"，实测却只拿到 18.6 倍中的 13000 倍：

            P2  oo=2 6.89e-2  oo=3 2.27e-2  oo=4 7.19e-6  oo=5 1.74e-6
            P3  oo=3 6.26e-2  oo=4 2.66e-2  oo=5 3.37e-3  oo=6 4.80e-6
        """
        for order in (2, 3):
            below = _median_rel_error(order, 2 * order - 1)
            at = _median_rel_error(order, 2 * order)
            assert at < below / 100.0, (
                f"P{order}: oo={2*order} 的误差 {at:.3e} 相对 oo={2*order-1} 的 "
                f"{below:.3e} 只改善了 {below/at:.1f} 倍——断崖不在 2*order 处了，"
                f"`rule*order` 这条经验法则的依据需要重新评估")

    def test_linear_couette_case_cannot_detect_the_cap(self):
        """记录方法论：线性 Couette 算例量不出上限的代价。

        它的能量通量只有三次，`over_order=3` 已精确捕捉，所以 P2 的
        3 与 4 在**舍入水平上不可分辨**（实测相对差约 5e-10，不是零）。
        这条测试把这个事实钉住，避免有人再用那个算例去"验证放开上限没用"。
        """
        order = 2
        ref_c, D_c = build_native_tet_operators(order)
        nodes = np.array([[0, 0, 0], [1.1, 0.1, -0.05],
                          [0.05, 0.95, 0.1], [-0.1, 0.05, 1.05]])
        phys = map_native_tet_to_physical(ref_c, nodes)
        # 线性 Couette：u = U*y/H，能量按 0.5*rho*u^2 反推（三次通量）
        y = phys[:, 1]
        u = U * y / 1.0
        Qc = np.zeros((phys.shape[0], 5))
        Qc[:, 0] = RHO
        Qc[:, 1] = RHO * u
        Qc[:, 4] = P0 / (GAMMA - 1.0) + 0.5 * RHO * u ** 2

        d3 = _div(nodes, order, 3, Qc, ref_c, D_c)
        d4 = _div(nodes, order, 4, Qc, ref_c, D_c)
        rel = np.abs(d3 - d4).max() / max(np.abs(d3).max(), 1e-30)
        # 1e-8 而不是 1e-10：实测两者相对差约 5e-10，是舍入而非零
        # （第一版按 1e-10 断言，被这条测试自己抓到）
        assert rel < 1e-8, (
            f"线性 Couette 上 over_order 3 与 4 的相对差 {rel:.3e} 已超出"
            f"舍入水平 —— 那说明该算例现在能分辨上限，本文件的算例设计"
            f"说明需要更新"
        )
