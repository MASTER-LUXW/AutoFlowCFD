"""微分矩阵的精确常数零化（`D @ 1 = 0`）—— 自由流保持性的直接来源。

## 缺口

参考空间微分矩阵按 `D = V_grad @ V^{-1}` 构造（代码里用 LU 求解而不是显式
求逆），解析上必然满足 `D @ 1 = 0`，但浮点下只成立到 `eps*cond(V)`。

对**直边四面体**这个残余就是全部：Jacobian 逐单元为常数，均匀自由流下
体积项散度恰好是

    div = f2c @ [ sum_m adj(J)_m * F_m * (D_m @ 1) ] / det(J)

所以算出来的伪残差**完全**来自 `D_m @ 1`，再被 `adj(J)/det(J)` 在小体积
单元上放大。实测（`build_native_tet_operators`）：

    order/oo    2        3        4        5        6
    max|D@1|  9.8e-16  3.7e-15  9.3e-14  5.0e-13  1.7e-12

于是**提高过积分阶数会同时抬高自由流保持性的误差底**——这与坍缩基当年
"上限不能放宽到 4"的依据是同一机制，只是量级温和得多。放开四面体过积分
上限（`test_tet_overintegration_cap_raised.py`）必须连同这一项一起做，否则
拿到的去混叠精度会被自由流误差吃回去一部分。

## 修正

`enforce_constant_annihilation`：把每行的行和从该行对角元减掉，

    D'[i, j] = D[i, j] - delta_ij * sum_k D[i, k]

代价是把 D 扰动了 `max|D@1|`（最坏 1.7e-12，相对 `max|D|~14` 是 1.2e-13），
即"常数模态精确，其余模态多一个 1e-13 量级扰动"换"自由流保持性落到 eps"。
这是曲线坐标下的标准做法（Pulliam & Steger 的度量抵消）。

## 真实网格实测（plate_coarse_volume，168340 单元，均匀自由流，max / L2）

    P1  四面体 oo=2  修正前 8.55e-5 / 5.66e-7   修正后 6.27e-5 / 5.62e-7
        棱柱         修正前 4.67e-4 / 1.986e-5  修正后 4.60e-4 / 1.984e-5
    P2  四面体 oo=3  修正前 4.22e-4 / 1.67e-6   修正后 2.19e-4 / 1.10e-6
        四面体 oo=4  修正前 3.06e-3 / 1.81e-5   修正后 3.90e-4 / 2.17e-6
        棱柱         修正前 4.7204e-3           修正后 4.7213e-3
    P3  四面体 oo=6  修正前 8.55e-2 / 3.15e-4   修正后 1.64e-3 / 9.54e-6
                                                    （52 倍 / 33 倍）

**P3 那一行是本修正真正的分量所在**：不修正的话把 P3 放到理想的 oo=6 会把
四面体自由流从 oo=3 时的 3.25e-4 恶化到 8.55e-2（263 倍），那正是坍缩基当年
"上限不能放宽到 4"的同一个失败模式；修正把它压到 1.64e-3，相对旧基线只差
5 倍，而去混叠精度换来 13000 倍。**所以"放开过积分上限"与本修正是同一件事
的两半，不能只做一半。**

**棱柱段的改善是可以忽略的**（+0.02% on max、-0.005% on L2），原因是坍缩基
`max|D_fine| = 560`，点积自身的求和舍入 `~eps*max|D|*n` 就已经是 1e-13
量级，行和残余不是它的主导项。这条限制写在这里而不是留成含糊表述：棱柱段
的 4.72e-3 是全网格自由流误差的主导项，它不是本改动能解决的，要靠降低
坍缩基的 `max|D|`（或换掉坍缩基）。修正仍然对棱柱施加，因为"微分矩阵必须
零化常数"这条不变量应当一致成立，且扰动在 1e-16 相对量级、不可能改变任何
已验证结论。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_prism.basis import build_native_prism_operators
from autoflowcfd.fr.native_prism.overintegration import (
    build_native_prism_overintegration_operators,
)
from autoflowcfd.fr.overintegration_order import (
    resolve_prism_overintegration_order,
)
from autoflowcfd.fr.diff_matrix_consistency import (
    constant_annihilation_error,
    enforce_constant_annihilation,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre
from autoflowcfd.fr.native_tet.basis import build_native_tet_operators
from autoflowcfd.fr.native_tet.overintegration import (
    build_native_tet_overintegration_operators,
)


def _prism_coarse_ref(order):
    """与 `generate_fr_operators` 完全相同的 SPs 构造。

    **必须是 Gauss-Legendre 而不是 Gauss-Lobatto**：坍缩坐标基在
    `b = 1`（坍缩点）处退化，Lobatto 点集含端点，模态 Vandermonde 奇异、
    `D` 全是 NaN。第一版测试用了 Lobatto 并因此拿到 NaN——那不是代码缺陷
    而是测试自己给错了点集，这条注释把它钉住。
    """
    sps, _ = gauss_legendre(order + 1)
    a, b, c = np.meshgrid(sps, sps, sps, indexing="ij")
    return np.column_stack([a.ravel(), b.ravel(), c.ravel()])


#: 修正后的行和上界。注意它**不是** 0：把行和从对角元减掉之后再用
#: `np.sum` 重新加一遍，浮点加法不满足结合律，剩下的是求和自身的舍入
#: `~eps*max|D|*sqrt(n)`。所以这里按 `max|D|` 归一化来断言。
_RESUM_FLOOR_FACTOR = 1e-14


class TestHelperSemantics:
    def test_makes_row_sums_vanish_to_resummation_floor(self):
        """真实的微分矩阵（行和 = 舍入）经修正后行和落到求和舍入底。"""
        _, D = build_native_tet_operators(4)
        # 人为注入一个舍入量级的行和残余，模拟修正前的状态
        rng = np.random.default_rng(20260917)
        D = D + rng.standard_normal(D.shape) * 1e-13
        before = constant_annihilation_error(D)
        assert before > 1e-14, "注入的残余应当明显高于求和舍入底"
        enforce_constant_annihilation(D)
        after = constant_annihilation_error(D)
        limit = _RESUM_FLOOR_FACTOR * max(np.abs(D).max(), 1.0)
        assert after <= limit, f"修正后 {after:.4e} > {limit:.4e}"
        assert after < before / 10.0

    def test_in_place_by_default_and_copy_on_request(self):
        rng = np.random.default_rng(1)
        D = rng.standard_normal((5, 5, 3)) * 1e-13
        D0 = D.copy()
        out = enforce_constant_annihilation(D)
        assert out is D
        D2 = D0.copy()
        out2 = enforce_constant_annihilation(D2, copy=True)
        assert out2 is not D2
        np.testing.assert_array_equal(D2, D0)

    def test_accepts_2d_single_direction(self):
        D = np.array([[1.0, -1.0 + 1e-15], [0.5, -0.5]])
        enforce_constant_annihilation(D)
        assert abs(D.sum(axis=1)).max() < 1e-16

    def test_rejects_matrices_that_are_not_differentiation_operators(self):
        """护栏：行和远超舍入量级时报错而不是静默修正。

        那种情形说明矩阵构造本身有误（基/节点/模态索引不匹配），行和修正
        会把真实 bug 抹成看起来正常的结果——正是不能接受的那类"用容差
        掩盖缺陷"。
        """
        D = np.ones((4, 4, 3))  # 行和 = 4，绝非舍入
        with pytest.raises(ValueError, match="远超舍入量级"):
            enforce_constant_annihilation(D)

    def test_rejects_non_finite_matrices(self):
        """NaN/inf 必须报错。它们会让"行和是否超限"的比较恒为 False 而
        静默穿过——本文件第一版就因为给错点集拿到了全 NaN 的棱柱 D，而
        修正函数当时一声不响地放过了它。"""
        D = np.zeros((4, 4, 3))
        D[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="非有限元素"):
            enforce_constant_annihilation(D)
        D2 = np.zeros((4, 4, 3))
        D2[1, 2, 1] = np.inf
        with pytest.raises(ValueError, match="非有限元素"):
            enforce_constant_annihilation(D2)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="既不是"):
            enforce_constant_annihilation(np.zeros((3, 3, 3, 3)))
        with pytest.raises(ValueError, match="前两轴必须同长"):
            enforce_constant_annihilation(np.zeros((3, 4, 3)))


class TestNativeTetDiffMatrices:
    """native 四面体：这里的常数零化是自由流保持的**充分**条件。"""

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_row_sums_at_resummation_floor(self, order):
        _, D = build_native_tet_operators(order)
        err = constant_annihilation_error(D)
        limit = _RESUM_FLOOR_FACTOR * max(np.abs(D).max(), 1.0)
        assert err <= limit, (
            f"order={order}: max|D@1| = {err:.4e} 超过求和舍入底 {limit:.4e}"
            f" —— enforce_constant_annihilation 没有生效，自由流保持性会退化")

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_polynomial_exactness_survives(self, order):
        """修正只动对角元、量级 1e-13，degree<=order 的精确微分必须保持。"""
        ref, D = build_native_tet_operators(order)
        r, s, t = ref[:, 0], ref[:, 1], ref[:, 2]
        rng = np.random.default_rng(31 + order)
        for _ in range(4):
            coef = {}
            f = np.zeros_like(r)
            dfdr = np.zeros_like(r)
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    for k in range(order + 1 - i - j):
                        c = rng.normal()
                        coef[(i, j, k)] = c
                        f += c * r ** i * s ** j * t ** k
                        if i >= 1:
                            dfdr += c * i * r ** (i - 1) * s ** j * t ** k
            got = D[:, :, 0] @ f
            assert np.abs(got - dfdr).max() <= 1e-10 * max(
                np.abs(dfdr).max(), 1.0), f"order={order} 的 r 方向精确性丢失"

    @pytest.mark.parametrize("over_order", [2, 3, 4, 5, 6])
    def test_full_overintegration_chain_annihilates_constants(self, over_order):
        """整条 c2f -> D_fine -> f2c 链对常数场给出零散度。

        这就是均匀自由流在四面体上的体积项残差（度量逐单元常数可以提到
        求和外面），所以它必须落到 eps 量级。
        """
        _, c2f, D_fine, f2c = build_native_tet_overintegration_operators(
            2, over_order)
        const_coarse = np.ones(c2f.shape[1])
        fine = c2f @ const_coarse
        assert np.abs(fine - 1.0).max() < 1e-14, "c2f 必须精确保持常数"
        rows = np.stack([D_fine[:, :, m] @ fine for m in range(3)], axis=1)
        out = np.abs(f2c @ rows).max()
        limit = _RESUM_FLOOR_FACTOR * max(np.abs(D_fine).max(), 1.0) * 10.0
        assert out <= limit, (
            f"over_order={over_order}: 全链路常数散度 {out:.4e} 超过 "
            f"{limit:.4e} —— 均匀自由流会产生这个量级的伪残差")

    def test_higher_over_order_no_longer_degrades_constant_annihilation(self):
        """修正前 oo=2 -> 6 的常数零化残余涨了约 1700 倍（9.8e-16 ->
        1.7e-12），那正是"放开上限有自由流代价"的来源。修正后各阶数都在
        求和舍入底上，阶数之间不再有这个单调恶化。"""
        errs = {}
        for oo in (2, 3, 4, 5, 6):
            _, D = build_native_tet_operators(oo)
            errs[oo] = constant_annihilation_error(D) / max(np.abs(D).max(), 1.0)
        worst_ratio = max(errs.values()) / max(min(errs.values()), 1e-300)
        assert worst_ratio < 50.0, (
            f"归一化常数零化残余在 oo=2..6 之间相差 {worst_ratio:.1f} 倍"
            f"（修正前约 1700 倍）：{errs}")


class TestPrismDiffMatrices:
    """棱柱（原生基，唯一实现）：同一条不变量也施加。

    2026-09-23 之前这一类跑的是**坍缩**棱柱算子
    （`build_collapsed_diff_matrices("prism", ...)` +
    `build_overintegration_operators("prism", ...)`），随坍缩棱柱基一并
    删除。已按原生算子重新标定，见 `test_prism_floor_is_dot_product_
    rounding` 里的实测表。
    """

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_coarse_row_sums_at_resummation_floor(self, order):
        _ref, D = build_native_prism_operators(order)
        err = constant_annihilation_error(D)
        limit = _RESUM_FLOOR_FACTOR * max(np.abs(D).max(), 1.0)
        assert err <= limit, f"P{order} 棱柱 coarse D: {err:.4e} > {limit:.4e}"

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_fine_row_sums_at_resummation_floor(self, order):
        over_order = resolve_prism_overintegration_order(order)
        _, _, D_fine, _ = build_native_prism_overintegration_operators(
            order, over_order)
        err = constant_annihilation_error(D_fine)
        limit = _RESUM_FLOOR_FACTOR * max(np.abs(D_fine).max(), 1.0)
        assert err <= limit, (
            f"P{order} oo={over_order} 棱柱 D_fine: {err:.4e} > {limit:.4e}")

    def test_prism_floor_is_dot_product_rounding(self):
        """记录方法论：棱柱段的行和残余**远低于**点积自身的求和舍入底。

        实测（原生棱柱，默认 `rule=2x` 下的生产 over_order）：

            阶数  oo  n_fine  max|D_fine|  行和残余    点积舍入底
            P1     2      18       2.582   2.85e-16    2.43e-15
            P2     4      75       7.702   8.90e-16    1.48e-14
            P3     6     196      15.083   1.67e-15    4.69e-14

        行和残余比舍入底小约一个数量级，所以"强制行和为零"在原生棱柱上
        不可能成为自由流误差的主导改善项 —— 这与已删除的坍缩棱柱基上
        得到的**同一个结论**（那边 `max|D_fine|` 约 560、两者同阶，真实
        网格实测 4.7204e-3 -> 4.7213e-3，+0.02%）来自不同的原因：坍缩是
        "两者同阶所以只消掉一部分"，原生是"行和残余本来就远低于舍入底"。
        这条测试把这个事实钉成可核对的数字，避免有人以为修正没生效。
        """
        for order in (1, 2, 3):
            oo = resolve_prism_overintegration_order(order)
            _, _, D_fine, _ = build_native_prism_overintegration_operators(
                order, oo)
            max_d = np.abs(D_fine).max()
            n = D_fine.shape[0]
            dot_floor = np.finfo(float).eps * max_d * np.sqrt(n)
            err = constant_annihilation_error(D_fine)
            assert err < dot_floor, (
                f"P{order} oo={oo}: 行和残余 {err:.3e} 已超过点积求和舍入底 "
                f"{dot_floor:.3e} —— 那时行和修正才会成为棱柱段的主导改善"
                f"项，本文件关于『棱柱没有改善』的结论需要重测")

