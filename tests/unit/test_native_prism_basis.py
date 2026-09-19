"""棱柱原生基（三角形 PKD/Dubiner ⊗ 直线 Legendre）。

## 它要解决什么

棱柱至今用**坍缩坐标基**，实测三条后果（完整数据见
`fr/native_prism/triangle_basis.py` 模块文档）：

  1. 微分矩阵元素量级随阶数爆炸（`max|D_3d_prism|` P1 2.05 -> P4 33929）；
  2. 自由流保持性只有 ~1e-9，且实测严格等于 `eps * max|D| / det(J)`
     —— 是舍入被算子量级放大，不降低 `max|D|` 就改不掉；
  3. 伪横流：只有被三角化的两个参考轴方向会长出非物理横流速度，
     P1 上饱和、P2 上无界增长导致发散。

四面体当年有完全同类的病理，解法是整套换 native PKD 基。本模块是棱柱侧
的同一条路。

## 本文件钉住的性质

**精确性**（任何正确实现都必须满足，是硬判据）：
  * 节点数 == 模态数 == `(order+1)^2 (order+2)/2`，节点落在参考棱柱内；
  * `D` 对空间内**全部**多项式给出精确导数；
  * `D @ 1` 是机器零（直边单元的自由流保持性完全由它决定）。

**改善**（相对坍缩基的定量收益，修复的意义所在）：
  * `max|D|` 大幅下降且随阶数**线性**增长而非指数；
  * `D @ 1` 的残余同样大幅下降；
  * 自由度数少 33~40%（顺带缓解 P2 的内存天花板）。
"""

import numpy as np
import pytest

from autoflowcfd.fr.collapsed_basis import build_collapsed_diff_matrices
from autoflowcfd.fr.native_prism.basis import (
    build_native_prism_nodes,
    build_native_prism_operators,
    build_native_prism_vandermonde,
    native_prism_n_sps,
    restricted_prism_modes,
)
from autoflowcfd.fr.native_prism.triangle_basis import (
    build_native_tri_vandermonde,
    restricted_tri_modes,
    warp_blend_nodes_2d,
)
from autoflowcfd.fr.quadrature_points import gauss_legendre

_ORDERS = [1, 2, 3, 4]


def _collapsed_prism_D(order):
    """同阶坍缩棱柱微分矩阵（对照组）。"""
    n1d = order + 1
    s1, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(s1, s1, s1, indexing="ij")
    cube = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return build_collapsed_diff_matrices("prism", order, cube)


class TestTriangleNodes:
    @pytest.mark.parametrize("order", [0] + _ORDERS)
    def test_node_count_matches_mode_count(self, order):
        r, s = warp_blend_nodes_2d(order)
        n_expect = (order + 1) * (order + 2) // 2
        assert len(r) == len(s) == n_expect
        assert len(restricted_tri_modes(order)) == n_expect

    @pytest.mark.parametrize("order", [0] + _ORDERS)
    def test_nodes_lie_inside_the_reference_triangle(self, order):
        """节点必须落在 `{r>=-1, s>=-1, r+s<=0}` 内。

        Warp & Blend 是对等距节点的**扰动**，扰动过大会把点推出单元 ——
        那会让 Vandermonde 在单元外取值，是一个不会崩但结果全错的状态。
        """
        r, s = warp_blend_nodes_2d(order)
        tol = 1e-12
        assert np.all(r >= -1.0 - tol), f"r 最小 {r.min()}"
        assert np.all(s >= -1.0 - tol), f"s 最小 {s.min()}"
        assert np.all(r + s <= tol), f"r+s 最大 {(r + s).max()}"

    @pytest.mark.parametrize("order", _ORDERS)
    def test_vandermonde_is_well_conditioned(self, order):
        """三角形 Vandermonde 的条件数必须是个位数~几十，不是几千。

        实测 1.00 / 1.63 / 4.04 / 8.55 / 18.5（P0~P4）。上界留足余量；
        它变大意味着节点分布退化。
        """
        r, s = warp_blend_nodes_2d(order)
        V, _, _ = build_native_tri_vandermonde(order, r, s)
        cond = float(np.linalg.cond(V))
        assert cond < 100.0, f"order={order}: cond(V)={cond:.3e}"


class TestPrismNodesAndModes:
    @pytest.mark.parametrize("order", _ORDERS)
    def test_counts_are_consistent(self, order):
        ref = build_native_prism_nodes(order)
        n = native_prism_n_sps(order)
        assert ref.shape == (n, 3)
        assert len(restricted_prism_modes(order)) == n
        assert n == (order + 1) ** 2 * (order + 2) // 2

    @pytest.mark.parametrize("order", _ORDERS)
    def test_nodes_lie_inside_the_reference_prism(self, order):
        ref = build_native_prism_nodes(order)
        r, s, t = ref[:, 0], ref[:, 1], ref[:, 2]
        tol = 1e-12
        assert np.all(r >= -1.0 - tol) and np.all(s >= -1.0 - tol)
        assert np.all(r + s <= tol)
        assert np.all(np.abs(t) <= 1.0 + tol)

    @pytest.mark.parametrize("order", _ORDERS)
    def test_vandermonde_is_invertible(self, order):
        ref = build_native_prism_nodes(order)
        V, _, _, _ = build_native_prism_vandermonde(order, ref)
        assert V.shape[0] == V.shape[1]
        assert float(np.linalg.cond(V)) < 1e3


class TestOperatorExactness:
    """**硬判据**：`D` 必须精确微分空间内的全部多项式。"""

    @pytest.mark.parametrize("order", _ORDERS)
    def test_differentiates_every_polynomial_in_the_space(self, order):
        ref, D = build_native_prism_operators(order)
        r, s, t = ref[:, 0], ref[:, 1], ref[:, 2]
        worst = 0.0
        detail = ""
        for i in range(order + 1):
            for j in range(order + 1 - i):
                for k in range(order + 1):
                    f = (r ** i) * (s ** j) * (t ** k)
                    d_r = ((i * r ** (i - 1)) if i else np.zeros_like(r)) \
                        * (s ** j) * (t ** k)
                    d_s = (r ** i) * ((j * s ** (j - 1)) if j
                                      else np.zeros_like(s)) * (t ** k)
                    d_t = (r ** i) * (s ** j) * ((k * t ** (k - 1)) if k
                                                 else np.zeros_like(t))
                    for m, (exact, nm) in enumerate(
                            ((d_r, "r"), (d_s, "s"), (d_t, "t"))):
                        err = float(np.max(np.abs(D[:, :, m] @ f - exact)))
                        rel = err / max(float(np.max(np.abs(exact))), 1.0)
                        if rel > worst:
                            worst, detail = rel, f"r^{i}s^{j}t^{k} d/d{nm}"
        assert worst < 1e-12, f"order={order}: 最大相对误差 {worst:.3e} ({detail})"

    @pytest.mark.parametrize("order", _ORDERS)
    def test_annihilates_constants_to_machine_zero(self, order):
        """`D @ 1 = 0` —— 直边单元的自由流保持性完全由这个残余决定。

        实测 1.1e-16 ~ 7.7e-16；坍缩基同阶是 2.2e-16 / 4.1e-15 / 1.9e-13
        （P3 差 300 倍）。
        """
        _, D = build_native_prism_operators(order)
        one = np.ones(D.shape[0])
        worst = max(float(np.max(np.abs(D[:, :, m] @ one))) for m in range(3))
        assert worst < 1e-14, f"order={order}: max|D@1| = {worst:.3e}"


class TestImprovementOverCollapsed:
    """相对坍缩基的**定量收益** —— 这次修复的意义所在。

    判据带下界：若某天收益消失（比例掉到 1 附近），说明要么坍缩基被改好
    了、要么原生实现退化了，两种情况都必须重新评估而不是让测试静默通过。
    """

    #: 实测 `max|D_collapsed| / max|D_native|`：P1 2.4 / P2 8.7 / P3 115 /
    #: P4 4405。下界留足余量。
    _MIN_GAIN = {1: 1.5, 2: 4.0, 3: 30.0, 4: 500.0}

    @pytest.mark.parametrize("order", _ORDERS)
    def test_operator_magnitude_is_far_smaller(self, order):
        _, dn = build_native_prism_operators(order)
        dc = _collapsed_prism_D(order)
        mag_n = float(np.max(np.abs(dn)))
        mag_c = float(np.max(np.abs(dc)))
        gain = mag_c / mag_n
        assert gain >= self._MIN_GAIN[order], (
            f"order={order}: max|D| 只改善了 {gain:.1f} 倍 "
            f"(native {mag_n:.3f} vs collapsed {mag_c:.3f})，"
            f"低于已记录的收益")

    def test_native_magnitude_grows_linearly_not_exponentially(self):
        """native 的 `max|D|` 随阶数线性增长；坍缩的是指数增长。

        这条是"为什么高阶尤其受益"的依据：P4 上收益已经是 4405 倍。
        """
        mags_n = [float(np.max(np.abs(build_native_prism_operators(p)[1])))
                  for p in _ORDERS]
        mags_c = [float(np.max(np.abs(_collapsed_prism_D(p)))) for p in _ORDERS]
        ratio_n = [mags_n[i + 1] / mags_n[i] for i in range(len(mags_n) - 1)]
        ratio_c = [mags_c[i + 1] / mags_c[i] for i in range(len(mags_c) - 1)]
        assert max(ratio_n) < 4.0, f"native 每阶增长比 {ratio_n} 不像线性"
        assert max(ratio_c) > 8.0, f"collapsed 每阶增长比 {ratio_c} 不像指数"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_fewer_degrees_of_freedom(self, order):
        """自由度少 33~40% —— 顺带缓解 P2 的内存天花板。"""
        n_native = native_prism_n_sps(order)
        n_collapsed = (order + 1) ** 3
        assert n_native < n_collapsed
        saving = 1.0 - n_native / n_collapsed
        assert saving > 0.3, f"order={order}: 只省了 {100 * saving:.1f}%"

    @pytest.mark.parametrize("order", _ORDERS)
    def test_constant_annihilation_residual_is_no_worse(self, order):
        """原生基的 `D@1` 残余不得比坍缩基差。"""
        _, dn = build_native_prism_operators(order)
        dc = _collapsed_prism_D(order)
        one_n = np.ones(dn.shape[0])
        one_c = np.ones(dc.shape[0])
        rn = max(float(np.max(np.abs(dn[:, :, m] @ one_n))) for m in range(3))
        rc = max(float(np.max(np.abs(dc[:, :, m] @ one_c))) for m in range(3))
        assert rn <= max(rc, 1e-15), (
            f"order={order}: native 的 D@1 残余 {rn:.3e} 比 collapsed 的 "
            f"{rc:.3e} 还差")
