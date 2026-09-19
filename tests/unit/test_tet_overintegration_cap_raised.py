"""四面体过积分上限与棱柱**解耦**并放开（2026-09-17）。

## 缺口

`collapsed_basis.OVERINTEGRATION_MAX_ORDER = 3` 是**坍缩坐标模态
Vandermonde 的条件数极限**（cond 在 N=4 达约 1e14、`max|D|` 暴涨约 6.3 万
倍）。native 四面体走受限 PKD 基，实测到 over_order=6 才 cond=3856、
`max|D|` 从 3 到 6 只长 3.5 倍（`test_native_tet_overintegration_
conditioning.py`）——那条论证对它不成立，但它一直在无依据地继承这个上限。

代价是量过的（`test_overintegration_cap_cost.py`，以 over_order=8 为参照的
体积项去混叠相对误差中位数）：

    P2  oo=3 -> oo=4    2.38e-2 -> 7.01e-6      3400 倍
    P3  oo=3 -> oo=6    9.17e-2 -> 4.92e-6     18600 倍
                （P3 的 oo=3 == order，过积分完全退化为恒等）

## 两条约束

`resolve_tet_overintegration_order(order)`：

  1. 去混叠经验法则 `rule * order`（`AFCFD_OVERINT_ORDER_RULE`，默认 2x）
  2. 四面体自己的上限（`AFCFD_TET_OVERINT_MAX_ORDER`，默认 6 = 不设限）

曾经有第三条"布局约束" `native_tet_n_fine(oo) <= (oo_prism+1)^3`，因为四面体
段的细点度量是切共用数组 `mesh.jacobians_fine` 的前 `n_fine_tet` 列。它把
P3 从理想的 oo=6（84 个细点）夹到 5（56 <= 64），而去混叠误差在
`oo = 2*order` 处**断崖式**下降：

    P3  oo=3  6.26e-2   oo=4  2.66e-2   oo=5  3.37e-3   oo=6  4.80e-6
    P2  oo=2  6.89e-2   oo=3  2.27e-2   oo=4  7.19e-6   oo=5  1.74e-6

所以被夹到 5 只拿到 18.6 倍（可得 13000 倍）——这不是可以接受的余量。已把
四面体段的度量改成"取第 0 列**广播**"（直边四面体逐单元常数，第 0 列就是那个
常数），布局宽度不再相关，那条约束被移除。

结果：

    P1  棱柱 oo=2  四面体 oo=2  细点 10   = 理想（与改动前相同）
    P2  棱柱 oo=3  四面体 oo=4  细点 35   = 理想
    P3  棱柱 oo=3  四面体 oo=6  细点 84   = 理想（超过棱柱的 64 列，无妨）

## 真实网格实测（plate_coarse_volume，168340 单元）

### P2 去混叠精度（以 oo=5 为参照，四面体段完整无粘残差的相对误差）

    oo=3（旧）  中位 8.83e-9   p90 7.05e-6   最大 7.03e-4
    oo=4（新）  中位 1.43e-9   p90 4.58e-9   最大 9.82e-8
                          -> p90 改善 1540 倍、最大改善 7160 倍

### 均匀自由流保持性（解析残差恒为零；四面体段 max / L2）

放开上限本身会抬高这个误差底（`D_fine` 对常数的零化残余随阶数增长），所以
同批一起做了 `diff_matrix_consistency.enforce_constant_annihilation`：

    P2  修正前  oo=3  4.22e-4 / 1.67e-6      oo=4  3.06e-3 / 1.81e-5
        修正后  oo=3  2.19e-4 / 1.10e-6      oo=4  3.90e-4 / 2.17e-6

    P3  修正前  oo=3  3.25e-4 / 2.16e-6      oo=6  8.55e-2 / 3.15e-4
        修正后  oo=3  (未测)                 oo=6  1.64e-3 / 9.54e-6

**修正是放开上限的前提条件，不是可选的附带项**：
  * P2：修正后的 oo=4（3.90e-4）比**修正前的 oo=3**（旧生产基线 4.22e-4）
    还低 —— 放开完全没有自由流代价。
  * P3：不修正的话 oo=6 会把四面体自由流从 3.25e-4 恶化到 8.55e-2
    （263 倍），那正是坍缩基当年"上限不能放宽"的失败模式；修正把它压到
    1.64e-3（52 倍改善），相对旧基线只差 5 倍，而去混叠精度换来 13000 倍。
    而且 1.64e-3 仍比**棱柱段**的 2.95e-1（max）/ 6.30e-3（L2）低 180/660
    倍 —— 全网格的自由流误差底由棱柱决定，四面体这 5 倍不改变它。

棱柱段在全部组合下都不变（P2 4.72e-3、P3 2.95e-1）—— 它由坍缩基
`max|D_fine|=560` 下点积自身的求和舍入决定，与本改动无关。
"""

import numpy as np
import pytest

from autoflowcfd.fr.collapsed_basis import (
    OVERINTEGRATION_MAX_ORDER,
    resolve_overintegration_order_rule,
)
from autoflowcfd.fr.native_tet_overintegration import (
    NATIVE_TET_OVERINTEGRATION_MAX_ORDER,
    native_tet_n_fine,
    resolve_tet_overintegration_max_order,
    resolve_tet_overintegration_order,
)
from autoflowcfd.fr.operators import generate_fr_operators


class TestNFineFormula:
    def test_matches_closed_form(self):
        for oo in range(0, 9):
            assert native_tet_n_fine(oo) == (oo + 1) * (oo + 2) * (oo + 3) // 6

    def test_is_strictly_smaller_than_tensor_product(self):
        """native 细点数严格小于同阶张量积——这正是不该继承棱柱上限的
        另一面：同样的 over_order 下四面体的工作量本来就少 2~3 倍。"""
        for oo in range(1, 9):
            assert native_tet_n_fine(oo) < (oo + 1) ** 3


class TestMaxOrderResolver:
    def test_default_is_six(self):
        assert NATIVE_TET_OVERINTEGRATION_MAX_ORDER == 6
        assert resolve_tet_overintegration_max_order() == 6

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AFCFD_TET_OVERINT_MAX_ORDER", "4")
        assert resolve_tet_overintegration_max_order() == 4

    def test_blank_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AFCFD_TET_OVERINT_MAX_ORDER", "   ")
        assert resolve_tet_overintegration_max_order() == 6

    @pytest.mark.parametrize("bad", ["abc", "2.5", "four"])
    def test_non_integer_raises_not_silently_defaults(self, monkeypatch, bad):
        """非法取值必须报错。静默回退会让一次拼写错误伪装成默认行为，把
        A/B 的两条运行悄悄变成同一档（同一原则见
        `kernels.py::resolve_ausm_precond_mode`）。"""
        monkeypatch.setenv("AFCFD_TET_OVERINT_MAX_ORDER", bad)
        with pytest.raises(ValueError, match="不是整数"):
            resolve_tet_overintegration_max_order()

    @pytest.mark.parametrize("bad", ["0", "-1", "9", "20"])
    def test_out_of_range_raises(self, monkeypatch, bad):
        monkeypatch.setenv("AFCFD_TET_OVERINT_MAX_ORDER", bad)
        with pytest.raises(ValueError, match=r"超出 \[1, 8\]"):
            resolve_tet_overintegration_max_order()


class TestOrderResolverConstraints:
    """两条约束各自都要真的起作用，且**没有**第三条布局约束。"""

    def test_rule_is_the_binding_constraint_by_default(self):
        """布局足够宽、上限足够高时，取的就是经验法则 rule*order。"""
        if resolve_overintegration_order_rule() != 2:
            pytest.skip("AFCFD_OVERINT_ORDER_RULE 非默认 2x")
        for order in (1, 2, 3):
            assert resolve_tet_overintegration_order(order, 10 ** 6) == 2 * order

    def test_cap_binds(self, monkeypatch):
        monkeypatch.setenv("AFCFD_TET_OVERINT_MAX_ORDER", "3")
        assert resolve_tet_overintegration_order(3, 10 ** 6) == 3

    def test_layout_no_longer_binds(self):
        """布局宽度参数已被忽略——传什么都不改变结果。

        它曾是真实约束并且把 P3 夹到 5（代价 700 倍，见模块文档）。形参
        保留只为签名兼容。这条测试把"已移除"钉住，避免有人因为
        `n_fine_tet > n_fine_prism` 看起来越界又把限制加回去。
        """
        if resolve_overintegration_order_rule() != 2:
            pytest.skip("AFCFD_OVERINT_ORDER_RULE 非默认 2x")
        for layout in (1, 8, 20, 27, 35, 56, 64, 84, 165, None):
            assert resolve_tet_overintegration_order(3, layout) == 6
            assert resolve_tet_overintegration_order(2, layout) == 4
            assert resolve_tet_overintegration_order(1, layout) == 2

    def test_never_drops_below_order(self):
        """结果恒 >= order：低于 order 的"细"网格表示不了解本身，那不是
        "过积分变弱"而是把解截断，是错误而非退化。"""
        for order in (1, 2, 3, 4, 5, 6, 7, 8):
            assert resolve_tet_overintegration_order(order) >= order

    def test_cap_can_never_push_below_order(self, monkeypatch):
        monkeypatch.setenv("AFCFD_TET_OVERINT_MAX_ORDER", "1")
        for order in (1, 2, 3):
            assert resolve_tet_overintegration_order(order) == order


class TestProductionOrders:
    """生产默认（rule=2x）下各阶数实际取到的过积分阶数。"""

    EXPECTED = {1: (2, 2, 10), 2: (3, 4, 35), 3: (3, 6, 84)}

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_operators_expose_decoupled_orders(self, order):
        if resolve_overintegration_order_rule() != 2:
            pytest.skip("AFCFD_OVERINT_ORDER_RULE 非默认 2x")
        oo_prism_exp, oo_tet_exp, n_fine_exp = self.EXPECTED[order]
        ops = generate_fr_operators(order)
        assert ops.overint_order_prism == oo_prism_exp, "棱柱阶数不应被本改动影响"
        assert ops.overint_order_tet == oo_tet_exp
        assert ops.overint_n_fine_tet == n_fine_exp
        assert ops.overint_D_fine_tet.shape == (n_fine_exp, n_fine_exp, 3)

    def test_p1_unchanged_from_before_the_raise(self):
        """P1（当前生产阶数）的四面体阶数与放开上限之前**完全相同**。

        旧行为 `min(2*1, OVERINTEGRATION_MAX_ORDER=3) = 2`，新行为
        `min(2*1, 6) = 2`，布局 27 >= 10 不约束——所以 P1 的过积分算子不受
        "放开上限"这一项影响，已验证的 P1 生产结果不因它改变。
        （`enforce_constant_annihilation` 是同批的**另一项**改动，它确实会
        在舍入量级上改动 P1，见 `test_diff_matrix_constant_annihilation.py`。）
        """
        assert resolve_tet_overintegration_order(1, (2 + 1) ** 3) == 2
        assert min(2 * 1, OVERINTEGRATION_MAX_ORDER) == 2

    def test_p2_reaches_the_ideal(self):
        """P2 的理想 oo=4 完整达到——这是本改动的主要收益（3400 倍）。"""
        assert resolve_tet_overintegration_order(2, (3 + 1) ** 3) == 4
        assert native_tet_n_fine(4) == 35
        assert native_tet_n_fine(4) <= 64

    def test_p3_reaches_the_ideal_despite_exceeding_the_prism_layout(self):
        """P3 取到理想的 oo=6，尽管它的 84 个细点超过棱柱的 64 列。

        这是把度量改成"第 0 列广播"之后才成立的（切列版本会被夹到 5）。
        P3 的去混叠误差因此从 6.26e-2（oo=3，完全无操作）降到 4.80e-6，
        而不是停在 oo=5 的 3.37e-3。
        """
        assert native_tet_n_fine(6) == 84
        assert native_tet_n_fine(6) > (3 + 1) ** 3, "84 确实超过棱柱布局 64"
        assert resolve_tet_overintegration_order(3) == 6
        ops = generate_fr_operators(3)
        assert ops.overint_order_tet == 6
        assert ops.overint_n_fine_tet == 84 > (ops.overint_order_prism + 1) ** 3


class TestLayoutInvariantHolds:
    """`volume_contract` 第 0 列广播的前提必须成立。"""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_shared_context_accepts_the_raised_order(self, order):
        """共享 helper 的形状校验对放开后的阶数仍然通过。"""
        from autoflowcfd.core.fr_operators.volume_contract import (
            get_overintegration_context,
        )

        class _FakeMesh:
            pass

        ops = generate_fr_operators(order)
        n_fine_prism = (ops.overint_order_prism + 1) ** 3
        n_cells, n_prism = 7, 3
        mesh = _FakeMesh()
        mesh.n_cells = n_cells
        mesh.n_prism_cells = n_prism
        mesh.n_sps_per_cell_fine = n_fine_prism
        mesh.jacobians_fine = {
            "det_jacs": np.ones(n_cells * n_fine_prism),
            "inv_jacs": np.tile(np.eye(3), (n_cells * n_fine_prism, 1, 1)),
        }
        oi = get_overintegration_context(mesh, ops)
        assert oi is not None
        segs = oi["segs"]
        assert len(segs) == 2
        assert segs[0][2] == n_fine_prism
        assert segs[1][2] == ops.overint_n_fine_tet
        # 四面体段的度量已按**自己的**细点数给出（P3 上它比棱柱宽度还大，
        # 靠广播而不是切列）
        assert segs[1][3].shape == (n_cells - n_prism, ops.overint_n_fine_tet)
        assert segs[1][4].shape == (n_cells - n_prism, ops.overint_n_fine_tet, 3, 3)
