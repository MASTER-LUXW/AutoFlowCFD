"""
AutoFlowCFD V2.0 - native 四面体过积分算子的条件数/量级实测钉桩。

## 为什么需要这个文件

`fr/collapsed_basis.py::OVERINTEGRATION_MAX_ORDER = 3` 这个上限的论证链条是：

    坍缩坐标模态 Vandermonde 条件数随阶数爆炸（N=2 ~1e5 / N=3 ~1e9 /
    N=4 ~1e14）-> over_order=4 时 D_fine 的**绝对量级**暴涨约 6.3 万倍
    （6.0e7 -> 3.8e12）-> 相对误差即使保持在 float64 极限 ~1e-16，乘上
    这个绝对量级再经几何度量项求和就成了不可忽略的绝对噪声 -> 黄金标准
    判据（均匀自由流场残差本该恒为零）实测从 1.06e-5 恶化到 5.6e-3

这整条链条是对**坍缩坐标张量积**基做的。坍缩四面体基已于 2026-09-03
删除，四面体现在走 native 受限 PKD/Dubiner 基——单纯形上的**正交**基。
而 `fr/native_tet/overintegration.py` 的模块文档当时写的是"两套基没有
理由用不同的过积分阶数选择"，把两件独立的事混成了一件：

  * `over_order = rule * order` 是与基无关的去混叠经验法则（这句对）
  * 上限 3 是坍缩基的**数值条件数**极限（这句不该被 native 继承）

2026-09-16 实测（本文件即该实测的可执行形式）：

    over_order  n_pts   cond(V)     max|D|    D@1/max|D|    D@r - 1
             1      4   2.828e+00  5.000e-01   3.791e-16   0.000e+00
             2     10   1.326e+01  2.000e+00   4.886e-16   4.441e-16
             3     20   5.637e+01  4.045e+00   9.057e-16   2.998e-15
             4     35   2.070e+02  6.757e+00   1.376e-14   2.243e-14
             5     56   8.972e+02  1.014e+01   4.971e-14   6.584e-14
             6     84   3.856e+03  1.420e+01   1.202e-13   6.610e-13

`cond(V)` 在 over_order=6 才 3856（坍缩基 N=4 就 ~1e14），`max|D|` 从
over_order=3 到 6 只长 3.5 倍（坍缩基同区间暴涨 6.3 万倍）。**上限 3
的那条论证对 native 完全不适用。**

## 上限已于 2026-09-17 放开（本文件的数据是那次放开的依据）

四面体的过积分阶数已独立出去（`fr/native_tet/overintegration.py::
NATIVE_TET_OVERINTEGRATION_MAX_ORDER = 6`，env `AFCFD_TET_OVERINT_MAX_ORDER`），
P1/P2/P3 都取到理想的 `2*order`（2/4/6）。

本段曾记录一条"架构约束"作为不放开的理由：`mesh.jacobians_fine` 是棱柱与
四面体共用一个 `n_sps_per_cell_fine` 维度的合并数组，"四面体用 4、棱柱用 3"
要按较大者分配（125 槽 vs 64 槽），plate_demo P2 的这块内存会从约 1.86 GB
涨到约 3.63 GB。**那条论证是错的，被自己的代码推翻**：
`high_order_mesh_order.compute_native_tet_jacobians` 对直边四面体只算一个
逐单元常数、**原样广播**填满全部槽位，所以四面体段的度量取第 0 列广播到
本段自己的 `n_fine_tet` 即可，完全不需要更宽的数组（见
`core/fr_operators/volume_contract.get_overintegration_context`）。

放开的收益已量化（`test_overintegration_cap_cost.py`）：去混叠误差在
`oo = 2*order` 处断崖式下降，P2 3400 倍、P3 13000 倍。放开本身会抬高自由流
保持性的误差底（下表 `D@1/max|D|` 那一列随阶数增长就是它），已同批用
`fr/diff_matrix_consistency.enforce_constant_annihilation` 消掉——**所以下表
的 `D@1/max|D|` 与 `D@r - 1` 两列现在只反映"修正前"的构造，本文件仍按那个
构造独立复算，用途是监控 Vandermonde 条件数本身的变化。**

本文件的作用相应是：
(a) 任何人读到坍缩基那个上限时不会再以为它对 native 也有数值依据，
(b) 把 native 到 over_order=6 仍然良态这件事钉成可执行的证据。
"""

import numpy as np
import pytest

from autoflowcfd.fr.collapsed_basis import OVERINTEGRATION_MAX_ORDER
from autoflowcfd.fr.native_tet.basis import (
    build_native_tet_operators,
    restricted_tet_modes,
    rst_to_abc,
    simplex3d_value,
)

#: 2026-09-16 实测值（见模块文档的表）。上界留约 2 倍余量：这些量完全由
#: 基函数与节点集决定，是确定性的，不该随平台浮动；留余量只为容忍
#: LAPACK 实现差异带来的 cond 末位差别。
_MEASURED = {
    #  over_order: (n_pts, cond_max, dmax_max, const_rel_max, lin_err_max)
    1: (4,  6.0e0, 1.0e0, 1e-14, 1e-14),
    2: (10, 3.0e1, 4.0e0, 1e-14, 1e-14),
    3: (20, 1.2e2, 8.0e0, 1e-14, 1e-13),
    4: (35, 4.2e2, 1.4e1, 1e-13, 1e-12),
    5: (56, 1.8e3, 2.1e1, 1e-12, 1e-12),
    6: (84, 8.0e3, 3.0e1, 1e-12, 1e-11),
}


def _stats(over_order):
    rst, D = build_native_tet_operators(over_order)
    rst = np.asarray(rst)
    D = np.asarray(D)
    modes = restricted_tet_modes(over_order)
    a, b, c = rst_to_abc(rst[:, 0], rst[:, 1], rst[:, 2])
    V = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])

    n = rst.shape[0]
    dmax = float(np.abs(D).max())
    const_err = max(float(np.abs(D[:, :, d] @ np.ones(n)).max()) for d in range(3))
    lin_err = float(np.abs(D[:, :, 0] @ rst[:, 0] - 1.0).max())
    return n, float(np.linalg.cond(V)), dmax, const_err / dmax, lin_err


@pytest.mark.parametrize('over_order', sorted(_MEASURED))
def test_native_tet_conditioning_stays_mild(over_order):
    """native 四面体细网格算子的条件数/量级必须保持温和。

    这不是"越小越好"的软指标——它是"上限 3 对 native 没有数值依据"这个
    结论的**全部依据**。若哪天这些数真的爆了（比如换了节点集或基函数
    归一），那个结论就必须重新审，而不是继续引用。
    """
    n_exp, cond_max, dmax_max, const_max, lin_max = _MEASURED[over_order]
    n, cond, dmax, const_rel, lin_err = _stats(over_order)

    assert n == n_exp, f"节点数变了：{n} != {n_exp}（(p+1)(p+2)(p+3)/6）"
    assert cond < cond_max, (
        f"over_order={over_order}: cond(V)={cond:.3e} 超出实测上界 {cond_max:.1e}"
    )
    assert dmax < dmax_max, (
        f"over_order={over_order}: max|D|={dmax:.3e} 超出实测上界 {dmax_max:.1e}"
    )
    assert const_rel < const_max, (
        f"over_order={over_order}: 常数场导数相对误差 {const_rel:.3e} "
        f"超出 {const_max:.1e}（D@1 解析恒为零）"
    )
    assert lin_err < lin_max, (
        f"over_order={over_order}: 线性场 r-导数误差 {lin_err:.3e} "
        f"超出 {lin_max:.1e}（解析恒为 1）"
    )


def test_native_conditioning_growth_is_polynomial_not_explosive():
    """native 的 cond(V) 增长必须是温和的多项式式，而不是坍缩基那种爆炸。

    定量判据：从 over_order=3 到 6，`max|D|` 的增长倍数必须小于 10。
    坍缩基在 N=3 -> N=4 一步就涨了约 6.3 万倍（6.0e7 -> 3.8e12，见
    collapsed_basis.py 该常量处的记录）——两者不在同一个量级，这正是
    "上限不能照搬"的核心证据。
    """
    _, _, d3, _, _ = _stats(3)
    _, _, d6, _, _ = _stats(6)
    assert d6 / d3 < 10.0, (
        f"max|D| 从 over_order=3 到 6 增长 {d6/d3:.1f} 倍，"
        f"已不再是温和增长，上限照搬与否的结论需要重新评估"
    )


def test_cap_currently_truncates_dealiasing_at_p2_and_above():
    """把"上限确实在截断 P2/P3 的去混叠"这一事实钉住。

    理想去混叠阶数是 `rule*order`（默认 rule=2）：
        P1 需要 2  -> min(2,3)=2  完整
        P2 需要 4  -> min(4,3)=3  **被截断**
        P3 需要 6  -> min(6,3)=3  **被截断，且 over_order==order，
                                    等价于完全不做过积分**
    这条测试不是要求上限改掉（改它有本文件模块文档说明的架构/内存代价），
    而是保证这个已知缺口不会因为有人改了常量而悄悄变成"看起来没问题"。
    """
    assert OVERINTEGRATION_MAX_ORDER == 3, (
        f"OVERINTEGRATION_MAX_ORDER 变成了 {OVERINTEGRATION_MAX_ORDER}；"
        f"若是有意放开，请同步更新本测试与 "
        f"test_native_tet_overintegration_conditioning 的模块文档，"
        f"并在真实网格上重新量过 P2/P3 的内存与自由流场保持性"
    )
    rule = 2
    assert min(rule * 1, OVERINTEGRATION_MAX_ORDER) == 2, 'P1 应当完整'
    assert min(rule * 2, OVERINTEGRATION_MAX_ORDER) == 3 < rule * 2, 'P2 被截断'
    assert min(rule * 3, OVERINTEGRATION_MAX_ORDER) == 3 == 3, \
        'P3 的 over_order 退化到等于 order，过积分完全失效'
