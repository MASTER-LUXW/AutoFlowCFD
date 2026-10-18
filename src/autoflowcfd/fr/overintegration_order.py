"""AutoFlowCFD V2.0 - 过积分（体积项去混叠）阶数的唯一来源。

`over_order = 2 * order`（两种原生基、全部体积项共用；P0 走有限体积路径，返回 0）。

## 为什么是 2*order

* 去混叠精度在 `oo = 2*order` 处断崖式下降（四面体 P3：oo=5 3.37e-3 vs oo=6 4.80e-6，差 18600 倍；P2：oo=3 2.38e-2
  vs oo=4 7.01e-6，3400 倍）。
* 3*order 在真实网格上买不到东西：plate_coarse_volume（168,340 单元，P1+SST，同一初场，预热 4 步后计时 8 步）
  rule=2x 11.996 s/step、rule=3x 18.967 s/step，**残差 7 位有效数字完全相同**；平板边界层算例粘性体积项的能量
  分量 2x 与 3x 逐位相同。
* 原生基不需要额外上限：原生棱柱 oo=8 时 cond(V)=1.86e3（坍缩棱柱 oo=4 已到 4.72e5、`max|D|` 3.4 万），原生
  四面体 oo=6 时 cond=3.86e3。求解器最高阶数 3，对应 oo=6。

2026-10-18 以前这三条由环境变量 `AFCFD_OVERINT_ORDER_RULE`（2x/3x）、`AFCFD_TET_OVERINT_MAX_ORDER`、
`AFCFD_PRISM_OVERINT_MAX_ORDER`（默认 6，P<=3 时从不起作用）选择，随 I-22 删除。
"""

#: 过积分阶数 = 本倍率 x 求解阶数
OVERINTEGRATION_ORDER_RULE = 2


def overintegration_order(order: int) -> int:
    """体积项过积分阶数 `2 * order`；`order <= 0`（P0 有限体积路径）时为 0。"""
    order = int(order)
    return 0 if order <= 0 else OVERINTEGRATION_ORDER_RULE * order


def prism_n_fine(over_order: int) -> int:
    """棱柱在 `over_order` 下的细点数 `(oo+1)^2 (oo+2)/2`。

    这个值同时是 `mesh.n_sps_per_cell_fine`（`jacobians_fine` 的每单元
    布局宽度）。原生棱柱的细点是 Warp&Blend 三角形点 x Gauss 挤出点，
    **不填充到立方体宽度 `(oo+1)^3`** —— 填充槽位恒为零、零贡献，却让整条
    过积分链在空点上白算，而 `D_fine` 的收缩是 O(n_fine^2)。

    四面体段不受这个宽度约束：它的细点度量逐单元一份、使用时广播
    （见 `core/fr_operators/volume_contract.get_overintegration_context`）。
    """
    from .native_prism.basis import native_prism_n_sps
    oo = int(over_order)
    if oo <= 0:
        return 0
    return native_prism_n_sps(oo)
