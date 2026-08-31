"""AutoFlowCFD V2.0 - native 四面体（路径C）体积项去混叠（过积分）算子。

背景：`fr/collapsed_basis.py::build_overintegration_operators` 文档
完整记录了去混叠的物理动机（欧拉通量×度量项的非线性乘积真实多项式
次数远高于求解阶数，直接在 coarse SPs 上微分等价于先混叠再求导，真实
数值实验测得 P2 残差可以是真值的 43~62 倍）——这个机制与坍缩坐标/
native 哪套基函数无关，native 单纯形基同样需要它，`inviscid.py`
目前对 native 网格完全跳过这一步（`jacobians_fine=None`），只是因为
这套算子此前不存在（见 Part8 文档"四、明确未做的后续工作"第4条）。

本文件的三件套与 `build_overintegration_operators` 逐项对应，只是
换成 native 单纯形基（`native_simplex_basis.py` 的受限 PKD 模态 +
Warp&Blend 节点），数学结构完全一致：
1. `interp_c2f`：**COARSE 阶数**模态基在 FINE 点上取值——Q 本身次数
   <=order，这一步是精确插值，不引入混叠。
2. `D_fine`：**FINE 阶数**原生单纯形微分矩阵（直接复用
   `build_native_tet_operators(over_order)`，不是新公式）。
3. `restrict_f2c`：**FINE 阶数**模态基在 COARSE 点上取值——把微分后
   的场从细网格精确插值（不是模态截断投影：过积分的目的正是要保留
   非线性通量的高阶内容,不能在算完导数后又把它截断掉,这一步只是把
   已经算好的、真实值得信赖的细网格结果在 COARSE 点上原样取值）回
   coarse SPs。

与坍缩坐标版本的关键差异：native 单纯形基节点数随阶数增长的方式不同
（`(order+1)(order+2)(order+3)/6` vs `(order+1)^3`），过积分阶数
`over_order` 仍然沿用与坍缩坐标完全相同的 `min(2*order,
OVERINTEGRATION_MAX_ORDER)` 经验法则（二次非线性去混叠的标准做法，
两套基没有理由用不同的过积分阶数选择）。
"""

from typing import Tuple

import numpy as np

from .native_simplex_basis import (
    build_native_tet_operators,
    restricted_tet_modes,
    simplex3d_value,
    rst_to_abc,
)


def _native_modal_vandermonde(ref_rst: np.ndarray, modes) -> np.ndarray:
    """给定一批参考坐标 (r,s,t) 和一组 (i,j,k) 模态索引，返回
    Vandermonde 矩阵 (n_pts, n_modes)——`build_native_tet_lift`/
    `build_native_tet_boundary_extrap` 已经在各自函数体内重复写过
    这个三行代码，这里单独提出来给过积分算子复用，避免第四份重复。
    """
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    return np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])


def build_native_tet_overintegration_operators(
    order: int, over_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造 native 四面体体积项去混叠三件套，与
    `build_overintegration_operators("tet", order, over_order, ...)`
    同样的用途和消费方式（`fr_residual/inviscid.py` 的过积分分支）。

    Args:
        order: 当前求解阶数 P（coarse）
        over_order: 过积分阶数（`min(2*order, OVERINTEGRATION_MAX_ORDER)`，
            与坍缩坐标版本同一个经验法则，见模块文档）

    Returns:
        (ref_rst_fine, interp_c2f, D_fine, restrict_f2c)：
        `ref_rst_fine` 形状 (n_fine,3)；`interp_c2f` 形状
        (n_fine,n_coarse)；`D_fine` 形状 (n_fine,n_fine,3)；
        `restrict_f2c` 形状 (n_coarse,n_fine)——与坍缩坐标版本完全
        对应的形状约定，`inviscid.py` 的过积分分支可以按同一套
        `contract_shared_operator_1axis/2axis` 消费，不需要改动那段
        通用代码本身。
    """
    from scipy.linalg import lu_factor, lu_solve

    ref_coarse, _ = build_native_tet_operators(order)
    ref_fine, D_fine = build_native_tet_operators(over_order)

    modes_coarse = restricted_tet_modes(order)
    modes_fine = restricted_tet_modes(over_order)

    V_coarse_sps = _native_modal_vandermonde(ref_coarse, modes_coarse)
    V_coarse_at_fine = _native_modal_vandermonde(ref_fine, modes_coarse)
    V_fine_at_fine = _native_modal_vandermonde(ref_fine, modes_fine)
    V_fine_at_coarse = _native_modal_vandermonde(ref_coarse, modes_fine)

    lu_coarse = lu_factor(V_coarse_sps.T)
    interp_c2f = lu_solve(lu_coarse, V_coarse_at_fine.T).T

    lu_fine = lu_factor(V_fine_at_fine.T)
    restrict_f2c = lu_solve(lu_fine, V_fine_at_coarse.T).T

    return ref_fine, interp_c2f, D_fine, restrict_f2c
