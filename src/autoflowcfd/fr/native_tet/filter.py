"""AutoFlowCFD V2.0 - native 四面体（路径C）指数模态滤波器。

背景：`fr/modal_filter.py`（坍缩坐标方案）是抑制节点配置法混叠失稳的
标准对策（该模块文档有完整原理/真实复现记录），但它的 `build_tet_
modal_filter` 按 `max(i,j,k)/order` 归一化——这是坍缩坐标"扩展张量积"
基（i,j,k 各自独立取 0..order，`(order+1)^3` 个模态，其中大量是超出
真实四面体多项式空间维度的冗余自由度）的专属判据，该模块文档明确记录
了"总阶数 i+j+k 归一化在四面体上会放大随机白噪声"这一真实反例。

native 基（`native_tet/basis.py`）不是同一个数学对象：模态严格是
`i+j+k<=order` 的最小 PKD/Dubiner 单纯形基，`i+j+k` 本身就是标准意义
下的多项式总阶数（不含任何冗余自由度）——这是 Hesthaven-Warburton
《Nodal DG》原书对单纯形谱元用的标准指数滤波器归一化方式，不是需要
重新发明的判据，与坍缩坐标那套判据的差异来自两套基底本身的数学结构
不同，不是"抄近路"或"想当然套用"。

衰减公式、`off` 档短路与 sigma 的选取全部复用
`fr/modal_filter.py::assemble_modal_filter`（同一个 Hesthaven-Warburton
推荐值，两套基没有理由用不同的衰减陡度），本文件只提供**归一化判据**与
Vandermonde —— 那才是两套基真正不同的地方。
"""

import numpy as np

from ..modal_filter import assemble_modal_filter
from .basis import (
    build_native_tet_operators,
    restricted_tet_modes,
    simplex3d_value,
    rst_to_abc,
)


def build_native_tet_modal_filter(order: int) -> np.ndarray:
    """native 四面体模态滤波矩阵，形状 (n_native_sps, n_native_sps)。

    `eta=(i+j+k)/order`（总多项式阶数归一化，见模块文档），
    `sigma(eta)=exp(-alpha*eta^(2s))`——低阶（真正被解析到的物理场）
    模态 sigma≈1 不衰减，只有最高阶模态被压到机器精度量级。

    Args:
        order: 多项式阶数 P

    Returns:
        F: (n_native_sps, n_native_sps)，`F @ field(native SPs)` 给出
            滤波后的节点值——与 `build_tet_modal_filter` 同样的消费
            方式（`F @ 节点值`），供 `fr/operators.py` 按
            `tet_basis_mode=="native"` 构造并接入求解器主循环。
    """
    if order == 0:
        return np.eye(1)

    # `off` 档的短路现在在 `modal_filter.assemble_modal_filter` 里 ——
    # 它当年在本函数里被漏掉过（后果是 AFCFD_FILTER_MODE=off 只关掉了
    # 棱柱的滤波器，而那张网格上四面体占 82.7%），收到一处之后再加一条
    # 基不可能漏掉它。完整记录见那个函数的文档。
    ref_rst, _ = build_native_tet_operators(order)
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    modes = restricted_tet_modes(order)
    V = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])

    # sigma 的选取（含 project 档要求严格取 {0,1} 以保证幂等）在
    # `assemble_modal_filter` 里，见 modal_filter.py 模块文档
    # "为什么需要 project 档"一节。
    etas = [(i + j + k) / order for (i, j, k) in modes]
    return assemble_modal_filter(V, etas)
