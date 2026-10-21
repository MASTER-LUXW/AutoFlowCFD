"""AutoFlowCFD V2.0 - DES 网格尺度(h_max / h_wn)"""

import numpy as np


def compute_h_max_and_h_wn(mesh) -> tuple:
    """逐单元 IDDES 网格尺度所需的两个几何量：h_max（最大边长）与 h_wn（近壁法向网格间距）。

    边长取网格质量检查同一份函数（`quality_metrics.tetrahedron_edge_lengths / prism_edge_lengths`）；单元顺序为
    HighOrderMesh 的约定：前 n_prism_cells 个是棱柱，其余是四面体。

    h_wn 的取值：
    - 棱柱（边界层挤出层，本项目唯一刻意各向异性的单元）：9 条边中第 6/7/8 列是竖直边（v_i -> w_i，挤出方向，
      构造上即壁面法向），取三者最小值——法向间距是最薄的那个方向，且不依赖任何距离场估计。
    - 四面体（核心 / 尾流区，生成时接近各向同性）：h_wn = h_max。h_wn 只进入 IDDES 的 `f_e2` 近壁抬升项，而近壁区
      按网格架构全部是棱柱；四面体所在处 `d_wall` 大、`alpha = 0.25 - d_w/h_max` 深度为负，f_e2 的近壁修正不激活，
      这个取值不进入起作用的那一项。

    不用"单元顶点在当地壁面法向上的投影跨度、法向取质心减最近壁面节点"（2026-09-14 实测证伪）：单元厚度远小于面内
    尺寸时（边界层单元的典型形态），该方向被面内偏移主导。真实法向跨度 0.03、面内尺寸约 1 的棱柱上，随壁面节点
    间距 0.40 -> 0.005 得到的相对误差为 21.8、22.5、0.0、12.4、7.0、3.6 倍，并不随节点变密单调收敛。要做对需要真正的
    法向场（如 `grad(d_wall)` 归一化）；棱柱竖直边本来就是法向跨度。

    Args:
        mesh: HighOrderMesh（读 n_cells / n_prism_cells 与修正朝向后的节点坐标、棱柱 / 四面体连接）

    Returns:
        (h_max, h_wn): 均为形状 (n_cells,) 的数组
    """
    from autoflowcfd.grid.validation.quality_metrics import (
        prism_edge_lengths, tetrahedron_edge_lengths,
    )

    n_cells = mesh.n_cells
    n_prism = mesh.n_prism_cells
    h_max = np.zeros(n_cells, dtype=np.float64)
    h_wn = np.zeros(n_cells, dtype=np.float64)

    if n_prism > 0:
        prism_edges = prism_edge_lengths(mesh._node_coords, mesh._fixed_prism_conn)
        h_max[:n_prism] = np.max(prism_edges, axis=1)
        h_wn[:n_prism] = np.min(prism_edges[:, 6:9], axis=1)

    n_tet = n_cells - n_prism
    if n_tet > 0:
        tet_edges = tetrahedron_edge_lengths(mesh._node_coords, mesh._fixed_tet_conn)
        tet_h_max = np.max(tet_edges, axis=1)
        h_max[n_prism:] = tet_h_max
        h_wn[n_prism:] = tet_h_max

    return h_max, h_wn
