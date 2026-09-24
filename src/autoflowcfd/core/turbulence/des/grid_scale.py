"""AutoFlowCFD V2.0 - DES 网格尺度(h_max / h_wn)

从 `src/autoflowcfd/core/turbulence/des.py`(原 657 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np


def compute_h_max_and_h_wn(mesh) -> tuple:
    """逐单元 IDDES 网格尺度所需的两个几何量：h_max（最大边长）与
    h_wn（近壁法向网格间距的估计）。

    复用 grid/validation/quality_metrics.py 中已验证的边长计算函数
    （tetrahedron_edge_lengths/prism_edge_lengths），不重新实现单元边长
    几何——这两个函数原本用于网格质量的长宽比计算，这里直接对同一批
    (nodes, cells) 数据复用。单元顺序遵循 HighOrderMesh 自身的约定
    （见 grid/high_order/high_order_mesh.py：cell_type = "prism" if
    i < n_prism_cells else "tet"）：前 n_prism_cells 个是棱柱，其余是
    四面体。

    h_wn（wall-normal spacing）取值约定：
    - 棱柱单元（本项目里唯一真正各向异性设计的单元类型，边界层挤出层）：
      prism_edge_lengths 返回的 9 条边中，第 6/7/8 列（索引 6:9）是竖直边
      （v_i -> w_i，边界层挤出方向，近似壁面法向），取这 3 条边的最小值
      作为该单元的法向间距估计——法向间距应该是"最薄"的那个方向。
    - 四面体单元（核心/尾流 LES 区域，本项目网格生成上设计为接近各向
      同性，不像棱柱层刻意拉伸）：没有明确的"法向"方向概念，取
      h_wn = h_max。
      **这条标注已更正（2026-09-14）**：此前写"是标准简化"，那个说法不
      准确，正确的理由是——h_wn 只进入 IDDES 的 `f_e2` **近壁抬升项**，
      而本项目的近壁区按网格架构设计**全部是棱柱**（边界层挤出层，见
      项目记忆 `tet_collapsed_coord_anisotropy`：近壁/高剪切区必须用
      棱柱，这是架构要求而不是偏好）。四面体只出现在核心/尾流区，那里
      `d_wall` 很大、`alpha = 0.25 - d_w/h_max` 深度为负、f_e2 的近壁
      修正本来就不激活——所以这个取值实际上从不进入起作用的那一项。
      它不是"用近似换简单"，而是在该项不激活的区域里取的合理定义。

    被证伪的替代方案（2026-09-14 实测，记录下来避免重试）：
      曾尝试改用"h_wn 的定义本身"——单元顶点在**当地壁面法向**上的投影
      跨度，法向取 `centroid - 最近壁面节点`（KD-Tree 查询，壁面距离场
      本来就是这么算的）。**这个估计量在近壁区是坏的**：当单元厚度远
      小于它自身的面内尺寸时（边界层单元的典型形态），"质心减最近壁面
      节点"的方向被**面内偏移**主导，与真实法向相差很大。
      用一个真实法向跨度 0.03、面内尺寸 ~1 的棱柱实测（壁面为 z=0 平面，
      扫描壁面节点间距）：
          间距 0.4000 -> h_wn=0.683（相对误差 21.8 倍）
          间距 0.1818 -> h_wn=0.704（22.5 倍）
          间距 0.0833 -> h_wn=0.030（0.00，恰好有节点落在质心正下方）
          间距 0.0400 -> h_wn=0.402（12.4 倍）
          间距 0.0100 -> h_wn=0.241（7.0 倍）
          间距 0.0050 -> h_wn=0.139（3.6 倍）
      注意它**随壁面节点变密并不单调收敛**——最近节点在面内邻域里基本
      是任意的，方向因此始终带着一个 O(面内间距) 的横向分量。
      要做对需要一个真正的法向场估计（例如 `grad(d_wall)` 归一化，或
      直接用边界面法向沿法线传播），那是另一项独立工作。
      现有的棱柱处理**恰恰是更稳健的那个**：棱柱的挤出方向在构造上就是
      壁面法向，取竖直边就是取法向跨度，不依赖任何距离场估计。

    精确 h_wn（2026-09-14，用户明确要求"本项目从不接受简化"后补齐）：
    传入 `wall_coords`（壁面节点坐标）时，h_wn 改用它的**定义本身**算，
    对任何单元形状都精确、不需要"各向同性"这类前提：
      1. 用 KD-Tree 找每个单元质心最近的壁面点，得到该处的壁面法向
         `n_hat = (centroid - nearest_wall_point) / |...|`；
      2. h_wn = 该单元全部顶点在 `n_hat` 上投影的**跨度**
         （max - min）——这正是"单元在壁面法向上的网格间距"，也就是
         Shur et al. 2008 里 h_wn 的物理含义。
    这条定义同时取代了棱柱的特例处理（"取 3 条竖直边的最小值"）：对
    真正沿壁面法向挤出的棱柱，两者数值上几乎相同（3 条竖直边近似等长），
    但投影跨度对斜挤出/扭曲棱柱仍然正确，而"最小竖直边"会低估。
    统一用一条定义也消掉了棱柱/四面体的分支差异。

    Args:
        mesh: HighOrderMesh 实例（需要 n_cells/n_prism_cells/_node_coords/
            _fixed_prism_conn/_fixed_tet_conn，均为该类已建立的内部
            几何属性，fr/face_flux_points*.py 已有跨模块访问这些属性的
            先例）
        wall_coords: (n_wall, 3) 壁面节点坐标。给出时 h_wn 走上面的精确
            定义；为 None 时退回几何启发式（棱柱取最小竖直边、四面体取
            h_max）——**这条退路只在调用方确实拿不到壁面节点时使用**
            （例如 DDES 只需要 h_max、根本不消费 h_wn 的场景）。IDDES
            的生产路径一定会先算壁面距离，那时壁面节点是已知的，见
            `fr_solver/turbulence.py::compute_wall_distance_for_solver`。

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
