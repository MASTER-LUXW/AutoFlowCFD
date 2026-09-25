"""AutoFlowCFD V2.0 - 单机求解器的壁面距离场（施加与跨阶数重算）。

壁距是**纯几何量**，不是解多项式场。它只由一个与阶数、分区、后端都无关的
来源决定（`core/utils/wall_distance_source.py::WallDistanceSource`：壁面节点
坐标的 KD-Tree，或节点级 Eikonal 距离），每个阶数都在当时的 SP 坐标上重新
查询一次——2026-09-05 的一个真实缺陷正是把它当解场在阶数切换时插值/广播
（见 `recompute_wall_distance_for_current_order`）。

2026-09-25：来源对象取代了此前的两个专用缓存（`_wall_coords_for_recompute` /
`_eikonal_recompute_cache`）与"单元中心 / 全域平均值"两级兜底——兜底会让
近壁湍流行为静默变坏，单机的 CLI 入口本来就在没有壁面时报错，这里与之
一致。分布式与 GPU 后端用同一个来源（此前它们各有一份把 `BoundaryMap`
的数组当字典读的拷贝，真实网格上构造即崩溃，见来源模块文档）。
"""

import numpy as np
from loguru import logger

from autoflowcfd.core.utils.wall_distance_source import WallDistanceSource

#: 需要壁面距离场的湍流模型
WALL_DISTANCE_MODELS = ("SST", "DDES", "IDDES", "WMLES", "LES")


def apply_wall_distance_source(solver, source: WallDistanceSource) -> None:
    """在求解器当前阶数的 SP 坐标上查询壁距，并记住来源供换阶重查。"""
    sps = getattr(solver.mesh, "sps_coords", None)
    if sps is None:
        raise RuntimeError(
            "网格没有 sps_coords，无法把壁面距离映射到解点——不退化为单元中心或全域"
            "平均值（那会静默破坏近壁湍流行为）。")
    n_cells, n_sps = solver.state.U.shape[:2]
    solver.wall_distance = source.query(np.asarray(sps).reshape(n_cells, n_sps, 3))
    solver._wall_distance_source = source
    logger.info(
        f"Wall distance ({source.kind}) mapped to SPs for P{getattr(solver, 'current_order', '?')}: "
        f"shape={solver.wall_distance.shape}, min={solver.wall_distance.min():.6e}, "
        f"max={solver.wall_distance.max():.6e}")


def compute_wall_distance_field(solver, mesh_nodes: np.ndarray, wall_indices: np.ndarray,
                                connectivity=None, use_eikonal: bool = False) -> None:
    """由网格节点与壁面节点索引构造来源并施加（`FRSolver.compute_wall_distance_field`）。

    Args:
        mesh_nodes: 全部网格节点坐标 `(n_nodes, 3)`
        wall_indices: WALL 边界面上的节点索引
        connectivity: 节点邻接表（`build_node_adjacency`），`use_eikonal=True` 时必需
        use_eikonal: True 用 Eikonal（沿网格拓扑传播，凹形/通道几何里比直线距离
            更符合"沿流场路径"的距离），False 用欧氏 KD-Tree
    """
    if solver.turb_model_name not in WALL_DISTANCE_MODELS:
        logger.warning(f"Turbulence model {solver.turb_model_name} does not require wall distance")
        return
    apply_wall_distance_source(solver, WallDistanceSource.from_nodes(
        mesh_nodes, wall_indices, connectivity=connectivity, use_eikonal=use_eikonal))


def recompute_wall_distance_for_current_order(solver) -> bool:
    """阶数切换后在新 SP 坐标上重新查询壁距（不插值、不平均）。

    **为什么不能插值/平均（2026-09-06 真实缺陷）**：此前阶数切换对
    `solver.wall_distance` 做与 U/k/omega 同一套 Lagrange 延拓（升阶）或
    `np.mean` 压缩（resume 重置到 P0）。但壁距不是解多项式场：P0 只有 1 个
    SP，其"插值"就是把形心附近的值原样广播给新阶数全部 SP——边界层棱柱
    单元内近壁与远壁 SP 的真实壁距可以差几个数量级，广播后被完全抹平；
    `np.mean` 同样丢失单元内分辨率且不可逆。后果：Wilcox 壁面 omega 目标值
    `omega_wall = 60*nu/(beta1*d1^2)` 里的 `d1 = min(wall_distance[owner])`
    系统性偏大，目标值偏小（平方反比放大），近壁约束弱化、omega 更易塌陷，
    经涡粘持续向平均流注入过量粘性（cube_demo 真实网格决定性验证）。

    调用时机：必须在 `solver.mesh.set_order(target_p)` **之后**（`sps_coords`
    那时才反映新阶数）。

    Returns:
        True：已按来源重新查询；False：求解器从未施加过壁距（例如层流），
        调用方不需要壁距。
    """
    source = getattr(solver, "_wall_distance_source", None)
    if source is None or getattr(solver, "wall_distance", None) is None:
        return False
    apply_wall_distance_source(solver, source)
    return True
