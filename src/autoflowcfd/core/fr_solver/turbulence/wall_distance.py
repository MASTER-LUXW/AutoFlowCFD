"""AutoFlowCFD V2.0 - 壁面距离场的计算与跨阶数重算。

从 `core/fr_solver/turbulence.py` 拆出（2026-09-24）。纯搬家，逻辑未改。

**为什么它是独立一块**：壁距是**纯几何量**，不是解多项式场。2026-09-05
的一个真实缺陷正是把它当解场在阶数切换时插值/广播，正确做法是重新做
KD-Tree 查询 —— `recompute_wall_distance_for_current_order` 就是为此存在。
"""

from typing import Optional

import numpy as np
from loguru import logger

from autoflowcfd.core.utils.wall_distance import compute_wall_distance


def compute_wall_distance_field(
    solver,
    mesh_nodes: np.ndarray,
    wall_indices: np.ndarray,
    connectivity: Optional[np.ndarray] = None,
    use_eikonal: bool = False,
) -> None:
    """计算壁面距离场（用于 DDES/WMLES/SST），映射到 SPs。

    Args:
        solver: FRSolver 实例
        mesh_nodes: 全部网格节点坐标，shape=(n_nodes, 3)
        wall_indices: WALL 边界节点索引
        connectivity: 节点邻接表（见 grid.node_connectivity.
            build_node_adjacency），use_eikonal=True 时必须提供，否则
            忽略——Eikonal（图最短路径近似）沿网格拓扑传播距离，需要这张图
        use_eikonal: True 时用 Eikonal 方程近似求解壁面距离（更符合复杂/
            凹形几何的真实"沿流场路径"距离，例如轮腔、地板下这类通道里，
            几何最近的墙面点可能隔着一层薄壁——纯欧氏 KD-Tree 会算出一个
            物理上不成立、偏小的距离，Eikonal 沿网格边传播就不会有这个
            问题）。False（默认）用纯欧氏 KD-Tree，更快，对开阔区域足够
    """
    if solver.turb_model_name not in ["SST", "DDES", "IDDES", "WMLES", "LES"]:
        logger.warning(f"Turbulence model {solver.turb_model_name} does not require wall distance")
        return

    logger.info("Computing wall distance field...")
    node_distances = compute_wall_distance(
        mesh_nodes, wall_indices, connectivity=connectivity, use_eikonal=use_eikonal
    )
    logger.info(f"Node-level wall distance computed: min={node_distances.min():.6f}, max={node_distances.max():.6f}")

    n_cells, n_sps = solver.state.U.shape[:2]

    if use_eikonal:
        # Eikonal 距离只在网格节点上定义（沿网格拓扑传播的结果），必须从
        # 最近节点的 node_distances 取值映射到 SP/单元中心 - 不能像下面
        # use_eikonal=False 分支那样直接对查询点坐标重新做一次"到 WALL
        # 节点最近欧氏距离"的独立几何查询,那样等于完全无视了 Eikonal 沿
        # 网格拓扑传播出来的结果,直接退化回它原本要避免的那种纯直线距离。
        query_points = (
            solver.mesh.sps_coords.reshape(-1, 3)
            if hasattr(solver.mesh, "sps_coords") and solver.mesh.sps_coords is not None
            else getattr(solver.mesh, "cell_centers", None)
        )
        if query_points is not None:
            mapped = _map_node_distances_to_points(mesh_nodes, node_distances, query_points)
            solver.wall_distance = (
                mapped.reshape(n_cells, n_sps)
                if mapped.shape[0] == n_cells * n_sps
                else np.tile(mapped[:, np.newaxis], (1, n_sps))
            )
            logger.info(
                f"Eikonal wall distance field mapped: shape={solver.wall_distance.shape}, "
                f"min={solver.wall_distance.min():.6f}, max={solver.wall_distance.max():.6f}"
            )
            # 见下面 KD-Tree 分支同一处缓存的文档：Eikonal 求解出的
            # `node_distances`（节点级、拓扑传播结果）与 `mesh_nodes`
            # 同样是与阶数无关的量——缓存下来供阶数切换时重新映射到新
            # SPs，不需要重新解一遍昂贵的 Eikonal 图最短路径问题。
            solver._eikonal_recompute_cache = (mesh_nodes, node_distances)
            return
        solver.wall_distance = np.ones((n_cells, n_sps)) * node_distances.mean()
        logger.info(f"Eikonal wall distance field initialized (no SP/cell-center coords available, using mean): "
                    f"{solver.wall_distance.mean():.6f}")
        return

    if hasattr(solver.mesh, "sps_coords") and solver.mesh.sps_coords is not None:
        sps_coords = solver.mesh.sps_coords
        flat_sps = sps_coords.reshape(-1, 3)
        try:
            from scipy.spatial import cKDTree

            wall_coords = mesh_nodes[wall_indices]
            tree = cKDTree(wall_coords)
            dist_flat, _ = tree.query(flat_sps, k=1)
            solver.wall_distance = dist_flat.reshape(n_cells, n_sps)
            logger.info(
                f"Wall distance field mapped to SPs: shape={solver.wall_distance.shape}, "
                f"min={solver.wall_distance.min():.6f}, max={solver.wall_distance.max():.6f}"
            )
            # 真实 bug 修复（2026-09-06，cube_demo 真实网格 Order
            # Continuation P0->P1 升阶后 k_mean 持续增长排查发现，见
            # `recompute_wall_distance_for_current_order` 文档完整推导）：
            # 缓存这个纯几何量（`wall_coords`，WALL 节点物理坐标，与阶数
            # 完全无关）供阶数切换时重新查询用，不是留给 Order
            # Continuation 去插值/平均这个已经算好的解——壁面距离不是
            # 解多项式场，插值/平均它在数学上没有依据（见该函数文档）。
            solver._wall_coords_for_recompute = wall_coords
        except Exception as e:
            logger.warning(f"SP-level mapping failed ({e}), falling back to cell-center mapping")
            _map_wall_distance_fallback(solver, node_distances, mesh_nodes, wall_indices, n_cells, n_sps)
    else:
        _map_wall_distance_fallback(solver, node_distances, mesh_nodes, wall_indices, n_cells, n_sps)


def _map_node_distances_to_points(
    mesh_nodes: np.ndarray, node_distances: np.ndarray, query_points: np.ndarray
) -> np.ndarray:
    """把节点级标量场（这里是 Eikonal 壁面距离）映射到任意查询点：每个
    查询点取其最近网格节点的场值。

    这是"节点上有定义、别处没有的标量场"映射到任意坐标最标准的做法（在没有
    另外接入 FR 基函数插值的前提下）——查询点到最近节点之间还有一段真实的
    几何偏移误差，量级受限于局部网格尺寸，是这类映射固有的、可接受的近似
    误差，不是本函数的缺陷。

    Args:
        mesh_nodes: 全部网格节点坐标，shape=(n_nodes, 3)
        node_distances: 节点级壁面距离，shape=(n_nodes,)
        query_points: 待映射的坐标点，shape=(n_query, 3)

    Returns:
        shape=(n_query,) 每个查询点对应的（最近节点的）壁面距离
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(mesh_nodes)
    _, nearest_node = tree.query(query_points, k=1)
    return node_distances[nearest_node]


def _map_wall_distance_fallback(solver, node_distances, mesh_nodes, wall_indices, n_cells, n_sps) -> None:
    """壁面距离映射的回退策略：基于单元中心或节点平均。"""
    if hasattr(solver.mesh, "cell_centers") and solver.mesh.cell_centers is not None:
        centers = solver.mesh.cell_centers
        try:
            from scipy.spatial import cKDTree

            wall_coords = mesh_nodes[wall_indices]
            tree = cKDTree(wall_coords)
            dist_centers, _ = tree.query(centers, k=1)
            solver.wall_distance = np.tile(dist_centers[:, np.newaxis], (1, n_sps))
            # 见 compute_wall_distance_field 主 KD-Tree 分支同一处缓存的
            # 文档。这条回退路径本来就是单元中心近似（同一单元内全部 SP
            # 共享一个值），阶数切换后重新查询不会恢复出真正的逐 SP
            # 分辨率（那需要 sps_coords，这条分支恰恰是它不可用时才走到
            # 这里）——但缓存下来仍然让"阶数切换后重新查询"这个统一机制
            # 对这条分支也保持诚实一致，而不是让它继续被插值/平均污染。
            solver._wall_coords_for_recompute = wall_coords
            return
        except Exception as e:
            # 第四次评审修复：此前静默吞掉异常且不记录原因（对比姊妹分支
            # 会 logger.warning(f"...failed ({e})...")）——壁面距离对
            # SST/DDES/WMLES 的近壁阻尼函数、F1/F2 混合函数、DDES 长度
            # 尺度都是关键输入，下面"全域单一平均值"的兜底会显著且静默地
            # 破坏这些模型的近壁行为，至少要把被吞掉的异常原因记下来。
            logger.warning(
                f"Cell-center wall distance KD-tree query failed ({e}), "
                f"falling back to a single mean value for the whole domain "
                f"- this will noticeably degrade near-wall turbulence model behavior."
            )

    solver.wall_distance = np.ones((n_cells, n_sps)) * node_distances.mean()
    logger.info(f"Wall distance field initialized (fallback): mean={solver.wall_distance.mean():.6f}")


def recompute_wall_distance_for_current_order(solver) -> bool:
    """真实 bug 修复（2026-09-06，cube_demo 791,492 单元真实网格 Order
    Continuation P0->P1 升阶后长程续算 k_mean 持续增长排查发现）：阶数
    切换（`run_order_continuation`/`interpolate_to_new_order`）此前对
    `solver.wall_distance` 做的是和 U/k_field/omega_field 同一套精确
    Lagrange 延拓插值（升阶时）或 `np.mean` 压缩（resume 场景重置到 P0
    时）——但壁面距离**不是解多项式场**，它是纯几何量（每个 SP 到最近
    WALL 节点的欧氏/Eikonal 距离），插值/平均它在数学上没有依据：
    - 升阶时：P0 只有 1 个 SP，其 Lagrange 基函数恒为常数 1（见
      `_lagrange_basis_matrix_1d` 文档"n=1 时退化为常数基函数"一节），
      所以"插值"实际上是把 P0 那 1 个 SP（本质是单元形心附近的壁面
      距离）**原样广播**给新阶数的全部 SP——边界层棱柱单元内近壁 SP
      和远壁 SP 的真实壁面距离可以相差好几个数量级，广播后这个差异
      被完全抹平，所有 SP 拿到同一个（通常偏大，因为是形心附近而不是
      最近壁面的那个 SP）值。
    - resume 重置到 P0 时：`np.mean` 把原阶数逐 SP 的精确值压缩成 1 个
      单元平均值，同样丢失了单元内的空间分辨率，且这个丢失是**不可逆
      的**——因为后续升阶用的插值就是把这个已经丢了分辨率的 P0 值
      原样广播回去，见上一条。

    真实后果（真实网格决定性验证，见项目记忆 cube_demo_gradvel_and_
    omega_wall_fixes_2026_09_05）：壁面距离是 SST F1 blending 判据、
    Wilcox 壁面 omega 目标值（`omega_wall=60*nu/(beta1*d1^2)`，
    `d1=np.min(wall_distance[owner_cells],axis=1)`）的关键输入——一旦
    同一单元内全部 SP 共享同一个值，`np.min(...)` 这一步彻底失效（常数
    数组取 min 恒等于其自身），"该单元里离墙最近的 SP"这个信息在阶数
    切换时就已经在上游被抹掉了，d1 系统性偏大，omega_wall 目标值系统性
    偏小（平方反比放大误差），`enforce_omega_wall_relaxation` 把 omega
    往一个本来就偏低的目标值松弛，近壁约束天然弱化，omega 更容易塌陷，
    经 nu_t 近零分母被放大，持续向平均流注入过量涡粘——这是 bug②
    （omega 壁面扩散侧未闭合）修复后问题被推迟而非根治的直接原因：
    "病灶"从"omega 扩散侧未闭合"变成了"约束用的 d1 输入值本身在阶数
    切换后就已经失真"。

    修复：不插值/不平均，阶数切换后直接用缓存下来的、与阶数无关的纯
    几何量（`solver._wall_coords_for_recompute`——WALL 节点物理坐标；
    或 `solver._eikonal_recompute_cache`——Eikonal 节点级距离场，避免
    重新解一遍图最短路径问题）+ 当前阶数真实的 `solver.mesh.sps_coords`
    重新做一次 KD-Tree/最近节点映射查询，与 `compute_wall_distance_
    field` 首次构造时完全同一套逻辑，只是复用缓存、不重新遍历边界节点。

    调用时机：必须在 `solver.mesh.set_order(target_p)`（或 P0 重置分支
    的 `set_order(0)`）**之后**调用——`solver.mesh.sps_coords` 只有在
    `set_order` 完成后才反映新阶数的真实 SP 坐标。

    Returns:
        True：成功用缓存的几何量重新查询，`solver.wall_distance` 已是
            当前阶数下精确的逐 SP 值。
        False：没有可用的缓存（例如从未调用过 `compute_wall_distance_
            field`，或该次调用走的是"既没有 sps_coords 也没有
            cell_centers"的最终兜底分支）——调用方应保留原有的插值/
            平均结果作为退化但至少形状正确的后备，不能让
            `solver.wall_distance` 变成 None 或形状不匹配。
    """
    if getattr(solver, "wall_distance", None) is None:
        return False

    n_cells = solver.mesh.n_cells
    n_sps = solver.mesh.n_sps_per_cell

    wall_coords = getattr(solver, "_wall_coords_for_recompute", None)
    if wall_coords is not None:
        from scipy.spatial import cKDTree

        tree = cKDTree(wall_coords)
        if hasattr(solver.mesh, "sps_coords") and solver.mesh.sps_coords is not None:
            flat_sps = solver.mesh.sps_coords.reshape(-1, 3)
            dist_flat, _ = tree.query(flat_sps, k=1)
            solver.wall_distance = dist_flat.reshape(n_cells, n_sps)
        elif hasattr(solver.mesh, "cell_centers") and solver.mesh.cell_centers is not None:
            dist_centers, _ = tree.query(solver.mesh.cell_centers, k=1)
            solver.wall_distance = np.tile(dist_centers[:, np.newaxis], (1, n_sps))
        else:
            return False
        logger.info(
            f"[Order Continuation] Wall distance recomputed (not interpolated) for P{solver.current_order}: "
            f"shape={solver.wall_distance.shape}, min={solver.wall_distance.min():.6e}, "
            f"max={solver.wall_distance.max():.6e}"
        )
        return True

    eikonal_cache = getattr(solver, "_eikonal_recompute_cache", None)
    if eikonal_cache is not None and hasattr(solver.mesh, "sps_coords") and solver.mesh.sps_coords is not None:
        mesh_nodes, node_distances = eikonal_cache
        query_points = solver.mesh.sps_coords.reshape(-1, 3)
        mapped = _map_node_distances_to_points(mesh_nodes, node_distances, query_points)
        solver.wall_distance = mapped.reshape(n_cells, n_sps)
        logger.info(
            f"[Order Continuation] Eikonal wall distance remapped (not interpolated) for "
            f"P{solver.current_order}: shape={solver.wall_distance.shape}"
        )
        return True

    return False
