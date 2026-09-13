"""
AutoFlowCFD V2.0 - FRSolver 湍流模型管理 (从 fr_solver.py 拆分)

本文件把 FRSolver 里与湍流模型初始化/壁面距离/源项计算/涡粘度耦合相关
的逻辑拆出来，避免 fr_solver.py 单文件过长（>400行需拆分的项目规范）。
函数签名都以 `solver: FRSolver` 为第一参数，FRSolver 里保留同名的薄
委托方法，调用方式不变——与代码库里 solver_helpers.py/order_continuation.py
已经在用的委托模式一致。
"""

from typing import Optional

import numpy as np
from loguru import logger

from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.des import DDESModel, IDDESModel, compute_h_max_and_h_wn
from autoflowcfd.core.turbulence.wmles import WMLESModel
from autoflowcfd.core.turbulence.sgs import WALEModel
from autoflowcfd.core.utils.wall_distance import compute_wall_distance
from autoflowcfd.core.fr_residual.viscous import compute_gradients as _compute_gradients_generic


def _set_freestream_turbulence(solver) -> tuple:
    """根据来流条件从 Tu/VR 推导物理自洽的 k/omega 初值。

    工业 RANS 标准做法（Fluent 用户手册 Section 7.3.2、OpenFOAM 通用实践）：
    不直接指定 k 和 omega，而是从湍流强度 Tu 和粘性比 VR 推导，
    确保 k 和 omega 通过 nu_t 物理耦合，避免拍脑袋组合导致源项失衡。

    公式：
        k_inf   = 1.5 * (U_inf * Tu)^2
        nu_t    = VR * nu（nu = mu/rho 运动粘度）
        omega_inf = k_inf / nu_t

    Returns:
        (k_inf, omega_inf): 来流湍动能和比耗散率
    """
    vel_inf = solver.freestream.get("vel_inf", 33.33)
    rho_inf = solver.freestream.get("rho_inf", 1.225)
    mu = getattr(solver, 'mu_molecular', 1.8e-5)
    nu = mu / max(rho_inf, 1e-10)

    # 外部气动默认值（参考 Fluent 手册：Tu ≤ 1%, VR = 2-10）
    Tu = getattr(solver, '_turbulence_intensity', 0.01)
    VR = getattr(solver, '_viscosity_ratio', 5.0)

    k_inf = 1.5 * (vel_inf * Tu) ** 2
    nu_t_inf = VR * nu
    omega_inf = k_inf / max(nu_t_inf, 1e-30)

    logger.debug(
        f"Freestream turbulence: Tu={Tu}, VR={VR}, "
        f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
        f"tau={1.0/(0.09*omega_inf):.6e}s"
    )
    return k_inf, omega_inf


def _set_turbulence_bounds(solver) -> None:
    """根据来流条件设置 k/omega 物理上界。

    k_max = 0.5 * vel_inf^2：湍动能不可能超过平均流动能（湍流强度 100% 的极限）。
    omega_max = 1e6：远大于任何工程壁面 omega 值（壁面 omega ~ U_tau^2/nu ~ 1e4
    量级，1e6 留 100 倍裕度）。

    不设上界时，SST 输运方程的源项+输运项正反馈会导致 k/omega 指数增长到
    1e260+ 量级（实测 cube_demo 100 步内即达到），而平均流完全不受影响
    （nu_t 被 SST a1 限幅保持合理），形成隐蔽的发散失效模式。
    """
    if not hasattr(solver, 'turb_model') or solver.turb_model is None:
        return
    if not hasattr(solver.turb_model, 'k_max'):
        return  # 不是 SST 模型，无上界属性
    vel_inf = solver.freestream.get("vel_inf", 33.33)
    solver.turb_model.k_max = 0.5 * vel_inf ** 2  # 湍动能 ≤ 平均流动能
    solver.turb_model.omega_max = 1e6  # 保守上界
    logger.debug(
        f"Turbulence bounds set: k_max={solver.turb_model.k_max:.2f}, "
        f"omega_max={solver.turb_model.omega_max:.0e} "
        f"(vel_inf={vel_inf:.2f})"
    )


def _update_production_ramp(solver) -> None:
    """更新湍流产项渐变因子。

    前 N 步内 production_factor 从 0 线性增加到 1，防止初始流场未发展时
    P_k >> D_k（产生项超过耗散项 8 个量级）导致 k/omega 指数爆炸。
    工业 RANS 求解器（Fluent、OpenFOAM）的标准做法。

    渐变完成时设置 _turb_production_ramp_complete = True（一次性标记），
    供 Order Continuation 等上层逻辑检测并重置残差基准值。
    """
    if not hasattr(solver, 'turb_model') or solver.turb_model is None:
        return
    if not hasattr(solver.turb_model, 'production_factor'):
        return
    ramp_steps = getattr(solver, '_turb_production_ramp_steps', 0)
    current_step = getattr(solver, '_turb_ramp_step', 0)
    if ramp_steps <= 0 or current_step >= ramp_steps:
        solver.turb_model.production_factor = 1.0
        # 渐变完成：一次性标记（之前未完成且现在已完成）
        if not getattr(solver, '_turb_production_ramp_complete', False):
            solver._turb_production_ramp_complete = True
            logger.info(
                f"[ProductionRamp] Ramp complete after {ramp_steps} steps, "
                f"production_factor = 1.0"
            )
    else:
        solver.turb_model.production_factor = current_step / ramp_steps
    # 递增计数器（每调用一次代表一个迭代步）
    solver._turb_ramp_step = current_step + 1


def init_turbulence_models(solver, n_cells: int, n_sps: int) -> None:
    """初始化湍流模型（对应 FRSolver._init_turbulence_models）。"""
    # 湍流产项渐变计数器（与迭代步数同步，控制 production_factor 从 0 渐增到 1）
    solver._turb_ramp_step = 0
    # 渐变完成标记（_update_production_ramp 在渐变完成时设为 True）
    solver._turb_production_ramp_complete = False
    # 渐变步数：前 turb_production_ramp_steps 步内，产生项从 0 线性增加到全量。
    # 工业 RANS 标准做法：防止初始流场未发展时 P_k >> D_k 导致 k/omega 指数爆炸。
    # 50 步足够：配合物理上界限制（k_max, omega_max），k/omega 在此步数内达到准平衡。
    # Fluent 默认 ~50 步，OpenFOAM ~100 步；过长的 ramp 浪费收敛机会。
    solver._turb_production_ramp_steps = 50

    # 从 Tu/VR 推导物理自洽的 k/omega 初值（工业标准）
    k_inf, omega_inf = _set_freestream_turbulence(solver)

    if solver.turb_model_name == "SST":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        _update_production_ramp(solver)
        print(f"   [OK] SST k-omega model initialized "
              f"(k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
              f"production ramp: {solver._turb_production_ramp_steps} steps)")

    elif solver.turb_model_name == "DDES":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        _update_production_ramp(solver)
        solver.ddes_model = DDESModel()
        # h_max（2026-09-02 补齐，与下面 IDDES 分支同一处几何量、同一个
        # 一次性缓存策略）：`apply_to_sst_model` 现在优先用各向异性感知
        # 的 max_edge 网格尺度而不是 cube_root(V)，见该方法文档——本项目
        # 高度依赖棱柱边界层网格，cube_root 会系统性低估扁平单元的 Δ。
        # 只需要 h_max（第一个返回值），h_wn 是 IDDES 专属几何量，DDES
        # 不用，但 compute_h_max_and_h_wn 只有一个返回两者的接口，丢弃
        # 用不到的 h_wn 即可。
        solver._iddes_h_max, _ = compute_h_max_and_h_wn(solver.mesh)
        print(f"   [OK] DDES model initialized (based on SST, "
              f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
              f"production ramp: {solver._turb_production_ramp_steps} steps)")

    elif solver.turb_model_name == "IDDES":
        solver.turb_model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        _set_turbulence_bounds(solver)
        _update_production_ramp(solver)
        solver.ddes_model = IDDESModel()
        # h_max/h_wn 只依赖网格几何（边长），与流场状态无关——mesh 在
        # 整个求解过程中不变，初始化时算一次并缓存在 solver 上，避免
        # 每步都重新调用 quality_metrics 的边长几何计算（见
        # compute_turbulence_source 里 solver._iddes_h_max/_iddes_h_wn
        # 的消费点）。
        solver._iddes_h_max, solver._iddes_h_wn = compute_h_max_and_h_wn(solver.mesh)
        print(f"   [OK] IDDES model initialized (based on SST, "
              f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e}, "
              f"production ramp: {solver._turb_production_ramp_steps} steps)")

    elif solver.turb_model_name == "WMLES":
        # `solver.wmles_model` 已在 `FRSolver.__init__` 第 3 步提前构造
        # （必须先于 boundary_ghost_provider 构造，见该处说明——2026-
        # 09-02 修复的构造顺序 bug）；这里不再重复构造，只在它意外为
        # None 时（例如某个不经过 FRSolver.__init__ 第 3 步、直接调用
        # 本函数的测试/脚本场景）按 GPU 版同一个公式补建，避免真正生产
        # 路径下出现两个物理等价但对象不同的 WMLESModel 实例。
        if getattr(solver, "wmles_model", None) is None:
            rho_inf = solver.freestream.get("rho_inf", 1.225)
            solver.wmles_model = WMLESModel(nu=solver.mu_molecular / max(rho_inf, 1e-10))
        solver.sgs_model = WALEModel()
        print(f"   [OK] WMLES model initialized")

    elif solver.turb_model_name == "LES":
        solver.sgs_model = WALEModel()
        print(f"   [OK] LES with WALE SGS model initialized")

    elif solver.turb_model_name == "NONE":
        print(f"   [OK] Laminar flow (no turbulence model)")

    else:
        raise ValueError(f"Unknown turbulence model: {solver.turb_model_name}")


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


def compute_turbulence_source(solver, dt) -> Optional[tuple]:
    """计算湍流模型源项（对应 FRSolver.compute_turbulence_source）。

    Args:
        dt: k/omega 场显式更新使用的时间步长，直接转发给
            `turb_model.update_fields`。调用方（fr_solver/step.py）
            按 scheme 传入不同的量：稳态加速模式（SSP-RK/IMEX）传
            逐 SP 的局部 CFL 步长数组 dt_local（形状 (n_cells, n_sps)，
            与 dk_total/domega_total 广播兼容），DUAL_TIME 模式传标量
            物理 dt。之前这里统一收到的是原始物理 dt 标量，未经
            cfl.py 的阶数/粘性/几何刚性收紧，真实复现过在合成 Couette
            +SST 算例与 cube_demo 生产网格上都会让 omega 场显式积分
            失稳（一步内放大几十倍，Order Continuation 升阶后几步内
            发散至 inf/NaN）——修复见 step.py::step 文档。
    """
    if solver.turb_model is None:
        return None

    # 更新湍流产项渐变因子（每步调用，production_factor 从 0 渐增到 1）
    _update_production_ramp(solver)

    Q = solver.state.Q
    # 真实 bug 修复（2026-09-03，cube_demo 791,492 单元真实网格 Order
    # Continuation P0->P1 跨阶后延迟发散排查发现）：此前这里对*守恒*变量
    # U 求梯度、直接切片 [1:4] 当速度梯度用——U[...,1:4] 是动量
    # (rho*u,rho*v,rho*w)，grad(rho*u) != rho*grad(u)，除非密度梯度处处
    # 为零。低马赫数流场里密度接近均匀，这个误差通常小到不可见，一旦
    # 出现哪怕很小的局部密度扰动（真实复现：Order Continuation 插值截断
    # 误差），这里算出的"应变率"就会混入一个虚假的 u_i*grad(rho)/rho
    # 分量，經 SST 产生项(P_k~nu_t*S^2)反馈进涡粘系数，涡粘再反馈进动量
    # 残差放大速度/密度扰动——形成真实的正反馈失稳（真实网格上表现为
    # P1 阶段前~20步几乎不动、随后 100 步内速度峰值从 50 m/s 涨到 600+
    # m/s，k/omega 双双撞上安全上限）。同一代码库里
    # `fr_residual/viscous_flux.py` 的主残差路径一直是对的（先
    # conserved_to_primitive 转 Q 再求梯度、再切片），这里改成同一模式。
    grad_vel = _compute_gradients_generic(Q[:, :, 1:4], solver.ops, solver.mesh)

    d_wall = solver.wall_distance
    if d_wall is not None:
        expected_shape = (solver.state.n_cells, solver.state.n_sps)
        if d_wall.shape != expected_shape:
            logger.warning(
                f"Wall distance shape mismatch: expected {expected_shape}, got {d_wall.shape}. "
                f"Rescaling to match current state..."
            )
            if d_wall.ndim == 2:
                mean_d = np.mean(d_wall, axis=1, keepdims=True)
                d_wall = np.tile(mean_d, (1, solver.state.n_sps))
                solver.wall_distance = d_wall
            else:
                raise RuntimeError(f"Cannot rescale wall distance from shape {d_wall.shape}")

    if d_wall is None:
        if solver.turb_model_name in ["SST", "DDES", "IDDES", "WMLES", "LES"]:
            raise RuntimeError(
                f"Wall distance field not computed for turbulence model '{solver.turb_model_name}'. "
                f"Please call compute_wall_distance_field() before solving, or ensure wall distance "
                f"is provided during solver initialization. Industrial-grade calculation requires "
                f"accurate wall distance, not simplified estimates."
            )
        else:
            n_cells, n_sps = solver.state.U.shape[:2]
            volumes = solver._get_cell_volumes()
            h_char = np.power(np.abs(volumes), 1.0 / 3.0)
            d_wall = np.tile(h_char[:, np.newaxis], (1, n_sps))
            logger.warning(f"Using characteristic length scale as wall distance estimate")

    mu = getattr(solver, 'mu_molecular', 1.8e-5)  # 分子粘度（k/omega方程自身扩散系数用分子粘度，与平均流粘性应力
    # 张量所用的有效粘度[core/fr_solver.py::_get_turbulent_viscosity_field]是两个不同量）

    grad_k = None
    grad_omega = None
    if solver.turb_model_name in ["SST", "DDES", "IDDES"]:
        k_expanded = solver.turb_model.k_field[:, :, np.newaxis]
        omega_expanded = solver.turb_model.omega_field[:, :, np.newaxis]

        from autoflowcfd.core.fr_residual.viscous import compute_scalar_gradient

        grad_k = compute_scalar_gradient(k_expanded, solver.ops, solver.mesh)
        grad_omega = compute_scalar_gradient(omega_expanded, solver.ops, solver.mesh)

        # 正性保持检查：防止梯度过大导致负值（工业计算的梯度限幅处理）
        #
        # errstate 包裹（真实 bug，2026-08-22，用户直接在真实网格 P0->P1
        # 转阶后的输出里看到这条警告发现）：跟 turbulence/transport.py
        # 同一处（见该文件对应注释的完整原理）完全同一个失效模式——退化
        # 单元上理论为常数的 k/omega 场求梯度，度量比值 adj(J)/det(J) 把
        # 浮点噪声放大到 >1e150，np.linalg.norm 内部 x*x 先于下面的裁剪
        # 逻辑溢出到 inf。2026-08-21 修 transport.py 那份独立副本时，这里
        # （同一份裁剪逻辑最早的位置，transport.py 的注释原文引用的正是
        # 本函数）被遗漏了——两处形状不同（这里是 (n_cells,n_sps)，
        # transport.py 是 (n_cells,n_sps,3)->(...,)但语义一致）但都是同一
        # 组 np.linalg.norm+裁剪代码，只补了一处。inf 输入不影响下面裁剪
        # 结果的正确性（inf>max_grad_mag 恒真，scale=max_grad_mag/inf=0，
        # 裁剪趋于 0 不是 nan），errstate 只是抑制警告噪音。
        with np.errstate(over='ignore', invalid='ignore'):
            max_grad_mag = 1e6
            grad_k_mag = np.linalg.norm(grad_k, axis=-1)
            grad_omega_mag = np.linalg.norm(grad_omega, axis=-1)

            if np.any(grad_k_mag > max_grad_mag):
                scale_k = max_grad_mag / np.maximum(grad_k_mag, 1e-10)
                grad_k *= np.clip(scale_k[:, :, np.newaxis], 0, 1)

            if np.any(grad_omega_mag > max_grad_mag):
                scale_omega = max_grad_mag / np.maximum(grad_omega_mag, 1e-10)
                grad_omega *= np.clip(scale_omega[:, :, np.newaxis], 0, 1)

    # DDES 的有效长度尺度 (sst_model.des_length_scale) 依赖涡粘 nu_t，而
    # nu_t 只在 compute_source_terms 内部才会被重新计算（sst_model.nu_t 是
    # 上一次调用留下的缓存），两者互相依赖对方的输出，天然只能"慢一拍"：
    # apply_to_sst_model 必须排在 compute_source_terms 之后，用这一步刚算
    # 出的 nu_t 算出 des_length_scale，供*下一步*使用——这是有意为之的
    # 近似（同阶数内连续迭代时物理上完全合理），不是疏忽，因此调用顺序
    # 本身不能颠倒（真实网格已验证：颠倒后 nu_t 变成读取上一步的旧维度
    # 缓存，问题只是从 des_length_scale 转移到 nu_t，没有解决）。
    #
    # 真正需要处理的是"跨阶数切换"这一刻：des_length_scale 是按上一个阶数
    # 的 SPs 维度算出的，阶数切换后与已经正确插值过的 k_field 形状不匹配。
    # 修复见 order_continuation.py：阶数切换时显式清空 des_length_scale，
    # 让切换后的第一步自动退回标准 RANS 耗散项（不依赖过期维度的缓存），
    # 而不是在这里颠倒调用顺序。
    Sk, S_omega = solver.turb_model.compute_source_terms(Q, grad_vel, d_wall, mu, grad_k=grad_k, grad_omega=grad_omega)

    if solver.ddes_model is not None:
        rho = Q[:, :, 0]
        nu_field = mu / np.maximum(rho, 1e-10)
        if isinstance(solver.ddes_model, IDDESModel):
            # IDDES 的 Δ/f_B/f_e 公式需要逐单元 h_max/h_wn（见
            # init_turbulence_models 里 IDDES 分支的一次性缓存），与
            # DDESModel 基类只需要 cell_volumes 的 cube_root(V) 网格
            # 尺度公式结构不同，走独立的 apply_to_sst_model_iddes。
            solver.ddes_model.apply_to_sst_model_iddes(
                solver.turb_model, d_wall, solver._iddes_h_max, solver._iddes_h_wn,
                nu_field, grad_vel,
            )
        else:
            cell_volumes = solver._get_cell_volumes()
            # h_max（2026-09-02）：`init_turbulence_models` 的 DDES 分支
            # 现在也会设置 `solver._iddes_h_max`（与 IDDES 同一处几何量、
            # 同一个一次性缓存，见该分支文档）——传入后 apply_to_sst_
            # model 改用各向异性感知的 max_edge 网格尺度，不再是
            # cube_root(V)。`getattr` 兜底：极少数不经过 init_
            # turbulence_models（例如脱离 solver 直接构造 DDESModel 的
            # 测试场景）没有这个属性时，退化为 cube_root，不报错。
            solver.ddes_model.apply_to_sst_model(
                solver.turb_model, d_wall, cell_volumes, nu_field, grad_vel,
                h_max=getattr(solver, '_iddes_h_max', None),
            )

    # Sk/S_omega 是 compute_source_terms 按标准 SST 公式算出的 rho*k、
    # rho*omega 方程源项（P_k/D_k/P_omega/D_omega/CD_omega 都显式带 rho
    # 因子），但 turb_model.k_field/omega_field 存的是 k、omega 本身
    # （不是 rho*k/rho*omega，初值 1e-6/1.0 也是 k/omega 量级而非 rho*k/
    # rho*omega 量级）——直接 self.k_field += dt*Sk 会缺一个 1/rho，
    # 量纲不对。这里换算成 dk/dt ≈ Sk/rho（对缓变 rho 的标准近似：
    # d(rho*k)/dt = rho*dk/dt + k*drho/dt ≈ rho*dk/dt）再传给
    # update_fields。
    rho = Q[:, :, 0]
    dk_dt = Sk / np.maximum(rho, 1e-10)
    domega_dt = S_omega / np.maximum(rho, 1e-10)

    # 完整输运项（对流+扩散）：对 SST/DDES 模型计算 k/omega 的 FR 空间输运
    # 残差，使 k/omega 不再仅是逐点 ODE 源项弛豫，而是真正随流场对流、
    # 跨单元扩散。见 core/turbulence_transport.py 模块文档。
    transport_k = None
    transport_omega = None
    if solver.turb_model_name in ["SST", "DDES", "IDDES"]:
        from autoflowcfd.core.turbulence.transport import compute_turbulence_transport_residual
        # grad_vel 复用上面已经为 compute_source_terms 算过的同一份值
        # （同一个 solver.state.U，两处之间没有任何修改），避免
        # compute_physical_gradient 这个已知热点被重复调用——见
        # compute_turbulence_transport_residual 参数文档的性能说明。
        #
        # grad_k/grad_omega 刻意不复用、仍让本函数内部重新计算：上面
        # 那两份在传给 compute_source_terms 之前经过了一次条件触发的
        # 梯度幅值裁剪（grad_k_mag/grad_omega_mag > 1e6 时 *= 缩放，
        # 见上方"正性保持检查"），而 transport 内部的 CD_kw/F1
        # 计算历来用的是未裁剪的原始梯度——两者在裁剪实际触发的
        # （罕见）情形下不是同一个数值，复用会在那个边界情形下悄悄
        # 改变 transport 的 F1/CD_kw 取值，不是纯粹的性能优化。这里
        # 没有把握判断"两处都用裁剪后的值"在物理上是否更对，宁可
        # 保持这部分原有行为不变，只拿 grad_vel 这个确定安全（两处
        # 之间毫无改动、任何情形下都是同一个数值）的部分。
        #
        # 之前这里 `try/except Exception` 把任何失败（包括真正的编程
        # 错误——形状不匹配、numba 编译失败等）都静默降级为"仅源项
        # 更新"，只打一条 warning，不中断求解——与本项目在别处反复强调
        # 的"不允许静默地什么都不做"（见 boundary/fr_ghost_state.py::
        # BoundaryGhostStateProvider 文档）、"必须先查清原因，不能静默
        # 截断/忽略"（见 face_flux_points_merge.py 文档）等原则相悖：
        # 真实 bug 会被这个 except 吞掉，求解器带着一个悄悄退化、外部
        # 毫无察觉的湍流模型继续跑完整个仿真。真实复现过的输运计算失败
        # 目前没有已知的"预期内、可安全忽略"的情形，故不再兜底捕获，
        # 让真正的错误照常抛出、中断求解。
        # grad_k/grad_omega 同样复用（#7 内存修复，2026-08-28，
        # cube_demo 79万单元 P2+DDES 首次真实 CLI 冒烟测试触发 OOM
        # 崩溃后追查发现）：本函数上面几行刚为 compute_source_terms
        # 算好、裁剪过的同一份 grad_k/grad_omega，此前这里只传了
        # grad_vel、没有一并传 grad_k/grad_omega——
        # compute_turbulence_transport_residual 本身早就支持接收这两者
        # （见该函数文档），调用方一直没有真正利用，导致内部又重新算
        # 一遍完全相同的梯度（~1GB 冗余数组，79万单元 P2 阶段）。数学上
        # 严格等价：本函数上面的裁剪是原地 `grad_k *= clip(...)`，
        # compute_turbulence_transport_residual 内部对已经满足裁剪阈值
        # 的输入重新检查同一个阈值必然是 no-op，不会改变数值结果。
        transport_k, transport_omega = compute_turbulence_transport_residual(
            solver, grad_vel=grad_vel, grad_k=grad_k, grad_omega=grad_omega,
            flat_face_override=getattr(solver, "_turbulence_flat_face_override", None),
        )

    solver.turb_model.update_fields(dt, dk_dt, domega_dt,
                                     transport_k=transport_k,
                                     transport_omega=transport_omega)

    # 真实 bug 修复（2026-09-12，cube_demo 791,492 单元真实网格 P1 直连
    # 长程测试发现）：k/omega 场同样需要与平均流一致的模态滤波，见
    # fr_solver/filter.py::filter_scalar_field 完整推导——此前"湍流走
    # 单步显式更新、不经过多级 RK 因此不会积累混叠"的排除理由已被真实
    # 数据证伪（P1 独立发散，omega 8.6% 单元逼近安全上限，定位到具体
    # 单元内部相邻解点间出现数量级跳变，外插到面后被上风格式放大成
    # 巨大虚假对流残差）。用与平均流完全同一套 filter_prism/filter_tet
    # 矩阵，P0（n_sps=1）下矩阵退化为单位矩阵，天然是无操作。
    if solver.mesh.n_sps_per_cell > 1:
        from autoflowcfd.core.fr_solver.filter import filter_scalar_field
        n_prism = solver.mesh.n_prism_cells
        solver.turb_model.k_field = filter_scalar_field(
            solver.turb_model.k_field, n_prism, solver.ops.filter_prism, solver.ops.filter_tet,
        )
        solver.turb_model.omega_field = filter_scalar_field(
            solver.turb_model.omega_field, n_prism, solver.ops.filter_prism, solver.ops.filter_tet,
        )
        # 滤波可能把场值推到正性下限以下（滤波器系数含负权重，理论上
        # 可能），滤波后必须重新过一遍正性/上界限制器，不能假设滤波
        # 输出天然满足这些约束。
        solver.turb_model.apply_positivity_limiter()

    # 真实 bug 修复（2026-09-04）：omega 壁面 Wilcox 解析值只在对流项
    # （近壁趋于零，因为无滑移）生效，扩散项（近壁 omega 动力学的主导
    # 机制）此前完全没有把这个约束传递进去——见 transport.py::
    # enforce_omega_wall_relaxation 文档，这是 grad_vel 修复后长程复现
    # 里仍持续发散的第二个独立根因（边界层单元 omega 长期不受约束地
    # 衰减到下界，经 nu_t 近零分母奇点放大湍流粘性比，持续向平均流
    # 注入过量粘性应力）。
    #
    # 2026-09-05 曾尝试把下面这个松弛改成按 dt/d1/扩散系数物理推导的
    # "点隐式"动态系数（数学上无条件稳定），真实网格验证证伪（细网格
    # 近壁单元动态系数天然趋近 1，几步内把 k_mean 从 38 打到 0.17）
    # 已完整撤销，见 enforce_omega_wall_relaxation 文档。**当前实现是
    # 固定 relax=0.5，`dt` 只是为了不破坏调用方签名而保留的未使用参数
    # ——不要被这行调用误导，真正的行为以被调用函数的文档为准。**
    if solver.turb_model_name in ["SST", "DDES", "IDDES"]:
        from autoflowcfd.core.turbulence.transport import enforce_omega_wall_relaxation
        enforce_omega_wall_relaxation(
            solver, dt, flat_face_override=getattr(solver, "_turbulence_flat_face_override", None),
        )

    return (Sk, S_omega)


def apply_turbulence_corrections(solver) -> None:
    """应用湍流模型的修正（SGS 涡粘系数）。

    WMLES 壁面剪应力**不**在这里施加，见
    FRSolver.compute_viscous_residual()/apply_turbulence_corrections()
    文档（T-05 修复：必须在残差组装阶段生效，这里在 step() 中排在状态
    更新之后，为时已晚）。
    """
    if solver.sgs_model is not None:
        # 真实 bug 修复（2026-09-03）：同 compute_turbulence_source 里的
        # grad_vel 修复，理由见该函数文档——LES/WMLES 的 SGS 涡粘同样
        # 不能用动量梯度冒充速度梯度。
        grad_u = _compute_gradients_generic(solver.state.Q[:, :, 1:4], solver.ops, solver.mesh)
        delta = solver._get_grid_scale()
        nu_t = solver.sgs_model.compute_eddy_viscosity(grad_u, delta)

        if hasattr(solver.turb_model, "nu_t"):
            solver.turb_model.nu_t += nu_t
            logger.debug(f"SGS eddy viscosity added to turbulence model: mean={nu_t.mean():.6e}")
        else:
            solver.sgs_model.nu_t = nu_t
            logger.debug(f"SGS eddy viscosity computed: mean={nu_t.mean():.6e}, max={nu_t.max():.6e}")


def get_turbulent_viscosity_field(solver) -> Optional[np.ndarray]:
    """汇总当前激活的湍流模型给出的动力涡粘度场 mu_t = rho * nu_t。"""
    rho = solver.state.Q[:, :, 0]
    mu_t_total = None

    if solver.turb_model is not None and hasattr(solver.turb_model, "nu_t"):
        mu_t_total = rho * solver.turb_model.nu_t
    if solver.sgs_model is not None and hasattr(solver.sgs_model, "nu_t") and solver.sgs_model.nu_t is not None:
        sgs_contrib = rho * solver.sgs_model.nu_t
        mu_t_total = sgs_contrib if mu_t_total is None else mu_t_total + sgs_contrib

    return mu_t_total
