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

from autoflowcfd.core.turbulence_sst import SSTModelFR
from autoflowcfd.core.turbulence_des import DDESModel
from autoflowcfd.core.turbulence_wmles import WMLESModel
from autoflowcfd.core.turbulence_sgs import WALEModel
from autoflowcfd.core.wall_distance import compute_wall_distance


def init_turbulence_models(solver, n_cells: int, n_sps: int) -> None:
    """初始化湍流模型（对应 FRSolver._init_turbulence_models）。"""
    if solver.turb_model_name == "SST":
        solver.turb_model = SSTModelFR(n_cells, n_sps)
        print(f"   [OK] SST k-omega model initialized")

    elif solver.turb_model_name == "DDES":
        solver.turb_model = SSTModelFR(n_cells, n_sps)
        solver.ddes_model = DDESModel()
        print(f"   [OK] DDES model initialized (based on SST)")

    elif solver.turb_model_name == "WMLES":
        solver.wmles_model = WMLESModel()
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
    if solver.turb_model_name not in ["SST", "DDES", "WMLES", "LES"]:
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
            return
        except Exception:
            pass

    solver.wall_distance = np.ones((n_cells, n_sps)) * node_distances.mean()
    logger.info(f"Wall distance field initialized (fallback): mean={solver.wall_distance.mean():.6f}")


def compute_turbulence_source(solver, dt: float) -> Optional[tuple]:
    """计算湍流模型源项（对应 FRSolver.compute_turbulence_source）。"""
    if solver.turb_model is None:
        return None

    Q = solver.state.Q
    grad_U = solver._compute_gradients()
    grad_vel = grad_U[:, :, 1:4, :]

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
        if solver.turb_model_name in ["SST", "DDES", "WMLES", "LES"]:
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

    mu = 1.8e-5  # 空气动力粘度（k/omega方程自身扩散系数用分子粘度，与平均流粘性应力
    # 张量所用的有效粘度[core/fr_solver.py::_get_turbulent_viscosity_field]是两个不同量）

    grad_k = None
    grad_omega = None
    if solver.turb_model_name in ["SST", "DDES"]:
        k_expanded = solver.turb_model.k_field[:, :, np.newaxis]
        omega_expanded = solver.turb_model.omega_field[:, :, np.newaxis]

        from autoflowcfd.core.fr_residual_viscous import compute_scalar_gradient

        grad_k = compute_scalar_gradient(k_expanded, solver.ops, solver.mesh)
        grad_omega = compute_scalar_gradient(omega_expanded, solver.ops, solver.mesh)

        # 正性保持检查：防止梯度过大导致负值（工业计算的梯度限幅处理）
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
        cell_volumes = solver._get_cell_volumes()
        solver.ddes_model.apply_to_sst_model(solver.turb_model, d_wall, cell_volumes, grad_vel)

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
    if solver.turb_model_name in ["SST", "DDES"]:
        try:
            from autoflowcfd.core.turbulence_transport import compute_turbulence_transport_residual
            transport_k, transport_omega = compute_turbulence_transport_residual(solver)
        except Exception as e:
            logger.warning(f"湍流输运项计算失败，退化为仅源项更新: {e}")

    solver.turb_model.update_fields(dt, dk_dt, domega_dt,
                                     transport_k=transport_k,
                                     transport_omega=transport_omega)

    return (Sk, S_omega)


def apply_turbulence_corrections(solver) -> None:
    """应用湍流模型的修正（SGS 涡粘系数）。

    WMLES 壁面剪应力**不**在这里施加，见
    FRSolver.compute_viscous_residual()/apply_turbulence_corrections()
    文档（T-05 修复：必须在残差组装阶段生效，这里在 step() 中排在状态
    更新之后，为时已晚）。
    """
    if solver.sgs_model is not None:
        grad_U = solver._compute_gradients()
        grad_u = grad_U[:, :, 1:4, :]
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
