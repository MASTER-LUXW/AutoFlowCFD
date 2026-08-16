"""
AutoFlowCFD V2.0 - HighOrderMesh 的阶数相关几何构建 (G-01/G-03, CL-02)

从 high_order_mesh.py 拆出来（控制单文件行数，>400 行需拆分的项目
规范）：参考点集生成、（含体积项去混叠用的 fine 网格）Jacobian
批量计算、Order Continuation 的按阶数几何缓存/切换。签名以
`mesh: HighOrderMesh` 为第一参数，HighOrderMesh 上保留同名薄委托
方法，调用方式不变。
"""

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from loguru import logger

from autoflowcfd.grid.curved_mapping import (
    CurvedMapping,
    map_prism_to_physical,
    map_tet_to_physical,
)
from autoflowcfd.fr.operators import generate_fr_operators

if TYPE_CHECKING:
    from autoflowcfd.grid.high_order_mesh import HighOrderMesh


def generate_reference_cube_sps(mesh: "HighOrderMesh", order: Optional[int] = None) -> np.ndarray:
    """生成计算立方体 [-1,1]^3 内的张量积 Gauss-Legendre SPs 坐标。

    四面体、棱柱共用同一套计算立方体坐标（Duffy 坍缩坐标的计算域）；
    单元类型差异完全体现在 curved_mapping 的物理映射函数中，这里不再
    像旧版本那样对四面体/棱柱分别生成不同的"参考点"。

    Args:
        order: 目标阶数；None 时使用 mesh.order（当前活动阶数）。
            Order Continuation 需要在切换到某个阶数*之前*为该阶数生成
            SPs，此时 mesh.order 还是旧阶数，必须显式传入。
    """
    from autoflowcfd.fr.operators import gauss_legendre

    n_points_1d = (order + 1) if order is not None else mesh.n_points_1d
    sps_1d, _ = gauss_legendre(n_points_1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def compute_jacobians_at_ref_points(
    mesh: "HighOrderMesh", mapper: CurvedMapping, ref_pts: np.ndarray, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """在给定的参考点集 ref_pts 上，对全部单元算一遍精确 Jacobian
    （逐单元循环，共用给同一 CurvedMapping 实例）——从 build_order_geometry
    拆出来，供体积项去混叠（over-integration）在 FINE 参考点集上复用
    同一套逻辑，而不是另写一份（CurvedMapping.compute_jacobian 对
    tet/prism 走解析精确公式，不依赖构造时传的 order，见该类文档，
    因此同一个 mapper 实例可以安全地在不同参考点集上反复调用）。
    """
    from autoflowcfd.core.fr_troubled_cell import compute_scaled_jacobian_quality

    n_prisms = mesh.n_prism_cells
    all_dets, all_inv_jacs, all_scaled_quality = [], [], []

    if mesh._fixed_prism_conn is not None and n_prisms > 0:
        for i in range(n_prisms):
            cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
            phys_pts = map_prism_to_physical(ref_pts, cell_nodes)
            jac_data = mapper.compute_jacobian(
                phys_pts, cell_id=i, cell_type="prism", cell_nodes=cell_nodes, ref_cube_sps=ref_pts
            )
            all_dets.append(jac_data["det_jacs"])
            all_inv_jacs.append(jac_data["inv_jacs"])
            if want_scaled_quality:
                all_scaled_quality.append(
                    compute_scaled_jacobian_quality(jac_data["jacobians"], jac_data["det_jacs"])
                )

    if mesh._fixed_tet_conn is not None and len(mesh._fixed_tet_conn) > 0:
        n_tets = len(mesh._fixed_tet_conn)
        for i in range(n_tets):
            cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i]]
            phys_pts = map_tet_to_physical(ref_pts, cell_nodes)
            jac_data = mapper.compute_jacobian(
                phys_pts, cell_id=n_prisms + i, cell_type="tet", cell_nodes=cell_nodes, ref_cube_sps=ref_pts
            )
            all_dets.append(jac_data["det_jacs"])
            all_inv_jacs.append(jac_data["inv_jacs"])
            if want_scaled_quality:
                all_scaled_quality.append(
                    compute_scaled_jacobian_quality(jac_data["jacobians"], jac_data["det_jacs"])
                )

    if not all_dets:
        return None
    result = {
        "det_jacs": np.concatenate(all_dets, axis=0),
        "inv_jacs": np.concatenate(all_inv_jacs, axis=0),
    }
    if want_scaled_quality:
        result["scaled_quality"] = np.concatenate(all_scaled_quality, axis=0)
    return result


def build_order_geometry(mesh: "HighOrderMesh", order: int) -> Dict[str, np.ndarray]:
    """在给定阶数下，从已修正朝向的 connectivity/节点坐标重新推导
    SPs 物理坐标与 Jacobian（不依赖 mesh.order/mesh.n_points_1d 的当前值，
    可在切换阶数*之前*安全调用）。

    Order Continuation（CL-02）的核心前提：FR 方法的解自由度与几何量
    必须共享同一组 SPs——只换 FR 微分算子（mesh.operators）而不重新
    推导这里的量，P0/P1 阶段的梯度/残差计算会直接用错误维度的
    Jacobian（真实网格已复现：reshape 到 27 SPs/cell 的 Jacobian 硬套
    1 SP/cell 的状态场，直接崩溃）。

    Returns:
        {"sps_coords", "jacobians", "ref_cube_sps", "jacobians_fine",
        "n_sps_per_cell_fine"}
    """
    n_points_1d = order + 1
    n_sps_per_cell = n_points_1d**3
    sps_coords = np.zeros((mesh.n_cells, n_sps_per_cell, 3))
    mapper = CurvedMapping(order)

    ref_cube_sps = generate_reference_cube_sps(mesh, order)
    n_prisms = mesh.n_prism_cells

    if mesh._fixed_prism_conn is not None and n_prisms > 0:
        for i in range(n_prisms):
            cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
            sps_coords[i] = map_prism_to_physical(ref_cube_sps, cell_nodes)

    if mesh._fixed_tet_conn is not None and len(mesh._fixed_tet_conn) > 0:
        n_tets = len(mesh._fixed_tet_conn)
        for i in range(n_tets):
            cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i]]
            sps_coords[n_prisms + i] = map_tet_to_physical(ref_cube_sps, cell_nodes)

    jacobians = compute_jacobians_at_ref_points(mesh, mapper, ref_cube_sps, want_scaled_quality=True)

    # 体积项去混叠（over-integration，V2.0 二次评审 Tier 0 #2）用的
    # 细网格几何：过积分阶数 over_order=2*order，与 fr/operators.py::
    # generate_fr_operators 里构造 overint_interp_c2f_*/overint_D_fine_*
    # 用的过积分阶数必须一致（否则 fr_residual_inviscid.py 里插值/
    # 微分/限制三个算子的形状与这里的 jacobians_fine 对不上）。
    # order==0（P0）没有意义（P0 走独立的有限体积残差路径，见
    # fr_residual_inviscid.py::compute_inviscid_residual_fr 的
    # n_points_1d==1 分支），跳过以节省内存/构建时间。
    jacobians_fine = None
    n_sps_per_cell_fine = 0
    if order >= 1:
        from autoflowcfd.fr.operators import gauss_legendre
        from autoflowcfd.fr.collapsed_basis import OVERINTEGRATION_MAX_ORDER

        over_order = min(2 * order, OVERINTEGRATION_MAX_ORDER)
        n_points_1d_fine = over_order + 1
        n_sps_per_cell_fine = n_points_1d_fine**3
        fine_1d, _ = gauss_legendre(n_points_1d_fine)
        xf, yf, zf = np.meshgrid(fine_1d, fine_1d, fine_1d, indexing="ij")
        ref_cube_sps_fine = np.column_stack([xf.ravel(), yf.ravel(), zf.ravel()])
        jacobians_fine = compute_jacobians_at_ref_points(
            mesh, mapper, ref_cube_sps_fine, want_scaled_quality=False
        )

    return {
        "sps_coords": sps_coords,
        "jacobians": jacobians,
        "ref_cube_sps": ref_cube_sps,
        "jacobians_fine": jacobians_fine,
        "n_sps_per_cell_fine": n_sps_per_cell_fine,
    }


def set_order(mesh: "HighOrderMesh", order: int) -> None:
    """切换网格当前活动的多项式阶数（Order Continuation 专用）。

    SPs 坐标、Jacobian、Flux Points 几何（含 Newton 面点位定位）全部
    随阶数重新推导——这些量不是"复用同一套再插值"就够的，FR 方法要求
    解自由度与几何在同一组 SPs/FPs 上重合。按阶数缓存：目标阶数（网格
    加载时已经构建过）与之前访问过的阶数直接复用缓存，不重复触发昂贵
    的逐面 Newton 点位定位重建。

    Args:
        order: 目标阶数
    """
    if order == mesh._active_order:
        return

    if order not in mesh._order_geometry_cache:
        geom = build_order_geometry(mesh, order)

        # 临时切到新阶数的基础几何量：build_face_flux_points /
        # precompute_cell_face_misalignment 直接读取 mesh.n_points_1d /
        # mesh.jacobians / mesh.operators，必须先落地才能调用。
        mesh.order = order
        mesh.n_points_1d = order + 1
        mesh.n_sps_per_cell = mesh.n_points_1d**3
        mesh.sps_coords = geom["sps_coords"]
        mesh.jacobians = geom["jacobians"]
        mesh._ref_cube_sps = geom["ref_cube_sps"]
        mesh.jacobians_fine = geom["jacobians_fine"]
        mesh.n_sps_per_cell_fine = geom["n_sps_per_cell_fine"]
        mesh.operators = generate_fr_operators(order)

        face_flux_points = None
        cell_face_misalignment = None
        if mesh.face_connectivity is not None:
            from autoflowcfd.fr.face_flux_points_merge import build_face_flux_points

            logger.info(f"Order continuation: building Flux Points geometry for P{order}...")
            face_flux_points = build_face_flux_points(mesh.face_connectivity, mesh)
            mesh.face_flux_points = face_flux_points
            logger.info(f"Order continuation: Flux Points geometry built for P{order}")

            if mesh.jacobians is not None:
                from autoflowcfd.core.fr_troubled_cell import precompute_cell_face_misalignment

                cell_face_misalignment = precompute_cell_face_misalignment(mesh)
                mesh.cell_face_misalignment = cell_face_misalignment

        mesh._order_geometry_cache[order] = {
            "n_points_1d": mesh.n_points_1d,
            "n_sps_per_cell": mesh.n_sps_per_cell,
            "sps_coords": mesh.sps_coords,
            "jacobians": mesh.jacobians,
            "ref_cube_sps": mesh._ref_cube_sps,
            "operators": mesh.operators,
            "face_flux_points": face_flux_points,
            "cell_face_misalignment": cell_face_misalignment,
            "jacobians_fine": mesh.jacobians_fine,
            "n_sps_per_cell_fine": mesh.n_sps_per_cell_fine,
        }

    cached = mesh._order_geometry_cache[order]
    mesh.order = order
    mesh.n_points_1d = cached["n_points_1d"]
    mesh.n_sps_per_cell = cached["n_sps_per_cell"]
    mesh.sps_coords = cached["sps_coords"]
    mesh.jacobians = cached["jacobians"]
    mesh._ref_cube_sps = cached["ref_cube_sps"]
    mesh.operators = cached["operators"]
    mesh.face_flux_points = cached["face_flux_points"]
    mesh.cell_face_misalignment = cached["cell_face_misalignment"]
    mesh.jacobians_fine = cached["jacobians_fine"]
    mesh.n_sps_per_cell_fine = cached["n_sps_per_cell_fine"]
    mesh._active_order = order
