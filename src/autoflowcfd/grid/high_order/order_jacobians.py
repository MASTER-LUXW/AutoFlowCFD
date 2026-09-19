"""AutoFlowCFD V2.0 - 按阶数计算 Jacobian 的函数族。

从 `high_order_mesh_order.py`（原 561 行）拆出（2026-09-19，项目"单
文件不超 500 行"规范）：那边留的是参考点集生成、`build_order_geometry`
总装与 Order Continuation 的按阶数几何缓存；这里是"在给定参考点集上
算全部单元 Jacobian"这一族（坍缩/原生 x 四面体/棱柱，以及两段拼接与
细点度量的恒定性校验）。纯搬家，逻辑未改。
"""

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ..curved_mapping.curved_mapping import (
    CurvedMapping,
    map_prism_to_physical,
    map_tet_to_physical,
)

if TYPE_CHECKING:
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


def compute_jacobians_at_ref_points(
    mesh: "HighOrderMesh", mapper: CurvedMapping, ref_pts: np.ndarray, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """在给定的参考点集 ref_pts 上，对全部单元算一遍精确 Jacobian
    （逐单元循环，共用给同一 CurvedMapping 实例）——从 build_order_geometry
    拆出来，供体积项去混叠（超过-积分）在 FINE 参考点集上复用
    同一套逻辑，而不是另写一份（CurvedMapping.compute_jacobian 对
    tet/prism 走解析精确公式，不依赖构造时传的 顺序，见该类文档，
    因此同一个 mapper 实例可以安全地在不同参考点集上反复调用）。
    """
    from autoflowcfd.core.fr_operators.troubled_cell import compute_scaled_jacobian_quality

    n_prisms = mesh.n_prism_cells
    n_tets = len(mesh._fixed_tet_conn) if mesh._fixed_tet_conn is not None else 0
    n_total = n_prisms + n_tets
    if n_total == 0:
        return None

    # 预分配直写（2026-08-26，P2 专项 OOM 修复）：原实现先把逐单元的
    # (n_ref,)/(n_ref,3,3) 小数组累积进三个 Python 列表（79 万单元→79 万个
    # 独立小数组，仅对象开销就上百 MB），再 np.concatenate——拼接期间小数组
    # 列表与完整拼接副本同时驻留，峰值直接翻倍；P2 过积分点集（64 pts/单元）
    # 下仅 inv_jacs 就 3.6GB/份→峰值 ~7.3GB，叠加尚未清理的旧阶数几何缓存与
    # 求解器状态，79 万单元生产网格 P1→P2 切换时分配失败崩溃（两次独立复现）。
    # 改为预分配单块大数组逐单元就地写入：峰值降为结果单份 + 单元级瞬态，
    # 数值与原实现逐位一致（写入顺序/拼接顺序都是 prism 在前、tet 在后）。
    n_ref = ref_pts.shape[0]
    all_dets = np.empty(n_total * n_ref)
    all_inv_jacs = np.empty((n_total * n_ref, 3, 3))
    all_scaled_quality = np.empty(n_total * n_ref) if want_scaled_quality else None

    def _fill_cell(i, cell_nodes, cell_type):
        phys_pts = (map_prism_to_physical if cell_type == "prism"
                    else map_tet_to_physical)(ref_pts, cell_nodes)
        jac_data = mapper.compute_jacobian(
            phys_pts, cell_id=i, cell_type=cell_type, cell_nodes=cell_nodes, ref_cube_sps=ref_pts
        )
        lo, hi = i * n_ref, (i + 1) * n_ref
        all_dets[lo:hi] = jac_data["det_jacs"].ravel()
        all_inv_jacs[lo:hi] = jac_data["inv_jacs"].reshape(-1, 3, 3)
        if want_scaled_quality:
            all_scaled_quality[lo:hi] = compute_scaled_jacobian_quality(
                jac_data["jacobians"], jac_data["det_jacs"]
            ).ravel()

    if mesh._fixed_prism_conn is not None and n_prisms > 0:
        for i in range(n_prisms):
            _fill_cell(i, mesh._node_coords[mesh._fixed_prism_conn[i]], "prism")

    if mesh._fixed_tet_conn is not None and n_tets > 0:
        for i in range(n_tets):
            _fill_cell(n_prisms + i, mesh._node_coords[mesh._fixed_tet_conn[i]], "tet")

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result


def compute_native_tet_jacobians(
    mesh: "HighOrderMesh", order: int, n_sps_per_cell: int, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """native 四面体（路径C）版本的 Jacobian 构造——见 Part8 文档"一、
    核心不变量：零填充块对角"第3点：直边单元 Jacobian 是每单元一个
    常数（`compute_native_tet_jacobian`，不依赖参考点位置），不像坍缩
    坐标那样需要在 `ref_cube_sps` 上逐点求值；这个常数原样广播填满
    该单元全部 `n_sps_per_cell` 槽位（真实 `n_native` 行与填充行一视
    同仁，因为它们对应同一个 det_j——这也是本设计"填充行必须有限"这个
    强制要求最容易满足的一种情形：native 直边单元根本没有"填充行专属
    的合理值"这个问题，复用同一个真实常数即可，不需要另外编造占位值）。

    `scaled_quality`（诊断量）用与坍缩坐标分支
    `troubled_cell.py::compute_scaled_jacobian_quality` 完全相同的定义
    （`det_j/prod(col_norms(J))`）手算——直边单元只有一个 J，不需要
    调用那个批量/逐点版本。
    """
    from autoflowcfd.fr.native_tet.basis import compute_native_tet_jacobian

    n_tets = len(mesh._fixed_tet_conn) if mesh._fixed_tet_conn is not None else 0
    if n_tets == 0:
        return None

    all_dets = np.empty(n_tets * n_sps_per_cell)
    all_inv_jacs = np.empty((n_tets * n_sps_per_cell, 3, 3))
    all_scaled_quality = np.empty(n_tets * n_sps_per_cell) if want_scaled_quality else None

    for i in range(n_tets):
        cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i]]
        det_j, adj_j = compute_native_tet_jacobian(cell_nodes)
        inv_j = adj_j / det_j
        lo, hi = i * n_sps_per_cell, (i + 1) * n_sps_per_cell
        all_dets[lo:hi] = det_j
        all_inv_jacs[lo:hi] = inv_j
        if want_scaled_quality:
            p0, p1, p2, p3 = cell_nodes
            J = 0.5 * np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            col_norms = np.linalg.norm(J, axis=0)
            quality = det_j / max(np.prod(col_norms), 1e-300)
            all_scaled_quality[lo:hi] = quality

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result


def _verify_tet_fine_metric_is_cellwise_constant(
    tet_part: Optional[Dict[str, np.ndarray]], n_sps_per_cell_fine: int
) -> None:
    """校验四面体细点度量在单元内逐槽位完全相同。

    `get_overintegration_context` 的四面体段只取第 0 列再广播到该段自己的
    `n_fine_tet` 宽度——这让四面体的过积分阶数不再受棱柱布局宽度约束
    （P3 因此能取到理想的 over_order=6，去混叠误差 3.37e-3 -> 4.80e-6）。
    这个等价性依赖"直边单元 Jacobian 不依赖参考点位置"，也就是
    `compute_native_tet_jacobians` 的常数广播。

    判据用**逐位相同**而不是容差：那些值本来就是同一次赋值广播出来的，
    任何差异都说明构造方式变了（例如引入曲边四面体后改成逐点求值），
    那时必须显式改掉广播路径，而不是让它悄悄给出"第 0 个细点的度量"。
    """
    if tet_part is None:
        return
    for key in ("det_jacs", "inv_jacs"):
        arr = tet_part.get(key)
        if arr is None:
            continue
        per_cell = arr.reshape((-1, n_sps_per_cell_fine) + arr.shape[1:])
        if per_cell.shape[0] == 0:
            continue
        ref_col = per_cell[:, :1]
        if not np.array_equal(per_cell, np.broadcast_to(ref_col, per_cell.shape)):
            bad = int((per_cell != np.broadcast_to(ref_col, per_cell.shape)).any(
                axis=tuple(range(1, per_cell.ndim))).sum())
            raise ValueError(
                f"四面体细点度量 '{key}' 在 {bad} 个单元内部不是逐槽位常数"
                f"——`get_overintegration_context` 的四面体段取第 0 列广播的"
                f"前提不再成立（典型原因：引入了曲边四面体、Jacobian 改成"
                f"逐点求值）。必须改掉那条广播，不能让它静默只用第 0 个"
                f"细点的度量。"
            )


def _combine_prism_and_tet_jacobians(
    prism_part: Optional[Dict[str, np.ndarray]], tet_part: Optional[Dict[str, np.ndarray]]
) -> Optional[Dict[str, np.ndarray]]:
    """把只算了棱柱部分（native 四面体分支专用，见 `build_order_geometry`）
    的 jacobians 与单独算好的 native 四面体部分按 prism-在前/tet-在后
    的既有顺序拼接——与 `compute_jacobians_at_ref_points` 一次性算全部
    单元时的拼接顺序保持一致，下游（`mesh.jacobians["det_jacs"].reshape
    (n_cells, n_sps)` 等）不需要关心这里是分两段算的。"""
    if prism_part is None:
        return tet_part
    if tet_part is None:
        return prism_part
    result = {}
    for key in prism_part:
        result[key] = np.concatenate([prism_part[key], tet_part[key]])
    return result


def _compute_prism_only_jacobians(
    mesh: "HighOrderMesh", mapper: CurvedMapping, ref_pts: np.ndarray, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """`compute_jacobians_at_ref_points` 的棱柱专属子集——四面体部分走
    `compute_native_tet_jacobians`（native 单纯形基），两部分分别算好
    再按 prism-在前/tet-在后拼接，见 `build_order_geometry`。逻辑与
    `compute_jacobians_at_ref_points` 的棱柱分支逐行一致，只是提取出来
    单独调用，不是重新实现一遍。
    """
    from autoflowcfd.core.fr_operators.troubled_cell import compute_scaled_jacobian_quality

    n_prisms = mesh.n_prism_cells
    if mesh._fixed_prism_conn is None or n_prisms == 0:
        return None
    n_ref = ref_pts.shape[0]
    all_dets = np.empty(n_prisms * n_ref)
    all_inv_jacs = np.empty((n_prisms * n_ref, 3, 3))
    all_scaled_quality = np.empty(n_prisms * n_ref) if want_scaled_quality else None

    for i in range(n_prisms):
        cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
        phys_pts = map_prism_to_physical(ref_pts, cell_nodes)
        jac_data = mapper.compute_jacobian(
            phys_pts, cell_id=i, cell_type="prism", cell_nodes=cell_nodes, ref_cube_sps=ref_pts
        )
        lo, hi = i * n_ref, (i + 1) * n_ref
        all_dets[lo:hi] = jac_data["det_jacs"].ravel()
        all_inv_jacs[lo:hi] = jac_data["inv_jacs"].reshape(-1, 3, 3)
        if want_scaled_quality:
            all_scaled_quality[lo:hi] = compute_scaled_jacobian_quality(
                jac_data["jacobians"], jac_data["det_jacs"]
            ).ravel()

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result


def compute_native_prism_jacobians(
    mesh: "HighOrderMesh", ref_native: np.ndarray, n_sps_per_cell: int,
    want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """原生棱柱基（`AFCFD_PRISM_BASIS=native`）版本的 Jacobian 构造。

    与四面体那条（`compute_native_tet_jacobians`）的**关键差别**：直边
    四面体的 Jacobian 是逐单元常数，而棱柱即便直边也一般**随点变化**
    （只有顶面是底面纯平移的右棱柱才恒定），所以这里必须在原生节点上
    **逐点**求值，不能算一个常数广播。解析求导用
    `native_prism_basis.native_prism_exact_jacobian`（不经过谱微分矩阵，
    理由与坍缩档的 `prism_exact_jacobian` 相同：谱微分会把算子舍入带进
    度量，而度量误差直接进自由流保持性）。

    填充槽位（`[n_native, n_sps_per_cell)`）复制真实 SP #0 的度量 ——
    与 `build_order_geometry` 里 `sps_coords` 的填充约定一致（有限、
    物理上合法的占位值）。**不能留 0**：`det_jacs` 是除数。

    `scaled_quality`（诊断量）调用与坍缩分支**同一个**
    `troubled_cell.py::compute_scaled_jacobian_quality`，不另写一份定义。

    Args:
        ref_native: `(n_native, 3)` 原生参考棱柱坐标 `(r,s,t)`。**由调用
            方传入而不是这里按 order 现算**：coarse 几何那一处已经为
            `sps_coords` 生成过同一批节点（不重算），而体积项去混叠的
            FINE 几何要的是 `over_order` 的节点集——两者只差参考点集，
            度量公式完全相同，没有理由写两个函数。
        n_sps_per_cell: 该段的布局宽度（coarse 是 `(order+1)^3`，fine 是
            `overintegration_order.prism_n_fine(over_order)`，原生档下后者
            恰好等于 `ref_native.shape[0]`、不产生填充槽位）。
    """
    from autoflowcfd.core.fr_operators.troubled_cell import compute_scaled_jacobian_quality
    from autoflowcfd.fr.native_prism.basis import native_prism_exact_jacobian

    n_prisms = mesh.n_prism_cells
    if mesh._fixed_prism_conn is None or n_prisms == 0:
        return None

    ref_native = np.asarray(ref_native, dtype=np.float64)
    n_native = ref_native.shape[0]
    if n_native > n_sps_per_cell:
        raise ValueError(
            f"原生棱柱参考点数 {n_native} 超过布局宽度 {n_sps_per_cell}"
            f"——零填充设计要求前者不多于后者，出现相反情形说明上游"
            f"传错了点集/宽度，不应静默截断")

    all_dets = np.empty(n_prisms * n_sps_per_cell)
    all_inv_jacs = np.empty((n_prisms * n_sps_per_cell, 3, 3))
    all_scaled_quality = (np.empty(n_prisms * n_sps_per_cell)
                          if want_scaled_quality else None)

    for i in range(n_prisms):
        cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
        jac = native_prism_exact_jacobian(ref_native, cell_nodes)   # (n_native,3,3)
        det = np.linalg.det(jac)
        inv = np.linalg.inv(jac)
        lo = i * n_sps_per_cell
        all_dets[lo:lo + n_native] = det
        all_inv_jacs[lo:lo + n_native] = inv
        # 填充槽位：复制真实 SP #0
        all_dets[lo + n_native:lo + n_sps_per_cell] = det[0]
        all_inv_jacs[lo + n_native:lo + n_sps_per_cell] = inv[0]
        if want_scaled_quality:
            q = compute_scaled_jacobian_quality(jac, det)
            all_scaled_quality[lo:lo + n_native] = q
            all_scaled_quality[lo + n_native:lo + n_sps_per_cell] = q[0]

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result
