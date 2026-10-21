"""AutoFlowCFD V2.0 - 按阶数计算 Jacobian 的函数族。

`high_order_mesh_order.py` 是参考点集生成、`build_order_geometry` 总装与 Order Continuation 的按阶数
几何缓存；这里是"在给定参考点集上
算全部单元 Jacobian"这一族（原生四面体/棱柱、两段拼接与细点度量的恒定性校验）。
"""

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np


if TYPE_CHECKING:
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


def _require_positive_det(det_per_cell: np.ndarray, cell_offset: int, cell_kind: str) -> None:
    """全部真实求值点上的 Jacobian 行列式必须为正，否则抛 MeshDistortionError（列出最坏单元）。

    Args:
        det_per_cell: (n_cells, n_points) 真实求值点上的行列式（不含填充槽位）
        cell_offset: 本段单元在全局单元编号中的起点（棱柱段 0，四面体段 n_prism）
        cell_kind: 报错用的单元类型名
    """
    from autoflowcfd.grid.curved_mapping.curved_mapping import MeshDistortionError

    min_per_cell = det_per_cell.min(axis=1)
    bad = np.flatnonzero(~(min_per_cell > 0.0))
    if bad.size == 0:
        return
    worst = bad[np.argsort(min_per_cell[bad])[:10]]
    listing = ", ".join(f"{cell_offset + int(c)}({min_per_cell[c]:.3e})" for c in worst)
    raise MeshDistortionError(
        f"{bad.size} 个{cell_kind}单元在求值点上的 Jacobian 行列式非正（翻转或退化），"
        f"最坏单元（全局编号(最小 det)）：{listing}。须在网格生成/修复阶段处理这些单元。")


def compute_native_tet_jacobians(
    mesh: "HighOrderMesh", order: int, n_sps_per_cell: int, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """原生四面体的 Jacobian 构造：直边单元 Jacobian 是每单元一个常数（`compute_native_tet_jacobian`，不依赖参考
    点位置），原样广播填满该单元全部 `n_sps_per_cell` 槽位（真实行与零填充槽位同值，填充行因此天然有限）。

    `scaled_quality`（诊断量）与棱柱同一定义 `det_j/prod(col_norms(J))`
    （`troubled_cell.py::compute_scaled_jacobian_quality`）；直边单元只有一个 J，直接手算。
    """
    from autoflowcfd.fr.native_tet.basis import compute_native_tet_jacobian

    n_tets = len(mesh._fixed_tet_conn) if mesh._fixed_tet_conn is not None else 0
    if n_tets == 0:
        return None

    nodes = mesh._node_coords[mesh._fixed_tet_conn]                       # (n_tets,4,3)
    J_all = 0.5 * np.stack([nodes[:, 1] - nodes[:, 0], nodes[:, 2] - nodes[:, 0],
                            nodes[:, 3] - nodes[:, 0]], axis=2)            # 列 = dphys/d(r,s,t)
    _require_positive_det(np.linalg.det(J_all)[:, None], mesh.n_prism_cells, "四面体")

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


def build_fine_metrics(
    prism_part: Optional[Dict[str, np.ndarray]], tet_part: Optional[Dict[str, np.ndarray]],
    n_fine_prism: int, n_prisms: int, n_tets: int, cell_slots: np.ndarray,
) -> Dict[str, np.ndarray]:
    """过积分细点度量（`mesh.jacobians_fine`），按单元类型分段存储：

        prism_det (n_prism, n_fine)       prism_inv (n_prism, n_fine, 3, 3)
        tet_det   (n_tet,)                tet_inv   (n_tet, 3, 3)

    棱柱的度量随细点变化（只有顶面是底面纯平移的右棱柱才恒定），必须逐点存；
    直边四面体的 Jacobian 逐单元为常数（`compute_native_tet_jacobian` 与参考点
    无关），只存一份，消费方（`volume_contract.get_overintegration_context`、
    GPU `gpu_overintegration`）广播到自己的细点数。

    此前两段共用一个按棱柱细点数展开的 `(n_cells*n_fine,)` 数组：四面体段把同一
    个常数写满 n_fine 个槽位。plate_demo（15.9 万四面体）P3、原生棱柱 oo=6
    （n_fine=196）下这一段约 2.5 GB，实际只需要 12.7 MB；GPU 端再按
    det/inv/adj 各上传一份，显存里约 4.7 GB。

    `tet_part` 须按每单元 1 个点求值（`compute_native_tet_jacobians(mesh, ..., 1)`）。

    另存逐单元的无粘体积算子组合编号 `prism_k_combo (n_prism,)` / `tet_k_combo (n_tet,)`
    （由三角形面坍缩顶点槽位 `cell_slots` 折叠，`fr/triangle_apex.k_combo_ids`）：它与
    细点度量一样是逐单元几何量，随同一份字典在单机、分布式各路径里切片与分发。
    """
    def _prism(key, tail):
        if prism_part is None:
            return np.zeros((0, n_fine_prism) + tail)
        return np.ascontiguousarray(prism_part[key].reshape((n_prisms, n_fine_prism) + tail))

    def _tet(key, tail):
        if tet_part is None:
            return np.zeros((0,) + tail)
        arr = tet_part[key]
        if arr.shape[0] != n_tets:
            raise ValueError(
                f"四面体细点度量 '{key}' 行数 {arr.shape[0]} 不等于四面体数 {n_tets}"
                f"——须按每单元 1 个点求值")
        return np.ascontiguousarray(arr.reshape((n_tets,) + tail))

    from autoflowcfd.fr.triangle_apex import k_combo_ids

    prism_combo, tet_combo = k_combo_ids(cell_slots, n_prisms)
    if prism_combo.shape[0] != n_prisms or tet_combo.shape[0] != n_tets:
        raise ValueError(f"cell_slots 行数 {np.shape(cell_slots)[0]} 与单元数 {n_prisms}+{n_tets} 不符")
    return {
        "prism_det": _prism("det_jacs", ()),
        "prism_inv": _prism("inv_jacs", (3, 3)),
        "tet_det": _tet("det_jacs", ()),
        "tet_inv": _tet("inv_jacs", (3, 3)),
        "prism_k_combo": prism_combo,
        "tet_k_combo": tet_combo,
    }


def select_fine_metrics(fine: Dict[str, np.ndarray], cell_ids: np.ndarray,
                        n_prism_global: int) -> Dict[str, np.ndarray]:
    """按单元编号（全局空间）选出一份分段细点度量（分布式紧凑空间、分区网格用）。

    本项目的单元存储顺序是"棱柱在前、四面体在后"，过积分按本地 `n_prism` 分两段
    （`get_overintegration_context`），所以 `cell_ids` 也必须满足棱柱全部在前；
    不满足直接报错，不静默错位。
    """
    ids = np.asarray(cell_ids, dtype=np.int64)
    is_prism = ids < int(n_prism_global)
    n_p = int(is_prism.sum())
    if not (is_prism[:n_p].all() and not is_prism[n_p:].any()):
        raise ValueError("cell_ids 不满足\"棱柱在前、四面体在后\"——过积分分段会错位")
    tet_ids = ids[n_p:] - int(n_prism_global)
    return {
        "prism_det": np.ascontiguousarray(fine["prism_det"][ids[:n_p]]),
        "prism_inv": np.ascontiguousarray(fine["prism_inv"][ids[:n_p]]),
        "tet_det": np.ascontiguousarray(fine["tet_det"][tet_ids]),
        "tet_inv": np.ascontiguousarray(fine["tet_inv"][tet_ids]),
        "prism_k_combo": np.ascontiguousarray(fine["prism_k_combo"][ids[:n_p]]),
        "tet_k_combo": np.ascontiguousarray(fine["tet_k_combo"][tet_ids]),
    }


def _combine_prism_and_tet_jacobians(
    prism_part: Optional[Dict[str, np.ndarray]], tet_part: Optional[Dict[str, np.ndarray]]
) -> Optional[Dict[str, np.ndarray]]:
    """把只算了棱柱部分（native 四面体分支专用，见 `build_order_geometry`）
    的 jacobians 与单独算好的 native 四面体部分按 prism-在前/tet-在后
    的既有顺序拼接，下游（`mesh.jacobians["det_jacs"].reshape
    (n_cells, n_sps)` 等）不需要关心这里是分两段算的。"""
    if prism_part is None:
        return tet_part
    if tet_part is None:
        return prism_part
    result = {}
    for key in prism_part:
        result[key] = np.concatenate([prism_part[key], tet_part[key]])
    return result


def compute_native_prism_jacobians(
    mesh: "HighOrderMesh", ref_native: np.ndarray, n_sps_per_cell: int,
    want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """原生棱柱基版本的 Jacobian 构造。

    与四面体那条（`compute_native_tet_jacobians`）的**关键差别**：直边
    四面体的 Jacobian 是逐单元常数，而棱柱即便直边也一般**随点变化**
    （只有顶面是底面纯平移的右棱柱才恒定），所以这里必须在原生节点上
    **逐点**求值，不能算一个常数广播。解析求导用 `fr/native_prism/basis.py::native_prism_exact_jacobian`
    （不经过谱微分矩阵：谱微分会把算子舍入带进度量，而度量误差直接进自由流保持性）。

    填充槽位（`[n_native, n_sps_per_cell)`）复制真实 SP #0 的度量 ——
    与 `build_order_geometry` 里 `sps_coords` 的填充约定一致（有限、
    物理上合法的占位值）。**不能留 0**：`det_jacs` 是除数。

    `scaled_quality`（诊断量）调用 `troubled_cell.py::compute_scaled_jacobian_quality`（与四面体同一定义）。

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

    real_dets = np.empty((n_prisms, n_native))
    for i in range(n_prisms):
        cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
        jac = native_prism_exact_jacobian(ref_native, cell_nodes)   # (n_native,3,3)
        det = np.linalg.det(jac)
        real_dets[i] = det
        if not (det > 0.0).all():
            continue                      # 循环后统一报错（求逆会在奇异点先抛出不相干的异常）
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
    _require_positive_det(real_dets, 0, "棱柱")

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result
