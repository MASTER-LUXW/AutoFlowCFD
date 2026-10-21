"""FR Flux Point 精确逐点法向量 / 面积权重 / adj(J) 行（numpy 参照实现 + 法向派生）。

`true_normal`/`true_area_weight` 与两侧的 adj(J) 行在**每个 Flux Point 自己的精确参考坐标处**解析求值，而不是每个
面取一个常数：棱柱四边形侧面在物理上一般不是平面（4 个角点定义的是双线性曲面，真实网格实测最大翘曲 11.13%），
P1 起一个面上的多个 Flux Points 分布在面内不同位置，共用一个常数法向等于对非平面曲面用错误的局部切平面方向做
黎曼求解。

生产构造（`face_metrics.py`）用 numba 版 `exact_normal_kernel.compute_exact_adj_rows_fast` 求 adj 行，再由
`normals_and_weights_from_adj_rows` 派生法向与面积权重；本模块的 `compute_exact_adj_rows` /
`compute_exact_face_normals_and_weights` 是独立的 numpy 参照实现（测试用它交叉核对 numba 版）。

面编码一律是原生编码（四面体 6~9 = 6 + 被排除的局部顶点，棱柱 10~14 = 10 + 原生面序号）。原生面的 adj 行已按
外法向定向：`true_normal = adj_row / |adj_row|`，`true_area_weight = |adj_row| * w_fp`（沿 FP 求和即物理面积）。
"""

from typing import Optional, Tuple

import numpy as np

from autoflowcfd.grid.connectivity.face_connectivity import (
    NATIVE_PRISM_FACE_CODE_RANGE,
    NATIVE_TET_FACE_CODE_RANGE,
)


def _native_tet_adj_row_batched(
    excluded_vertex: int, cell_nodes: np.ndarray, n1d: int, sps_1d: np.ndarray
) -> np.ndarray:
    """native 四面体面的外向 adj(J) 行：方向是外法向，模长是该 Flux Point 参考采样坐标 (a,b) 下的局部面积
    Jacobian（`true_area_weight = mag * w_fp` 沿 FP 求和给出真实物理面积）。

    法向**方向**在平面三角形上处处相同，但模长不是常数：面 Flux Points 用坍缩三角形采样
    （`cube_to_tri_rs`/`tri_barycentric`，与棱柱三角形封盖相同）生成，(a,b) 到物理面坐标之间含 `(1-b)/4` 型
    因子，面积微元必须逐 FP 按采样自身的切向量叉积算（用常数行广播会让面积求和差 4 倍）。

    Returns:
        (k, n_fp, 3)
    """
    face_vertex_idx = [v for v in range(4) if v != excluded_vertex]
    p_i = cell_nodes[:, face_vertex_idx[0]]  # (k,3)
    p_j = cell_nodes[:, face_vertex_idx[1]]
    p_k = cell_nodes[:, face_vertex_idx[2]]
    p_excluded = cell_nodes[:, excluded_vertex]
    e1 = p_j - p_i  # (k,3)，常数（平面三角形仿射映射）
    e2 = p_k - p_i

    aa, bb = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    a, b = aa.ravel(), bb.ravel()  # (n_fp,)

    # phys(a,b) = l1(r_tri,s_tri)*p_i + l2*p_j + l3*p_k，其中
    # (r_tri,s_tri)=cube_to_tri_rs(a,b)：见 curved_mapping.py/
    # face_flux_points/ref_geometry_nb.py::_cube_to_tri_rs_nb 同一约定
    # （r=(1+a)(1-b)/2-1, s=b）与 _tri_barycentric_nb（l1=-(r+s)/2,
    # l2=(1+r)/2, l3=(1+s)/2）——链式法则直接展开：
    # d phys/da = (1-b)/4 * e1；d phys/db = -(1+a)/4*e1 + 0.5*e2。
    dphys_da = ((1.0 - b) / 4.0)[None, :, None] * e1[:, None, :]
    dphys_db = (-(1.0 + a) / 4.0)[None, :, None] * e1[:, None, :] + 0.5 * e2[:, None, :]
    raw = np.cross(dphys_da, dphys_db)  # (k,n_fp,3)

    # 定向：outward = 远离被排除顶点的方向
    to_excluded = p_excluded[:, None, :] - p_i[:, None, :]  # (k,1,3)，对 n_fp 广播
    sign = np.where(np.einsum('kpd,kpd->kp', raw, np.broadcast_to(to_excluded, raw.shape)) > 0, -1.0, 1.0)
    return raw * sign[:, :, None]


def compute_exact_adj_rows(
    n_faces: int,
    n1d: int,
    sps_1d: np.ndarray,
    n_prism: int,
    cell_arr: np.ndarray,
    code_arr: np.ndarray,
    prism_conn: np.ndarray,
    tet_conn: np.ndarray,
    node_coords: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对每个面，在 `(cell_arr[f], code_arr[f])`（owner 或 neighbor 侧）下逐 Flux Point 求该单元自身映射的外向
    adj(J) 行，形状 `(n_faces, n_fp, 3)`。

    Args:
        cell_arr, code_arr: (n_faces,)，某一侧的单元与原生面编码
        valid_mask: (n_faces,) bool，None 时全部有效；neighbor 侧传 `~face_conn.is_boundary`，无效项保持为零

    Raises:
        ValueError: 有效项里出现非原生面编码（面连接关系没有经过原生编码翻译）。
    """
    n_fp = n1d * n1d
    adj_row_out = np.zeros((n_faces, n_fp, 3))
    valid = np.ones(n_faces, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    codes = np.asarray(code_arr, dtype=np.int64)
    tet_lo, tet_hi = NATIVE_TET_FACE_CODE_RANGE
    prism_lo, prism_hi = NATIVE_PRISM_FACE_CODE_RANGE
    bad = valid & ~(((codes >= tet_lo) & (codes < tet_hi)) | ((codes >= prism_lo) & (codes < prism_hi)))
    if np.any(bad):
        raise ValueError(f"面编码 {sorted(set(codes[bad].tolist()))} 不是原生编码（四面体 [{tet_lo},{tet_hi})、"
                         f"棱柱 [{prism_lo},{prism_hi})）")

    for ev in range(tet_hi - tet_lo):
        faces_here = np.nonzero(valid & (codes == tet_lo + ev))[0]
        if len(faces_here) == 0:
            continue
        cell_nodes = node_coords[tet_conn[cell_arr[faces_here] - n_prism]]  # (k,4,3)
        adj_row_out[faces_here] = _native_tet_adj_row_batched(ev, cell_nodes, n1d, sps_1d)

    from ..native_prism.face import native_prism_face_adj_rows

    order = n1d - 1
    for fid in range(prism_hi - prism_lo):
        faces_here = np.nonzero(valid & (codes == prism_lo + fid))[0]
        if len(faces_here) == 0:
            continue
        cell_nodes = node_coords[prism_conn[cell_arr[faces_here]]]  # (k,6,3)
        for idx, f in enumerate(faces_here):
            adj_row_out[f] = native_prism_face_adj_rows(order, fid, cell_nodes[idx])

    return adj_row_out


def compute_exact_face_normals_and_weights(
    n_faces: int,
    n1d: int,
    sps_1d: np.ndarray,
    weights_1d: np.ndarray,
    n_prism: int,
    owner_cell: np.ndarray,
    owner_code: np.ndarray,
    prism_conn: np.ndarray,
    tet_conn: np.ndarray,
    node_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """owner 侧逐 Flux Point 的外法向与面积微元（numpy 参照实现）。

    Returns:
        (true_normal, true_area_weight)：形状 (n_faces,n_fp,3) / (n_faces,n_fp)；neighbor 侧外法向取负号。
    """
    adj_row = compute_exact_adj_rows(n_faces, n1d, sps_1d, n_prism, owner_cell, owner_code,
                                     prism_conn, tet_conn, node_coords)
    return normals_and_weights_from_adj_rows(adj_row, weights_1d)


def normals_and_weights_from_adj_rows(adj_row: np.ndarray, weights_1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """owner 侧外向 adj(J) 行 `(n_faces, n_fp, 3)` -> `(true_normal, true_area_weight)`。"""
    wx, wy = np.meshgrid(weights_1d, weights_1d, indexing="ij")
    w_fp = (wx * wy).ravel()  # (n_fp,) 原始（未归一化）张量积求积权重，覆盖 [-1,1]^2
    mag = np.linalg.norm(adj_row, axis=-1)
    true_normal = adj_row / np.maximum(mag, 1e-300)[..., None]
    true_area_weight = mag * w_fp[None, :]
    return true_normal, true_area_weight
