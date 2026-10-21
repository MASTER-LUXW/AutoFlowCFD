"""AutoFlowCFD V2.0 - 逐通量点的精确法向 / 面积权重 / adj(J) 行。

adj 行一律由 **numba 版** `exact_normal_kernel.compute_exact_adj_rows_fast` 计算，法向 / 面积权重由 owner 侧
adj 行派生（`exact_normal.normals_and_weights_from_adj_rows`）；numpy 版 `exact_normal.compute_exact_adj_rows`
只作为测试里的交叉验证参照（两者实测一致到 4e-16）。
"""

from dataclasses import dataclass

import numpy as np

from autoflowcfd.grid.connectivity.face_connectivity import FRFaceConnectivity
from .exact_normal import normals_and_weights_from_adj_rows


@dataclass
class FaceMetrics:
    """字段名与 `merge.py` 里原来的局部变量同名（去掉前导下划线）。"""

    true_normal: np.ndarray            # (n_faces, n_fp, 3)
    true_area_weight: np.ndarray       # (n_faces, n_fp)
    owner_adj_row_exact: np.ndarray    # (n_faces, n_fp, 3)
    neighbor_adj_row_exact: np.ndarray  # (n_faces, n_fp, 3)


def build_face_metrics(
    face_conn: FRFaceConnectivity,
    mesh,
    n_faces: int,
    n1d: int,
    n_prism: int,
    sps_1d: np.ndarray,
    weights_1d: np.ndarray,
    is_bnd: np.ndarray,
) -> FaceMetrics:
    """逐通量点求精确法向/面积权重与两侧的 adj 行。"""
    # 逐 Flux Point 精确求值（不是每个面一个常数）：棱柱四边形侧面一般非平面，一个面上的多个通量点用同一个常数
    # 法向等于对曲面用错误的局部切平面方向做黎曼求解（见 `exact_normal.py` 模块文档）。两侧各自的精确 adj(J) 行
    # 供界面核直接查表读取；neighbor 侧对边界面无意义，用 `~is_bnd` 排除。numba 版与 numpy 参照实现一致到 ~3e-16。
    from .exact_normal_kernel import compute_exact_adj_rows_fast

    _owner_adj_row_exact = compute_exact_adj_rows_fast(
        n_faces, n1d, sps_1d, n_prism,
        cell_arr=np.asarray(face_conn.owner_cell, dtype=np.int64),
        prism_conn=(mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None
                    else np.empty((0, 6), dtype=np.int64)),
        tet_conn=(mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None
                  else np.empty((0, 4), dtype=np.int64)),
        node_coords=mesh._node_coords,
        code_arr=np.asarray(face_conn.owner_cube_face, dtype=np.int64),
    )
    # 法向与面积权重由同一份 owner adj 行派生（不另走 numpy 参照实现重算一遍）。
    _all_normals, _all_area_w = normals_and_weights_from_adj_rows(_owner_adj_row_exact, weights_1d)
    _neighbor_adj_row_exact = compute_exact_adj_rows_fast(
        n_faces, n1d, sps_1d, n_prism,
        cell_arr=np.asarray(face_conn.neighbor_cell, dtype=np.int64),
        prism_conn=(mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None
                    else np.empty((0, 6), dtype=np.int64)),
        tet_conn=(mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None
                  else np.empty((0, 4), dtype=np.int64)),
        node_coords=mesh._node_coords,
        valid_mask=~is_bnd,
        code_arr=np.asarray(face_conn.neighbor_cube_face, dtype=np.int64),
    )

    return FaceMetrics(
        true_normal=_all_normals,
        true_area_weight=_all_area_w,
        owner_adj_row_exact=_owner_adj_row_exact,
        neighbor_adj_row_exact=_neighbor_adj_row_exact,
    )
