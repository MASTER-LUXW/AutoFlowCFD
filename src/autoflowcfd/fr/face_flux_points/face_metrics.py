"""AutoFlowCFD V2.0 - 逐通量点的精确法向 / 面积权重 / adj(J) 行。

从 `merge.py`（原 670 行）拆出（2026-09-19，项目"单文件不超 500 行"
规范）。整段原样搬运，逻辑一字未改；只把原来的局部变量改成形参
（去掉前导下划线）。

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
    neighbor_adj_row_exact: np.ndarray # (n_faces, n_fp, 3)


def build_face_metrics(
    face_conn: FRFaceConnectivity,
    mesh,
    n_faces: int,
    n1d: int,
    n_prism: int,
    sps_1d: np.ndarray,
    weights_1d: np.ndarray,
    o_side_arr: np.ndarray,
    is_bnd: np.ndarray,
) -> FaceMetrics:
    """逐通量点求精确法向/面积权重与两侧的 adj 行。"""
    # 真实 bug 修复（2026-08-23，见 face_flux_points/exact_normal.py 模块
    # 文档完整原理）：此前这里对每个面只取*一个*常数法向/面积（`_normals`/
    # `_areas`，来自三角化的平面近似），在该面全部 n_fp 个 Flux Points 上
    # 重复使用（`np.repeat`）——核心通量 kernel（inviscid_kernel.py 等）
    # 因此对棱柱四边形侧面这类物理上一般非平面的面，用同一个错误的局部
    # 切平面方向做黎曼求解，与 troubled_cell.py 机制2诊断的"面法向失配"
    # 是同一个问题，只是此前只用于事后诊断、从未修正过通量本身实际使用
    # 的法向。改为逐 Flux Point 精确求值（`compute_exact_face_normals_
    # and_weights`，用已经验证过的解析精确 Jacobian，不是新的近似）——
    # 对平面面（绝大多数四面体面、未翘曲的棱柱四边形面）结果与旧的常数
    # 值逐位一致（见对应单元测试），只有真正非平面的棱柱四边形侧面才会
    # 表现出per-FP的真实差异。

    # 真实 bug 修复的第二部分（2026-08-23）：P1+ 内部面主通量（
    # inviscid_kernel.py）与 troubled_cell.py
    # 机制2诊断此前用的"自洽方向"（`own_dir_outward`/`a0,a1,a2`）是对
    # SP 网格上的 adj(J) 做 Lagrange 外插到 FP，不是本次同款的逐点精确
    # 求值——这里预计算 owner/neighbor 两侧各自的精确 adj(J) 行（未归一化、
    # 未按 side 定向的原始值，与下游消费点 `a0,a1,a2` 变量的语义一致），
    # 供这些 kernel 直接查表读取，取代它们内部的 `_extrap_matmul(adj_j
    # [cell,:,axis,:], E)` 外插调用。neighbor 侧对边界面无意义（
    # neighbor_axis/side 是 -1/0.0 哨兵值），用 `~is_bnd` 排除。
    # 用 numba 并行加速版替代纯 NumPy 桶循环版（第四次评审接入）：
    # compute_exact_adj_rows 在 P2+ 大网格上是真实存在的性能瓶颈（逐面
    # 类型分桶 + 批量 NumPy 临时数组，峰值内存 ~5GB，P2 网格初始化卡住
    # 10+ 分钟——见 face_flux_points/exact_normal_kernel.py 模块文档）。
    # `compute_exact_adj_rows_fast` 此前虽已实现且接口完全一致，但从未
    # 被接入生产路径；本轮评审用 order 1/2/3 × tet/prism 合成算例逐位
    # 交叉验证（最大绝对误差 ~3e-16，机器精度量级），确认可以安全替换。
    from .exact_normal_kernel import compute_exact_adj_rows_fast

    # code_arr 透传（native 四面体分派，见 compute_exact_adj_rows_kernel/
    # _native_tet_adj_row_at 文档"真实 bug 修复"说明——owner_axis/
    # owner_side 对 native 面存的是复用的 excluded_vertex/哑值，此前
    # 这个"fast" kernel 完全不知道这一点，见该函数文档完整原理）。
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
    # 法向与面积权重由同一份 owner adj 行派生（I-35，2026-10-19）：此前先用 numpy 参照实现
    # （`compute_exact_face_normals_and_weights`，内部重算一遍 owner adj 行——就是上面那条峰值 ~5GB、P2 初始化 10+
    # 分钟的路径）只为求这两个量，随后 numba 版再算一遍 owner adj 行。两者一致到 ~4e-16。
    _all_normals, _all_area_w = normals_and_weights_from_adj_rows(
        _owner_adj_row_exact, weights_1d, o_side_arr.astype(np.float64),
        np.asarray(face_conn.owner_cube_face, dtype=np.int64))
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
