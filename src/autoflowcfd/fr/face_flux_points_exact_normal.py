"""FR Flux Point 精确逐点法向量/面积权重/自洽方向计算 (2026-08-23)。

真实 bug 修复：`true_normal`/`true_area_weight`（core/fr_residual/
inviscid_kernel.py 等核心通量 kernel 实际使用的法向/面积）此前对每个面
只取*一个*常数值（`face_connectivity.py::build_face_connectivity` 三角化
出的平面近似），在该面所有 Flux Points 上重复使用（
`face_flux_points_merge.py`: `_all_normals = np.repeat(_normals, n_fp, ...)`）。

棱柱四边形侧面在物理上一般不是平面（4 个角点定义的是一个双线性曲面，
只有角点严格共面时才退化为平面）——这不是网格缺陷，是棱柱+曲面挤出这
个几何表示组合固有的性质（真实网格实测最大翘曲 11.13%，见
face_flux_points_merge.py 的既有记录）。P0（每个面只有 1 个"平均"通量
点）感受不到这个差异；P1 起，一个面上有多个 Flux Points 分布在面内部
不同位置，用同一个常数法向计算通量，实质上是对一个非平面曲面用错误的
局部切平面方向做黎曼求解——`core/fr_operators/troubled_cell.py` 的机制2
诊断（face-normal misalignment）测的正是这个：owner 自身映射隐含的局部
法向（`own_dir_outward`，来自 adj(J)）与这个全面共用的常数 true_normal
之间的偏差，真实网格上最大到 37°，95%+ 单元超过 1°。真实长程 P1 生产
网格测试（cube_demo，2026-08-23）显示这个不一致与 P1 阶段残差异常/
无法收敛直接相关。

本模块用`grid/curved_mapping/curved_mapping_exact_jacobian.py`已经验证过
的解析精确 Jacobian（`tet_exact_jacobian`/`prism_exact_jacobian`——纯闭式
表达式，无谱微分截断误差，该模块文档已证实 GCL 残差从~1e-14 降到
~1e-19），直接在每个 Flux Point 自己的精确参考坐标处求值，取该点所属
单元在该点的 adj(J) 对应行。两个用途：

1. `compute_exact_face_normals_and_weights`：取 owner 侧的值，归一化+按
   `side` 定向作为 `true_normal`，模长乘以原始求积权重作为
   `true_area_weight`——供该面全部 Flux Points 使用（owner 算出的值同时
   供 neighbor 侧使用，取负号，与现有 inviscid_kernel.py 的
   "neighbor 视角外法向恒为 -true_normal" 约定完全一致，不改变通量
   守恒结构）。
2. `compute_exact_adj_rows`：给定任意一侧（owner 或 neighbor）各自的
   (cell,axis,side) 赋值，返回*未归一化、未按 side 定向*的原始 adj(J)
   行——供 `inviscid_kernel.py`/`inviscid_kernel_colored.py`/
   `viscous_flux.py` 的"自洽方向"计算（P1+ 内部面主通量的实际输入，见
   `5_重大问题修复-黎曼求解器法向.md`）和 `troubled_cell.py` 机制2诊断
   使用——此前这两处都是对 SP 网格上的 adj(J) 做 Lagrange 外插到 FP
   （`_extrap_matmul(adj_j[cell,:,axis,:], E)`），外插本身对坍缩坐标下
   本质是有理函数的 adj(J) 有不可忽略的截断误差（P1 尤其明显，是"P1
   GCL 0.105 度量混叠"同一个误差来源）；改为直接在 FP 精确坐标求值，
   消除这部分截断误差，不是新的近似。

对平面面（全部四面体面、未翘曲的棱柱四边形面）本模块给出的值应与旧的
三角化常数法向/SP 外插值逐位一致或高度接近（不是近似退化）——
tests/unit/test_face_flux_points_exact_normal.py 显式验证了这一点。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from autoflowcfd.grid.curved_mapping.curved_mapping import (
    TET_CUBE_FACES, PRISM_CUBE_FACES,
)
from autoflowcfd.grid.curved_mapping.curved_mapping_exact_jacobian import (
    tet_exact_jacobian, prism_exact_jacobian,
)
from autoflowcfd.fr.face_flux_points import CUBE_FACE_AXIS_SIDE, face_ref_grid


def _tet_exact_jacobian_batched(ref_pts: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """`tet_exact_jacobian` 的批量版本（多个面各自不同的单元/角点，共享
    同一个 (axis,side) 因而同一组参考坐标）。

    Args:
        ref_pts: (n_fp, 3)，该 (axis,side) 面固定的参考坐标网格（所有面共用）
        cell_nodes: (n_faces, 4, 3)，每个面各自单元的 4 个角点

    Returns:
        (n_faces, n_fp, 3, 3)，J[...,phys,m]，与 tet_exact_jacobian 单点版
        逐位一致（已用单点版逐面循环数值核对，见对应单元测试）
    """
    a, b, c = ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2]  # (n_fp,)
    p0, p1, p2, p3 = (cell_nodes[:, i] for i in range(4))  # each (n_faces,3)
    e1 = (p1 - p0) / 2.0
    e2 = (p2 - p0) / 2.0
    e3 = (p3 - p0) / 2.0

    s = (1.0 + b) * (1.0 - c) / 2.0 - 1.0
    t = c

    n_faces = cell_nodes.shape[0]
    n_fp = ref_pts.shape[0]
    jac = np.zeros((n_faces, n_fp, 3, 3))
    coef_a = -(s + t) / 2.0  # (n_fp,)
    jac[:, :, :, 0] = coef_a[None, :, None] * e1[:, None, :]

    coef_b1 = -(1.0 + a) * (1.0 - c) / 4.0
    coef_b2 = (1.0 - c) / 2.0
    jac[:, :, :, 1] = coef_b1[None, :, None] * e1[:, None, :] + coef_b2[None, :, None] * e2[:, None, :]

    coef_c1 = -(1.0 + a) * (1.0 - b) / 4.0
    coef_c2 = -(1.0 + b) / 2.0
    jac[:, :, :, 2] = (
        coef_c1[None, :, None] * e1[:, None, :]
        + coef_c2[None, :, None] * e2[:, None, :]
        + e3[:, None, :]
    )
    return jac


def _prism_exact_jacobian_batched(ref_pts: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """`prism_exact_jacobian` 的批量版本，参数/返回值约定同
    `_tet_exact_jacobian_batched`（cell_nodes 改为 (n_faces,6,3)）。"""
    a, b, c = ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2]
    p0, p1, p2, p3, p4, p5 = (cell_nodes[:, i] for i in range(6))

    d_bottom_da = ((1.0 - b) / 4.0)[None, :, None] * (p1 - p0)[:, None, :]
    d_top_da = ((1.0 - b) / 4.0)[None, :, None] * (p4 - p3)[:, None, :]
    d_bottom_db = (
        (-(1.0 - a) / 4.0)[None, :, None] * p0[:, None, :]
        + (-(1.0 + a) / 4.0)[None, :, None] * p1[:, None, :]
        + 0.5 * p2[:, None, :]
    )
    d_top_db = (
        (-(1.0 - a) / 4.0)[None, :, None] * p3[:, None, :]
        + (-(1.0 + a) / 4.0)[None, :, None] * p4[:, None, :]
        + 0.5 * p5[:, None, :]
    )

    r = (1.0 + a) * (1.0 - b) / 2.0 - 1.0
    s = b
    l1 = -(r + s) / 2.0
    l2 = (1.0 + r) / 2.0
    l3 = (1.0 + s) / 2.0
    bottom = l1[None, :, None] * p0[:, None, :] + l2[None, :, None] * p1[:, None, :] + l3[None, :, None] * p2[:, None, :]
    top = l1[None, :, None] * p3[:, None, :] + l2[None, :, None] * p4[:, None, :] + l3[None, :, None] * p5[:, None, :]

    n_faces = cell_nodes.shape[0]
    n_fp = ref_pts.shape[0]
    jac = np.zeros((n_faces, n_fp, 3, 3))
    jac[:, :, :, 0] = 0.5 * (1.0 - c)[None, :, None] * d_bottom_da + 0.5 * (1.0 + c)[None, :, None] * d_top_da
    jac[:, :, :, 1] = 0.5 * (1.0 - c)[None, :, None] * d_bottom_db + 0.5 * (1.0 + c)[None, :, None] * d_top_db
    jac[:, :, :, 2] = 0.5 * (top - bottom)[None, :, :]
    return jac


_FACE_DEFS = [
    (name, axis, side, jac_fn, is_prism)
    for is_prism, valid_names, jac_fn in (
        (True, PRISM_CUBE_FACES, _prism_exact_jacobian_batched),
        (False, TET_CUBE_FACES, _tet_exact_jacobian_batched),
    )
    for name, (axis, side) in CUBE_FACE_AXIS_SIDE.items()
    if name in valid_names
]

def _native_tet_adj_row_batched(
    excluded_vertex: int, cell_nodes: np.ndarray, n1d: int, sps_1d: np.ndarray
) -> np.ndarray:
    """native 四面体版本的"adj(J) 行"——与坍缩坐标版本
    `compute_exact_adj_rows` 返回值同一约定：方向是 outward 法向，模长是
    该 Flux Point 参考采样坐标 (a,b) 下的局部面积 Jacobian（`true_area_
    weight=mag*w_fp` 直接积分给出真实物理面积）。

    真实 bug 修复过程中的一个教训（如实记录）：**不能**简单套用"直边
    四面体 Jacobian 是常数矩阵，所以这一行对该面所有 FP 都相同"这个
    直觉——法向量*方向*确实处处相同（平面三角形），但 native 四面体
    自己的面 Flux Points 是用 `native_tet_face_points_physical`/
    `_native_tet_face_points_nb` 同一套**坍缩三角形**采样
    （`cube_to_tri_rs`/`tri_barycentric`，与棱柱三角形封盖完全相同）
    生成的物理点位置，(a,b) 立方体坐标到物理面坐标之间本身就含有
    `(1-b)/4` 型非线性因子（与 `_prism_exact_jacobian_batched` 对
    c=-1/c=+1 面的处理是同一个道理，那里全程没有走过"常数捷径"），
    面积微元必须逐 FP 按这套采样自身的切向量叉积算，不能用常数行
    广播——最初实现漏了这一步，被本模块单元测试（真实叉积面积交叉
    验证）当场测出（面积微元求和与参考值相差整整 4 倍），已按下面的
    正确公式重新实现。

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
    # face_flux_points_helpers_numba.py::_cube_to_tri_rs_nb 同一约定
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
    axis_arr: np.ndarray,
    side_arr: np.ndarray,
    prism_conn: np.ndarray,
    tet_conn: np.ndarray,
    node_coords: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    code_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对每个面，在其（可以是 owner 侧也可以是 neighbor 侧的）
    `(cell_arr[f], axis_arr[f], side_arr[f])` 赋值下，逐 Flux Point 精确
    求出该单元自身映射在该点的 adj(J) 第 `axis_arr[f]` 行——*未归一化、
    未按 side 定向*的原始值（与 `own_dir_outward` 计算里 `a0,a1,a2`
    这一步等价，见模块文档）。

    Args:
        cell_arr, axis_arr, side_arr: (n_faces,)，可以是 owner 或
            neighbor 侧的赋值（对边界面，neighbor 侧无意义，用
            `valid_mask` 排除，不要求 axis_arr/side_arr 对这些条目有效）
        valid_mask: (n_faces,) bool，None 时视为全部有效（例如 owner 侧
            对任意面都有效）；neighbor 侧调用时传
            `~face_conn.is_boundary`，跳过边界面（其 neighbor_axis 是
            -1 的哨兵值，不是一个真实的坍缩方向）
        code_arr: (n_faces,) 或 None——原始 cube face 编码（`CUBE_FACE_CODES`
            约定，0~5 坍缩坐标、6~9 native 四面体、10~14 native 棱柱）。**真实 bug 修复**
            （Part7 文档阶段2"执行状态更新"节）：`axis_arr`/`side_arr` 对
            native 四面体面（`face_flux_points_merge.py` 里 `_CF_AXIS`/
            `_CF_SIDE` 把 axis 槽位复用成 excluded_vertex）给出的值与
            坍缩坐标的 (axis,side) 语义**在数值上会碰撞**（如
            excluded_vertex=0 与坍缩坐标 "a=-1" 面同样是 (axis=0,side=-1)，
            excluded_vertex=3 则不落在任何坍缩坐标条目里，得到全零）——
            不传 `code_arr` 时这个碰撞会被静默放行（对应旧行为，纯坍缩
            坐标网格不受影响，因为那时 `axis_arr` 从不会取到"实为
            excluded_vertex"的含义）；传入且其中有 `>=6` 的项时，那些
            face 改用 `_native_tet_adj_row_batched`（excluded_vertex=
            code-6）单独计算，不再落入下面按 (axis,side) 匹配的坍缩坐标
            分支。
    """
    n_fp = n1d * n1d
    adj_row_out = np.zeros((n_faces, n_fp, 3))
    valid = np.ones(n_faces, dtype=bool) if valid_mask is None else valid_mask
    is_prism_cell = cell_arr < n_prism

    is_native = valid & (code_arr >= 6) if code_arr is not None else np.zeros(n_faces, dtype=bool)

    for name, axis, side, jac_fn, is_prism in _FACE_DEFS:
        type_mask = is_prism_cell if is_prism else ~is_prism_cell
        face_mask = valid & ~is_native & type_mask & (axis_arr == axis) & (side_arr == side)
        faces_here = np.nonzero(face_mask)[0]
        if len(faces_here) == 0:
            continue

        cells_here = cell_arr[faces_here]
        if is_prism:
            node_ids = prism_conn[cells_here]
        else:
            node_ids = tet_conn[cells_here - n_prism]
        cell_nodes = node_coords[node_ids]  # (k,6,3) or (k,4,3)

        ref_pts = face_ref_grid(n1d, axis, side, sps_1d)  # (n_fp,3)，本桶所有面共用
        J = jac_fn(ref_pts, cell_nodes)  # (k,n_fp,3,3)

        det_J = np.linalg.det(J)
        inv_J = np.linalg.inv(J)
        adj_row_out[faces_here] = det_J[..., None] * inv_J[:, :, axis, :]

    if np.any(is_native):
        # 两类原生面分开：四面体是 [6,10)、棱柱是 [10,15)。**必须分开**：
        # 棱柱面走 `ev = code - 6` 会取到 4~8，落在
        # `_native_tet_adj_row_batched` 的 `else` 分支上（那个函数只认
        # 0~3），静默按"排除顶点 3"的四面体公式算出完全错误的法向。
        for ev in range(4):
            faces_here = np.nonzero(is_native & (code_arr == 6 + ev))[0]
            if len(faces_here) == 0:
                continue
            cells_here = cell_arr[faces_here]
            node_ids = tet_conn[cells_here - n_prism]
            cell_nodes = node_coords[node_ids]  # (k,4,3)
            adj_row_out[faces_here] = _native_tet_adj_row_batched(ev, cell_nodes, n1d, sps_1d)

        # 原生棱柱面（编码 10~14）：`native_prism_face_adj_rows` 给出的行
        # 已按 outward 定向（参考余向量自带符号），与下游
        # `side_factor = 1.0 if code >= 6` 的既有处理自洽。
        #
        # 实测这批行与坍缩路径逐点等价（法向 3.3e-16、物理面积权重
        # 2.4e-15），所以换基**不改变面几何的量**，只是走了自带定向的
        # 那条公式，从而让"原生面 side_factor 恒为 1"这个统一约定成立。
        for fid in range(5):
            faces_here = np.nonzero(is_native & (code_arr == 10 + fid))[0]
            if len(faces_here) == 0:
                continue
            from .native_prism.face import native_prism_face_adj_rows

            order = n1d - 1
            cells_here = cell_arr[faces_here]
            node_ids = prism_conn[cells_here]
            cell_nodes = node_coords[node_ids]  # (k,6,3)
            for idx, f in enumerate(faces_here):
                adj_row_out[f] = native_prism_face_adj_rows(
                    order, fid, cell_nodes[idx])

    return adj_row_out


def compute_exact_face_normals_and_weights(
    n_faces: int,
    n1d: int,
    sps_1d: np.ndarray,
    weights_1d: np.ndarray,
    n_prism: int,
    owner_cell: np.ndarray,
    owner_axis: np.ndarray,
    owner_side: np.ndarray,
    prism_conn: np.ndarray,
    tet_conn: np.ndarray,
    node_coords: np.ndarray,
    owner_code: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """对每个面的每个 Flux Point，用其 owner 单元在该 FP 精确参考坐标处
    的解析 Jacobian，算出局部外法向 + 面积微元（取代此前面级别常数值）。

    Args:
        owner_code: (n_faces,) 或 None——owner 侧原始 cube face 编码，
            透传给 `compute_exact_adj_rows` 做 native 四面体分派（见该
            函数文档"真实 bug 修复"说明）；native 四面体面（code>=6）
            对应的 `owner_side` 条目本函数不使用（`_native_tet_adj_row_
            batched` 已经返回按 outward 定向好的行，不需要再乘 side），
            调用方仍然可以照常传入 `_CF_SIDE` 查表得到的占位值。

    Returns:
        (true_normal, true_area_weight)：形状 (n_faces,n_fp,3) / (n_faces,n_fp)，
        与原 `_all_normals`/`_all_area_w` 完全相同的语义和消费方式（owner 侧
        外法向按 owner FP 网格顺序排列；neighbor 侧取负号，调用方不变）。
    """
    n_fp = n1d * n1d
    wx, wy = np.meshgrid(weights_1d, weights_1d, indexing="ij")
    w_fp = (wx * wy).ravel()  # (n_fp,) 原始（未归一化）张量积求积权重，覆盖 [-1,1]^2

    adj_row = compute_exact_adj_rows(
        n_faces, n1d, sps_1d, n_prism, owner_cell, owner_axis, owner_side,
        prism_conn, tet_conn, node_coords, code_arr=owner_code,
    )
    mag = np.linalg.norm(adj_row, axis=-1)
    mag_safe = np.maximum(mag, 1e-300)
    # native 四面体（owner_code>=6）：_native_tet_adj_row_batched 已经是
    # 按 outward 定向好的行，这里的 side 因子必须视同 +1，不能用调用方
    # 传入的 owner_side 占位值（对 excluded_vertex 复用的 axis 槽位没有
    # +1/-1 语义，见 face_flux_points.py::CUBE_FACE_AXIS_SIDE 文档）。
    if owner_code is not None:
        side_factor = np.where(owner_code >= 6, 1.0, owner_side)
    else:
        side_factor = owner_side
    true_normal = (adj_row / mag_safe[..., None]) * side_factor[:, None, None]
    true_area_weight = mag * w_fp[None, :]

    return true_normal, true_area_weight
