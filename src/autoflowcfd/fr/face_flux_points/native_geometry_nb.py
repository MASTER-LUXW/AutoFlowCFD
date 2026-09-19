"""AutoFlowCFD V2.0 - **原生**基参考几何的 numba 原语 + 插值共用入口。

原生四面体（路径C）的 numba 版几何原语，以及三条基共用的"面点 -> 体积
节点"插值矩阵入口 `interp_matrix_from_cube_coords_nb`（坍缩棱柱 / 坍缩
四面体 / 原生棱柱）。

从原 `kernel_helpers.py`（699 行）按功能拆出（2026-09-19，项目"单文件
不超 500 行"规范）：坍缩坐标那一半在 `ref_geometry_nb.py`，面编码查找表
在 `face_code_tables.py`。纯搬家，未改任何逻辑。
"""

import numpy as np
from numba import njit

from autoflowcfd.fr.collapsed_basis import (
    prism_modal_basis_and_grad,
    tet_modal_basis_and_grad,
)
from autoflowcfd.fr.native_prism.interp_numba import (
    native_prism_interp_matrix_nb,
)
from autoflowcfd.fr.native_tet.basis import simplex3d_value
from .ref_geometry_nb import _cube_to_tri_rs_nb, _tri_barycentric_nb


# ============================================================================
# native 四面体（路径C，非坍缩坐标）numba 版几何原语
#
# 与上面坍缩坐标（(axis,side)）版本平行——见
# `fr/native_tet/basis.py`/`fr/face_flux_points/locate.py::
# locate_native_tet_face_point`/`fr/face_flux_points/geometry.py::
# native_tet_face_points_physical` 的纯 Python 参考实现，本节是它们的
# 逐行 numba 移植，供 `face_flux_points/kernel.py::build_fp_newton_
# parallel` 内部对 native 四面体面（cube face code>=6，excluded_vertex=
# code-6）分支消费，避免退回到假设 (axis,side) 语义的 `_newton_locate_nb`/
# `_face_ref_grid_nb`。face code>=6 只可能出现在 native 四面体的真实面上
# （见 grid/connectivity/face_connectivity.py::with_native_face_codes
# 文档），棱柱四边形侧面恒为 0~5，因此本节所有分支判据只需要检查
# `code>=6`，不需要额外的 is_prism 判据。
# ============================================================================

# excluded_vertex(0~3) -> 该真实面 3 个顶点的局部下标（升序），与
# native_tet/basis.py::face_node_indices/face_flux_points.py::
# native_tet_face_points_physical 的 `tuple(v for v in range(4) if v !=
# excluded_vertex)` 完全一致。
_NATIVE_FACE_VI = np.array([1, 0, 0, 0], dtype=np.int32)
_NATIVE_FACE_VJ = np.array([2, 2, 1, 1], dtype=np.int32)
_NATIVE_FACE_VK = np.array([3, 3, 3, 2], dtype=np.int32)


@njit(cache=True)
def _rst_to_abc_nb(r, s, t):
    """`native_tet.basis.rst_to_abc` 逐行移植（numba 版，逐点标量
    分支代替 `np.where`）——只用于取值，不用于求导，见该函数文档。"""
    n = r.shape[0]
    a = np.empty(n)
    b = np.empty(n)
    c = np.empty(n)
    for p in range(n):
        c[p] = t[p]
        denom_b = 1.0 - c[p]
        if abs(denom_b) > 1e-12:
            b[p] = 2.0 * (s[p] + 1.0) / denom_b - 1.0
        else:
            b[p] = -1.0
        denom_a = s[p] + t[p]
        if abs(denom_a) > 1e-12:
            a[p] = -2.0 * (1.0 + r[p]) / denom_a - 1.0
        else:
            a[p] = -1.0
    return a, b, c


@njit(cache=True)
def _map_native_tet_to_physical_nb(rst, cell_nodes):
    """`native_tet.basis.map_native_tet_to_physical` 逐行移植：重心
    坐标仿射组合，直接从 (r,s,t) 到物理坐标，不经过任何坍缩坐标中间量。"""
    n = rst.shape[0]
    result = np.empty((n, 3))
    for p in range(n):
        r, s, t = rst[p, 0], rst[p, 1], rst[p, 2]
        L1 = -(1.0 + r + s + t) / 2.0
        L2 = (1.0 + r) / 2.0
        L3 = (1.0 + s) / 2.0
        L4 = (1.0 + t) / 2.0
        for d in range(3):
            result[p, d] = (
                L1 * cell_nodes[0, d] + L2 * cell_nodes[1, d]
                + L3 * cell_nodes[2, d] + L4 * cell_nodes[3, d]
            )
    return result


@njit(cache=True)
def _native_tet_face_points_nb(n1d, excluded_vertex, cell_nodes, sps_1d):
    """`face_flux_points/geometry.py::native_tet_face_points_physical` 逐行移植：
    在 native 四面体某个真实面上，用与棱柱三角形封盖相同的坍缩三角形
    采样（`_cube_to_tri_rs_nb`/`_tri_barycentric_nb`，本文件已有）生成
    `n1d*n1d` 个物理点——只用于选取物理点位置，与用哪套基函数表示体积场
    无关，是安全复用（见 Python 参考实现文档）。"""
    vi = _NATIVE_FACE_VI[excluded_vertex]
    vj = _NATIVE_FACE_VJ[excluded_vertex]
    vk = _NATIVE_FACE_VK[excluded_vertex]
    Pi = cell_nodes[vi]
    Pj = cell_nodes[vj]
    Pk = cell_nodes[vk]
    n_fp = n1d * n1d
    result = np.empty((n_fp, 3))
    for i in range(n1d):
        for j in range(n1d):
            flat = i * n1d + j
            r_tri, s_tri = _cube_to_tri_rs_nb(sps_1d[i], sps_1d[j])
            l1, l2, l3 = _tri_barycentric_nb(r_tri, s_tri)
            for d in range(3):
                result[flat, d] = l1 * Pi[d] + l2 * Pj[d] + l3 * Pk[d]
    return result


@njit(cache=True)
def _tet_native_locate_nb(cell_nodes, excluded_vertex, targets_phys):
    """`face_flux_points/locate.py::locate_native_tet_face_point` 逐行
    移植：native 四面体某个真实面（`excluded_vertex` 给定）上一批目标
    物理点的精确点位定位——与坍缩坐标版本 `_tet_exact_locate_nb` 共享
    同一个重心坐标最小二乘解算（同样的 alpha/beta 公式），区别只在收尾：
    直接换算 (r,s,t)，不经过含显式除法、在退化轴附近病态的 (a,b,c)
    坍缩坐标转换——这正是 native 模式点位定位不会出现条件数病态的原因。

    Returns:
        (free, rst, resid)：
        `free` 形状 (n_pts,2)，(alpha,beta) 两个自由重心坐标分量
        （与坍缩坐标版本的 2 个自由方向坐标同一存储宽度，供
        `nb_fc`/`ow_fc` 复用同一个数组形状）；
        `rst` 形状 (n_pts,3)，供调用方立即构造插值矩阵，不需要从
        `free` 反推（`excluded_vertex` 已知，反推是多余的重复计算）；
        `resid` 形状 (n_pts,)，每点物理残差（绝对长度单位），
        与 `_newton_locate_nb` 同一惯例，不在这里归约成单一标量。
    """
    vi = _NATIVE_FACE_VI[excluded_vertex]
    vj = _NATIVE_FACE_VJ[excluded_vertex]
    vk = _NATIVE_FACE_VK[excluded_vertex]
    Pi = cell_nodes[vi]
    Pj = cell_nodes[vj]
    Pk = cell_nodes[vk]
    e1 = Pj - Pi
    e2 = Pk - Pi
    n_pts = targets_phys.shape[0]
    e1_n = np.sqrt(max(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2, 1e-300))
    e2_n = np.sqrt(max(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2, 1e-300))
    e1h = e1 / e1_n
    e2h = e2 / e2_n
    a11 = e1h[0] ** 2 + e1h[1] ** 2 + e1h[2] ** 2
    a12 = e1h[0] * e2h[0] + e1h[1] * e2h[1] + e1h[2] * e2h[2]
    a22 = e2h[0] ** 2 + e2h[1] ** 2 + e2h[2] ** 2

    free = np.empty((n_pts, 2))
    rst = np.empty((n_pts, 3))
    for p in range(n_pts):
        r0 = targets_phys[p, 0] - Pi[0]
        r1 = targets_phys[p, 1] - Pi[1]
        r2 = targets_phys[p, 2] - Pi[2]
        b1 = r0 * e1h[0] + r1 * e1h[1] + r2 * e1h[2]
        b2 = r0 * e2h[0] + r1 * e2h[1] + r2 * e2h[2]
        det = a11 * a22 - a12 * a12
        ds = det if abs(det) > 1e-300 else 1e-300
        alpha = (b1 * a22 - b2 * a12) / ds / e1_n
        beta = (a11 * b2 - a12 * b1) / ds / e2_n
        free[p, 0] = alpha
        free[p, 1] = beta

        L = np.zeros(4)
        L[vi] = 1.0 - alpha - beta
        L[vj] = alpha
        L[vk] = beta
        rst[p, 0] = 2.0 * L[1] - 1.0
        rst[p, 1] = 2.0 * L[2] - 1.0
        rst[p, 2] = 2.0 * L[3] - 1.0

    phys_check = _map_native_tet_to_physical_nb(rst, cell_nodes)
    resid = np.empty(n_pts)
    for p in range(n_pts):
        dx = phys_check[p, 0] - targets_phys[p, 0]
        dy = phys_check[p, 1] - targets_phys[p, 1]
        dz = phys_check[p, 2] - targets_phys[p, 2]
        resid[p] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return free, rst, resid


@njit(cache=True)
def _native_alpha_beta_to_rst_nb(excluded_vertex, free):
    """把 `_tet_native_locate_nb` 存下来的 (alpha,beta) 自由坐标（与
    `nb_fc`/`ow_fc` 复用同一个 (n_fp,2) 数组槽位）还原成 (r,s,t)——供
    `face_flux_points/kernel_multisource.py` 的 "primary interp" 分支使用：那里
    只有已经存好的 (alpha,beta)，没有重新调用一次定位（Newton/闭式解）
    的必要，与 `_tet_native_locate_nb` 内部收尾算的是同一个代数关系。"""
    vi = _NATIVE_FACE_VI[excluded_vertex]
    vj = _NATIVE_FACE_VJ[excluded_vertex]
    vk = _NATIVE_FACE_VK[excluded_vertex]
    n_pts = free.shape[0]
    rst = np.empty((n_pts, 3))
    for p in range(n_pts):
        alpha = free[p, 0]
        beta = free[p, 1]
        L = np.zeros(4)
        L[vi] = 1.0 - alpha - beta
        L[vj] = alpha
        L[vk] = beta
        rst[p, 0] = 2.0 * L[1] - 1.0
        rst[p, 1] = 2.0 * L[2] - 1.0
        rst[p, 2] = 2.0 * L[3] - 1.0
    return rst


@njit(cache=True)
def _native_interp_matrix_nb(rst, native_mode_i, native_mode_j, native_mode_k, v_sps_inv_native, n_sps):
    """给定一批目标点的 native 原生参考坐标 (r,s,t)，构造它们相对某个
    native 四面体体积节点的插值矩阵，形状 (n_pts, n_sps)——`n_sps` 是
    全局统一宽度（`n1d**3`），按 Part6/7"补位对齐"原则只写前
    `n_native=len(native_mode_i)` 列，其余列保持零（`np.zeros` 初始化）。
    `build_fp_newton_parallel`（主 kernel）与 `build_ms_interp_parallel`
    （multi-source kernel）的 native 分支共用同一份实现，避免重复。"""
    n_pts = rst.shape[0]
    a_v, b_v, c_v = _rst_to_abc_nb(rst[:, 0], rst[:, 1], rst[:, 2])
    n_native = native_mode_i.shape[0]
    V_t = np.empty((n_pts, n_native))
    for m in range(n_native):
        V_t[:, m] = simplex3d_value(a_v, b_v, c_v, native_mode_i[m], native_mode_j[m], native_mode_k[m])
    interp = np.zeros((n_pts, n_sps))
    for p in range(n_pts):
        for s in range(n_native):
            val = 0.0
            for m in range(n_native):
                val += V_t[p, m] * v_sps_inv_native[m, s]
            interp[p, s] = val
    return interp


@njit(cache=True)
def interp_matrix_from_cube_coords_nb(
    abc, is_prism, is_native_prism, n1d, n_sps,
    v_sps_inv_prism, v_sps_inv_tet,
    v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k,
):
    """给定目标点的**坍缩立方体**参考坐标，构造"面点 -> 体积节点"插值矩阵。

    三条基共用这**一个**入口：坍缩棱柱 / 坍缩四面体 / 原生棱柱。

    ## 为什么抽出来

    主 kernel 与 multi-source kernel 里原本有 **6 处**逐字相同的
    "构造 V_t、再手写三重循环做矩阵乘"。给原生棱柱加支持如果按老样子
    在每处再加一个分支，就是 6 份要同步的实现 —— 而本项目已多次因为
    "两份实现只改了一份"出真实缺陷，且这里改错不会报错、只会静默拿到
    错的邻居插值。

    ## 原生棱柱为什么也能用**坍缩**坐标进来

    坐标往返恰好是恒等（`cube_to_tri_rs` 之后 `rs_to_ab` 回到原值），所以
    原生棱柱模态可以直接在 `(a,b,c)` 上求值、零换算；实测与走
    `build_native_prism_vandermonde` 吻合 9.4e-16 相对。完整推导见
    `fr/native_prism/interp_numba.py` 模块文档。

    Args:
        abc: `(n_pts, 3)` 目标点的坍缩立方体坐标
        is_prism: 目标单元是棱柱（否则四面体）
        is_native_prism: 目标面是**原生棱柱**面（编码 [10,15)）
        n1d: `order + 1`
        n_sps: 全局统一宽度 `n1d**3`
        v_sps_inv_prism / v_sps_inv_tet: 两条坍缩基的节点 Vandermonde 逆
        v_sps_inv_np / np_mode_*: 原生棱柱的 Vandermonde 逆与模态索引

    Returns:
        `(n_pts, n_sps)`。原生棱柱只写前 `n_native` 列、其余为零
        （"补位对齐"约定，与原生四面体那条一致）。
    """
    if is_native_prism:
        return native_prism_interp_matrix_nb(
            abc, np_mode_i, np_mode_j, np_mode_k, v_sps_inv_np, n_sps)
    n_pts = abc.shape[0]
    if is_prism:
        V_t = prism_modal_basis_and_grad(
            abc[:, 0], abc[:, 1], abc[:, 2], n1d - 1)[0]
        V_inv = v_sps_inv_prism
    else:
        V_t = tet_modal_basis_and_grad(
            abc[:, 0], abc[:, 1], abc[:, 2], n1d - 1)[0]
        V_inv = v_sps_inv_tet
    out = np.zeros((n_pts, n_sps))
    for p in range(n_pts):
        for s in range(n_sps):
            val = 0.0
            for m in range(n_sps):
                val += V_t[p, m] * V_inv[m, s]
            out[p, s] = val
    return out
