"""
AutoFlowCFD - FP 几何构建 numba 辅助函数

坐标变换、物理映射、参考面网格、Newton 迭代定位和查找表。
被 face_flux_points_numba（主 kernel）和 face_flux_points_ms_numba
（multi-source kernel）共同依赖。
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
from autoflowcfd.fr.face_flux_points import CUBE_FACE_AXIS_SIDE
from autoflowcfd.fr.face_flux_points_data import _PRISM_QUAD_CODES
from autoflowcfd.fr.native_simplex_basis import simplex3d_value
from autoflowcfd.grid.connectivity.face_connectivity import CUBE_FACE_CODES


# ============================================================================
# numba 版坐标变换
# ============================================================================


@njit(cache=True)
def _cube_to_tet_rst_nb(a, b, c):
    t = c
    s = (1.0 + b) * (1.0 - c) / 2.0 - 1.0
    r = -(1.0 + a) * (s + t) / 2.0 - 1.0
    return r, s, t


@njit(cache=True)
def _tet_barycentric_nb(r, s, t):
    L1 = -(1.0 + r + s + t) / 2.0
    L2 = (1.0 + r) / 2.0
    L3 = (1.0 + s) / 2.0
    L4 = (1.0 + t) / 2.0
    return L1, L2, L3, L4


@njit(cache=True)
def _cube_to_tri_rs_nb(a, b):
    s = b
    r = (1.0 + a) * (1.0 - b) / 2.0 - 1.0
    return r, s


@njit(cache=True)
def _tri_barycentric_nb(r, s):
    l1 = -(r + s) / 2.0
    l2 = (1.0 + r) / 2.0
    l3 = (1.0 + s) / 2.0
    return l1, l2, l3


# ============================================================================
# numba 版物理映射
# ============================================================================


@njit(cache=True)
def _map_tet_to_physical_nb(ref_pts, cell_nodes):
    n = ref_pts.shape[0]
    result = np.empty((n, 3))
    for p in range(n):
        a, b, c = ref_pts[p, 0], ref_pts[p, 1], ref_pts[p, 2]
        r, s, t = _cube_to_tet_rst_nb(a, b, c)
        L1, L2, L3, L4 = _tet_barycentric_nb(r, s, t)
        for d in range(3):
            result[p, d] = (
                L1 * cell_nodes[0, d] + L2 * cell_nodes[1, d]
                + L3 * cell_nodes[2, d] + L4 * cell_nodes[3, d]
            )
    return result


@njit(cache=True)
def _map_prism_to_physical_nb(ref_pts, cell_nodes):
    n = ref_pts.shape[0]
    result = np.empty((n, 3))
    for p in range(n):
        a, b, c = ref_pts[p, 0], ref_pts[p, 1], ref_pts[p, 2]
        r, s = _cube_to_tri_rs_nb(a, b)
        l1, l2, l3 = _tri_barycentric_nb(r, s)
        z = c
        for d in range(3):
            bottom = l1 * cell_nodes[0, d] + l2 * cell_nodes[1, d] + l3 * cell_nodes[2, d]
            top = l1 * cell_nodes[3, d] + l2 * cell_nodes[4, d] + l3 * cell_nodes[5, d]
            result[p, d] = 0.5 * (1.0 - z) * bottom + 0.5 * (1.0 + z) * top
    return result


@njit(cache=True)
def _map_ref_nb(is_prism, ref_pts, cell_nodes):
    if is_prism:
        return _map_prism_to_physical_nb(ref_pts, cell_nodes)
    return _map_tet_to_physical_nb(ref_pts, cell_nodes)


# ============================================================================
# numba 版参考面网格
# ============================================================================


@njit(cache=True)
def _face_ref_grid_nb(n1d, axis, side, sps_1d):
    n_fp = n1d * n1d
    pts = np.empty((n_fp, 3))
    o0, o1 = 0, 0
    idx = 0
    for a in range(3):
        if a != axis:
            if idx == 0:
                o0 = a
            else:
                o1 = a
            idx += 1
    for i in range(n1d):
        for j in range(n1d):
            flat = i * n1d + j
            pts[flat, axis] = side
            pts[flat, o0] = sps_1d[i]
            pts[flat, o1] = sps_1d[j]
    return pts


# ============================================================================
# numba 版 Newton 点位定位
# ============================================================================

_NEWTON_MAX_ITER = 50
_NEWTON_TOL_REL = 1e-10


@njit(cache=True)
def _tet_exact_locate_nb(cell_nodes, fixed_axis, fixed_val, targets_phys):
    """四面体面解析闭式解。"""
    if fixed_axis == 0 and fixed_val < 0.0:
        vi, vj, vk = 0, 2, 3
    elif fixed_axis == 0 and fixed_val > 0.0:
        vi, vj, vk = 1, 2, 3
    elif fixed_axis == 1 and fixed_val < 0.0:
        vi, vj, vk = 0, 1, 3
    else:
        vi, vj, vk = 0, 1, 2

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

        L = np.zeros(4)
        L[vi] = 1.0 - alpha - beta
        L[vj] = alpha
        L[vk] = beta

        r = 2.0 * L[1] - 1.0
        s = 2.0 * L[2] - 1.0
        t = 2.0 * L[3] - 1.0
        c_v = t
        db = 1.0 - c_v
        if abs(db) < 1e-300:
            db = 1e-300
        b_c = 2.0 * (s + 1.0) / db - 1.0
        da = s + t
        if abs(da) < 1e-300:
            da = 1e-300
        a_c = -2.0 * (r + 1.0) / da - 1.0

        full = np.empty(3)
        full[0] = a_c
        full[1] = b_c
        full[2] = c_v
        ix = 0
        for ax in range(3):
            if ax != fixed_axis:
                free[p, ix] = full[ax]
                ix += 1
    return free


@njit(cache=True)
def _newton_locate_nb(is_prism, cell_nodes, fixed_axis, fixed_val, targets_phys, char_length):
    """在单元面上 Newton 迭代定位目标物理点。四面体走闭式解，棱柱走迭代。

    Returns:
        (free_coords, resid_per_point)：resid_per_point 形状 (n_pts,)，
        每个目标点各自的最终物理残差（绝对长度单位），不在这里归约成
        单一标量——调用方（face_flux_points_merge.py）对多源棱柱四边形
        侧面（同一批 targets_phys 里混有真正属于该 cell、和落在对角线
        另一半、根本不在该 cell 面上的点）需要按半区各自的掩码分别取
        max，若在这里就归约成全批次的单一 max，两个半区的残差会被
        混在一起——落在"错误"那一半的目标点在这个 cell 的面上无论
        Newton 怎么迭代都不可能收敛（它们物理上属于对角线另一侧、
        映射到另一个真实相邻单元的面），残差会远超真实的双线性曲面
        翘曲量级，误把这当作整张面的残差会对多源面产生系统性误报。
    """
    o0, o1 = 0, 0
    ix = 0
    for ax in range(3):
        if ax != fixed_axis:
            if ix == 0:
                o0 = ax
            else:
                o1 = ax
            ix += 1

    n_pts = targets_phys.shape[0]
    scale = char_length if char_length > 1e-300 else 1e-300

    if not is_prism:
        x = _tet_exact_locate_nb(cell_nodes, fixed_axis, fixed_val, targets_phys)
        full = np.empty((n_pts, 3))
        for p in range(n_pts):
            full[p, fixed_axis] = fixed_val
            full[p, o0] = x[p, 0]
            full[p, o1] = x[p, 1]
        phys = _map_ref_nb(False, full, cell_nodes)
        resid_arr = np.empty(n_pts)
        for p in range(n_pts):
            dx = phys[p, 0] - targets_phys[p, 0]
            dy = phys[p, 1] - targets_phys[p, 1]
            dz = phys[p, 2] - targets_phys[p, 2]
            resid_arr[p] = np.sqrt(dx * dx + dy * dy + dz * dz)
        return x, resid_arr

    # 棱柱 Newton
    x = np.zeros((n_pts, 2))
    eps = 1e-6

    full = np.empty((n_pts, 3))
    for p in range(n_pts):
        full[p, fixed_axis] = fixed_val
        full[p, o0] = x[p, 0]
        full[p, o1] = x[p, 1]
    phys = _map_ref_nb(True, full, cell_nodes)
    rn = np.empty(n_pts)
    for p in range(n_pts):
        dx = phys[p, 0] - targets_phys[p, 0]
        dy = phys[p, 1] - targets_phys[p, 1]
        dz = phys[p, 2] - targets_phys[p, 2]
        rn[p] = np.sqrt(dx * dx + dy * dy + dz * dz)

    tol = 1e-13
    if _NEWTON_TOL_REL * scale > tol:
        tol = _NEWTON_TOL_REL * scale

    for _it in range(_NEWTON_MAX_ITER):
        mx = 0.0
        for p in range(n_pts):
            if rn[p] > mx:
                mx = rn[p]
        if mx < tol:
            break

        full = np.empty((n_pts, 3))
        for p in range(n_pts):
            full[p, fixed_axis] = fixed_val
            full[p, o0] = x[p, 0]
            full[p, o1] = x[p, 1]
        phys = _map_ref_nb(True, full, cell_nodes)

        res = np.empty((n_pts, 3))
        for p in range(n_pts):
            res[p, 0] = phys[p, 0] - targets_phys[p, 0]
            res[p, 1] = phys[p, 1] - targets_phys[p, 1]
            res[p, 2] = phys[p, 2] - targets_phys[p, 2]

        # 有限差分 Jacobi
        xp0 = x.copy(); xp0[:, 0] += eps
        fp0 = np.empty((n_pts, 3))
        for p in range(n_pts):
            fp0[p, fixed_axis] = fixed_val
            fp0[p, o0] = xp0[p, 0]; fp0[p, o1] = xp0[p, 1]
        pp0 = _map_ref_nb(True, fp0, cell_nodes)

        xp1 = x.copy(); xp1[:, 1] += eps
        fp1 = np.empty((n_pts, 3))
        for p in range(n_pts):
            fp1[p, fixed_axis] = fixed_val
            fp1[p, o0] = xp1[p, 0]; fp1[p, o1] = xp1[p, 1]
        pp1 = _map_ref_nb(True, fp1, cell_nodes)

        J0 = (pp0 - phys) / eps
        J1 = (pp1 - phys) / eps

        # Jacobi 预条件 + 正规方程 + 回溯线搜索
        J0n = np.empty(n_pts)
        J1n = np.empty(n_pts)
        for p in range(n_pts):
            J0n[p] = max(np.sqrt(J0[p, 0] ** 2 + J0[p, 1] ** 2 + J0[p, 2] ** 2), 1e-300)
            J1n[p] = max(np.sqrt(J1[p, 0] ** 2 + J1[p, 1] ** 2 + J1[p, 2] ** 2), 1e-300)

        dx = np.empty((n_pts, 2))
        for p in range(n_pts):
            j0h0, j0h1, j0h2 = J0[p, 0] / J0n[p], J0[p, 1] / J0n[p], J0[p, 2] / J0n[p]
            j1h0, j1h1, j1h2 = J1[p, 0] / J1n[p], J1[p, 1] / J1n[p], J1[p, 2] / J1n[p]
            a11 = j0h0 ** 2 + j0h1 ** 2 + j0h2 ** 2
            a12 = j0h0 * j1h0 + j0h1 * j1h1 + j0h2 * j1h2
            a22 = j1h0 ** 2 + j1h1 ** 2 + j1h2 ** 2
            b1 = -(j0h0 * res[p, 0] + j0h1 * res[p, 1] + j0h2 * res[p, 2])
            b2 = -(j1h0 * res[p, 0] + j1h1 * res[p, 1] + j1h2 * res[p, 2])
            det = a11 * a22 - a12 * a12
            ds = det if abs(det) > 1e-300 else 1e-300
            dx[p, 0] = (b1 * a22 - b2 * a12) / ds / J0n[p]
            dx[p, 1] = (a11 * b2 - a12 * b1) / ds / J1n[p]

        # 回溯线搜索。真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：
        # 此前分两段循环——第一段算试探点更新 xb/rb，第二段对全部点
        # 重新算一遍**相同的**试探点，用 rt<rb[p] 判断是否要减半步长；
        # 但对刚在第一段改进过的点，rb[p] 此时已经等于这次算出的 rt，
        # rt<rb[p] 恒为 False，导致刚成功改进的点在本轮也被错误地减半
        # 步长。与 Python 参考实现（face_flux_points_locate.py::
        # newton_locate_on_face）不一致——那里每次迭代只算一次试探残差，
        # 用同一次结果同时驱动 xb/rb 更新与 step 调整决策。这里改成同一
        # 结构：不影响最终收敛结果（历史最优 xb/rb 不会被撤销），但消除
        # 了对已改进点的多余步长收缩，在偏斜/退化单元上收敛更稳健。
        step = np.ones(n_pts)
        xb = x.copy()
        rb = rn.copy()
        for _ls in range(20):
            all_imp = True
            all_sml = True
            for p in range(n_pts):
                xt0 = x[p, 0] + step[p] * dx[p, 0]
                xt1 = x[p, 1] + step[p] * dx[p, 1]
                fp = np.empty((1, 3))
                fp[0, fixed_axis] = fixed_val; fp[0, o0] = xt0; fp[0, o1] = xt1
                pt = _map_ref_nb(True, fp, cell_nodes)
                ddx = pt[0, 0] - targets_phys[p, 0]
                ddy = pt[0, 1] - targets_phys[p, 1]
                ddz = pt[0, 2] - targets_phys[p, 2]
                rt = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                improved = rt < rb[p]
                if improved:
                    xb[p, 0] = xt0; xb[p, 1] = xt1; rb[p] = rt
                else:
                    all_imp = False
                    step[p] *= 0.5
                if step[p] >= 1e-8:
                    all_sml = False
            if all_imp or all_sml:
                break
        x = xb
        rn = rb

    return x, rn


# ============================================================================
# 查找表
# ============================================================================

# cube face code -> (axis, side) 查表。**必须覆盖全部 15 个编码**：numba
# nopython 不做边界检查，对 code>=len 索引会读到未定义内存（原生四面体
# 编码当年就踩过这条，见 `face_flux_points_exact_normal_kernel.py` 里那段
# "真实 bug 修复背景"）。
#
#   [0, 6)   坍缩立方体面 a=-1/a=+1/b=-1/b=+1/c=-1/c=+1
#   [6, 10)  原生四面体面 —— axis 槽位复用成 excluded_vertex、side 是哑值
#            （那条分支不走 (axis,side) 语义，见主 kernel 里的分派）
#   [10, 15) 原生棱柱面 —— 填的是**真实**的 (axis, side)：原生棱柱面的
#            通量点与坍缩立方体面的通量点已验证是同一批物理点、同一顺序，
#            所以定位仍走坍缩的 `_newton_locate_nb`，它需要真实的 axis/side。
#
# **由 `face_flux_points.py::CUBE_FACE_AXIS_SIDE` 逐项派生**（2026-09-19）。
# 此前这里是一份手抄的字面量数组，靠注释+一条测试"钉住两者不漂移" ——
# 而原生棱柱接入时正是从这类重复里漏出真实缺陷（`_PQ_CODES` 漏了原生那
# 三个侧面编码，`_PRISM_QUAD_CODES` 同样漏了，导致原生档第一次端到端
# 构造直接 `KeyError: (0, 14)`）。派生之后"漂移"在结构上不可能发生。
#
# 导入方向安全：`face_flux_points.py` 与 `face_flux_points_data.py` 都不
# import 本模块（只有 `face_flux_points_numba.py`/`_ms_numba.py` 会），
# 所以这里反向 import 不构成环。
_FACE_AXIS = np.zeros(len(CUBE_FACE_CODES), dtype=np.int32)
_FACE_SIDE = np.zeros(len(CUBE_FACE_CODES), dtype=np.float64)
for _name, _code in CUBE_FACE_CODES.items():
    _ax, _sd = CUBE_FACE_AXIS_SIDE[_name]
    _FACE_AXIS[_code] = _ax
    _FACE_SIDE[_code] = _sd
del _name, _code, _ax, _sd

#: **多源面**（棱柱的三个四边形侧面）的 cube face code —— 这类面在对侧
#: 被三角化成两个面，插值矩阵要走 multi-source kernel。坍缩编码是
#: a=-1/a=+1/b=-1 = 0/1/2，原生棱柱的对应面是 f3/f2/f4 = 13/12/14。
#: **漏掉原生那三个会让棱柱侧面走单源路径、静默拿到错的邻居插值**。
#:
#: 从 `face_flux_points_data.py::_PRISM_QUAD_CODES` 派生（那个集合是唯一
#: 定义；这里只是把它变成 numba 能索引的有序数组）。
_PQ_CODES = np.array(sorted(_PRISM_QUAD_CODES), dtype=np.int32)

#: 原生面编码区间（与 `grid/connectivity/face_connectivity.py` 的
#: `NATIVE_TET_FACE_CODE_RANGE`/`NATIVE_PRISM_FACE_CODE_RANGE` 同源）。
#: numba 分支判据统一用这两个常量，不要再写 `code >= 6` 这种字面量 ——
#: 加了棱柱编码之后那个字面量的含义从"是原生四面体面"变成了"是任意
#: 原生面"，而两者要分派到**不同**的算子组。
_NATIVE_TET_LO = 6
_NATIVE_TET_HI = 10
_NATIVE_PRISM_LO = 10
_NATIVE_PRISM_HI = 15

# ============================================================================
# native 四面体（路径C，非坍缩坐标）numba 版几何原语
#
# 与上面坍缩坐标（(axis,side)）版本平行——见
# `fr/native_simplex_basis.py`/`fr/face_flux_points_locate.py::
# locate_native_tet_face_point`/`fr/face_flux_points.py::
# native_tet_face_points_physical` 的纯 Python 参考实现，本节是它们的
# 逐行 numba 移植，供 `face_flux_points_numba.py::build_fp_newton_
# parallel` 内部对 native 四面体面（cube face code>=6，excluded_vertex=
# code-6）分支消费，避免退回到假设 (axis,side) 语义的 `_newton_locate_nb`/
# `_face_ref_grid_nb`。face code>=6 只可能出现在 native 四面体的真实面上
# （见 grid/connectivity/face_connectivity.py::with_native_face_codes
# 文档），棱柱四边形侧面恒为 0~5，因此本节所有分支判据只需要检查
# `code>=6`，不需要额外的 is_prism 判据。
# ============================================================================

# excluded_vertex(0~3) -> 该真实面 3 个顶点的局部下标（升序），与
# native_simplex_basis.py::face_node_indices/face_flux_points.py::
# native_tet_face_points_physical 的 `tuple(v for v in range(4) if v !=
# excluded_vertex)` 完全一致。
_NATIVE_FACE_VI = np.array([1, 0, 0, 0], dtype=np.int32)
_NATIVE_FACE_VJ = np.array([2, 2, 1, 1], dtype=np.int32)
_NATIVE_FACE_VK = np.array([3, 3, 3, 2], dtype=np.int32)


@njit(cache=True)
def _rst_to_abc_nb(r, s, t):
    """`native_simplex_basis.rst_to_abc` 逐行移植（numba 版，逐点标量
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
    """`native_simplex_basis.map_native_tet_to_physical` 逐行移植：重心
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
    """`face_flux_points.py::native_tet_face_points_physical` 逐行移植：
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
    """`face_flux_points_locate.py::locate_native_tet_face_point` 逐行
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
    `face_flux_points_ms_numba.py` 的 "primary interp" 分支使用：那里
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
