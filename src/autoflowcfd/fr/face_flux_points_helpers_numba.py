"""
AutoFlowCFD - FP 几何构建 numba 辅助函数

坐标变换、物理映射、参考面网格、Newton 迭代定位和查找表。
被 face_flux_points_numba（主 kernel）和 face_flux_points_ms_numba
（multi-source kernel）共同依赖。
"""

import numpy as np
from numba import njit


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
    """在单元面上 Newton 迭代定位目标物理点。四面体走闭式解，棱柱走迭代。"""
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
        mr = 0.0
        for p in range(n_pts):
            dx = phys[p, 0] - targets_phys[p, 0]
            dy = phys[p, 1] - targets_phys[p, 1]
            dz = phys[p, 2] - targets_phys[p, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r > mr:
                mr = r
        return x, mr

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

        # 回溯线搜索
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
                if rt < rb[p]:
                    xb[p, 0] = xt0; xb[p, 1] = xt1; rb[p] = rt
                else:
                    all_imp = False
                if step[p] >= 1e-8:
                    all_sml = False
            if all_imp or all_sml:
                break
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
                if not (rt < rb[p]):
                    step[p] *= 0.5
        x = xb
        rn = rb

    fr = 0.0
    for p in range(n_pts):
        if rn[p] > fr:
            fr = rn[p]
    return x, fr


# ============================================================================
# 查找表
# ============================================================================

_FACE_AXIS = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
_FACE_SIDE = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_PQ_CODES = np.array([0, 1, 2], dtype=np.int32)  # a=-1, a=+1, b=-1
