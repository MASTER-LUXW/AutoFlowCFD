"""
AutoFlowCFD - FR Flux Points 精确点位定位（numpy 版；生产构造用 numba 版 `ref_geometry_nb.py`/`native_geometry_nb.py`）。

对 owner 的每个 FP 物理坐标，在 neighbor 的对应局部面上反解出 neighbor 的精确参考坐标，使映射恰好给出该
物理点——四面体面走原生参考坐标下的解析闭式解（`locate_native_tet_face_point`），棱柱面走向量化批量 Newton
迭代（棱柱侧面是真双线性曲面，没有闭式解；在立方体坐标里定位，与原生映射逐位恒等）。不假设两侧网格对齐。
"""

from typing import Tuple

import numpy as np

from autoflowcfd.grid.curved_mapping.curved_mapping import map_prism_to_physical

_NEWTON_MAX_ITER = 50
_NEWTON_TOL_REL = 1e-10  # 收敛判据：相对局部面特征尺度（不是绝对长度单位）

# 容忍阈值：相对局部面特征尺度，超过则视为真正的定位失败（报错中止，而非
# 静默放行）。取值有真实网格数据支撑，不是拍脑袋：在 cube_demo 网格
# （545597 单元）上直接、独立于 Newton 之外，对全部 322758 个棱柱四边形
# 侧面算了"第4个角点到其余3点所在平面的距离/四边形对角线长度"这个翘曲度
# 量——这是棱柱侧面（双线性曲面）与相邻四面体（平面三角形）只能在角点
# 严格重合、内部存在真实几何偏差的直接度量。结果：99% 的四边形翘曲度
# <2.87%，全网格最大值 11.13%（0 个超过 15%）。Newton 最小二乘解已经是
# 该目标点在双线性曲面上能达到的最优逼近，这个偏差是网格本身的固有几何
# 特征（混合棱柱/四面体网格在等参 FR 框架下的截断误差，类比任何数值
# 离散化都有的截断误差），不是算法缺陷；15% 的阈值留有余量覆盖这一真实
# 分布的同时，仍能可靠地把"真正不相容/断裂的面"（残差应远超此量级，
# 通常是数量级的差异）与"翘曲但合法的面"区分开——超过阈值报错中止而不是
# 静默放宽，接受阈值内的情形也会被完整记录（见 face_flux_points/merge.py
# 的 tolerated 汇总日志），不是简化或掩盖。
_ACCEPT_WARN_REL = 0.15


def _tet_solve_barycentric_on_face(
    cell_nodes: np.ndarray, face_vertex_idx: Tuple[int, int, int], targets_phys: np.ndarray
) -> np.ndarray:
    """给定四面体某个真实面（由 3 个局部顶点下标确定，第 4 个顶点的重心坐标恒为 0），在物理空间解出一批目标点
    的重心坐标 L1..L4。

    "3 点确定一个平面，求平面内一点的重心坐标"恒为良态的 3×2 最小二乘问题——条件数只取决于这 3 点自身构成的
    三角形的边长比。

    Returns:
        L: (n_pts, 4) 重心坐标
    """
    i, j, k = face_vertex_idx
    P_i, P_j, P_k = cell_nodes[i], cell_nodes[j], cell_nodes[k]
    e1 = P_j - P_i
    e2 = P_k - P_i
    rhs = targets_phys - P_i[None, :]  # (n_pts,3)

    # 与 Newton 迭代内部同一套 Jacobi 预条件手法（条件数只取决于三角形自身边长比）。
    e1_norm = max(float(np.linalg.norm(e1)), 1e-300)
    e2_norm = max(float(np.linalg.norm(e2)), 1e-300)
    e1_hat = e1 / e1_norm
    e2_hat = e2 / e2_norm
    a11 = float(np.dot(e1_hat, e1_hat))
    a12 = float(np.dot(e1_hat, e2_hat))
    a22 = float(np.dot(e2_hat, e2_hat))
    b1 = rhs @ e1_hat
    b2 = rhs @ e2_hat
    det = a11 * a22 - a12 * a12
    det_safe = det if abs(det) > 1e-300 else 1e-300
    alpha = (b1 * a22 - b2 * a12) / det_safe / e1_norm
    beta = (a11 * b2 - a12 * b1) / det_safe / e2_norm

    n_pts = targets_phys.shape[0]
    L = np.zeros((n_pts, 4))
    L[:, i] = 1.0 - alpha - beta
    L[:, j] = alpha
    L[:, k] = beta
    return L


def locate_native_tet_face_point(
    cell_nodes: np.ndarray, excluded_vertex: int, targets_phys: np.ndarray, char_length: float = 1.0
) -> Tuple[np.ndarray, float]:
    """四面体某个真实面（对面被排除的局部顶点 `excluded_vertex` 给定）上一批目标物理点的精确点位定位，直接给出
    它们在四面体**原生参考坐标** `(r,s,t)` 里的位置：解出重心坐标 L1..L4 后 `r=2*L2-1, s=2*L3-1, t=2*L4-1`
    （不经过立方体坐标 `(a,b,c)`，没有含显式除法的退化轴转换）。

    Returns:
        (rst, final_resid)：rst 形状 (n_pts, 3)；final_resid 是该批点的最大物理残差（绝对长度单位），与
        `newton_locate_on_prism_face` 同一惯例，供调用方按 `char_length` 分级处理。
    """
    from ...grid.curved_mapping.curved_mapping import tet_barycentric

    face_vertex_idx = tuple(v for v in range(4) if v != excluded_vertex)
    L = _tet_solve_barycentric_on_face(cell_nodes, face_vertex_idx, targets_phys)

    r = 2.0 * L[:, 1] - 1.0
    s = 2.0 * L[:, 2] - 1.0
    t = 2.0 * L[:, 3] - 1.0
    rst = np.column_stack([r, s, t])

    L1, L2, L3, L4 = tet_barycentric(r, s, t)
    p0, p1, p2, p3 = cell_nodes
    phys_check = L1[:, None] * p0 + L2[:, None] * p1 + L3[:, None] * p2 + L4[:, None] * p3
    final_resid = float(np.max(np.linalg.norm(phys_check - targets_phys, axis=1)))

    scale = max(char_length, 1e-300)
    warn_tol = max(1e-9, _ACCEPT_WARN_REL * scale)
    if final_resid > warn_tol:
        raise RuntimeError(
            f"Native tet face exact point-location failed to converge: max residual {final_resid:.3e} "
            f"({100 * final_resid / scale:.2f}% of local face scale {scale:.3e}). "
            f"This indicates a genuinely non-conforming mesh face (target point not actually "
            f"coplanar with this tet's face) rather than a numerical precision issue."
        )
    return rst, final_resid


def newton_locate_on_prism_face(
    cell_nodes: np.ndarray,
    fixed_axis: int,
    fixed_val: float,
    targets_phys: np.ndarray,
    char_length: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """在棱柱单元立方体坐标的固定 fixed_axis=fixed_val 局部面上，为一批目标物理点 (n_pts,3) 定位另外两个
    自由坐标，使映射结果与目标重合。棱柱侧面是真双线性曲面，没有闭式解，走向量化批量 Newton 迭代（有限差分
    雅可比）——映射在给定面上光滑且（对非退化单元）局部可逆，通常 <5 次迭代到机器精度。

    Returns:
        (free_coords, final_resid): free_coords 形状 (n_pts, 2)（两个自由
        方向的坐标，顺序 = other_axes 升序）；final_resid 为该批点的最大
        物理残差（绝对长度单位），供调用方按 char_length 分级处理。
    """
    other_axes = [a for a in range(3) if a != fixed_axis]
    n_pts = targets_phys.shape[0]
    scale = max(char_length, 1e-300)

    def full_points(xy: np.ndarray) -> np.ndarray:
        pts = np.zeros((n_pts, 3))
        pts[:, fixed_axis] = fixed_val
        pts[:, other_axes[0]] = xy[:, 0]
        pts[:, other_axes[1]] = xy[:, 1]
        return pts

    x = np.zeros((n_pts, 2))  # 初值取面中心 (0,0)
    eps = 1e-6
    resid_norm = np.linalg.norm(map_prism_to_physical(full_points(x), cell_nodes) - targets_phys, axis=1)
    tol_converge = max(1e-13, _NEWTON_TOL_REL * scale)

    for _ in range(_NEWTON_MAX_ITER):
        if np.max(resid_norm) < tol_converge:
            break

        phys = map_prism_to_physical(full_points(x), cell_nodes)
        residual = phys - targets_phys  # (n_pts,3)

        x_p0 = x.copy()
        x_p0[:, 0] += eps
        x_p1 = x.copy()
        x_p1[:, 1] += eps
        phys_p0 = map_prism_to_physical(full_points(x_p0), cell_nodes)
        phys_p1 = map_prism_to_physical(full_points(x_p1), cell_nodes)
        J0 = (phys_p0 - phys) / eps  # (n_pts,3) d(phys)/d(x0)
        J1 = (phys_p1 - phys) / eps  # (n_pts,3) d(phys)/d(x1)

        # 逐点最小二乘解 2x2 正规方程 (J^T J) dx = -J^T residual（3方程2未知量，
        # 对确实落在该面上的目标点是相容超定系统，最小二乘解=精确解）。
        # 求解前先把 J0,J1 各自按列归一化（Jacobi 预条件）：偏斜（对角线
        # 方向远长于另一方向）单元的两个参考方向映射到物理空间的尺度可以
        # 相差数倍，直接对原始 J0,J1 形成正规方程会把这个尺度差平方，
        # 显著放大条件数（对四面体面这种严格仿射映射，理论上一步就该精确
        # 收敛，条件数差导致的舍入误差却会把残差顶在 1e-3 量级的精度地板
        # 上、永远到不了收敛判据——已在真实网格的偏斜单元上复现、验证
        # 该修复后收敛到机器精度）。归一化后 a11=a22=1，det=1-a12^2=
        # sin^2(J0,J1 夹角)，只在两个参考方向映射到几乎平行的物理方向时
        # 才会真正病态（这才是物理上无法定位，而不是数值精度问题）。
        J0_norm = np.maximum(np.linalg.norm(J0, axis=1), 1e-300)
        J1_norm = np.maximum(np.linalg.norm(J1, axis=1), 1e-300)
        J0_hat = J0 / J0_norm[:, None]
        J1_hat = J1 / J1_norm[:, None]

        a11 = np.einsum("pi,pi->p", J0_hat, J0_hat)
        a12 = np.einsum("pi,pi->p", J0_hat, J1_hat)
        a22 = np.einsum("pi,pi->p", J1_hat, J1_hat)
        b1 = -np.einsum("pi,pi->p", J0_hat, residual)
        b2 = -np.einsum("pi,pi->p", J1_hat, residual)

        det = a11 * a22 - a12 * a12
        det_safe = np.where(np.abs(det) < 1e-300, 1e-300, det)
        dx0_hat = (b1 * a22 - b2 * a12) / det_safe
        dx1_hat = (a11 * b2 - a12 * b1) / det_safe
        dx = np.column_stack([dx0_hat / J0_norm, dx1_hat / J1_norm])

        # 回溯线搜索（逐点独立）：棱柱侧面是双线性曲面，满步 Newton 对强
        # 非线性/畸变单元可能过冲甚至发散，用简单的按点减半步长直到残差
        # 下降，比放宽收敛容差更能保证物理正确性（不掩盖真正的不相容面）。
        step = np.ones(n_pts)
        x_best, resid_best = x, resid_norm
        for _ls in range(20):
            x_trial = x + step[:, None] * dx
            resid_trial = np.linalg.norm(
                map_prism_to_physical(full_points(x_trial), cell_nodes) - targets_phys, axis=1
            )
            improved = resid_trial < resid_best
            x_best = np.where(improved[:, None], x_trial, x_best)
            resid_best = np.where(improved, resid_trial, resid_best)
            if np.all(improved) or np.all(step < 1e-8):
                break
            step = np.where(improved, step, step * 0.5)
        x, resid_norm = x_best, resid_best

    final_resid = np.max(resid_norm)
    warn_tol = max(1e-9, _ACCEPT_WARN_REL * scale)
    if final_resid > warn_tol:
        raise RuntimeError(
            f"Newton face point-location failed to converge: max residual {final_resid:.3e} "
            f"({100 * final_resid / scale:.2f}% of local face scale {scale:.3e}). "
            f"This indicates a genuinely non-conforming mesh face (target point not actually "
            f"on this cell's face) rather than a slow-convergence issue."
        )
    return x, final_resid
