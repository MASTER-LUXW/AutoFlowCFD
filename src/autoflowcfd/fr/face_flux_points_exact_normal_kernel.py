"""Numba 并行 kernel：精确逐面法向量/adj(J) 行计算（2026-08-24 性能优化）。

将 face_flux_points_exact_normal.py 中 compute_exact_adj_rows 的核心计算
（逐面类型桶循环 + 批量 Jacobian/det/inv）替换为单个 numba prange 并行 kernel。

性能瓶颈（优化前）：
    compute_exact_adj_rows 对 188 万个面按 (axis, side, cell_type) 分成 ~10 个桶，
    每个桶用纯 NumPy 批量计算 Jacobian (k, n_fp, 3, 3) → det → inv → adj_row。
    P2 时 n_fp=9，中间数组 J 约 1.2 GB，det/inv 各需等量临时内存，峰值 ~5 GB。
    纯 Python 桶循环 + 巨量 NumPy 临时数组导致 P2 初始化卡住 10+ 分钟。

优化策略：
    1. 单个 prange kernel 遍历全部 n_faces，每线程处理一个面全部 n_fp 个通量点
    2. 手动 inline 3×3 det/inv/adj（numba 不支持 np.linalg.det/inv）
    3. 手动 inline tet/prism 精确 Jacobian 闭式公式（消除函数调用开销）
    4. 逐 (face, fp) 计算 → 无巨量中间数组，内存 O(n_faces × n_fp) 直接输出
    5. 预计算所有 (axis, side) 组合的参考坐标网格（6 种 × n_fp × 3），
       kernel 内通过查表访问，避免逐面重新构建

与原版数值一致性：
    使用完全相同的数学公式（tet/prism 精确 Jacobian 闭式 + Duffy 变换），
    只是计算顺序从"批量 NumPy"改为"逐面 numba 并行"，浮点结果应逐位一致。
    tests/unit/test_face_flux_points_exact_normal.py 验证。
"""

import numpy as np
from numba import njit, prange


# ======================================================================
# 3×3 线性代数内联辅助函数
# ======================================================================

@njit(inline='always')
def _det3(m):
    """3×3 矩阵行列式（手动展开，无 LAPACK 依赖）。"""
    return (m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]))


@njit(inline='always')
def _adj3(m):
    """3×3 矩阵伴随矩阵（adjugate / classical adjugate）。

    adj(J) = det(J) * inv(J)，即余因子矩阵的转置。
    直接计算 adj(J) 而非先求 inv 再乘 det，减少一次矩阵运算。
    """
    adj = np.empty((3, 3), dtype=np.float64)
    adj[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    adj[0, 1] = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]
    adj[0, 2] = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
    adj[1, 0] = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
    adj[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
    adj[1, 2] = m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]
    adj[2, 0] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    adj[2, 1] = m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]
    adj[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    return adj


# ======================================================================
# 精确 Jacobian 内联计算（与 batched 版数学等价）
# ======================================================================

@njit(inline='always')
def _tet_jac_at(a, b, c, p0, p1, p2, p3):
    """直边四面体在单个参考坐标 (a,b,c) 处的精确 Jacobian。

    与 _tet_exact_jacobian_batched 在单点处的值逐位一致。
    """
    e1_0 = (p1[0] - p0[0]) * 0.5
    e1_1 = (p1[1] - p0[1]) * 0.5
    e1_2 = (p1[2] - p0[2]) * 0.5
    e2_0 = (p2[0] - p0[0]) * 0.5
    e2_1 = (p2[1] - p0[1]) * 0.5
    e2_2 = (p2[2] - p0[2]) * 0.5
    e3_0 = (p3[0] - p0[0]) * 0.5
    e3_1 = (p3[1] - p0[1]) * 0.5
    e3_2 = (p3[2] - p0[2]) * 0.5

    s = (1.0 + b) * (1.0 - c) * 0.5 - 1.0
    t = c

    jac = np.empty((3, 3), dtype=np.float64)

    coef_a = -(s + t) * 0.5
    jac[0, 0] = coef_a * e1_0
    jac[1, 0] = coef_a * e1_1
    jac[2, 0] = coef_a * e1_2

    coef_b1 = -(1.0 + a) * (1.0 - c) * 0.25
    coef_b2 = (1.0 - c) * 0.5
    jac[0, 1] = coef_b1 * e1_0 + coef_b2 * e2_0
    jac[1, 1] = coef_b1 * e1_1 + coef_b2 * e2_1
    jac[2, 1] = coef_b1 * e1_2 + coef_b2 * e2_2

    coef_c1 = -(1.0 + a) * (1.0 - b) * 0.25
    coef_c2 = -(1.0 + b) * 0.5
    jac[0, 2] = coef_c1 * e1_0 + coef_c2 * e2_0 + e3_0
    jac[1, 2] = coef_c1 * e1_1 + coef_c2 * e2_1 + e3_1
    jac[2, 2] = coef_c1 * e1_2 + coef_c2 * e2_2 + e3_2

    return jac


@njit(inline='always')
def _prism_jac_at(a, b, c, p0, p1, p2, p3, p4, p5):
    """直边棱柱在单个参考坐标 (a,b,c) 处的精确 Jacobian。

    与 _prism_exact_jacobian_batched 在单点处的值逐位一致。
    """
    # bottom/top 对 a 的偏导
    fdb_0 = (1.0 - b) * 0.25 * (p1[0] - p0[0])
    fdb_1 = (1.0 - b) * 0.25 * (p1[1] - p0[1])
    fdb_2 = (1.0 - b) * 0.25 * (p1[2] - p0[2])
    fdt_0 = (1.0 - b) * 0.25 * (p4[0] - p3[0])
    fdt_1 = (1.0 - b) * 0.25 * (p4[1] - p3[1])
    fdt_2 = (1.0 - b) * 0.25 * (p4[2] - p3[2])

    # bottom/top 对 b 的偏导
    fddb_0 = (-(1.0 - a) * 0.25 * p0[0] - (1.0 + a) * 0.25 * p1[0] + 0.5 * p2[0])
    fddb_1 = (-(1.0 - a) * 0.25 * p0[1] - (1.0 + a) * 0.25 * p1[1] + 0.5 * p2[1])
    fddb_2 = (-(1.0 - a) * 0.25 * p0[2] - (1.0 + a) * 0.25 * p1[2] + 0.5 * p2[2])
    fdtb_0 = (-(1.0 - a) * 0.25 * p3[0] - (1.0 + a) * 0.25 * p4[0] + 0.5 * p5[0])
    fdtb_1 = (-(1.0 - a) * 0.25 * p3[1] - (1.0 + a) * 0.25 * p4[1] + 0.5 * p5[1])
    fdtb_2 = (-(1.0 - a) * 0.25 * p3[2] - (1.0 + a) * 0.25 * p4[2] + 0.5 * p5[2])

    # 重心坐标
    r = (1.0 + a) * (1.0 - b) * 0.5 - 1.0
    s = b
    l1 = -(r + s) * 0.5
    l2 = (1.0 + r) * 0.5
    l3 = (1.0 + s) * 0.5

    bot_0 = l1 * p0[0] + l2 * p1[0] + l3 * p2[0]
    bot_1 = l1 * p0[1] + l2 * p1[1] + l3 * p2[1]
    bot_2 = l1 * p0[2] + l2 * p1[2] + l3 * p2[2]
    top_0 = l1 * p3[0] + l2 * p4[0] + l3 * p5[0]
    top_1 = l1 * p3[1] + l2 * p4[1] + l3 * p5[1]
    top_2 = l1 * p3[2] + l2 * p4[2] + l3 * p5[2]

    cm = 0.5
    cp = 0.5
    jac = np.empty((3, 3), dtype=np.float64)

    jac[0, 0] = cm * (1.0 - c) * fdb_0 + cp * (1.0 + c) * fdt_0
    jac[1, 0] = cm * (1.0 - c) * fdb_1 + cp * (1.0 + c) * fdt_1
    jac[2, 0] = cm * (1.0 - c) * fdb_2 + cp * (1.0 + c) * fdt_2

    jac[0, 1] = cm * (1.0 - c) * fddb_0 + cp * (1.0 + c) * fdtb_0
    jac[1, 1] = cm * (1.0 - c) * fddb_1 + cp * (1.0 + c) * fdtb_1
    jac[2, 1] = cm * (1.0 - c) * fddb_2 + cp * (1.0 + c) * fdtb_2

    jac[0, 2] = 0.5 * (top_0 - bot_0)
    jac[1, 2] = 0.5 * (top_1 - bot_1)
    jac[2, 2] = 0.5 * (top_2 - bot_2)

    return jac


# ======================================================================
# 主 kernel：compute_exact_adj_rows 的 numba 并行版
# ======================================================================

@njit(parallel=True, cache=True)
def compute_exact_adj_rows_kernel(
    n_faces,
    n_fp,
    n_prism,
    cell_arr,
    axis_arr,
    side_arr,
    prism_conn,
    tet_conn,
    node_coords,
    valid_mask,
    ref_pts_table,
    adj_row_out,
):
    """compute_exact_adj_rows 的 numba 并行 kernel。

    每个线程处理一个面，遍历该面全部 n_fp 个通量点，
    内联计算 Jacobian → adj(J) → 提取 axis 行写入输出。

    Args:
        ref_pts_table: (6, n_fp, 3)，预计算的参考坐标网格。
            索引 = axis * 2 + (0 if side < 0 else 1)
    """
    for f in prange(n_faces):
        if not valid_mask[f]:
            continue

        cell = cell_arr[f]
        ax = axis_arr[f]
        sd = side_arr[f]
        is_prism = cell < n_prism

        # 查表获取该面的参考坐标网格
        sd_idx = 0 if sd < 0.0 else 1
        table_idx = ax * 2 + sd_idx

        # 获取角点物理坐标
        if is_prism:
            p0 = node_coords[prism_conn[cell, 0]]
            p1 = node_coords[prism_conn[cell, 1]]
            p2 = node_coords[prism_conn[cell, 2]]
            p3 = node_coords[prism_conn[cell, 3]]
            p4 = node_coords[prism_conn[cell, 4]]
            p5 = node_coords[prism_conn[cell, 5]]
        else:
            tc = cell - n_prism
            p0 = node_coords[tet_conn[tc, 0]]
            p1 = node_coords[tet_conn[tc, 1]]
            p2 = node_coords[tet_conn[tc, 2]]
            p3 = node_coords[tet_conn[tc, 3]]

        # 逐通量点计算 Jacobian → adj(J) → 提取 axis 行
        for fp in range(n_fp):
            a = ref_pts_table[table_idx, fp, 0]
            b = ref_pts_table[table_idx, fp, 1]
            c = ref_pts_table[table_idx, fp, 2]

            if is_prism:
                J = _prism_jac_at(a, b, c, p0, p1, p2, p3, p4, p5)
            else:
                J = _tet_jac_at(a, b, c, p0, p1, p2, p3)

            adj = _adj3(J)
            adj_row_out[f, fp, 0] = adj[ax, 0]
            adj_row_out[f, fp, 1] = adj[ax, 1]
            adj_row_out[f, fp, 2] = adj[ax, 2]


# ======================================================================
# Python 封装：预计算参考坐标表 + 调用 kernel
# ======================================================================

def precompute_ref_pts_table(n1d, sps_1d):
    """预计算所有 (axis, side) 组合的参考坐标网格。

    Returns:
        ref_pts_table: (6, n_fp, 3) 数组。
        索引方式: table[axis * 2 + (0 if side<0 else 1)]
    """
    n_fp = n1d * n1d
    table = np.zeros((6, n_fp, 3), dtype=np.float64)
    for axis in range(3):
        for side_idx, side in enumerate([-1.0, 1.0]):
            other_axes = [a for a in range(3) if a != axis]
            g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
            pts = np.zeros((n_fp, 3), dtype=np.float64)
            pts[:, axis] = side
            pts[:, other_axes[0]] = g1.ravel()
            pts[:, other_axes[1]] = g2.ravel()
            table[axis * 2 + side_idx] = pts
    return table


def compute_exact_adj_rows_fast(
    n_faces, n1d, sps_1d, n_prism,
    cell_arr, axis_arr, side_arr,
    prism_conn, tet_conn, node_coords,
    valid_mask=None,
):
    """compute_exact_adj_rows 的 numba 加速版（接口与原版完全一致）。

    用单个 prange kernel 替代 Python 桶循环 + 批量 NumPy，
    内存从 O(n_faces × n_fp × 27)（中间 Jacobian 数组）
    降到 O(n_faces × n_fp × 3)（仅输出数组）。
    """
    n_fp = n1d * n1d
    adj_row_out = np.zeros((n_faces, n_fp, 3), dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.ones(n_faces, dtype=np.bool_)

    ref_pts_table = precompute_ref_pts_table(n1d, sps_1d)

    compute_exact_adj_rows_kernel(
        n_faces, n_fp, n_prism,
        cell_arr, axis_arr, side_arr,
        prism_conn, tet_conn, node_coords,
        valid_mask, ref_pts_table, adj_row_out,
    )
    return adj_row_out
