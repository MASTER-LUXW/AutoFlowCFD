"""Numba 并行 kernel：精确逐面法向量/adj(J) 行计算（2026-08-24 性能优化）。

将 face_flux_points/exact_normal.py 中 compute_exact_adj_rows 的核心计算
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

逐点的 Jacobian / adj 行原语已拆到 `exact_normal_primitives.py`
（2026-09-19，项目"单文件不超 500 行"规范），本文件只留 prange kernel
与它的 Python 封装（参考点表预计算 + dtype 归一化）。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.grid.connectivity.face_connectivity import (
    NATIVE_PRISM_FACE_CODE_RANGE,
    NATIVE_TET_FACE_CODE_RANGE,
)
from .exact_normal_primitives import (
    _adj3,
    _native_prism_adj_row_at,
    _native_tet_adj_row_at,
    _prism_jac_at,
    _tet_jac_at,
)


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
    code_arr,
    n1d,
    sps_1d,
    np_ref_pts_table,
    np_cov,
    np_duffy,
):
    """compute_exact_adj_rows 的 numba 并行 kernel。

    每个线程处理一个面，遍历该面全部 n_fp 个通量点，
    内联计算 Jacobian → adj(J) → 提取 axis 行写入输出。

    Args:
        ref_pts_table: (6, n_fp, 3)，预计算的参考坐标网格。
            索引 = axis * 2 + (0 if side < 0 else 1)
        code_arr: (n_faces,) 原始 cube face 编码，>=6 即 native 四面体
            真实面（excluded_vertex=code-6）——真实 bug 修复（见
            `_native_tet_adj_row_at` 文档）：`axis_arr`/`side_arr` 对
            native 面存的是复用的 excluded_vertex/哑值，**不能**像
            此前那样直接当 (axis,side) 用于 `table_idx` 查表——
            `excluded_vertex` 可以是 0~3，`table_idx=ax*2+sd_idx`
            在 `excluded_vertex==3` 时越界读取只有 6 行的
            `ref_pts_table`（numba 不做边界检查，读到未定义内存），
            `excluded_vertex<3` 时也会被当成错误的坍缩坐标 (axis,side)
            语义算出完全错误的 Jacobian，必须用这个原始编码分派到
            `_native_tet_adj_row_at` 单独处理。
        n1d, sps_1d: native 分支需要按 (a,b) 参考坐标（`flat=i*n1d+j`
            约定，与 `native_tet_face_points_physical`/
            `_native_tet_face_points_nb` 同一套采样顺序）算出每个
            fp 的 (a,b)，不能复用 `ref_pts_table`（那是坍缩坐标 6 个
            面专属，不含 native 面的参考坐标含义）。
    """
    for f in prange(n_faces):
        if not valid_mask[f]:
            continue

        cell = cell_arr[f]
        code = code_arr[f]

        if code >= _NP_LO:
            # 原生棱柱真实面（编码 [10,15)）：**判据必须是区间**——写成
            # `code >= 6` 会把它们送进下面的四面体分支、按 `code - 6` 取到
            # excluded_vertex 4~8，而那个函数只认 0~3，numba nopython 不做
            # 边界检查，会读到未定义内存或静默给出完全错误的法向。
            fid = code - _NP_LO
            cov0 = np_cov[fid, 0]
            cov1 = np_cov[fid, 1]
            cov2 = np_cov[fid, 2]
            q0 = node_coords[prism_conn[cell, 0]]
            q1 = node_coords[prism_conn[cell, 1]]
            q2 = node_coords[prism_conn[cell, 2]]
            q3 = node_coords[prism_conn[cell, 3]]
            q4 = node_coords[prism_conn[cell, 4]]
            q5 = node_coords[prism_conn[cell, 5]]
            for fp in range(n_fp):
                r0, r1, r2 = _native_prism_adj_row_at(
                    cov0, cov1, cov2, np_duffy[fid, fp],
                    np_ref_pts_table[fid, fp, 0],
                    np_ref_pts_table[fid, fp, 1],
                    np_ref_pts_table[fid, fp, 2],
                    q0, q1, q2, q3, q4, q5)
                adj_row_out[f, fp, 0] = r0
                adj_row_out[f, fp, 1] = r1
                adj_row_out[f, fp, 2] = r2
            continue

        if code >= _NT_LO:
            # native 四面体真实面（编码 [6,10)）：直边常数 Jacobian（性质上
            # 不依赖参考坐标），但面法向"原始向量"（未归一化）本身随采样点
            # (a,b) 变化（见 _native_tet_adj_row_at/_native_tet_adj_row_
            # batched 文档——坍缩三角形采样引入的非线性，不是常数捷径能
            # 处理的）。
            tc = cell - n_prism
            p0 = node_coords[tet_conn[tc, 0]]
            p1 = node_coords[tet_conn[tc, 1]]
            p2 = node_coords[tet_conn[tc, 2]]
            p3 = node_coords[tet_conn[tc, 3]]
            ev = code - _NT_LO
            for fp in range(n_fp):
                i = fp // n1d
                j = fp % n1d
                a = sps_1d[i]
                b = sps_1d[j]
                r0, r1, r2 = _native_tet_adj_row_at(ev, a, b, p0, p1, p2, p3)
                adj_row_out[f, fp, 0] = r0
                adj_row_out[f, fp, 1] = r1
                adj_row_out[f, fp, 2] = r2
            continue

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

#: 原生面编码区间（唯一事实来源在
#: `grid/connectivity/face_connectivity.py`，这里做模块级常量以便 numba
#: kernel 里用具名判据而不是 `>= 6` 这种字面量）。
_NT_LO = NATIVE_TET_FACE_CODE_RANGE[0]
_NP_LO = NATIVE_PRISM_FACE_CODE_RANGE[0]


def precompute_native_prism_face_tables(order):
    """预计算原生棱柱 5 个面的 `(参考点, 余向量, Duffy 因子)` 三张表。

    Returns:
        `(ref_pts (5,n_fp,3), cov (5,3), duffy (5,n_fp))`。三张表都从
        `fr/native_prism/face.py` 里那份唯一定义派生（面点生成器与
        `_FACE_REF_COVECTOR`），不在这里重抄公式。
    """
    from ..native_prism.face import (
        PRISM_FACE_IDS,
        _FACE_REF_COVECTOR,
        native_prism_face_points,
    )

    n_faces_np = len(PRISM_FACE_IDS)
    n_fp = (order + 1) ** 2
    ref_pts = np.zeros((n_faces_np, n_fp, 3), dtype=np.float64)
    cov = np.zeros((n_faces_np, 3), dtype=np.float64)
    duffy = np.ones((n_faces_np, n_fp), dtype=np.float64)
    for fid in PRISM_FACE_IDS:
        pts = native_prism_face_points(order, fid)
        ref_pts[fid] = pts
        c, is_cap = _FACE_REF_COVECTOR[fid]
        cov[fid] = c
        if is_cap:
            duffy[fid] = (1.0 - pts[:, 1]) / 2.0
    return ref_pts, cov, duffy


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
    code_arr=None,
):
    """compute_exact_adj_rows 的 numba 加速版（接口与原版基本一致，
    新增可选 `code_arr` 供 native 四面体分派，见 `compute_exact_adj_rows_
    kernel`/`_native_tet_adj_row_at` 文档"真实 bug 修复"说明）。

    用单个 prange kernel 替代 Python 桶循环 + 批量 NumPy，
    内存从 O(n_faces × n_fp × 27)（中间 Jacobian 数组）
    降到 O(n_faces × n_fp × 3)（仅输出数组）。

    Args:
        code_arr: (n_faces,) 或 None——原始 cube face 编码。None 时
            （既有调用点未显式传入的情形）视为全部走坍缩坐标分支，
            与本次修复前完全一致的行为（传一个全 -1 数组，恒小于 6，
            `is_native` 恒为 False）。
    """
    n_fp = n1d * n1d
    adj_row_out = np.zeros((n_faces, n_fp, 3), dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.ones(n_faces, dtype=np.bool_)
    if code_arr is None:
        code_arr = np.full(n_faces, -1, dtype=np.int64)

    # ===== 两张连接表必须**同一个**整数 dtype（真实段错误修复） =====
    #
    # kernel 里 `p0 = node_coords[prism_conn[cell, 0]]` 与
    # `p0 = node_coords[tet_conn[tc, 0]]` 是同一个变量的 if/else 两支。
    # 两张表的索引 dtype 不同时，`parallel=True` 下的 numba 会生成**直接
    # 段错误**的代码（不是异常、没有任何 Python 栈，只有一串
    # "Windows fatal exception: access violation"）。
    #
    # 生产里这个组合真实存在且**必然命中**：`face_flux_points/merge.py`
    # 在 `mesh._fixed_prism_conn is None`（纯四面体网格）时传的兜底是
    # `np.empty((0, 6), dtype=np.int64)`，而真实的 `_fixed_tet_conn` 是
    # int32 —— 于是任何纯四面体网格在建面几何时段错误。实测：order 1/2/3
    # 全部 exit 139（`tests/unit/test_exact_adj_rows_dtype_mix.py`）。
    #
    # 修在这里而不是去改那两处兜底的 dtype：本函数已经在对
    # `code_arr`/`sps_1d` 做同样的归一化，调用方传什么 dtype 都不该让
    # kernel 崩。只改兜底的话，将来任何一个 int64 连接表的网格会从另一
    # 边再踩一次同一个坑。
    prism_conn = np.ascontiguousarray(prism_conn, dtype=np.int64)
    tet_conn = np.ascontiguousarray(tet_conn, dtype=np.int64)

    ref_pts_table = precompute_ref_pts_table(n1d, sps_1d)
    np_ref_pts_table, np_cov, np_duffy = precompute_native_prism_face_tables(
        n1d - 1)

    compute_exact_adj_rows_kernel(
        n_faces, n_fp, n_prism,
        cell_arr, axis_arr, side_arr,
        prism_conn, tet_conn, node_coords,
        valid_mask, ref_pts_table, adj_row_out,
        np.asarray(code_arr, dtype=np.int64),
        n1d, np.asarray(sps_1d, dtype=np.float64),
        np_ref_pts_table, np_cov, np_duffy,
    )
    return adj_row_out
