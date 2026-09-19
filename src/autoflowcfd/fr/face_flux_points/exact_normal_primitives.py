"""AutoFlowCFD V2.0 - 逐通量点 Jacobian / adj 行的 numba 内联原语。

3x3 线代（det/adjugate）与三类单元在**单个**参考点处的解析 Jacobian /
adj 行：坍缩四面体、坍缩棱柱、原生四面体、原生棱柱。全部是
`@njit(inline='always')` 的纯标量/小数组函数，由
`exact_normal_kernel.py` 的 prange kernel 逐点调用。

从原 `exact_normal_kernel.py`（572 行）拆出（2026-09-19，项目"单文件
不超 500 行"规范）。纯搬家，未改任何逻辑。
"""

import numpy as np
from numba import njit


# ======================================================================
# 3×3 线性代数内联辅助函数
# ======================================================================

@njit(inline='always')
def _native_prism_adj_row_at(cov0, cov1, cov2, duffy,
                             r, s_, t, p0, p1, p2, p3, p4, p5):
    """原生棱柱某个面在参考点 `(r,s_,t)` 处的 `adj_row`（未归一化的
    "物理外法向 x 面积微元"），返回三个分量。

    与 `fr/native_prism/face.py::native_prism_face_adj_rows` **同一个公式**
    （那里是 numpy 批量版，这里是 numba 逐点版，本项目生产路径走的是
    numba 这条，见 `face_flux_points/merge.py` 的调用点）：

        adj_row_d = duffy * sum_m cov_m * adj(J)[m, d]

    `J[d, m] = d(phys_d)/d(xi_m)`（与 `native_prism_exact_jacobian` 的
    约定一致），`adj(J) = det(J) inv(J)`，`cov` 是该面的参考外向余向量，
    `duffy` 是三角封盖的 `(1-s)/2`（侧四边形传 1.0）—— 因子乘在行里而
    不是乘在求积权重里，理由与实测数据见 `native_prism/face.py` 里
    `_FACE_REF_COVECTOR` 上方那段。

    **全标量实现、不分配任何临时数组**：`prange` 体内的 `np.empty` 会让
    numba 的并行数组分析尝试把分配提升出循环，实测在本 kernel 里直接
    段错误（不是异常、没有任何输出）。既有的 `_adj3`/`_tet_jac_at` 走的是
    坍缩分支那条已经编译通过的路径，不能据此推断新分支也安全。
    """
    l1 = -0.5 * (r + s_)
    l2 = 0.5 * (1.0 + r)
    l3 = 0.5 * (1.0 + s_)
    hm = 0.5 * (1.0 - t)
    hp = 0.5 * (1.0 + t)

    # J[d, m]，逐分量展开
    b0 = l1 * p0[0] + l2 * p1[0] + l3 * p2[0]
    b1 = l1 * p0[1] + l2 * p1[1] + l3 * p2[1]
    b2 = l1 * p0[2] + l2 * p1[2] + l3 * p2[2]
    t0 = l1 * p3[0] + l2 * p4[0] + l3 * p5[0]
    t1 = l1 * p3[1] + l2 * p4[1] + l3 * p5[1]
    t2 = l1 * p3[2] + l2 * p4[2] + l3 * p5[2]

    j00 = hm * 0.5 * (p1[0] - p0[0]) + hp * 0.5 * (p4[0] - p3[0])
    j10 = hm * 0.5 * (p1[1] - p0[1]) + hp * 0.5 * (p4[1] - p3[1])
    j20 = hm * 0.5 * (p1[2] - p0[2]) + hp * 0.5 * (p4[2] - p3[2])
    j01 = hm * 0.5 * (p2[0] - p0[0]) + hp * 0.5 * (p5[0] - p3[0])
    j11 = hm * 0.5 * (p2[1] - p0[1]) + hp * 0.5 * (p5[1] - p3[1])
    j21 = hm * 0.5 * (p2[2] - p0[2]) + hp * 0.5 * (p5[2] - p3[2])
    j02 = 0.5 * (t0 - b0)
    j12 = 0.5 * (t1 - b1)
    j22 = 0.5 * (t2 - b2)

    # adj(J)[m, d]，与 `_adj3` 逐项同式
    a00 = j11 * j22 - j12 * j21
    a01 = j02 * j21 - j01 * j22
    a02 = j01 * j12 - j02 * j11
    a10 = j12 * j20 - j10 * j22
    a11 = j00 * j22 - j02 * j20
    a12 = j02 * j10 - j00 * j12
    a20 = j10 * j21 - j11 * j20
    a21 = j01 * j20 - j00 * j21
    a22 = j00 * j11 - j01 * j10

    return (duffy * (cov0 * a00 + cov1 * a10 + cov2 * a20),
            duffy * (cov0 * a01 + cov1 * a11 + cov2 * a21),
            duffy * (cov0 * a02 + cov1 * a12 + cov2 * a22))


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
def _native_tet_adj_row_at(excluded_vertex, a, b, p0, p1, p2, p3):
    """native 四面体（路径C）版本的单点 adj 行——`face_flux_points_
    exact_normal.py::_native_tet_adj_row_batched` 的逐点 numba 内联版
    （该函数是批量 numpy 版本，这里是单个 (face,fp) 点的等价移植，供
    `compute_exact_adj_rows_kernel` 内联调用，理由同 `_tet_jac_at`/
    `_prism_jac_at`——numba 不支持调用那个批量函数）。

    真实 bug 修复背景：`owner_adj_row_exact`/`neighbor_adj_row_exact`
    此前完全由这个模块的 `compute_exact_adj_rows_fast`（一个独立于
    `face_flux_points/exact_normal.py::compute_exact_adj_rows` 的性能
    优化版重复实现）产出，从未被 Part7/8 的 native 分支 code_arr 修复
    覆盖到——native 面的 excluded_vertex（0~3）被当成坍缩坐标的
    `axis_arr` 直接使用，`excluded_vertex=3` 时 `table_idx=ax*2+sd_idx`
    越界读取 `ref_pts_table`（形状只有 6 行）未定义内存，
    `excluded_vertex<3` 时也会用错误的坍缩坐标 Jacobian 公式——用真实
    order=1 网格复现：native 四面体单元的无粘残差从应有的精确 0 暴涨到
    相对残差 1e6 量级，如实记录（不是理论推导出来的，是被这次接入
    native kernel 后新增的端到端残差测试当场测出）。
    """
    if excluded_vertex == 0:
        pi0, pi1, pi2 = p1[0], p1[1], p1[2]
        pj0, pj1, pj2 = p2[0], p2[1], p2[2]
        pk0, pk1, pk2 = p3[0], p3[1], p3[2]
        pe0, pe1, pe2 = p0[0], p0[1], p0[2]
    elif excluded_vertex == 1:
        pi0, pi1, pi2 = p0[0], p0[1], p0[2]
        pj0, pj1, pj2 = p2[0], p2[1], p2[2]
        pk0, pk1, pk2 = p3[0], p3[1], p3[2]
        pe0, pe1, pe2 = p1[0], p1[1], p1[2]
    elif excluded_vertex == 2:
        pi0, pi1, pi2 = p0[0], p0[1], p0[2]
        pj0, pj1, pj2 = p1[0], p1[1], p1[2]
        pk0, pk1, pk2 = p3[0], p3[1], p3[2]
        pe0, pe1, pe2 = p2[0], p2[1], p2[2]
    else:
        pi0, pi1, pi2 = p0[0], p0[1], p0[2]
        pj0, pj1, pj2 = p1[0], p1[1], p1[2]
        pk0, pk1, pk2 = p2[0], p2[1], p2[2]
        pe0, pe1, pe2 = p3[0], p3[1], p3[2]

    e1_0 = pj0 - pi0
    e1_1 = pj1 - pi1
    e1_2 = pj2 - pi2
    e2_0 = pk0 - pi0
    e2_1 = pk1 - pi1
    e2_2 = pk2 - pi2

    c_a = (1.0 - b) / 4.0
    c_b = -(1.0 + a) / 4.0
    dpda_0 = c_a * e1_0
    dpda_1 = c_a * e1_1
    dpda_2 = c_a * e1_2
    dpdb_0 = c_b * e1_0 + 0.5 * e2_0
    dpdb_1 = c_b * e1_1 + 0.5 * e2_1
    dpdb_2 = c_b * e1_2 + 0.5 * e2_2

    r0 = dpda_1 * dpdb_2 - dpda_2 * dpdb_1
    r1 = dpda_2 * dpdb_0 - dpda_0 * dpdb_2
    r2 = dpda_0 * dpdb_1 - dpda_1 * dpdb_0

    tox0 = pe0 - pi0
    tox1 = pe1 - pi1
    tox2 = pe2 - pi2
    dotv = r0 * tox0 + r1 * tox1 + r2 * tox2
    if dotv > 0.0:
        r0 = -r0
        r1 = -r1
        r2 = -r2
    return r0, r1, r2


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
