"""
AutoFlowCFD V2.0 - Hesthaven & Warburton "Warp & Blend" 四面体优化节点
分布，从 Nodal-DG 参考实现（https://github.com/tcew/nodal-dg,
Codes1.1/Codes3D/，2026-08-30 抓取核实）逐行移植，不是重新发明。
原始 MATLAB 文件：Nodes3D.m, EquiNodes3D.m, WarpShiftFace3D.m,
evalshift.m, evalwarp.m, xyztorst.m。

用于 `native_tet/basis.py::build_native_tet_operators`（Part6
整改计划阶段0，四面体路径C——独立于坍缩坐标的体积微分算子构造）。
优化节点分布本身不是这次修复的必要条件（`native_tet/basis.py`
的正规原生梯度公式在任意节点分布上都同样规避坐标奇点），选用
Warp & Blend 是因为它同时给出比等距节点更好的插值 Lebesgue 常数，
是标准做法，不引入额外假设。

输出：给定阶数 p 的 (r,s,t) 参考四面体坐标（标准 [-1,1] 参考四面体，
AutoFlowCFD 约定 v1=(-1,-1,-1),v2=(1,-1,-1),v3=(-1,1,-1),v4=(-1,-1,1)，
与 Warburton 原始约定的顶点标签不同但通过 xyztorst 精确换算），
(p+1)(p+2)(p+3)/6 个节点。
"""

import numpy as np

ALPHA_STORE = [0, 0, 0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577, 1.1603,
               1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655]


def _jacobi_gl(p: int) -> np.ndarray:
    """JacobiGL(0,0,p)：p+1 个 Gauss-Lobatto-Legendre 点，升序，含端点 ±1。"""
    if p == 0:
        return np.array([0.0])
    if p == 1:
        return np.array([-1.0, 1.0])
    from numpy.polynomial import legendre as L

    interior_roots = L.Legendre.basis(p).deriv().roots()
    pts = np.concatenate([[-1.0], np.sort(interior_roots), [1.0]])
    return pts


def _equi_nodes_3d(N: int):
    """EquiNodes3D.m 逐行移植：等距 (r,s,t)，MATLAB 1-index 循环转 0-index。"""
    Np = (N + 1) * (N + 2) * (N + 3) // 6
    X = np.zeros(Np)
    Y = np.zeros(Np)
    Z = np.zeros(Np)
    sk = 0
    for n in range(1, N + 2):
        for m in range(1, N + 3 - n):
            for q in range(1, N + 4 - n - m):
                X[sk] = -1 + (q - 1) * 2.0 / N
                Y[sk] = -1 + (m - 1) * 2.0 / N
                Z[sk] = -1 + (n - 1) * 2.0 / N
                sk += 1
    return X, Y, Z


def _evalwarp(p: int, xnodes: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """evalwarp.m 逐行移植：一维边缘 warp 函数（显式 Lagrange 基）。"""
    warp = np.zeros_like(xout)
    xeq = np.array([-1.0 + 2.0 * (p - i) / p for i in range(p + 1)])  # i=0..p (MATLAB i=1..p+1)
    for i in range(p + 1):
        d = np.full_like(xout, xnodes[i] - xeq[i])
        for j in range(1, p):  # MATLAB j=2..p -> 0-index j=1..p-1
            if i != j:
                d = d * (xout - xeq[j]) / (xeq[i] - xeq[j])
        if i != 0:
            d = -d / (xeq[i] - xeq[0])
        if i != p:
            d = d / (xeq[i] - xeq[p])
        warp = warp + d
    return warp


def _evalshift(p: int, pval: float, L1: np.ndarray, L2: np.ndarray, L3: np.ndarray):
    """evalshift.m 逐行移植：二维 warp & blend 变换。"""
    gaussX = -_jacobi_gl(p)
    blend1, blend2, blend3 = L2 * L3, L1 * L3, L1 * L2
    warpfactor1 = 4 * _evalwarp(p, gaussX, L3 - L2)
    warpfactor2 = 4 * _evalwarp(p, gaussX, L1 - L3)
    warpfactor3 = 4 * _evalwarp(p, gaussX, L2 - L1)
    warp1 = blend1 * warpfactor1 * (1 + (pval * L1) ** 2)
    warp2 = blend2 * warpfactor2 * (1 + (pval * L2) ** 2)
    warp3 = blend3 * warpfactor3 * (1 + (pval * L3) ** 2)
    dx = 1 * warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    dy = 0 * warp1 + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3
    return dx, dy


def _warp_shift_face_3d(p: int, pval: float, pval2: float, L1, L2, L3, L4):
    """WarpShiftFace3D.m 逐行移植（注意 evalshift 只用到 L2,L3,L4，L1 未使用，
    与原 MATLAB 签名的参数命名习惯一致——不是遗漏）。"""
    return _evalshift(p, pval, L2, L3, L4)


def warp_blend_nodes_3d(p: int):
    """Nodes3D.m 逐行移植。返回 (r,s,t)，形状各 (Np,)，标准参考四面体
    （AutoFlowCFD 约定 v1=(-1,-1,-1),v2=(1,-1,-1),v3=(-1,1,-1),v4=(-1,-1,1)，
    与 Warburton 原始约定的顶点标签不同但通过 xyztorst 精确换算）。

    p=0 特判：受限 PKD 模态数 `(0+1)(0+2)(0+3)/6=1`，只有常数模态
    (i=j=k=0)，常数函数的插值结果与采样点位置无关（数学上任意一点都
    精确成立），Warp & Blend 优化节点分布本身的意义是改善*多点*插值的
    Lebesgue 常数，对单点情形不适用，也不能直接调用——原始 MATLAB 算法
    的 `EquiNodes3D.m`/`evalwarp.m` 都以阶数 N/p 做分母，N=0 时是纯粹的
    未定义除法（不是"数学上等于0"，是"这一步压根不该跑到这里"），必须
    在进入那条流水线之前分流。取参考四面体形心
    （四个顶点坐标平均，`v1..v4` 见下方，代入得 (-0.5,-0.5,-0.5)）作为
    唯一节点，这与 `build_native_tet_operators`/`_native_mode_norm_squared`
    等下游代码要求"n_native_sps 个节点、几何位置在参考四面体内部即可"
    的约定完全兼容（不要求该点是任何特定的传统求积点）。
    """
    if p == 0:
        return np.array([-0.5]), np.array([-0.5]), np.array([-0.5])

    alpha = ALPHA_STORE[p] if p <= 14 else 1.0

    r0, s0, t0 = _equi_nodes_3d(p)
    L1 = (1 + t0) / 2.0
    L2 = (1 + s0) / 2.0
    L3 = -(1 + r0 + s0 + t0) / 2.0
    L4 = (1 + r0) / 2.0

    v1 = np.array([-1.0, -1.0 / np.sqrt(3), -1.0 / np.sqrt(6)])
    v2 = np.array([1.0, -1.0 / np.sqrt(3), -1.0 / np.sqrt(6)])
    v3 = np.array([0.0, 2.0 / np.sqrt(3), -1.0 / np.sqrt(6)])
    v4 = np.array([0.0, 0.0, 3.0 / np.sqrt(6)])

    t1 = np.zeros((4, 3))
    t2 = np.zeros((4, 3))
    t1[0] = v2 - v1
    t1[1] = v2 - v1
    t1[2] = v3 - v2
    t1[3] = v3 - v1
    t2[0] = v3 - 0.5 * (v1 + v2)
    t2[1] = v4 - 0.5 * (v1 + v2)
    t2[2] = v4 - 0.5 * (v2 + v3)
    t2[3] = v4 - 0.5 * (v1 + v3)
    for n in range(4):
        t1[n] /= np.linalg.norm(t1[n])
        t2[n] /= np.linalg.norm(t2[n])

    tol = 1e-10
    XYZ = np.outer(L3, v1) + np.outer(L4, v2) + np.outer(L2, v3) + np.outer(L1, v4)
    shift = np.zeros_like(XYZ)

    Ls_all = [L1, L2, L3, L4]
    face_perm = {
        0: (0, 1, 2, 3),  # face=1: La=L1,Lb=L2,Lc=L3,Ld=L4
        1: (1, 0, 2, 3),  # face=2: La=L2,Lb=L1,Lc=L3,Ld=L4
        2: (2, 0, 3, 1),  # face=3: La=L3,Lb=L1,Lc=L4,Ld=L2
        3: (3, 0, 2, 1),  # face=4: La=L4,Lb=L1,Lc=L3,Ld=L2
    }
    for face in range(4):
        ia, ib, ic, id_ = face_perm[face]
        La, Lb, Lc, Ld = Ls_all[ia], Ls_all[ib], Ls_all[ic], Ls_all[id_]

        warp1, warp2 = _warp_shift_face_3d(p, alpha, alpha, La, Lb, Lc, Ld)
        blend = Lb * Lc * Ld
        denom = (Lb + 0.5 * La) * (Lc + 0.5 * La) * (Ld + 0.5 * La)
        ids = denom > tol
        blend = blend.copy()
        blend[ids] = (1 + (alpha * La[ids]) ** 2) * blend[ids] / denom[ids]

        shift = shift + np.outer(blend * warp1, t1[face]) + np.outer(blend * warp2, t2[face])

        ids2 = (La < tol) & (((Lb > tol).astype(int) + (Lc > tol).astype(int) + (Ld > tol).astype(int)) < 3)
        shift[ids2] = np.outer(warp1[ids2], t1[face]) + np.outer(warp2[ids2], t2[face])

    XYZ = XYZ + shift

    # xyztorst.m
    rhs = XYZ.T - 0.5 * np.outer(v2 + v3 + v4 - v1, np.ones(XYZ.shape[0]))
    A = np.column_stack([0.5 * (v2 - v1), 0.5 * (v3 - v1), 0.5 * (v4 - v1)])
    RST = np.linalg.solve(A, rhs)
    r, s, t = RST[0], RST[1], RST[2]
    return r, s, t
