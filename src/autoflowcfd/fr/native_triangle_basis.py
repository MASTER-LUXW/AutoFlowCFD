"""AutoFlowCFD V2.0 - 三角形原生 PKD/Dubiner 基与 Warp & Blend 节点。

从 Nodal-DG 参考实现（https://github.com/tcew/nodal-dg, Codes1.1/Codes2D/）
逐行移植，不是重新发明。原始 MATLAB 文件：`Nodes2D.m`、`xytors.m`、
`rstoab.m`、`Simplex2DP.m`、`GradSimplex2DP.m`、`Warpfactor.m`。

## 为什么需要它（2026-09-18）

棱柱至今用**坍缩坐标基**（`collapsed_basis.py::prism_modal_basis_and_grad`，
在 (a,b) 截面上做 Duffy 三角形坍缩）。实测两条后果：

1. **微分矩阵的元素量级随阶数爆炸**：

       max|D_3d_prism|:   P1 2.049 -> P2 22.51 -> P3 560.1   （每阶约 x25）
       max|D_native_tet|: P1 0.500 -> P2 2.000 -> P3 4.045   （对照，正常）

   坍缩把节点挤向退化边，节点微分算子因此病态。

2. **自由流保持性只有 ~1e-9**（仿射结构化棱柱网格上本应是机器零）。
   实测三阶数的误差与 `eps * max|D| / det(J)` 严格成比例：

       order  max|D|   det_J min   自由流误差   误差/(max|D|/det_J)
         P1    2.049   1.99e-6     3.01e-9      2.9e-15
         P2   22.51    1.06e-6     7.80e-9      3.7e-16
         P3   560.1    6.54e-7     1.64e-7      1.9e-16

   后两行就是机器 epsilon —— 所以那个误差**是纯舍入被算子量级放大**，
   不是算法 bug；`enforce_constant_annihilation` 已经在用（行和残余相对
   `max|D|` 正好是 eps 量级），**不降低 `max|D|` 就无法改善**。

3. 更严重的是**伪横流**：只有被三角化的两个参考轴方向会长出非物理的
   横流速度（把展向换成挤出轴后保持机器零，相差 11 个数量级），它在
   P1 上饱和、在 P2 上无界增长导致发散（1152 单元干净网格上第 75 步）。

四面体当年有**完全同类**的病理，解法就是整套换成 native PKD/Dubiner 基
（见 `native_simplex_basis.py` 与项目记忆 `tet-collapsed-coord-anisotropy` /
`native-tet-basis-production-readiness`）。本模块是棱柱侧的同一条路：
棱柱 = 三角形 ⊗ 直线，三角形用这里的原生 PKD 基 + Warp & Blend 节点，
挤出方向保持现有的 Legendre/Gauss-Legendre（那个方向本来就是精确的，
不需要改）。

## 关键点：基的公式用到 (a,b)，但**节点不坍缩**

Dubiner 基的**表达式**里确实出现 `a = 2(1+r)/(1-s) - 1` 这个坍缩坐标
（那是 Dubiner 1991 构造这套正交多项式时用的写法），但：

  * **节点**取在真实参考三角形 `(r,s)` 上（Warp & Blend），不是把张量积
    方格坍缩过来 —— 所以节点不会挤向退化边；
  * **梯度**直接对 `(r,s)` 给出闭式（`GradSimplex2DP`），全程只用非负
    整数次幂、不含除法，退化轴上处处有限 —— 与 `simplex3d_grad` 同一条
    技巧，不经过会奇异的 `d(r,s)/d(a,b)` 逆。

这两点正是坍缩方案与原生方案的分水岭。
"""

from typing import List, Tuple

import numpy as np

# Jacobi 多项式与其导数复用坍缩基模块里那一份（正交多项式本身与"节点
# 怎么取"无关，是同一个数学对象；再抄一份等于多一个要同步的事实来源）。
from .collapsed_basis import grad_jacobi_polynomial, jacobi_polynomial
from .warp_blend_nodes import _evalshift

#: `Nodes2D.m` 的 alpha 优化表（三角形专用，与三维那张表不同）。
#: N >= 16 时原实现取 5/3。
_ALPHA_OPT_2D = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                 1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223,
                 1.6258]


def restricted_tri_modes(order: int) -> List[Tuple[int, int]]:
    """三角形最小 PKD/Dubiner 模态索引集合：`i+j<=order`。

    个数是 `(order+1)(order+2)/2`，与同阶 Warp & Blend 节点数相等 ——
    两者必须相等才能构成一个可逆的 Vandermonde（有断言检查）。

    与坍缩基的 `(order+1)^2` 相比少了将近一半：坍缩基用的是"扩展张量积"
    `i,j` 各自独立取 0..order，其中 `i+j>order` 的那些模态在三角形上
    不是独立的自由度。
    """
    return [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]


def rs_to_ab(r: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`rstoab.m` 逐行移植：参考三角形 `(r,s)` -> 坍缩坐标 `(a,b)`。

    `s == 1`（退化顶点）处 `a` 取 -1 —— 这不是 fallback 近似，而是极限值
    本身：那一点上 `a` 的取值不影响任何基函数的值（`(1-b)^i` 因子在
    `i>0` 时为零，`i=0` 时基函数与 `a` 无关）。
    """
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    denom = 1.0 - s
    a = np.where(np.abs(denom) > 1e-12, 2.0 * (1.0 + r) / np.where(
        np.abs(denom) > 1e-12, denom, 1.0) - 1.0, -1.0)
    return a, s.copy()


def simplex2d_value(a: np.ndarray, b: np.ndarray, i: int, j: int) -> np.ndarray:
    """`Simplex2DP.m` 逐行移植：三角形正交模态基在 `(a,b)` 处的取值。

        P = sqrt(2) * P_i^{0,0}(a) * P_j^{2i+1,0}(b) * (1-b)^i

    处处有限（`b=1` 时 `(1-b)^i` 在 `i>0` 直接为零，`i=0` 时该因子为 1），
    不需要任何 L'Hopital 或 fallback。
    """
    h1 = jacobi_polynomial(a, 0.0, 0.0, i)
    h2 = jacobi_polynomial(b, float(2 * i + 1), 0.0, j)
    return np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i


def simplex2d_grad(a: np.ndarray, b: np.ndarray, i: int,
                   j: int) -> Tuple[np.ndarray, np.ndarray]:
    """`GradSimplex2DP.m` 逐行移植：对**参考三角形原生坐标 `(r,s)`** 的
    梯度（不是对 `(a,b)` 的梯度）。

    与 `native_simplex_basis.simplex3d_grad` 同一条技巧：把 `(1-b)` 权重
    因子的幂次**代数地减一**，等价于提前吸收了链式法则本该出现、原本
    会奇异的那个因子 —— 全程只用非负整数次幂、不含除法，因此在退化轴
    （`b=1`）上同样处处有限。

    Returns:
        `(dP_dr, dP_ds)`，形状与 `a`/`b` 相同。
    """
    fa = jacobi_polynomial(a, 0.0, 0.0, i)
    dfa = grad_jacobi_polynomial(a, 0.0, 0.0, i)
    gb = jacobi_polynomial(b, float(2 * i + 1), 0.0, j)
    dgb = grad_jacobi_polynomial(b, float(2 * i + 1), 0.0, j)

    half_1mb = 0.5 * (1.0 - b)

    # r 方向
    dmode_dr = dfa * gb
    if i > 0:
        dmode_dr = dmode_dr * (half_1mb ** (i - 1))

    # s 方向
    dmode_ds = dfa * (gb * (0.5 * (1.0 + a)))
    if i > 0:
        dmode_ds = dmode_ds * (half_1mb ** (i - 1))
    tmp = dgb * (half_1mb ** i)
    if i > 0:
        tmp = tmp - 0.5 * i * gb * (half_1mb ** (i - 1))
    dmode_ds = dmode_ds + fa * tmp

    norm = 2.0 ** (i + 0.5)
    return dmode_dr * norm, dmode_ds * norm


def warp_blend_nodes_2d(p: int) -> Tuple[np.ndarray, np.ndarray]:
    """`Nodes2D.m` + `xytors.m` 逐行移植：参考三角形上的 Warp & Blend 节点。

    Args:
        p: 多项式阶数（`p=0` 时返回单个形心点）

    Returns:
        `(r, s)`，各 `(p+1)(p+2)/2` 个，落在标准参考三角形
        `{(r,s) : r>=-1, s>=-1, r+s<=0}` 上。
    """
    if p == 0:
        # 单点：取形心。等距/warp 构造在 N=0 时除以 N，必须单独处理。
        return np.array([-1.0 / 3.0]), np.array([-1.0 / 3.0])

    alpha = _ALPHA_OPT_2D[p - 1] if p < 16 else 5.0 / 3.0

    n_p = (p + 1) * (p + 2) // 2
    l1 = np.zeros(n_p)
    l3 = np.zeros(n_p)
    sk = 0
    for n in range(1, p + 2):
        for m in range(1, p + 3 - n):
            l1[sk] = (n - 1) / p
            l3[sk] = (m - 1) / p
            sk += 1
    l2 = 1.0 - l1 - l3

    x = -l2 + l3
    y = (-l2 - l3 + 2.0 * l1) / np.sqrt(3.0)

    # `_evalshift` 就是 Nodes2D 里那段 blend+warp（三维路径也调用它处理
    # 面内），签名 (p, alpha, L1, L2, L3) -> (dx, dy)。
    #
    # `Warpfactor.m` 末尾那个端点修正**不需要**在这里再写一遍：它已经被
    # `_evalshift` 里的 blend 因子（L2*L3 等在端点为零）抵消掉。本文件
    # 曾经为"与 MATLAB 逐行一致"单独实现过一份 `_warpfactor`，但它从未
    # 被调用过（零引用、零测试覆盖），2026-09-18 删除 —— 留一份不跑的
    # 实现只会让后来人以为要同步维护它。
    dx, dy = _evalshift(p, alpha, l1, l2, l3)
    x = x + dx
    y = y + dy

    # xytors.m：等边三角形坐标 -> 标准参考三角形坐标
    m1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    m2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    m3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    r = -m2 + m3 - m1
    s = -m2 - m3 + m1
    return r, s


def eval_tri_modes(order: int, r: np.ndarray, s: np.ndarray):
    """在**任意**点集上求全部受限 PKD 模态及其对 `(r,s)` 的梯度。

    Returns:
        `(V, Vr, Vs)`，各 `(n_pts, (order+1)(order+2)/2)`，列序与
        `restricted_tri_modes(order)` 一致。

    与 `build_native_tri_vandermonde` 的区别只有一条：这里**不要求**
    点数等于模态数。棱柱基要在 `(order+1)^2(order+2)/2` 个棱柱节点上求
    三角形模态（点数远多于三角形模态数），走的就是这条；把模态求值循环
    留在一处，棱柱侧不再抄一遍公式（抄一遍的代价是两份要同步的事实
    来源，本项目已多次因此出真实缺陷）。
    """
    modes = restricted_tri_modes(order)
    a, b = rs_to_ab(r, s)
    V = np.empty((len(a), len(modes)))
    Vr = np.empty_like(V)
    Vs = np.empty_like(V)
    for m, (i, j) in enumerate(modes):
        V[:, m] = simplex2d_value(a, b, i, j)
        Vr[:, m], Vs[:, m] = simplex2d_grad(a, b, i, j)
    return V, Vr, Vs


def build_native_tri_vandermonde(order: int, r: np.ndarray, s: np.ndarray):
    """给定三角形**节点**，构造方阵 Vandermonde `(V, Vr, Vs)`。

    Raises:
        ValueError: 节点数与模态数不相等（两者理论上都必须是
            `(order+1)(order+2)/2`；不等说明节点生成或模态索引有 bug，
            不应当静默继续）。非方阵的取值请用 `eval_tri_modes`。
    """
    n_pts = len(r)
    n_modes = (order + 1) * (order + 2) // 2
    if n_pts != n_modes:
        raise ValueError(
            f"三角形节点数 {n_pts} 与受限 PKD 模态数 {n_modes} 不一致"
            f"（order={order}）——两者理论上必须相等"
            f"（(order+1)(order+2)/2）。")
    return eval_tri_modes(order, r, s)
