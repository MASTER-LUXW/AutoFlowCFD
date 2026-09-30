"""AutoFlowCFD V2.0 - 粘性通量的边界处理：逐面边界种类、边界"另一侧"梯度的镜像。

## 为什么需要按边界种类分派（2026-09-30 修复的真实缺陷）

粘性界面项在边界面上的公共通量是 `G(Q_avg, 梯度_avg)`，其中 `Q_avg` 用幽灵态、
"另一侧"梯度由本侧构造。此前除绝热类（WALL/SYMMETRY）把 `∇T` 法向镜像外，
**其余一律取本侧梯度**，且除速度罚项外不加任何罚项。后果是：凡幽灵态**延拓**
内部值的分量（出口的速度与温度、入口回流点、对称面与滑移壁的切向速度、
远场与入口的温度），该面上的公共法向粘性通量就等于本单元自己的通量，
FR 修正量为零 —— 那个分量**在扩散算子里没有施加任何边界条件**。弱形式里
于是留下一项不定号的边界积分 `-∮ φ (κ ∇φ)·n`，离散扩散算子失去强制性
（coercivity）。

实测（均匀静止基态、等压温度扰动、逐 DOF 组装纯热传导算子，质量矩阵范数下
对称部分的最大广义特征值，>0 即不强制）：

    P2 全对称边界          -1.6e-6（只剩常数模态）
    P2 Blasius 同款混合边界  +2.4e-2，16 个正方向，算子本身 3 个正实部特征值
    同上 + 长宽比 8、nz=1    +5.8e-1

最不强制的方向集中在远场与入口/出口相交的角部单元。动量在全对称边界下也有
34 个正方向（对称面切向应力原样穿过边界）。Blasius P2 隐式稳态第 40 步起的
发散就是它：贴壁第一层的等压密度/温度模态被放大（NK 固定 CFL 130 下每步
x1.23），把热传导整个关掉后同一模态每步衰减（x0.79~0.99）。

## 各边界种类的处理（每一条都给出强制的边界积分）

    VBC_NOSLIP_WALL  无滑移壁：速度 Dirichlet（本侧梯度 + 速度罚项）；
                     绝热：∇T 法向镜像，公共法向热通量恰为零
    VBC_MIRROR       对称面 / 滑移壁：幽灵态是本侧状态的镜像，梯度同样取镜像
                     场的梯度（速度 R∇uR、温度 R∇T，R = I - 2nn^T）。面平均后
                     应变的法-切分量与法向温度梯度恰为零，公共牵引只剩法向分量
                     （法向速度由罚项弱施加为零）——正是对称/滑移的精确条件
    VBC_DIRICHLET    远场：幽灵态整份给定（自由来流），速度与温度都按 Dirichlet
                     施加（本侧梯度 + 速度罚项 + 以热传导率为系数的温度罚项）
    VBC_NEUMANN      出口：粘性上一切延拓分量取零法向通量（无牵引、绝热），
                     即公共法向粘性通量为零、不加罚项
    VBC_INLET        入口：逐通量点按本侧法向速度判定（与 `inlet_ghost_state` 同一
                     条件）—— 流入（给定入口状态）按 VBC_DIRICHLET，回流（幽灵态
                     延拓内部）按 VBC_NEUMANN

`VBC_INTERIOR`（0）表示内部面，罚项取内部形式（见 `viscous_ip_penalty_tilde`）。
逐面种类由 `boundary/fr_ghost_state.py::build_viscous_boundary_kind` 按边界组
BC 类型给出；本模块只放 kernel 侧（numba）要用的常量与逐点函数。
"""

import numpy as np
from numba import njit

VBC_INTERIOR = 0
VBC_NOSLIP_WALL = 1
VBC_MIRROR = 2
VBC_DIRICHLET = 3
VBC_NEUMANN = 4
VBC_INLET = 5


@njit(cache=True, inline='always')
def mirror_normal_component(g: np.ndarray, adj_row: np.ndarray) -> np.ndarray:
    """把向量 g 关于面（法向由 `adj_row` 给出方向）做**法向分量镜像**：

        g_mirror = g - 2 (g·n) n,    n = adj_row / |adj_row|

    取面平均后 `g_avg = g - (g·n) n` 法向分量**精确为零**，于是投影到该面的
    离散传导热通量 `a·q_avg = -k (adj_row·∇T_avg)` 恒等于零——这是绝热/对称
    的精确离散表述，不是"近似到某个容差"。

    方向说明：用的是 `adj_row`（逆变行，即 det(J)∇ξ，与面法向平行），
    **不是** `true_normal`。两者平行，但用 `adj_row` 才能保证"恒等于零"的是
    真正进入残差的那个投影量 `a0*q_x+a1*q_y+a2*q_z` 本身。镜像对 n 取反不变，
    因此内/外法向朝向约定无关紧要。

    退化保护：`|adj_row|` 为零（退化面）时原样返回 g——此时该面的通量投影
    本来就是零，镜不镜像都不影响结果。
    """
    m2 = adj_row[0] * adj_row[0] + adj_row[1] * adj_row[1] + adj_row[2] * adj_row[2]
    out = np.empty(3)
    if m2 <= 0.0:
        for a in range(3):
            out[a] = g[a]
        return out
    # 不必显式开方归一化：(g·a)/|a|^2 * a 就是 (g·n)n
    d = (g[0] * adj_row[0] + g[1] * adj_row[1] + g[2] * adj_row[2]) / m2
    for a in range(3):
        out[a] = g[a] - 2.0 * d * adj_row[a]
    return out


@njit(cache=True, inline='always')
def mirror_velocity_gradient(gv: np.ndarray, adj_row: np.ndarray) -> np.ndarray:
    """镜像速度场的梯度 `R gv R`，`R = I - 2 n n^T`，`gv[i,j] = d(u_i)/d(x_j)`。

    对称面幽灵态是本侧速度关于该面的镜像 `u_g(x) = R u(Rx)`，其梯度恰为
    `R ∇u R`。与本侧取平均后：应变张量的法-切分量 `n·S·t` 反号相消为零，
    法-法与切-切分量保留，于是 `tau_avg·n = (n·tau·n) n` —— 切向牵引为零、
    法向正应力保留，是对称面/滑移壁的精确条件。

    `R g R = g - 2 n (n^T g) - 2 (g n) n^T + 4 (n^T g n) n n^T`，直接按分量展开，
    不构造 R。退化面（`|adj_row|=0`）原样返回。
    """
    m2 = adj_row[0] * adj_row[0] + adj_row[1] * adj_row[1] + adj_row[2] * adj_row[2]
    out = np.empty((3, 3))
    if m2 <= 0.0:
        for a in range(3):
            for b in range(3):
                out[a, b] = gv[a, b]
        return out
    inv = 1.0 / np.sqrt(m2)
    n = np.empty(3)
    for a in range(3):
        n[a] = adj_row[a] * inv
    nTg = np.empty(3)   # (n^T g)_j
    gn = np.empty(3)    # (g n)_i
    for a in range(3):
        nTg[a] = n[0] * gv[0, a] + n[1] * gv[1, a] + n[2] * gv[2, a]
        gn[a] = gv[a, 0] * n[0] + gv[a, 1] * n[1] + gv[a, 2] * n[2]
    ngn = n[0] * gn[0] + n[1] * gn[1] + n[2] * gn[2]
    for a in range(3):
        for b in range(3):
            out[a, b] = (gv[a, b] - 2.0 * n[a] * nTg[b] - 2.0 * gn[a] * n[b]
                         + 4.0 * ngn * n[a] * n[b])
    return out


@njit(cache=True, inline='always')
def resolve_point_kind(kind: int, Q_s: np.ndarray, adj_row: np.ndarray) -> int:
    """把入口（`VBC_INLET`）按本通量点实际流入/回流落到 Dirichlet / Neumann。

    判据与 `inlet_ghost_state` 同一条物理条件：本侧法向速度 `u·n < 0`（外法向，
    原生面的 `adj_row` 已 outward 定向）为流入、幽灵态是给定入口状态；否则回流、
    幽灵态延拓本侧。**不能**用"幽灵态是否与本侧逐位相同"来判：回流时幽灵态与
    kernel 里的本侧外插值来自两次独立的外插，舍入差会把回流点误判成 Dirichlet
    （解析 Jacobian 的复合差分里本侧平移后的迹与平移场外插出的幽灵态也不逐位
    相同，实测单元块偏差 11%）。其余种类原样返回。
    """
    if kind != VBC_INLET:
        return kind
    un = Q_s[1] * adj_row[0] + Q_s[2] * adj_row[1] + Q_s[3] * adj_row[2]
    if un < 0.0:
        return VBC_DIRICHLET
    return VBC_NEUMANN


@njit(cache=True, inline='always')
def boundary_other_gradients(gv_s: np.ndarray, gT_s: np.ndarray, adj_row: np.ndarray, kind: int):
    """边界（含混合拆分面的边界半区）上"另一侧"的梯度 `(gv_x, gT_x)`。

    入口不必先落到 Dirichlet / Neumann：两者都取本侧梯度。无滑移壁：速度梯度取本侧（壁面切向
    速度的法向导数就是壁面剪应力本身），温度梯度法向镜像（绝热）；对称面 /
    滑移壁：两者都取镜像场的梯度；Dirichlet / Neumann：取本侧（Neumann 的
    零法向通量由 `viscous_jump_point` 直接给出，不经梯度）。
    """
    gv_x = np.empty((3, 3))
    gT_x = np.empty(3)
    if kind == VBC_MIRROR:
        gv_x = mirror_velocity_gradient(gv_s, adj_row)
        gT_x = mirror_normal_component(gT_s, adj_row)
        return gv_x, gT_x
    for a in range(3):
        for b in range(3):
            gv_x[a, b] = gv_s[a, b]
    if kind == VBC_NOSLIP_WALL:
        gT_x = mirror_normal_component(gT_s, adj_row)
    else:
        for a in range(3):
            gT_x[a] = gT_s[a]
    return gv_x, gT_x
