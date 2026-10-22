"""AutoFlowCFD V2.0 - 粘性物理通量与 IP 罚项（边界梯度镜像见 `viscous_bc.py`）。"""

import numpy as np

from numba import njit, prange
from autoflowcfd.core.thermo import CP_AIR, R_AIR

from .constants import VISCOUS_IP_C_BASE


def resolve_viscous_ip_constant(order: int) -> float:
    """按多项式阶数给出 IP 罚项常数 `c_ip`。

    ## 为什么必须随阶数增长

    IP 罚项要保证强制性（coercivity），常数必须压过**迹不等式常数**
    （trace inequality）：对 d 维单元上的 p 次多项式，
    `||v||_{dK}^2 <= C_tr * (p+1)(p+d)/d * |dK|/|K| * ||v||_K^2`。
    所以 `c_ip` 必须 `~ (p+1)(p+d)/d`；3D 下即 `(p+1)(p+3)/3`
    （P0 1.0、P1 2.67、P2 5.0、P3 8.0 —— 以 P0 归一后是 1 / 2.67 / 5 / 8）。

    ## 本项目的格式是 IIPG，不是 SIPG —— 它需要更大的罚项

    `compute_physical_gradient` 拿不到面数据，所以 `sigma = grad(u)` 没有
    BR1/SIPG 要求的提升项，缺的正是伴随一致性那一项 `-∮{∇v}·n[[u]]`。
    这样的格式是 **IIPG**（incomplete interior penalty）：它收敛、稳定，
    但强制性对罚项的下界要求比 SIPG 高。`VISCOUS_IP_C_BASE` 的取值因此是
    **实测标定**的（见下），不是照 SIPG 的经验值抄的。

    ## 标定依据（实测，2026-09-23）

    ### 精度：Blasius 解析初场、原生 P1、nx=16、CFL 0.1、`cf/cf_exact`

        BASE  c_ip    2000步(中位/最差)   4000步(中位/最差)   10000步
        1.0   2.667   0.9912 / 0.0162    1.0074 / 0.0421    1.0169 / 0.1327
        2.0   5.333   0.9955 / 0.0094    1.0126 / 0.0536    —
        4.0  10.667   0.9976 / 0.0062    1.0149 / 0.0581    —

    注意**排序在 2000~4000 步之间反转**：大罚项早期更准、长窗口反而更差。
    项目有"短窗口反复给出符号相反结论"的历史教训，所以以长窗口为准。

    ### 谱：均匀基态、16 单元、纯粘性算子数值雅可比逐块谱（原生）

        BASE  c_ip    动量块 max Re  正实部   能量块 max Re  正实部
        0.0   0.00    +4.09e-10      0/288    +5.70e+01      78/96
        1.0   2.667   +0.00e+00      0/288    +4.99e+01      40/96
        2.0   5.333   +0.00e+00      0/288    +4.83e+01      33/96
        4.0  10.667   +0.00e+00      0/288    +4.70e+01      33/96

    动量块与罚项无关（无罚项时就已严格耗散）；能量块的改善在 `BASE>=2`
    **饱和**。两张表合起来给出 `VISCOUS_IP_C_BASE = 2.0`（见该常量处的
    四条判据）。
    """
    if order < 0:
        raise ValueError(f"order={order} 不合法")
    return VISCOUS_IP_C_BASE * float((order + 1) * (order + 3)) / 3.0


@njit(cache=True, inline='always')
def smooth_nonnegative(x: float, band: float) -> float:
    """非解点处（界面通量点的外插值、过积分细点的插值）扩散系数的取用值：

        x <= 0          ->  0
        0 < x < band    ->  band t^2 (2 - t)，t = x / band     （三次 Hermite，两端值与一阶导数都接上）
        x >= band       ->  x

    一阶导数连续（0 处为 0，`band` 处为 1）；`band <= 0` 时退化为 `max(x, 0)`。`x >= band` 的值逐位不变，
    窄带里略小于原值（最多差 0.148 band）。数组版（numpy / cupy）是 `smooth_nonnegative_array`，导数是
    `smooth_nonnegative_derivative`。
    """
    if x <= 0.0:
        return 0.0
    if x >= band:
        return x
    t = x / band
    return band * t * t * (2.0 - t)


@njit(cache=True, inline='always')
def smooth_nonnegative_derivative(x: float, band: float) -> float:
    """`d smooth_nonnegative / dx`：0（x <= 0）、`t (4 - 3 t)`（窄带内）、1（x >= band）。"""
    if x <= 0.0:
        return 0.0
    if x >= band:
        return 1.0
    t = x / band
    return t * (4.0 - 3.0 * t)


def smooth_nonnegative_array(x, band: float, xp=np):
    """`smooth_nonnegative` 的数组版（同一个分段函数）。"""
    if band <= 0.0:
        return xp.maximum(x, 0.0)
    t = xp.clip(x / band, 0.0, 1.0)
    return xp.where(x >= band, x, band * t * t * (2.0 - t))


@njit(cache=True, inline='always')
def effective_eddy_viscosity(mu_t: float, mu: float) -> float:
    """非解点处涡粘的取用值：`smooth_nonnegative(mu_t, mu)`——负的外插值取 0，不小于分子粘度的原样，中间光滑
    过渡。层流（`mu_t` 恒为零）与充分湍流区逐位不变。为什么需要、为什么不用硬截断 `max(mu_t, 0)`，见
    `viscous_physical_flux_point` 文档"涡粘按非负取用"。数组版是 `effective_eddy_viscosity_array`。
    """
    return smooth_nonnegative(mu_t, mu)


def effective_eddy_viscosity_array(mu_t, mu: float, xp=np):
    """`effective_eddy_viscosity` 的数组版（GPU 与后处理用）。"""
    return smooth_nonnegative_array(mu_t, mu, xp)


@njit(cache=True, inline='always')
def viscous_physical_flux_point(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray,
    mu: float, Pr: float, mu_t: float, Pr_t: float,
) -> np.ndarray:
    """粘性物理通量，单点版。`mu_t` 是该点处涡粘场的值（可以是外插 / 插值出来的），在这里统一换成
    `effective_eddy_viscosity`；已经换算过的调用方（`face_point_jumps.viscous_common_flux_point` 的面平均）
    用 `viscous_physical_flux_point_effective`。
    """
    return viscous_physical_flux_point_effective(Q, grad_vel, grad_T, mu, Pr, effective_eddy_viscosity(mu_t, mu), Pr_t)


@njit(cache=True, inline='always')
def viscous_physical_flux_point_effective(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray,
    mu: float, Pr: float, mu_t: float, Pr_t: float,
) -> np.ndarray:
    """粘性物理通量，单点版（`mu_t` 已是取用值，原样使用）。

    Args:
        Q: (5,) (rho,u,v,w,p)
        grad_vel: (3,3)，grad_vel[i,j]=d(u_i)/d(x_j)
        grad_T: (3,)
        mu, Pr: 分子动力粘度/普朗特数（标量）
        mu_t, Pr_t: 湍流涡粘度/湍流普朗特数（标量，层流传 0.0/任意值）

    Returns:
        G: (3,5)，与 core/fr_residual/viscous_flux/pointwise.py::viscous_physical_flux 的公式
        逐一对应（质量分量恒为0；G[i,1+j]=tau[i,j]（对称）；
        G[i,4]=work[i]-q[i]）。

    ## 能量分量的符号（2026-09-26 修复的一阶物理缺陷）

    能量方程 `d(rho E)/dt + div((rho E + p) u) = div(tau.u) - div(q)`，Fourier
    定律 `q = -k grad(T)`。残差按 `dU/dt = -div F + div G` 组装（粘性项取正号），
    所以 `G_E = tau.u - q = tau.u + k grad(T)`。此前写成 `work + q`（即
    `tau.u - k grad(T)`），热传导整体反号、成了**反扩散**：均匀静止基态上纯粘性
    算子的线性化谱有 64 个正实部特征值、特征向量全部落在能量分量上，且加大 IP
    罚项 16 倍几乎不变（罚项本身符号正确，只能在跨面跳跃上部分抵消）；热斑上
    `d(rho E)/dt > 0`。plate_demo P1+SST 上它在锐边分离区（对流弱、湍流热传导
    `mu_t cp/Pr_t` 大）表现为等压下的密度/温度噪声持续放大（温度到 3.8e3 K、
    密度到 0.01），平均流 Newton 步在这些单元被物理性限幅冻住、全局残差回升。
    CPU 两份实现与 GPU 版同一次改正（GPU 版 2026-09-03 曾被"对齐"到这个错误
    符号，那次对照只验证了两边一致，没有对照物理方向）。

    ## 涡粘按非负取用（2026-10-11 修复的一阶物理缺陷）

    `mu_t` 在解点上非负，但本函数的求值点不只是解点：界面通量点上的值是单元内涡粘多项式的**外插**，过积分
    细点上是插值。涡粘在贴壁第一层里从接近零陡升到几十上百倍分子粘度，P1 线性外插到壁面就是负的，而且
    绝对值超过分子粘度——总粘度 `mu + mu_t < 0`，粘性通量与壁面罚项从耗散变成**反耗散**，往贴壁层里注入
    动量。物理上壁面的涡粘是零（SA-neg 对负的 `nu_tilde` 同样规定 `mu_t = 0`）。实测：
    * 平板（无滑移壁、第一层 1 mm、来流 30 m/s、SA P1 隐式稳态），来流粘性比 5 / 50 / 500：收敛后第一层内
      最大速度 30.6 / 78.3 / 90.4 m/s（残差都收敛到 6e-5，是"收敛的错解"）；取非负后 30.6 / 30.8 / 30.5；
    * cube_demo SA P1：稳态升阶后 20 / 40 步最大速度 82 / 162 m/s，接着算隐式瞬态第 4 / 10 步 176 / 307 m/s
      （来流 30，M 到 0.94，同一单元内解点速度反号）；|v| > 100 的 3398 个棱柱里约 2670 个是底面在壁面上的
      第一层。稳态 P1 停滞、双时间步 dt 上限 1e-4、内迭代逐步变差都是它的表现。
    取用值由 `effective_eddy_viscosity` 给出，施加在 `viscous_physical_flux_point`——界面、体积、解析 Jacobian
    的差分都经过的唯一逐点定义——不在各处外插之后分别补。

    **不用硬截断 `max(mu_t, 0)`**：它在零点有折角。同一个平板算例来流粘性比 500 时，硬截断给出物理的解
    （最大速度 30.8 m/s）但隐式稳态不收敛（残差停在 8e5，CFL 在 2~30 之间反复）；外插值在零附近的通量点让
    Newton 的线性化来回跳。
    """
    mu_total = mu + mu_t

    S00 = grad_vel[0, 0]
    S11 = grad_vel[1, 1]
    S22 = grad_vel[2, 2]
    S01 = 0.5 * (grad_vel[0, 1] + grad_vel[1, 0])
    S02 = 0.5 * (grad_vel[0, 2] + grad_vel[2, 0])
    S12 = 0.5 * (grad_vel[1, 2] + grad_vel[2, 1])
    div_u = grad_vel[0, 0] + grad_vel[1, 1] + grad_vel[2, 2]
    lam = -2.0 / 3.0 * mu_total

    tau00 = 2.0 * mu_total * S00 + lam * div_u
    tau11 = 2.0 * mu_total * S11 + lam * div_u
    tau22 = 2.0 * mu_total * S22 + lam * div_u
    tau01 = 2.0 * mu_total * S01
    tau02 = 2.0 * mu_total * S02
    tau12 = 2.0 * mu_total * S12

    k_cond = mu * CP_AIR / Pr + mu_t * CP_AIR / Pr_t
    qx = -k_cond * grad_T[0]
    qy = -k_cond * grad_T[1]
    qz = -k_cond * grad_T[2]

    u = Q[1]
    v = Q[2]
    w = Q[3]
    # work[j] = sum_i u_i * tau[i,j]，tau 对称
    work_x = u * tau00 + v * tau01 + w * tau02
    work_y = u * tau01 + v * tau11 + w * tau12
    work_z = u * tau02 + v * tau12 + w * tau22

    G = np.zeros((3, 5))
    G[0, 1] = tau00
    G[0, 2] = tau01
    G[0, 3] = tau02
    G[0, 4] = work_x - qx

    G[1, 1] = tau01
    G[1, 2] = tau11
    G[1, 3] = tau12
    G[1, 4] = work_y - qy

    G[2, 1] = tau02
    G[2, 2] = tau12
    G[2, 3] = tau22
    G[2, 4] = work_z - qz
    return G


@njit(cache=True, inline='always')
def viscous_ip_penalty_tilde(
    Q_o: np.ndarray, Q_other: np.ndarray, mu_total: float, k_total: float,
    h: float, adj_mag: float, c_ip: float,
) -> np.ndarray:
    """粘性 Interior Penalty (IP) 罚项，已转成 tilde（逆变）单位。

    ## 调用方式：内部面与边界面同一个公式（一个事实来源）

    * **内部面**：`k_total` 传**面平均**热传导率，`Q_other` 是邻居侧外插值。
    * **边界面**：`Q_other` 是幽灵态，`k_total` 按粘性边界种类给（`flux_kernels/viscous_bc.py`）：Dirichlet
      （远场、入口）传本侧热传导率，温度被弱施加到幽灵态温度；无滑移壁、镜像类（对称面 / 滑移壁）与出口传 0.0
      （绝热由 ∇T 法向镜像精确施加；出口的幽灵态速度就是本侧速度，跳跃为零，罚项只剩零）。边界面"另一侧"的梯度
      由 `boundary_other_gradients` 按种类给出：无滑移壁的速度梯度取本侧，`G_common` 与本侧通量的动量分量逐位
      相等，固壁的无滑移剪应力全部经本罚项弱施加。

    ## 长度尺度 `h`：面法向的单元厚度

    调用方传 `h = FlatFaceGeometry.ip_length[face]` = `min(V_owner, V_neighbor) / A_face`（边界面取 owner；`V` 是
    `get_all_cell_volumes()` 的精确体积，`A_face` 是 `true_area_weight` 逐面求和）。两侧单值，罚通量守恒。
    不能用 `mean(det_jacs)`（"体积 ÷ 参考单元体积"，随单元类型而变）或 `vol**(1/3)`（各向异性贴壁单元上是几何
    平均，实测平板贴壁单元比壁法向厚度大 2.48 倍，且随壁面层加密越来越偏）。

    ## 为什么内部面也必须加

    `compute_physical_gradient` 的签名是 `(field, mesh, ops)`——拿不到面数据，辅助变量 `sigma = grad(u)` 是
    纯单元内局部梯度，没有 BR1 要求的提升项 `lift({u} - u|_dK)`；界面耦合只有粘性通量的两侧算术平均。在
    Arnold-Brezzi-Cockburn-Marini (2002) 的统一框架里这是"提升项与罚项都为零"的那一档，不满足强制性：离散
    扩散算子不控制跨面跳跃。实测均匀基态纯粘性算子的数值雅可比谱（不加内部面罚项）：最大实部 +1.16e2、最小
    -5.94e1，最不稳定特征向量全部落在能量分量上（热传导那一支）；纯扩散算子本应全部 `Re(lambda) <= 0`。
    Couette 与均匀流场探不到这一点：精确解连续且可精确表示，跨面跳跃恒为零。

    **P0 是同一缺陷的极端形态**：分片常数的局部梯度恒为零，`G_common` 与 `G_own` 全分量恒为零——罚项是 P0 唯一
    的粘性耦合（形式上等价于有限体积的两点扩散通量）。

    ## 能量分量的两项

        pen[4] = -(eta_v*{u}.[[u]] + eta_T*[[T]]) * adj_mag

    * `eta_T*[[T]]`：热传导那一支的罚项，系数按 IP 惯例用**热传导率**
      `k = mu*cp/Pr + mu_t*cp/Pr_t`（不是 mu）。
    * `eta_v*{u}.[[u]]`：动量罚项做的功。把罚项看成一份附加应力
      `tau_pen = eta_v*[[u]]`，它对能量方程的贡献就是 `{u}.tau_pen`——
      与物理通量里 `u.tau` 那一项同构。不加它，动量罚项做的功在能量方程
      里没有对应，总能量不闭合。
    * 边界面同样叠加这一项：静止无滑移壁 `{u}=0`、镜像类 `{u}` 纯切向而 `[[u]]` 纯法向，两处都恒为零；只有
      远场/入口流入点上它非零，而那里罚项牵引确实在做功。

    温度给定的边界（远场、入口流入点）必须加温度罚项：∇T 取内部值时公共法向热通量就等于本单元自己的热通量，
    FR 修正为零，那个面上没有热边界条件，弱形式留下不定号的 `-∮ T k∂T/∂n`，热传导算子失去强制性（见
    `flux_kernels/viscous_bc.py` 模块文档）。

    Args:
        Q_o: (5,) 面上本侧原始变量外插值 (rho,u,v,w,p)
        Q_other: (5,) 对侧原始变量 —— 边界面是幽灵态（含真实 BC，如 WALL
            无滑移镜像），内部面是邻居侧在同一批 FP 上的外插值
        mu_total: 分子+湍流动力粘度之和（边界面取本侧值、内部面取面平均，
            与 `G_common` 用的 `mut_avg` 保持一致）
        k_total: 热传导率 `mu*cp/Pr + mu_t*cp/Pr_t`；绝热类边界面传 0.0
            （见上面"调用方式"）
        h: 面法向的单元厚度 `ip_length`（见上面"长度尺度"一节）
        adj_mag: 该 FP 处本侧逆变行范数（`adj_row` 已外向定向，不再乘定向因子）
        c_ip: 罚项常数，`resolve_viscous_ip_constant(order)`（随阶数增长，见该函数）

    Returns:
        pen: (5,)。`[1:4]` 是动量分量，`[4]` 是能量分量。可直接
        `G_tilde_common[v] += pen[v]` for v in range(1, 5)
    """
    pen = np.zeros(5)
    h_safe = h
    if h_safe < 1e-300:
        h_safe = 1e-300
    eta = c_ip * mu_total / h_safe
    scale = eta * adj_mag
    for v in range(1, 4):
        pen[v] = -scale * (Q_o[v] - Q_other[v])
    work = 0.0
    for v in range(1, 4):
        work += 0.5 * (Q_o[v] + Q_other[v]) * (Q_o[v] - Q_other[v])
    eta_T = c_ip * k_total / h_safe
    scale_T = eta_T * adj_mag
    T_o = Q_o[4] / (Q_o[0] * R_AIR)
    T_other = Q_other[4] / (Q_other[0] * R_AIR)
    pen[4] = -scale * work - scale_T * (T_o - T_other)
    return pen


@njit(cache=True, parallel=True)
def viscous_physical_flux_batch(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray,
    mu: float, Pr: float, mu_t: np.ndarray, Pr_t: float,
) -> np.ndarray:
    """`viscous_physical_flux_point` 的批量版：Q (N,5), grad_vel (N,3,3),
    grad_T (N,3), mu_t (N,) -> G (N,3,5)。理由同 `euler_physical_flux_batch`
    （含多核并行说明——同样是纯 gather，无需私有缓冲区）。
    """
    n = Q.shape[0]
    G = np.zeros((n, 3, 5))
    for i in prange(n):
        G[i] = viscous_physical_flux_point(Q[i], grad_vel[i], grad_T[i], mu, Pr, mu_t[i], Pr_t)
    return G
