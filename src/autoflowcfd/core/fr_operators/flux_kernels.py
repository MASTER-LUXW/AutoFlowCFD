"""欧拉/粘性物理通量的逐点标量 numba 版本 (性能优化配套)。

`core/fr_residual_inviscid.py::euler_physical_flux` 和
`core/fr_viscous_flux.py::viscous_physical_flux` 是向量化 numpy 实现
（`np.stack`/`np.zeros`/`np.swapaxes`/`np.eye`/`einsum`），在
`fr_residual_inviscid_kernel.py`/`fr_viscous_flux_kernel.py` 的逐点
numba `@njit` 主循环里会被反复调用（每个 Flux Point 调一次）——numba
nopython 模式不支持 `einsum`/`swapaxes`，所以不能直接复用，必须重新写
逐点标量版。这是整个性能优化里除了 AUSM+up 之外风险最高的新代码
（尤其 `viscous_physical_flux_point` 涉及真实物理：Boussinesq 假设下
`mu_total=mu+mu_t` 统一处理应力张量、`k_cond` 混合分子/湍流普朗特数、
`work=vel·tau` 粘性功）——因此单独在这里、用随机输入与现有向量化实现
逐位对比验证（见 `tests/unit/test_fr_flux_kernels_pointwise.py`），不
把这一步的验证并入端到端残差对比，出问题能立刻定位到这里而不是别处。

两个函数的公式必须与 `fr_residual_inviscid.py::euler_physical_flux`/
`fr_viscous_flux.py::viscous_physical_flux` 严格一致，改动前者时必须
同步检查后者是否也要改。
"""

import numpy as np
from numba import njit, prange

GAMMA = 1.4
R_AIR = 287.0  # 空气比气体常数 J/(kg*K)，须与 fr_viscous_flux.py 保持一致

#: 粘性 Interior Penalty 罚项常数。标准 DG 惯例取 O(1)~O(10)，理论要求
#: `c > C_trace(p)`（3D P1 的 trace 常数约 2.7）。**边界面与内部面用同一个
#: 值**：两者是同一个罚项、同一套量纲推导，没有理由给两个数。
#:
#: 此前这个常数在 `fr_residual/viscous_flux_kernel.py`、
#: `fr_residual/viscous_p0_kernel.py`、`gpu/residual/gpu_viscous.py` **各有
#: 一份独立的 `= 4.0`**，GPU 那边还把罚项公式整个抄了一遍。本项目已多次因
#: "两份实现只改了一份"出真实缺陷（滤波档双解析器、CFL 三处硬编码兜底、
#: 配置层与 CLI 相差 20 倍），所以统一到这里、由各 kernel 导入。
#: **实测标定值（2026-09-23）**，见 `resolve_viscous_ip_constant` 文档里的
#: 标定表。取 2.0 的四条判据：
#:   ① 理论硬要求 `c_ip > C_trace = (p+1)(p+3)/3`（3D P1 = 2.667）——
#:      BASE=1.0 给出的 c_ip 恰好等于下界、零余量，作为稳定化参数不可取；
#:   ② 长窗口精度：Blasius 4000 步最差偏离 0.0536（BASE=4.0 是 0.0581）；
#:   ③ 能量块谱改善在 `BASE>=2` 已饱和（33/96，与 BASE=4.0 相同），
#:      再加大没有收益；
#:   ④ 刚性因子 17（BASE=4.0 是 33）—— 虽然真实网格上实测代价为零
#:      （粘性从来不是约束方），但没有理由白付。
VISCOUS_IP_C_BASE = 2.0


def resolve_viscous_ip_constant(order: int) -> float:
    """按多项式阶数给出 IP 罚项常数 `c_ip`。

    ## 为什么必须随阶数增长

    IP 罚项要保证强制性（coercivity），常数必须压过**迹不等式常数**
    （trace inequality）：对 d 维单元上的 p 次多项式，
    `||v||_{dK}^2 <= C_tr * (p+1)(p+d)/d * |dK|/|K| * ||v||_K^2`。
    所以 `c_ip` 必须 `~ (p+1)(p+d)/d`；3D 下即 `(p+1)(p+3)/3`
    （P0 1.0、P1 2.67、P2 5.0、P3 8.0 —— 以 P0 归一后是 1 / 2.67 / 5 / 8）。

    固定用一个与阶数无关的数是旧实现的真实缺口（原文档自己标着"未做
    多项式阶数相关的最优 trace-inequality 常数标定"）。

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

#: 定压比热 J/(kg*K)。此前在 `viscous_physical_flux_point` 里逐点算一次
#: `cp = GAMMA * R_AIR / (GAMMA - 1.0)`；内部面罚项的热传导系数要用同一个
#: cp，所以提到模块级（同一个表达式在导入期求值一次，逐位相同）。
CP_AIR = GAMMA * R_AIR / (GAMMA - 1.0)


@njit(cache=True)
def euler_physical_flux_point(Q: np.ndarray) -> np.ndarray:
    """欧拉物理通量，单点版。Q=(rho,u,v,w,p) -> F，形状 (3,5)。

    与 fr_residual_inviscid.py::euler_physical_flux 的公式逐一对应。
    """
    rho = Q[0]
    u = Q[1]
    v = Q[2]
    w = Q[3]
    p = Q[4]
    rho_safe = max(rho, 1e-10)

    ke = 0.5 * (u * u + v * v + w * w)
    e_internal = p / ((GAMMA - 1.0) * rho_safe)
    rhoE = rho * (e_internal + ke)
    H = (rhoE + p) / rho_safe

    mf0 = rho * u
    mf1 = rho * v
    mf2 = rho * w

    F = np.zeros((3, 5))
    F[0, 0] = mf0
    F[0, 1] = mf0 * u + p
    F[0, 2] = mf0 * v
    F[0, 3] = mf0 * w
    F[0, 4] = rho * H * u

    F[1, 0] = mf1
    F[1, 1] = mf1 * u
    F[1, 2] = mf1 * v + p
    F[1, 3] = mf1 * w
    F[1, 4] = rho * H * v

    F[2, 0] = mf2
    F[2, 1] = mf2 * u
    F[2, 2] = mf2 * v
    F[2, 3] = mf2 * w + p
    F[2, 4] = rho * H * w
    return F


@njit(cache=True, inline='always')
def viscous_physical_flux_point(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray,
    mu: float, Pr: float, mu_t: float, Pr_t: float,
) -> np.ndarray:
    """粘性物理通量，单点版。

    Args:
        Q: (5,) (rho,u,v,w,p)
        grad_vel: (3,3)，grad_vel[i,j]=d(u_i)/d(x_j)
        grad_T: (3,)
        mu, Pr: 分子动力粘度/普朗特数（标量）
        mu_t, Pr_t: 湍流涡粘度/湍流普朗特数（标量，层流传 0.0/任意值）

    Returns:
        G: (3,5)，与 fr_viscous_flux.py::viscous_physical_flux 的公式
        逐一对应（质量分量恒为0；G[i,1+j]=tau[i,j]（对称）；
        G[i,4]=work[i]+q[i]）。
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
    G[0, 4] = work_x + qx

    G[1, 1] = tau01
    G[1, 2] = tau11
    G[1, 3] = tau12
    G[1, 4] = work_y + qy

    G[2, 1] = tau02
    G[2, 2] = tau12
    G[2, 3] = tau22
    G[2, 4] = work_z + qz
    return G


@njit(cache=True, inline='always')
def mirror_normal_component(g: np.ndarray, adj_row: np.ndarray) -> np.ndarray:
    """把向量 g 关于面（法向由 `adj_row` 给出方向）做**法向分量镜像**：

        g_mirror = g - 2 (g·n) n,    n = adj_row / |adj_row|

    用于 BR1 边界面的温度梯度（见 boundary/fr_ghost_state.py::
    ADIABATIC_THERMAL_BC_TYPES）。取 BR1 面平均后

        g_avg = 0.5 (g + g_mirror) = g - (g·n) n

    其法向分量**精确为零**，于是投影到该面的离散传导热通量
    `a·q_avg = -k (adj_row·∇T_avg)` 恒等于零——这是绝热/对称壁的精确
    离散表述，不是"近似到某个容差"。

    方向说明：这里用的是 `adj_row`（逆变行，即 det(J)∇ξ，与面法向平行），
    **不是** `true_normal`。两者平行，但用 `adj_row` 才能保证"恒等于零"
    的是真正进入残差的那个投影量 `a0*q_x+a1*q_y+a2*q_z` 本身，而不是
    一个与之只差截断误差的替代量。镜像对 n 取反不变，因此内/外法向
    朝向约定无关紧要。

    退化保护：`|adj_row|` 为零（退化面）时原样返回 g——此时该面的通量
    投影本来就是零，镜不镜像都不影响结果。
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
def viscous_ip_penalty_tilde(
    Q_o: np.ndarray, Q_other: np.ndarray, mu_total: float, k_total: float,
    h: float, adj_mag: float, side: float, c_ip: float,
    include_work: bool,
) -> np.ndarray:
    """粘性 Interior Penalty (IP) 罚项，已转成 tilde（逆变）单位。

    ## 两种调用方式，同一个公式（一个事实来源）

    * **边界面**：`k_total=0.0`、`include_work=False`，`Q_other` 是幽灵态。
      能量分量恒为 `0.0`，退化成 2026-09-15 起的既有行为。
    * **内部面**：`k_total` 传**面平均**热传导率、`include_work=True`，
      `Q_other` 是邻居侧外插值。见下面"为什么内部面也必须加"。

    ## 长度尺度 `h`：由调用方给出，**不再是 `vol**(1/3)`**（2026-09-23 修复）

    此前本函数内部算 `h = vol**(1/3)`，而调用方传的 `vol` 是
    `mean(det_jacs)`。两处都错：

    1. `mean(det_jacs)` **不是单元体积**，是"体积 ÷ 参考单元体积"，而参考
       体积是**基相关**的（坍缩张量积立方体 8、原生棱柱 4、原生四面体
       4/3）。同一个物理单元在两条基下因此拿到相差 2 倍（四面体 6 倍）的
       `vol`，罚项强度相差 21%（四面体 45%）。
       `HighOrderMesh.get_all_cell_volumes()` 的文档里早就写明
       "det(J)均值*8 是错的、已为 CFL/网格尺度那条路径改成正确的加权
       积分"——**罚项这个消费点被漏掉了**。
    2. 各向异性贴壁单元上 `vol**(1/3)` 是几何平均，远大于壁法向间距。IP
       罚项的标准长度尺度是**面法向的单元厚度** `vol / A_face`（直棱柱
       贴壁单元上恰好等于 `dy`）。实测平板算例贴壁单元
       `vol**(1/3) = 3.0457e-02` vs `vol/A_face = 1.2275e-02`，**大 2.48
       倍**，且该偏差随壁面层加密越来越严重（`dy` 减半时 `vol/A` 减半，
       而 `vol**(1/3)` 只降 1.26 倍）。

    现在由调用方传 `h = cell_volume[cell] / face_area[face]`，其中
    `cell_volume` 取 `get_all_cell_volumes()` 那份正确实现、`face_area` 取
    `true_area_weight` 逐面求和（该和等于物理面积这一点已独立验证：平板
    算例六个边界平面与解析面积之比全部 1.000000）。

    ## 为什么内部面也必须加（2026-09-23，真实缺陷修复）

    本项目的粘性离散被文档称作 BR1，但 `compute_physical_gradient` 的签名
    是 `(field, mesh, ops)` —— 它**拿不到任何面数据**，所以辅助变量
    `sigma = grad(u)` 是**纯单元内局部梯度**，没有 BR1 要求的提升项
    `lift({u} - u|_dK)`。界面耦合只存在于第二个方程（粘性通量取两侧算术
    平均），而内部面**没有任何跳跃罚项**。在 Arnold-Brezzi-Cockburn-Marini
    (2002) 的统一框架里这是"提升项与罚项都为零"的那一档 —— 它不满足强制性
    （coercivity）：离散扩散算子完全不控制跨面跳跃。

    **实测**（均匀基态、纯粘性算子的数值雅可比谱，两个差分步长逐位相同、
    均匀基态与 Blasius 基态逐位相同）：

        基         自由度  max Re(lambda)  无量纲 Re*h^2/nu  min Re(lambda)
        native     2160    +1.1553e+02     +92.8            -5.9374e+01
        collapsed  2880    +9.2344e+02     +742.1           -5.2781e+02

    纯扩散算子必须**全部** `Re(lambda) <= 0`；这里正的一侧比负的一侧还大，
    最不稳定特征向量 100% 落在能量分量上（两条基都是），即 BR1 热传导那一支。

    为什么既有验证算例探不到：**Couette 的精确解连续且可精确表示**，跨面
    跳跃恒为零，缺失的提升项与罚项贡献都恰好是零 —— "Couette 精确到
    8.4e-9"对这条路径没有约束力；均匀流场同理（梯度恒零）。

    **P0 是同一缺陷的极端形态**：分片常数的局部梯度恒为零，于是 `G_common`
    与 `G_own` 全分量恒等于零 —— 内部面粘性通量**恒为零**，P0 阶单元之间
    完全没有粘性扩散。罚项正是 P0 唯一可能的粘性耦合（形式上等价于有限
    体积的两点扩散通量）。

    ## 能量分量的两项

        pen[4] = -(eta_v*{u}.[[u]] + eta_T*[[T]]) * adj_mag * side

    * `eta_T*[[T]]`：热传导那一支的罚项，系数按 IP 惯例用**热传导率**
      `k = mu*cp/Pr + mu_t*cp/Pr_t`（不是 mu）。
    * `eta_v*{u}.[[u]]`：动量罚项做的功。把罚项看成一份附加应力
      `tau_pen = eta_v*[[u]]`，它对能量方程的贡献就是 `{u}.tau_pen` ——
      与物理通量里 `u.tau` 那一项同构。不加它，动量罚项做的功在能量方程
      里没有对应，总能量不闭合。
    * **静止无滑移壁上这一项恒为零**（`{u} = 0`），所以既有边界实现
      "只有动量分量"在静止壁上本来就是一致的；`include_work=False` 保留
      那个行为，避免改动已验证的 FARFIELD/INLET/OUTLET 边界。

    根因：`viscous_physical_flux_point` 算出的应力张量 tau 只依赖速度梯度
    `grad_vel`，不依赖状态 `Q` 本身；而边界面的梯度按本代码库既定策略镜像
    内部值（`gv_ghost=gv_owner`，见 viscous_flux_kernel.py 模块文档"边界面
    梯度处理"一节），于是 `G_common` 与 `G_own` 的动量分量在任意边界条件
    下逐位相等——`jump_owner` 恒为零，等价于固壁上无滑移剪应力不存在
    （数值验证：WALL/SLIP_WALL/FARFIELD 给出逐位相同的动量残差，V2.0
    专家组评审新发现的阻塞级问题）。

    标准 DG/FR 文献（S-03 允许的 "IP" 方案，与 LDG 并列）对此的解法是在
    共同数值粘性通量里补一个正比于状态跳跃 [[u]]=u_owner-u_ghost 的耗散
    罚项（Interior Penalty / SIPG，Arnold et al. 2002 统一分析框架；系数
    形式取自 Shahbazi (2005) 的标准 penalty parameter η=C·μ/h）：

        G_num·n = {G(∇u)}·n - η·[[u]]，η = c_ip·μ_eff/h

    h 用本项目自己在粘性 CFL（core/fr_solver/cfl.py::_compute_local_time_step
    的 `dt_visc ∝ V^(2/3)/mu_eff` 隐含的长度尺度约定）里已经采用的
    "cell volume^(1/3)" 做局部特征长度，不引入新的长度尺度定义，量纲上
    `mu_total*(Δu)/h` 与 tau 同为 Pa；再乘以 `adj_mag*oside`
    （与本文件其余通量把物理量转成 tilde/逆变量的方式完全一致，
    见 fr_residual_inviscid_kernel.py 里 `F_tilde_common = F_common_n *
    adj_mag * oside` 的同一套惯例）得到可以直接叠加进 `G_tilde_common`
    的量。

    **2026-09-23 更正**：上面这段原文写的是"只在边界面调用……内部面两侧的
    梯度本就是各自独立算出的真实局部梯度（不是镜像），已有非零、物理有
    意义的耦合，不属于本次修复范围，不额外加罚项"。那个理由**为真但不
    充分**，结论是错的：BR1 的病不是"内部面耦合为零"，而是**不控制跨面
    跳跃**（缺强制性）。实测均匀基态纯粘性算子谱有正实部 328/2160，纯扩散
    算子本应全部 <= 0。现在内部面也加罚项（`k_total` 传面平均热传导率、
    `include_work=True`），完整依据见本函数开头"为什么内部面也必须加"。

    **为什么"边界面"这一档只有动量分量（2026-09-15 结论，2026-09-23
    复核仍然成立，仅适用范围收窄到边界面）**：IP 罚项在边界面存在的理由是
    "梯度被镜像 ⇒ 该分量的跳跃恒为零 ⇒ Dirichlet 型边界条件在扩散算子里
    完全没被施加"。对能量方程，这个理由按热边界类型逐类检查后都不成立
    （所以边界面传 `k_total=0.0`）——**但这条论证只覆盖边界面**，内部面
    的能量跳跃既非镜像也非零，强制性要求它必须被罚，见上面那段更正：

    - **绝热类（WALL/SYMMETRY）**：正确的边界条件是 q_n = 0，是
      Neumann 型而不是 Dirichlet 型——它已经由 ∇T 的法向分量镜像**精确**
      施加（见 `mirror_normal_component` 与 boundary/fr_ghost_state.py::
      ADIABATIC_THERMAL_BC_TYPES），罚项在这里无事可做，加了反而是往
      "零热通量"这个恒等式上叠加一个非零项。
    - **透射类（INLET/OUTLET/FARFIELD）**：∇T 不镜像、取内部值，能量
      跳跃项本来就非零，不存在"约束没被施加"的问题。这些边界位于远场、
      对流主导（Pe>>1），温度由无粘特征通量携带的 ghost 态施加。
    - **等温壁**：这才是真正需要能量罚项（η_T=c·k_eff/h 乘以 [[T]]，注意
      系数是热传导率 k=μ·cp/Pr 而不是 μ）的情形——而**本项目没有等温壁
      BC，也没有壁面热通量模型**（见 fr_ghost_state.py::wall_ghost_state
      的"热边界条件"一节）。将来若新增带传热的壁面类型，必须在这里同时
      补上对应的能量罚项，否则壁面温度条件在扩散算子里不会被施加。

    Args:
        Q_o: (5,) 面上本侧原始变量外插值 (rho,u,v,w,p)
        Q_other: (5,) 对侧原始变量 —— 边界面是幽灵态（含真实 BC，如 WALL
            无滑移镜像），内部面是邻居侧在同一批 FP 上的外插值
        mu_total: 分子+湍流动力粘度之和（边界面取本侧值、内部面取面平均，
            与 `G_common` 用的 `mut_avg` 保持一致）
        k_total: 热传导率 `mu*cp/Pr + mu_t*cp/Pr_t`。**边界面传 0.0**
            （理由见下面"为什么只有动量分量"一节：绝热壁是 Neumann 型、
            已由 ∇T 法向镜像精确施加，加 Dirichlet 型罚项反而是错的）
        h: 面法向的单元厚度 `cell_volume / face_area`（见上面"长度尺度"
            一节；**不要再传 `mean(det_jacs)` 或任何 `vol**(1/3)`**）
        adj_mag: 该 FP 处本侧逆变行范数（与本文件其余处一致的度量量）
        side: side 因子。**坍缩面传 `owner_side`/`neighbor_side`，原生面
            （cube face 编码 >= 6）必须传 +1** —— 原生面的 `adj_row` 已按
            outward 定向，再乘一次 side 会让 `side = -1` 的面上罚项反号、
            从耗散变成往单元注入动量（2026-09-22 修复的真实缺陷，见
            `viscous_flux_kernel.py` 里 `pen_side_o` 那段注释）
        include_work: 是否叠加动量罚项做的功到能量分量（内部面 True、
            边界面 False，理由见上面"能量分量的两项"）
        c_ip: 罚项常数（标准 DG 惯例取 O(1)~O(10)，本实现固定用 4.0，
            未做多项式阶数相关的最优 trace-inequality 常数标定——这是
            稳定性调优参数，不影响"罚项存在与否/符号是否耗散"这一
            正确性核心，若未来观测到边界层数值振荡可调大）

    Returns:
        pen: (5,)。`[1:4]` 是动量分量，`[4]` 是能量分量（`k_total=0.0` 且
        `include_work=False` 时恒为 `0.0`）。可直接
        `G_tilde_common[v] += pen[v]` for v in range(1, 5)
    """
    pen = np.zeros(5)
    h_safe = h
    if h_safe < 1e-300:
        h_safe = 1e-300
    eta = c_ip * mu_total / h_safe
    scale = eta * adj_mag * side
    for v in range(1, 4):
        pen[v] = -scale * (Q_o[v] - Q_other[v])
    work = 0.0
    if include_work:
        for v in range(1, 4):
            work += 0.5 * (Q_o[v] + Q_other[v]) * (Q_o[v] - Q_other[v])
    eta_T = c_ip * k_total / h_safe
    scale_T = eta_T * adj_mag * side
    T_o = Q_o[4] / (Q_o[0] * R_AIR)
    T_other = Q_other[4] / (Q_other[0] * R_AIR)
    pen[4] = -scale * work - scale_T * (T_o - T_other)
    return pen


@njit(cache=True, parallel=True)
def euler_physical_flux_batch(Q: np.ndarray) -> np.ndarray:
    """`euler_physical_flux_point` 的批量版：Q (N,5) -> F (N,3,5)。

    体积项性能优化配套（原体积项调用的是 `fr_residual_inviscid.py::
    euler_physical_flux` 的向量化 numpy 实现，逐点重复分配 `np.zeros`
    大数组+`np.stack`，是新的性能瓶颈来源之一，见 py-spy 对生产网格的
    实测采样）。直接复用已经逐位验证过的 `euler_physical_flux_point`，
    不是新公式，只是换一种循环方式；调用方负责把任意形状的
    `(...,5)` 输入展平成 `(N,5)` 再调用，输出展平成 `(N,3,5)` 后自行
    reshape 回原始前导维度。

    多核并行（阶段二）：这是纯 gather——每次迭代 i 只写自己的输出行
    `F[i]`，不同 i 之间零索引冲突，`prange` 直接安全，不需要像两个
    界面 kernel（fr_residual_inviscid_kernel.py/fr_viscous_flux_kernel.py）
    那样用私有缓冲区+归约处理 scatter-add。线程数由 numba 运行时环境
    （`numba.set_num_threads`，求解器启动时设置一次）决定，这里不接收
    也不查询线程数参数。
    """
    n = Q.shape[0]
    F = np.zeros((n, 3, 5))
    for i in prange(n):
        F[i] = euler_physical_flux_point(Q[i])
    return F


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


@njit(cache=True, inline='always')
def _log_mean_point(a: float, b: float) -> float:
    """Ismail & Roe (2009) 数值稳定对数平均：ln_mean(a,b) = (a-b)/ln(a/b)，
    a≈b 时用泰勒展开避免 0/0（与
    `8_算法重构-Entropy-Stable_Split-Form通量重构-Part1/2.md` 决定性
    验证脚本 `log_mean` 同一公式，这里是单点标量版供 numba 逐点核使用）。
    """
    xi = a / b
    f = (xi - 1.0) / (xi + 1.0)
    u = f * f
    if u < 1e-4:
        F = 1.0 + u / 3.0 + u * u / 5.0 + u * u * u / 7.0
    else:
        F = np.log(xi) / (2.0 * f) if abs(f) > 1e-300 else np.log(xi)
    return (a + b) / (2.0 * F)


@njit(cache=True, inline='always')
def chandrashekar_flux_point(QL: np.ndarray, QR: np.ndarray) -> np.ndarray:
    """Chandrashekar (2013) 熵守恒 + 动能守恒两点数值通量，单点对版。

    QL, QR: (5,) 原始变量 (rho,u,v,w,p)。
    Returns: (3,5)，三个物理方向的两点通量向量（与
    `euler_physical_flux_point` 输出形状一致，QL==QR 时代数上精确退化
    为 `euler_physical_flux_point(QL)`——两点通量的标准一致性要求）。

    公式来源与验证：`8_算法重构-Entropy-Stable_Split-Form通量重构-
    Part1.md` 二、4 节（对照 arXiv:1209.4994 核实），已在
    `8_算法重构-Entropy-Stable_Split-Form通量重构-Part2.md` 用真实
    生产 `D_3d_tet` 矩阵决定性验证（180 个随机四面体样本，P1/P2/P3，
    与对称平均度量项 `{{adj(J)}}_ij=0.5*(adj_i+adj_j)` 配合使用，见
    `core/fr_residual/inviscid.py::compute_entropy_stable_volume_divergence`
    调用处）。
    """
    rhoL, uL, vL, wL, pL = QL[0], QL[1], QL[2], QL[3], QL[4]
    rhoR, uR, vR, wR, pR = QR[0], QR[1], QR[2], QR[3], QR[4]

    betaL = rhoL / (2.0 * pL)
    betaR = rhoR / (2.0 * pR)
    rho_ln = _log_mean_point(rhoL, rhoR)
    beta_ln = _log_mean_point(betaL, betaR)
    rho_bar = 0.5 * (rhoL + rhoR)
    beta_bar = 0.5 * (betaL + betaR)
    u_bar = 0.5 * (uL + uR)
    v_bar = 0.5 * (vL + vR)
    w_bar = 0.5 * (wL + wR)
    p_tilde = rho_bar / (2.0 * beta_bar)
    ke_bar = 0.5 * (u_bar * u_bar + v_bar * v_bar + w_bar * w_bar)
    e_term = 1.0 / (2.0 * (GAMMA - 1.0) * beta_ln) - ke_bar

    F = np.zeros((3, 5))
    # x 方向
    f_rho = rho_ln * u_bar
    fx1 = u_bar * f_rho + p_tilde
    fx2 = v_bar * f_rho
    fx3 = w_bar * f_rho
    F[0, 0] = f_rho
    F[0, 1] = fx1
    F[0, 2] = fx2
    F[0, 3] = fx3
    F[0, 4] = e_term * f_rho + u_bar * fx1 + v_bar * fx2 + w_bar * fx3
    # y 方向
    f_rho = rho_ln * v_bar
    fy1 = u_bar * f_rho
    fy2 = v_bar * f_rho + p_tilde
    fy3 = w_bar * f_rho
    F[1, 0] = f_rho
    F[1, 1] = fy1
    F[1, 2] = fy2
    F[1, 3] = fy3
    F[1, 4] = e_term * f_rho + u_bar * fy1 + v_bar * fy2 + w_bar * fy3
    # z 方向
    f_rho = rho_ln * w_bar
    fz1 = u_bar * f_rho
    fz2 = v_bar * f_rho
    fz3 = w_bar * f_rho + p_tilde
    F[2, 0] = f_rho
    F[2, 1] = fz1
    F[2, 2] = fz2
    F[2, 3] = fz3
    F[2, 4] = e_term * f_rho + u_bar * fz1 + v_bar * fz2 + w_bar * fz3
    return F


@njit(cache=True, parallel=True)
def entropy_stable_volume_divergence_batch(
    Q: np.ndarray, adj_j: np.ndarray, D_fine: np.ndarray
) -> np.ndarray:
    """体积项散度，entropy-stable/split-form 版本（逐单元、逐 SP 对
    two-point flux + 对称平均度量项，见
    `8_算法重构-Entropy-Stable_Split-Form通量重构-Part1/2.md` 完整推导，
    张量收缩约定与生产强形式 `np.matmul(adj_j, F_phys)` 逐一对应）。

    Args:
        Q: (n_cells, n_fine, 5) 原始变量，过积分 FINE 点集上
        adj_j: (n_cells, n_fine, 3, 3) 几何度量项（det(J)*inv(J)），FINE 点
        D_fine: (n_fine, n_fine, 3) FINE 点集自身的微分矩阵，单元间共享

    Returns:
        div_comp: (n_cells, n_fine, 5)，与强形式 `contract_shared_
        operator_2axis(D_fine, np.matmul(adj_j, F_phys))` 同一个量，
        供调用方按同样方式 `restrict_f2c @ div_comp` 后除以 coarse
        det(J)。

    复杂度是 O(n_cells * n_fine^2)（两点通量需要遍历 SP 对），比强形式的
    O(n_cells * n_fine) 更贵——这是 entropy-stable 方案的固有代价（两点
    通量结构性要求，不是实现效率问题），只作为可选项（默认关闭）供用户
    在能接受这个额外开销时启用。用 `prange` 按单元并行、内层 (i,j) 双重
    循环逐对累加而不是先构造 (n_fine,n_fine,...) 的密集张量，把每个单元
    的峰值内存控制在 O(n_fine)（几十~一百多个 FP 量级），避免 O(n_fine^2)
    的中间数组在生产网格规模（数十万单元）下引发内存问题（这类问题此前
    在 P2 SST 输运项上真实出现过，见项目记忆 `p2_sst_performance_and_
    oom_fixes`）。
    """
    n_cells, n_fine, _ = Q.shape
    div_comp = np.zeros((n_cells, n_fine, 5))
    for c in prange(n_cells):
        for i in range(n_fine):
            Qi = Q[c, i]
            adj_i = adj_j[c, i]
            acc = np.zeros(5)
            for j in range(n_fine):
                Qj = Q[c, j]
                adj_jj = adj_j[c, j]
                F_pair = chandrashekar_flux_point(Qi, Qj)  # (3,5)
                for m in range(3):
                    coeff = D_fine[i, j, m]
                    if coeff == 0.0:
                        continue
                    for v in range(5):
                        s = 0.0
                        for cc in range(3):
                            adj_sym = 0.5 * (adj_i[m, cc] + adj_jj[m, cc])
                            s += adj_sym * F_pair[cc, v]
                        acc[v] += coeff * s
            div_comp[c, i] = 2.0 * acc
    return div_comp
