"""AutoFlowCFD V2.0 - 粘性物理通量、边界梯度镜像与 IP 罚项

从 `src/autoflowcfd/core/fr_operators/flux_kernels.py`(原 615 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np

from numba import njit, prange
from .constants import CP_AIR, R_AIR, VISCOUS_IP_C_BASE


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
