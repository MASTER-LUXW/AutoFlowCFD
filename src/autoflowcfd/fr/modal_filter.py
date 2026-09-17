"""
AutoFlowCFD V2.0 - 坍缩坐标单元指数模态滤波器 (Tier-0 修复)

背景（2026-08-14 Couette 合成算例定量验证过程中定位）：真实复现——对
精确解析解（Couette 线性剪切场，无粘残差理论上处处严格为零）施加一次
机器精度量级（~1e-9 相对幅度）的随机扰动后，无粘残差立刻放大到 1e-5~
1e-2 量级（放大倍数约 17~45 倍/次微分）；粘性残差因涉及"先求梯度、
再求散度"两次微分，放大倍数约为无粘的平方量级。这个放大本身不是
bug——是任何节点配置（nodal collocation）谱型/DG 方法的已知固有属性
（高阶模态对应的插值多项式在退化边附近条件数更差，参见 Hesthaven &
Warburton《Nodal DG Methods》Ch.5"混叠不稳定性"一节；Boyd《Chebyshev
and Fourier Spectral Methods》Ch.11 "去混叠与滤波"一节）。但若不加抑制，
这个放大在显式时间推进里逐步复合：本次真实复现（棱柱网格与四面体网格
均复现），残差在连续几步内从 ~1e-6 量级爆炸式增长到 1e100+ 量级并
NaN——即使把时间步长人为缩小到原来的 1/100，也只能推迟 2~3 步发生，
不能消除，证实这不是 CFL 稳定性余量问题，是真正的（与步长无关的）
混叠驱动不稳定性。

`suppress_residual_outliers`（fr_troubled_cell.py 机制3）无法压制这种
失稳——因为噪声放大是同一个单元里*大面积、渐变分布*的（不是单个 SP
异常凸出），跟同单元其余 SP 的中位数比没有显著的相对倍数差异，机制3
的"单元内相对异常"判据结构性地检测不到。

本模块实现谱方法/DG 方法处理这类混叠失稳的标准、教科书级别对策——
指数模态滤波器：把节点值变换到模态系数空间，对*高阶*模态按其阶数
指数衰减，再变换回节点值。低阶（物理上真正被解析到的）模态权重
恒为 1（不衰减，不影响真实物理精度），只有最高阶、本就处于插值多项式
数值噪声主导区间的模态被压制——这是谱方法/DG 方法里与"迎风数值耗散"
同等地位的标准组成部分，不是本次调试引入的权宜手段。

参考：Hesthaven & Warburton (2008) §5.3；Boyd (2001) Ch.11；
Karniadakis & Sherwin (2005) §4.3（谱/hp 方法里坍缩坐标单元同样需要
滤波，该书明确讨论了棱柱/四面体坍缩坐标模态的滤波器构造）。

2026-08-29 调查记录（未采用的修复尝试，供后续参考）：曾尝试过两种
"减少滤波强度"的改法——(a) order<=1 时直接返回单位矩阵（理由：P1
唯一的非常数模态在 eta=max(i,j,k)/order 归一化下恒为 eta=1，被当成
"最高阶噪声模态"整个滤掉，等效把 P1 拍平成 P0，真实复现：cube_demo
P1 层流/DDES 60 步求解后单元内 8 个 SP 的速度分量彼此只差 ~1e-14，
单元间却相差 27 m/s，壁面粘性力积分 F_viscous≈1e-19）；(b) order>=2
时改成"只有 max(i,j,k)==order 的模态才压制、其余模态 sigma 严格等于
1.0"的硬截断。两者都在真实验证中被证伪：(a) 直接用于 cube_demo 真实
P1 层流求解，10 步内即发散到 NaN（比原来的"缓慢发散"更差）；(b) 用于
tests/validation/test_tgv.py（P2，三方向周期四面体网格）与
tests/validation/test_couette.py::test_couette_prism_residual_trend
（P2，棱柱网格），双双在数步内复现本模块文档开头描述的原始灾难性
失稳（TGV 场值放大到 ~1e78 量级）。说明这个滤波器对"中间"模态的
光滑衰减不是可以随意去掉的多余抑制，而是真实承担着抑制混叠失稳的
数值稳定性职责，即使它同时也会在足够多次迭代后磨灭这些模态携带的
真实物理梯度内容——这是当前设计里两个目标（长期迭代精度 vs 短期
数值稳定性）之间一个尚未解决的真实张力，不是可以通过简单调整
sigma 分段/截断就能兼得的问题。改动已回退到本文档描述的原始实现；
根治需要更深入的数值方法研究（如真正的 entropy-stable/split-form
通量构造，或者只在残差诊断出真正的局部退化时才局部加强滤波，而不是
对整个网格全局施加同一套滤波强度），不在这次调查的范围内解决。
"""

import os

import numpy as np

from autoflowcfd.fr.collapsed_basis import prism_modal_basis_and_grad, tet_modal_basis_and_grad

# 标准指数滤波器参数（Hesthaven & Warburton 推荐值）：
# sigma(eta) = exp(-ALPHA * eta^(2*FILTER_ORDER))，eta=归一化模态阶数∈[0,1]。
# ALPHA = -ln(machine_eps)：使得 eta=1（最高阶模态）处滤波系数衰减到
# 双精度机器精度量级，彻底压制该模态携带的纯数值噪声，同时 eta=0
# （常数模态）处 sigma=1 恒成立（自由流场保持性不受影响，见本模块
# 单元测试）。FILTER_ORDER=4：中低阶模态（eta 明显小于 1）衰减因子
# 接近 1，只有最高一两阶模态被显著压制，不牺牲已解析到的真实物理精度。
#
# **2026-09-15 实测更正：这段"不牺牲已解析到的真实物理精度"的说明与实现
# 不符。** ALPHA=-ln(eps) 使 sigma(eta=1)=2.2e-16，那是**清零**而不是
# "显著压制"；而 eta=max(i,j,k)/order 这个判据下 eta=1 恰好覆盖该阶
# **全部新增模态**。实测滤波矩阵的秩：
#     order=1 -> 秩 1（只剩常数模态）      => P1 实际是 P0
#     order=2 -> 秩 8 = 2^3（只剩双线性）  => P2 实际是 P1
#     order=3 -> 秩 27 = 3^3               => P3 实际是 P2
# 即**每个阶数都精确损失一整阶**。算子层面直接验证（order=1，线性场 x）：
# 胞内变化 5.77e-3 -> 2.06e-18，保留比 3.6e-16。
# 真实网格印证（79 万单元 P1 iter=300 检查点）：胞内 |grad u| 相对参照
# 剪切率 U/h 只有 7.7e-16，粘性残差几乎只来自边界 IP 罚项。
#
# alpha=-ln(eps) 本身是 Hesthaven & Warburton 的标准取值，但那是为**高阶**
# 设计的：在 P8 上清掉第 8 阶无关紧要，在 P1 上清掉第 1 阶就只剩常数。
# 本项目只跑 P1/P2，正好落在这个取值最糟的区间。
#
# 另一个关键点：滤波器是**每个 RK stage 都施加**的，因此任何 sigma<1 都会
# 随步数复合累积（sigma=0.9 时每步 0.9^3=0.73，100 步后 ~1e-14）。平衡点
# 取决于"物理每步再生该模态的幅度"与"滤波每步压制的幅度"之比——单纯把
# alpha 调小只是把清零推迟，不改变"滤波压倒物理"的性质。所以正确的方向
# 是**按需施加**（逐单元用传感器门控），而不是全局调强度。
#
# `AFCFD_FILTER_MODE` 环境变量（默认 legacy = 保持既有行为，不改变默认）：
#   legacy  当前行为（alpha=-ln(eps)，全局每 stage 施加）
#   off     恒等滤波（完全不施加），用于对照"滤波是否必需"
#   mild    sigma(eta=1)=AFCFD_FILTER_SIGMA_TOP（默认 0.99），其余同形式
#   project 精确投影：sigma 只取 0 或 1 —— eta<1 的模态严格保留 1，
#           eta==1（最高阶）严格取 0。**幂等**，见下。
#   sensor  逐单元门控（在应用层实现），矩阵与 project 相同 —— 门控的
#           算子语义要求幂等，理由见 `filter_sigma` 里的注释。
# 这四档供受控 A/B 用；"按传感器逐单元门控"那一档在应用层实现
# （core/fr_solver/filter.py），不在这里改矩阵。
#
# ===== 为什么需要 project 档（2026-09-17 实测，一处此前被漏掉的缺陷）=====
#
# 上面那句"任何 sigma<1 都会随步数复合累积"此前只被当成 `mild` 档
# （sigma_top=0.99）的问题。但实测各阶的 sigma 谱是：
#
#   P1  eta=[0, 1]                sigma=[1, 2.22e-16]
#   P2  eta=[0, 0.5, 1]           sigma=[1, 0.868667, 2.22e-16]
#   P3  eta=[0, 1/3, 2/3, 1]      sigma=[1, 0.994521, 0.245032, 2.22e-16]
#
# **`legacy` 档自己在 P2/P3 就有严格介于 0 和 1 的中间模态 sigma。**
# 于是每个 RK stage 乘一次、每步三次：
#
#   P2  中间模态每步残留 0.6555  -> 约 100 步后 ~1e-18（实际退化到 P0，不是 P1）
#   P3  eta=2/3 的模态每步残留 0.0147 -> **一步内就基本清零**（两阶）；
#       eta=1/3 的模态每步 0.9837 -> 100 步后 0.192
#
# 项目记忆 `modal_filter_annihilates_one_order` 记的"每阶恰好损失一整阶"
# 是用**单次施加**的矩阵秩验证的（P1 1/8、P2 8/27、P3 27/64 恰好等于低
# 一阶的维数）——那个观测本身没错，但秩只反映"一次施加后哪些模态落到
# 数值零"，**不反映中间模态被反复乘以 0.87 / 0.25 的累积效应**。所以
# "P2≡P1 / P3≡P2"对长程运行是**不成立**的。
#
# 直接后果：矩阵不幂等。实测 |F@F - F| 的最大元
#
#   P1  2.2e-16（真投影）   P2  6.3e-02   P3  3.5e-01
#
# 而"按传感器逐单元门控"这套用法需要的算子语义恰恰是**"把这个单元降
# 一阶"**——那必须是幂等投影，否则被标记的单元会被反复削、一路掉到 P0，
# 并且退化后的单元更容易再次越界，形成正反馈。等熵涡精确解上的实测：
# 门控在 P1（真投影）上渐近失活、收敛阶保住 2.16/2.18；在 P2（非幂等）
# 上标记比例平台在 ~19%、收敛阶掉到 1.3。
#
# `project` 档就是把 sigma 取成严格的 {0,1}，于是
#   * 幂等：F@F == F 到机器精度，逐 stage 施加不累积；
#   * 语义精确："恰好削掉最高阶"，不多不少。
# legacy 档保持逐位不变，供回归对照。
_FILTER_MODE = os.environ.get("AFCFD_FILTER_MODE", "legacy").lower()
_SIGMA_TOP = float(os.environ.get("AFCFD_FILTER_SIGMA_TOP", "0.99"))

#: 合法档位。**必须校验**：此前未知取值会落进下面 `else` 分支、静默按
#: legacy 跑——一次拼写错误（例如 `AFCFD_FILTER_MODE=projct`）就会让一整
#: 轮 A/B 的两条运行悄悄变成同一档，而这在日志里完全看不出来（启动日志
#: 只打印读到的字符串本身）。同类静默回退在本项目已出过多次事故，见
#: `fr_operators/kernels.py::resolve_ausm_precond_mode` 文档。
#: `sensor` 档不改矩阵（在应用层 `core/fr_solver/filter.py` 实现），但仍
#: 是这里的合法取值——它走下面的 else 分支、用与 legacy 相同的矩阵。
_VALID_FILTER_MODES = ("legacy", "off", "mild", "project", "sensor")
if _FILTER_MODE not in _VALID_FILTER_MODES:
    raise ValueError(
        f"AFCFD_FILTER_MODE={_FILTER_MODE!r} 不是合法取值；"
        f"合法值 {sorted(_VALID_FILTER_MODES)}。"
        f"不静默回退到 legacy——那会让一次拼写错误伪装成默认行为。"
    )

if _FILTER_MODE == "mild":
    # 由 sigma(1)=exp(-alpha) 反解 alpha
    FILTER_ALPHA = -np.log(max(min(_SIGMA_TOP, 1.0 - 1e-300), 1e-300))
else:
    FILTER_ALPHA = -np.log(np.finfo(np.float64).eps)
FILTER_ORDER = 4
FILTER_MODE = _FILTER_MODE


def _exp_filter_sigma(degree_frac: np.ndarray) -> np.ndarray:
    """指数滤波器系数 sigma(eta)=exp(-alpha*eta^(2s))，eta=degree_frac。"""
    return np.exp(-FILTER_ALPHA * degree_frac ** (2 * FILTER_ORDER))


def filter_sigma(degree_frac):
    """按当前 `FILTER_MODE` 返回模态衰减系数 sigma(eta)。

    `project` 档返回严格的 {0,1}（eta>=1-1e-12 取 0，其余取 1），使滤波
    矩阵成为**幂等投影**；其余档沿用指数型 `_exp_filter_sigma`。
    完整理由见模块顶部"为什么需要 project 档"一节。

    单点标量与 numpy 数组都支持——两个坍缩/棱柱构造函数逐模态调用，
    native 四面体构造函数按整个模态列表向量化调用。
    """
    eta = np.asarray(degree_frac, dtype=np.float64)
    # `sensor` 与 `project` 共用投影型 sigma。**为什么 sensor 必须是投影**
    # （2026-09-17）：sensor 档的语义是"按传感器逐单元门控，对被标记单元
    # 施加这个矩阵"，而它每个 RK stage 都施加一次。所要表达的操作是
    # "把这个单元降一阶"——那在数学上就是一个**幂等投影**。用指数型
    # sigma（P2 中间模态 0.8687、P3 的 0.2450）会让被标记单元被反复削、
    # 一路掉到 P0，而退化后的单元更容易再次被标记，形成正反馈。等熵涡
    # 精确解实测：P2 上非幂等版本标记比例平台在 ~19%、收敛阶掉到 1.3。
    # P1 上两者**逐位相同**（legacy 的 P1 sigma 本来就是 {1, 2.2e-16}），
    # 所以此前用 sensor 档跑的 P1 对照结果不受影响。
    if FILTER_MODE in ("project", "sensor"):
        out = np.where(eta >= 1.0 - 1e-12, 0.0, 1.0)
        return out if out.ndim else float(out)
    return _exp_filter_sigma(eta)


def build_tet_modal_filter(order: int, ref_cube_sps: np.ndarray) -> np.ndarray:
    """四面体坍缩坐标模态滤波矩阵，形状 (n_sps,n_sps)。

    模态 (i,j,k) 的滤波强度按 max(i,j,k)/order 归一化——本模块复用的是
    fr/collapsed_basis.py 里 i,j,k 各自独立 0..order 的"扩展张量积"基
    （该基已用真实单纯形坐标 (r,s,t) 上的线性/二次多项式验证到机器精度
    是正确的完备插值基，见开发过程记录，不是本次要重新设计的对象），
    i,j,k 各自独立取 0..order，"某一根轴自己的索引逼近 order"就意味着
    该模态处于该轴插值多项式的最高阶、数值噪声主导区间——不能用总阶数
    i+j+k 归一化：真实验证过，总阶数判据在四面体上对随机白噪声的滤波
    结果是*放大*（噪声标准差放大 17.5 倍，算子谱范数 192.9），因为
    c 轴（模态基里退化程度最深的轴，见 fr/collapsed_basis.py 文档）
    权重形如 (1-c)^(i+j) 更容易在节点空间产生病态放大，用只统计"总阶数"
    而忽视"具体是哪根轴逼近其自身上限"的判据无法覆盖到；改用
    max(i,j,k)/order 后随机白噪声被正确压制（标准差降到 0.63 倍，谱
    范数 5.78），常数场保持性同样在机器精度量级不受影响。

    Args:
        order: 多项式阶数 P
        ref_cube_sps: (n_sps,3) 计算立方体 SPs 坐标，与
            fr/operators.py::generate_fr_operators 构造 D_3d_tet 用的
            完全一致

    Returns:
        F: (n_sps,n_sps) 滤波矩阵，F @ field(SPs) 给出滤波后的节点值
    """
    if order == 0 or FILTER_MODE == "off":
        # off：恒等滤波（AFCFD_FILTER_MODE=off，见模块顶部说明）
        return np.eye(ref_cube_sps.shape[0])

    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    V, _, _, _ = tet_modal_basis_and_grad(a, b, c, order)

    n1d = order + 1
    sigma = np.zeros(n1d ** 3)
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                flat = i * n1d * n1d + j * n1d + k
                sigma[flat] = filter_sigma(max(i, j, k) / order)

    return V @ np.diag(sigma) @ np.linalg.inv(V)


def build_prism_modal_filter(order: int, ref_cube_sps: np.ndarray) -> np.ndarray:
    """棱柱坍缩坐标模态滤波矩阵，形状 (n_sps,n_sps)。

    与 build_tet_modal_filter 同一套 max(i,j,k)/order 归一化判据（原因
    见该函数文档——按"某一根轴自己的索引逼近 order"而非总阶数判断，
    棱柱上两种判据（总阶数 vs max）实测都能有效压制随机白噪声，统一
    用 max(i,j,k)/order 与四面体保持同一套判据，便于维护）。

    Args:
        order: 多项式阶数 P
        ref_cube_sps: (n_sps,3) 计算立方体 SPs 坐标

    Returns:
        F: (n_sps,n_sps) 滤波矩阵
    """
    if order == 0 or FILTER_MODE == "off":
        # off：恒等滤波（AFCFD_FILTER_MODE=off，见模块顶部说明）
        return np.eye(ref_cube_sps.shape[0])

    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    V, _, _, _ = prism_modal_basis_and_grad(a, b, c, order)

    n1d = order + 1
    sigma = np.zeros(n1d ** 3)
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                flat = i * n1d * n1d + j * n1d + k
                sigma[flat] = filter_sigma(max(i, j, k) / order)

    return V @ np.diag(sigma) @ np.linalg.inv(V)
