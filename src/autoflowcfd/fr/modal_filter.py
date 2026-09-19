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
# `AFCFD_FILTER_MODE` 环境变量（**默认 off**，2026-09-19 从 sensor 改，
# 依据见本文件下方"默认值 2026-09-19 改为 off"那一节 —— 简述：修掉原生
# 四面体路径两个重大缺陷之后，用 Blasius 的 cf 与 TGV 的解析耗散率重新
# 判定，`off` 是唯一在两个算例上都物理自洽的档）：
#   legacy  当前行为（alpha=-ln(eps)，全局每 stage 施加）
#   off     恒等滤波（完全不施加），用于对照"滤波是否必需"
#   mild    sigma(eta=1)=AFCFD_FILTER_SIGMA_TOP（默认 0.99），其余同形式
#   project 精确投影：sigma 只取 0 或 1 —— eta<1 的模态严格保留 1，
#           eta==1（最高阶）严格取 0。**幂等**，见下。
#   sensor  逐单元门控（在应用层实现）；矩阵自 2026-09-18 起用 mild 型
#           **有界衰减**（顶模态 sigma=AFCFD_FILTER_SIGMA_TOP，默认 0.99）
#           而不再是精确投影，理由见 filter_sigma 里那一节 —— 门控的
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
#
# ===== 默认值 2026-09-19 改为 off（依据见下，取代 2026-09-17 那次）=====
#
# 先说为什么此前那轮标定**整体失效**：2026-09-19 在已上线的原生四面体路径上
# 查出并修掉两个重大缺陷（见 `core/fr_operators/face_kernels.py::
# FlatFaceGeometry.ref_area_weight` 与 `core/fr_operators/troubled_cell.py::
# _outlier_ref_and_flag_kernel` 的文档）：
#   * DG 提升算子的界面项多乘了一个 `|adj_row| ~ h^2`，界面耦合与**上风
#     耗散**在细网格上被系统性压制（细长四面体实测差 5 个数量级）；
#   * P2/P3 的四面体残差被机制3 **整体清零**，单元完全不演化。
# 也就是说：此前"必须靠滤波才稳定"这个印象，是在格式本身缺了应有的上风
# 耗散（甚至 P2/P3 干脆没有四面体动力学）的前提下形成的。那批标定不能沿用。
#
# 修完之后用**有解析解**的两个算例重新判定（都不含激波，所以任何限制器
# 动作本来就不该发生）：
#
#   档       Blasius cf 中位偏差      TGV 动能（解析耗散率判据）
#   off      +9.52%                   通过（单调衰减，dK/dt 为解析值的
#                                     0.31~0.54 倍 —— 欠分辨下欠耗散是
#                                     可预期的）
#   sensor   +9.52%（与 off **逐位相同**）  **失败：能量净增长 +5.14%**
#   project  **-87.69%**              通过
#   legacy   --                       **失败：过耗散 7.6 倍**
#
# 三条由此确定的事实：
#   1. `sensor` 档（2026-09-18 起用 mild 型有界衰减）在 Blasius 上与 `off`
#      **逐位相同** —— 它实质上什么都没做。这与另一条既有观测是同一事实的
#      两面：「sensor + persson == off，因为 Persson 掩码实测 0.000%」。
#   2. 但它并不是真的无操作：TGV 上 BJ 判据对这个**欠分辨光滑场 100% 标记**
#      （正弦每波长只有 4 个单元、曲率强，单元内极值确实超出邻域均值包络
#      —— 这是 BJ 这类判据的固有行为，不是缺陷），于是退化成"全局每 stage
#      施加 mild 非幂等衰减"，450 次累积出**非物理的能量增长**。这与既有
#      记录「非幂等 -> 反复削 -> 正反馈」吻合。
#   3. `project`（幂等、语义精确）在 P1 上等于把被标记单元**拍平成 P0**，
#      Blasius 的壁面剪应力因此塌掉 88% —— P1 的顶模态就是全部非常数内容。
#      所以"把动作换成幂等投影"这条路在生产阶数 P1 上不可用。
#
# 结论：门控滤波这套机制在当前实现下没有提供可测的正面价值，而两种全局
# 强档各自破坏一个物理量。默认值定为 `off`，稳定性交给格式本身应有的机制
# —— 上风通量耗散（AUSM+up，界面项现已修正）、体积项去混叠
# （`AFCFD_VISC_OVERINT`，默认开）、以及真有激波时才该出现的人工粘性。
# 这与工业/文献里高阶 DG/FR 的标准做法一致：限制器/滤波只在真实不连续处
# 动作，不作为常规耗散来源。
#
# 其余四档全部保留为合法取值：`legacy` 是唯一能复现历史结果的档（回归
# 对照），`project`/`mild`/`sensor` 供将来真有激波的算例与受控 A/B。
#
# ===== 默认值 2026-09-17 从 legacy 改为 sensor（已被上面那轮取代）=====
#
# 三档在 P1（生产阶数）上的实测（平板边界层算例，2304 单元，同一初场
# 同一 CFL，80 步）：
#
#   filter=legacy  sensor=persson   176.2 ms/step   res 7.9426e+04
#   filter=project sensor=persson   156.9 ms/step   res 7.9426e+04   <- 与 legacy 逐位相同
#   filter=off     sensor=persson   179.7 ms/step   res 1.6658e+05
#   filter=sensor  sensor=persson   154.3 ms/step   res 1.6658e+05   <- 与 off 逐位相同
#   filter=sensor  sensor=bounds    183.1 ms/step   res 7.7280e+04   <- 残差最低
#
# 两条"逐位相同"各自印证了一件事：
#   * legacy == project 在 P1 上成立，因为 P1 的顶模态**就是**全部非常数
#     内容，两档都把它清零 -> 两档都让 P1 退化成 P0；
#   * sensor + persson == off，因为 Persson-Peraire 在 order=1 上原理性
#     退化、且它探的是守恒密度（真实解上光滑），掩码实测 0.000%
#     （同一时刻 rho_v/rho_w 是 98.8%），等于没有门控。
#
# 所以 sensor 与 bounds 必须成对改：只把 filter 改成 sensor 而传感器还是
# persson，等于把默认值悄悄改成了 off。
#
# 为什么 sensor + bounds 是每个阶数上都不差的那一档（等熵涡精确解）：
#
#   阶数   legacy / project                     sensor + bounds
#   P1     退化成 P0（阶 1）                    阶 2.16/2.18，达到设计阶 2
#   P2     legacy 非幂等、约 100 步退到 P0；     只在被标记的约 17.6% 单元
#          project 精确降一阶（阶 2 vs 设计 3）  里降阶，全局阶约 2.1
#
# 代价：相对 legacy 每步 +3.9%（183.1 vs 176.2 ms）。真实网格上的决定性
# 证据（plate_demo_volume_les，179,237 单元）：legacy 在 iter 112 发散，
# 而 sensor+bounds 跑出 216 步残差单调下降 3.5 倍，Cd 漂移从零曲率的线性
# 0.0167/步变成负曲率的 0.0071 -> 0.0042/步。
#
# legacy 保留为合法档，专供回归对照（它是唯一能复现历史结果的档）。
_FILTER_MODE = os.environ.get("AFCFD_FILTER_MODE", "off").lower()
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

if _FILTER_MODE in ("mild", "sensor"):
    # 由 sigma(1)=exp(-alpha) 反解 alpha。
    # `sensor` 档 2026-09-18 起也走这一支（此前用精确投影），理由见下方
    # filter_sigma 里"为什么 sensor 从精确投影改成有界衰减"一节。
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
    # ===== 为什么 sensor 从精确投影改成有界衰减（2026-09-18 更正）=====
    #
    # 此前这里写的是"sensor 与 project 共用投影型 sigma"，论证是：门控的
    # 语义是"把这个单元降一阶"，而它每个 RK stage 都施加一次，所以算子
    # 必须**幂等**，否则被标记单元会被反复削到 P0。那条论证的**前提**
    # 是"非投影"只能取 legacy 的 alpha —— 那个 alpha 下 P2 中间模态是
    # 0.8687、P3 是 0.2450，确实会把中间模态一起削掉，等熵涡上实测标记
    # 比例平台 ~19%、收敛阶掉到 1.3。
    #
    # 但 `mild` 的 alpha 完全不同。sigma_top=0.99 -> alpha=0.01005，
    # sigma(eta)=exp(-alpha*eta^8)：
    #
    #     P2  eta={0, 0.5, 1}          sigma={1, 0.999961, 0.99}
    #     P3  eta={0, 1/3, 2/3, 1}     sigma={1, 1-1.5e-6, 0.99961, 0.99}
    #
    # 也就是说**只有顶模态被动，每次只衰减 1%**；"中间模态被反复削"这条
    # 担忧在这个 alpha 下不成立，非幂等性被限制在顶模态自身的 0.99^n 上
    # ——那正是想要的"只要传感器还在报，就持续慢慢耗散"。
    #
    # 为什么必须改（Blasius 平板，本项目唯一有精确解的粘性算例，nx=16、
    # 400 步、固定 CFL 0.03、同步数的干净对照；壁面剪应力用胞内两点斜率
    # 提取，该提取已用"把解析解放到同一网格上"验证到中位 -0.49%）：
    #
    #     档位                          贴壁命中率  du/dy 中位   cf vs Blasius
    #     off（无滤波，会发散）              --      1312.68      -6.33%
    #     sensor+bounds 精确投影           0.831%     561.94     -74.54%
    #     sensor+bounds 有界衰减 0.99      7.510%    1312.68      -6.33%
    #     sensor+bounds 有界衰减 0.90      3.167%    1312.68      -6.33%
    #
    # 精确投影把被标记单元一次清零，而单元内扩散重建壁面梯度约需
    # `h^2/nu/dt ~ 26` 步——0.83%/stage 意味着平均每 40 步就再清一次，
    # 梯度永远恢复不到位（实测 du/dy 的**最大值**是对的、中位被拉低
    # 2.3 倍）。有界衰减下 cf 与 `off` **完全一致**，而滤波仍然在工作
    # （贴壁层每 stage 仍标记 3~7.5% 并施加衰减）。
    #
    # 命中率反而比投影档**高**是应当的：投影把单元压平之后它就不再越界，
    # 而衰减保留梯度、于是持续（合理地）越界。
    if FILTER_MODE == "project":
        out = np.where(eta >= 1.0 - 1e-12, 0.0, 1.0)
        return out if out.ndim else float(out)
    return _exp_filter_sigma(eta)


def _tensor_max_etas(order: int) -> np.ndarray:
    """坍缩基（"扩展张量积" `i,j,k` 各自独立 0..order）的 `max(i,j,k)/order`
    归一化模态阶数，排列与 `tet_modal_basis_and_grad`/
    `prism_modal_basis_and_grad` 的列序一致（`flat = i*n1d^2 + j*n1d + k`）。

    坍缩四面体与坍缩棱柱用的是**同一个**判据与**同一个**模态排列，所以
    只留一份。为什么是 `max` 而不是总阶数 `i+j+k`，见
    `build_tet_modal_filter` 文档里那段实测（总阶数判据在四面体上把随机
    白噪声*放大* 17.5 倍）。
    """
    n1d = order + 1
    etas = np.empty(n1d ** 3)
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                etas[i * n1d * n1d + j * n1d + k] = max(i, j, k) / order
    return etas


def assemble_modal_filter(V: np.ndarray, etas) -> np.ndarray:
    """`F = V @ diag(sigma(eta)) @ V^-1` —— 全部基**共用**的滤波器装配步。

    Args:
        V: `(n, n)` 模态 Vandermonde（节点 x 模态）
        etas: 长度 `n` 的归一化模态阶数，逐项 `in [0, 1]`，列序与 `V` 的
            列一致。各基自己决定怎么归一化（坍缩四面体/坍缩棱柱用
            `max(i,j,k)/order`，原生四面体用 `(i+j+k)/order`，原生棱柱用
            `max(i+j, k)/order`），理由分别见各构造函数文档。

    Returns:
        `(n, n)` 滤波矩阵，`F @ 节点值` 给出滤波后的节点值。

    ## 为什么把这几行单独抽出来

    `FILTER_MODE == "off"` 这条短路**当年真的被漏掉过**：坍缩四面体与
    坍缩棱柱两个构造都有，原生四面体那个没有，后果是
    `AFCFD_FILTER_MODE=off` **只关掉了棱柱的滤波器**，而那张 79 万单元
    网格上四面体占 82.7% —— 也就是说"关掉滤波器"的对照实验里绝大多数
    单元根本没被关掉，排查时因为日志只打印 `filter_prism` 的秩而没发现
    （完整记录见 `native_tet_filter.py` 里那段注释）。

    把装配与短路收到一处之后，再加一条基**不可能**漏掉它。
    """
    n = V.shape[0]
    etas = np.asarray(etas, dtype=np.float64)
    if etas.shape != (n,):
        raise ValueError(
            f"assemble_modal_filter: etas 形状 {etas.shape} 与 V 的列数 "
            f"{n} 不一致 —— 两者必须逐项对应，不一致说明调用方把模态顺序"
            f"弄错了，那会让滤波强度施加到错误的模态上（不会报错、只会"
            f"静默地压掉物理内容）")
    if FILTER_MODE == "off":
        # off：恒等滤波（AFCFD_FILTER_MODE=off，见模块顶部说明）
        return np.eye(n)
    sigma = np.array([filter_sigma(float(e)) for e in etas])
    return V @ np.diag(sigma) @ np.linalg.inv(V)


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
    if order == 0:
        return np.eye(ref_cube_sps.shape[0])

    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    V, _, _, _ = tet_modal_basis_and_grad(a, b, c, order)
    return assemble_modal_filter(V, _tensor_max_etas(order))


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
    if order == 0:
        return np.eye(ref_cube_sps.shape[0])

    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    V, _, _, _ = prism_modal_basis_and_grad(a, b, c, order)
    return assemble_modal_filter(V, _tensor_max_etas(order))
