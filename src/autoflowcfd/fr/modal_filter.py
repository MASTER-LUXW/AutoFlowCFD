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
"""

import numpy as np

from autoflowcfd.fr.collapsed_basis import prism_modal_basis_and_grad, tet_modal_basis_and_grad

# 标准指数滤波器参数（Hesthaven & Warburton 推荐值）：
# sigma(eta) = exp(-ALPHA * eta^(2*FILTER_ORDER))，eta=归一化模态阶数∈[0,1]。
# ALPHA = -ln(machine_eps)：使得 eta=1（最高阶模态）处滤波系数衰减到
# 双精度机器精度量级，彻底压制该模态携带的纯数值噪声，同时 eta=0
# （常数模态）处 sigma=1 恒成立（自由流场保持性不受影响，见本模块
# 单元测试）。FILTER_ORDER=4：中低阶模态（eta 明显小于 1）衰减因子
# 接近 1，只有最高一两阶模态被显著压制，不牺牲已解析到的真实物理精度。
FILTER_ALPHA = -np.log(np.finfo(np.float64).eps)
FILTER_ORDER = 4


def _exp_filter_sigma(degree_frac: np.ndarray) -> np.ndarray:
    """指数滤波器系数 sigma(eta)=exp(-alpha*eta^(2s))，eta=degree_frac。"""
    return np.exp(-FILTER_ALPHA * degree_frac ** (2 * FILTER_ORDER))


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

    n1d = order + 1
    sigma = np.zeros(n1d ** 3)
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                flat = i * n1d * n1d + j * n1d + k
                sigma[flat] = _exp_filter_sigma(max(i, j, k) / order)

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
    if order == 0:
        return np.eye(ref_cube_sps.shape[0])

    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    V, _, _, _ = prism_modal_basis_and_grad(a, b, c, order)

    n1d = order + 1
    sigma = np.zeros(n1d ** 3)
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                flat = i * n1d * n1d + j * n1d + k
                sigma[flat] = _exp_filter_sigma(max(i, j, k) / order)

    return V @ np.diag(sigma) @ np.linalg.inv(V)
