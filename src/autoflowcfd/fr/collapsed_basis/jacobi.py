"""AutoFlowCFD V2.0 - Jacobi 正交多项式及其导数。

从 `fr/collapsed_basis.py`（原 610 行）拆出（2026-09-19，项目"单文件不超
500 行"规范）。这两个函数是**与基无关**的基础设施：坍缩基、原生三角形/
四面体 PKD 基、原生棱柱的挤出方向 Legendre 都在用。纯搬家，逻辑未改。
"""

import numpy as np
from numba import njit


@njit(cache=True)
def jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """计算未归一化 Jacobi 多项式 P_n^(alpha,beta)(x) 在 x（数组）处的取值。

    标准三项递推（Hesthaven & Warburton 附录 A / Abramowitz & Stegun）。
    只作为构造 Vandermonde 矩阵的基，不需要正交归一化——微分矩阵
    D=V_xi@inv(V) 与具体选用哪组（可逆的）基无关，只要 V 可逆。

    条件数随阶数增长（真实验证：四面体 N<=2 时 cond(V)<=1e5，N=3 时
    ~1e9，N=4 时 ~1e14）——本代码库当前实际使用的多项式阶数 N=2（P=2）
    在这个范围内工作良好（体积项散度残差已在真实网格上验证到 1e-12
    量级）。曾尝试用标准 L2 归一化 Jacobi 多项式降低高阶条件数
    （Hesthaven-Warburton 参考实现的做法），但发现对本模块这种
    "P_n^(alpha,0) 外面再乘 (1-b)^i 权重因子"的复合结构，单独归一化
    多项式因子本身并不能改善（真实测得反而在部分阶数下更差：N=4 时
    从 2.8e14 变成 8.2e16）——权重因子必须和多项式的正交性一起联合
    归一化才是正确做法，这是比单独归一化 Jacobi 多项式更复杂的构造，
    N>=3 时的条件数改善留作后续工作（不影响当前 N=2 生产阶数的正确性
    与数值稳健性，已充分验证）。

    numba @njit 编译（性能优化：这个函数在真实网格上被调用数百万次——
    每次 owner/neighbor 跨单元插值都要重新构造 Vandermonde 矩阵，
    130 万面的生产网格上单是网格加载阶段就要跑约 300 万次调用，纯
    Python 函数调用开销占了 FP 几何构建约 2/3 的时间，见开发过程记录
    的 cProfile 剖析）。数学公式与递推逻辑完全不变，只是编译成原生
    代码执行——已用随机输入对比新旧实现逐位一致（200 组随机 n/alpha/
    beta/x 全部 0.0 误差），实测单次调用提速约 13 倍。numba 要求输入
    是具体类型的 ndarray，不再接受 list 等其它可迭代对象——本模块内外
    全部调用点传入的都已经是 float64 ndarray（见调用处），这不是放宽/
    简化数值行为，只是收紧了函数签名对输入类型的隐式假设。
    """
    P0 = np.ones_like(x)
    if n == 0:
        return P0
    P1 = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x)
    if n == 1:
        return P1
    Pnm1, Pn = P0, P1
    for k in range(1, n):
        a1 = 2.0 * (k + 1) * (k + alpha + beta + 1) * (2 * k + alpha + beta)
        a2 = (2 * k + alpha + beta + 1) * (alpha**2 - beta**2)
        a3 = (2 * k + alpha + beta) * (2 * k + alpha + beta + 1) * (2 * k + alpha + beta + 2)
        a4 = 2.0 * (k + alpha) * (k + beta) * (2 * k + alpha + beta + 2)
        Pnp1 = ((a2 + a3 * x) * Pn - a4 * Pnm1) / a1
        Pnm1, Pn = Pn, Pnp1
    return Pn


@njit(cache=True)
def grad_jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """P_n^(alpha,beta) 对 x 的导数：(n+alpha+beta+1)/2 * P_{n-1}^(alpha+1,beta+1)(x)，n=0 时恒为 0。
    numba @njit 编译，理由/验证同 jacobi_polynomial 文档。"""
    if n == 0:
        return np.zeros_like(x)
    return 0.5 * (n + alpha + beta + 1.0) * jacobi_polynomial(x, alpha + 1.0, beta + 1.0, n - 1)
