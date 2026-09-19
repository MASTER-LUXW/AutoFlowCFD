"""AutoFlowCFD V2.0 - 矩阵自由 Jacobian-向量乘（Fréchet 差分）。

Newton-Krylov 的核心：Krylov 子空间迭代只需要 `J v`，不需要 `J` 本身。
用残差函数的一阶 Fréchet 差分近似：

    J v  ≈  [ R(U + eps*v) - R(U) ] / eps

一次 `J v` 恰好一次残差求值，与一次显式 RK 子级同量级。**不组装、不存
储任何 Jacobian**，所以内存占用与显式推进相同 —— 这对本项目很关键：
P2 下 79 万单元的一个 `(n_cells, n_sps, n_var)` 场就是 1.2 GB，稀疏
Jacobian（每单元 27x27x5x5 的块 + 面耦合块）在同一网格上是不可接受的。

## eps 的取法：逐变量按参考量级无量纲化

标准取法（Brown & Saad 1990；Pernice & Walker 1998）是

    eps = sqrt(eps_mach) * (1 + ||U||) / ||v||

它隐含假设 `U` 的各分量量级相当。本项目**不满足**这个前提：守恒变量
横跨 5 个数量级（`rho` ~ 1.2、`rho*u` ~ 40、`rho*E` ~ 2.5e5）。所以这里
先按参考量级无量纲化（复用
`fr_solver/residual_diagnostics.py::_reference_scales` 那一套，与残差
诊断、BJ 判据的绝对地板同一个尺度来源），在无量纲空间里取标量 eps：

    U~ = U / s,  v~ = v / s              （s = 参考量级，逐变量）
    eps = sqrt(eps_mach) * (1 + rms(U~)) / rms(v~)

`rms` 而不是 `||.||_2` 是为了让 eps 不随网格规模漂移（`||U~||_2` 随单元
数按 sqrt(N) 增长，用它会让同一个物理问题在不同网格上取不同的差分步长）。

### 实测收益：4~13 倍，不是"决定性"（如实记录）

写这段时曾声称"单标量 eps 会让质量方程那几行基本是噪声"。**那条声称被
自己的测量否掉了**，必须更正。用一个有解析 Jacobian 的非线性合成系统
（变量量级刻意按 `[1.225, 40, 40, 40, 2.5e5]` 构造）对照 `J v` 的相对
误差：

    v 的方向      逐变量无量纲    单标量 eps
    e_rho          1.81e-09      8.56e-09
    e_rho_u        6.08e-10      4.92e-09
    e_rho_v        1.41e-09      1.89e-08
    e_rho_w        1.72e-09      1.11e-08
    e_rho_E        1.04e-08      4.66e-09     <- 这一档反而差 1.6 倍
    按量级随机      1.05e-08      6.39e-09     <- 同上

所以真实情况是：在**集中于小量级变量**的方向上无量纲化好 4~13 倍，在
`rho_E` 主导的方向上差约 1.6 倍；两者都在 1e-9 量级，都远低于
inexact-Newton 的容差下限（`forcing.py` 的 `_ETA_MIN = 1e-4`），**都不是
收敛的限制因素**。

保留无量纲化的理由因此是"最坏方向上的误差更小、且与项目其它地方的尺度
约定一致"，不是"另一种不可用"。

（方法论记录：第一版对照用的是**线性** R，得出的结论是相反的 ——
线性 R 没有截断误差，eps 越大舍入越小，于是单标量 eps 的"大 eps"反而
赢。必须用非线性 R 并与解析 Jacobian 对照才测得到真实取舍。）

## 为什么不用二阶中心差分

中心差分 `[R(U+eps v) - R(U-eps v)] / (2 eps)` 精度更高，但每次 `J v`
要**两次**残差求值，而 GMRES 的每次迭代都要一次 `J v`。一阶差分的误差
是 `O(eps) = O(sqrt(eps_mach)) ~ 1e-8` 相对量级，远小于 inexact-Newton
容差（见 `forcing.py`），不是收敛的限制因素。基态 `R(U)` 在整个 Krylov
求解期间只算一次并缓存，所以一次 GMRES 迭代 = 一次残差求值。
"""

from typing import Callable

import numpy as np

#: `sqrt(eps_mach)`，一阶 Fréchet 差分的最优步长量级（截断误差
#: `O(eps)` 与舍入误差 `O(eps_mach/eps)` 在此处平衡）。
_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))


class MatrixFreeJacobian:
    """`R` 在基态 `U0` 处的矩阵自由 Jacobian 作用算子。

    一个实例对应一个 Newton 外迭代：基态 `U0` 与基残差 `R0` 固定，
    Krylov 迭代反复调用 `matvec`。

    **做成类而不是闭包**：闭包会把创建它的整个作用域（含 solver、网格、
    算子等大对象）一起留活，而这个对象要在整个 Krylov 求解期间存在。
    类只持有它真正需要的几个字段（项目规范：长生命周期对象不要用闭包
    捕获环境）。
    """

    __slots__ = ("_residual", "_u0", "_r0", "_scales_flat", "_inv_scales_flat",
                 "_u0_rms_scaled", "n_matvec")

    def __init__(self, residual: Callable[[np.ndarray], np.ndarray],
                 u0_flat: np.ndarray, r0_flat: np.ndarray,
                 scales: np.ndarray):
        """
        Args:
            residual: `R(U_flat) -> (N, n_var)`，与 `step.py` 里
                `mean_flow_residual` 同一个约定（`dU/dt = -R`）。
            u0_flat: `(N, n_var)` 基态。
            r0_flat: `(N, n_var)` 基态残差 `R(U0)`，由调用方算好传入
                （整个 Krylov 求解期间复用，不重算）。
            scales: `(n_var,)` 每个守恒变量的参考量级。
        """
        u0_flat = np.ascontiguousarray(u0_flat, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64).ravel()
        if scales.shape[0] != u0_flat.shape[1]:
            raise ValueError(
                f"参考量级长度 {scales.shape[0]} 与变量数 "
                f"{u0_flat.shape[1]} 不符 —— 逐变量无量纲化是这里 eps "
                f"取法的前提，形状不符不能静默按广播处理")
        if not np.all(scales > 0.0):
            raise ValueError(f"参考量级必须全为正，收到 {scales}")

        self._residual = residual
        self._u0 = u0_flat
        self._r0 = np.ascontiguousarray(r0_flat, dtype=np.float64)
        self._scales_flat = scales[None, :]
        self._inv_scales_flat = (1.0 / scales)[None, :]
        # ||U~||_rms，只依赖基态，整个 Krylov 求解期间是常数
        self._u0_rms_scaled = float(
            np.sqrt(np.mean((u0_flat * self._inv_scales_flat) ** 2)))
        self.n_matvec = 0

    def matvec(self, v_flat: np.ndarray) -> np.ndarray:
        """`J v`，形状与 `v_flat` 相同 `(N, n_var)`。

        `v` 按**物理**量纲给出（与 `U` 同量纲），内部自行无量纲化。
        `v` 为零向量时直接返回零（GMRES 的初始/退化情形），不去做一次
        无意义的残差求值。
        """
        v_flat = np.ascontiguousarray(v_flat, dtype=np.float64)
        v_scaled = v_flat * self._inv_scales_flat
        v_rms = float(np.sqrt(np.mean(v_scaled ** 2)))
        if v_rms == 0.0:
            return np.zeros_like(v_flat)

        eps = _SQRT_EPS * (1.0 + self._u0_rms_scaled) / v_rms
        r_pert = self._residual(self._u0 + eps * v_flat)
        self.n_matvec += 1
        return (np.asarray(r_pert, dtype=np.float64) - self._r0) / eps
