"""AutoFlowCFD V2.0 - 伪瞬态（PTC）对角预处理。

## 要解的线性系统

Newton 步本身是 `J dU = -R`。从一个远离解的初场直接解它是发散的（Newton
只有局部收敛性），工业做法是**伪瞬态延拓**（pseudo-transient
continuation，Kelley & Keyes 1998）：在对角上加一个伪时间项

    ( I/dtau + J ) dU = -R

* `dtau -> 0`：退化成显式前向 Euler（`dU = -dtau R`），无条件鲁棒但慢；
* `dtau -> inf`：退化成纯 Newton，接近解时二次收敛；
* 中间：随残差下降把 `dtau` 逐步放大，从鲁棒平滑过渡到快速。

这正是"用同一套代码既有显式的鲁棒性、又有 Newton 的速度"的标准机制，
也是本项目 `--cfl-start/--cfl-max` 那套自适应 CFL 控制器可以**原样复用**
的地方：`dtau` 就是 `cfl.py::compute_local_time_step` 已经算好的逐 SP
`dt_local`，CFL 越大 `dtau` 越大、越靠近 Newton。

## 预处理子

`I/dtau + J` 在 `dtau` 小的时候由对角项主导，此时它本身就良态；`dtau`
大的时候 `J` 主导，而 `J` 是对流-扩散算子、条件数随网格加密与阶数变差。
这里用**逐 SP 的对角预处理**

    P = I / dtau_local      ->      P^{-1} v = dtau_local * v

它有三个性质让它成为合理的第一实现：

1. **零额外成本**：`dtau_local` 已经被显式路径算出来了，没有任何新的
   矩阵构造或分解；
2. **它恰好抵消对角项**：`P^{-1}(I/dtau + J) = I + dtau*J`，在 `dtau`
   小的时候这就是 `I` 的小扰动，GMRES 几步就收敛 —— 也就是"鲁棒档"下
   预处理是接近最优的；
3. **逐 SP 而不是逐单元一个标量**：`dt_local` 在本项目里本来就是逐 SP
   的（随阶数/粘性刚性/几何退化收紧），退化单元与贴壁单元的 `dtau` 可以
   相差几百倍（实测 178~790 倍），用单元平均会让那些单元的对角项错得
   很远。

**它不是块 Jacobi**：块 Jacobi 见 `block_jacobi.py`（生产默认，本类是它
的基类，并在块 Jacobi 超内存上限时单独使用）。本文档旧版写过"矩阵自由地
构造块需要每单元 `n_sps*n_var` 次残差求值、成本不可接受"——那条论证漏掉
了**着色**：同色单元可以同时扰动，总次数是 `色数 x 解点数 x 变量数`、与
单元数无关（plate_demo P1 是 150 次）。实测在 CFL 34 下块 Jacobi 把 GMRES
从 300 次（仍未收敛）降到 20 次，见 `block_jacobi.py` 模块文档。
"""

import numpy as np


class PseudoTransientDiagonal:
    """`P^{-1} v = dtau_local * v` 的对角预处理，同时提供 PTC 对角项本身。

    **做成类而不是闭包**：它要在整个 Krylov 求解期间存活，闭包会把创建
    它的整个作用域一起留活（项目规范）。
    """

    __slots__ = ("_dtau", "_dtau_col")

    def __init__(self, dtau_flat: np.ndarray, n_var: int):
        """
        Args:
            dtau_flat: `(N,)` 逐 SP 伪时间步长（就是 `cfl.py` 算出的
                `dt_local`，已按阶数/粘性/几何收紧）。
            n_var: 守恒变量数，只用来校验广播形状意图明确。
        """
        from autoflowcfd.core.utils.array_module import array_module

        xp = array_module(dtau_flat)
        dtau = xp.ascontiguousarray(dtau_flat, dtype=xp.float64).ravel()
        if dtau.ndim != 1:
            raise ValueError(f"dtau 必须是一维逐 SP 数组，收到 {dtau.shape}")
        if not bool(xp.all(dtau > 0.0)):
            bad = int(xp.count_nonzero(dtau <= 0.0))
            raise ValueError(
                f"dtau 里有 {bad} 个非正值 —— 伪瞬态项 `I/dtau` 要求它严格"
                f"为正，非正值只可能来自局部步长计算本身出了问题，"
                f"不静默钳制")
        self._dtau = dtau
        self._dtau_col = dtau[:, None]
        if n_var <= 0:
            raise ValueError(f"n_var 必须为正，收到 {n_var}")

    @property
    def dtau(self) -> np.ndarray:
        """逐 SP 的 `dtau`，形状 `(N,)`。"""
        return self._dtau

    def add_ptc_term(self, jv_flat: np.ndarray, v_flat: np.ndarray
                     ) -> np.ndarray:
        """把 `J v` 变成 `(I/dtau + J) v`（**原地**加到 `jv_flat` 上）。"""
        jv_flat += v_flat / self._dtau_col
        return jv_flat

    def apply(self, v_flat: np.ndarray) -> np.ndarray:
        """`P^{-1} v = dtau_local * v`。"""
        return v_flat * self._dtau_col
