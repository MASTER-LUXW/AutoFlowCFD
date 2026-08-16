"""体积项/梯度里"共享算子 (不依赖 cell) 张量收缩"的高效实现 (性能优化配套)。

`fr_residual_inviscid.py`（体积项 over-integration 三件套 + 无 fine 几何回退
分支）、`fr_viscous_flux.py`（体积项散度）、`fr_gradients.py`（物理空间梯度）
里大量出现形如::

    np.einsum("fs,csv->cfv", D, X)      # D 不依赖 cell，只收缩 1 个轴
    np.einsum("fjm,cjmv->cfv", D, X)    # D 不依赖 cell，收缩 2 个轴 (j,m)

的调用——`D`（微分/插值/限制算子）对所有 cell 共享、完全相同，只有 `X`
随 cell 变化。这类收缩在数学上等价于"共享矩阵 @ 逐 cell 展平后的大矩阵"，
应当归约成一次 BLAS gemm；但 `np.einsum` 不传 `optimize=True` 时走的是
通用逐元素求和路径，不会自动识别并利用这个结构——在 545,597 cell 的
生产网格上实测是体积项的主要热点（py-spy 对 P2 阶数残差求值的采样，
`euler_physical_flux`/`einsum` 相关帧占据几乎全部采样）。

`np.tensordot` 对两个操作数的收缩，内部本身就是"reshape 成 2D + 调用
`np.dot`（BLAS gemm）+ reshape 回去"（numpy 自己的实现，非本项目新写的
数值算法）——用它替换这些 einsum 调用，是完全等价的同一个求和公式，
只是换一条计算路径，不改变数学结果（浮点重结合误差与此前 numba 化
界面项时处理的是同一类、已用相对容差验证过的现象，见
`fr_residual_inviscid_kernel.py` 模块文档）。

两个小函数只处理"D 在前、X 在后，D 的收缩轴固定是紧跟在输出轴之后的
1 或 2 个轴"这一种调用形状——这正是本代码库里所有出现该模式的调用点
的共同结构，不做成通用任意轴收缩工具。
"""

import numpy as np


def contract_shared_operator_1axis(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """等价于 `np.einsum("fs,csv->cfv", D, X)`。

    Args:
        D: (F, S) —— 不依赖 cell 的共享算子（如 interp_c2f、restrict_f2c）
        X: (C, S, V) —— 逐 cell 的场

    Returns:
        (C, F, V)
    """
    out = np.tensordot(D, X, axes=([1], [1]))  # (F, C, V)
    return np.ascontiguousarray(np.moveaxis(out, 0, 1))  # (C, F, V)


def contract_shared_operator_2axis(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """等价于 `np.einsum("fjm,cjmv->cfv", D, X)`。

    Args:
        D: (F, J, M) —— 不依赖 cell 的共享算子（如 D_3d_tet/prism、D_fine）
        X: (C, J, M, V) —— 逐 cell 的场

    Returns:
        (C, F, V)
    """
    out = np.tensordot(D, X, axes=([1, 2], [1, 2]))  # (F, C, V)
    return np.ascontiguousarray(np.moveaxis(out, 0, 1))  # (C, F, V)
