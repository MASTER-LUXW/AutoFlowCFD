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

实现选 `np.matmul` 广播、不用 `np.tensordot`：`tensordot(D, X, ...)` 把
不依赖 cell 的小算子 `D` 当作 `a`、逐 cell 的大数组 `X` 当作 `b`，内部
按 `newaxes_b = contract轴 + notcontract轴` 转置 `b`——X 的 cell 轴不在
被收缩的轴里，会被搬到中间，产生一份完整的转置副本（`.reshape` 无法
把这种转置合并成 no-copy view）。在 364,555 cell 的生产网格上 P2 阶数
实测触发过 `Unable to allocate 1.10 GiB`（cell 轴 153,950 个 prism 撞上
J×M×V 的转置副本）——`X` 的 cell 轴原本就在最前面（数组按 cell 存储，
下游按 cell 消费），不应该为了凑 tensordot 的轴序被搬到别处再搬回来。
`np.matmul(D_flat, X_flat)`（`D_flat`: (F,K)，`X_flat`: (C,K,V)）走
批量矩阵乘广播语义，把 (C,K,V) 当作 C 个 (K,V) 矩阵、循环与共享的
(F,K) 做 gemm，全程不触碰、不转置 X 的 cell 轴，不产生这份大副本；
数学上与 `tensordot`+`moveaxis` 完全等价（已用随机张量数值验证，
两者输出最大绝对误差在 float64 机器精度量级）。
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
    # np.matmul 对 2D @ 3D 按批量矩阵乘广播：D (F,S) 广播为逐 cell 共享的
    # 左矩阵，与 X 的每个 (S,V) 切片做 gemm，得到 (C,F,V)——不转置 X 的
    # cell 轴，见模块文档。
    return np.ascontiguousarray(np.matmul(D, X))


def contract_shared_operator_2axis(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """等价于 `np.einsum("fjm,cjmv->cfv", D, X)`。

    Args:
        D: (F, J, M) —— 不依赖 cell 的共享算子（如 D_3d_tet/prism、D_fine）
        X: (C, J, M, V) —— 逐 cell 的场

    Returns:
        (C, F, V)
    """
    F, J, M = D.shape
    C, _, _, V = X.shape
    D_flat = D.reshape(F, J * M)  # 小数组，reshape 是 no-copy view
    X_flat = X.reshape(C, J * M, V)  # J,M 相邻，合并成 K 轴同样是 no-copy view
    return np.ascontiguousarray(np.matmul(D_flat, X_flat))
