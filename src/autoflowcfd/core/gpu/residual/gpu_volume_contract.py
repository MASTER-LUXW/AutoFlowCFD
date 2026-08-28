"""
AutoFlowCFD V2.0 - GPU 版共享算子张量收缩（体积项核心）

与 core/fr_operators/volume_contract.py 完全对应的 CuPy 版本。

数学公式完全一致：
- contract_shared_operator_1axis: 等价于 np.einsum("fs,csv->cfv", D, X)
- contract_shared_operator_2axis: 等价于 np.einsum("fjm,cjmv->cfv", D, X)

实现用 `cp.matmul` 广播、不用 `cp.tensordot`：`tensordot(D, X, ...)` 把
不依赖 cell 的小算子 `D` 当作 `a`、逐 cell 的大数组 `X` 当作 `b`，内部
按 `newaxes_b = contract轴 + notcontract轴` 转置 `b`——X 的 cell 轴不在
被收缩的轴里，会被搬到中间，产生一份完整的转置副本。CPU 版
（`core/fr_operators/volume_contract.py`）在 364,555 cell 的生产网格上
P2 阶数实测触发过 `Unable to allocate 1.10 GiB`（cell 轴 153,950 个
prism 撞上转置副本），GPU 显存比主存更紧张，同一模式风险更高，因此
同步改为 `cp.matmul` 广播：X 的 cell 轴留在最前面全程不转置，数学上
与 `tensordot`+`moveaxis` 完全等价（已用 CPU 端随机张量数值验证过，
两者输出最大绝对误差在 float64 机器精度量级，见
`core/fr_operators/volume_contract.py` 模块文档）。
"""

from autoflowcfd.core.gpu import get_cupy


def gpu_contract_shared_operator_1axis(D, X):
    """GPU 版 contract_shared_operator_1axis。

    等价于 np.einsum("fs,csv->cfv", D, X)。

    Args:
        D: (F, S) CuPy 数组，不依赖 cell 的共享算子
        X: (C, S, V) CuPy 数组，逐 cell 的场

    Returns:
        (C, F, V) CuPy 数组
    """
    cp = get_cupy()
    # cp.matmul 对 2D @ 3D 按批量矩阵乘广播：D (F,S) 广播为逐 cell 共享
    # 的左矩阵，与 X 的每个 (S,V) 切片做 gemm，得到 (C,F,V)——不转置 X
    # 的 cell 轴，见模块文档。
    return cp.ascontiguousarray(cp.matmul(D, X))


def gpu_contract_shared_operator_2axis(D, X):
    """GPU 版 contract_shared_operator_2axis。

    等价于 np.einsum("fjm,cjmv->cfv", D, X)。

    Args:
        D: (F, J, M) CuPy 数组，不依赖 cell 的共享算子
        X: (C, J, M, V) CuPy 数组，逐 cell 的场

    Returns:
        (C, F, V) CuPy 数组
    """
    cp = get_cupy()
    F, J, M = D.shape
    C, _, _, V = X.shape
    D_flat = D.reshape(F, J * M)
    X_flat = X.reshape(C, J * M, V)
    return cp.ascontiguousarray(cp.matmul(D_flat, X_flat))
