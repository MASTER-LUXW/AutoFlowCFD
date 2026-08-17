"""
AutoFlowCFD V2.0 - GPU 版共享算子张量收缩（体积项核心）

与 core/fr_volume_contract.py 完全对应的 CuPy 版本。
将 np.tensordot 替换为 cp.tensordot，底层自动走 cuBLAS gemm。

数学公式完全一致：
- contract_shared_operator_1axis: 等价于 np.einsum("fs,csv->cfv", D, X)
- contract_shared_operator_2axis: 等价于 np.einsum("fjm,cjmv->cfv", D, X)

性能：CuPy 的 tensordot 内部调用 cuBLAS gemm，对大规模网格
（545K cells, P2, 27 SPs/cell）的张量收缩有显著加速。
"""

import numpy as np
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
    out = cp.tensordot(D, X, axes=([1], [1]))  # (F, C, V)
    return cp.ascontiguousarray(cp.moveaxis(out, 0, 1))  # (C, F, V)


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
    out = cp.tensordot(D, X, axes=([1, 2], [1, 2]))  # (F, C, V)
    return cp.ascontiguousarray(cp.moveaxis(out, 0, 1))  # (C, F, V)
