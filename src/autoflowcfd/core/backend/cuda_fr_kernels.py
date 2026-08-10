"""
AutoFlowCFD V2.0 - FR 求解器 GPU 加速内核 (B-01)

本模块使用 Numba 实现 FR 核心计算逻辑的 GPU 并行化。
"""

import numpy as np
from numba import cuda, float64


@cuda.jit
def fr_residual_kernel(U, Q, dU_dt, diff_matrix, n_cells, n_sps):
    """
    并行计算每个单元内 SPs 上的残差。
    """
    cell_id = cuda.grid(1)
    if cell_id >= n_cells:
        return
    
    # 局部求导逻辑 (简化版)
    for var in range(5): # 5个守恒变量
        for sp in range(n_sps):
            grad_sum = 0.0
            for k in range(n_sps):
                grad_sum += diff_matrix[sp, k] * Q[cell_id, k, var]
            dU_dt[cell_id, sp, var] = -grad_sum # 简化残差

def launch_fr_residual(U, Q, dU_dt, diff_matrix):
    """启动 GPU 内核。"""
    n_cells, n_sps, _ = U.shape
    threads_per_block = 256
    blocks_per_grid = (n_cells + threads_per_block - 1) // threads_per_block
    
    fr_residual_kernel[blocks_per_grid, threads_per_block](
        cuda.to_device(U), cuda.to_device(Q), cuda.to_device(dU_dt), 
        cuda.to_device(diff_matrix), n_cells, n_sps
    )
    
    return dU_dt