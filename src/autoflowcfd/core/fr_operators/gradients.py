"""
AutoFlowCFD V2.0 - 解点上的物理空间梯度（链式法则经逐点度量项转换）。

参考空间微分算子给出 ∂φ/∂ξ_m，物理梯度按链式法则

    ∂φ/∂x_n = Σ_m (∂ξ_m/∂x_n) * ∂φ/∂ξ_m = Σ_m inv_jac[m,n] * (D[:,:,m] @ φ)

棱柱与四面体各用自己原生基的微分矩阵（`fr/operators/container.py::FROperators`）。线性物理函数的梯度精确恢复为
其常数梯度（机器精度，`tests/unit/test_fr_gradients.py`）。
"""

import numpy as np

from numba import njit, prange

from autoflowcfd.core.fr_operators.volume_contract import cell_blocks, contract_local, grad_to_physical_point


@njit(cache=True, parallel=True)
def _physical_gradient_kernel(field, lo, hi, D2, inv_jacs, out, n_blocks):
    """单元 `lo..hi-1`：`out[c, s, v, n] = sum_m inv_jacs[c, s, m, n] * (D2 field[c])[3 s + m, v]`。"""
    n = hi - lo
    n_sps = field.shape[1]
    for b in prange(n_blocks):
        g_c = np.empty((3 * n_sps, field.shape[2]))
        for i in range(b * n // n_blocks, (b + 1) * n // n_blocks):
            c = lo + i
            contract_local(D2, field[c], g_c)
            for s in range(n_sps):
                grad_to_physical_point(g_c[3 * s:3 * s + 3], inv_jacs[c, s], out[c, s])


def compute_physical_gradient(field: np.ndarray, mesh, ops) -> np.ndarray:
    """计算场变量在物理空间中的梯度（逐解点度量项）。

    Args:
        field: 形状 (n_cells, n_sps, n_field_vars)，SPs 上的场值
        mesh: HighOrderMesh 实例（需要 jacobians['inv_jacs']）
        ops: FROperators（需要 D_3d）

    Returns:
        grad: 形状 (n_cells, n_sps, n_field_vars, 3)，物理空间梯度
    """
    n_cells, n_sps, n_field_vars = field.shape
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # 棱柱 / 四面体原生基各自的微分矩阵（`fr/operators/container.py::FROperators`）。
    #
    # 性能优化：D（形状 (s,j,m)）不依赖 cell，只有 field（c,j,v）依赖
    # cell——把 D 的 (s,m) 两个输出轴摊平、j 挪到最后一维，转成
    # `contract_shared_operator_1axis` 认识的 (S*M, J) 2D 形状，收缩后
    # 再 reshape 回 (c,s,m,v)；数学上与 `np.einsum("sjm,cjv->csmv", D,
    # field)` 严格等价（同一个求和，只是换一条 BLAS gemm 计算路径），
    # 原因/验证方式见 core/fr_operators/volume_contract.py 模块文档——生产网格上
    # `compute_physical_gradient` 的 einsum 是体积项性能优化里的另一个
    # 主要热点（py-spy 采样证实）。
    # 链式法则转物理空间：grad_phys[c,s,v,n] = sum_m inv_jac[c,s,m,n] * grad_comp[c,s,m,v]
    # 输出维度顺序 (n_cells,n_sps,n_field_vars,3)，与代码库既有 grad_U 约定一致。
    # 逐点链式法则 `volume_contract.grad_to_physical_point`（性能优化 2026-09-13：原
    # `np.matmul(np.swapaxes(grad_comp,-1,-2), inv_jacs)` 要先物化一份非连续转置副本、再做数百万次
    # (V,3)@(3,3) 微型 gemm，开销主导且不随核数并行）。对 m 求和顺序与 np.matmul 一致，结果逐位相同。
    #
    # **逐单元融合**（2026-10-15）：每个单元先收缩出计算空间梯度、再按链式法则转物理空间，中间量只在块内
    # 局部工作数组里（`_physical_gradient_kernel`）。此前按 32768 单元分块，每块分配收缩结果与转换结果两份
    # 块大小的临时数组（P2/V=5 每份约 106 MiB）——Windows 上新分配的大数组每 MB 约 0.21 ms 花在缺页上，是
    # 复用内存的 10 倍以上，一次残差求值要做 6 次梯度。求和顺序取自同一份内联定义（`contract_local` /
    # `grad_to_physical_point`），与分块版逐位相同。prism/tet 两段用不同的微分矩阵，单元存储本来就是
    # prism 在前 tet 在后。
    n_prism = mesh.n_prism_cells
    grad_phys = np.empty((n_cells, n_sps, n_field_vars, 3))
    D2_prism = None
    if n_prism > 0:
        D2_prism = np.ascontiguousarray(
            np.transpose(ops.D_native_prism_padded, (0, 2, 1))).reshape(n_sps * 3, n_sps)
    # 原生四面体：零填充到 n_sps 宽度的 `D_native_tet_padded`（填充行的导数恒为 0）。
    D2_tet = None
    if n_cells > n_prism:
        tet_ops = ops.D_native_tet_padded
        D2_tet = np.ascontiguousarray(
            np.transpose(tet_ops, (0, 2, 1))).reshape(n_sps * 3, n_sps)

    for seg_lo, seg_hi, D2 in ((0, n_prism, D2_prism), (n_prism, n_cells, D2_tet)):
        if D2 is None or seg_hi <= seg_lo:
            continue
        _physical_gradient_kernel(field, seg_lo, seg_hi, D2, inv_jacs, grad_phys, cell_blocks(seg_hi - seg_lo))
    return grad_phys


def compute_physical_scalar_gradient(scalar_field: np.ndarray, mesh, ops) -> np.ndarray:
    """标量场版本（去掉 field_vars 维度的便捷包装）。

    Args:
        scalar_field: 形状 (n_cells, n_sps) 或 (n_cells, n_sps, 1)

    Returns:
        grad: 形状 (n_cells, n_sps, 3)

    输入是 2D 还是"3D 且末轴为 1"都返回 (n_cells, n_sps, 3)（末尾挤掉单位轴）。
    """
    if scalar_field.ndim == 2:
        scalar_field = scalar_field[:, :, np.newaxis]
    grad = compute_physical_gradient(scalar_field, mesh, ops)  # (n_cells,n_sps,1,3)
    return grad[:, :, 0, :]
