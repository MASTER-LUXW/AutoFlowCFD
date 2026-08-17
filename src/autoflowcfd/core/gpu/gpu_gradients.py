"""
AutoFlowCFD V2.0 - GPU 版物理空间梯度计算

与 core/fr_gradients.py 对应的 CuPy 版本。
核心操作：共享算子张量收缩 + 度量项链式法则。

数学公式完全一致：
1. 参考空间梯度：grad_comp = D @ field（通过 gpu_contract_shared_operator）
2. 物理空间梯度：grad_phys = grad_comp^T @ inv_jacs（通过 cp.matmul）
"""

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.gpu_volume_contract import gpu_contract_shared_operator_1axis


def compute_physical_gradient_gpu(field, mesh_data, ops_data):
    """GPU 版物理空间梯度计算。

    与 core/fr_gradients.py::compute_physical_gradient 公式完全一致。

    Args:
        field: CuPy 数组 (n_cells, n_sps, n_field_vars)
        mesh_data: dict，包含 'inv_jacs', 'n_prism' 等
        ops_data: dict，包含 'D_3d_tet', 'D_3d_prism' 等

    Returns:
        grad: CuPy 数组 (n_cells, n_sps, n_field_vars, 3)
    """
    cp = get_cupy()
    n_cells, n_sps, n_field_vars = field.shape
    inv_jacs = mesh_data['inv_jacs']  # (n_cells, n_sps, 3, 3)
    n_prism = mesh_data.get('n_prism', 0)

    # 参考空间梯度
    grad_comp = cp.zeros((n_cells, n_sps, 3, n_field_vars), dtype=cp.float64)

    if n_prism > 0:
        D_3d_prism = ops_data['D_3d_prism']  # (n_sps, n_sps, 3)
        D2 = cp.ascontiguousarray(D_3d_prism.transpose(0, 2, 1).reshape(n_sps * 3, n_sps))
        grad_comp[:n_prism] = gpu_contract_shared_operator_1axis(
            D2, field[:n_prism]
        ).reshape(n_prism, n_sps, 3, n_field_vars)

    if n_cells > n_prism:
        n_tet = n_cells - n_prism
        D_3d_tet = ops_data['D_3d_tet']  # (n_sps, n_sps, 3)
        D2 = cp.ascontiguousarray(D_3d_tet.transpose(0, 2, 1).reshape(n_sps * 3, n_sps))
        grad_comp[n_prism:] = gpu_contract_shared_operator_1axis(
            D2, field[n_prism:]
        ).reshape(n_tet, n_sps, 3, n_field_vars)

    # 链式法则：grad_phys[c,s,v,n] = sum_m inv_jac[c,s,m,n] * grad_comp[c,s,m,v]
    # 转置 grad_comp 的 (m,v) → (v,m)，与 inv_jacs 的 (m,n) 相乘
    grad_phys = cp.matmul(cp.swapaxes(grad_comp, -1, -2), inv_jacs)
    return grad_phys


def compute_physical_scalar_gradient_gpu(scalar_field, mesh_data, ops_data):
    """GPU 版标量场物理空间梯度。

    Args:
        scalar_field: CuPy 数组 (n_cells, n_sps) 或 (n_cells, n_sps, 1)
        mesh_data: dict
        ops_data: dict

    Returns:
        grad: CuPy 数组 (n_cells, n_sps, 3)
    """
    cp = get_cupy()
    if scalar_field.ndim == 2:
        scalar_field = scalar_field[..., None]  # (n_cells, n_sps, 1)

    grad = compute_physical_gradient_gpu(scalar_field, mesh_data, ops_data)
    return grad[..., 0]  # (n_cells, n_sps, 3)
