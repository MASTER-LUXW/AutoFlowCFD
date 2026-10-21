"""
AutoFlowCFD V2.0 - GPU 版物理空间梯度计算

与 core/fr_operators/gradients.py 对应的 CuPy 版本。
核心操作：共享算子张量收缩 + 度量项链式法则。

数学公式完全一致：
1. 参考空间梯度：grad_comp = D @ field（通过 gpu_contract_shared_operator）
2. 物理空间梯度：grad_phys = grad_comp^T @ inv(J)，inv(J) = adj(J) / det(J)（设备上只存 det 与 adj，
   `gpu_inviscid_volume.prepare_mesh_data`；先乘 adj 再除以 det，不物化 (n_cells, n_sps, 3, 3) 的 inv(J)）
"""

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.residual.gpu_volume_contract import gpu_contract_shared_operator_1axis


def compute_physical_gradient_gpu(field, mesh_data, ops_data):
    """GPU 版物理空间梯度计算。

    与 core/fr_operators/gradients.py::compute_physical_gradient 公式完全一致。

    Args:
        field: CuPy 数组 (n_cells, n_sps, n_field_vars)
        mesh_data: dict，包含 'det_jacs'、'adj_j'、'n_prism' 等
        ops_data: dict，包含 'D_native_prism_padded'、'D_native_tet_padded'

    Returns:
        grad: CuPy 数组 (n_cells, n_sps, n_field_vars, 3)
    """
    cp = get_cupy()
    n_cells, n_sps, n_field_vars = field.shape
    adj_j = mesh_data['adj_j']  # (n_cells, n_sps, 3, 3)
    det_jacs = mesh_data['det_jacs']  # (n_cells, n_sps)
    n_prism = mesh_data.get('n_prism', 0)

    # 参考空间梯度
    grad_comp = cp.zeros((n_cells, n_sps, 3, n_field_vars), dtype=cp.float64)

    if n_prism > 0:
        D_prism = ops_data['D_native_prism_padded']  # (n_sps, n_sps, 3)
        D2 = cp.ascontiguousarray(D_prism.transpose(0, 2, 1).reshape(n_sps * 3, n_sps))
        grad_comp[:n_prism] = gpu_contract_shared_operator_1axis(
            D2, field[:n_prism]
        ).reshape(n_prism, n_sps, 3, n_field_vars)

    if n_cells > n_prism:
        n_tet = n_cells - n_prism
        D_tet_op = ops_data['D_native_tet_padded']  # (n_sps, n_sps, 3)
        D2 = cp.ascontiguousarray(D_tet_op.transpose(0, 2, 1).reshape(n_sps * 3, n_sps))
        grad_comp[n_prism:] = gpu_contract_shared_operator_1axis(
            D2, field[n_prism:]
        ).reshape(n_tet, n_sps, 3, n_field_vars)

    # 链式法则：grad_phys[c,s,v,n] = sum_m inv_jac[c,s,m,n] * grad_comp[c,s,m,v]，inv_jac = adj / det
    grad_phys = cp.matmul(cp.swapaxes(grad_comp, -1, -2), adj_j)
    grad_phys /= det_jacs[:, :, None, None]
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
    if scalar_field.ndim == 2:
        scalar_field = scalar_field[..., None]  # (n_cells, n_sps, 1)

    grad = compute_physical_gradient_gpu(scalar_field, mesh_data, ops_data)
    # 真实 bug 修复（V2.0 专家组盲审第四轮，2026-08-28）：`grad` 形状是
    # (n_cells, n_sps, n_field_vars=1, 3)——`grad[..., 0]` 对末轴（3 个
    # 空间分量那一维）取索引 0，等价于"只留 x 分量、丢掉 y/z"，还留了一个
    # 多余的 n_field_vars=1 轴，输出形状 (n_cells, n_sps, 1) 而不是文档
    # 承诺的 (n_cells, n_sps, 3)——本机没有 CuPy，这个函数此前从未被
    # 实际执行验证过；`compute_turbulence_source_gpu`（SST 源项唯一调用
    # 处，用于 grad_k/grad_omega）会因此拿到被截断成标量的假梯度。正确
    # 做法是去掉 n_field_vars 这个多余轴（大小恒为 1），保留完整的 3
    # 空间分量：`grad[:, :, 0, :]`。
    return grad[:, :, 0, :]  # (n_cells, n_sps, 3)
