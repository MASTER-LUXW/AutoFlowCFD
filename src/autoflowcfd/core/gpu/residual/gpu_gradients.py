"""
AutoFlowCFD V2.0 - GPU 版物理空间梯度计算

与 core/fr_gradients.py 对应的 CuPy 版本。
核心操作：共享算子张量收缩 + 度量项链式法则。

数学公式完全一致：
1. 参考空间梯度：grad_comp = D @ field（通过 gpu_contract_shared_operator）
2. 物理空间梯度：grad_phys = grad_comp^T @ inv_jacs（通过 cp.matmul）
"""

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.residual.gpu_volume_contract import gpu_contract_shared_operator_1axis


def compute_physical_gradient_gpu(field, mesh_data, ops_data):
    """GPU 版物理空间梯度计算。

    与 core/fr_gradients.py::compute_physical_gradient 公式完全一致。

    真实 bug 修复（2026-09-03，排查"native 是否已补全"时用真实含
    native 四面体的合成网格 + numpy-as-cupy 替身首次决定性对照 CPU
    才发现——本机没有真实 CuPy，这个函数此前从未被真正执行验证过）：
    四面体分支此前无条件用 `ops_data['D_3d_tet']`（坍缩坐标微分算子），
    从未像 `gpu_inviscid_volume.py::compute_volume_term_gpu` 那样按
    `tet_basis_mode` 分派到 `D_native_tet_padded`——`tet_basis_mode=
    "native"` 时四面体的物理梯度因此用了完全错误的参考空间微分算子，
    与 CPU 版 `core/fr_gradients.py::compute_physical_gradient`（早已
    正确按 `mesh.tet_basis_mode` 分派）不一致，误差量级与场本身同阶
    （不是浮点噪声）。棱柱不受影响（棱柱不区分 collapsed/native，两种
    模式下都用同一个 `D_3d_prism`），这也是此前"均匀/近似均匀"这类
    测试从未捕捉到的原因——需要真正非零、且四面体单元占比不小的场景
    才会暴露。

    修复：不需要额外的 `tet_basis_mode` 显式参数——`ops_data` 本身已经
    足够自描述：`generate_fr_operators`/`prepare_ops_data` 只在
    `tet_basis_mode="native"` 时才会往 `ops_data` 里写
    `D_native_tet_padded` 这个 key（collapsed 模式下该 key 根本不存在，
    见 `compute_volume_term_gpu` 同一处判据"纯坍缩坐标网格下
    `ops.D_native_tet_padded` 恒为 None，不写入 data"的说明）——用
    `'D_native_tet_padded' in ops_data` 自描述判断，不需要改动本函数
    任何一个调用点的签名（分布在 gpu_viscous.py/gpu_solver_io.py/
    gpu_distributed_init.py/gpu_scalar_transport.py 共 7 处）。

    Args:
        field: CuPy 数组 (n_cells, n_sps, n_field_vars)
        mesh_data: dict，包含 'inv_jacs', 'n_prism' 等
        ops_data: dict，包含 'D_3d_tet', 'D_3d_prism'，native 网格上
            还含 'D_native_tet_padded' 等

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
        D_tet_op = (
            ops_data['D_native_tet_padded']
            if 'D_native_tet_padded' in ops_data
            else ops_data['D_3d_tet']
        )  # (n_sps, n_sps, 3)
        D2 = cp.ascontiguousarray(D_tet_op.transpose(0, 2, 1).reshape(n_sps * 3, n_sps))
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
