"""
AutoFlowCFD V2.0 - 物理空间梯度计算（度量项一致，修复版）

此前 fr_residual_viscous.py 的 compute_gradients / compute_scalar_gradient
把计算立方体微分算子 D_3d（给出 ∂φ/∂ξ_m，即相对计算坐标 a,b,c 的导数）
直接当作物理空间导数 ∂φ/∂x_n 使用，没有做任何度量项变换。这对笛卡尔
张量积（未曲变的规则六面体）单元恰好凑巧正确，但对本代码库中的每一个
四面体/棱柱单元（曲边/坍缩坐标映射，见 grid/curved_mapping.py）都是
错误的导数——物理正确的链式法则是：

    ∂φ/∂x_n = Σ_m (∂ξ_m/∂x_n) * ∂φ/∂ξ_m = Σ_m inv_jac[m,n] * (D_3d[:,:,m] @ φ)

已用线性函数解析解验证（对任意非退化四面体/棱柱，线性物理函数的梯度
应精确恢复为其真实常数梯度，验证误差在机器精度量级，见
tests/unit/test_fr_gradients.py）。
"""

import numpy as np


def compute_physical_gradient(field: np.ndarray, mesh, ops) -> np.ndarray:
    """计算场变量在物理空间中的梯度，正确处理曲边/坍缩坐标度量项。

    Args:
        field: 形状 (n_cells, n_sps, n_field_vars)，SPs 上的场值
        mesh: HighOrderMesh 实例（需要 jacobians['inv_jacs']）
        ops: FROperators（需要 D_3d）

    Returns:
        grad: 形状 (n_cells, n_sps, n_field_vars, 3)，物理空间梯度
    """
    n_cells, n_sps, n_field_vars = field.shape
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)

    # 四面体/棱柱专用坍缩坐标微分矩阵（不能用朴素张量积 D_3d，理由见
    # fr/operators.py::FROperators.D_3d_tet/D_3d_prism 文档）。
    n_prism = mesh.n_prism_cells
    grad_comp = np.zeros((n_cells, n_sps, 3, n_field_vars))
    if n_prism > 0:
        grad_comp[:n_prism] = np.einsum("sjm,cjv->csmv", ops.D_3d_prism, field[:n_prism])
    if n_cells > n_prism:
        grad_comp[n_prism:] = np.einsum("sjm,cjv->csmv", ops.D_3d_tet, field[n_prism:])
    # 链式法则转物理空间：grad_phys[c,s,v,n] = sum_m inv_jac[c,s,m,n] * grad_comp[c,s,m,v]
    # 输出维度顺序 (n_cells,n_sps,n_field_vars,3)，与代码库既有 grad_U 约定一致
    grad_phys = np.einsum("csmn,csmv->csvn", inv_jacs, grad_comp)
    return grad_phys


def compute_physical_scalar_gradient(scalar_field: np.ndarray, mesh, ops) -> np.ndarray:
    """标量场版本（去掉 field_vars 维度的便捷包装）。

    Args:
        scalar_field: 形状 (n_cells, n_sps) 或 (n_cells, n_sps, 1)

    Returns:
        grad: 形状 (n_cells, n_sps, 3)

    修复记录：此前只在输入是 2D (n_cells,n_sps) 时才补一个 field_vars 轴再
    在最后挤掉；但唯一真实调用方（core/fr_solver_turbulence.py 计算
    SST/DDES 的 grad_k/grad_omega）传入的是已经手动加过 (n_cells,n_sps,1)
    这个轴的 3D 数组，导致这里的挤压条件从未触发，返回值多出一个不该有
    的轴 (n_cells,n_sps,1,3)——下游 turbulence_sst.py 的交叉扩散项
    `np.sum(grad_k*grad_omega, axis=2)` 会把这个多余的单位轴当成待求和的
    维度（axis=2 现在指向它而不是本该求和的空间维），真正的 3 分量空间
    向量完全没被点积掉；再与形状 (n_cells,n_sps) 的 rho/F1 等场相乘时，
    错位广播成 (n_cells,n_cells,3)（真实网格已复现该崩溃）。现在统一在
    末尾挤掉这个轴，与输入是 2D 还是"3D 且末轴为1"无关。
    """
    if scalar_field.ndim == 2:
        scalar_field = scalar_field[:, :, np.newaxis]
    grad = compute_physical_gradient(scalar_field, mesh, ops)  # (n_cells,n_sps,1,3)
    return grad[:, :, 0, :]
