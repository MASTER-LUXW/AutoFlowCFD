"""Q 准则涡识别（Hunt, Wray & Moin 1988）：Q = 0.5 (||Omega||^2 - ||S||^2)，Q > 0 为旋转主导区。

速度梯度与湍流模型源项用同一个实现（`core/fr_operators/corrected_gradient.py::source_velocity_gradient`）：
P>=1 为解点上的单元内多项式导数，P0 为提升修正的 Green–Gauss 梯度（边界取与残差同一个幽灵态）。
解点上的 Q 按求积权重求单元积分平均后供低阶 VTK 导出。
"""

import numpy as np


def q_criterion_from_velocity_gradient(grad: np.ndarray) -> np.ndarray:
    """`grad[..., i, j] = du_i/dx_j` -> Q（形状去掉最后两维）。"""
    S = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    Omega = 0.5 * (grad - np.swapaxes(grad, -1, -2))
    return 0.5 * (np.sum(Omega * Omega, axis=(-2, -1)) - np.sum(S * S, axis=(-2, -1)))


def q_criterion_cell_field(solver) -> np.ndarray:
    """求解器当前状态的 Q 准则单元积分平均 `(n_cells,)`（单机求解器或其主机视图）。"""
    from autoflowcfd.core.fr_operators.corrected_gradient import source_velocity_gradient
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.fr.native_padding import cell_integration_weights, weighted_cell_average

    host = solver.host_view() if hasattr(solver, "host_view") else solver
    flat = get_flat_face_geometry(host.mesh, host.ops)
    grad = source_velocity_gradient(host.state.Q, host.mesh, host.ops, flat, host.boundary_ghost_provider)
    return weighted_cell_average(q_criterion_from_velocity_gradient(grad), cell_integration_weights(host))
