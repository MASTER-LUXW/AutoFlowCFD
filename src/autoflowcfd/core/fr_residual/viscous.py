"""
AutoFlowCFD V2.0 - 守恒变量 / 标量场的物理梯度入口（S-03）。

粘性物理通量、体积项与界面耦合在 `core/fr_residual/viscous_flux/`（`compute_viscous_residual_fr`）；
本文件给求解器与湍流模型提供梯度函数，委托给 `core/fr_operators/gradients.py`。
"""

import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient, compute_physical_scalar_gradient


def compute_gradients(U: np.ndarray, ops, mesh=None) -> np.ndarray:
    """计算守恒变量的物理空间梯度（度量项一致）。

    Args:
        U: 守恒变量，形状 (n_cells, n_sps, n_vars)
        mesh: 提供度量项（jacobians）；缺失时报错（没有度量项就算不出物理梯度）。

    Returns:
        grad_U: 形状 (n_cells, n_sps, n_vars, 3)
    """
    if mesh is None:
        raise ValueError(
            "compute_gradients requires `mesh` to apply the metric-term chain rule "
            "(see core/fr_operators/gradients.py)"
        )
    return compute_physical_gradient(U, mesh, ops)


def compute_scalar_gradient(scalar_field: np.ndarray, ops, mesh=None) -> np.ndarray:
    """标量场版本，见 compute_gradients 的说明。"""
    if mesh is None:
        raise ValueError(
            "compute_scalar_gradient now requires `mesh` to apply the metric-term chain rule."
        )
    return compute_physical_scalar_gradient(scalar_field, mesh, ops)
