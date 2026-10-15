"""
AutoFlowCFD V2.0 - FR 粘性残差计算入口 (S-03, Tier-0 重建版)

真正的物理通量函数、度量项一致的体积项、基于真实单元-面连接关系的
BR1 界面耦合都在 core/fr_viscous_flux.py 中实现（该文件的历史版本
体积过大且混杂了从未被真正触发的"完整LDG"死代码分支，已拆分——
理由与验证记录见该文件模块文档）。本文件保留梯度的公开函数名
（compute_gradients/compute_scalar_gradient），函数体委托给已验证正确的新实现。粘性残差
直接调用 `compute_viscous_residual_fr`（2026-10-15 删掉了旧签名包装 `compute_viscous_residual`：
它的 `state_Q` 参数从未被使用，分布式路径为了凑这个参数每次多算一遍原始变量）。
"""

import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient, compute_physical_scalar_gradient


def compute_gradients(U: np.ndarray, ops, mesh=None) -> np.ndarray:
    """计算守恒变量的物理空间梯度（度量项一致）。

    Args:
        U: 守恒变量，形状 (n_cells, n_sps, n_vars)
        mesh: 需要提供 jacobians；为兼容旧调用位置参数保持在最后且可选，
            但物理正确的梯度离不开度量项，缺失时直接报错而不是静默退化
            为错误的（旧版本行为——直接用 D_3d 冒充物理导数）。

    Returns:
        grad_U: 形状 (n_cells, n_sps, n_vars, 3)
    """
    if mesh is None:
        raise ValueError(
            "compute_gradients now requires `mesh` to apply the metric-term chain rule "
            "(see core/fr_gradients.py) - physical gradients cannot be computed correctly "
            "without it for curved/collapsed-coordinate cells."
        )
    return compute_physical_gradient(U, mesh, ops)


def compute_scalar_gradient(scalar_field: np.ndarray, ops, mesh=None) -> np.ndarray:
    """标量场版本，见 compute_gradients 的说明。"""
    if mesh is None:
        raise ValueError(
            "compute_scalar_gradient now requires `mesh` to apply the metric-term chain rule."
        )
    return compute_physical_scalar_gradient(scalar_field, mesh, ops)
