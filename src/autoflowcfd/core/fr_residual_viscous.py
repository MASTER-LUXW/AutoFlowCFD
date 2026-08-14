"""
AutoFlowCFD V2.0 - FR 粘性残差计算入口 (S-03, Tier-0 重建版)

真正的物理通量函数、度量项一致的体积项、基于真实单元-面连接关系的
BR1 界面耦合都在 core/fr_viscous_flux.py 中实现（该文件的历史版本
体积过大且混杂了从未被真正触发的"完整LDG"死代码分支，已拆分——
理由与验证记录见该文件模块文档）。本文件保留对外的公开函数名
（compute_viscous_residual/compute_gradients/compute_scalar_gradient），
避免影响其他调用方，但函数体委托给已验证正确的新实现。
"""

import numpy as np

from autoflowcfd.core.fr_gradients import compute_physical_gradient, compute_physical_scalar_gradient
from autoflowcfd.core.fr_viscous_flux import compute_viscous_residual_fr


def compute_viscous_residual(
    state_U: np.ndarray,
    state_Q: np.ndarray,
    ops,
    mesh,
    mu: float = 1.8e-5,
    Pr: float = 0.72,
    gamma: float = 1.4,
    mu_t_field: np.ndarray = None,
    Pr_t: float = 0.9,
    boundary_ghost_provider=None,
) -> np.ndarray:
    """计算粘性残差。真正的实现见 fr_viscous_flux.compute_viscous_residual_fr。

    Args:
        state_U: 守恒变量 (n_cells, n_sps, n_vars)
        state_Q: 原始变量（保留参数以兼容旧调用签名；新实现从 state_U
            自行重新计算原始变量，因为需要与梯度计算共用同一份、
            保证一致性的 conserved_to_primitive 转换）
        mu: **有效**动力粘度（分子粘度 + 湍流涡粘度，调用方负责传入
            湍流模型算出的 mu_t 之和——这是本次修复的一部分：旧版本
            调用方从不传 mu，粘性残差永远只用分子粘度，湍流模型算出的
            涡粘系数从未真正进入粘性应力张量，见 core/fr_solver.py
            compute_viscous_residual 的调用处）

    Returns:
        viscous_res: 形状与 state_U 相同（n_vars>5 时高阶湍流分量补零，
            湍流量自身的输运方程仍由 compute_turbulence_source 单独处理）
    """
    n_vars = state_U.shape[-1]
    res_euler = compute_viscous_residual_fr(
        state_U, mesh, ops, mu=mu, Pr=Pr, mu_t_field=mu_t_field, Pr_t=Pr_t,
        boundary_ghost_provider=boundary_ghost_provider,
    )
    if n_vars > 5:
        res_full = np.zeros(state_U.shape)
        res_full[:, :, :5] = res_euler
        return res_full
    return res_euler


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
