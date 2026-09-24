"""AutoFlowCFD V2.0 - 湍流涡粘修正的施加与涡粘场取用。

从 `core/fr_solver/turbulence.py` 拆出（2026-09-24）。纯搬家，逻辑未改。
"""

from typing import Optional

import numpy as np
from loguru import logger

from autoflowcfd.core.fr_residual.viscous import compute_gradients as _compute_gradients_generic


def apply_turbulence_corrections(solver) -> None:
    """应用湍流模型的修正（SGS 涡粘系数）。

    WMLES 壁面剪应力**不**在这里施加，见
    FRSolver.compute_viscous_residual()/apply_turbulence_corrections()
    文档（T-05 修复：必须在残差组装阶段生效，这里在 step() 中排在状态
    更新之后，为时已晚）。
    """
    if solver.sgs_model is not None:
        # 真实 bug 修复（2026-09-03）：同 compute_turbulence_source 里的
        # grad_vel 修复，理由见该函数文档——LES/WMLES 的 SGS 涡粘同样
        # 不能用动量梯度冒充速度梯度。
        grad_u = _compute_gradients_generic(solver.state.Q[:, :, 1:4], solver.ops, solver.mesh)
        delta = solver._get_grid_scale()
        nu_t = solver.sgs_model.compute_eddy_viscosity(grad_u, delta)

        if hasattr(solver.turb_model, "nu_t"):
            solver.turb_model.nu_t += nu_t
            logger.debug(f"SGS eddy viscosity added to turbulence model: mean={nu_t.mean():.6e}")
        else:
            solver.sgs_model.nu_t = nu_t
            logger.debug(f"SGS eddy viscosity computed: mean={nu_t.mean():.6e}, max={nu_t.max():.6e}")


def get_turbulent_viscosity_field(solver) -> Optional[np.ndarray]:
    """汇总当前激活的湍流模型给出的动力涡粘度场 mu_t = rho * nu_t。"""
    rho = solver.state.Q[:, :, 0]
    mu_t_total = None

    if solver.turb_model is not None and hasattr(solver.turb_model, "nu_t"):
        mu_t_total = rho * solver.turb_model.nu_t
    if solver.sgs_model is not None and hasattr(solver.sgs_model, "nu_t") and solver.sgs_model.nu_t is not None:
        sgs_contrib = rho * solver.sgs_model.nu_t
        mu_t_total = sgs_contrib if mu_t_total is None else mu_t_total + sgs_contrib

    return mu_t_total
