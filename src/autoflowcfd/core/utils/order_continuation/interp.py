"""AutoFlowCFD V2.0 - 单机 CPU 的升阶（解场 / 湍流场的精确延拓）"""

from typing import Any

from autoflowcfd.core.fr_solver.state import FRState
from autoflowcfd.core.utils.order_continuation.order_switch import (
    install_turbulence, reset_order_dependent_state, switched_fields,
)


def interpolate_to_new_order(solver: Any, new_order: int) -> None:
    """单机 CPU 升阶：解与湍流场精确延拓（`order_switch.switched_fields`），换阶后失效的跨步状态复位。

    算子、网格几何、壁面距离、幽灵态由调用方 `FRSolver._interpolate_to_new_order` 切换。状态整体换成新阶数的
    `FRState`（此前就地改 `U` / `Q`，残差数组 `dU_dt` 留着旧阶数的形状）。
    """
    old_order = solver.current_order
    if new_order == old_order:
        return
    turb = solver.turb_model
    switched = switched_fields(old_order, new_order, int(solver.mesh.n_prism_cells), solver.freestream,
                               solver.state.U, turb, nu_t=None if turb is None else turb.nu_t)
    state = FRState(switched.U.shape[0], switched.U.shape[1])
    state.U = switched.U
    state._update_primitives()
    solver.state = state
    install_turbulence(turb, switched)
    reset_order_dependent_state(solver, solver.sgs_model)
    solver.current_order = new_order
