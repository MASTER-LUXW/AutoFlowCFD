"""AutoFlowCFD V2.0 - 全新求解器在 Order Continuation 起步时重建到 P0"""

from autoflowcfd.core.fr_solver.state import FRState
from autoflowcfd.core.utils.order_continuation.order_switch import (
    install_turbulence, reset_order_dependent_state, switch_mesh_order, switched_fields,
)


def _reset_state_to_p0(solver) -> None:
    """把一个**全新**（非 resume）求解器的状态重建到 P0。

    Order Continuation 从 P0 起步；全新求解器构造时按目标阶数分配了状态，这里用均匀自由来流（速度方向取自攻角 /
    侧滑角）、模型的来流湍流值、零涡粘在 P0 上重建它（`order_switch.switched_fields` 降阶分支），并把算子、网格
    阶数、壁面距离、SGS 涡粘、双时间步上一时间层与 Newton 跨步状态一并切到 P0。
    """
    from autoflowcfd.core.fr_solver.turbulence import recompute_wall_distance_for_current_order

    switched = switched_fields(solver.current_order, 0, 0, solver.freestream, solver.state.U, solver.turb_model)
    state = FRState(switched.U.shape[0], switched.U.shape[1])
    state.U = switched.U
    state._update_primitives()
    solver.state = state
    install_turbulence(solver.turb_model, switched)
    solver.current_order = 0
    solver.ops = switch_mesh_order(solver.mesh, 0)
    # 壁面距离是纯几何量：切到 P0 之后由来源在 P0 解点上重新查询（不取单元平均）
    recompute_wall_distance_for_current_order(solver)
    reset_order_dependent_state(solver, solver.sgs_model)
