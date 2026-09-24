"""AutoFlowCFD V2.0 - 全新求解器在 Order Continuation 起步时重建到 P0

从 `src/autoflowcfd/core/utils/order_continuation.py`(原 926 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


import numpy as np




from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence


def _reset_state_to_p0(solver, expected_p0_n_sps: int = 1) -> None:
    """把一个**全新**（非 resume）求解器的状态重建到 P0。

    Order Continuation 从 P0 起步；全新求解器构造时按目标阶数分配了
    状态，这里用均匀自由来流在 P0 上重建它，并把湍流场、DES 长度尺度、
    SGS 涡粘、壁面距离、算子、网格阶数一并切到 P0。

    从 `run_order_continuation` 抽出（2026-09-24，项目「单文件不超 500 行」
    规范）。抽取时顺带修掉一个真实缺陷：初场速度此前写死成
    `(vel_inf, 0, 0)`，攻角/侧滑角被丢掉，见下方同名注释。
    """
    from autoflowcfd.core.fr_solver.state import FRState
    from autoflowcfd.fr.operators import generate_fr_operators

    print(f"[INFO] Current state has {solver.state.U.shape[1]} SPs/cell, reinitializing from P0...")

    from autoflowcfd.core.utils.flow_direction import direction_from_freestream

    p0_state = FRState(solver.state.n_cells, expected_p0_n_sps, solver.state.n_vars)
    # 速度方向必须取自 aoa/aos（2026-09-24 修复）：此前这里写死 (vel_inf, 0, 0)，
    # 而边界 Q_free 用的是正确方向，`--aoa` 非零时初场与边界不一致。
    # 8 处同类写法已统一到 `freestream_conservative_state`（见其文档）。
    # 这一处是**每个目标阶数 >= 2 的全新算例都必经**的（FRSolver.solve
    # 里 `self.order >= 2` 才进 Order Continuation）：FRSolver.__init__
    # 的初场本来是对的，随即被这次重建覆盖成零攻角。
    _v0 = float(solver.freestream["vel_inf"]) * direction_from_freestream(solver.freestream)
    p0_state.initialize_uniform(
        rho=solver.freestream["rho_inf"],
        u=float(_v0[0]), v=float(_v0[1]), w=float(_v0[2]),
        p=solver.freestream["p_inf"],
    )
    solver.state = p0_state

    if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "k_field"):
        # 用 Tu/VR 推导的物理自洽值重置（与 init_turbulence_models 一致）
        k_inf, omega_inf = _set_freestream_turbulence(solver)
        solver.turb_model.k_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * k_inf
        solver.turb_model.omega_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * omega_inf
        # nu_t 同样必须重置到 P0 维度——理由同 interpolate_to_new_order
        # 里的 nu_t 插值处理：它不会自动跟着 k_field/omega_field 变形，
        # 只在 compute_source_terms 被调用时才按当时的 k/omega 重新
        # 算出，遗漏会让它保留重置前的形状，被 _compute_local_time_step
        # 在 compute_turbulence_source 刷新它之前读取时引发同一类形状
        # 不匹配问题。用 SSTModelFR.__init__ 同样的初值约定（零）。
        if hasattr(solver.turb_model, "nu_t"):
            solver.turb_model.nu_t = np.zeros((solver.state.n_cells, expected_p0_n_sps))
        print(f"[INFO] Turbulence fields reset to P0 dimensions")

    if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "des_length_scale"):
        # 同 interpolate_to_new_order 里的处理：清空而不是插值，理由见
        # 该函数文档。
        solver.turb_model.des_length_scale = None

    if getattr(solver, "sgs_model", None) is not None and hasattr(solver.sgs_model, "nu_t"):
        solver.sgs_model.nu_t = None

    if solver.wall_distance is not None:
        old_wall_dist = solver.wall_distance
        if old_wall_dist.ndim == 2 and old_wall_dist.shape[1] > 1:
            mean_wall_dist = np.mean(old_wall_dist, axis=1, keepdims=True)
            solver.wall_distance = np.tile(mean_wall_dist, (1, expected_p0_n_sps))
            print(f"[INFO] Wall distance field reset to P0 dimensions")

    solver.current_order = 0
    solver.ops = generate_fr_operators(0)
    solver.mesh.set_order(0)

    # 真实 bug 修复（2026-09-06）：上面第 434-439 行的 `np.mean` 压缩
    # 只是权宜的形状占位，不是壁面距离在 P0 下的正确值——见
    # `recompute_wall_distance_for_current_order` 文档完整推导。
    # `set_order(0)` 之后 `solver.mesh.sps_coords` 才反映 P0 真实的
    # 单 SP 坐标，这里重新查询覆盖掉那个被压缩、失真的值；如果没有
    # 缓存（没调用过 `compute_wall_distance_field`，或本来就没有
    # wall_distance），保留上面的均值压缩结果作为退化但形状正确的
    # 后备。
    from autoflowcfd.core.fr_solver.turbulence import recompute_wall_distance_for_current_order
    recompute_wall_distance_for_current_order(solver)

    print(f"[INFO] Reinitialized to P0 ({expected_p0_n_sps} SP/cell)")
