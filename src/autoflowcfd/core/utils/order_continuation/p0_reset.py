"""AutoFlowCFD V2.0 - 全新求解器在 Order Continuation 起步时重建到 P0

从 `src/autoflowcfd/core/utils/order_continuation.py`(原 926 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


import numpy as np






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

    p0_state = FRState(solver.state.n_cells, expected_p0_n_sps)
    # 速度方向必须取自 aoa/aos（2026-09-24 修复）：此前这里写死 (vel_inf, 0, 0)，
    # 而边界 Q_free 用的是正确方向，`--aoa` 非零时初场与边界不一致。
    # 8 处同类写法已统一到 `freestream_conservative_state`（见其文档）。
    # 这一处是**每个目标阶数 >= 1 的全新算例都必经**的（`uses_order_continuation`
    # 为真才进 Order Continuation）：FRSolver.__init__
    # 的初场本来是对的，随即被这次重建覆盖成零攻角。
    _v0 = float(solver.freestream["vel_inf"]) * direction_from_freestream(solver.freestream)
    p0_state.initialize_uniform(
        rho=solver.freestream["rho_inf"],
        u=float(_v0[0]), v=float(_v0[1]), w=float(_v0[2]),
        p=solver.freestream["p_inf"],
    )
    solver.state = p0_state

    if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "transported_fields"):
        # 重置为模型的来流值（构造时由 Tu/VR 推导，与 init_turbulence_models 同一组值）
        solver.turb_model.set_transported_fields(
            solver.turb_model.freestream_fields((solver.state.n_cells, expected_p0_n_sps), np))
        # nu_t 同样必须重置到 P0 维度——理由同 interpolate_to_new_order
        # 里的 nu_t 插值处理：它不会自动跟着输运场变形，
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

    solver.current_order = 0
    solver.ops = generate_fr_operators(0)
    solver.mesh.set_order(0)

    # 壁面距离是纯几何量：`set_order(0)` 之后 `sps_coords` 才是 P0 的解点，由来源重新查询
    # （不取单元平均——见 `recompute_wall_distance_for_current_order` 文档）。
    from autoflowcfd.core.fr_solver.turbulence import recompute_wall_distance_for_current_order
    recompute_wall_distance_for_current_order(solver)

    print(f"[INFO] Reinitialized to P0 ({expected_p0_n_sps} SP/cell)")
