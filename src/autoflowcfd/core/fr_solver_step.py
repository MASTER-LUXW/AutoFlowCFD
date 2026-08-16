"""
AutoFlowCFD V2.0 - FRSolver 单时间步推进 (从 fr_solver.py 拆分)

从 fr_solver.py 拆出来（控制单文件行数，>400 行需拆分的项目规范）。
签名以 `solver: FRSolver` 为第一参数，FRSolver 上保留同名薄委托方法，
调用方式不变。
"""

import numpy as np

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.fr_solver_filter import build_filter_func


def step(solver, dt: float) -> float:
    """
    执行一个时间步长 (S-05)。

    平均流（5个欧拉变量）真正通过 solver.time_integrator 推进
    （SSP-RK2/RK3/IMEX/Dual-Time，由构造时的 time_scheme 决定），
    取代旧版本里恒定不变的单级前向欧拉——此前不管 CLI 传
    --time-method rk3/imex/dual-time 哪一个，step() 内部都硬编码执行
    `U = U + dt_local*residual`，`solver.time_integrator` 被构造出来后
    从未被调用过。

    湍流量 (k,omega) 的输运方程仍用独立的单步显式更新（算子分裂：
    平均流走高阶 RK 子迭代，湍流方程走更简单、专门做过刚性限制
    的更新，是工业 RANS/DES 求解器常见做法，避免把湍流源项的强
    非线性刚性直接卷入平均流的多级残差重新求值）。

    dt 参数的语义按 time_scheme 分两种情况（此前不管哪种 scheme，
    dt 参数都被完全忽略，实际步长恒由 solver._compute_local_time_step()
    的逐单元局部 CFL 步长决定——这对稳态 RANS 收敛加速是对的，但
    意味着 CLI `solve transient`（DES/LES）传入的 `--dt`/`--physical-time`
    从未真正生效，瞬态仿真没有时间精度，这是本次修复的问题）：
    - SSP-RK2/RK3/IMEX（稳态收敛加速模式）：dt 参数确实被忽略，
      步长仍由局部 CFL 决定——用于收敛到定常解，不要求时间精度，
      局部时间步是标准且正确的加速手段。
    - DUAL_TIME（DES/LES 等真正非稳态仿真应使用的模式）：dt 现在
      是真正的物理时间步长，通过 BDF1/BDF2 时间导数项耦合进伪残差
      （见 TimeIntegrator.step_dual_time），伪时间迭代收敛后得到的
      解在物理时间上精确前进了 dt；局部 CFL 步长只用作内层伪时间
      迭代的加速手段，不影响物理时间精度。

    Args:
        solver: FRSolver 实例
        dt: 见上——SSP-RK/IMEX 模式下被忽略，DUAL_TIME 模式下是真正
            生效的物理时间步长

    Returns:
        residual_norm: 残差范数
    """
    from autoflowcfd.core.fr_solver import logger  # 延迟导入避免循环依赖

    try:
        solver.state._update_primitives()

        # BD-02：合成湍流入口 (SEM) 涡核对流——每个物理步调用一次
        # advance()，不在每次残差求值/RK 子迭代里调用（见
        # boundary/fr_ghost_state.py::InletSEMGhostState 文档）。
        # solver._sem_instances 由 _build_boundary_ghost_provider 在
        # LES/DDES 模式下、存在 VELOCITY_INLET 组时填充，否则是空列表。
        for sem in getattr(solver, "_sem_instances", []):
            sem.advance(dt, mean_velocity=np.array([solver.freestream["vel_inf"], 0.0, 0.0]))

        # 湍流源项在当前状态下求值一次（沿用旧有的单步显式-半隐式
        # 阻尼更新，见 turbulence_sst.py::update_fields）
        turb_source = solver.compute_turbulence_source(dt)

        n_cells, n_sps, n_vars = solver.state.U.shape
        dt_local = solver._compute_local_time_step()  # (n_cells, n_sps)

        U_flat = solver.state.U.reshape(n_cells * n_sps, n_vars)
        dt_local_flat = dt_local.reshape(n_cells * n_sps)

        def mean_flow_residual(U_flat_trial: np.ndarray) -> np.ndarray:
            """TimeIntegrator 约定：dU/dt = -residual_func(U)。"""
            U_trial = U_flat_trial.reshape(n_cells, n_sps, n_vars)
            saved_U = solver.state.U
            solver.state.U = U_trial
            try:
                inv_res = solver.compute_inviscid_residual()
                visc_res = solver.compute_viscous_residual()
            finally:
                solver.state.U = saved_U
            total = inv_res + visc_res  # 已是 dU/dt，形状 (n_cells,n_sps,n_vars)
            return -total.reshape(n_cells * n_sps, n_vars)

        def convective_residual_only(U_flat_trial: np.ndarray) -> np.ndarray:
            """IMEX 显式项：只含无粘对流残差，供 step_imex 使用。"""
            U_trial = U_flat_trial.reshape(n_cells, n_sps, n_vars)
            saved_U = solver.state.U
            solver.state.U = U_trial
            try:
                inv_res = solver.compute_inviscid_residual()
            finally:
                solver.state.U = saved_U
            return -inv_res.reshape(n_cells * n_sps, n_vars)

        def diffusive_residual_only(U_flat_trial: np.ndarray) -> np.ndarray:
            """IMEX 隐式项：只含粘性残差（含湍流涡粘耦合的扩散项），
            供 step_imex 的 Picard 子迭代反复重新求值。"""
            U_trial = U_flat_trial.reshape(n_cells, n_sps, n_vars)
            saved_U = solver.state.U
            solver.state.U = U_trial
            try:
                visc_res = solver.compute_viscous_residual()
            finally:
                solver.state.U = saved_U
            return -visc_res.reshape(n_cells * n_sps, n_vars)

        # residual0 是 TimeIntegrator 自身的 R(U) 约定（dU/dt=-R），
        # 复用它既避免重复计算 Stage 0 残差，也用来更新
        # solver.state.dU_dt——收敛监控 (get_residual_norm) 依赖这个量，
        # 重构 step() 时若遗漏这一步，会让残差历史恒为 0（表面上"已收敛"，
        # 实际只是从未被更新过），已用非均匀扰动初场验证发现并修复。
        residual0 = mean_flow_residual(U_flat)
        solver.state.dU_dt = (-residual0).reshape(n_cells, n_sps, n_vars)

        # 模态滤波回调（S-05 补充修复）：见 fr_solver_filter.py 文档——
        # 必须传给 TimeIntegrator，由它在*每个* RK stage 的正定性投影
        # 之后立即施加，抑制坍缩坐标节点配置法固有的混叠噪声放大；
        # 只在最终组合结果上滤波一次不够，真实复现噪声在中间 stage
        # 就已放大到 NaN。
        filter_func = build_filter_func(solver)

        if solver.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME:
            # 真正时间精度的物理时间推进：dt 是物理时间步长（不再被
            # 忽略），dt_local 只用作内层伪时间迭代的局部加速步长，
            # 两者不能混用——见 TimeIntegrator.step_dual_time 文档。
            U_new_flat = solver.time_integrator.step_dual_time(
                U_flat,
                mean_flow_residual,
                dt_local_flat,
                dt_physical=dt,
                solution_prev=solver._dual_time_U_prev,
                max_inner_iter=solver.time_integrator.dual_time_steps,
                filter_func=filter_func,
            )
            solver._dual_time_U_prev = U_flat.copy()
        elif solver.time_integrator.scheme == TimeIntegrationScheme.IMEX_EULER:
            # 显式处理无粘对流项、隐式处理粘性+湍流扩散项——通用的
            # step(...) 单一残差入口表达不了这个拆分（见该方法里的
            # 说明），必须直接调用 step_imex 并传入两个独立的残差
            # 闭包。
            U_new_flat = solver.time_integrator.step_imex(
                U_flat, convective_residual_only, diffusive_residual_only,
                dt_local_flat, p_floor=1.0,
            )
        else:
            U_new_flat = solver.time_integrator.step(
                U_flat, mean_flow_residual, dt_local_flat, p_floor=1.0, residual0=residual0,
                filter_func=filter_func,
            )
        solver.state.U = U_new_flat.reshape(n_cells, n_sps, n_vars)

        # 湍流量 (k,omega) 的更新已经在上面 compute_turbulence_source()
        # 内部通过 turb_model.update_fields() 完成（真正被
        # _get_turbulent_viscosity_field/nu_t 消费的是 turb_model.
        # k_field/omega_field，不是 state.U[:,:,5:7]）。此前这里还有
        # 一段用 dt_local（逐 SP 局部 CFL 步长）对 state.U[:,:,5:7]
        # 做的第二次更新——用的是同一份 Sk/S_omega，却是与
        # update_fields 内部用的 dt（全局步长）不同的 dt_local，且
        # state.U[:,:,5:7] 全仓库没有任何代码读取（已核实），是纯粹
        # 的死代码+双重更新，删除。
        solver.apply_turbulence_corrections()
        solver.state._update_primitives()

        residual_norm = solver.state.get_residual_norm()
        return residual_norm

    except Exception as e:
        logger.error(f"Step failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
