"""
AutoFlowCFD V2.0 - FRSolver 单时间步推进 (从 fr_solver.py 拆分)

从 fr_solver.py 拆出来（控制单文件行数，>400 行需拆分的项目规范）。
签名以 `solver: FRSolver` 为第一参数，FRSolver 上保留同名薄委托方法，
调用方式不变。
"""

import numpy as np

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.fr_solver.filter import build_filter_func


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

    湍流场更新用的时间步长（真实 bug，已修复）：此前不管哪种
    scheme，`compute_turbulence_source` 都直接拿 step() 收到的原始
    `dt` 参数去更新 k_field/omega_field（纯显式前向欧拉），完全没有
    经过 `solver._compute_local_time_step()` 算出来的、真正随阶数
    /粘性刚性/几何退化收紧过的逐 SP 局部步长 dt_local——CLI
    `solve steady` 固定传 dt=1e-3，Order Continuation 各阶数共用
    同一个值，从不随阶数收紧。真实复现（cube_demo 生产网格 + 合成
    Couette+SST 算例）：dt_local 的最小值比这个固定 dt 小 178~790 倍
    （合成算例实测，随阶数提升而恶化），k/omega 场显式积分因此在
    P0 就已经临界不稳定（omega 一步内被放大 23 倍），Order
    Continuation 提升到 P1 后（真正的 FR 梯度重构启用，湍流输运项
    量级增大）在几步内失控发散，与真实网格报告的"P0->P1 残差暴涨
    ~1000倍、P2 变成 NaN"精确吻合。现在湍流场更新与平均流一样使用
    dt_local（工业 RANS 求解器标准做法：湍流量与平均流共用局部时间
    步加速），但 DUAL_TIME 模式下例外，见下方。

    dt 参数的语义按 time_scheme 分两种情况：
    - SSP-RK2/RK3/IMEX（稳态收敛加速模式）：dt 参数确实被忽略，
      平均流步长与湍流场步长都改用局部 CFL 决定——用于收敛到定常解，
      不要求时间精度，局部时间步是标准且正确的加速手段。
    - DUAL_TIME（DES/LES 等真正非稳态仿真应使用的模式）：dt 是真正
      的物理时间步长，通过 BDF1/BDF2 时间导数项耦合进伪残差（见
      TimeIntegrator.step_dual_time），伪时间迭代收敛后得到的解在
      物理时间上精确前进了 dt；局部 CFL 步长只用作内层伪时间迭代的
      加速手段，不影响物理时间精度。湍流场更新在这个模式下仍然用
      物理 dt（不是 dt_local）——必须与平均流站在同一个物理时间基准
      上前进，换成伪时间步长会让两者时间不同步，物理时间精度失去
      意义。

    Args:
        solver: FRSolver 实例
        dt: 见上——SSP-RK/IMEX 模式下被忽略，DUAL_TIME 模式下是真正
            生效的物理时间步长

    Returns:
        residual_norm: 残差范数
    """
    from autoflowcfd.core.fr_solver.solver import logger  # 延迟导入避免循环依赖

    try:
        solver.state._update_primitives()

        # BD-02：合成湍流入口 (SEM) 涡核对流——每个物理步调用一次
        # advance()，不在每次残差求值/RK 子迭代里调用（见
        # boundary/fr_ghost_state.py::InletSEMGhostState 文档）。
        # solver._sem_instances 由 _build_boundary_ghost_provider 在
        # LES/DDES 模式下、存在 VELOCITY_INLET 组时填充，否则是空列表。
        for sem in getattr(solver, "_sem_instances", []):
            sem.advance(dt, mean_velocity=np.array([solver.freestream["vel_inf"], 0.0, 0.0]))

        n_cells, n_sps, n_vars = solver.state.U.shape
        dt_local = solver._compute_local_time_step()  # (n_cells, n_sps)

        # 湍流源项在当前状态下求值一次（沿用旧有的单步显式-半隐式
        # 阻尼更新，见 turbulence_sst.py::update_fields）。
        #
        # 关键修复（真实复现：cube_demo 生产网格 + 小合成 Couette+SST 算例
        # 均可复现）：此前这里传的是 step() 收到的原始物理 dt（CLI
        # `solve steady` 固定传 1e-3，且 Order Continuation 各阶数共用
        # 同一个值，从不随阶数/网格收紧），而不是刚算出来的、真正随阶数
        # /粘性刚性/几何退化收紧过的 dt_local（cfl.py::compute_local_
        # time_step，三种机制取最小值，专门为压制包括湍流交叉扩散在内的
        # 刚性子系统设计——见该文件文档第2条）。k/omega 场的更新
        # （SSTModelFR.update_fields）是纯显式前向欧拉
        # `k_field += dt*dk_total`，用一个未经稳定性检验、其量级由
        # dt_local 算出来恰好是 178~790 倍还是保守值（合成算例实测，
        # 真实网格上更极端）的固定步长积分，在合成 Couette+SST 算例上
        # 已实测复现：omega 场在 P0 第一步内就从初值 1.0 冲到 23.34（放大
        # 23 倍），P1 第一步冲到 1438，P2 数步内到 1e14~1e28 直至 inf——
        # 与真实网格报告的"P0 结束到 P1 开始残差暴涨约1000倍，P2 完全
        # 发散为 NaN"精确吻合。cfl.py 的阶数收紧/粘性稳定性限制/几何
        # 退化限制全部正确算出了 dt_local，只是从未被传给这条路径使用；
        # 平均流经 solver.time_integrator.step(..., dt_local_flat, ...)
        # 正确使用了它。现在湍流场显式更新也使用同一个逐 SP 局部时间
        # 步长（工业 RANS 求解器的标准做法：湍流量与平均流共用同一套
        # 局部时间步加速策略），而不是一个与它完全脱节的固定物理 dt。
        #
        # DUAL_TIME 例外：该模式下 dt 是真正生效的物理时间步长（BDF1/
        # BDF2 时间精度要求，见本函数顶部文档与下方 U 的推进分支），
        # 湍流场必须与平均流用同一个物理时间基准前进，不能像稳态加速
        # 模式那样换成逐 SP 伪时间步长——否则湍流场与平均流各自站在
        # 不同的"时间"上，物理时间精度失去意义。稳态收敛加速模式
        # （SSP-RK/IMEX）下 dt 参数定义上就应被忽略（见文档），
        # 用 dt_local 才是这里的一致行为。
        turb_dt = dt if solver.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME else dt_local
        turb_source = solver.compute_turbulence_source(turb_dt)

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
