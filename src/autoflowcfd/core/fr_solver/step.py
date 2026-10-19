"""
AutoFlowCFD V2.0 - FRSolver 单时间步推进 (从 fr_solver.py 拆分)

从 fr_solver.py 拆出来（控制单文件行数，>400 行需拆分的项目规范）。
签名以 `solver: FRSolver` 为第一参数，FRSolver 上保留同名薄委托方法，
调用方式不变。
"""

from functools import partial

import numpy as np

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.turbulence.registry import has_transport_equations
from autoflowcfd.core.fr_solver.turbulence.implicit import (
    CpuCoupledBackend,
    single_machine_cell_colors,
    single_machine_coupling_graph,
)


def step(solver, dt: float) -> float:
    """
    执行一个时间步长 (S-05)。

    平均流（5个欧拉变量）真正通过 solver.time_integrator 推进
    （SSP-RK2/RK3/Dual-Time/Newton-Krylov，由构造时的 time_scheme 决定），
    取代旧版本里恒定不变的单级前向欧拉——此前不管 CLI 传
    哪一种时间方法，step() 内部都硬编码执行
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

    dt 参数的语义按 time_scheme 分三种情况：
    - SSP-RK2/RK3（稳态收敛加速模式）：dt 参数被忽略，
      平均流步长与湍流场步长都改用局部 CFL 决定——用于收敛到定常解，
      不要求时间精度，局部时间步是标准且正确的加速手段。
    - SSP-RK2/RK3 + 时间精确（`solver.time_accurate`，瞬态的显式档）：求解循环经
      `time_integration/time_accurate.py::advance` 把物理步切成子步，本函数每次推进一个子步，
      局部步长函数返回全场同一步长 h（`uniform_time_step`），dt 是该物理步的剩余时间。
    - DUAL_TIME（DES/LES 等真正非稳态仿真应使用的模式）：dt 是真正
      的物理时间步长（BDF1/BDF2）；局部 CFL 步长只是内迭代里 Newton 步被拒时的回退手段，
      不影响物理时间精度。每个物理步的 BDF 方程用隐式内迭代求解
      （`time_integration/implicit/newton/physical_time.py`：物理时间项并入稳态的
      Newton-Krylov 步，带输运方程的湍流模型与平均流紧耦合、同一个物理时间）。

    Args:
        solver: FRSolver 实例
        dt: 见上——稳态 SSP-RK 下被忽略；时间精确子步里是物理步的剩余时间；DUAL_TIME 下是物理时间步长

    Returns:
        residual_norm: 残差范数
    """
    from loguru import logger

    try:
        solver.state._update_primitives()

        n_cells, n_sps, n_vars = solver.state.U.shape
        # 低马赫数伪时间预处理（2026-09-14 新增）：`dt_local` 是**平均流**用
        # 的步长（启用预处理时按预处理波速 (|un|+c_precond) 放大，M~0.1 下
        # 约 5 倍）；`dt_physical` 是按物理波速算出的那份，留给湍流标量
        # 更新用——完整理由见 cfl.py::compute_local_time_step 文档
        # "低马赫数伪时间预处理"一节（湍流输运的显式更新刻意没有 point-
        # implicit 阻尼，不跟着放大步长）。未启用预处理时两者是同一个数组。
        dt_local, dt_physical = solver._compute_local_time_step(return_physical_too=True)

        # BD-02：合成湍流入口 (SEM) 涡核对流——每个物理步一次，不在每次残差求值/RK 子迭代里
        # （见 boundary/fr_ghost_state.py::InletSEMGhostState 文档）；全部后端共用。时间精确子步按实际步长推进
        from autoflowcfd.boundary.synthetic_inlet import advance_synthetic_inlets
        from autoflowcfd.core.time_integration.time_accurate import sem_time_step
        advance_synthetic_inlets(solver.boundary_ghost_provider, solver.freestream, sem_time_step(solver, dt))

        # 累计伪时间（2026-09-17）：局部时间步进下不存在单一的"当前时间"，
        # 所以按单元累加。这个量此前算得出来但**从来没被报告过**，导致日志
        # 里看不出"残差在降但物理场只走了 1.8% 个特征时间"——那次误判的
        # 完整记录见 `pseudotime_budget.py` 模块文档。累加 O(n_cells) 一次
        # 加法，相对一步残差求值可忽略。
        from autoflowcfd.core.fr_solver.pseudotime_budget import accumulate_pseudo_time

        if solver.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME:
            # 双时间步：每个单元都前进同一个物理时间步（伪时间步只是内迭代的回退手段，不是"走过的时间"）
            accumulate_pseudo_time(solver, np.full(n_cells, float(dt)))
        else:
            accumulate_pseudo_time(solver, dt_local.reshape(n_cells, n_sps)[:, 0] if dt_local.ndim > 1
                                   else dt_local[::max(n_sps, 1)])

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
        # 不同的“时间”上，物理时间精度失去意义。稳态收敛加速模式
        # （SSP-RK）下 dt 参数定义上就应被忽略（见文档），
        # 用 dt_local 才是这里的一致行为。
        # 隐式稳态 + k-omega：湍流与平均流在下面的 NEWTON_KRYLOV 分支里紧耦合求解
        # （time_integration/implicit/newton/coupled_step.py 模块文档：分离式在大 CFL 下的块
        # Gauss-Seidel 外迭代会振荡），这里不再单独推进湍流
        scheme = solver.time_integrator.scheme
        dual_time = scheme == TimeIntegrationScheme.DUAL_TIME
        # 隐式步（稳态 Newton-Krylov，或双时间步的隐式内迭代）里带输运方程的湍流模型与平均流紧耦合求解；
        # 其余情形（显式格式、代数/亚格子模型）湍流按算子分裂先更新
        coupled_nk = (scheme in (TimeIntegrationScheme.NEWTON_KRYLOV, TimeIntegrationScheme.DUAL_TIME)
                      and solver.turb_model is not None
                      and has_transport_equations(solver.turb_model_name))
        if not coupled_nk:
            solver.compute_turbulence_source(dt_physical)

        # 本步冻结的湍流涡粘（算子分裂：湍流已更新，本步全部残差求值共用
        # 同一份，与 GPU/分布式后端同一约定，见 compute_viscous_residual 文档）
        from autoflowcfd.core.fr_solver.turbulence.corrections import get_turbulent_viscosity_field

        mu_t_step = get_turbulent_viscosity_field(solver)
        # 本步冻结的人工扩散系数（同一个算子分裂约定；未启用时为 None）
        nu_av_step = solver.compute_artificial_diffusivity_field()

        U_flat = solver.state.U.reshape(n_cells * n_sps, n_vars)
        dt_local_flat = dt_local.reshape(n_cells * n_sps)

        def mean_flow_residual_raw(U_flat_trial: np.ndarray) -> np.ndarray:
            """未经预处理的原始残差 R（TimeIntegrator 约定 dU/dt = -R）。

            残差监控（`state.dU_dt` -> `residual_norm.py::monitor_residuals`）与自适应 CFL
            都必须用这一份**物理**残差，不能用预处理后的：Gamma 可逆、
            两者同时趋零，但量级不同，用预处理值会让打印出来的残差、
            收敛判据以及与历史算例的对比全部失去可比性。
            """
            U_trial = U_flat_trial.reshape(n_cells, n_sps, n_vars)
            saved_U = solver.state.U
            solver.state.U = U_trial
            try:
                return solver.compute_mean_flow_residual(mu_t_turb=mu_t_step, nu_av=nu_av_step)
            finally:
                solver.state.U = saved_U

        def mean_flow_residual(U_flat_trial: np.ndarray) -> np.ndarray:
            """供 TimeIntegrator 推进用的残差：启用低马赫数预处理时返回
            `Gamma R`，否则就是原始 R。

            `Gamma` 线性，直接作用在 R 上与作用在 dU/dtau 上等价。它
            **必须**与 cfl.py 里按预处理波速取的 dt 成对出现，缺一个就是
            2026-08-24 那次失稳（完整推导/正确性论证见
            core/utils/preconditioning.py 模块末尾"伪时间预处理矩阵 Gamma"）。
            `solver.state.Q` 此刻正是 U_trial 对应的原始变量——
            `compute_mean_flow_residual` 入口的 `_update_primitives()` 是用
            U_trial 算的（闭包里刚把 state.U 换成 U_trial），不是基态；
            这一点是这里能直接用它的前提。
            """
            res = mean_flow_residual_raw(U_flat_trial)
            if solver.low_mach_precond_enabled:
                from autoflowcfd.core.utils.preconditioning import (
                    apply_low_mach_preconditioner,
                )
                res = apply_low_mach_preconditioner(
                    res, solver.state.Q, solver.freestream["mach_ref"], out=res,
                )
            return res.reshape(n_cells * n_sps, n_vars)

        # residual0 是 TimeIntegrator 自身的 R(U) 约定（dU/dt=-R），
        # 复用它既避免重复计算 Stage 0 残差，也用来更新
        # solver.state.dU_dt——收敛监控（monitor_residuals）依赖这个量，
        # 重构 step() 时若遗漏这一步，会让残差历史恒为 0（表面上"已收敛"，
        # 实际只是从未被更新过），已用非均匀扰动初场验证发现并修复。
        # `residual0` 复用给积分器省掉一次 Stage 0 残差求值；但监控用的
        # `state.dU_dt` 必须是**未预处理**的物理残差（见
        # `mean_flow_residual_raw` 文档）。下面先算原始残差、取负存进
        # dU_dt（`-res` 本身产生独立副本），再就地把同一块内存预处理成
        # 积分器要的 `Gamma R`——不额外分配 1.2GiB（P2 规模）的数组。
        residual0_raw = mean_flow_residual_raw(U_flat)
        solver.state.dU_dt = (-residual0_raw).reshape(n_cells, n_sps, n_vars)
        residual0 = None
        if scheme not in (TimeIntegrationScheme.DUAL_TIME, TimeIntegrationScheme.NEWTON_KRYLOV):
            # 只有显式积分器复用它；隐式步自己在基态上求残差，这里留着它只会在整个 Newton 步里多占一个
            # 全尺寸数组（cube_demo 粗网格 P2 为 0.29 GiB，正落在线性求解的内存峰值上）
            if solver.low_mach_precond_enabled:
                from autoflowcfd.core.utils.preconditioning import (
                    apply_low_mach_preconditioner,
                )
                residual0_raw = apply_low_mach_preconditioner(
                    residual0_raw, solver.state.Q, solver.freestream["mach_ref"],
                    out=residual0_raw,
                )
            residual0 = residual0_raw.reshape(n_cells * n_sps, n_vars)
        del residual0_raw

        # 守恒的正性保持限制器（Zhang–Shu 型，取代逐点硬钳；缘由见
        # time_integration/positivity/__init__.py）。按阶数缓存。
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter
        positivity_func = get_positivity_limiter(solver)

        if scheme in (TimeIntegrationScheme.DUAL_TIME, TimeIntegrationScheme.NEWTON_KRYLOV):
            # 隐式步（矩阵自由 Newton-Krylov + 伪瞬态延拓），与其它后端共用同一个实现：带输运方程的湍流
            # 模型时平均流与湍流紧耦合（`implicit/newton/coupled_step.py`），否则只解平均流 5 个变量
            # （`implicit/newton/mean_flow_step.py`）。稳态一步一个 Newton 步；双时间步每个
            # 物理步做若干个带 BDF 物理时间项的 Newton 步（`implicit/newton/physical_time.py`）。
            from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
            from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import (
                step_mean_flow_newton,
            )
            from autoflowcfd.core.fr_residual.jacobian.backend import (
                MeanFlowBlockAssembler, unsupported_reason,
            )

            def newton_step(dtau_flat, physical_time=None, advance_ramp=True):
                """一个 Newton 步（`dtau_flat` 是伪时间步长天花板，CFL 律见 adaptive_cfl/ser.py），
                新状态写回求解器，返回诊断 info。"""
                if coupled_nk:
                    from autoflowcfd.core.time_integration.implicit.newton.coupled_step import step_coupled_newton

                    return step_coupled_newton(solver, CpuCoupledBackend(solver, nu_av_step), dtau_flat,
                                               physical_time=physical_time, advance_ramp=advance_ramp)
                u_now = solver.state.U.reshape(n_cells * n_sps, n_vars)
                u_next, info = step_mean_flow_newton(
                    solver, mean_flow_residual, u_now, dtau_flat,
                    _reference_scales(solver.freestream, n_vars),
                    red=LocalReductions(np),
                    cell_is_prism=np.arange(n_cells) < int(solver.mesh.n_prism_cells),
                    cell_colors=lambda: single_machine_cell_colors(solver),
                    coupling_graph=partial(single_machine_coupling_graph, solver),
                    order=_current_order(solver), positivity=positivity_func, physical_time=physical_time,
                    block_assembler=None if unsupported_reason(
                        order=_current_order(solver)) else MeanFlowBlockAssembler(
                        mesh=solver.mesh, ops=solver.ops, ghost_provider=solver.boundary_ghost_provider,
                        mu=solver.mu_molecular, mach_ref=solver.freestream["mach_ref"],
                        low_mach=solver.low_mach_precond_enabled,
                        mu_t=solver._get_turbulent_viscosity_field(mu_t_step), nu_av=nu_av_step, n_sps=n_sps,
                        wall_model=solver.wall_model))
                solver.state.U = u_next.reshape(n_cells, n_sps, n_vars)
                solver.state._update_primitives()
                return info

            if dual_time:
                from autoflowcfd.core.time_integration.implicit.newton.physical_time import (
                    advance_physical_step, previous_newton_state, store_newton_state,
                )

                x_now = (CpuCoupledBackend(solver, nu_av_step).state() if coupled_nk else U_flat.copy())
                advance_physical_step(
                    solver, x_now, previous_newton_state(solver, np, n_cells, n_sps),
                    lambda term, first: newton_step(
                        solver._compute_local_time_step().reshape(n_cells * n_sps), term, first), dt)
                store_newton_state(solver, x_now, n_cells, n_sps, n_vars)
                _nk_info = solver._newton_last_info
            else:
                _nk_info = newton_step(dt_local_flat)
            U_new_flat = solver.state.U.reshape(n_cells * n_sps, n_vars)
        else:
            U_new_flat = solver.time_integrator.step(
                U_flat, mean_flow_residual, dt_local_flat, residual0=residual0, positivity_func=positivity_func,
            )
        solver.state.U = U_new_flat.reshape(n_cells, n_sps, n_vars)

        # 湍流输运场在上面 compute_turbulence_source() 里由模型对象自己更新（状态数组只含平均流）
        solver.apply_turbulence_corrections()
        solver.state._update_primitives()

        from autoflowcfd.core.time_integration.residual_norm import explicit_turbulence_residual, monitor_residuals
        from autoflowcfd.fr.native_padding import order_from_n_sps

        # 逐方程 RMS 留给求解循环记进收敛历史（`convergence_history.py`，每个物理步一条）
        red = LocalReductions(np)
        is_prism = np.arange(n_cells) < int(solver.mesh.n_prism_cells)
        residual_norm, solver._residual_per_equation = monitor_residuals(
            red, solver.state.dU_dt, is_prism, order_from_n_sps(n_sps))
        solver._residual_turbulence = explicit_turbulence_residual(
            red, solver.turb_model, is_prism, order_from_n_sps(n_sps))

        # 自适应 CFL 更新（2026-08-24）：根据本步残差调节下一步的 CFL 数。
        # 放在 step() 末尾（不是 solve() 循环里），这样 solve() 和
        # order_continuation 两条路径都自动受益。
        _cfl_ctrl = getattr(solver, '_cfl_controller', None)
        if dual_time:
            pass          # 伪时间 CFL 在每个物理步的内迭代里更新（implicit/newton/physical_time.py）
        elif _cfl_ctrl is not None:
            if solver.time_integrator.scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
                # SER 看 Newton 实际在解的系统的残差 ||Gamma R||（步前、与上面
                # 物理残差同一时刻），并区分"残差上升但步被完整接受"（物理暂态，
                # 保持）与"步没被完整接受"（收缩），见 adaptive_cfl/ser.py
                from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import newton_step_ok

                _cfl_ctrl.update(_nk_info["res_norm"], step_ok=newton_step_ok(_nk_info))
            else:
                _cfl_ctrl.update(residual_norm)

        return residual_norm

    except Exception as e:
        logger.error(f"Step failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise



def _current_order(solver) -> int:
    order = getattr(solver, "current_order", None)
    return int(order if order is not None else solver.order)

