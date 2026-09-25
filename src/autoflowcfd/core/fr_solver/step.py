"""
AutoFlowCFD V2.0 - FRSolver 单时间步推进 (从 fr_solver.py 拆分)

从 fr_solver.py 拆出来（控制单文件行数，>400 行需拆分的项目规范）。
签名以 `solver: FRSolver` 为第一参数，FRSolver 上保留同名薄委托方法，
调用方式不变。
"""

import os

import numpy as np

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.fr_solver.filter import build_filter_func
from autoflowcfd.core.fr_solver.turbulence.implicit import (
    IMPLICIT_TURBULENCE_MODELS,
    step_turbulence_newton,
)


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
    from loguru import logger

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
        # 低马赫数伪时间预处理（2026-09-14 新增）：`dt_local` 是**平均流**用
        # 的步长（启用预处理时按预处理波速 (|un|+c_precond) 放大，M~0.1 下
        # 约 5 倍）；`dt_physical` 是按物理波速算出的那份，留给湍流标量
        # 更新用——完整理由见 cfl.py::compute_local_time_step 文档
        # "低马赫数伪时间预处理"一节（湍流输运的显式更新刻意没有 point-
        # implicit 阻尼，不跟着放大步长）。未启用预处理时两者是同一个数组。
        dt_local, dt_physical = solver._compute_local_time_step(return_physical_too=True)

        # 累计伪时间（2026-09-17）：局部时间步进下不存在单一的"当前时间"，
        # 所以按单元累加。这个量此前算得出来但**从来没被报告过**，导致日志
        # 里看不出"残差在降但物理场只走了 1.8% 个特征时间"——那次误判的
        # 完整记录见 `pseudotime_budget.py` 模块文档。累加 O(n_cells) 一次
        # 加法，相对一步残差求值可忽略。
        _tau = getattr(solver, "tau_accum", None)
        _dt_cell = dt_local.reshape(n_cells, n_sps)[:, 0] if dt_local.ndim > 1             else dt_local[::max(n_sps, 1)]
        if _tau is None or np.shape(_tau) != np.shape(_dt_cell):
            solver.tau_accum = np.array(_dt_cell, dtype=float)
        else:
            solver.tau_accum = _tau + _dt_cell
        # **本步**的逐单元 dt 也要单独留一份：伪时间预算里"到 tau/T = 1
        # 还需要多少步"是 `T / dt_median`，用累计量代替 dt 会把这个数按
        # 步数成比例低估（第一版真犯过：12 步后报"需要 105 步"，真实是
        # 1250 步）。
        solver.dt_cell_last = np.array(_dt_cell, dtype=float)

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
        # （SSP-RK/IMEX）下 dt 参数定义上就应被忽略（见文档），
        # 用 dt_local 才是这里的一致行为。
        turb_dt = (dt if solver.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME
                   else dt_physical)
        if (solver.time_integrator.scheme == TimeIntegrationScheme.NEWTON_KRYLOV
                and solver.turb_model is not None
                and solver.turb_model_name in IMPLICIT_TURBULENCE_MODELS):
            # 隐式稳态：k-omega 也走分离式 PTC-Newton（平均流冻结），显式
            # 输运更新在隐式 CFL 下必然失稳，见 turbulence/implicit.py 文档
            step_turbulence_newton(solver, turb_dt)
        else:
            solver.compute_turbulence_source(turb_dt)

        U_flat = solver.state.U.reshape(n_cells * n_sps, n_vars)
        dt_local_flat = dt_local.reshape(n_cells * n_sps)

        def mean_flow_residual_raw(U_flat_trial: np.ndarray) -> np.ndarray:
            """未经预处理的原始残差 R（TimeIntegrator 约定 dU/dt = -R）。

            残差监控（`state.dU_dt` -> `get_residual_norm`）与自适应 CFL
            都必须用这一份**物理**残差，不能用预处理后的：Gamma 可逆、
            两者同时趋零，但量级不同，用预处理值会让打印出来的残差、
            收敛判据以及与历史算例的对比全部失去可比性。
            """
            U_trial = U_flat_trial.reshape(n_cells, n_sps, n_vars)
            saved_U = solver.state.U
            solver.state.U = U_trial
            try:
                inv_res = solver.compute_inviscid_residual()
                visc_res = solver.compute_viscous_residual()
            finally:
                solver.state.U = saved_U
            # B-12 P2 OOM 修复第⑤级（2026-08-26）：原 `total = inv_res + visc_res`
            # 再 `-total` 会先后多分配两个 (n_cells,n_sps,n_vars)≈1.2GB 的全场数组，
            # 且峰值时三者共存；visc_res 是刚算出的新数组，对它原地加与原地取负严格等价，
            # 峰值降 2.4GB。见 time_integration/base.py 的 del L0/L1 同类注释。
            visc_res += inv_res
            visc_res *= -1  # dU/dt → R(U)（TimeIntegrator 约定 dU/dt=-R）
            return visc_res

        def mean_flow_residual(U_flat_trial: np.ndarray) -> np.ndarray:
            """供 TimeIntegrator 推进用的残差：启用低马赫数预处理时返回
            `Gamma R`，否则就是原始 R。

            `Gamma` 线性，直接作用在 R 上与作用在 dU/dtau 上等价。它
            **必须**与 cfl.py 里按预处理波速取的 dt 成对出现，缺一个就是
            2026-08-24 那次失稳（完整推导/正确性论证见
            core/utils/preconditioning.py 模块末尾"伪时间预处理矩阵 Gamma"）。
            `solver.state.Q` 此刻正是 U_trial 对应的原始变量——
            `compute_inviscid_residual` 入口的 `_update_primitives()` 是用
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

        def convective_residual_only(U_flat_trial: np.ndarray) -> np.ndarray:
            """IMEX 显式项：只含无粘对流残差，供 step_imex 使用。"""
            U_trial = U_flat_trial.reshape(n_cells, n_sps, n_vars)
            saved_U = solver.state.U
            solver.state.U = U_trial
            try:
                inv_res = solver.compute_inviscid_residual()
            finally:
                solver.state.U = saved_U
            inv_res *= -1  # 原地取负，理由同 mean_flow_residual 的 B-12 注释
            return inv_res.reshape(n_cells * n_sps, n_vars)

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
            visc_res *= -1  # 原地取负，理由同 mean_flow_residual 的 B-12 注释
            return visc_res.reshape(n_cells * n_sps, n_vars)

        # residual0 是 TimeIntegrator 自身的 R(U) 约定（dU/dt=-R），
        # 复用它既避免重复计算 Stage 0 残差，也用来更新
        # solver.state.dU_dt——收敛监控 (get_residual_norm) 依赖这个量，
        # 重构 step() 时若遗漏这一步，会让残差历史恒为 0（表面上"已收敛"，
        # 实际只是从未被更新过），已用非均匀扰动初场验证发现并修复。
        # `residual0` 复用给积分器省掉一次 Stage 0 残差求值；但监控用的
        # `state.dU_dt` 必须是**未预处理**的物理残差（见
        # `mean_flow_residual_raw` 文档）。下面先算原始残差、取负存进
        # dU_dt（`-res` 本身产生独立副本），再就地把同一块内存预处理成
        # 积分器要的 `Gamma R`——不额外分配 1.2GiB（P2 规模）的数组。
        residual0_raw = mean_flow_residual_raw(U_flat)
        solver.state.dU_dt = (-residual0_raw).reshape(n_cells, n_sps, n_vars)
        if solver.low_mach_precond_enabled:
            from autoflowcfd.core.utils.preconditioning import (
                apply_low_mach_preconditioner,
            )
            residual0_raw = apply_low_mach_preconditioner(
                residual0_raw, solver.state.Q, solver.freestream["mach_ref"],
                out=residual0_raw,
            )
        residual0 = residual0_raw.reshape(n_cells * n_sps, n_vars)

        # 模态滤波回调（S-05 补充修复）：见 fr_solver_filter.py 文档——
        # 必须传给 TimeIntegrator，由它在*每个* RK stage 的正定性投影
        # 之后立即施加，抑制坍缩坐标节点配置法固有的混叠噪声放大；
        # 只在最终组合结果上滤波一次不够，真实复现噪声在中间 stage
        # 就已放大到 NaN。
        # `AFCFD_FILTER_MODE=sensor`：按 Persson-Peraire 传感器逐单元门控
        # （见 filter.py::build_sensor_gated_filter_func）。默认 legacy
        # 行为不变，这一档供受控 A/B 与"光滑区不损失阶数"的正式方案用。
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        if resolve_filter_mode("cpu-single") == "sensor":
            from autoflowcfd.core.fr_solver.filter import (
                build_sensor_gated_filter_func,
            )
            filter_func = build_sensor_gated_filter_func(solver)
        else:
            filter_func = build_filter_func(solver)

        # 守恒的正性保持限制器（Zhang–Shu 型，取代逐点硬钳；缘由见
        # time_integration/positivity/__init__.py）。按阶数缓存。
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter
        positivity_func = get_positivity_limiter(solver)

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
                positivity_func=positivity_func,
            )
            solver._dual_time_U_prev = U_flat.copy()
        elif solver.time_integrator.scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
            # 隐式稳态步（矩阵自由 Newton-Krylov + 伪瞬态延拓）。
            #
            # `dt_local_flat` 同时充当 PTC 的 `dtau` 与对角预处理 ——
            # 于是 `--cfl-start/--cfl-max` 那套自适应控制器原样生效：
            # CFL 小 -> 对角项主导、接近显式、鲁棒；CFL 大 -> 接近纯
            # Newton、快。参考量级用残差诊断那一套唯一来源，不另定义。
            #
            # **一次 step() 做一个 Newton 步**：外层 `solver.solve()` 已经
            # 在做残差监控、自适应 CFL、checkpoint、Order Continuation，
            # Newton 外迭代放在它里面让这些机制原样生效（见
            # `implicit/jfnk.py` 模块文档）。
            #
            # `filter_func` 在这条路径上**不施加**：模态滤波是显式 RK
            # 逐 stage 的正定性/去噪手段，而 Newton 步里"解"是线性系统
            # 的解、不存在 stage 的概念；在 Newton 步之后滤一次会改变
            # 被求解的那个不动点方程本身（`R(U)=0` 变成
            # `F(U)=0` 的另一个问题），让残差与收敛判据失去意义。默认档
            # `FILTER_MODE=off` 本来就不构造 filter_func；显式指定了非
            # off 档时下面会明确报错而不是静默忽略。
            from autoflowcfd.core.fr_solver.residual_diagnostics import (
                _reference_scales,
            )
            from autoflowcfd.core.time_integration.implicit import (
                EisenstatWalkerForcing, step_newton_krylov,
            )

            if filter_func is not None:
                raise ValueError(
                    "NEWTON_KRYLOV（隐式稳态）与模态滤波不能同时启用："
                    "滤波会改变被求解的不动点方程本身（R(U)=0 变成另一个"
                    "问题），使残差与收敛判据失去意义。请用 "
                    "AFCFD_FILTER_MODE=off（默认值），或改用显式格式。"
                )
            if solver._newton_forcing is None:
                solver._newton_forcing = EisenstatWalkerForcing()
            # Newton 的未知量只有平均流 5 个守恒变量。湍流模型开启时状态向量
            # 带两个 k/omega 槽位（`state.U[:,:,5:7]`），它们全仓库无人读取、
            # 残差恒为零（2026-09-25 实测 inv/visc 残差这两列 max = 0.0），
            # k/omega 真正的值在 `turb_model` 上、由上面的隐式湍流步更新。
            # 带着它们解会让 Krylov 向量、块 Jacobi 的块尺寸与装配次数都
            # 白白多出 7/5 倍（plate_demo P1+SST：196 次 vs 140 次残差求值）。
            n_mf = min(n_vars, 5)
            nk_residual = _MeanFlowSlice(mean_flow_residual, U_flat, n_mf)
            if solver._newton_block_precond is None:
                solver._newton_block_precond = _build_block_precond_cache(solver, n_sps, n_mf)
            # `_newton_dtau_scale` 把 PTC 的 dtau 缩放状态跨步带下去：
            # 一步不被接受时 `step_newton_krylov` 会当场缩小 dtau 重试，
            # 用不完的档数由下一步继续（见 `implicit/dtau_control.py`
            # 里那段"固定 CFL 下永久停滞"的真实运行记录）。
            U_mf_new, _nk_info = step_newton_krylov(
                nk_residual, U_flat[:, :n_mf], dt_local_flat,
                _reference_scales(solver.freestream, n_vars)[:n_mf],
                forcing=solver._newton_forcing,
                dtau_scale=solver._newton_dtau_scale,
                block_precond=solver._newton_block_precond,
            )
            U_new_flat = U_flat.copy()
            U_new_flat[:, :n_mf] = U_mf_new
            solver._newton_last_info = _nk_info
            solver._newton_dtau_scale = _nk_info["dtau_scale"]
            if _nk_info["theta"] <= 0.0:
                logger.warning(
                    "Newton 步未能前进（theta=0, gmres_info=%s, "
                    "gmres_iters=%d, dtau_scale=%.3e, 本步已缩 %d 档）"
                    "——dtau 缩到下限仍拿不到被接受的步，那不再是步长"
                    "问题（dtau->0 即显式前向 Euler、必然被接受），"
                    "检查残差求值在当前状态上是否已经非物理"
                    % (_nk_info["gmres_info"], _nk_info["gmres_iters"],
                       _nk_info["dtau_scale"], _nk_info["n_dtau_cuts"]))
            elif _nk_info["n_dtau_cuts"] > 0:
                logger.info(
                    "Newton 步缩 %d 档 dtau 后被接受"
                    "（dtau_scale=%.3e, theta=%.3f）"
                    % (_nk_info["n_dtau_cuts"], _nk_info["dtau_scale"],
                       _nk_info["theta"]))
        elif solver.time_integrator.scheme == TimeIntegrationScheme.IMEX_EULER:
            # 显式处理无粘对流项、隐式处理粘性+湍流扩散项——通用的
            # step(...) 单一残差入口表达不了这个拆分（见该方法里的
            # 说明），必须直接调用 step_imex 并传入两个独立的残差
            # 闭包。
            U_new_flat = solver.time_integrator.step_imex(
                U_flat, convective_residual_only, diffusive_residual_only,
                dt_local_flat, positivity_func=positivity_func,
            )
        else:
            U_new_flat = solver.time_integrator.step(
                U_flat, mean_flow_residual, dt_local_flat, residual0=residual0,
                filter_func=filter_func, positivity_func=positivity_func,
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

        # 自适应 CFL 更新（2026-08-24）：根据本步残差调节下一步的 CFL 数。
        # 放在 step() 末尾（不是 solve() 循环里），这样 solve() 和
        # order_continuation 两条路径都自动受益。
        _cfl_ctrl = getattr(solver, '_cfl_controller', None)
        if _cfl_ctrl is not None:
            if solver.time_integrator.scheme == TimeIntegrationScheme.NEWTON_KRYLOV:
                # SER 看 Newton 实际在解的系统的残差 ||Gamma R||（步前、与上面
                # 物理残差同一时刻），并区分"残差上升但步被完整接受"（物理暂态，
                # 保持）与"步没被完整接受"（收缩），见 adaptive_cfl/ser.py
                _cfl_ctrl.update(
                    _nk_info["res_norm"],
                    step_ok=(_nk_info["theta"] >= 1.0 and _nk_info["n_dtau_cuts"] == 0))
            else:
                _cfl_ctrl.update(residual_norm)

        return residual_norm

    except Exception as e:
        logger.error(f"Step failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def _build_block_precond_cache(solver, n_sps: int, n_vars: int):
    """按当前阶数构造单元块 Jacobi 缓存（`implicit/block_jacobi.py`）。

    单机 CPU 的单元排列是"棱柱在前、四面体在后"（`mesh.n_prism_cells`），
    着色用的面相邻关系取自残差本身用的同一份展平面几何（带缓存）。
    """
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.time_integration.implicit import BlockJacobiCache
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    order = getattr(solver, "current_order", None)
    if order is None:
        order = solver.order
    n_cells = solver.state.U.shape[0]
    cell_is_prism = np.arange(n_cells) < int(solver.mesh.n_prism_cells)
    n_real_prism, n_real_tet = real_sps_per_cell(int(order))
    ffg = get_flat_face_geometry(solver.mesh, solver.ops)
    return BlockJacobiCache(
        owner_cell=ffg.owner_cell, neighbor_cell=ffg.neighbor_cell,
        cell_is_prism=cell_is_prism, n_sps=n_sps,
        n_real_prism=n_real_prism, n_real_tet=n_real_tet, n_var=n_vars)


class _MeanFlowSlice:
    """把 `(N, n_vars)` 的残差函数限制到前 `n_mf` 个（平均流）变量上。

    其余列（湍流槽位）固定在步前的值；做成类而不是闭包，理由同
    `implicit/jacobian_vector.py::MatrixFreeJacobian`（在整个 Krylov 求解
    期间存活，只持有需要的字段）。
    """

    __slots__ = ("_residual", "_full", "_n")

    def __init__(self, residual, u_full: np.ndarray, n_mf: int):
        self._residual = residual
        self._full = np.array(u_full, dtype=np.float64, copy=True)
        self._n = n_mf

    def __call__(self, u_mf: np.ndarray) -> np.ndarray:
        if self._n == self._full.shape[1]:
            return self._residual(u_mf)
        u = self._full.copy()
        u[:, :self._n] = u_mf
        return self._residual(u)[:, :self._n]

