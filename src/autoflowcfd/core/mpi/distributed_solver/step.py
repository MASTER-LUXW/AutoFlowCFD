"""AutoFlowCFD V2.0 - CPU MPI 分布式的单步推进与局部时间步长（求解循环在 solve_loop.py）

`DistributedFRSolver` 的一组方法（mixin）。

**只含方法，没有状态**：全部属性由 `DistributedFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from typing import Any

from functools import partial

import numpy as np

from autoflowcfd.core.mpi.reductions import MPIReductions
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme


class _DistributedStepMixin:
    """CPU MPI 分布式的单步推进与局部时间步长"""

    # 宿主类在 __init__ / 其它 mixin 中提供的属性（仅类型声明）
    _cfl_controller: Any
    _distributed_positivity_limiter: Any
    _newton_last_info: Any
    _prev_mu_t_local: Any
    artificial_viscosity_alpha: Any
    artificial_viscosity_enabled: Any
    ddes_model: Any
    dist_flat_face: Any
    fixed_cfl_number: Any
    halo_exchange: Any
    iddes_h_max_compact: Any
    iddes_h_wn_compact: Any
    local_solver: Any
    low_mach_precond_enabled: Any
    mesh: Any
    ops: Any
    order: Any
    partition: Any
    sgs_model: Any
    state: Any
    time_integrator: Any
    turb_model: Any
    turb_model_name: Any
    wall_distance_compact: Any
    wall_model: Any

    def _compute_distributed_local_time_step(self, U_local, mu_t_local=None,
                                             return_physical_too: bool = False):
        """逐单元局部 CFL 步长（2026-09-14 补齐，取代此前的全局固定 dt）。

        此前这条路径用调用方传入的全局固定 dt 铺满所有单元，源码里记作
        "已接受的简化"。那条"简化"有三重真实代价（步长被全场最苛刻单元
        卡死、逐 SP 的几何/度量 CFL 保护完全失效、自适应 CFL 与低马赫数
        预处理都无从生效），完整论证与实现见
        `core/mpi/distributed_cfl.py` 模块文档。

        需要一次额外的 halo 交换：局部 dt 的谱半径求和必须读到邻居单元
        （含 halo）的速度与声速，这是这项功能的内在需求，不是可以省掉的
        开销（相比之下 RK3 每个 stage 各有一次交换）。
        """
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        from autoflowcfd.core.mpi.comm import allreduce_min
        from autoflowcfd.core.mpi.distributed_cfl import (
            compute_distributed_local_time_step,
        )
        from autoflowcfd.core.time_integration.time_accurate import uniform_time_step

        dist_fc = self.dist_flat_face
        U_extended = self.halo_exchange.exchange(U_local)
        U_compact = U_extended[dist_fc.perm]
        Q_compact = conserved_to_primitive(U_compact[..., :5])

        # 几何量（jacobians/cell_volumes）复用 DistributedMeshAdapter 的
        # 抽取逻辑——它已经处理好"传统模式 vs 完全分布式加载"两种索引
        # 语义的区别（见该类文档），这里不重复那段判断。
        from autoflowcfd.core.mpi.distributed_compute import distributed_mesh_adapter
        adapter = distributed_mesh_adapter(self.partition, dist_fc, self.mesh, self.ops)

        # 时间精确子步：全场同一步长（跨 rank 最小值），见 time_integration/time_accurate.py
        return uniform_time_step(self, compute_distributed_local_time_step(
            U_compact, Q_compact, dist_fc, self.mesh,
            jacobians=adapter.jacobians,
            cell_volumes=adapter.cell_volumes,
            n_local_cells=self.partition.n_local_cells,
            mu_molecular=self.local_solver.mu_molecular,
            freestream=self.local_solver.freestream,
            cfl_controller=self._cfl_controller,
            fixed_cfl_number=self.fixed_cfl_number,
            current_order=getattr(self, "current_order", self.order),
            low_mach_precond_enabled=getattr(
                self, "low_mach_precond_enabled", False),
            mu_t_local=mu_t_local,
            return_physical_too=return_physical_too,
        ), return_physical_too, reduce_min=allreduce_min)

    def _viscous_enabled(self) -> bool:
        """粘性残差是否计算：完全分布式加载的替身显式给出，真实 `FRSolver`（传统模式）恒为真
        （它没有 `.config`，见 `step` 里 2026-09-02 的修复记录）。"""
        cfg = getattr(self.local_solver, "config", None)
        return cfg.physics.enable_viscous if cfg is not None else True

    def _inviscid_dudt_local(self, U_local: np.ndarray) -> np.ndarray:
        """无粘 dU/dt（含本次求值的 halo 交换），`(n_local, n_sps, n_vars)`。"""
        from autoflowcfd.core.mpi.distributed_compute import distributed_compute_inviscid_residual

        ls = self.local_solver
        return distributed_compute_inviscid_residual(
            U_local, self.partition, self.halo_exchange, self.dist_flat_face, self.mesh, self.ops,
            ls.boundary_ghost_provider, mach_ref=ls.freestream["mach_ref"])

    def _viscous_dudt_local(self, U_local: np.ndarray, mu_t_compact, nu_av_compact) -> np.ndarray:
        """粘性（含湍流涡粘耦合与问题单元人工扩散）dU/dt（含本次求值的 halo 交换）。"""
        from autoflowcfd.core.mpi.distributed_compute import distributed_compute_viscous_residual

        ls = self.local_solver
        dudt = distributed_compute_viscous_residual(
            U_local, self.partition, self.halo_exchange, self.dist_flat_face, self.mesh, self.ops,
            ls.mu_molecular, ls.boundary_ghost_provider, mu_t_field_compact=mu_t_compact,
            wall_model=self.wall_model)
        if nu_av_compact is not None:
            from autoflowcfd.core.mpi.distributed_artificial_viscosity import (
                distributed_artificial_diffusion_dudt,
            )
            dudt = dudt + distributed_artificial_diffusion_dudt(
                U_local, nu_av_compact, self.partition, self.halo_exchange, self.dist_flat_face, self.mesh,
                self.ops)
        return dudt

    def _mean_flow_dudt_local(self, U_local: np.ndarray, mu_t_compact, nu_av_compact) -> np.ndarray:
        """平均流 dU/dt（无粘 + 粘性），`(n_local, n_sps, n_vars)`。"""
        dudt = self._inviscid_dudt_local(U_local)
        if self._viscous_enabled():
            dudt = dudt + self._viscous_dudt_local(U_local, mu_t_compact, nu_av_compact)
        return dudt

    def step(self, dt: float) -> float:
        """执行一步时间推进（分布式版本）。

        真正的 3-stage Shu-Osher SSP-RK3（与单机 FRSolver 同一套实现，
        `TimeIntegrator._ssp_rk_stage_step`，见 core/fr_solver/step.py::
        step 的 mean_flow_residual，这里的 `residual_func` 是它的分布式
        版本，两套约定必须严格一致）：每个 stage 都要用该 stage 的中间解
        重新做一次完整的 halo 交换 + 残差求值——不能像旧版本那样只算
        一次残差就套用 RK3 的名字（旧实现自己的注释承认"简化为单步
        Euler"，与 SSP-RK3 的时间精度/稳定域完全不是一回事）。

        两个容易踩错、已用独立数值脚本验证过的约定，都严格照抄
        `fr_solver/step.py::step`：
        1. `_ssp_rk_stage_step` 期望的 `solution`/`dt_local` 是展平成
           `(n_local_cells*n_sps, n_vars)` / `(n_local_cells*n_sps,)` 的
           2D/1D 数组（`dt = dt_local[:, None]` 只能对 2D solution 广播），
           不是 `(n_local_cells, n_sps, n_vars)` 的 3D 数组——直接传 3D
           会在第一个 stage 就因广播形状不匹配抛 ValueError。
        2. `compute_*_residual_fr` 返回的是 dU/dt 本身，`residual_func`
           必须返回其**负值**（`TimeIntegrator` 的约定是 dU/dt=-R(U)）；
           旧的 Euler 实现 `U += dt*total_residual` 直接用未取负的和，
           这一点在那个实现里恰好是自洽的（因为它没有经过 R→L=-R 这层
           转换），但复用 `_ssp_rk_stage_step` 就必须显式取负，否则解会
           往错误的时间方向积分。

        SST 湍流模型（2026-09-02）：与单机 `fr_solver/step.py::step` 同一个
        算子分裂设计——湍流源项+输运在物理步开始时求值一次（用当前状态，
        不在每个 RK 子迭代里重算），产出的 `mu_t_field_compact` 在本步
        全部 RK 子阶段内保持不变，供粘性残差 BR1 界面项消费；k/omega 场
        本身用独立于平均流 RK 的显式-半隐式更新（`SSTModelFR.update_
        fields`），不在这里更新。`__init__` 只接受 'none'/'SST'。

        Args:
            dt: 物理时间步长（分布式路径目前用全局固定步长，不做单机
                路径那种逐 cell 局部 CFL 时间步——旧实现本来就是全局
                dt，这里不新增自适应步长这个单独的功能点）

        Returns:
            residual_norm: 全局残差 L2 范数（RK3 第 0 阶段的 dU/dt 范数，
                与旧实现的报告口径一致，用于跨迭代收敛监控）
        """
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales

        mu = self.local_solver.mu_molecular
        boundary_ghost_provider = self.local_solver.boundary_ghost_provider
        mach_ref = self.local_solver.freestream["mach_ref"]
        n_local = self.partition.n_local_cells
        n_sps = self.state.n_sps
        n_vars = self.state.n_vars

        # SST 湍流源项+输运（算子分裂，物理步开始时求值一次，见本方法
        # 文档）——用当前（上一步末尾的）状态，产出的 mu_t_field_compact
        # 供本步全部 RK 子阶段的粘性残差使用。
        # 逐单元局部 CFL 步长（2026-09-14 取代全局固定 dt，见
        # `_compute_distributed_local_time_step` 与
        # core/mpi/distributed_cfl.py 模块文档）。
        # `dt_mean_local` 是平均流用的（启用低马赫数预处理时按预处理
        # 波速放大），`dt_phys_local` 是按物理波速那一份——湍流标量的**显式**
        # 更新用后者（k/omega 的显式更新刻意没有 point-implicit 阻尼），隐式紧耦合
        # Newton 用前者，与单机 `fr_solver/step.py` 一致（见 coupled_step.py 模块文档）。
        dual_time = self.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME
        # 隐式步：稳态 Newton-Krylov，或双时间步的隐式内迭代（implicit/newton/physical_time.py）
        is_newton = dual_time or self.time_integrator.scheme == TimeIntegrationScheme.NEWTON_KRYLOV
        dist_fc = self.dist_flat_face
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism

        cell_is_prism = native_cell_is_prism(dist_fc)[:n_local]
        order_now = int(getattr(self, "current_order", self.order))
        U_local_now = self.state.get_local_U()
        # 本步冻结的问题单元人工扩散系数（compact 排列，与单机 step.py 同一算子
        # 分裂约定；未启用时 None）。粘性步长限制按 rho*nu 计入（单机
        # `_get_cfl_viscosity_field` 同一口径）。
        nu_av_compact = None
        mu_cfl_local = self._prev_mu_t_local
        if self.artificial_viscosity_enabled:
            from autoflowcfd.core.mpi.distributed_artificial_viscosity import (
                distributed_artificial_diffusivity, local_from_compact,
            )
            nu_av_compact = distributed_artificial_diffusivity(
                U_local_now[..., :5], self.partition, self.halo_exchange, dist_fc, self.mesh, self.ops,
                order=order_now, alpha_av=self.artificial_viscosity_alpha)
            mu_av_local = U_local_now[..., 0] * local_from_compact(nu_av_compact, dist_fc, n_local)
            mu_cfl_local = mu_av_local if mu_cfl_local is None else mu_cfl_local + mu_av_local
        dt_mean_local, dt_phys_local = self._compute_distributed_local_time_step(
            U_local_now, mu_t_local=mu_cfl_local, return_physical_too=True,
        )
        # 合成湍流入口（SEM）每个物理步推进一次（与单机同一个函数；各 rank 的 SEM 由确定性种子构造、
        # 推进序列相同，入口脉动在分区间一致）；时间精确子步按实际步长推进
        from autoflowcfd.boundary.synthetic_inlet import advance_synthetic_inlets
        from autoflowcfd.core.time_integration.time_accurate import sem_time_step
        advance_synthetic_inlets(boundary_ghost_provider, self.local_solver.freestream, sem_time_step(self, dt))

        mu_t_field_compact = None
        from autoflowcfd.core.turbulence.registry import has_transport_equations

        coupled_nk = (is_newton and self.turb_model is not None
                      and has_transport_equations(self.turb_model_name))
        if coupled_nk:
            # 隐式步 + 带输运方程的湍流模型：湍流与平均流在下面紧耦合求解（单机同一个算法，
            # time_integration/implicit/newton/coupled_step.py；适配器见 distributed_implicit.py）。
            # 这里只取当前状态的涡粘供残差监控
            from autoflowcfd.core.mpi.distributed_implicit import DistributedTurbulenceBackend

            turb_backend = DistributedTurbulenceBackend(self, cell_is_prism, order_now)
            turb_backend.prepare_inputs()
            turb_backend.rates()
            mu_t_field_compact = turb_backend.trial_mu_t_compact()
        elif self.turb_model is not None:
            from autoflowcfd.core.mpi.distributed_turbulence import (
                distributed_compute_turbulence_source_and_viscosity,
            )
            mu_t_field_compact = distributed_compute_turbulence_source_and_viscosity(
                self.state.get_local_U()[..., :5], self.partition, self.halo_exchange,
                self.dist_flat_face, self.mesh, self.ops,
                self.turb_model, mu, self.wall_distance_compact, dt_phys_local,
                ramp_owner=self,
                turb_model_name=self.turb_model_name, ddes_model=self.ddes_model,
                iddes_h_max_compact=self.iddes_h_max_compact,
                iddes_h_wn_compact=self.iddes_h_wn_compact,
                # compact 面空间的 provider（group_code 已重切），湍流输运的
                # 壁面/来流条件靠它按边界组取类型
                boundary_ghost_provider=boundary_ghost_provider,
            )
        elif self.sgs_model is not None:
            # LES（2026-09-02）：WALE 纯代数模型，用当前状态现算，见
            # distributed_compute_les_viscosity 文档。
            from autoflowcfd.core.mpi.distributed_turbulence import (
                distributed_compute_les_viscosity,
            )
            mu_t_field_compact = distributed_compute_les_viscosity(
                self.state.get_local_U()[..., :5], self.partition, self.halo_exchange,
                self.dist_flat_face, self.mesh, self.ops, self.sgs_model, boundary_ghost_provider,
            )

        def residual_func_raw(U_flat_trial: np.ndarray) -> np.ndarray:
            """未经预处理的物理残差（TimeIntegrator 约定 dU/dt = -R）。RK3 每个
            stage 都会调用一次：对该 stage 的中间解重新做 halo 交换 +
            残差求值（halo 数据在每个 stage 之间会变化，不能复用上一个
            stage 交换到的邻居数据）。"""
            U_stage_local = U_flat_trial.reshape(n_local, n_sps, n_vars)
            return -self._mean_flow_dudt_local(U_stage_local, mu_t_field_compact, nu_av_compact)

        def residual_func(U_flat_trial: np.ndarray) -> np.ndarray:
            """供时间积分器推进用：启用低马赫数预处理时返回 `Gamma R`。

            Gamma 线性、逐点，作用在 R 上与作用在 dU/dtau 上等价；它
            **必须**与上面按预处理波速取的 `dt_mean_local` 成对出现
            （完整推导见 core/utils/preconditioning.py 模块末尾；只改
            步长不改方程就是 2026-08-24 那次失稳）。
            Gamma 需要的原始变量由 `apply_low_mach_preconditioner` 从
            试探态算出的 Q 提供——这里显式转换，不依赖任何调用顺序上的
            隐式副作用（与单机 CPU 路径复用 `state.Q` 的做法不同：残差函数按参数接收试探态，
            不保证求解器上的原始变量与之同步）。
            """
            res = residual_func_raw(U_flat_trial)
            if self.low_mach_precond_enabled:
                from autoflowcfd.core.utils.preconditioning import (
                    apply_low_mach_preconditioner,
                )
                U_trial = U_flat_trial.reshape(n_local, n_sps, n_vars)
                Q_trial = conserved_to_primitive(U_trial[..., :5])
                res = apply_low_mach_preconditioner(
                    res, Q_trial, mach_ref, out=res)
            return res.reshape(n_local * n_sps, n_vars)

        U_flat = self.state.get_local_U().reshape(n_local * n_sps, n_vars)
        dt_local_flat = dt_mean_local.reshape(n_local * n_sps)

        # 正性保持（与单机同一个限制器、同一个核）：几何取 local 段、原生
        # 排列 —— adapter 的 jacobians 在紧凑排列，经 inv_perm 换回后切片，
        # 与上面的 cell_is_prism 同一换序。只在缓存失效（阶数切换）时构建。
        positivity_func = self._distributed_positivity_limiter()

        # Stage 0 残差单独算一次：既用于收敛监控（与旧实现报告口径一致），
        # 也通过 residual0= 传给 _ssp_rk_stage_step 复用，避免它内部再重复
        # 算一次同样的 R(U^n)。
        # `residual0_raw` 是**物理**残差：收敛监控（state.dU_dt ->
        # residual_norm.py::monitor_residuals）与自适应 CFL 都必须用它，不能用
        # 预处理值（Gamma 可逆、两者同时趋零，但量级不同，用预处理值会
        # 让打印的残差与历史算例失去可比性）。与单机 step.py 同一分工。
        residual0_raw = residual_func_raw(U_flat)
        self.state.dU_dt[:n_local] = -residual0_raw
        if self.low_mach_precond_enabled:
            from autoflowcfd.core.utils.preconditioning import (
                apply_low_mach_preconditioner,
            )
            residual0 = apply_low_mach_preconditioner(
                residual0_raw, self.state.Q[:n_local], mach_ref,
                out=residual0_raw,
            ).reshape(n_local * n_sps, n_vars)
        else:
            residual0 = residual0_raw.reshape(n_local * n_sps, n_vars)

        if is_newton:
            # 隐式步：与单机 CPU/GPU 同一个实现（带输运方程的湍流模型时紧耦合 implicit/newton/coupled_step.py，
            # 否则 implicit/newton/mean_flow_step.py），归约换成跨 rank 的 MPIReductions，块 Jacobi 着色是全局
            # 一致着色里本 rank 那一段（见 core/mpi/distributed_implicit.py 模块文档）。稳态一步一个
            # Newton 步；双时间步每个物理步做若干个带 BDF 物理时间项的 Newton 步（implicit/newton/physical_time.py）
            from autoflowcfd.core.mpi.distributed_coarse import CompactCellValues, coarse_comm_context
            from autoflowcfd.core.mpi.distributed_implicit import (
                distributed_block_jacobi_colors, distributed_coupling_graph, distributed_mean_flow_assembler,
            )
            from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import (
                step_mean_flow_newton,
            )

            coarse_ctx = coarse_comm_context(
                self.partition, CompactCellValues(self.halo_exchange, self.dist_flat_face.perm, np))
            coupled = None
            if coupled_nk:
                from autoflowcfd.core.mpi.distributed_implicit import DistributedCoupledBackend
                from autoflowcfd.core.time_integration.implicit.newton.coupled_step import step_coupled_newton

                coupled = DistributedCoupledBackend(self, cell_is_prism, order_now, nu_av_compact, coarse_ctx)

            def newton_step(dtau_flat, physical_time=None, advance_ramp=True):
                """一个 Newton 步，新状态写回 `state`（与湍流模型），返回诊断 info。"""
                if coupled is not None:
                    return step_coupled_newton(self, coupled, dtau_flat, physical_time=physical_time,
                                               advance_ramp=advance_ramp)
                u_next, info = step_mean_flow_newton(
                    self, residual_func, self.state.get_local_U().reshape(n_local * n_sps, n_vars), dtau_flat,
                    _reference_scales(self.local_solver.freestream, n_vars),
                    red=MPIReductions(np), cell_is_prism=cell_is_prism,
                    cell_colors=lambda: distributed_block_jacobi_colors(self),
                    coupling_graph=partial(distributed_coupling_graph, self, coarse_ctx), global_coarse=coarse_ctx,
                    order=order_now, positivity=positivity_func, physical_time=physical_time,
                    block_assembler=distributed_mean_flow_assembler(
                        self, self.local_solver, order=order_now, mu_t_compact=mu_t_field_compact,
                        exchange=self.halo_exchange.exchange, perm=self.dist_flat_face.perm, n_sps=n_sps,
                        nu_av_compact=nu_av_compact))
                u_next = u_next.reshape(n_local, n_sps, n_vars)
                self.state.U[:n_local] = u_next
                self.state.Q[:n_local] = conserved_to_primitive(u_next[..., :5])
                return info

            if dual_time:
                from autoflowcfd.core.time_integration.implicit.newton.physical_time import (
                    advance_physical_step, previous_newton_state, store_newton_state,
                )

                def pseudo_time_step():
                    return self._compute_distributed_local_time_step(
                        self.state.get_local_U(), mu_t_local=mu_cfl_local).reshape(n_local * n_sps)

                x_now = coupled.state() if coupled is not None else U_flat.copy()
                advance_physical_step(
                    self, x_now, previous_newton_state(self, np, n_local, n_sps),
                    lambda term, first: newton_step(pseudo_time_step(), term, first), dt)
                store_newton_state(self, x_now, n_local, n_sps, n_vars)
                nk_info = self._newton_last_info
            else:
                nk_info = newton_step(dt_local_flat)
            U_new_flat = self.state.get_local_U().reshape(n_local * n_sps, n_vars)
            if coupled is not None:
                mu_t_field_compact = coupled.turb.mu_t_compact
        else:
            # `TimeIntegrator.step()` 对隐式格式显式报错（不静默退化成前向 Euler）；它们上面都已分派。
            U_new_flat = self.time_integrator.step(
                U_flat, residual_func, dt_local_flat, residual0=residual0,
                positivity_func=positivity_func,
            )

        U_new_local = U_new_flat.reshape(n_local, n_sps, n_vars)
        self.state.U[:n_local] = U_new_local
        self.state.Q[:n_local] = conserved_to_primitive(U_new_local[..., :5])

        # 下一步的粘性 CFL 限制要用本步算出的涡粘（与单机读取湍流模型
        # 已存字段是同一时序）。转成 local 排列缓存：mu_t_field_compact
        # 在"棱柱在前"紧凑空间，必须先 inv_perm 换回原生排列再切 local
        # （紧凑空间的前 n_local 段**不是** local 单元，见
        # distributed_cfl.py 模块文档）。
        if mu_t_field_compact is not None:
            self._prev_mu_t_local = (
                mu_t_field_compact[self.dist_flat_face.inv_perm][:n_local])

        from autoflowcfd.core.time_integration.residual_norm import explicit_turbulence_residual, monitor_residuals

        red = MPIReductions(np)
        residual_norm, self._residual_per_equation = monitor_residuals(
            red, self.state.dU_dt[:n_local], cell_is_prism, order_now,
            _reference_scales(self.local_solver.freestream, n_vars))
        self._residual_turbulence = explicit_turbulence_residual(red, self.turb_model, cell_is_prism, order_now)
        # 自适应 CFL 按**全局**残差范数更新：所有 rank 喂同一个值，因此
        # 得到同一个 CFL 数（按各自局部残差更新会让 rank 间 CFL 漂移）。
        # 双时间步的伪时间 CFL 在每个物理步的内迭代里更新（implicit/newton/physical_time.py）
        if self._cfl_controller is not None and not dual_time:
            if is_newton:
                # SER 看 Newton 实际在解的系统的全局残差 ||Gamma R||，并区分
                # "残差上升但步被完整接受"与"步没被完整接受"（adaptive_cfl/ser.py）
                from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import newton_step_ok

                self._cfl_controller.update(nk_info["res_norm"], step_ok=newton_step_ok(nk_info))
            else:
                self._cfl_controller.update(residual_norm)
        return residual_norm
