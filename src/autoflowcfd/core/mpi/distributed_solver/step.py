"""AutoFlowCFD V2.0 - CPU MPI 分布式的单步推进、求解循环与局部时间步长

从 `src/autoflowcfd/core/mpi/distributed_solver.py` 的 `DistributedFRSolver` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `DistributedFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np
from typing import Optional
from loguru import logger
from autoflowcfd.core.mpi import is_root
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme


class _DistributedStepMixin:
    """CPU MPI 分布式的单步推进、求解循环与局部时间步长"""

    def compute_global_residual_norm(self) -> float:
        """全局残差 L2 范数。"""
        return self.state.global_residual_norm()

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
        from autoflowcfd.core.mpi.distributed_cfl import (
            compute_distributed_local_time_step,
        )
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        dist_fc = self.dist_flat_face
        U_extended = self.halo_exchange.exchange(U_local)
        U_compact = U_extended[dist_fc.perm]
        Q_compact = conserved_to_primitive(U_compact[..., :5])

        # 几何量（jacobians/cell_volumes）复用 DistributedMeshAdapter 的
        # 抽取逻辑——它已经处理好"传统模式 vs 完全分布式加载"两种索引
        # 语义的区别（见该类文档），这里不重复那段判断。
        from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter
        adapter = DistributedMeshAdapter(self.partition, dist_fc, self.mesh, self.ops)

        return compute_distributed_local_time_step(
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
        )

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
        from autoflowcfd.core.mpi.distributed_compute import (
            distributed_compute_inviscid_residual,
            distributed_compute_viscous_residual,
        )
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        # 真实 bug 修复（2026-09-02，"传统模式"主 __init__ + 真正 step()
        # 端到端测试此前从未存在，用户明确要求补齐测试覆盖后才发现）：
        # `self.local_solver.config.physics.enable_viscous` 假设
        # `local_solver` 是"完全分布式加载"专用的 `types.SimpleNamespace`
        # 鸭子类型替身（见 `from_fully_distributed_package` 里
        # `config=types.SimpleNamespace(physics=types.SimpleNamespace(
        # enable_viscous=...))` 的构造），但"传统模式"下 `local_solver`
        # 是 `local_solver` 这个 @property 真正构造出的、货真价实的
        # `FRSolver` 实例——`FRSolver` 类本身完全没有 `.config` 属性
        # （单机路径的粘性残差本来就无条件计算，从未有过开关），这里
        # 无条件访问 `.config` 必然 `AttributeError`——"传统模式"这条
        # CLI 生产路径（`solve steady --n-ranks>1`，不加 `--fully-
        # distributed`）因此 100% 必现崩溃，此前从未被任何测试捕捉到。
        # 修复：`local_solver` 没有 `.config` 时（真实 FRSolver 场景）
        # 退回 `True`，与单机 `FRSolver` 的真实行为（粘性残差恒计算）
        # 一致；"完全分布式加载"的替身仍按其显式提供的值。
        _local_config = getattr(self.local_solver, 'config', None)
        enable_viscous = (
            _local_config.physics.enable_viscous if _local_config is not None else True
        )
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
        # 波速放大），`dt_phys_local` 是按物理波速那一份——湍流标量必须
        # 用后者，与单机 `fr_solver/step.py` 里 `turb_dt = dt_physical`
        # 完全一致（k/omega 的显式更新刻意没有 point-implicit 阻尼）。
        U_local_now = self.state.get_local_U()
        dt_mean_local, dt_phys_local = self._compute_distributed_local_time_step(
            U_local_now, mu_t_local=self._prev_mu_t_local, return_physical_too=True,
        )

        mu_t_field_compact = None
        if self.turb_model is not None:
            from autoflowcfd.core.mpi.distributed_turbulence import (
                distributed_compute_turbulence_source_and_viscosity,
            )
            dt_local = dt_phys_local
            mu_t_field_compact, self._turb_ramp_step = distributed_compute_turbulence_source_and_viscosity(
                self.state.get_local_U()[..., :5], self.partition, self.halo_exchange,
                self.turb_halo_exchange, self.dist_flat_face, self.mesh, self.ops,
                self.turb_model, mu, self.wall_distance_compact, dt_local,
                turb_ramp_step=self._turb_ramp_step,
                turb_ramp_steps=self._turb_production_ramp_steps,
                turb_model_name=self.turb_model_name, ddes_model=self.ddes_model,
                iddes_h_max_compact=self.iddes_h_max_compact,
                iddes_h_wn_compact=self.iddes_h_wn_compact,
                des_length_scale_halo_exchange=self.des_length_scale_halo_exchange,
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
                self.dist_flat_face, self.mesh, self.ops, self.sgs_model,
            )

        def _inviscid_dudt(U_stage_local: np.ndarray) -> np.ndarray:
            """无粘 dU/dt（含本 stage 的 halo 交换）。"""
            if n_sps == 1:
                # P0（Order Continuation 最低阶）：单机无粘残差在这个
                # 阶数完全绕开 flat-face 压缩抽象（见 core/fr_residual/
                # inviscid.py::compute_inviscid_residual_fr 的
                # `mesh.n_points_1d==1` 分支文档），P1+ 路径共用的
                # compact/halo 机制在这里不适用，见
                # distributed_order_continuation.py::compute_
                # distributed_p0_inviscid_residual 文档。
                from autoflowcfd.core.mpi.distributed_order_continuation import (
                    compute_distributed_p0_inviscid_residual,
                )
                return compute_distributed_p0_inviscid_residual(self, U_stage_local)
            return distributed_compute_inviscid_residual(
                U_stage_local, self.partition, self.halo_exchange,
                self.dist_flat_face, self.mesh, self.ops,
                boundary_ghost_provider, mach_ref=mach_ref,
            )

        def _viscous_dudt(U_stage_local: np.ndarray) -> np.ndarray:
            """粘性（含湍流涡粘耦合）dU/dt（含本 stage 的 halo 交换）。"""
            return distributed_compute_viscous_residual(
                U_stage_local, self.partition, self.halo_exchange,
                self.dist_flat_face, self.mesh, self.ops,
                mu, boundary_ghost_provider,
                mu_t_field_compact=mu_t_field_compact,
                wmles_model=self.wmles_model,
                wall_distance_compact=self.wall_distance_compact,
            )

        def residual_func_raw(U_flat_trial: np.ndarray) -> np.ndarray:
            """未经预处理的物理残差（TimeIntegrator 约定 dU/dt = -R）。RK3 每个
            stage 都会调用一次：对该 stage 的中间解重新做 halo 交换 +
            残差求值（halo 数据在每个 stage 之间会变化，不能复用上一个
            stage 交换到的邻居数据）。"""
            U_stage_local = U_flat_trial.reshape(n_local, n_sps, n_vars)
            total_dudt = _inviscid_dudt(U_stage_local)
            if enable_viscous:
                total_dudt = total_dudt + _viscous_dudt(U_stage_local)
            return -total_dudt

        def residual_func(U_flat_trial: np.ndarray) -> np.ndarray:
            """供时间积分器推进用：启用低马赫数预处理时返回 `Gamma R`。

            Gamma 线性、逐点，作用在 R 上与作用在 dU/dtau 上等价；它
            **必须**与上面按预处理波速取的 `dt_mean_local` 成对出现
            （完整推导见 core/utils/preconditioning.py 模块末尾；只改
            步长不改方程就是 2026-08-24 那次失稳）。
            Gamma 需要的原始变量由 `apply_low_mach_preconditioner` 从
            试探态算出的 Q 提供——这里显式转换，不依赖任何调用顺序上的
            隐式副作用（与单机 CPU 路径复用 `state.Q` 的做法不同，
            理由同 GPU 侧，见 core/gpu/gpu_preconditioning.py 文档）。
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

        # 模态滤波（2026-09-14 补齐）：单机 `fr_solver/step.py` 每个 RK
        # stage 后都施加（`build_filter_func`），多 GPU 分布式也有
        # `filter_func_gpu`——CPU 分布式此前是唯一没有施加的路径。它是
        # P>=1 的稳定性机制，不是可选项（坍缩坐标/配置点法对高阶模态
        # 混叠天然敏感，真实复现记录见 fr_solver/filter.py）。
        # local 排列里棱柱/四面体交错，所以用按单元类型掩码分派的变体；
        # 单元类型取自 `dist_fc.compact_cell_type`（0=棱柱/1=四面体，
        # 紧凑排列），换回原生排列后切 local 段。
        dist_fc = self.dist_flat_face
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism

        cell_is_prism = native_cell_is_prism(dist_fc)[:n_local]
        filter_func = None
        if n_sps > 1:
            from autoflowcfd.core.fr_solver.filter import (
                build_filter_func_by_cell_type, resolve_filter_mode,
            )
            mode = resolve_filter_mode("cpu-mpi")
            if mode == "sensor":
                filter_func = self._build_sensor_gated_filter_func_distributed(
                    n_local, n_sps, cell_is_prism)
            else:
                filter_func = build_filter_func_by_cell_type(
                    self.ops, n_local, n_sps, cell_is_prism)

        # 正性保持（与单机同一个限制器、同一个核）：几何取 local 段、原生
        # 排列 —— adapter 的 jacobians 在紧凑排列，经 inv_perm 换回后切片，
        # 与上面的 cell_is_prism 同一换序。只在缓存失效（阶数切换）时构建。
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        def _local_geometry():
            from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter
            adapter = DistributedMeshAdapter(self.partition, dist_fc, self.mesh, self.ops)
            det_compact = np.asarray(adapter.jacobians["det_jacs"]).reshape(-1, n_sps)
            return det_compact[dist_fc.inv_perm][:n_local], cell_is_prism

        positivity_func = get_positivity_limiter(self, geometry=_local_geometry)

        # Stage 0 残差单独算一次：既用于收敛监控（与旧实现报告口径一致），
        # 也通过 residual0= 传给 _ssp_rk_stage_step 复用，避免它内部再重复
        # 算一次同样的 R(U^n)。
        # `residual0_raw` 是**物理**残差：收敛监控（state.dU_dt ->
        # compute_global_residual_norm）与自适应 CFL 都必须用它，不能用
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

        # 真实 bug 修复（2026-09-02，见 __init__ 里 self._time_integrator
        # 构造处同一处说明）：此前这里无条件调用 `_ssp_rk_stage_step`，
        # DUAL_TIME（真正时间精度的瞬态仿真）请求了也无路可走——现在
        # 与单机 `fr_solver/step.py::step` 同一个分派方式：`residual_
        # func` 本身就是 `spatial_residual(U) -> R(U)`（`dU/dt=-R(U)`
        # 约定，与 `step_dual_time` 需要的语义完全一致，不需要额外
        # 包装），直接复用。
        if self._time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME:
            U_new_flat = self._time_integrator.step_dual_time(
                U_flat, residual_func, dt_local_flat, dt_physical=dt,
                solution_prev=self._dual_time_U_prev,
                max_inner_iter=self._time_integrator.dual_time_steps,
                filter_func=filter_func, positivity_func=positivity_func,
            )
            self._dual_time_U_prev = U_flat.copy()
        elif self._time_integrator.scheme == TimeIntegrationScheme.IMEX_EULER:
            # 显式无粘对流 + 隐式粘性（阻尼 Picard），与单机
            # `fr_solver/step.py` 同一个拆分、同一个积分器。**2026-09-25 补齐**：
            # 此前这里无条件走 `_ssp_rk_stage_step`，而 IMEX_EULER 在系数表里
            # 没有条目、回退成 1 级的前向 Euler 表 —— `--time-method imex
            # --n-ranks N` 静默跑成了前向 Euler（假选项）。
            def _explicit_R(U_flat_trial):
                U3 = U_flat_trial.reshape(n_local, n_sps, n_vars)
                return (-_inviscid_dudt(U3)).reshape(n_local * n_sps, n_vars)

            def _implicit_R(U_flat_trial):
                if not enable_viscous:
                    return np.zeros_like(U_flat_trial)
                U3 = U_flat_trial.reshape(n_local, n_sps, n_vars)
                return (-_viscous_dudt(U3)).reshape(n_local * n_sps, n_vars)

            U_new_flat = self._time_integrator.step_imex(
                U_flat, _explicit_R, _implicit_R, dt_local_flat,
                positivity_func=positivity_func,
            )
        else:
            # `step()` 对 IMEX/DUAL_TIME/NEWTON_KRYLOV 显式报错（不静默退化成
            # 前向 Euler）；前两者上面已分派，NEWTON_KRYLOV 在构造时即拒绝。
            U_new_flat = self._time_integrator.step(
                U_flat, residual_func, dt_local_flat, residual0=residual0,
                filter_func=filter_func, positivity_func=positivity_func,
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

        residual_norm = self.compute_global_residual_norm()
        # 自适应 CFL 按**全局**残差范数更新：所有 rank 喂同一个值，因此
        # 得到同一个 CFL 数（按各自局部残差更新会让 rank 间 CFL 漂移）。
        if self._cfl_controller is not None:
            self._cfl_controller.update(residual_norm)
        return residual_norm

    def solve(self, n_steps: int, dt: float, output_interval: int = 100, checkpoint_callback=None,
              tol: float = 1e-6, phase_max_iter: Optional[int] = None,
              residual_drop_threshold: float = 1e2):
        """运行分布式求解循环。

        真实 bug 修复（2026-09-02，用户明确要求"不允许出现完成度不是
        100%的功能点"后排查发现）：此前 `output_interval` 只控制
        `logger.info` 进度打印的频率，从未触发任何中间 checkpoint
        保存——分布式路径此前只在 CLI 里 `solve()` 返回*之后*保存一次
        最终 checkpoint（见 `solve_steady_command.py`），跑到一半被
        杀掉/崩溃会丢失全部进度，且没有任何"从分布式 checkpoint 继续
        跑"的机制（`solve resume` 命令对 `--n-ranks`/`--multi-gpu`
        完全没有感知）。与单机路径 `FRSolver.solve(...,
        checkpoint_callback=...)` 同一个设计补上回调机制：调用方
        （CLI）传入的 `checkpoint_callback(solver, iteration)` 在每步
        结束后被调用，由回调自己决定何时/如何保存（通常内部判断
        `iteration % checkpoint_interval`），不在这里耦合具体的保存
        格式——与单机路径的分工完全一致。

        Order Continuation 自动分派（2026-09-02，见 core/mpi/
        distributed_order_continuation.py 模块文档）：与单机
        `FRSolver.solve()`（`self.order_continuation_enabled and
        self.order >= 2` 时自动改用逐阶爬坡）同一个判据——`self.order`
        （目标阶数）>= 2 时自动委托给 `run_distributed_order_
        continuation`，不需要 CLI/调用方显式请求。P0/P1 直接求解
        （真实数值复核见 order_continuation.py 文档"曾经在这里跳过
        P=1"一节，两条阶数下均匀自由流场残差都很好，不需要爬坡）。

        Args:
            n_steps: 最大时间步数
            dt: 时间步长
            output_interval: 输出间隔
            checkpoint_callback: 可选，`callback(solver, iteration)`，
                每步结束后调用一次（与单机 `FRSolver.solve` 同名参数
                同一个约定）
            tol, phase_max_iter, residual_drop_threshold: 仅在触发
                Order Continuation（`self.order >= 2`）时生效，与单机
                `run_order_continuation` 同名参数同一含义。
        """
        if getattr(self, 'order_continuation_enabled', True) and self.order >= 2:
            from autoflowcfd.core.mpi.distributed_order_continuation import (
                run_distributed_order_continuation,
            )
            return run_distributed_order_continuation(
                self, n_steps, dt, tol,
                checkpoint_callback=checkpoint_callback,
                phase_max_iter=phase_max_iter,
                residual_drop_threshold=residual_drop_threshold,
            )

        if is_root():
            logger.info(f"Starting distributed solve: {n_steps} steps, dt={dt}")

        for step_idx in range(n_steps):
            # 执行一步
            residual_norm = self.step(dt)

            # 输出进度
            if step_idx % output_interval == 0 and is_root():
                logger.info(
                    f"Step {step_idx}/{n_steps}, "
                    f"residual_norm={residual_norm:.6e}"
                )

            if checkpoint_callback is not None:
                # 全部 rank 都要调用（checkpoint_callback 内部的
                # distributed_save_checkpoint 本身就是集体操作——需要
                # 每个 rank 各自贡献 local cells 数据才能在 root 组装
                # 出正确的全局状态，只在 root 调用会在非 root rank 的
                # gather 那一侧永久阻塞）。
                checkpoint_callback(self, step_idx + 1)

            # 同步（可选，用于调试）
            # barrier()

        if is_root():
            logger.info("Distributed solve completed.")
