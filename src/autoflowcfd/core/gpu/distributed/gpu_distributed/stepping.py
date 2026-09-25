"""AutoFlowCFD V2.0 - `MultiGPUDistributedSolver` 的单步推进与求解循环（mixin，只含方法）。

从 `core/gpu/distributed/gpu_distributed.py` 拆出（2026-09-25）。
"""

import time
from typing import Any, Dict, Optional

from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.mpi import is_root
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme


class _MultiGPUSteppingMixin:
    """完全分布式加载入口、阶数切换、单步推进与求解循环。"""

    @classmethod
    def from_fully_distributed_package(cls, package: dict, n_ranks: int,
                                        device_id=None, rank=None, root_context=None):
        """真正的"完全分布式加载"构造入口（#1，2026-09-02）——与主
        `__init__`（"传统模式"：每个 rank 独立加载完整全局网格）的关键
        区别：`package` 是 root rank 预先算好、已经按本 rank 的 compact
        索引空间切好的紧凑数据，本 rank 从未持有、也不需要持有完整
        全局网格。与 CPU `DistributedFRSolver.from_fully_distributed_
        package` 共用同一套 `build_fully_distributed_rank_package`/
        `distributed_mesh_load_v2`（package 构造逻辑与后端无关）。

        实现拆到独立模块 `gpu_distributed_fully_distributed.py`（控制
        单文件行数，与 `_interpolate_to_new_order`/
        `gpu_distributed_order_continuation.py` 同一个拆分动机），见该
        模块文档完整说明（范围边界：支持 turbulence_model='none'/'sst'/
        'ddes'/'iddes'/'wmles'/'les'，DUAL_TIME/checkpoint/Order
        Continuation 均已接入）。

        Args:
            package: `build_fully_distributed_rank_package` 的返回值
                （或 `distributed_mesh_load_v2` 经 MPI 收发后本 rank
                收到的那一份）
            n_ranks: MPI rank 总数
            device_id: GPU 设备号（None 时按 rank 轮询分配）
            rank: 当前 rank（None 时从 MPI 获取）
            root_context: 仅 root rank 需要非 None——`distributed_mesh_
                load_v2` 返回的第二个值，供 Order Continuation 使用，
                见 `gpu_distributed_fully_distributed.py` 模块文档。

        Returns:
            MultiGPUDistributedSolver 实例
        """
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            build_multi_gpu_solver_from_fully_distributed_package,
        )
        return build_multi_gpu_solver_from_fully_distributed_package(
            cls, package, n_ranks, device_id=device_id, rank=rank, root_context=root_context,
        )

    def _interpolate_to_new_order(self, target_p: int) -> None:
        """阶数切换（2026-09-02，见 core/gpu/distributed/
        gpu_distributed_order_continuation.py 模块文档）——与 CPU
        `DistributedFRSolver._interpolate_to_new_order` 同一个命名/
        调用约定，供 `run_distributed_order_continuation`（CPU/GPU
        共用同一份迭代循环，见 core/mpi/distributed_order_
        continuation.py）统一调用。"""
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )
        gpu_interpolate_to_new_order(self, target_p)

    def step(self, dt: float = 0.0) -> float:
        """执行一个分布式时间步。

        时间推进走与 CPU/单机 GPU **同一个**积分器（`TimeIntegrator`：SSP-RK
        stage 推进 / 双时间步），每个 stage 的残差求值都重新做 halo 交换
        （`_spatial_residual` -> `_compute_total_residual_gpu`），每个 stage
        收尾先滤波、再施加守恒的正性限制器（`_get_positivity_limiter_gpu`）。

        Args:
            dt: 时间步长

        Returns:
            residual_norm: 全局残差范数
        """
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell

        # #1（2026-08-28）：此前这里用 self.mesh.n_cells（全局单元数）
        # 展平/reshape self.U_gpu——self.U_gpu 现在只有 n_local 个单元，
        # 用全局尺寸 reshape 会形状不匹配崩溃。改为统一使用 n_local。
        #
        # 逐单元局部 CFL 步长（2026-09-14）：此前这里直接铺用调用方传入
        # 的全局固定 `dt`，并把它记作"与 CPU 分布式路径一致的、已被接受
        # 的简化"。用户明确指出本项目不接受简化，本轮与 CPU 分布式同批
        # 补齐——`_compute_local_time_step_gpu` 已重写（修掉两处潜伏 bug
        # 与"只用 SP0"的简化，见该方法文档），这里改用它。
        # 返回值是**紧凑排列**（棱柱在前），按 inv_perm 换回原生排列后
        # 切 local 段才能与 `self.U_gpu` 对齐。
        # 真实 bug 修复（2026-09-02，与 CPU 分布式 `DistributedFRSolver.
        # step` 同一处修复、同一个理由——用户明确要求"不允许出现完成度
        # 不是100%的功能点"后排查发现）：此前这里无论 `self.time_
        # integrator.scheme` 是什么都无条件走下面手动展开的 RK stage
        # 逻辑，`scheme` 只被用来查系数表——DUAL_TIME（真正时间精度的
        # 瞬态仿真，DES/LES 场景理应使用的模式）请求了也完全无路可走。
        # 与单机 GPU `gpu_solver.py::step` 同一个分派方式（`GPUTime
        # Integrator.step_dual_time` 本身早已实现，只是从未被这条
        # 分布式路径调用过）：构造一个把 trial U 临时写入 `self.U_gpu`
        # 再调用既有 `_compute_total_residual_gpu`（自带 halo 交换）的
        # 残差闭包，直接复用，不需要另起一套残差组装逻辑。
        mu_t_field = None

        def _spatial_residual(U_flat_trial, inviscid=True, viscous=True):
            """物理残差 R（约定 dU/dt = -R）。残差组装与 halo 交换都读
            `self.U_gpu`，所以试探态临时写进去、算完恢复。`mu_t_field` 在
            下面各分支里、第一次调用之前求好（本步内固定，算子分裂）。
            `inviscid`/`viscous` 供 IMEX 分别取显式/隐式两半。"""
            U_trial = U_flat_trial.reshape(n_local, n_sps, 5)
            saved_U = self.U_gpu
            self.U_gpu = U_trial
            try:
                if inviscid and viscous:
                    res = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
                else:
                    res = self._compute_total_residual_gpu(
                        mu_t_field=mu_t_field, inviscid=inviscid, viscous=viscous)
            finally:
                self.U_gpu = saved_U
            return (-res).reshape(n_local * n_sps, 5)

        positivity_func = self._get_positivity_limiter_gpu()

        if self.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME:
            # DUAL_TIME 下预处理不启用，局部 dt 只有一份；湍流仍用它
            # （物理波速那一份）。
            _dtm = self._compute_local_time_step_gpu()
            _dt_turb = float(cp.mean(_dtm[self._inv_perm_gpu][:n_local]))
            mu_t_field = self._compute_turbulence_source_distributed(_dt_turb)
            U_flat = self.U_gpu.reshape(n_local * n_sps, 5)

            # 内层伪时间迭代的局部加速步长：与单机一致用局部 CFL 步长
            # （`dt` 仍然是真正的物理时间步长，通过 dt_physical= 传入）。
            dt_mean_c = self._compute_local_time_step_gpu()
            dt_mean_local = dt_mean_c[self._inv_perm_gpu][:n_local]
            pseudo_dt = cp.broadcast_to(
                dt_mean_local[:, None], (n_local, n_sps)).reshape(n_local * n_sps)
            U_new_flat = self.time_integrator.step_dual_time(
                U_flat, _spatial_residual, pseudo_dt, dt_physical=dt,
                solution_prev=self._dual_time_U_prev,
                max_inner_iter=self.time_integrator.dual_time_steps,
                filter_func=self.filter_func_gpu, positivity_func=positivity_func,
            )
            self._dual_time_U_prev = U_flat.copy()
            self.U_gpu = U_new_flat.reshape(n_local, n_sps, 5)
            final_res_flat = _spatial_residual(U_new_flat)
            residual_norm = self._global_residual_norm(final_res_flat)
            self.residual_history.append(residual_norm)
            self.iteration += 1
            self._update_cfl_controller(residual_norm)
            return residual_norm

        dt_mean_c, dt_phys_c = self._compute_local_time_step_gpu(
            return_physical_too=True)
        dt_mean_local = dt_mean_c[self._inv_perm_gpu][:n_local]
        dt_phys_local = dt_phys_c[self._inv_perm_gpu][:n_local]
        dt_flat = cp.broadcast_to(
            dt_mean_local[:, None], (n_local, n_sps)).reshape(n_local * n_sps)

        # 湍流源项求值（算子分裂，每个 step 开始时计算一次，用当前——
        # 上一步末尾——的状态，与 CPU 分布式 SST 同一个时序，见
        # distributed_solver.py::step 文档）。turb_model_gpu 为 None
        # （turbulence_model='none'）时恒返回 None。
        # 湍流标量用**物理**波速算出的那一份 dt（见
        # gpu_distributed_init.py::_compute_turbulence_source_distributed
        # 第 5 步的说明与单机 step.py 的 `turb_dt = dt_physical`）。
        mu_t_field = self._compute_turbulence_source_distributed(
            float(cp.mean(dt_phys_local)))

        U_flat = self.U_gpu.reshape(n_local * n_sps, 5)

        def _precond(res_flat, U_state_flat):
            """施加低马赫数预处理 `Gamma R`（就地）。

            Gamma 线性、逐点，作用在残差上与作用在 dU/dt 上等价（见
            core/utils/preconditioning.py 模块末尾）。它**必须**与上面
            按预处理波速取的 `dt_flat` 成对出现——只改步长不改方程就是
            2026-08-24 那次失稳。未启用时原样返回。
            """
            if not self.low_mach_precond_enabled:
                return res_flat
            from autoflowcfd.core.gpu.gpu_preconditioning import (
                apply_low_mach_preconditioner_gpu,
            )
            R3 = res_flat.reshape(n_local, n_sps, 5)
            out = apply_low_mach_preconditioner_gpu(
                R3, U_state_flat.reshape(n_local, n_sps, 5),
                self.freestream["mach_ref"], out=R3)
            return out.reshape(n_local * n_sps, 5)

        # 时间推进走与 CPU/单机 GPU **同一个**积分器（`TimeIntegrator` 的
        # stage 推进，含每个 stage 的滤波与守恒正性限制）。此前这里手工展开
        # 了一份 RK stage：正性用逐点硬钳、且顺序是"先钳后滤"，与 CPU 那份
        # 已经不一致（2026-09-25 删除）。
        #
        # 残差范数取 **stage 0 的物理残差** `R(U^n)`，与 CPU 单机/CPU 分布式/
        # 单机 GPU 同一口径。此前这里报的是**最后一个 stage** 的残差
        # （RK3 下是 `R(U^(2))`），同一个算例在不同后端上打印的残差因此不是
        # 同一个量，自适应 CFL 控制器也吃到不同的信号。
        residual0_raw = _spatial_residual(U_flat)
        residual_norm = self._global_residual_norm(residual0_raw)

        if self.time_integrator.scheme == TimeIntegrationScheme.IMEX_EULER:
            # 显式无粘对流 + 隐式粘性（阻尼 Picard），与单机/CPU 分布式同一个
            # 拆分、同一个积分器（低马赫预处理在 IMEX 下不启用，理由见
            # `FRSolver.__init__`）。**2026-09-25 补齐**：此前这里只查 RK 系数
            # 表，IMEX_EULER 没有条目、回退成 1 级前向 Euler —— `--time-method
            # imex --multi-gpu` 静默跑成前向 Euler（假选项）。
            U_new_flat = self.time_integrator.step_imex(
                U_flat,
                lambda U: _spatial_residual(U, viscous=False),
                lambda U: _spatial_residual(U, inviscid=False),
                dt_flat, positivity_func=positivity_func,
            )
        else:
            residual0 = _precond(residual0_raw, U_flat)

            def _residual(U_flat_trial):
                return _precond(_spatial_residual(U_flat_trial), U_flat_trial)

            U_new_flat = self.time_integrator.step(
                U_flat, _residual, dt_flat, residual0=residual0,
                filter_func=self.filter_func_gpu, positivity_func=positivity_func,
            )
        self.U_gpu = U_new_flat.reshape(n_local, n_sps, 5)

        self.residual_history.append(residual_norm)
        self.iteration += 1
        self._update_cfl_controller(residual_norm)
        return residual_norm

    def solve(
        self,
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        output_interval: int = 10,
        checkpoint_callback=None,
        phase_max_iter: Optional[int] = None,
        residual_drop_threshold: float = 1e2,
    ) -> Dict[str, Any]:
        """执行分布式稳态求解循环。

        真实 bug 修复（2026-09-02，与 CPU MPI 分布式 `DistributedFRSolver.
        solve` 同一处修复、同一个理由——用户明确要求"不允许出现完成度
        不是100%的功能点"后排查发现）：此前 `output_interval` 只控制
        `print` 进度打印频率，没有任何中间 checkpoint 保存机制。补上
        与单机 `FRSolver.solve`/CPU 分布式 `DistributedFRSolver.solve`
        同一个约定的 `checkpoint_callback(solver, iteration)` 回调。

        Order Continuation 自动分派（2026-09-02，见 core/gpu/distributed/
        gpu_distributed_order_continuation.py 模块文档）：与 CPU
        `DistributedFRSolver.solve()` 同一个判据——`self.order`（目标
        阶数）>= 2 时自动改用逐阶爬坡（`run_distributed_order_
        continuation`，CPU/GPU 共用同一份迭代循环），不需要调用方显式
        请求。该函数返回 `SolverResult`（dataclass），这里适配转换成
        本方法一贯的 dict 返回约定，不改变调用方（CLI）已有的
        `result['final_residual']`/`result['iterations']` 访问方式。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 收敛容差
            output_interval: 输出间隔
            checkpoint_callback: 可选，`callback(solver, iteration)`，
                每步结束后调用一次

        Returns:
            结果字典
        """
        if getattr(self, 'order_continuation_enabled', True) and self.order >= 2:
            from autoflowcfd.core.mpi.distributed_order_continuation import (
                run_distributed_order_continuation,
            )
            result = run_distributed_order_continuation(
                self, max_iter, dt, tol, checkpoint_callback=checkpoint_callback,
                phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
            )
            return {
                'converged': result.converged,
                'iterations': result.iterations,
                'final_residual': result.final_residual,
            }

        if is_root():
            print(f"Starting multi-GPU solve: {self.n_ranks} ranks, max_iter={max_iter}")

        converged = False
        final_residual = 1e10
        _last_finite = None

        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res

            if is_root():
                if i == 0 or (i + 1) % output_interval == 0:
                    print(
                        f"Multi-GPU Iter {i+1}: Residual = {res:.6e} | "
                        f"Time/step: {t_end-t_start:.3f}s"
                    )

            # 发散检查必须在 checkpoint 回调**之前**（2026-09-16 修复）：
            # 此前顺序是先保存再检查，于是发散那一步的 NaN 状态会被如实
            # 写进 checkpoint。`res` 来自 `self.step()` 的全域 allreduce，
            # 各 rank 取值相同，所以全部 rank 会同时抛出、不会死锁。
            check_residual_finite(
                res, i + 1, last_finite=_last_finite,
                extra_hint=f"分布式路径：{self.n_ranks} 个 rank，"
                           f"残差是全域 allreduce 值（各 rank 一致）",
            )
            _last_finite = res

            if checkpoint_callback is not None:
                # 全部 rank 都要调用——保存需要每个 rank 各自贡献 local
                # cells 数据（见 CPU 分布式 solve 同一处注释）。
                checkpoint_callback(self, i + 1)

            if res < tol:
                converged = True
                if is_root():
                    print(f"✅ Multi-GPU Converged at iteration {i+1}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }
