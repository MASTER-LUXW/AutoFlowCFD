"""GPUFRSolver I/O 和湍流源项混入类。

从 gpu_solver.py 拆出，控制单文件行数。包含 checkpoint 保存/加载、
CPU↔GPU 状态传输、资源释放和湍流源项计算。
"""

import numpy as np
from loguru import logger

from autoflowcfd.core.gpu import get_cupy


class _GPUSolverIOMixin:
    """GPUFRSolver I/O 混入。

    子类需要提供：U_gpu, Q_gpu, array_mgr, mesh, iteration,
    residual_history, turb_model_gpu, _dual_time_U_prev 等属性。
    """

    def get_state_cpu(self):
        """将 GPU 状态下载回 CPU。"""
        return {
            'U': self.array_mgr.to_cpu(self.U_gpu),
            'Q': self.array_mgr.to_cpu(self.Q_gpu),
        }

    def set_state_from_cpu(self, U_np: np.ndarray):
        """从 CPU 设置求解器状态。"""
        self.U_gpu = self.array_mgr.to_gpu(U_np)
        self._update_primitives_gpu()

    def cleanup(self):
        """释放 GPU 资源。"""
        self.array_mgr.cleanup()

    def save_checkpoint(self, path: str):
        """保存 GPU 求解器状态到 checkpoint 文件。"""
        import h5py
        cp = get_cupy()

        U_cpu = cp.asnumpy(self.U_gpu)
        Q_cpu = cp.asnumpy(self.Q_gpu)

        with h5py.File(path, 'w') as f:
            f.create_dataset('U', data=U_cpu)
            f.create_dataset('Q', data=Q_cpu)
            f.attrs['iteration'] = self.iteration
            f.attrs['n_cells'] = self.mesh.n_cells
            f.attrs['n_sps'] = self.mesh.n_sps_per_cell
            f.attrs['order'] = self.order
            f.attrs['time_scheme'] = self.time_integrator.scheme
            f.attrs['cfl'] = self.time_integrator.cfl

            if self.residual_history:
                f.create_dataset('residual_history', data=np.array(self.residual_history))
            if self.turb_model_gpu is not None:
                f.create_dataset('k', data=cp.asnumpy(self.turb_model_gpu.k_field))
                f.create_dataset('omega', data=cp.asnumpy(self.turb_model_gpu.omega_field))
            if self._dual_time_U_prev is not None:
                f.create_dataset('U_prev', data=cp.asnumpy(self._dual_time_U_prev))

        logger.info(f"GPU checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """从 checkpoint 文件加载 GPU 求解器状态。"""
        import h5py
        cp = get_cupy()

        with h5py.File(path, 'r') as f:
            U_cpu = f['U'][:]
            Q_cpu = f['Q'][:]
            self.U_gpu = cp.asarray(U_cpu)
            self.Q_gpu = cp.asarray(Q_cpu)
            self.iteration = int(f.attrs['iteration'])

            if 'residual_history' in f:
                self.residual_history = f['residual_history'][:].tolist()
            if 'k' in f and 'omega' in f and self.turb_model_gpu is not None:
                self.turb_model_gpu.k_field = cp.asarray(f['k'][:])
                self.turb_model_gpu.omega_field = cp.asarray(f['omega'][:])
            if 'U_prev' in f:
                self._dual_time_U_prev = cp.asarray(f['U_prev'][:])

        logger.info(f"GPU checkpoint loaded from {path}, iteration={self.iteration}")

    def _update_production_ramp_gpu(self) -> None:
        """更新湍流产项渐变因子（GPU 版，与 CPU 侧
        fr_solver/turbulence.py::_update_production_ramp 同一机制）。

        第四次评审发现：GPU 路径的 `turb_model_gpu.production_factor`
        自身文档已承认"未接入渐变逻辑，恒为 1.0"——只靠 k_max/omega_max
        硬上限兜底，初始瞬态存在重新触发 CPU 侧已修复过的数值爆炸风险
        （CPU 侧修复正是本轮上一次提交"湍流强度产生项加渐变因子"）。
        前 N 步内 production_factor 从 0 线性增加到 1，防止初始流场
        未发展时 P_k >> D_k 导致 k/omega 指数爆炸。
        """
        if self.turb_model_gpu is None or not hasattr(self.turb_model_gpu, 'production_factor'):
            return
        ramp_steps = getattr(self, '_turb_production_ramp_steps', None)
        if ramp_steps is None:
            ramp_steps = 50  # 与 CPU 侧 init_turbulence_models 同一默认值
            self._turb_production_ramp_steps = ramp_steps
        current_step = getattr(self, '_turb_ramp_step', 0)
        if ramp_steps <= 0 or current_step >= ramp_steps:
            self.turb_model_gpu.production_factor = 1.0
            if not getattr(self, '_turb_production_ramp_complete', False):
                self._turb_production_ramp_complete = True
                logger.info(
                    f"[ProductionRamp][GPU] Ramp complete after {ramp_steps} steps, "
                    f"production_factor = 1.0"
                )
        else:
            self.turb_model_gpu.production_factor = current_step / ramp_steps
        self._turb_ramp_step = current_step + 1

    def compute_turbulence_source_gpu(self):
        """GPU 计算湍流模型源项。

        完整流程：
        1. 计算速度梯度（GPU）
        2. 计算 k/ω 的真实物理梯度（GPU）
        3. 使用预计算的壁面距离
        4. DDES/IDDES 长度尺度（可选，写入 turb_model_gpu.des_length_scale）
        5. 计算 SST 源项
        6. k/ω 完整输运（对流+扩散，#7 新增，见 gpu_scalar_transport.py）
        7. 更新 k/ω 场（含正性限制器）

        纯 WMLES/LES（无 SST 输运，`turb_model_gpu is None`）：mu_t 只
        来自 SGS 模型，读取上一步 `_apply_turbulence_corrections_gpu()`
        算出的 `sgs_model_gpu.nu_t`（一步滞后，与 CPU 版
        `apply_turbulence_corrections` 在 step() 末尾才更新 sgs_model.nu_t、
        供下一步粘性残差使用的操作分裂时序完全一致，见该方法调用点
        gpu_solver.py::step() 文档）。

        Returns:
            mu_t: 动力涡粘度 rho*nu_t (n_cells, n_sps) CuPy 数组，湍流模型
                与 SGS 模型都未激活时返回 None
        """
        cp = get_cupy()
        rho = self.Q_gpu[:, :, 0]

        if self.turb_model_gpu is None:
            if self.sgs_model_gpu is None or self.sgs_model_gpu.nu_t is None:
                return None
            return rho * self.sgs_model_gpu.nu_t

        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        self._update_production_ramp_gpu()

        from autoflowcfd.core.gpu.residual.gpu_gradients import (
            compute_physical_gradient_gpu,
            compute_physical_scalar_gradient_gpu,
        )
        # 真实 bug 修复（2026-09-03，与下面 296 行附近同一类，CPU 版
        # 见 fr_solver/turbulence.py::compute_turbulence_source 文档）：
        # 此前对*守恒*变量 U_gpu 求梯度再切片动量分量冒充速度梯度——
        # grad(rho*u) != rho*grad(u)，除非密度梯度处处为零。直接对
        # Q_gpu（原始变量，已经是真正的速度）求梯度。
        grad_vel = compute_physical_gradient_gpu(
            self.Q_gpu[..., 1:4], self.mesh_data, self.ops_data,
        )

        d_wall = self.wall_distance_gpu
        if d_wall is None:
            # 真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前静默
            # 回退到硬编码常量 0.01m，与 CPU 版 fr_solver/turbulence.py
            # 的既定原则矛盾（"Industrial-grade calculation requires
            # accurate wall distance, not simplified estimates"）。这个
            # 分支只在 SST 已激活（本方法只被 SST 生产路径调用）却没有
            # 壁面距离场时触发——正常初始化流程下 _init_wall_distance_gpu
            # 已在构造期设置好 wall_distance_gpu（见 gpu_solver.py 调用
            # 点），走到这里说明初始化被跳过或状态被破坏，直接失败而
            # 不是悄悄用一个和网格尺度无关的假常量继续算。
            raise RuntimeError(
                f"Wall distance field not computed for turbulence model "
                f"'{self.turb_model_name}'. Please ensure _init_wall_distance_gpu() "
                f"ran during solver initialization. Industrial-grade calculation "
                f"requires accurate wall distance, not simplified estimates."
            )

        grad_k = compute_physical_scalar_gradient_gpu(
            self.turb_model_gpu.k_field, self.mesh_data, self.ops_data,
        )
        grad_omega = compute_physical_scalar_gradient_gpu(
            self.turb_model_gpu.omega_field, self.mesh_data, self.ops_data,
        )

        max_grad_mag = 1e6
        grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
        if cp.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / cp.maximum(grad_k_mag, 1e-10)
            grad_k *= cp.clip(scale_k, 0, 1)[..., None]
        if cp.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / cp.maximum(grad_omega_mag, 1e-10)
            grad_omega *= cp.clip(scale_omega, 0, 1)[..., None]

        # 真实 bug 修复（2026-09-02，排查多GPU分布式SST时对照发现，与
        # 分布式本身无关，单机 GPU 路径同样中招，此前从未被端到端验证
        # 过）：`compute_source_terms_gpu` 的 `grad_U` 参数按文档/CPU版
        # `SSTModelFR.compute_source_terms` 同名参数实际期望的是**速度
        # 梯度**（(n_cells,n_sps,3,3)，内部 `compute_strain_rate_
        # magnitude_gpu` 直接对末两维做转置相加），不是 5 变量梯度
        # ((n_cells,n_sps,5,3))——此前这里传的是后者，
        # `compute_strain_rate_magnitude_gpu` 内部
        # `cp.transpose(grad_u,(0,1,3,2))` 会产出 (...,3,5) 与
        # (...,5,3) 无法广播相加，真实 CUDA 环境下必然 ValueError 崩溃。
        # 应该传上面已经算好的 `grad_vel`（2026-09-03 起直接对 Q_gpu 速度
        # 分量求梯度算出，不再是从 5 变量梯度切片，见上面 grad_vel 赋值
        # 处文档）。
        Sk, S_omega = self.turb_model_gpu.compute_source_terms_gpu(
            self.Q_gpu, grad_vel, d_wall, self.mu_molecular,
            grad_k, grad_omega,
        )

        # 真实 bug 修复（2026-09-02，排查"GPU侧DDES/IDDES是否有同类未
        # 发现的bug"时对照 CPU 版发现，与上面 grad_U/grad_vel 是两个
        # 独立的问题）：此前这里的 DDES/IDDES 长度尺度更新排在
        # `compute_source_terms_gpu` **之前**，注释还声称"与 CPU 版
        # 调用顺序完全一致"——但 CPU 版
        # `fr_solver_turbulence.py::compute_turbulence_source` 的真实
        # 顺序是反的：`compute_source_terms`（用*上一步*遗留的
        # `des_length_scale`，写死为"慢一拍"耦合，见 CPU 版 turbulence.py
        # "DDES 的有效长度尺度"一节注释的完整推导——`apply_to_sst_model`
        # 依赖当前 nu_t，而 nu_t 只有 `compute_source_terms` 算完才是
        # 当前值，两者互相依赖对方的输出，只能先后错开一步，且顺序不能
        # 颠倒）在前，`apply_to_sst_model[_iddes]`（为*下一步*写入新的
        # `des_length_scale`）在后。此前 GPU 版顺序颠倒，等价于让
        # DDES/IDDES 长度尺度提前一步生效（"快一拍"耦合，且用的还是
        # `apply_to_sst_model_gpu` 内部读取的 `nu_t`——此时 `nu_t` 还是
        # 上一次调用（或初始值）遗留的，语义上更混乱），与 CPU 版数值
        # 结果不一致——用真实非均匀状态实测：DDES 在这份测试数据下巧合
        # 数值一致（f_d 恰好被推到掩盖差异的区间），IDDES 上实测出
        # 2.2% 相对差异，暴露了这个顺序错误。现改为与 CPU 版逐字一致
        # 的顺序：`compute_source_terms_gpu` 在前，DDES/IDDES 长度尺度
        # 更新在后。
        if self.ddes_model_gpu is not None:
            nu_field = self.mu_molecular / cp.maximum(rho, 1e-10)
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUIDDESModel
            if isinstance(self.ddes_model_gpu, GPUIDDESModel):
                self.ddes_model_gpu.apply_to_sst_model_iddes_gpu(
                    self.turb_model_gpu, d_wall, self._iddes_h_max_gpu, self._iddes_h_wn_gpu,
                    nu_field, grad_vel,
                )
            else:
                cell_volumes = self.mesh_data.get('cell_volumes')
                if cell_volumes is None:
                    cell_volumes = cp.asarray(self.mesh.get_all_cell_volumes())
                self.ddes_model_gpu.apply_to_sst_model_gpu(
                    self.turb_model_gpu, d_wall, cell_volumes, nu_field, grad_vel,
                    h_max=getattr(self, '_iddes_h_max_gpu', None),
                )

        dk_dt = Sk / cp.maximum(rho, 1e-10)
        domega_dt = S_omega / cp.maximum(rho, 1e-10)

        # k/omega 完整输运（对流+扩散，#7 新增）：真正补齐 GPU SST 长期
        # 缺失的输运项——此前 update_fields_gpu 的 transport_k/
        # transport_omega 参数从未被调用方传入（见 gpu_turbulence_sst.py
        # 模块文档），k/omega 场只靠逐点源项 ODE 弛豫，没有跨单元对流/
        # 扩散。与 CPU 版 fr_solver_turbulence.py::compute_turbulence_source
        # 同一个触发条件（SST/DDES/IDDES 都需要）。
        transport_k = None
        transport_omega = None
        if self.turb_model_name.upper() in ("SST", "DDES", "IDDES"):
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
                compute_turbulence_transport_residual_gpu,
            )
            transport_k, transport_omega = compute_turbulence_transport_residual_gpu(
                self, grad_vel=grad_vel,
            )

        # 湍流标量必须用**物理**波速算出的那一份 dt（2026-09-14，低马赫数
        # 预处理接入 GPU 时同步）：启用预处理后平均流的 dt 按预处理波速
        # 放大约 7 倍，而 k/omega 的显式更新刻意没有做 point-implicit
        # 阻尼（见 turbulence/sst.py::update_fields 文档），不能跟着放大。
        # 与 CPU 侧 step.py 里 `turb_dt = dt_physical` 同一处理。
        _, dt_physical = self._compute_local_time_step_gpu(return_physical_too=True)
        dt_mean = cp.mean(dt_physical)
        self.turb_model_gpu.update_fields_gpu(
            float(dt_mean), dk_dt, domega_dt,
            transport_k=transport_k, transport_omega=transport_omega,
        )

        # 真实 bug 修复（2026-09-12，与 CPU 版
        # fr_solver/turbulence.py::compute_turbulence_source 同一处修复，
        # 完整推导见 gpu_modal_filter.py::filter_scalar_field_gpu 文档）：
        # k/omega 场同样需要模态滤波，理由/CPU-GPU一致性要求同上。
        if self.mesh.n_sps_per_cell > 1:
            from autoflowcfd.core.gpu.gpu_modal_filter import filter_scalar_field_gpu
            n_prism = self.mesh.n_prism_cells
            self.turb_model_gpu.k_field = filter_scalar_field_gpu(
                self.turb_model_gpu.k_field, n_prism, self.ops.filter_prism, self.ops.filter_tet,
            )
            self.turb_model_gpu.omega_field = filter_scalar_field_gpu(
                self.turb_model_gpu.omega_field, n_prism, self.ops.filter_prism, self.ops.filter_tet,
            )
            self.turb_model_gpu.apply_positivity_limiter_gpu()

        # 真实缺口修复（2026-09-05，代码复审发现）：CPU 版
        # fr_solver/turbulence.py::compute_turbulence_source 在
        # update_fields 之后调用 enforce_omega_wall_relaxation 修补
        # omega 壁面扩散侧未闭合的架构缺口（见该处文档完整推导），
        # GPU 版此前完全没有移植这一步——GPU SST/DDES/IDDES 长期运行
        # 会重现与 CPU 版修复前完全相同的中长期发散机制（边界层 omega
        # 衰减到下界 -> nu_t 近零分母奇点 -> 湍流粘性比失控）。
        if self.turb_model_name.upper() in ("SST", "DDES", "IDDES"):
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
                enforce_omega_wall_relaxation_gpu,
            )
            enforce_omega_wall_relaxation_gpu(cp, self)

        mu_t = rho * self.turb_model_gpu.nu_t
        if self.sgs_model_gpu is not None and self.sgs_model_gpu.nu_t is not None:
            mu_t = mu_t + rho * self.sgs_model_gpu.nu_t
        return mu_t

    def _apply_turbulence_corrections_gpu(self):
        """GPU 版 SGS（WALE）涡粘系数更新，与 CPU 版
        `fr_solver_turbulence.py::apply_turbulence_corrections` 完全同一个
        操作分裂时序：必须在 step() 里状态更新（`self.U_gpu = U_new_flat`+
        `_update_primitives_gpu()`）**之后**调用（见 gpu_solver.py::step()
        调用点），算出的 nu_t 供下一步 `compute_turbulence_source_gpu()`
        读取——不能提前到状态更新之前，那样用的是上上一步的状态，语义上
        更旧一步，且与 CPU 版行为不一致。

        无 sgs_model_gpu（NONE/SST/DDES/IDDES 场景）时是 no-op。
        """
        if self.sgs_model_gpu is None:
            return
        cp = get_cupy()
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu

        # 真实 bug 修复（2026-09-03）：同上面 156 行附近 compute_turbulence_
        # source_gpu 里的 grad_vel 修复，理由见该处文档——LES/WMLES 的
        # SGS 涡粘同样不能用动量梯度冒充速度梯度。
        grad_vel = compute_physical_gradient_gpu(
            self.Q_gpu[..., 1:4], self.mesh_data, self.ops_data,
        )
        nu_t = self.sgs_model_gpu.compute_eddy_viscosity_gpu(grad_vel, self._grid_scale_gpu)

        if self.turb_model_gpu is not None and hasattr(self.turb_model_gpu, "nu_t"):
            self.turb_model_gpu.nu_t = self.turb_model_gpu.nu_t + nu_t
