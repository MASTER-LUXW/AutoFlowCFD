"""GPUFRSolver I/O 和湍流源项混入类。

从 gpu_solver.py 拆出，控制单文件行数。包含资源释放和湍流源项计算（主机<->设备的状态读写见
`host_view.py`）。
"""

from autoflowcfd.core.gpu import get_cupy


class _GPUSolverIOMixin:
    """GPUFRSolver I/O 混入。

    子类需要提供：U_gpu, Q_gpu, array_mgr, mesh, iteration,
    residual_history, turb_model_gpu, _dual_time_U_prev 等属性。
    """

    def cleanup(self):
        """释放 GPU 资源。"""
        self.array_mgr.cleanup()

    def _update_production_ramp_gpu(self) -> None:
        """湍流产生项渐变（全部后端同一份，见
        `fr_solver/turbulence/init.py::advance_production_ramp`）。"""
        from autoflowcfd.core.fr_solver.turbulence.init import advance_production_ramp

        advance_production_ramp(self, self.turb_model_gpu)

    def compute_turbulence_source_gpu(self, turb_dt, physical_time=None, advance_ramp: bool = True):
        """GPU 上湍流场的一次更新（参数含义与 CPU `fr_solver/turbulence/source.py::compute_turbulence_source`
        相同）。

        Args:
            turb_dt: 一次更新的（伪）时间步长：按**物理**波速算出的逐单元局部步长 `(n_cells, 1)`（不跟
                低马赫预处理放大）。2026-09-25 以前这里自己取 `cp.mean(dt_physical)`：全场一个平均
                步长——小单元拿到超过自身稳定限的步长、大单元走得慢。
            physical_time: DUAL_TIME 的物理时间项（`core/turbulence/dual_time.py`），其余格式 None
            advance_ramp: 是否推进产生项渐变计数（DUAL_TIME 内迭代只有第一次推进）

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
        rho = self.Q_gpu[:, :, 0]

        if self.turb_model_gpu is None:
            if self.sgs_model_gpu is None or self.sgs_model_gpu.nu_t is None:
                return None
            return rho * self.sgs_model_gpu.nu_t

        if advance_ramp:
            self._update_production_ramp_gpu()

        grad_vel, d_wall = self._prepare_turbulence_inputs_gpu()
        rates = self._evaluate_turbulence_rates_gpu(grad_vel, d_wall, apply_des=True)

        self.turb_model_gpu.update_fields(turb_dt, rates.source, rates.transport)
        if physical_time is not None:
            physical_time.apply(self.turb_model_gpu, get_cupy(), turb_dt)

        self._finalize_turbulence_update_gpu()
        return self._turbulent_mu_t_gpu()

    def _turbulence_velocity_gradient_gpu(self):
        """湍流模型（SST/DDES/IDDES 源项、LES 亚格子涡粘）用的速度梯度（CPU 版
        `fr_solver/turbulence/source.py::turbulence_velocity_gradient` 的 GPU 对应：
        P0 提升修正、P>=1 单元内导数，见 `gpu/residual/gpu_corrected_gradient.py`）。"""
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.gpu.residual.gpu_corrected_gradient import source_velocity_gradient_gpu
        return source_velocity_gradient_gpu(
            get_cupy(), self.Q_gpu, self.mesh_data, self.ops_data, self.flat_face_gpu,
            get_flat_face_geometry(self.mesh, self.ops), self.boundary_ghost_provider, self.device_id)

    def _prepare_turbulence_inputs_gpu(self):
        """一步之内只依赖平均流的输入 `(grad_vel, d_wall)`（CPU 版
        `source.py::prepare_turbulence_inputs` 的 GPU 对应）。"""
        grad_vel = self._turbulence_velocity_gradient_gpu()

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

        return grad_vel, d_wall

    def _evaluate_turbulence_rates_gpu(self, grad_vel, d_wall, *, apply_des: bool):
        """在 `turb_model_gpu` 当前的 `k_field/omega_field` 上求 k 与 `w = ln(omega)` 的
        源项部分与输运部分 `(dk_dt, dw_dt, transport_k, transport_w)`（CPU 版
        `source.py::evaluate_turbulence_rates` 的 GPU 对应，同一组副作用约定：
        刷新模型上的 `nu_t` 等缓存；`apply_des=False` 时不改写 DES 长度尺度）。"""
        from autoflowcfd.core.turbulence.registry import is_sa_family

        if is_sa_family(self.turb_model_name):
            # 与 CPU 同一份算法（`core/turbulence/sa/rates.py`），注入 GPU 的标量输运原语
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport.primitives import GpuScalarTransport
            from autoflowcfd.core.turbulence.sa.rates import evaluate_sa_rates

            return evaluate_sa_rates(self.turb_model_gpu, GpuScalarTransport(self), self.Q_gpu, grad_vel,
                                     d_wall, self.mu_molecular)

        # 以下为 SST 族（SST/DDES/IDDES）
        cp = get_cupy()
        rho = self.Q_gpu[:, :, 0]
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu
        from autoflowcfd.core.turbulence.limits import clip_gradient_magnitude
        from autoflowcfd.core.turbulence.sst.log_omega import log_omega

        # 梯度对 k 与 w = ln(omega) 求（被输运的量），模长上限只作用在这两者上；模型项
        # 用物理梯度 grad(omega) = omega grad(w)（与 CPU 版同一处）
        omega = self.turb_model_gpu.omega_field
        grad_k = clip_gradient_magnitude(compute_physical_scalar_gradient_gpu(
            self.turb_model_gpu.k_field, self.mesh_data, self.ops_data), cp)
        grad_w = clip_gradient_magnitude(compute_physical_scalar_gradient_gpu(
            log_omega(omega, cp), self.mesh_data, self.ops_data), cp)
        grad_omega = omega[:, :, None] * grad_w

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
        if apply_des and self.ddes_model_gpu is not None:
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
        # w = ln(omega) 方程的源项部分（omega 由 exp(w) 产生、恒为正）
        dw_dt = S_omega / (cp.maximum(rho, 1e-10) * omega)

        # k/omega 完整输运（对流+扩散，#7 新增）：真正补齐 GPU SST 长期
        # 缺失的输运项——此前 GPU 显式更新的 transport_k/
        # transport_omega 参数从未被调用方传入（见 gpu_turbulence_sst.py
        # 模块文档），k/omega 场只靠逐点源项 ODE 弛豫，没有跨单元对流/
        # 扩散。与 CPU 版 fr_solver_turbulence.py::compute_turbulence_source
        # 同一个触发条件（SST/DDES/IDDES 都需要）。
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
            compute_turbulence_transport_residual_gpu,
        )
        transport_k, transport_w = compute_turbulence_transport_residual_gpu(
            self, grad_k=grad_k, grad_log_omega=grad_w,
        )

        from autoflowcfd.core.turbulence.transported import TurbulenceRates
        return TurbulenceRates((Sk, S_omega), (dk_dt, dw_dt), (transport_k, transport_w))

    def _finalize_turbulence_update_gpu(self):
        """`k/omega` 更新之后的后处理：模态滤波 + 正性限幅（非恒等滤波时）。显式与
        隐式路径共用。壁面 omega 只由扩散残差的面 Dirichlet 施加，不做步后松弛
        （2026-10-01 删除，理由见 CPU `fr_solver/turbulence/source.py::
        finalize_turbulence_update` 文档）。"""
        # 真实 bug 修复（2026-09-12，与 CPU 版
        # fr_solver/turbulence.py::compute_turbulence_source 同一处修复，
        # 完整推导见 gpu_modal_filter.py::filter_scalar_field_gpu 文档）：
        # k/omega 场同样需要模态滤波，理由/CPU-GPU一致性要求同上。
        # 与 CPU 同一份算法（`core/turbulence/unknown_filter.py`），作用在 Newton 未知量上
        from autoflowcfd.core.gpu.gpu_modal_filter import GpuFilterKernels
        from autoflowcfd.core.turbulence.unknown_filter import filter_turbulence_unknowns

        _order = int(getattr(self, "current_order", None) or getattr(self, "order", 0))
        frac = filter_turbulence_unknowns(self.turb_model_gpu, GpuFilterKernels(), self.mesh.n_prism_cells,
                                          _order, self.ops)
        if frac is not None:
            self._turb_filter_troubled_frac = frac

    def _turbulent_mu_t_gpu(self):
        """当前湍流场对应的动力涡粘 `rho*nu_t`（含 SGS 部分）。"""
        rho = self.Q_gpu[:, :, 0]
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
        # 与 SST 源项同一份速度梯度（原始变量速度 + 提升修正）
        nu_t = self.sgs_model_gpu.compute_eddy_viscosity_gpu(self._turbulence_velocity_gradient_gpu(),
                                                             self._grid_scale_gpu)

        if self.turb_model_gpu is not None and hasattr(self.turb_model_gpu, "nu_t"):
            self.turb_model_gpu.nu_t = self.turb_model_gpu.nu_t + nu_t
