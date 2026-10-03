"""AutoFlowCFD V2.0 - 多 GPU 分布式湍流源项（SST/DDES/IDDES/LES/WMLES）

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_init.py` 的 `_GPUDistributedInitMixin` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `_GPUDistributedInitMixin` 的 `__init__` 建立，
这里通过 `self` 访问。

## 结构：prepare / evaluate / finalize / write-back

与单机 CPU（`fr_solver/turbulence/source.py`）、单机 GPU（`gpu_solver_io.py`）
同一个拆分（2026-09-25）：显式更新 = prepare -> evaluate -> update_fields ->
finalize -> write-back；隐式 k-omega（`gpu_distributed_implicit.py` 的适配器）
在一个 Newton 步内冻结 prepare 的平均流输入，反复 evaluate。

compact 视图（`_TurbulenceView`）：k/omega 是 local 大小的持久状态，而源项与
输运要读 halo 单元，所以每次求值都经 2 变量 halo 交换写进一个 compact 大小的
临时 `GPUTurbulenceSST`，结果只把 local 段写回真正的模型——与 CPU 分布式
`core/mpi/distributed_turbulence.py` 同一个设计。
"""

import types

from autoflowcfd.core.gpu import get_cupy


class _GPUDistributedTurbSourceMixin:
    """多 GPU 分布式湍流源项（SST/DDES/IDDES/LES/WMLES）"""

    def _compute_turbulence_source_distributed(self, dt):
        """分布式湍流源项+输运的显式更新，返回 compact 排列的动力涡粘
        `(n_compact, n_sps)`（供粘性残差的 BR1 界面项），无湍流模型时 None。

        Args:
            dt: k/omega 显式更新的步长，与单机 CPU/GPU 同一规则：DUAL_TIME 下是
                物理时间步（标量），其余是按**物理**波速算出的逐单元局部步长，
                **紧凑排列** `(n_compact, 1)`（`_compute_local_time_step_gpu`
                的第二个返回值），与 compact 视图逐点对齐。2026-09-25 以前
                调用方传的是全场均值（单机 GPU 同一缺陷，见
                `gpu_solver_io.py::compute_turbulence_source_gpu` 文档）。
        """
        if self.turb_model_gpu is None:
            if self.sgs_model_gpu is None:
                return None
            return self._les_mu_t_compact()

        from autoflowcfd.core.fr_solver.turbulence.init import advance_production_ramp

        advance_production_ramp(self, self.turb_model_gpu)
        ctx = self._prepare_turbulence_view_distributed()
        self._sync_turbulence_view(ctx)
        dk_dt, dw_dt, transport_k, transport_w = self._evaluate_turbulence_rates_distributed(
            ctx, apply_des=True)
        # 场更新（k 与 w = ln omega 的点隐式阻尼 + 输运，见 sst/update.py::advance_k_log_omega）
        ctx.view.update_fields_gpu(dt, dk_dt, dw_dt, transport_k=transport_k, transport_log_omega=transport_w)
        self._finalize_turbulence_update_distributed(ctx)
        self._write_back_turbulence_distributed(ctx, fields=True)
        return ctx.rho * ctx.view.nu_t

    def _turbulence_velocity_gradient_compact(self, Q_compact):
        """紧凑空间的湍流模型速度梯度（单机 `_turbulence_velocity_gradient_gpu` 的多 GPU
        对应，同一份算法）。P0 的提升修正梯度在 halo 行上不完整（halo 单元外侧的面不在
        本 rank），经 `mpi/compact_halo.py`（与 CPU 分布式同一个类）取所属 rank 的值。"""
        from autoflowcfd.core.gpu.residual.gpu_corrected_gradient import source_velocity_gradient_gpu
        from autoflowcfd.core.mpi.compact_halo import CompactHaloRefresh

        return source_velocity_gradient_gpu(
            get_cupy(), Q_compact, self.mesh_data, self.ops_data, self.flat_face_gpu,
            self.dist_flat_face.base_flat, self.boundary_ghost_provider, self.device_id,
            halo_refresh=CompactHaloRefresh(self.gpu_halo, self._perm_gpu, self._inv_perm_gpu,
                                            self.partition.n_local_cells))

    def _les_mu_t_compact(self):
        """纯 LES（WALE）：代数模型、没有跨步状态，直接用当前（halo 交换后的
        compact）速度场现算 mu_t。与单机版在 step() 末尾缓存、供下一步使用
        在数学上是同一个值（WALE 不依赖历史状态）。"""
        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu

        U_compact = self._permute_to_compact(self.gpu_halo.exchange(self.U_gpu))
        Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
        grad_vel = self._turbulence_velocity_gradient_compact(Q_compact)
        nu_t_compact = self.sgs_model_gpu.compute_eddy_viscosity_gpu(grad_vel, self._grid_scale_compact)
        return Q_compact[:, :, 0] * nu_t_compact

    def _prepare_turbulence_view_distributed(self):
        """一步之内只依赖平均流的部分：平均流 halo 交换、compact 原始变量与
        速度梯度、壁距，以及 compact 视图本身。返回上下文对象 `ctx`。"""
        cp = get_cupy()
        n_sps = self.mesh.n_sps_per_cell
        n_compact = self.n_compact

        # 平均流 halo 交换 + 重排到 compact 索引空间（与无粘残差同一处理）
        U_compact = self._permute_to_compact(self.gpu_halo.exchange(self.U_gpu))

        # compact 视图。k_inf/omega_inf 不只是初值：开边界来流 ghost 取它们
        # 作为来流值（gpu_scalar_transport/residual.py），必须与真正的模型一致
        # ——2026-09-25 前视图用构造默认值 1e-6/1.0（CPU 分布式同一缺陷）
        from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST

        model = self.turb_model_gpu
        view = GPUTurbulenceSST(n_compact, n_sps, self.device_id,
                                k_inf=model.k_inf, omega_inf=model.omega_inf)
        for attr in (
            "sigma_k1", "sigma_k2", "sigma_w1", "sigma_w2", "beta1", "beta2",
            "a1", "kappa", "beta_star", "k_max", "omega_max", "production_factor",
        ):
            if hasattr(model, attr):
                setattr(view, attr, getattr(model, attr))

        # DDES/IDDES 的 des_length_scale 是跨步持久状态（上一步用上一步 nu_t
        # 算出、本步读取），与 k/omega 同一套 halo 交换 + compact 重排，不能把
        # n_local 大小的数组直接挂到 compact 视图上（2026-09-02 修复）
        view.des_length_scale = None
        if self.ddes_model_gpu is not None and getattr(model, "des_length_scale", None) is not None:
            view.des_length_scale = self._permute_to_compact(self.gpu_halo.exchange(model.des_length_scale))

        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu

        Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
        grad_vel = self._turbulence_velocity_gradient_compact(Q_compact)

        d_wall = self.wall_distance_gpu
        if d_wall is None:
            raise RuntimeError(
                f"Rank {self.rank}: 湍流模型 '{getattr(self, 'turb_model_name', '?')}' 需要"
                f"壁面距离，但 _init_wall_distance_distributed() 没有算出——不退化为估计值。")

        # 输运残差与壁面函数读取的最小鸭子类型（mesh 只需要 n_cells/
        # n_sps_per_cell/n_prism_cells/order；`dict.get(key, default)` 的默认值表达式
        # 会被无条件求值，所以 n_prism_cells 必须给；order 是扩散内罚常数的阶数，
        # 与平均流粘性残差 `compute_viscous_residual_gpu` 取同一处 `self.mesh.order`）
        transport = types.SimpleNamespace(
            turb_model_gpu=view,
            mesh=types.SimpleNamespace(
                n_cells=n_compact, n_sps_per_cell=n_sps,
                n_prism_cells=self.dist_flat_face.base_flat.n_prism,
                order=int(self.mesh.order)),
            Q_gpu=Q_compact, U_gpu=U_compact, mu_molecular=self.mu_molecular,
            mesh_data=self.mesh_data, ops_data=self.ops_data,
            flat_face_gpu=self.flat_face_gpu, wall_distance_gpu=d_wall,
            _wall_mask_k_gpu=self._wall_mask_k_gpu, _open_mask_gpu=self._open_mask_gpu,
        )
        return types.SimpleNamespace(view=view, Q=Q_compact, rho=Q_compact[:, :, 0],
                                     grad_vel=grad_vel, d_wall=d_wall, transport=transport,
                                     cp=cp)

    def _sync_turbulence_view(self, ctx) -> None:
        """把真正模型（local）当前的 k/omega 经 2 变量 halo 交换写进 compact 视图。"""
        cp = ctx.cp
        k_omega_local = cp.stack([self.turb_model_gpu.k_field, self.turb_model_gpu.omega_field],
                                 axis=-1)
        k_omega_compact = self._permute_to_compact(self.gpu_halo.exchange(k_omega_local))
        ctx.view.k_field = k_omega_compact[..., 0].copy()
        ctx.view.omega_field = k_omega_compact[..., 1].copy()

    def _evaluate_turbulence_rates_distributed(self, ctx, *, apply_des: bool):
        """在 compact 视图当前的 k/omega 上求 k 与 `w = ln(omega)` 的 `(dk/dt, dw/dt,
        transport_k, transport_w)`（compact 排列；源项部分已除以 rho，与单机
        `gpu_solver_io.py::_evaluate_turbulence_rates_gpu` 同一套变换）。

        副作用同单机：刷新视图上的 nu_t / 混合 beta；`apply_des=True` 时按刚算出
        的 nu_t 刷新 DES 长度尺度（供下一步用）——隐式路径的试探求值必须传 False。
        """
        cp = ctx.cp
        view = ctx.view
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu

        from autoflowcfd.core.turbulence.sst.bounds import clip_gradient_magnitude
        from autoflowcfd.core.turbulence.sst.log_omega import log_omega

        # 梯度对被输运的 k 与 w = ln(omega) 求；模型项用 grad(omega) = omega grad(w)
        omega = view.omega_field
        grad_k = clip_gradient_magnitude(
            compute_physical_scalar_gradient_gpu(view.k_field, self.mesh_data, self.ops_data), cp)
        grad_w = clip_gradient_magnitude(
            compute_physical_scalar_gradient_gpu(log_omega(omega, cp), self.mesh_data, self.ops_data), cp)
        grad_omega = omega[:, :, None] * grad_w

        Sk, S_omega = view.compute_source_terms_gpu(
            ctx.Q, ctx.grad_vel, ctx.d_wall, self.mu_molecular, grad_k, grad_omega)

        # DDES/IDDES 长度尺度必须排在源项**之后**（"慢一拍"耦合：源项用上一步
        # 的长度尺度，算完再用本步 nu_t 写下一步要用的，见单机 gpu_solver_io.py）
        if apply_des and self.ddes_model_gpu is not None:
            nu_field = self.mu_molecular / cp.maximum(ctx.rho, 1e-10)
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUIDDESModel
            if isinstance(self.ddes_model_gpu, GPUIDDESModel):
                self.ddes_model_gpu.apply_to_sst_model_iddes_gpu(
                    view, ctx.d_wall, self.iddes_h_max_compact, self.iddes_h_wn_compact,
                    nu_field, ctx.grad_vel)
            else:
                self.ddes_model_gpu.apply_to_sst_model_gpu(
                    view, ctx.d_wall, self.mesh_data.get('cell_volumes'), nu_field, ctx.grad_vel,
                    h_max=getattr(self, 'iddes_h_max_compact', None))

        dk_dt = Sk / cp.maximum(ctx.rho, 1e-10)
        dw_dt = S_omega / (cp.maximum(ctx.rho, 1e-10) * omega)

        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
            compute_turbulence_transport_residual_gpu,
        )
        transport_k, transport_w = compute_turbulence_transport_residual_gpu(
            ctx.transport, grad_k=grad_k, grad_log_omega=grad_w)
        return dk_dt, dw_dt, transport_k, transport_w

    def _finalize_turbulence_update_distributed(self, ctx) -> None:
        """k/omega 更新之后的后处理（compact 视图上）：模态滤波 + 正性限幅。与单机
        `finalize_turbulence_update` 同一份；壁面 omega 只由扩散残差的面 Dirichlet
        施加，不做步后松弛（2026-10-01 删除，理由见单机同名函数文档）。"""
        view = ctx.view
        n_sps = self.mesh.n_sps_per_cell
        if n_sps > 1:
            # k 与 w = ln(omega) 的模态滤波（与单机同一份，`GPUTurbulenceSST.filter_fields_gpu`）；
            # compact 排列"棱柱在前"
            order = int(getattr(self, "current_order", getattr(self, "order", 0)))
            frac = view.filter_fields_gpu(self.flat_face_gpu.n_prism, self.ops, order)
            if frac is not None:
                self._turb_filter_troubled_frac = frac


    def _write_back_turbulence_distributed(self, ctx, *, fields: bool) -> None:
        """compact 视图的结果换回原生排列、切 local 段写回真正的模型：nu_t 与
        DES 长度尺度总是写；`fields=True` 时连 k/omega 一起写（隐式路径的 k/omega
        由 Newton 步直接更新在模型上，只在 finalize 之后写回）。"""
        n_local = self.partition.n_local_cells
        model = self.turb_model_gpu
        view = ctx.view
        if fields:
            model.k_field[:] = self._unpermute_from_compact(view.k_field)[:n_local]
            model.omega_field[:] = self._unpermute_from_compact(view.omega_field)[:n_local]
        model.nu_t[:] = self._unpermute_from_compact(view.nu_t)[:n_local]
        if self.ddes_model_gpu is not None and getattr(view, "des_length_scale", None) is not None:
            model.des_length_scale = self._unpermute_from_compact(view.des_length_scale)[:n_local].copy()
