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
        dk_dt, domega_dt, transport_k, transport_omega = self._evaluate_turbulence_rates_distributed(
            ctx, apply_des=True)
        # 场更新（点隐式阻尼 + 输运，见 update_fields_gpu 文档）
        ctx.view.update_fields_gpu(dt, dk_dt, domega_dt,
                                   transport_k=transport_k, transport_omega=transport_omega)
        self._finalize_turbulence_update_distributed(ctx, omega_wall_relaxation=True)
        self._write_back_turbulence_distributed(ctx, fields=True)
        return ctx.rho * ctx.view.nu_t

    def _les_mu_t_compact(self):
        """纯 LES（WALE）：代数模型、没有跨步状态，直接用当前（halo 交换后的
        compact）速度场现算 mu_t。与单机版在 step() 末尾缓存、供下一步使用
        在数学上是同一个值（WALE 不依赖历史状态）。"""
        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu

        U_compact = self._permute_to_compact(self.gpu_halo.exchange(self.U_gpu))
        Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
        # 速度梯度对原始变量求（不是动量梯度，2026-09-03 修复）
        grad_vel = compute_physical_gradient_gpu(Q_compact[..., 1:4], self.mesh_data, self.ops_data)
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
            des_len_extended = self.des_length_scale_halo_gpu.exchange(
                model.des_length_scale[:, :, None])[:, :, 0]
            view.des_length_scale = self._permute_to_compact(des_len_extended)

        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu

        Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
        # `compute_source_terms_gpu` 要的是速度梯度 (n,n_sps,3,3)，对原始变量的
        # 速度分量求（对守恒变量求梯度再切动量冒充速度梯度是 2026-09-03 修过的缺陷）
        grad_vel = compute_physical_gradient_gpu(Q_compact[..., 1:4], self.mesh_data, self.ops_data)

        d_wall = self.wall_distance_gpu
        if d_wall is None:
            raise RuntimeError(
                f"Rank {self.rank}: 湍流模型 '{getattr(self, 'turb_model_name', '?')}' 需要"
                f"壁面距离，但 _init_wall_distance_distributed() 没有算出——不退化为估计值。")

        # 输运残差与壁面函数读取的最小鸭子类型（mesh 只需要 n_cells/
        # n_sps_per_cell/n_prism_cells；`dict.get(key, default)` 的默认值表达式
        # 会被无条件求值，所以 n_prism_cells 必须给）
        transport = types.SimpleNamespace(
            turb_model_gpu=view,
            mesh=types.SimpleNamespace(
                n_cells=n_compact, n_sps_per_cell=n_sps,
                n_prism_cells=self.dist_flat_face.base_flat.n_prism),
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
        k_omega_compact = self._permute_to_compact(self.turb_halo_gpu.exchange(k_omega_local))
        ctx.view.k_field = k_omega_compact[..., 0].copy()
        ctx.view.omega_field = k_omega_compact[..., 1].copy()

    def _evaluate_turbulence_rates_distributed(self, ctx, *, apply_des: bool):
        """在 compact 视图当前的 k/omega 上求 `(dk/dt, domega/dt, transport_k,
        transport_omega)`（compact 排列；源项部分已除以 rho）。

        副作用同单机：刷新视图上的 nu_t / 混合 beta；`apply_des=True` 时按刚算出
        的 nu_t 刷新 DES 长度尺度（供下一步用）——隐式路径的试探求值必须传 False。
        """
        cp = ctx.cp
        view = ctx.view
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu

        grad_k = compute_physical_scalar_gradient_gpu(view.k_field, self.mesh_data, self.ops_data)
        grad_omega = compute_physical_scalar_gradient_gpu(view.omega_field, self.mesh_data, self.ops_data)
        max_grad_mag = 1e6
        grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
        if cp.any(grad_k_mag > max_grad_mag):
            grad_k *= cp.clip(max_grad_mag / cp.maximum(grad_k_mag, 1e-10), 0, 1)[..., None]
        if cp.any(grad_omega_mag > max_grad_mag):
            grad_omega *= cp.clip(max_grad_mag / cp.maximum(grad_omega_mag, 1e-10), 0, 1)[..., None]

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
        domega_dt = S_omega / cp.maximum(ctx.rho, 1e-10)

        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
            compute_turbulence_transport_residual_gpu,
        )
        transport_k, transport_omega = compute_turbulence_transport_residual_gpu(
            ctx.transport, grad_vel=ctx.grad_vel)
        return dk_dt, domega_dt, transport_k, transport_omega

    def _finalize_turbulence_update_distributed(self, ctx, *, omega_wall_relaxation: bool) -> None:
        """k/omega 更新之后的后处理（compact 视图上）：模态滤波 + 正性限幅，
        以及（显式路径）omega 壁面松弛。与单机 `finalize_turbulence_update`
        同一顺序、同一开关语义（隐式路径把壁面条件放进残差，不能再投影）。"""
        cp = ctx.cp
        view = ctx.view
        n_sps = self.mesh.n_sps_per_cell
        if n_sps > 1:
            # k/omega 与平均流同一套模态滤波（2026-09-12）；compact 排列"棱柱在前"
            from autoflowcfd.core.fr_solver.filter import resolve_turb_filter_gate
            from autoflowcfd.core.gpu.gpu_modal_filter import (
                filter_scalar_field_gated_gpu, filter_scalar_field_gpu,
            )
            n_prism_compact = self.flat_face_gpu.n_prism
            if resolve_turb_filter_gate() == "sensor":
                from autoflowcfd.core.gpu.gpu_troubled_cell import compute_turb_troubled_mask_gpu

                order = int(getattr(self, "current_order", getattr(self, "order", 0)))
                troubled = compute_turb_troubled_mask_gpu(
                    view.k_field, view.omega_field, n_prism_compact, order)
                self._turb_filter_troubled_frac = float(cp.mean(troubled))
                view.k_field = filter_scalar_field_gated_gpu(
                    view.k_field, n_prism_compact, self.ops.filter_prism, self.ops.filter_tet, troubled)
                view.omega_field = filter_scalar_field_gated_gpu(
                    view.omega_field, n_prism_compact, self.ops.filter_prism, self.ops.filter_tet,
                    troubled)
            else:
                view.k_field = filter_scalar_field_gpu(
                    view.k_field, n_prism_compact, self.ops.filter_prism, self.ops.filter_tet)
                view.omega_field = filter_scalar_field_gpu(
                    view.omega_field, n_prism_compact, self.ops.filter_prism, self.ops.filter_tet)
            view.apply_positivity_limiter_gpu()

        # omega 壁面 Wilcox 解析值的扩散侧闭合（2026-09-05），显式路径专用
        if omega_wall_relaxation and getattr(self, "turb_model_name", "").upper() in ("SST", "DDES", "IDDES"):
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
                enforce_omega_wall_relaxation_gpu,
            )
            enforce_omega_wall_relaxation_gpu(cp, ctx.transport)

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
