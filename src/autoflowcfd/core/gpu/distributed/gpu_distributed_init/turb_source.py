"""AutoFlowCFD V2.0 - 多 GPU 分布式湍流源项（SST/DDES/IDDES/LES/WMLES）

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_init.py` 的 `_GPUDistributedInitMixin` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `_GPUDistributedInitMixin` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from autoflowcfd.core.gpu import get_cupy


class _GPUDistributedTurbSourceMixin:
    """多 GPU 分布式湍流源项（SST/DDES/IDDES/LES/WMLES）"""

    def _compute_turbulence_source_distributed(self, dt: float):
        """分布式 SST 源项+输运计算（真正实现，2026-09-02）。

        复用而不是重新实现单机数值逻辑（与 CPU MPI 路径 `core/mpi/
        distributed_turbulence.py::distributed_compute_turbulence_
        source_and_viscosity` 完全同一个设计）：k/omega 做与平均流
        `U_gpu` 同一套 halo 交换 + compact 索引空间重排，构造一个
        compact 大小的临时 `GPUTurbulenceSST`"视图"，调用完全相同的
        `compute_source_terms_gpu`/`compute_turbulence_transport_
        residual_gpu`/`update_fields_gpu`，只把 local cells 的结果
        写回真正的（n_local 大小的）`turb_model_gpu`。

        此前这里直接用 `self.U_gpu`（native 排列，只有 n_local，没有
        halo）算梯度、且引用了本类从未定义过的 `self.Q_gpu`/
        `self.ops_data`（后者已在 __init__ 补上）——两处都是真实 bug，
        不只是"wall_distance 索引空间不对齐"这一处，见本方法此前版本
        文档。

        Args:
            dt: 本步物理时间步长（标量）——与本类 mean-flow RK 推进
                本步的**逐单元局部物理步长的均值**（2026-09-14：分布式
                路径已补齐逐 cell 局部 CFL，此前"用全局固定步长"的既定
                简化已不存在，见 gpu_distributed.py::
                _compute_local_time_step_gpu 的重写说明），与单机 GPU
                路径的既定约定一致
                （`update_fields_gpu(dt: float, ...)` 本来就接受标量，
                单机版用 `float(cp.mean(dt_local))`，不是 CPU 路径那种
                per-cell dt_local 数组）。

        Returns:
            mu_t_field_compact: (n_compact, n_sps) 动力涡粘度场，供
            `compute_viscous_residual_gpu` 的 BR1 界面项消费；
            `turb_model_gpu is None` 时返回 None。
        """
        cp = get_cupy()

        if self.turb_model_gpu is None:
            if self.sgs_model_gpu is None:
                return None
            # 纯 LES（2026-09-02）：WALE 是纯代数模型，没有 SST 那种
            # k/omega 跨步 ODE 积分状态，不需要"halo 交换持久状态+构造
            # 临时视图"这套复杂度——直接用当前（halo 交换后的 compact）
            # 速度场现算 mu_t 即可，与单机版
            # `_apply_turbulence_corrections_gpu` 在操作分裂时序上的
            # 差异（单机版排在 step() 状态更新之后、供*下一步*用）在
            # 数学上等价：WALE 不依赖历史状态，"用当前状态现算"和"上一步
            # 结束时用同一个当前状态算好缓存"是同一个值，没有必要在
            # 分布式路径上复刻单机版那个只是为了性能缓存的调用时序。
            U_extended = self.gpu_halo.exchange(self.U_gpu)
            U_compact = self._permute_to_compact(U_extended)
            from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
            from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu
            Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
            rho_compact = Q_compact[:, :, 0]
            # 真实 bug 修复（2026-09-03，同下面 300 行附近 compute_source_
            # terms_gpu 调用点同一处修复）：不能对*守恒*变量 U_compact
            # 求梯度再切片动量分量冒充速度梯度，直接对 Q_compact 的速度
            # 分量求梯度。
            grad_vel = compute_physical_gradient_gpu(Q_compact[..., 1:4], self.mesh_data, self.ops_data)
            nu_t_compact = self.sgs_model_gpu.compute_eddy_viscosity_gpu(grad_vel, self._grid_scale_compact)
            return rho_compact * nu_t_compact

        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell
        n_compact = self.n_compact

        # 1. 平均流 halo 交换（与 mean-flow 残差同一个 self.gpu_halo）+
        # 重排到 compact 索引空间——只为了拿 Q（rho/velocity）用于
        # grad_vel，与 compute_inviscid_residual_gpu 同一处理。
        U_extended = self.gpu_halo.exchange(self.U_gpu)
        U_compact = self._permute_to_compact(U_extended)

        # 2. k/omega halo 交换（独立于平均流的 2-var 交换）。
        k_omega_local = cp.stack(
            [self.turb_model_gpu.k_field, self.turb_model_gpu.omega_field], axis=-1
        )  # (n_local,n_sps,2)
        k_omega_extended = self.turb_halo_gpu.exchange(k_omega_local)
        k_omega_compact = self._permute_to_compact(k_omega_extended)

        # 2b. des_length_scale halo 交换（DDES/IDDES 专用，2026-09-02，
        # 与 CPU MPI 路径 distributed_turbulence.py 同一处真实 bug 修复
        # 同一个根因——`des_length_scale` 是跨步持久状态，必须与
        # k_field/omega_field 同一套 halo 交换+compact 重排才能在第二
        # 次及以后的调用里正确使用，不能直接把 n_local 大小的数组
        # setattr 到 n_compact 大小的临时视图上）。只有 1 个分量，不能
        # 复用 2-var 的 turb_halo_gpu。
        des_length_scale_compact = None
        if self.ddes_model_gpu is not None and getattr(self.turb_model_gpu, "des_length_scale", None) is not None:
            des_len_extended = self.des_length_scale_halo_gpu.exchange(
                self.turb_model_gpu.des_length_scale[:, :, None]
            )[:, :, 0]
            des_length_scale_compact = self._permute_to_compact(des_len_extended)

        # 3. 构造 compact 索引空间的临时 GPUTurbulenceSST"视图"——不能
        # 直接复用真正的 turb_model_gpu（只有 n_local 大小），需要一个
        # 同形状为 n_compact 的临时对象供 compute_source_terms_gpu/
        # update_fields_gpu 读写，用完只把 local 部分写回真正的模型。
        from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
        turb_view = GPUTurbulenceSST(n_compact, n_sps, self.device_id)
        for attr in (
            "sigma_k1", "sigma_k2", "sigma_w1", "sigma_w2", "beta1", "beta2",
            "a1", "kappa", "beta_star", "k_max", "omega_max", "production_factor",
        ):
            if hasattr(self.turb_model_gpu, attr):
                setattr(turb_view, attr, getattr(self.turb_model_gpu, attr))
        # des_length_scale 已在上面 2b 步用正确的 halo 交换+compact 重排
        # 算好（None 表示还没有跨步长度尺度记忆，与首次调用/纯 SST 一致，
        # 不能像其余常量那样直接 setattr 原始 n_local 大小的值——见 2b
        # 步注释的真实 bug 说明）。
        turb_view.des_length_scale = des_length_scale_compact
        turb_view.k_field = k_omega_compact[..., 0].copy()
        turb_view.omega_field = k_omega_compact[..., 1].copy()

        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
        from autoflowcfd.core.gpu.residual.gpu_gradients import (
            compute_physical_gradient_gpu,
            compute_physical_scalar_gradient_gpu,
        )

        Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
        rho_compact = Q_compact[:, :, 0]

        # 真实 bug 修复（2026-09-02 的 shape 修复 + 2026-09-03 的进一步
        # 修正，与单机 GPU 路径 gpu_solver_io.py 同一处独立发现的 bug
        # 同一个根因，见该文件对应修复说明）：`compute_source_terms_gpu`
        # 的 `grad_U` 形参实际期望速度梯度 (n_cells,n_sps,3,3)（内部
        # compute_strain_rate_magnitude_gpu 直接对末两维转置相加）。
        # 2026-09-02 的修复只解决了"切片成 3 变量"这一半（避免 shape
        # 崩溃），但对*守恒*变量 U_compact 求梯度再切片动量分量仍然是
        # 冒充速度梯度——grad(rho*u) != rho*grad(u)，除非密度梯度处处
        # 为零。改为直接对 Q_compact 的速度分量求梯度。
        grad_vel = compute_physical_gradient_gpu(Q_compact[..., 1:4], self.mesh_data, self.ops_data)

        d_wall = self.wall_distance_gpu
        if d_wall is None:
            raise RuntimeError(
                f"Rank {self.rank}: wall distance field not computed for turbulence "
                f"model '{getattr(self, 'turb_model_name', '?')}'. Please ensure "
                f"_init_wall_distance_distributed() ran during solver initialization. "
                f"Industrial-grade calculation requires accurate wall distance, not "
                f"simplified estimates."
            )

        grad_k = compute_physical_scalar_gradient_gpu(turb_view.k_field, self.mesh_data, self.ops_data)
        grad_omega = compute_physical_scalar_gradient_gpu(turb_view.omega_field, self.mesh_data, self.ops_data)

        max_grad_mag = 1e6
        grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
        if cp.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / cp.maximum(grad_k_mag, 1e-10)
            grad_k *= cp.clip(scale_k, 0, 1)[..., None]
        if cp.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / cp.maximum(grad_omega_mag, 1e-10)
            grad_omega *= cp.clip(scale_omega, 0, 1)[..., None]

        Sk, S_omega = turb_view.compute_source_terms_gpu(
            Q_compact, grad_vel, d_wall, self.mu_molecular, grad_k, grad_omega,
        )

        # DDES/IDDES 长度尺度更新（2026-09-02）：必须排在 compute_
        # source_terms_gpu **之后**——与单机版 gpu_solver_io.py 同一处
        # 真实顺序 bug 修复同一个根因（该文件"必须在 compute_source_
        # terms_gpu 之前"的旧注释是错的，CPU 参照的真实顺序是
        # compute_source_terms 用*上一步*的 des_length_scale（"慢一拍"
        # 耦合），算完之后才用*本步*的 nu_t 写入*下一步*要用的新
        # des_length_scale，两者互相依赖对方的输出、只能这样错开一步，
        # 见该文件对应修复文档的完整推导）。
        if self.ddes_model_gpu is not None:
            nu_field = self.mu_molecular / cp.maximum(rho_compact, 1e-10)
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUIDDESModel
            if isinstance(self.ddes_model_gpu, GPUIDDESModel):
                self.ddes_model_gpu.apply_to_sst_model_iddes_gpu(
                    turb_view, d_wall, self.iddes_h_max_compact, self.iddes_h_wn_compact,
                    nu_field, grad_vel,
                )
            else:
                cell_volumes = self.mesh_data.get('cell_volumes')
                self.ddes_model_gpu.apply_to_sst_model_gpu(
                    turb_view, d_wall, cell_volumes, nu_field, grad_vel,
                    h_max=getattr(self, 'iddes_h_max_compact', None),
                )

        dk_dt = Sk / cp.maximum(rho_compact, 1e-10)
        domega_dt = S_omega / cp.maximum(rho_compact, 1e-10)

        # 4. 完整输运项（对流+扩散）：与单机版同一个 compute_turbulence_
        # transport_residual_gpu，用一个只暴露它实际读取属性的最小鸭子
        # 类型 adapter（mesh 只需要 n_cells/n_sps_per_cell/n_prism_cells，
        # 不是完整网格对象——`n_prism_cells` 即便 `self.mesh_data` 里
        # 已经有 `'n_prism'` 键，`dict.get('n_prism', solver.mesh.
        # n_prism_cells)` 的默认值表达式仍会被 Python 无条件求值，
        # 缺这个属性会直接 AttributeError，不是"用不到就不用给"）。
        import types
        n_prism_compact = self.dist_flat_face.base_flat.n_prism
        transport_adapter = types.SimpleNamespace(
            turb_model_gpu=turb_view,
            mesh=types.SimpleNamespace(
                n_cells=n_compact, n_sps_per_cell=n_sps, n_prism_cells=n_prism_compact,
            ),
            Q_gpu=Q_compact,
            U_gpu=U_compact,
            mu_molecular=self.mu_molecular,
            mesh_data=self.mesh_data,
            ops_data=self.ops_data,
            flat_face_gpu=self.flat_face_gpu,
            wall_distance_gpu=d_wall,
            _wall_mask_k_gpu=self._wall_mask_k_gpu,
            _open_mask_gpu=self._open_mask_gpu,
        )
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
            compute_turbulence_transport_residual_gpu,
        )
        transport_k, transport_omega = compute_turbulence_transport_residual_gpu(
            transport_adapter, grad_vel=grad_vel,
        )

        # 5. 场更新（点隐式阻尼+输运，见 update_fields_gpu 文档）。
        # `dt` 现在由调用方传入**逐单元局部物理步长的均值**（2026-09-14，
        # 与 CPU 分布式/单机同一时序：湍流标量必须用按**物理**波速算出的
        # 那一份 dt，不能跟着低马赫数预处理放大——k/omega 的显式更新刻意
        # 没有 point-implicit 阻尼）。此前这里用的是调用方传入的全局固定
        # dt，并注明"本类分布式路径目前统一用全局固定步长、
        # _compute_local_time_step_gpu 与 compact 索引空间不兼容"——那个
        # 前提已经不成立：该方法已重写为紧凑索引空间、逐 SP 取最小值。
        # `update_fields_gpu` 接标量，所以这里取均值（与该函数在其它路径
        # 上的既有用法一致，见 gpu_solver_io.py 的 `cp.mean(dt_physical)`）。
        turb_view.update_fields_gpu(
            float(dt), dk_dt, domega_dt,
            transport_k=transport_k, transport_omega=transport_omega,
        )

        # 5.4 真实 bug 修复（2026-09-12，与单机 GPU 路径 gpu_solver_io.py、
        # CPU 分布式路径（经 compute_turbulence_source 自动获得，见该处
        # 文档）同一处修复，完整推导见 gpu_modal_filter.py::
        # filter_scalar_field_gpu 文档）：k/omega 场同样需要模态滤波。
        # compact 索引空间下 n_prism 用 dist_fc.base_flat.n_prism（"棱柱
        # 在前"的既定 compact 排序约定，与 CPU 分布式路径的 mesh_adapter
        # 同一个量）。
        if n_sps > 1:
            from autoflowcfd.core.fr_solver.filter import resolve_turb_filter_gate
            from autoflowcfd.core.gpu.gpu_modal_filter import (
                filter_scalar_field_gated_gpu, filter_scalar_field_gpu,
            )
            n_prism_compact = self.flat_face_gpu.n_prism
        # 门控维度 `AFCFD_FILTER_TURB_GATE`（2026-09-15 系统性审计的 A 类
        # 发现）：这一维此前只在 CPU 路径接线过，GPU 这边无条件全场滤波
        # ——同一个环境变量在不同后端意味着不同的数值方案且无任何提示。
        # 现已用 GPU 版同一套传感器补齐（gpu_troubled_cell.py），默认
        # "all" 与此前行为逐位一致。
            if resolve_turb_filter_gate() == "sensor":
                from autoflowcfd.core.gpu.gpu_troubled_cell import (
                    compute_turb_troubled_mask_gpu,
                )
                _order = int(getattr(self, "current_order", None)
                             or getattr(self, "order", 0))
                troubled = compute_turb_troubled_mask_gpu(
                    turb_view.k_field, turb_view.omega_field,
                    n_prism_compact, _order)
                self._turb_filter_troubled_frac = float(cp.mean(troubled))
                turb_view.k_field = filter_scalar_field_gated_gpu(
                    turb_view.k_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet, troubled,
                )
                turb_view.omega_field = filter_scalar_field_gated_gpu(
                    turb_view.omega_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet, troubled,
                )
            else:
                turb_view.k_field = filter_scalar_field_gpu(
                    turb_view.k_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet,
                )
                turb_view.omega_field = filter_scalar_field_gpu(
                    turb_view.omega_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet,
                )
            turb_view.apply_positivity_limiter_gpu()

        # 5.5 真实缺口修复（2026-09-05，代码复审发现，与单机 GPU 路径
        # gpu_solver_io.py 同一处修复同一个根因）：omega 壁面 Wilcox
        # 解析值扩散侧未闭合，此前多GPU分布式路径完全没有移植这一步
        # ——直接复用上面已经构造好的 transport_adapter（它已经暴露了
        # enforce_omega_wall_relaxation_gpu 需要的全部属性：Q_gpu/
        # turb_model_gpu/flat_face_gpu/wall_distance_gpu/mu_molecular/
        # _wall_mask_k_gpu/mesh.n_cells），turb_view 是同一个对象引用，
        # 函数内部对 turb_model_gpu.omega_field 的原地修改直接反映到
        # turb_view 上。
        if getattr(self, "turb_model_name", "").upper() in ("SST", "DDES", "IDDES"):
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
                enforce_omega_wall_relaxation_gpu,
            )
            enforce_omega_wall_relaxation_gpu(cp, transport_adapter)

        # 6. 只把 local cells 的更新结果写回真正的 turb_model_gpu（native
        # 排列，见 dist_flat_face.inv_perm 文档——turb_view.k_field 是
        # compact 排列，需要先换回原生排列再切 [:n_local]，与平均流
        # 残差同一处理）。
        k_native = self._unpermute_from_compact(turb_view.k_field)
        omega_native = self._unpermute_from_compact(turb_view.omega_field)
        self.turb_model_gpu.k_field[:] = k_native[:n_local]
        self.turb_model_gpu.omega_field[:] = omega_native[:n_local]
        nu_t_native = self._unpermute_from_compact(turb_view.nu_t)
        self.turb_model_gpu.nu_t[:] = nu_t_native[:n_local]

        # DDES/IDDES：des_length_scale 是跨步持久状态（本步 apply_to_
        # sst_model[_iddes] 刚用本步 nu_t 算出、要写回供*下一步* 2b 步
        # 读取），与 k_field/omega_field 同一套写回处理。
        if self.ddes_model_gpu is not None and getattr(turb_view, "des_length_scale", None) is not None:
            des_len_native = self._unpermute_from_compact(turb_view.des_length_scale)
            self.turb_model_gpu.des_length_scale = des_len_native[:n_local].copy()

        # mu_t_field 保持 compact 排列（平均流粘性残差直接消费 compact
        # 索引空间的场，见 compute_viscous_residual_gpu 里 U_compact 的
        # 同一处理）。
        return rho_compact * turb_view.nu_t
