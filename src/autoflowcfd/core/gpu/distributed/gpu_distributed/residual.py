"""AutoFlowCFD V2.0 - `MultiGPUDistributedSolver` 的 halo 交换与残差组装（mixin，只含方法）。

从 `core/gpu/distributed/gpu_distributed.py` 拆出（2026-09-25）。
"""

import numpy as np

from autoflowcfd.core.gpu import get_cupy


class _MultiGPUResidualMixin:
    """halo 交换、压缩索引空间换序、残差组装、正性限制器与全局残差范数。"""

    def _halo_exchange_gpu(self):
        """GPU 直接 halo 交换（优化版）。

        支持两种模式：
        1. CUDA-aware MPI：GPU buffer 直接通信（零拷贝）
        2. Staging buffer：GPU→CPU→MPI→CPU→GPU（只传输必要数据）
        """
        self.U_extended_gpu = self.gpu_halo.exchange(self.U_gpu)

    def _permute_to_compact(self, U_native):
        """把 halo 交换协议原生排列（local在前、halo在后）的场数组重排到
        本类残差计算实际使用的"棱柱在前、四面体在后"压缩索引空间——见
        distributed_flat_face.py::DistributedFlatFaceGeometry.perm 文档。
        """
        return U_native[self._perm_gpu]

    def _unpermute_from_compact(self, field_compact):
        """`_permute_to_compact` 的逆操作：把"棱柱在前"压缩索引空间的
        结果换回 halo 交换协议原生排列，供 `[:n_local]` 切片取出本 rank
        真正拥有的 local cells 结果。"""
        return field_compact[self._inv_perm_gpu]

    def compute_inviscid_residual_gpu(self):
        """GPU 计算分布式无粘残差。

        #1（2026-08-28）：此前直接把 `self.U_gpu`（当时误按全局单元数
        分配）传给底层函数；现在 `self.U_gpu` 只有 n_local 个单元，
        真正参与残差计算（含分区边界处需要 halo 邻居数据的界面项）必须
        用 halo 交换后的扩展数组 `self.U_extended_gpu`（`_halo_exchange_
        gpu()` 产出，此前算出来后从未被消费——见该方法与
        `_compute_total_residual_gpu` 的调用关系）。

        halo 交换产出的 `U_extended_gpu` 是"local在前、halo在后"的原生
        排列，但 `self.mesh_data`/`self.flat_face_gpu` 是按"棱柱在前、
        四面体在后"的压缩索引空间构造的（体积项切片需要，见
        distributed_flat_face.py 模块文档）——两者不一致，必须先用
        `self._perm_gpu` 把 `U_extended_gpu` 重排到压缩索引空间，残差
        算完后再用 `self._inv_perm_gpu` 换回原生排列，才能在最后
        `[:n_local]` 切片时取到与 `self.U_gpu`（原生排列）对齐的本 rank
        local cells 结果。
        """
        U_compact = self._permute_to_compact(self.U_extended_gpu)
        from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu
        residual_compact = compute_inviscid_residual_fr_gpu(
            U_compact, self.mesh, self.ops,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mesh_data=self.mesh_data,
            ops_data=self.mesh_data,
            flat_face_gpu=self.flat_face_gpu,
            flat_face_cpu=self.dist_flat_face.base_flat,
            device_id=self.device_id,
            mach_ref=self.freestream["mach_ref"],
        )
        residual_native = self._unpermute_from_compact(residual_compact)
        return residual_native[: self.partition.n_local_cells]

    def compute_viscous_residual_gpu(self, mu_t_field=None, nu_av_compact=None):
        """GPU 计算分布式粘性残差。

        nu_av_compact: 本步冻结的问题单元人工扩散系数（compact 排列，可选），
        给出时叠加 `div(nu grad U)`（与单机同一项，见
        `compute_artificial_diffusivity_compact_gpu`）。

        mu_t_field: 湍流涡粘度场（可选）——此前本方法签名只有 self，
        但调用方 step() 以 mu_t_field=mu_t_field 关键字调用它，签名/
        调用不匹配，必然 TypeError（V2.0 专家组评审逐行核实）。

        halo 交换/压缩索引空间重排逻辑与 compute_inviscid_residual_gpu
        完全一致，见该方法文档。
        """
        U_compact = self._permute_to_compact(self.U_extended_gpu)
        from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu
        residual_compact = compute_viscous_residual_fr_gpu(
            U_compact, self.mesh, self.ops,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mesh_data=self.mesh_data,
            ops_data=self.mesh_data,
            flat_face_gpu=self.flat_face_gpu,
            flat_face_cpu=self.dist_flat_face.base_flat,
            device_id=self.device_id,
        )

        # WMLES 壁面剪应力修正：与单机同一份实现，在 compact 空间（几何取 `_compact_mesh_view`、面几何取
        # `dist_flat_face.base_flat`，与粘性残差同一个索引空间）上直接以 cupy 求值
        if self.wall_model is not None:
            correction_compact = self.wall_model.residual_correction(
                get_cupy(), self._compact_mesh_view, self.dist_flat_face.base_flat, self.boundary_ghost_provider,
                U_compact, self.mu_molecular)
            if correction_compact is not None:
                residual_compact = residual_compact + correction_compact[..., :residual_compact.shape[-1]]

        if nu_av_compact is not None:
            residual_compact = residual_compact + self._artificial_diffusion_compact_gpu(
                U_compact, nu_av_compact)[..., :residual_compact.shape[-1]]

        residual_native = self._unpermute_from_compact(residual_compact)
        return residual_native[: self.partition.n_local_cells]

    def compute_artificial_diffusivity_compact_gpu(self):
        """compact 排列的问题单元人工扩散系数 nu (n_compact, n_sps)；未启用返回 None。

        系数只依赖单元自身的解，halo 单元用 halo 数据算出的值与拥有方逐位相同
        （与 CPU MPI `distributed_artificial_viscosity.py` 同一约定）。
        """
        if not self.artificial_viscosity_enabled:
            return None
        from autoflowcfd.core.fr_operators.artificial_viscosity import compute_artificial_diffusivity
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu

        cp = get_cupy()
        U_compact = self._permute_to_compact(self.gpu_halo.exchange(self.U_gpu))
        n_compact = U_compact.shape[0]
        return compute_artificial_diffusivity(
            U_compact[..., :5], int(self.current_order), self.mesh_data['cell_volumes'],
            cp.arange(n_compact) < int(self.dist_flat_face.base_flat.n_prism),
            lambda phi: compute_physical_scalar_gradient_gpu(phi, self.mesh_data, self.mesh_data),
            alpha_av=self.artificial_viscosity_alpha)

    def _artificial_diffusion_compact_gpu(self, U_compact, nu_compact):
        """compact 空间的 `div(nu grad U_k)`，k = 0..4（GPU 标量扩散装配）。"""
        from autoflowcfd.core.fr_operators.artificial_viscosity import artificial_diffusion_residual
        from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport.residual import (
            compute_scalar_diffusion_residual_gpu,
        )

        n_compact, n_sps = U_compact.shape[0], U_compact.shape[1]
        n_prism = int(self.dist_flat_face.base_flat.n_prism)
        c_ip = resolve_viscous_ip_constant(int(self.current_order))
        return artificial_diffusion_residual(
            U_compact[..., :5], nu_compact,
            lambda phi, gamma: compute_scalar_diffusion_residual_gpu(
                phi, gamma, self.mesh_data, self.mesh_data, self.flat_face_gpu,
                n_compact, n_prism, n_sps, c_ip=c_ip, gamma_band=0.0))

    def _compute_total_residual_gpu(self, mu_t_field=None, nu_av_compact=None):
        """计算总的 **dU/dt**（无粘 + 粘性），先执行 halo 交换。

        符号约定与 `compute_*_residual_fr_gpu`（以及 CPU 的
        `compute_*_residual_fr`）一致：返回的是 dU/dt 本身，**不是**积分器
        约定里的 R（dU/dt = -R）。唯一的消费方 `step()` 里的
        `_spatial_residual` 负责取负。此前多 GPU 的手工 RK 分支把它当成 R
        （`L = -res`），在真实残差上逆时间积分（2026-09-25 修复）。

        Args:
            mu_t_field: 动力涡粘度 (n_cells, n_sps) CuPy 数组（可选）
            nu_av_compact: 问题单元人工扩散系数（compact 排列，可选）
        """
        self._halo_exchange_gpu()
        res = self.compute_inviscid_residual_gpu()
        res = res + self.compute_viscous_residual_gpu(mu_t_field=mu_t_field, nu_av_compact=nu_av_compact)
        return res

    def local_cell_geometry(self):
        """本 rank local 单元（**原生**排列，`self.U_gpu` 所在的空间）的 `(det_jacs (n_local, n_sps), 是否棱柱)`，
        主机数组。`mesh_data['det_jacs']` 在"棱柱在前"紧凑排列，经 `inv_perm` 换回。正性限制器与写盘共用。"""
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism

        cp = get_cupy()
        n_local = self.partition.n_local_cells
        dist_fc = self.dist_flat_face
        det_c = cp.asnumpy(self.mesh_data['det_jacs'])
        det_c = det_c.reshape(det_c.shape[0], -1)
        return det_c[np.asarray(dist_fc.inv_perm)][:n_local], native_cell_is_prism(dist_fc)[:n_local]

    def _get_positivity_limiter_gpu(self):
        """守恒的正性保持限制器（与 CPU 同一个，GPU 上走数组模块无关实现），几何取 `local_cell_geometry`。
        按阶数缓存，只在缓存失效（阶数切换）时下载一次 det(J)。"""
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        return get_positivity_limiter(self, xp=get_cupy(), geometry=self.local_cell_geometry)

    def _monitor_residuals(self, residual_local):
        """`(监控残差范数, 逐方程 RMS)`：本 rank local 单元（原生排列）的残差，全部后端同一个定义（真实自由度上的
        全局 RMS，`time_integration/residual_norm.py`）。"""
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism
        from autoflowcfd.core.mpi.reductions import MPIReductions
        from autoflowcfd.core.time_integration.residual_norm import monitor_residuals
        from autoflowcfd.fr.native_padding import order_from_n_sps

        n_local = self.partition.n_local_cells
        return monitor_residuals(MPIReductions(get_cupy()), residual_local,
                                 native_cell_is_prism(self.dist_flat_face)[:n_local],
                                 order_from_n_sps(self.mesh.n_sps_per_cell))
