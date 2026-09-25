"""AutoFlowCFD V2.0 - `MultiGPUDistributedSolver` 的 halo 交换与残差组装（mixin，只含方法）。

从 `core/gpu/distributed/gpu_distributed.py` 拆出（2026-09-25）。
"""

import numpy as np

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.mpi.comm import allreduce_sum


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

    def compute_viscous_residual_gpu(self, mu_t_field=None):
        """GPU 计算分布式粘性残差。

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

        # WMLES 壁面剪应力修正（2026-09-02）：与单机 GPUFRSolver.
        # compute_viscous_residual_gpu 同一个施加时机（残差组装阶段，
        # 时间积分之前），直接调用 CPU 核心函数而不是
        # gpu_turbulence_wmles.py 的单机 facade（后者用 `solver.mesh`
        # 构造 extrap 所需的 n_prism_cells/jacobians，单机语境下就是
        # compact 空间本身；分布式场景下 `self.mesh` 在"传统模式"下是
        # **全局**网格，必须换成 `self._compact_mesh_view`，见该属性
        # 构造处的说明），自己搭建同样结构的 facade——与 CPU MPI 分布式
        # `distributed_compute_viscous_residual` 完全同一个模式。
        if self.wmles_model is not None:
            cp = get_cupy()
            import types
            from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
            from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction

            U_compact_cpu = cp.asnumpy(U_compact)
            Q_compact_cpu = conserved_to_primitive(U_compact_cpu[..., :5])
            wall_distance_cpu = (
                cp.asnumpy(self.wall_distance_gpu) if self.wall_distance_gpu is not None else None
            )
            facade = types.SimpleNamespace(
                wmles_model=self.wmles_model, mesh=self._compact_mesh_view, ops=self.ops,
                wall_distance=wall_distance_cpu,
                state=types.SimpleNamespace(U=U_compact_cpu, Q=Q_compact_cpu),
                boundary_ghost_provider=self.boundary_ghost_provider,
            )
            correction_cpu = compute_wmles_wall_stress_correction(
                facade, flat_face_override=self.dist_flat_face.base_flat,
            )
            if correction_cpu is not None:
                correction_compact = cp.asarray(correction_cpu)
                residual_compact = residual_compact + correction_compact[..., :residual_compact.shape[-1]]

        residual_native = self._unpermute_from_compact(residual_compact)
        return residual_native[: self.partition.n_local_cells]

    def _compute_total_residual_gpu(self, mu_t_field=None, inviscid=True, viscous=True):
        """计算总的 **dU/dt**（无粘 + 粘性），先执行 halo 交换。

        符号约定与 `compute_*_residual_fr_gpu`（以及 CPU 的
        `compute_*_residual_fr`）一致：返回的是 dU/dt 本身，**不是**积分器
        约定里的 R（dU/dt = -R）。唯一的消费方 `step()` 里的
        `_spatial_residual` 负责取负。此前多 GPU 的手工 RK 分支把它当成 R
        （`L = -res`），在真实残差上逆时间积分（2026-09-25 修复）。

        Args:
            mu_t_field: 动力涡粘度 (n_cells, n_sps) CuPy 数组（可选）
            inviscid, viscous: 取哪几部分（IMEX 分别要显式/隐式两半；
                至少一个为 True）。
        """
        if not (inviscid or viscous):
            raise ValueError("inviscid 与 viscous 至少要取一个")
        self._halo_exchange_gpu()
        res = self.compute_inviscid_residual_gpu() if inviscid else None
        if viscous:
            visc_res = self.compute_viscous_residual_gpu(mu_t_field=mu_t_field)
            res = visc_res if res is None else res + visc_res
        return res

    def _get_positivity_limiter_gpu(self):
        """守恒的正性保持限制器（与 CPU 同一个，GPU 上走数组模块无关实现）。

        几何取 local 段、**原生**排列（`self.U_gpu` 所在的空间）：
        `mesh_data['det_jacs']` 在"棱柱在前"紧凑排列，经 `inv_perm` 换回。
        按阶数缓存，只在缓存失效（阶数切换）时下载一次 det(J)。
        """
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        cp = get_cupy()
        n_local = self.partition.n_local_cells
        dist_fc = self.dist_flat_face

        def _local_geometry():
            det_c = cp.asnumpy(self.mesh_data['det_jacs'])
            det_c = det_c.reshape(det_c.shape[0], -1)
            return (det_c[np.asarray(dist_fc.inv_perm)][:n_local],
                    native_cell_is_prism(dist_fc)[:n_local])

        return get_positivity_limiter(self, xp=cp, geometry=_local_geometry)

    def _global_residual_norm(self, res_flat) -> float:
        """MPI 全局残差归约。

        #1（2026-08-28）：分母此前是 `self.mesh.n_cells（全局）* n_sps *
        5 * self.n_ranks`——`self.mesh.n_cells` 在传统模式下本来就已经是
        全局单元数，再乘一次 `n_ranks` 会把分母错误放大 n_ranks 倍
        （与 self.U_gpu 此前错误按全局尺寸分配是两个独立的 bug，凑巧
        当时两边都用了同一个"全局 n_cells"所以没有立刻表现为形状错误，
        但归一化分母本身始终是错的）。改用 `partition.n_global_cells`
        （不确定 partition_mesh 是否严格均分负载，这是唯一真正的全局
        单元总数来源），不再额外乘 n_ranks。
        """
        cp = get_cupy()
        local_norm_sq = float(cp.sum(res_flat ** 2))
        global_norm_sq = allreduce_sum(local_norm_sq)
        n_sps = self.mesh.n_sps_per_cell
        n_global = self.partition.n_global_cells * n_sps * 5
        return np.sqrt(global_norm_sq / max(1, n_global))
