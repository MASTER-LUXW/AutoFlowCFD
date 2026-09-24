"""AutoFlowCFD V2.0 - 单机 GPU 的无粘/粘性残差与原始量更新

从 `src/autoflowcfd/core/gpu/solver/gpu_solver.py` 的 `GPUFRSolver` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `GPUFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np
from autoflowcfd.core.gpu import get_cupy


class _GPUSolverResidualMixin:
    """单机 GPU 的无粘/粘性残差与原始量更新"""

    def _update_primitives_gpu(self):
        """GPU 上更新原始变量。"""
        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
        self.Q_gpu = conserved_to_primitive_gpu(self.U_gpu[..., :5])

    def compute_inviscid_residual_gpu(self, U_trial=None):
        """GPU 计算无粘残差。

        Args:
            U_trial: CuPy 数组 (n_cells, n_sps, n_vars)，试验解（可选）

        Returns:
            residual: CuPy 数组 (n_cells, n_sps, 5)
        """
        cp = get_cupy()
        U = U_trial if U_trial is not None else self.U_gpu

        if self.mesh.n_points_1d == 1:
            # P0 路径：使用 CuPy RawKernel
            from autoflowcfd.core.gpu.residual.gpu_p0_inviscid import (
                compute_inviscid_residual_p0_cupy_gpu_resident,
            )
            Q_flat = self.Q_gpu[:, 0, :5].copy()
            # 需要面连接关系数据
            fc = self.mesh.face_connectivity
            owner = cp.asarray(fc.owner_cell)
            neighbor = cp.asarray(
                np.where(fc.is_boundary, 0, fc.neighbor_cell)
            )
            is_bnd = cp.asarray(fc.is_boundary)
            # 面法向和面积（性能修复，真实复现：本函数是 GPU P0 无粘残差
            # 每步都要调用的热路径，此前这里逐面 `ffp_list[f]` 索引——自
            # face_flux_points/merge.py 的 flat array 重构后
            # `mesh.face_flux_points` 是 `_KernelFaceData`，`[f]` 会按需
            # *构造*一个完整 FaceFluxPointGeometry 对象，187 万面级别的
            # 网格上每步都这样做是灾难级开销，与
            # core/fr_operators/troubled_cell.py::precompute_cell_face_
            # misalignment 是同一类遗漏、同一次真实复现中一并发现。改用
            # inviscid_p0.py 里已经过测试、CPU P0 热路径同样在用的快速
            # 路径提取函数，数学上是同一批数据，只是不再逐面构造对象。
            n_faces = fc.n_faces
            from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
            normal, area_w = _extract_p0_face_geometry(self.mesh.face_flux_points, fc, n_faces)
            normal_gpu = cp.asarray(normal)
            area_w_gpu = cp.asarray(area_w)
            volumes_gpu = self.mesh_data.get('cell_volumes')
            if volumes_gpu is None:
                volumes_gpu = cp.asarray(self.mesh.get_all_cell_volumes())

            # 边界幽灵态（正确性修复，B-8 一并暴露）：此前这里恒传全零，
            # kernel 会把边界面外部状态当成零态解 AUSM 黎曼问题。改为与
            # gpu_p0_inviscid.py::compute_inviscid_residual_p0_cupy 同款的
            # 逐面 ghost_provider 预计算；范围含混合拆分面的边界子面记录（B-8）。
            from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider
            ghost_provider = (
                self.boundary_ghost_provider
                if self.boundary_ghost_provider is not None
                else DefaultGhostProvider()
            )
            ffp_list = self.mesh.face_flux_points
            mixed_bnd_face = getattr(ffp_list, "mixed_bnd_face", None)
            if mixed_bnd_face is None:
                mixed_bnd_face = np.zeros(n_faces, dtype=np.bool_)
            mixed_bnd_frac = getattr(ffp_list, "mixed_p0_bnd_frac", None)
            if mixed_bnd_frac is None:
                mixed_bnd_frac = np.zeros(n_faces, dtype=np.float64)
            ghost_faces = np.nonzero(fc.is_boundary | mixed_bnd_face)[0]
            Q_flat_np = cp.asnumpy(Q_flat)
            Q_ghost_np = np.zeros((n_faces, 5), dtype=np.float64)
            for f in ghost_faces:
                oc = int(fc.owner_cell[f])
                Q_ghost_np[f, :] = ghost_provider(
                    f, Q_flat_np[oc: oc + 1], normal[f: f + 1]
                )[0]
            Q_ghost = cp.asarray(Q_ghost_np)
            mixed_bnd_frac_gpu = cp.asarray(mixed_bnd_frac)

            res = compute_inviscid_residual_p0_cupy_gpu_resident(
                Q_flat, owner, neighbor, is_bnd,
                normal_gpu, area_w_gpu, volumes_gpu,
                Q_ghost, mixed_bnd_frac_gpu, self.mesh.n_cells, n_faces,
                mach_ref=self.freestream["mach_ref"],
            )
            # 扩展到 (n_cells, n_sps, 5)
            return cp.broadcast_to(res, (self.mesh.n_cells, self.mesh.n_sps_per_cell, 5)).copy()
        else:
            # P>=1 高阶 FR GPU 路径
            from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu
            return compute_inviscid_residual_fr_gpu(
                U, self.mesh, self.ops,
                boundary_ghost_provider=self.boundary_ghost_provider,
                mesh_data=self.mesh_data,
                ops_data=self.ops_data,
                flat_face_gpu=self.flat_face_gpu,
                device_id=self.device_id,
                mach_ref=self.freestream["mach_ref"],
            )

    def compute_viscous_residual_gpu(self, U_trial=None, mu_t_field=None):
        """GPU 计算粘性残差。

        Args:
            U_trial: CuPy 数组（可选）
            mu_t_field: 湍流涡粘度 rho*nu_t (n_cells, n_sps) CuPy 数组（可选）

        Returns:
            viscous_residual: CuPy 数组 (n_cells, n_sps, 5)
        """
        from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu
        U = U_trial if U_trial is not None else self.U_gpu
        res = compute_viscous_residual_fr_gpu(
            U, self.mesh, self.ops,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mesh_data=self.mesh_data,
            ops_data=self.ops_data,
            flat_face_gpu=self.flat_face_gpu,
            device_id=self.device_id,
        )

        # WMLES 壁面剪应力修正（#7）：与 CPU 版
        # `FRSolver.compute_viscous_residual` 同一个调用点——必须叠加在
        # 这里（残差组装阶段），不能等 step() 状态更新之后才生效，见
        # gpu_turbulence_wmles.py 模块文档。
        if self.wmles_model is not None:
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_wmles import (
                compute_wmles_wall_stress_correction_gpu,
            )
            wall_stress_correction = compute_wmles_wall_stress_correction_gpu(self, U=U)
            if wall_stress_correction is not None:
                res = res + wall_stress_correction[..., : res.shape[-1]]

        return res
