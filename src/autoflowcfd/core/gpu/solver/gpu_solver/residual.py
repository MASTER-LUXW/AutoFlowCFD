"""AutoFlowCFD V2.0 - 单机 GPU 的无粘/粘性残差与原始量更新

`GPUFRSolver` 的一组方法（mixin）。

**只含方法，没有状态**：全部属性由 `GPUFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from typing import Any

from autoflowcfd.core.gpu import get_cupy


class _GPUSolverResidualMixin:
    """单机 GPU 的无粘/粘性残差与原始量更新"""

    # 宿主类在 __init__ / 其它 mixin 中提供的属性（仅类型声明）
    mesh: Any
    mesh_data: Any
    ops: Any

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
            # P0：有限体积 RawKernel（与多 GPU 分布式、CPU 同一个离散）
            from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
            from autoflowcfd.core.gpu.residual.gpu_p0_inviscid import compute_inviscid_residual_p0_gpu

            res = compute_inviscid_residual_p0_gpu(
                U, get_flat_face_geometry(self.mesh, self.ops), self._p0_face_arrays_gpu(),
                self.boundary_ghost_provider, self.freestream["mach_ref"])
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

    def _p0_face_arrays_gpu(self) -> dict:
        """P0 有限体积残差用到的面几何（设备端），按网格对象缓存（几何不随状态变化，换阶数时网格对象随之更换）。"""
        cache = getattr(self, "_p0_face_cache", None)
        if cache is not None and cache["mesh"] is self.mesh:
            return cache
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.fr_residual.inviscid_p0 import p0_face_geometry
        from autoflowcfd.core.gpu.residual.gpu_p0_inviscid import p0_face_arrays_gpu

        normal, area_w = p0_face_geometry(self.mesh.face_flux_points)
        volumes = self.mesh_data.get("cell_volumes")
        cache = p0_face_arrays_gpu(get_flat_face_geometry(self.mesh, self.ops), normal, area_w,
                                   volumes if volumes is not None else self.mesh.get_all_cell_volumes())
        cache["mesh"] = self.mesh
        self._p0_face_cache = cache
        return cache

    def compute_viscous_residual_gpu(self, U_trial=None, mu_t_field=None, nu_av=None):
        """GPU 计算粘性残差。

        Args:
            U_trial: CuPy 数组（可选）
            mu_t_field: 湍流涡粘度 rho*nu_t (n_cells, n_sps) CuPy 数组（可选）
            nu_av: 本步冻结的问题单元人工扩散系数（运动粘度，可选）；给出时叠加
                `div(nu grad U)`，与 CPU `FRSolver.compute_viscous_residual` 同一项

        Returns:
            viscous_residual: CuPy 数组 (n_cells, n_sps, 5)
        """
        from autoflowcfd.core.fr_residual.viscous_flux.constants import PRANDTL
        from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu
        U = U_trial if U_trial is not None else self.U_gpu
        res = compute_viscous_residual_fr_gpu(
            U, self.mesh, self.ops,
            mu=self.mu_molecular, Pr=PRANDTL,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mesh_data=self.mesh_data,
            ops_data=self.ops_data,
            flat_face_gpu=self.flat_face_gpu,
            device_id=self.device_id,
        )

        # WMLES 壁面剪应力修正：与 CPU 同一份实现（`core/turbulence/wall_model/`，数组模块无关），残差组装阶段施加
        if self.wall_model is not None:
            from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

            wall_stress_correction = self.wall_model.residual_correction(
                get_cupy(), self.mesh, get_flat_face_geometry(self.mesh, self.ops), self.boundary_ghost_provider,
                U, self.mu_molecular)
            if wall_stress_correction is not None:
                res = res + wall_stress_correction[..., : res.shape[-1]]

        if nu_av is not None:
            res = res + self._artificial_diffusion_residual_gpu(U, nu_av)[..., : res.shape[-1]]

        return res

    def compute_artificial_diffusivity_field_gpu(self, U=None):
        """问题单元人工扩散系数 nu (n_cells, n_sps)（运动粘度）；未启用返回 None。

        与 CPU `FRSolver.compute_artificial_diffusivity_field` 同一个函数，梯度用
        GPU 的单元内物理梯度。
        """
        if not self.artificial_viscosity_enabled:
            return None
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_artificial_diffusivity,
        )
        from autoflowcfd.core.gpu.residual.gpu_gradients import (
            compute_physical_scalar_gradient_gpu,
        )

        cp = get_cupy()
        U = self.U_gpu if U is None else U
        n_cells = U.shape[0]
        return compute_artificial_diffusivity(
            U, int(self.current_order), self._cell_volumes_gpu(),
            cp.arange(n_cells) < int(self.mesh.n_prism_cells),
            lambda phi: compute_physical_scalar_gradient_gpu(phi, self.mesh_data, self.ops_data),
            alpha_av=self.artificial_viscosity_alpha)

    def _artificial_diffusion_residual_gpu(self, U, nu_av):
        """`div(nu grad U_k)`，k = 0..4（GPU 标量扩散装配，与 CPU 逐项对应）。"""
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            artificial_diffusion_residual,
        )
        from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport.residual import (
            compute_scalar_diffusion_residual_gpu,
        )

        n_cells, n_sps = U.shape[0], U.shape[1]
        c_ip = resolve_viscous_ip_constant(int(self.current_order))
        return artificial_diffusion_residual(
            U, nu_av,
            lambda phi, gamma: compute_scalar_diffusion_residual_gpu(
                phi, gamma, self.mesh_data, self.ops_data, self.flat_face_gpu,
                n_cells, int(self.mesh.n_prism_cells), n_sps, c_ip=c_ip, gamma_band=0.0))
