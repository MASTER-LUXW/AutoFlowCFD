"""
AutoFlowCFD V2.0 - GPU FRSolver

完整的 GPU 版 FR 求解器，对应 core/fr_solver.py。
所有计算在 GPU 上完成，数据常驻显存，只在 I/O 时传输。

设计：
- 与 CPU 版 FRSolver 接口一致（solve/step/compute_*_residual）
- 内部使用 GPUArrayManager 管理 GPU 数据
- 残差计算全部走 CuPy（gpu_inviscid.py / gpu_viscous.py）
- 时间积分走 GPUTimeIntegrator（gpu_time_integration.py）
- 支持 P0 和 P>=1 两种路径
- 支持单 GPU 稳态/伪稳态求解

使用:
    solver = GPUFRSolver(mesh, ops, order=2, device_id=0)
    result = solver.solve(max_iter=1000, dt=1e-4, tol=1e-6)
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger

from autoflowcfd.core.gpu import gpu_available, get_cupy
from autoflowcfd.core.gpu.array_manager import GPUArrayManager
from autoflowcfd.core.gpu.gpu_time_integration import (
    GPUTimeIntegrator,
    enforce_positivity_gpu,
    compute_local_cfl_step_gpu,
)
from autoflowcfd.core.gpu.gpu_solver_init import _GPUSolverInitMixin
from autoflowcfd.core.gpu.gpu_solver_io import _GPUSolverIOMixin


class GPUFRSolver(_GPUSolverInitMixin, _GPUSolverIOMixin):
    """GPU 版 FR 求解器。

    与 CPU 版 FRSolver 接口一致，内部全程使用 CuPy 数组。
    网格数据和求解状态常驻 GPU 显存。

    Attributes:
        mesh: HighOrderMesh（CPU 侧引用，用于几何查询）
        ops: FROperators
        array_mgr: GPUArrayManager 实例
        time_integrator: GPUTimeIntegrator 实例
        U_gpu: 当前守恒变量（CuPy 数组，常驻 GPU）
        Q_gpu: 当前原始变量（CuPy 数组，常驻 GPU）
    """

    def __init__(
        self,
        mesh,
        ops,
        order: int = 2,
        n_vars: int = 5,
        device_id: int = 0,
        time_scheme: str = "ssp_rk3",
        cfl: float = 1.0,
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        mu_molecular: float = 1.8e-5,
        boundary_ghost_provider=None,
        turb_model: str = "NONE",
    ):
        """初始化 GPU FRSolver。

        Args:
            mesh: HighOrderMesh 实例
            ops: FROperators 实例
            order: 多项式阶数
            n_vars: 守恒变量数
            device_id: GPU 设备 ID
            time_scheme: 时间积分方案
            cfl: CFL 数
            rho_inf, vel_inf, p_inf: 自由来流条件
            mu_molecular: 分子动力粘度
            boundary_ghost_provider: 边界幽灵态提供者
        """
        if not gpu_available:
            raise RuntimeError(
                "CuPy is not available. Install with: pip install cupy-cuda12x"
            )

        self.mesh = mesh
        self.ops = ops
        self.order = order
        self.n_vars = n_vars
        self.device_id = device_id
        self.mu_molecular = mu_molecular
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
        self.boundary_ghost_provider = boundary_ghost_provider
        self.turb_model_name = turb_model
        self.turb_model_gpu = None  # GPU 湍流模型（可选）

        # GPU 数组管理器
        self.array_mgr = GPUArrayManager(device_id=device_id)

        # 上传网格数据
        self.mesh_data = self.array_mgr.upload_mesh_data(mesh, ops)
        self.ops_data = {k: v for k, v in self.mesh_data.items()}

        # 上传面几何
        self.flat_face_gpu = None
        self._init_face_geometry()

        # 时间积分器
        self.time_integrator = GPUTimeIntegrator(scheme=time_scheme, cfl=cfl)

        # 初始化求解状态
        n_cells = mesh.n_cells
        n_sps = mesh.n_sps_per_cell

        cp = get_cupy()
        with cp.cuda.Device(device_id):
            # 均匀初场
            self.U_gpu = cp.zeros((n_cells, n_sps, n_vars), dtype=cp.float64)
            self.U_gpu[:, :, 0] = rho_inf
            self.U_gpu[:, :, 1] = rho_inf * vel_inf
            self.U_gpu[:, :, 4] = p_inf / (1.4 - 1.0) + 0.5 * rho_inf * vel_inf**2

            self.Q_gpu = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
            self._update_primitives_gpu()

        # 初始化 GPU 湍流模型
        if turb_model == "SST":
            from autoflowcfd.core.gpu.gpu_turbulence_sst import GPUTurbulenceSST
            self.turb_model_gpu = GPUTurbulenceSST(n_cells, n_sps, device_id)
            logger.info(f"GPU SST k-omega model initialized on device {device_id}")
            print(f"   [OK] GPU SST k-omega model initialized")

        # 预计算壁面距离（用于湍流模型）
        self.wall_distance_gpu = None
        if self.turb_model_gpu is not None:
            self._init_wall_distance_gpu()

        # 初始化 GPU 模态滤波（抑制混叠噪声）
        self.filter_func_gpu = None
        self._init_modal_filter_gpu()

        # DUAL_TIME 专用：物理时间层 n-1 的解（BDF2 时间导数项需要）
        self._dual_time_U_prev = None

        # 残差范数历史
        self.residual_history = []
        self.iteration = 0

        logger.info(
            f"GPUFRSolver initialized: {n_cells} cells, P{order}, "
            f"device {device_id}, scheme={time_scheme}, CFL={cfl}"
        )
        print(f"✅ GPUFRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Device: {device_id} ({self.array_mgr._device_name})")
        print(f"   Time Scheme: {time_scheme}, CFL: {cfl}")


    def _update_primitives_gpu(self):
        """GPU 上更新原始变量。"""
        from autoflowcfd.core.gpu.gpu_flux import conserved_to_primitive_gpu
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
            from autoflowcfd.core.gpu.gpu_p0_inviscid import (
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
            # 面法向和面积
            ffp_list = self.mesh.face_flux_points
            n_faces = fc.n_faces
            normal = np.empty((n_faces, 3), dtype=np.float64)
            area_w = np.empty((n_faces,), dtype=np.float64)
            for f in range(n_faces):
                normal[f] = ffp_list[f].true_normal[0]
                area_w[f] = ffp_list[f].true_area_weight[0]
            normal_gpu = cp.asarray(normal)
            area_w_gpu = cp.asarray(area_w)
            volumes_gpu = self.mesh_data.get('cell_volumes')
            if volumes_gpu is None:
                volumes_gpu = cp.asarray(self.mesh.get_all_cell_volumes())

            # 边界幽灵态
            Q_ghost = cp.zeros((n_faces, 5), dtype=cp.float64)

            res = compute_inviscid_residual_p0_cupy_gpu_resident(
                Q_flat, owner, neighbor, is_bnd,
                normal_gpu, area_w_gpu, volumes_gpu,
                Q_ghost, self.mesh.n_cells, n_faces,
            )
            # 扩展到 (n_cells, n_sps, 5)
            return cp.broadcast_to(res, (self.mesh.n_cells, self.mesh.n_sps_per_cell, 5)).copy()
        else:
            # P>=1 高阶 FR GPU 路径
            from autoflowcfd.core.gpu.gpu_inviscid import compute_inviscid_residual_fr_gpu
            return compute_inviscid_residual_fr_gpu(
                U, self.mesh, self.ops,
                boundary_ghost_provider=self.boundary_ghost_provider,
                mesh_data=self.mesh_data,
                ops_data=self.ops_data,
                flat_face_gpu=self.flat_face_gpu,
                device_id=self.device_id,
            )

    def compute_viscous_residual_gpu(self, U_trial=None, mu_t_field=None):
        """GPU 计算粘性残差。

        Args:
            U_trial: CuPy 数组（可选）
            mu_t_field: 湍流涡粘度 rho*nu_t (n_cells, n_sps) CuPy 数组（可选）

        Returns:
            viscous_residual: CuPy 数组 (n_cells, n_sps, 5)
        """
        from autoflowcfd.core.gpu.gpu_viscous import compute_viscous_residual_fr_gpu
        U = U_trial if U_trial is not None else self.U_gpu
        return compute_viscous_residual_fr_gpu(
            U, self.mesh, self.ops,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            mesh_data=self.mesh_data,
            ops_data=self.ops_data,
            device_id=self.device_id,
        )

    def _compute_local_time_step_gpu(self):
        """GPU 计算局部 CFL 时间步长（使用所有 SP 的谱半径）。"""
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        fc = self.mesh.face_connectivity
        ffp_list = self.mesh.face_flux_points
        n_faces = fc.n_faces

        owner_cell = cp.asarray(fc.owner_cell)
        neighbor_cell = cp.asarray(
            np.where(fc.is_boundary, 0, fc.neighbor_cell)
        )
        is_boundary = cp.asarray(fc.is_boundary)

        normal = np.empty((n_faces, 3), dtype=np.float64)
        area_w = np.empty((n_faces,), dtype=np.float64)
        for f in range(n_faces):
            normal[f] = ffp_list[f].true_normal[0]
            area_w[f] = ffp_list[f].true_area_weight[0]
        normals_gpu = cp.asarray(normal)
        areas_gpu = cp.asarray(area_w)

        cell_volumes = self.mesh_data.get('cell_volumes')
        if cell_volumes is None:
            cell_volumes = cp.asarray(self.mesh.get_all_cell_volumes())

        # 使用所有 SP 计算谱半径（取最大值），而非仅 SP0
        # 对每个 SP 独立计算 CFL 步长，然后取 cell 内最小值
        dt_all_sps = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        for sp in range(n_sps):
            U_sp = self.U_gpu[:, sp:sp+1, :]  # (n_cells, 1, n_vars)
            dt_sp = compute_local_cfl_step_gpu(
                U_sp, cell_volumes,
                owner_cell, neighbor_cell, is_boundary,
                normals_gpu, areas_gpu,
                None, None,
                cfl=self.time_integrator.cfl,
            )
            dt_all_sps[:, sp] = dt_sp

        return cp.min(dt_all_sps, axis=1)  # (n_cells,)

    def step(self, dt: float = 0.0) -> float:
        """执行一个时间步。

        完整流程（与 CPU 版 step() 对应）：
        1. 更新原始变量
        2. 湍流源项求值（算子分裂：湍流走独立显式更新）
        3. 局部 CFL 步长
        4. 平均流残差计算（含湍流涡粘耦合）
        5. SSP-RK / IMEX / DUAL_TIME 时间推进
        6. 湍流场更新（k/ω 正性限制）

        Args:
            dt: 物理时间步长（稳态模式下被局部 CFL 步长覆盖，
                DUAL_TIME 模式下是真正的物理时间步长）

        Returns:
            residual_norm: 残差范数
        """
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        self._update_primitives_gpu()

        # 湍流源项在当前状态下求值（算子分裂）
        mu_t_field = self.compute_turbulence_source_gpu()

        # 局部 CFL 步长
        dt_local = self._compute_local_time_step_gpu()
        dt_local_full = cp.broadcast_to(
            dt_local[:, None], (n_cells, n_sps)
        ).reshape(n_cells * n_sps)

        # 展平 U 用于时间积分器
        U_flat = self.U_gpu.reshape(n_cells * n_sps, self.n_vars)

        # 构建平均流残差函数（含湍流涡粘耦合）
        def mean_flow_residual(U_flat_trial):
            U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
            inv_res = self.compute_inviscid_residual_gpu(U_trial)
            visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field)
            total = inv_res + visc_res
            return -total.reshape(n_cells * n_sps, self.n_vars)

        # 初始残差
        residual0 = mean_flow_residual(U_flat)

        # 根据时间方案选择推进方式
        scheme = self.time_integrator.scheme

        if scheme == "dual_time":
            # DUAL_TIME: 真正时间精度的物理时间推进
            if self._dual_time_U_prev is None:
                # 第一个物理步：BDF1
                U_new_flat = self.time_integrator.step_dual_time(
                    U_flat, mean_flow_residual, dt_local_full,
                    dt_physical=dt,
                    solution_prev=None,
                    max_inner_iter=self.time_integrator.dual_time_steps if hasattr(self.time_integrator, 'dual_time_steps') else 5,
                    filter_func=self.filter_func_gpu,
                )
            else:
                # 后续物理步：BDF2
                U_new_flat = self.time_integrator.step_dual_time(
                    U_flat, mean_flow_residual, dt_local_full,
                    dt_physical=dt,
                    solution_prev=self._dual_time_U_prev,
                    max_inner_iter=self.time_integrator.dual_time_steps if hasattr(self.time_integrator, 'dual_time_steps') else 5,
                    filter_func=self.filter_func_gpu,
                )
            # 保存当前解作为下一步的 prev
            self._dual_time_U_prev = U_flat.copy()

        elif scheme == "imex_euler":
            # IMEX: 显式处理对流，隐式处理粘性
            def convective_residual_only(U_flat_trial):
                U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
                inv_res = self.compute_inviscid_residual_gpu(U_trial)
                return -inv_res.reshape(n_cells * n_sps, self.n_vars)

            def diffusive_residual_only(U_flat_trial):
                U_trial = U_flat_trial.reshape(n_cells, n_sps, self.n_vars)
                visc_res = self.compute_viscous_residual_gpu(U_trial, mu_t_field=mu_t_field)
                return -visc_res.reshape(n_cells * n_sps, self.n_vars)

            U_new_flat = self.time_integrator.step_imex(
                U_flat, convective_residual_only, diffusive_residual_only,
                dt_local_full, p_floor=1.0,
            )

        else:
            # SSP-RK2/RK3 or Forward Euler
            U_new_flat = self.time_integrator.step(
                U_flat, mean_flow_residual, dt_local_full,
                p_floor=1.0, residual0=residual0,
                filter_func=self.filter_func_gpu,
            )

        self.U_gpu = U_new_flat.reshape(n_cells, n_sps, self.n_vars)
        self._update_primitives_gpu()

        # 残差范数
        residual_norm = float(cp.linalg.norm(residual0) / max(1, np.sqrt(residual0.size)))
        self.residual_history.append(residual_norm)
        self.iteration += 1

        return residual_norm

    def solve(
        self,
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        output_interval: int = 10,
    ) -> Dict[str, Any]:
        """执行稳态求解循环。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长（稳态模式下被 CFL 覆盖）
            tol: 收敛容差
            output_interval: 输出间隔

        Returns:
            结果字典
        """
        print(f"Starting GPU solve: max_iter={max_iter}, tol={tol}")
        converged = False
        final_residual = 1e10

        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res

            if i == 0 or (i + 1) % output_interval == 0:
                mem = self.array_mgr.get_memory_usage()
                print(
                    f"GPU Iter {i+1}: Residual = {res:.6e} | "
                    f"Time/step: {t_end-t_start:.3f}s | "
                    f"GPU mem: {mem['used_mb']:.0f}/{mem['total_mb']:.0f} MB"
                )

            if res < tol:
                converged = True
                print(f"✅ GPU Converged at iteration {i+1} with residual {res:.6e}")
                break

            if not np.isfinite(res):
                print(f"❌ GPU Diverged at iteration {i+1} with residual {res}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }

