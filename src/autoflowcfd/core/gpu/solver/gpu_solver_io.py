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

    def compute_turbulence_source_gpu(self):
        """GPU 计算湍流模型源项。

        完整流程：
        1. 计算速度梯度（GPU）
        2. 计算 k/ω 的真实物理梯度（GPU）
        3. 使用预计算的壁面距离
        4. 计算 SST 源项
        5. 更新 k/ω 场（含正性限制器）

        Returns:
            mu_t: 动力涡粘度 rho*nu_t (n_cells, n_sps) CuPy 数组，无湍流模型时返回 None
        """
        if self.turb_model_gpu is None:
            return None

        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        from autoflowcfd.core.gpu.gpu_gradients import (
            compute_physical_gradient_gpu,
            compute_physical_scalar_gradient_gpu,
        )
        grad_U = compute_physical_gradient_gpu(
            self.U_gpu[..., :5], self.mesh_data, self.ops_data,
        )

        d_wall = self.wall_distance_gpu
        if d_wall is None:
            d_wall = cp.ones((n_cells, n_sps), dtype=cp.float64) * 0.01

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

        Sk, S_omega = self.turb_model_gpu.compute_source_terms_gpu(
            self.Q_gpu, grad_U, d_wall, self.mu_molecular,
            grad_k, grad_omega,
        )

        rho = self.Q_gpu[:, :, 0]
        dk_dt = Sk / cp.maximum(rho, 1e-10)
        domega_dt = S_omega / cp.maximum(rho, 1e-10)

        dt_local = self._compute_local_time_step_gpu()
        dt_mean = cp.mean(dt_local)
        self.turb_model_gpu.update_fields_gpu(float(dt_mean), dk_dt, domega_dt)

        return rho * self.turb_model_gpu.nu_t
