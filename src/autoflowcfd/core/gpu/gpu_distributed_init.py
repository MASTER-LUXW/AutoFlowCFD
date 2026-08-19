"""MultiGPUDistributedSolver 初始化和 I/O 混入类。

从 gpu_distributed.py 拆出，控制单文件行数。
"""

import numpy as np
from loguru import logger

from autoflowcfd.core.gpu import get_cupy


class _GPUDistributedInitMixin:
    """MultiGPUDistributedSolver 初始化/I/O 混入。"""

    def _rebuild_partition(self, partition_info):
        """从分区信息字典重建 DistributedPartition。"""
        return partition_info

    def _init_wall_distance_distributed(self):
        """分布式壁面距离计算（使用局部网格）。"""
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell

        try:
            if hasattr(self.mesh, 'sps_coords') and self.mesh.sps_coords is not None:
                sps_coords = self.mesh.sps_coords.reshape(-1, 3)
            elif hasattr(self.mesh, 'cell_centers') and self.mesh.cell_centers is not None:
                sps_coords = np.tile(self.mesh.cell_centers, (1, n_sps)).reshape(-1, 3)
            else:
                logger.warning(f"Rank {self.rank}: No SP coords for wall distance")
                self.wall_distance_gpu = cp.ones((n_local, n_sps), dtype=cp.float64) * 0.01
                return

            wall_indices = None
            if hasattr(self.mesh, 'boundary_groups'):
                for bg_name, bg in self.mesh.boundary_groups.items():
                    if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                        wall_indices = bg.get('node_indices')
                        break

            if wall_indices is not None and len(wall_indices) > 0:
                from scipy.spatial import cKDTree
                wall_coords = self.mesh.nodes[wall_indices]
                tree = cKDTree(wall_coords)
                dist_flat, _ = tree.query(sps_coords, k=1)
                self.wall_distance_gpu = cp.asarray(
                    dist_flat.reshape(n_local, n_sps)
                )
            else:
                volumes = self.mesh_data.get('cell_volumes')
                if volumes is None:
                    volumes = cp.asarray(self.mesh.get_all_cell_volumes())
                h_char = volumes ** (1.0 / 3.0)
                self.wall_distance_gpu = cp.broadcast_to(
                    h_char[:, None], (n_local, n_sps)
                ).copy()
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Wall distance failed: {e}")
            self.wall_distance_gpu = cp.ones((n_local, n_sps), dtype=cp.float64) * 0.01

    def _init_modal_filter_distributed(self):
        """分布式模态滤波初始化。"""
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell
        n_prism = self.mesh.n_prism_cells

        try:
            filter_prism = self.ops.filter_prism
            filter_tet = self.ops.filter_tet

            if filter_prism is not None or filter_tet is not None:
                from autoflowcfd.core.gpu.gpu_modal_filter import build_gpu_filter_func
                self.filter_func_gpu = build_gpu_filter_func(
                    n_local, n_sps, n_prism,
                    filter_prism, filter_tet,
                    device_id=self.device_id,
                )
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Modal filter init failed: {e}")

    def _init_distributed_face_geometry(self):
        """初始化分布式面几何（GPU 版）。"""
        try:
            from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
            flat_face = get_flat_face_geometry(self.mesh, self.ops)
            # 构建分布式面几何
            from autoflowcfd.core.mpi.distributed_flat_face import (
                build_distributed_flat_face,
            )
            dist_flat_face = build_distributed_flat_face(
                flat_face, self.partition, self.rank, self.n_ranks
            )
            # 上传到 GPU
            from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
            self.flat_face_gpu = build_gpu_flat_face(dist_flat_face, self.device_id)
        except Exception as e:
            logger.warning(f"Distributed face geometry init failed: {e}")
            self.flat_face_gpu = None

    def _halo_exchange_gpu(self):
        """GPU 直接 halo 交换（优化版）。

        支持两种模式：
        1. CUDA-aware MPI：GPU buffer 直接通信（零拷贝）
        2. Staging buffer：GPU→CPU→MPI→CPU→GPU（只传输必要数据）
        """
        self.U_extended_gpu = self.gpu_halo.exchange(self.U_gpu)

    def _compute_turbulence_source_distributed(self):
        """分布式湍流源项计算（与单机版一致）。"""
        if self.turb_model_gpu is None:
            return None

        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell

        from autoflowcfd.core.gpu.gpu_gradients import (
            compute_physical_gradient_gpu,
            compute_physical_scalar_gradient_gpu,
        )

        # 速度梯度
        grad_U = compute_physical_gradient_gpu(
            self.U_gpu[..., :5], self.mesh_data, self.ops_data,
        )

        # 壁面距离
        d_wall = self.wall_distance_gpu
        if d_wall is None:
            d_wall = cp.ones((n_local, n_sps), dtype=cp.float64) * 0.01

        # k/ω 梯度
        grad_k = compute_physical_scalar_gradient_gpu(
            self.turb_model_gpu.k_field, self.mesh_data, self.ops_data,
        )
        grad_omega = compute_physical_scalar_gradient_gpu(
            self.turb_model_gpu.omega_field, self.mesh_data, self.ops_data,
        )

        # 梯度限幅
        max_grad_mag = 1e6
        grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
        if cp.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / cp.maximum(grad_k_mag, 1e-10)
            grad_k *= cp.clip(scale_k, 0, 1)[..., None]
        if cp.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / cp.maximum(grad_omega_mag, 1e-10)
            grad_omega *= cp.clip(scale_omega, 0, 1)[..., None]

        # SST 源项
        Sk, S_omega = self.turb_model_gpu.compute_source_terms_gpu(
            self.Q_gpu, grad_U, d_wall, self.mu_molecular,
            grad_k, grad_omega,
        )

        # 更新湍流场
        rho = self.Q_gpu[:, :, 0]
        dk_dt = Sk / cp.maximum(rho, 1e-10)
        domega_dt = S_omega / cp.maximum(rho, 1e-10)
        dt_local = self._compute_local_time_step_gpu()
        dt_mean = float(cp.mean(dt_local))
        self.turb_model_gpu.update_fields_gpu(dt_mean, dk_dt, domega_dt)

        return rho * self.turb_model_gpu.nu_t

    def get_state_cpu(self):
        """下载 GPU 状态到 CPU。"""
        cp = get_cupy()
        return {'U': cp.asnumpy(self.U_gpu)}

    def cleanup(self):
        """释放 GPU 资源。"""
        self.array_mgr.cleanup()

    def save_checkpoint_distributed(self, path: str):
        """分布式 checkpoint 保存。"""
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_checkpoint as _dist_save,
        )
        U_local_cpu = cp.asnumpy(self.U_gpu)
        local_cells = self.partition.local_cells
        n_global_cell = self.mesh.n_cells * self.n_ranks
        _dist_save(U_local_cpu, local_cells, n_global_cell, path, self.rank, self.n_ranks)

        from autoflowcfd.core.mpi import is_root
        if is_root():
            logger.info(f"Distributed GPU checkpoint saved to {path}")

    def load_checkpoint_distributed(self, path: str):
        """分布式 checkpoint 加载。"""
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_load_checkpoint as _dist_load,
        )
        local_cells = self.partition.local_cells
        U_local_cpu = _dist_load(path, local_cells, self.rank, self.n_ranks)
        self.U_gpu = cp.asarray(U_local_cpu)

        from autoflowcfd.core.mpi import is_root
        if is_root():
            logger.info(f"Distributed GPU checkpoint loaded from {path}")
