"""
AutoFlowCFD V2.0 - 多 GPU + MPI 分布式求解器

将 GPU 计算与 MPI 域分解结合：每个 MPI rank 使用一块 GPU 进行计算。

设计：
- 继承 DistributedFRSolver 的 MPI 基础设施（分区、halo 交换、全局归约）
- 残差计算使用 GPU 版本（gpu_inviscid.py / gpu_viscous.py）
- 数据常驻各 rank 的 GPU 显存
- GPU 直接 Halo 交换（gpu_halo_exchange.py）：
  - CUDA-aware MPI：零拷贝 GPU↔GPU
  - 非 CUDA-aware：staging buffer 优化（只传输 send/recv 列表数据）
- SSP-RK2/RK3 时间推进：每个 stage 执行 halo 交换 + 残差评估
- 全局残差归约：MPI Allreduce

使用:
    mpirun -np 4 autoflowcfd solve steady <grid> --backend gpu --multi-gpu
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
from autoflowcfd.core.mpi import get_rank, get_size, is_root, mpi_available
from autoflowcfd.core.mpi.partition import (
    partition_mesh, build_distributed_partition, DistributedPartition
)
from autoflowcfd.core.gpu.gpu_halo_exchange import GPUHaloExchange
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.distributed_flat_face import (
    DistributedFlatFaceGeometry, build_distributed_flat_face
)
from autoflowcfd.core.mpi.comm import allreduce_sum, allreduce_min, barrier
from autoflowcfd.core.gpu.gpu_distributed_init import _GPUDistributedInitMixin


class MultiGPUDistributedSolver(_GPUDistributedInitMixin):
    """多 GPU + MPI 分布式求解器。

    每个 MPI rank 绑定一块 GPU，使用 GPU 进行所有计算，
    通过 MPI 进行 halo 交换和全局归约。

    Attributes:
        partition: 本 rank 的分区信息
        halo_exchange: halo 交换管理器
        array_mgr: GPU 数组管理器
        time_integrator: GPU 时间积分器
        U_gpu: 本 rank 的守恒变量（GPU 常驻）
        rank: 当前 MPI rank
        n_ranks: 总 rank 数
        device_id: 本 rank 使用的 GPU 设备 ID
    """

    def __init__(
        self,
        mesh,
        ops,
        n_ranks: int,
        face_connectivity_data=None,
        partition_info=None,
        rank: Optional[int] = None,
        device_id: Optional[int] = None,
        time_scheme: str = "ssp_rk3",
        cfl: float = 1.0,
        mu_molecular: float = 1.8e-5,
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        turb_model: str = "NONE",
    ):
        """初始化多 GPU 分布式求解器。

        Args:
            mesh: HighOrderMesh（局部网格或完整网格）
            ops: FROperators
            n_ranks: MPI rank 总数
            face_connectivity_data: 局部面连接关系数据（分布式加载模式）
            partition_info: 分区信息（分布式加载模式）
            rank: 当前 rank（默认从 MPI 获取）
            device_id: GPU 设备 ID（默认 rank % n_gpus）
            time_scheme: 时间积分方案
            cfl: CFL 数
            mu_molecular: 分子动力粘度
            rho_inf, vel_inf, p_inf: 自由来流条件
        """
        if not gpu_available:
            raise RuntimeError("CuPy required for multi-GPU solver")
        if not mpi_available:
            raise RuntimeError("MPI required for distributed solver")

        cp = get_cupy()
        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        self.mesh = mesh
        self.ops = ops
        self.mu_molecular = mu_molecular
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
        self.turb_model_name = turb_model

        # GPU 设备选择：默认 round-robin 分配
        if device_id is None:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            device_id = self.rank % n_gpus
        self.device_id = device_id

        # 切换到指定 GPU
        with cp.cuda.Device(device_id):
            logger.info(f"Rank {self.rank} using GPU device {device_id}")

        # 初始化 GPU 数组管理器
        self.array_mgr = GPUArrayManager(device_id=device_id)

        # 分区
        if partition_info is not None:
            # 分布式加载模式：已有分区信息
            self.partition = self._rebuild_partition(partition_info)
            self._using_distributed_mesh = True
        else:
            # 传统模式：所有 rank 有完整网格，执行分区
            fc = mesh.face_connectivity
            if self.rank == 0:
                cell_partition = partition_mesh(fc, n_ranks)
            else:
                cell_partition = None
            from autoflowcfd.core.mpi.comm import bcast_from_root
            if n_ranks > 1:
                cell_partition = bcast_from_root(cell_partition)
            self.partition = build_distributed_partition(
                fc, cell_partition, self.rank, n_ranks
            )
            self._using_distributed_mesh = False

        # GPU 直接 Halo 交换（支持 CUDA-aware MPI 和 staging buffer 两种模式）
        self.gpu_halo = GPUHaloExchange(
            self.partition, n_sps=n_sps, n_vars=5, device_id=device_id
        )

        # 分布式状态
        n_cells = mesh.n_cells
        n_sps = mesh.n_sps_per_cell
        n_local = self.partition.n_local_cells
        self.state = DistributedFRState(n_cells, n_sps, 5, self.partition)

        # 上传局部网格数据到 GPU
        self.mesh_data = self.array_mgr.upload_mesh_data(mesh, ops)

        # 初始化局部面几何
        self.flat_face_gpu = None
        self._init_distributed_face_geometry()

        # 时间积分器
        self.time_integrator = GPUTimeIntegrator(scheme=time_scheme, cfl=cfl)

        # 初始化 GPU 湍流模型（与单机版一致）
        self.turb_model_gpu = None
        if turb_model == "SST":
            from autoflowcfd.core.gpu.gpu_turbulence_sst import GPUTurbulenceSST
            n_local_cells = self.partition.n_local_cells
            self.turb_model_gpu = GPUTurbulenceSST(n_local_cells, n_sps, device_id)
            logger.info(f"Rank {self.rank}: GPU SST model initialized")

        # 预计算壁面距离
        self.wall_distance_gpu = None
        if self.turb_model_gpu is not None:
            self._init_wall_distance_distributed()

        # 初始化 GPU 模态滤波
        self.filter_func_gpu = None
        self._init_modal_filter_distributed()

        # 初始化 GPU 上的求解状态
        with cp.cuda.Device(device_id):
            self.U_gpu = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
            self.U_gpu[:, :, 0] = rho_inf
            self.U_gpu[:, :, 1] = rho_inf * vel_inf
            self.U_gpu[:, :, 4] = p_inf / (1.4 - 1.0) + 0.5 * rho_inf * vel_inf**2

        self.residual_history = []
        self.iteration = 0

        barrier()
        if is_root():
            logger.info(
                f"MultiGPUDistributedSolver initialized: {n_ranks} ranks, "
                f"{n_cells} cells/rank (local: {n_local})"
            )
            print(f"✅ MultiGPUDistributedSolver Ready:")
            print(f"   Ranks: {n_ranks}, Cells/rank: {n_local}")
            print(f"   GPU device: {device_id} per rank")

    def _halo_exchange_gpu(self):
        """GPU 直接 halo 交换（优化版）。

        支持两种模式：
        1. CUDA-aware MPI：GPU buffer 直接通信（零拷贝）
        2. Staging buffer：GPU→CPU→MPI→CPU→GPU（只传输必要数据）
        """
        self.U_extended_gpu = self.gpu_halo.exchange(self.U_gpu)

    def compute_inviscid_residual_gpu(self):
        """GPU 计算分布式无粘残差。"""
        cp = get_cupy()
        from autoflowcfd.core.gpu.gpu_inviscid import compute_inviscid_residual_fr_gpu
        return compute_inviscid_residual_fr_gpu(
            self.U_gpu, self.mesh, self.ops,
            mesh_data=self.mesh_data,
            ops_data=self.mesh_data,
            flat_face_gpu=self.flat_face_gpu,
            device_id=self.device_id,
        )

    def compute_viscous_residual_gpu(self):
        """GPU 计算分布式粘性残差。"""
        from autoflowcfd.core.gpu.gpu_viscous import compute_viscous_residual_fr_gpu
        return compute_viscous_residual_fr_gpu(
            self.U_gpu, self.mesh, self.ops,
            mu=self.mu_molecular,
            mesh_data=self.mesh_data,
            ops_data=self.mesh_data,
            device_id=self.device_id,
        )

    def _compute_local_time_step_gpu(self):
        """GPU 计算局部 CFL 步长。"""
        cp = get_cupy()
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

        return compute_local_cfl_step_gpu(
            self.U_gpu, cell_volumes,
            owner_cell, neighbor_cell, is_boundary,
            normals_gpu, areas_gpu,
            None, None,
            cfl=self.time_integrator.cfl,
        )

    def _compute_total_residual_gpu(self, mu_t_field=None):
        """计算总残差（无粘 + 粘性），先执行 halo 交换。

        Args:
            mu_t_field: 动力涡粘度 (n_cells, n_sps) CuPy 数组（可选）
        """
        self._halo_exchange_gpu()
        inv_res = self.compute_inviscid_residual_gpu()
        visc_res = self.compute_viscous_residual_gpu(mu_t_field=mu_t_field)
        return inv_res + visc_res

    def step(self, dt: float = 0.0) -> float:
        """执行一个分布式时间步（SSP-RK 多 stage）。

        流程（SSP-RK3 为例，每个 stage 都重新计算残差+halo 交换）：
        Stage 0: 计算初始残差 R(U^n)
        Stage 1: U^(1) = U^n + dt*L(U^n); halo 交换; 计算 R(U^(1))
        Stage 2: U^(2) = 3/4*U^n + 1/4*(U^(1) + dt*L(U^(1))); halo; R(U^(2))
        Stage 3: U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt*L(U^(2)))

        对于 SSP-RK2 和 Forward Euler 类似处理。

        Args:
            dt: 时间步长

        Returns:
            residual_norm: 全局残差范数
        """
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        # 局部 CFL 步长
        dt_local = self._compute_local_time_step_gpu()
        dt_local_full = cp.broadcast_to(
            dt_local[:, None], (n_cells, n_sps)
        ).reshape(n_cells * n_sps)
        dt_flat = dt_local_full[:, None]  # (N, 1)

        # RK 系数表
        scheme = self.time_integrator.scheme
        table = self.time_integrator._table
        alpha = table["alpha"]
        beta = table["beta"]
        n_stages = table["stages"]

        # 湍流源项求值（算子分裂，每个 step 开始时计算一次）
        mu_t_field = self._compute_turbulence_source_distributed()

        U_flat = self.U_gpu.reshape(n_cells * n_sps, 5)
        U0 = U_flat.copy()

        # === Stage 0: 初始残差 ===
        res0 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res0_flat = res0.reshape(n_cells * n_sps, 5)
        L0 = -res0_flat

        # === Stage 1 ===
        U_stage1 = U0 + dt_flat * L0
        enforce_positivity_gpu(U_stage1)
        # 模态滤波
        if self.filter_func_gpu is not None:
            U_stage1 = self.filter_func_gpu(U_stage1)
        self.U_gpu = U_stage1.reshape(n_cells, n_sps, 5)

        if n_stages == 1:
            residual_norm = self._global_residual_norm(res0_flat)
            self.residual_history.append(residual_norm)
            self.iteration += 1
            return residual_norm

        # === Stage 2 ===
        res1 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res1_flat = res1.reshape(n_cells * n_sps, 5)
        L1 = -res1_flat
        U_stage2 = (alpha[1][0] * U0 + alpha[1][1] * U_stage1 + beta[1] * dt_flat * L1)
        enforce_positivity_gpu(U_stage2)
        if self.filter_func_gpu is not None:
            U_stage2 = self.filter_func_gpu(U_stage2)
        self.U_gpu = U_stage2.reshape(n_cells, n_sps, 5)

        if n_stages == 2:
            residual_norm = self._global_residual_norm(res1_flat)
            self.residual_history.append(residual_norm)
            self.iteration += 1
            return residual_norm

        # === Stage 3 (RK3) ===
        res2 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res2_flat = res2.reshape(n_cells * n_sps, 5)
        L2 = -res2_flat
        U_stage3 = (alpha[2][0] * U0 + alpha[2][1] * U_stage1 +
                    alpha[2][2] * U_stage2 + beta[2] * dt_flat * L2)
        enforce_positivity_gpu(U_stage3)
        if self.filter_func_gpu is not None:
            U_stage3 = self.filter_func_gpu(U_stage3)
        self.U_gpu = U_stage3.reshape(n_cells, n_sps, 5)

        residual_norm = self._global_residual_norm(res2_flat)
        self.residual_history.append(residual_norm)
        self.iteration += 1
        return residual_norm

    def _global_residual_norm(self, res_flat) -> float:
        """MPI 全局残差归约。"""
        cp = get_cupy()
        local_norm_sq = float(cp.sum(res_flat ** 2))
        global_norm_sq = allreduce_sum(local_norm_sq)
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell
        n_global = n_cells * n_sps * 5 * self.n_ranks
        return np.sqrt(global_norm_sq / max(1, n_global))

    def solve(
        self,
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        output_interval: int = 10,
    ) -> Dict[str, Any]:
        """执行分布式稳态求解循环。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 收敛容差
            output_interval: 输出间隔

        Returns:
            结果字典
        """
        if is_root():
            print(f"Starting multi-GPU solve: {self.n_ranks} ranks, max_iter={max_iter}")

        converged = False
        final_residual = 1e10

        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res

            if is_root():
                if i == 0 or (i + 1) % output_interval == 0:
                    print(
                        f"Multi-GPU Iter {i+1}: Residual = {res:.6e} | "
                        f"Time/step: {t_end-t_start:.3f}s"
                    )

            if res < tol:
                converged = True
                if is_root():
                    print(f"✅ Multi-GPU Converged at iteration {i+1}")
                break

            if not np.isfinite(res):
                if is_root():
                    print(f"❌ Multi-GPU Diverged at iteration {i+1}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }

