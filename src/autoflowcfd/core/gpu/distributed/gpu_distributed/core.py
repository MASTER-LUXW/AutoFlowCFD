"""
AutoFlowCFD V2.0 - 多 GPU + MPI 分布式求解器（主类与装配顺序）

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

2026-09-25 按职责拆成同目录的 mixin（`setup` 装配阶段、`residual` 残差、
`timestep` 局部步长、`stepping` 推进与求解循环）与 `compact_view`。
"""

from typing import Optional

from loguru import logger

from autoflowcfd.core.gpu.distributed.gpu_distributed_init import _GPUDistributedInitMixin
from autoflowcfd.core.mpi import is_root
from autoflowcfd.core.mpi.comm import barrier

from .residual import _MultiGPUResidualMixin
from .setup import _MultiGPUSetupMixin
from .stepping import _MultiGPUSteppingMixin
from .timestep import _MultiGPUTimeStepMixin


class MultiGPUDistributedSolver(_MultiGPUSetupMixin, _MultiGPUSteppingMixin, _MultiGPUResidualMixin,
                                _MultiGPUTimeStepMixin, _GPUDistributedInitMixin):
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
        mu_molecular: float = 1.8e-5,
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        # 攻角/侧滑角（度）。0/0 时来流严格沿 +x，与此前把方向硬编码
        # 成 +x 的行为逐位相同。约定见 core/utils/flow_direction.py。
        aoa_deg: float = 0.0,
        aos_deg: float = 0.0,
        turb_model: str = "NONE",
        turbulence_intensity: float = 0.01,
        viscosity_ratio: float = 5.0,
        cfl_start: Optional[float] = None,
        cfl_max: Optional[float] = None,
        cfl_min: Optional[float] = None,
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
            cfl_start, cfl_max, cfl_min: 自适应 CFL 参数（与单机 FRSolver/
                GPUFRSolver 同名参数同一语义；None 取控制器默认值，规则见
                `adaptive_cfl/policy.py::build_cfl_policy`）。
            mu_molecular: 分子动力粘度
            rho_inf, vel_inf, p_inf: 自由来流条件
        """
        self._setup_identity_and_device(
            mesh, ops, n_ranks, rank, device_id, mu_molecular,
            rho_inf, vel_inf, p_inf, aoa_deg, aos_deg, turb_model)
        n_sps, n_local = self._setup_partition_and_geometry(mesh, ops, n_ranks, partition_info)
        self._setup_time_integration(time_scheme, cfl_start, cfl_max, cfl_min)
        self._setup_turbulence(mesh, turb_model, n_sps, mu_molecular, rho_inf, vel_inf,
                               turbulence_intensity, viscosity_ratio)
        self._setup_state_and_boundary(n_local, n_sps)

        self.residual_history = []
        self.iteration = 0

        barrier()
        if is_root():
            logger.info(
                f"MultiGPUDistributedSolver initialized: {n_ranks} ranks, "
                f"{n_local} local cells/rank (+{self.partition.n_halo} halo)"
            )
            print("✅ MultiGPUDistributedSolver Ready:")
            print(f"   Ranks: {n_ranks}, Cells/rank: {n_local} (+{self.partition.n_halo} halo)")
            print(f"   GPU device: {self.device_id} per rank")
