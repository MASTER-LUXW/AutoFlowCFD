"""
AutoFlowCFD V2.0 - 多 GPU + MPI 分布式求解器（主类与装配顺序）

将 GPU 计算与 MPI 域分解结合：每个 MPI rank 使用一块 GPU 进行计算。

设计：
- 继承 DistributedFRSolver 的 MPI 基础设施（分区、halo 交换、全局归约）
- 残差计算使用 GPU 版本（gpu_inviscid/ / gpu_viscous/）
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

from autoflowcfd.core.gpu.distributed.gpu_distributed_init import _GPUDistributedInitMixin
from autoflowcfd.core.mpi.distributed_solver.assembly import DistributedRunSettings, canonical_turbulence_model

from .residual import _MultiGPUResidualMixin
from .setup import _MultiGPUSetupMixin
from .stepping import _MultiGPUSteppingMixin
from .timestep import _MultiGPUTimeStepMixin

from autoflowcfd.core.turbulence.wall_model.treatment import DEFAULT_WALL_TREATMENT
from autoflowcfd.core.fr_solver.boundary.constants import _SEM_DEFAULT_NUM_EDDIES
from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS
from autoflowcfd.core.fr_solver.solver.solve_loop import DistributedSolveLoopMixin


class MultiGPUDistributedSolver(_MultiGPUSetupMixin, _MultiGPUSteppingMixin, _MultiGPUResidualMixin,
                                _MultiGPUTimeStepMixin, _GPUDistributedInitMixin, DistributedSolveLoopMixin):
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
        *,
        time_scheme,
        mu_molecular: float,
        rho_inf: float,
        vel_inf: float,
        p_inf: float,
        turb_model_name: str,
        rank: Optional[int] = None,
        device_id: Optional[int] = None,
        # 攻角/侧滑角（度）。0/0 时来流严格沿 +x，与此前把方向硬编码
        # 成 +x 的行为逐位相同。约定见 core/utils/flow_direction.py。
        aoa_deg: float = 0.0,
        aos_deg: float = 0.0,
        turbulence_intensity: float = 0.01,
        viscosity_ratio: float = 5.0,
        cfl_start: Optional[float] = None,
        cfl_max: Optional[float] = None,
        cfl_min: Optional[float] = None,
        adaptive_cfl: bool = True,
        time_accurate: bool = False,
        low_mach_precond: bool = True,
        wall_distance_source=None,
        artificial_viscosity_enabled: bool = False,
        artificial_viscosity_alpha: float = 1.0,
        sem_num_eddies: Optional[int] = _SEM_DEFAULT_NUM_EDDIES,
        wall_treatment: str = DEFAULT_WALL_TREATMENT,
        dual_time_inner_iter: int = DEFAULT_DUAL_TIME_STEPS,
    ):
        """初始化多 GPU 分布式求解器（"传统模式"：每个 rank 持有完整全局网格，root 用 METIS 分区后广播）。
        只有 root 持有完整网格的"完全分布式加载"走 `from_fully_distributed_package`。

        Args:
            mesh: 完整全局 HighOrderMesh
            ops: FROperators
            n_ranks: MPI rank 总数
            rank: 当前 rank（默认从 MPI 获取）
            device_id: GPU 设备 ID（默认 rank % n_gpus）
            time_scheme: 时间积分方案
            cfl_start, cfl_max, cfl_min: 自适应 CFL 参数（与单机 FRSolver/
                GPUFRSolver 同名参数同一语义；None 取控制器默认值，规则见
                `adaptive_cfl/policy.py::build_cfl_policy`）。
            adaptive_cfl, time_accurate, low_mach_precond: 与单机同名参数同一语义
                （`core/time_integration/time_accurate.py::configure_time_stepping`）
            mu_molecular: 分子动力粘度
            rho_inf, vel_inf, p_inf: 自由来流条件
            artificial_viscosity_enabled, artificial_viscosity_alpha: 问题单元人工
                粘性（与单机 FRSolver 同名参数、同一实现）
            sem_num_eddies: 合成湍流入口（SEM）的涡核数（与单机同名参数；此前写死 200）
            wall_treatment: RANS 近壁处理方式（与单机同名参数，`turbulence/wall_model/treatment.py`）
            turb_model_name: 湍流模型名（大小写不敏感，与单机 / CPU 分布式同名参数）
            dual_time_inner_iter: dual-time 每个物理步的内迭代次数上限（与单机同名参数）；2026-10-20 以前多 GPU
                没有这个构造参数，由调用方在构造后改 `time_integrator.dual_time_steps`
        """
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref

        settings = DistributedRunSettings(
            turb_model_name=canonical_turbulence_model(turb_model_name),
            freestream={"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf,
                        "mach_ref": resolve_mach_ref(rho_inf, vel_inf, p_inf),
                        "aoa_deg": float(aoa_deg), "aos_deg": float(aos_deg)},
            mu_molecular=mu_molecular, turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
            sem_num_eddies=sem_num_eddies, artificial_viscosity_enabled=artificial_viscosity_enabled,
            artificial_viscosity_alpha=artificial_viscosity_alpha, wall_treatment=wall_treatment,
            time_scheme=time_scheme, dual_time_inner_iter=dual_time_inner_iter, time_accurate=time_accurate,
            adaptive_cfl=adaptive_cfl, cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            low_mach_precond=low_mach_precond)
        # 壁面距离来源（core/utils/wall_distance）：与单机 CPU 同一个，CLI 由体网格 WALL 边界面构造后传入；
        # 湍流模型需要壁距时必须提供
        self._wall_distance_source = wall_distance_source
        self._setup_identity_and_device(rank, n_ranks, device_id, mesh.order, settings)
        self._is_fully_distributed = False
        self.mesh = mesh
        self.ops = ops
        self._setup_partition_and_geometry(mesh, ops, n_ranks)
        self._setup_time_integration(settings)
        models = self._setup_turbulence_models(mesh.n_sps_per_cell)
        if models.des is not None:
            # DDES/IDDES：逐单元网格尺度只依赖几何，传统模式下 self.mesh 是完整全局网格，算一次、按
            # compact_global_ids 切片；全局结果保留供换阶后重新切片（几何量与阶数无关）
            from autoflowcfd.core.gpu import get_cupy
            from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn

            cp = get_cupy()
            self._iddes_h_max_global, self._iddes_h_wn_global = compute_h_max_and_h_wn(mesh)
            compact_global_ids = self.dist_flat_face.compact_global_ids
            with cp.cuda.Device(self.device_id):
                self.iddes_h_max_compact = cp.asarray(self._iddes_h_max_global[compact_global_ids])
                self.iddes_h_wn_compact = cp.asarray(self._iddes_h_wn_global[compact_global_ids])
        if self.turb_model_gpu is not None:
            self._init_wall_distance_distributed()

        # 边界幽灵态：按完整全局网格构建（全局面编号），再把 group_code 重映射到本 rank 的压缩面编号——残差核
        # 以压缩面编号查询，两套编号不是同一索引空间的子区间（2-rank 合成网格上 8/12 个面会查到错误的组）
        from autoflowcfd.boundary.fr_ghost_state import restrict_ghost_provider
        from autoflowcfd.core.fr_solver.boundary import boundary_physics, build_boundary_ghost_provider

        self._setup_state_and_boundary(restrict_ghost_provider(
            build_boundary_ghost_provider(self.mesh, self.ops, boundary_physics(self), {}),
            self.partition.local_faces))
        self._finish_setup("traditional mode")
