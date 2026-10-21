"""AutoFlowCFD V2.0 - `MultiGPUDistributedSolver` 的装配阶段（mixin，只含方法）。

传统模式 `__init__`（每个 rank 持有完整全局网格、自己分区并构建面几何）与完全分布式
`from_fully_distributed_package`（root 预先切好紧凑包下发，`gpu_distributed_fully_distributed/build.py`）共用这里
与网格来源无关的阶段：身份与设备、压缩索引空间的设备几何、时间推进、湍流模型、状态与边界；两条路径只给出几何与
湍流几何量（壁距、DES 网格尺度）的来源。设置取自同一个 `DistributedRunSettings`（与 CPU 分布式同一份，
`core/mpi/distributed_solver/assembly.py`）。

2026-10-22（I-110）以前两条路径各写一遍，完全分布式路径缺产生项渐变计数器初始化（输运模型第一步
`advance_production_ramp` 即 AttributeError）、曾缺 low_mach_precond_enabled。

**顺序是契约**：面几何（含 halo 扩展）必须先于 halo 交换器与状态缓冲区构造（它们按构造时刻的收发表预分配）；
壁面模型必须先于边界幽灵态构造（`boundary_physics` 据此决定 WALL 组是否为滑移镜像）。
"""

from typing import Any

from loguru import logger

from autoflowcfd.core.gpu import get_cupy, gpu_available
from autoflowcfd.core.gpu.array_manager import GPUArrayManager
from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
from autoflowcfd.core.gpu.gpu_time_integration import GPUTimeIntegrator
from autoflowcfd.core.mpi import get_rank, is_root, mpi_available
from autoflowcfd.core.mpi.comm import barrier
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.distributed_solver.assembly import DistributedRunSettings
from autoflowcfd.core.mpi.partition import build_distributed_partition, partition_mesh
from autoflowcfd.core.time_integration.base import require_distributed_scheme
from autoflowcfd.core.turbulence.dual_time import reset_dual_time_history
from autoflowcfd.core.turbulence.wall_model.treatment import build_wall_model, validate_wall_treatment

from .compact_view import _CompactMeshDataView


class _MultiGPUSetupMixin:
    """`MultiGPUDistributedSolver` 两条构造路径按依赖顺序调用的装配阶段。"""

    # 宿主类与其它 mixin 提供 / 两条路径各自设置的属性（仅类型声明）
    _init_distributed_face_geometry: Any
    dist_flat_face: Any
    flat_face_gpu: Any

    def _setup_identity_and_device(self, rank, n_ranks: int, device_id, order: int,
                                   settings: DistributedRunSettings) -> None:
        """环境校验、rank 与阶数、来流与决定物理解的参数、壁面模型、GPU 设备与数组管理器；湍流模型对象与壁距等
        几何量先置空。"""
        if not gpu_available:
            raise RuntimeError("CuPy required for multi-GPU solver")
        if not mpi_available(require=True):
            raise RuntimeError("MPI required for distributed solver")
        cp = get_cupy()

        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        # `order` = 目标阶数，`current_order` = 当前实际所在阶数（与 CPU 分布式、单机同一约定）
        self.order = int(order)
        self.current_order = self.order
        self.order_continuation_enabled = True
        self.turb_model_name = settings.turb_model_name
        self.freestream = dict(settings.freestream)
        self.mu_molecular = settings.mu_molecular
        # 来流湍流参数：湍流模型构造（`_set_freestream_turbulence` / `create_sa_model`）与合成湍流入口要读
        self._turbulence_intensity = settings.turbulence_intensity
        self._viscosity_ratio = settings.viscosity_ratio
        self._sem_num_eddies = None if settings.sem_num_eddies is None else int(settings.sem_num_eddies)
        self.artificial_viscosity_enabled = bool(settings.artificial_viscosity_enabled)
        self.artificial_viscosity_alpha = float(settings.artificial_viscosity_alpha)
        self.wall_treatment = validate_wall_treatment(settings.wall_treatment)
        # 壁面模型：WMLES 或 RANS 自动壁面处理（与单机同一个构造函数）
        self.wall_model = build_wall_model(self.turb_model_name, self.wall_treatment)
        self.turb_model_gpu = None
        self.ddes_model_gpu = None
        self.sgs_model_gpu = None
        self.iddes_h_max_compact = None
        self.iddes_h_wn_compact = None
        self.wall_distance_gpu = None

        # GPU 设备选择：默认 round-robin 分配
        if device_id is None:
            device_id = self.rank % cp.cuda.runtime.getDeviceCount()
        self.device_id = device_id
        with cp.cuda.Device(device_id):
            logger.info(f"Rank {self.rank} using GPU device {device_id}")
        self.array_mgr = GPUArrayManager(device_id=device_id)

    def _setup_partition_and_geometry(self, mesh, ops, n_ranks: int) -> None:
        """传统模式：root 用 METIS 分区后广播，构建本 rank 的面几何（含 halo 扩展），再把全局网格的逐单元几何
        限制并重映射到 local+halo 压缩索引空间（"棱柱在前"排列，`dist_flat_face.compact_global_ids`）。"""
        fc = mesh.face_connectivity
        cell_partition = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells) if self.rank == 0 else None
        from autoflowcfd.core.mpi.comm import bcast_from_root
        if n_ranks > 1:
            cell_partition = bcast_from_root(cell_partition)
        if cell_partition is None:
            raise RuntimeError("root 没有广播单元分区（partition_mesh 的结果）")
        self.partition = build_distributed_partition(fc, cell_partition, self.rank, n_ranks)
        # 供 _init_distributed_face_geometry 扩展 halo 层覆盖 FR 通量点多源交叉插值依赖用、换阶时重建分区用
        self.cell_partition = cell_partition

        # 面几何必须排在 halo 交换器 / 状态 / 设备几何上传之前：它会原地扩展 partition 的 halo 收发表（真实
        # cube_demo 网格上 16%~19% 的单元需要），设备几何按它给出的压缩索引空间构造
        self.flat_face_gpu = None
        self.dist_flat_face = None
        self._init_distributed_face_geometry()
        self._setup_compact_geometry(
            _CompactMeshDataView(mesh, self.dist_flat_face.compact_global_ids, self.dist_flat_face.base_flat.n_prism),
            ops, mesh.n_sps_per_cell)

    def _setup_compact_geometry(self, compact_mesh, ops, n_sps: int) -> None:
        """压缩索引空间的设备几何、halo 交换器与分布式状态（两条路径共用）。

        `compact_mesh`：local+halo 压缩索引空间的网格几何（传统模式为全局网格的 `_CompactMeshDataView`，完全分布式
        为 root 预先切好的 `PrecompactedMeshData`）。残差核读取的 Q/U 只有压缩索引空间大小，按全局单元编号的几何
        会读到不相关单元（2026-08-28 #1）。WMLES 的匹配点几何也读这个主机端视图。
        """
        cp = get_cupy()
        self._compact_mesh_view = compact_mesh
        self.mesh_data = self.array_mgr.upload_mesh_data(compact_mesh, ops)
        self.n_compact = compact_mesh.n_cells
        # perm/inv_perm 常驻 GPU，避免每步残差计算重新上传
        self._perm_gpu = cp.asarray(self.dist_flat_face.perm)
        self._inv_perm_gpu = cp.asarray(self.dist_flat_face.inv_perm)
        # 与单机 GPUFRSolver 同一约定：`upload_mesh_data` 把网格几何与 FR 算子打包进同一个字典
        self.ops_data = {k: v for k, v in self.mesh_data.items()}
        # GPU 直接 halo 交换（CUDA-aware MPI 零拷贝，或 staging buffer）
        self.gpu_halo = GPUHaloExchange(self.partition, n_sps=n_sps, n_vars=5, device_id=self.device_id)
        self.state = DistributedFRState(self.partition, n_sps)

    def _setup_time_integration(self, settings: DistributedRunSettings) -> None:
        """时间积分器、产生项渐变、上一时间层与 Newton 跨步状态、CFL 策略与低马赫预处理（各后端唯一来源）。
        CFL 控制器按**全局**残差范数更新，全部 rank 得到同一个 CFL 数。"""
        time_scheme = require_distributed_scheme(settings.time_scheme)
        self.time_integrator = GPUTimeIntegrator(scheme=time_scheme,
                                                 dual_time_steps=int(settings.dual_time_inner_iter))
        from autoflowcfd.core.fr_solver.turbulence.init import init_production_ramp
        init_production_ramp(self, time_scheme)
        from autoflowcfd.core.time_integration.time_accurate import configure_time_stepping

        configure_time_stepping(self, time_scheme, time_accurate=bool(settings.time_accurate),
                                adaptive_cfl=bool(settings.adaptive_cfl), cfl_start=settings.cfl_start,
                                cfl_max=settings.cfl_max, cfl_min=settings.cfl_min,
                                low_mach_precond=settings.low_mach_precond)
        # DUAL_TIME 的 BDF2 上一物理时间层（None：尚未跑过一个物理步，退化为 BDF1，与单机同一约定）
        reset_dual_time_history(self)
        from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import reset_newton_state

        reset_newton_state(self)

    def _setup_turbulence_models(self, n_sps: int):
        """湍流模型对象（输运模型 / DES 长度尺度 / WALE，与全部后端同一个构造处），只按 n_local 分配（local+halo 的
        湍流量经平均流同一个 halo 交换器实时获取）；返回 `TurbulenceModels`。壁距与 DES 网格尺度由调用方按几何
        来源给出。"""
        from autoflowcfd.core.fr_solver.turbulence.init import build_turbulence_models, describe_turbulence_models

        models = build_turbulence_models(self, self.turb_model_name, self.partition.n_local_cells, n_sps,
                                         cp=get_cupy(), device_id=self.device_id)
        self.turb_model_gpu, self.ddes_model_gpu, self.sgs_model_gpu = models
        logger.info(f"Rank {self.rank}: GPU {describe_turbulence_models(self.turb_model_name, models)}")
        return models

    def _setup_state_and_boundary(self, boundary_ghost_provider) -> None:
        """GPU 常驻求解状态（本 rank 拥有的 n_local 个单元，均匀来流初值、速度方向取自攻角 / 侧滑角）、边界幽灵态
        （已重映射到本 rank 的压缩面编号）与湍流输运的 WALL / 开放边界面掩码。"""
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.state.n_sps
        with cp.cuda.Device(self.device_id):
            from autoflowcfd.core.utils.flow_direction import freestream_conservative_state

            self.U_gpu = cp.empty((n_local, n_sps, 5), dtype=cp.float64)
            self.U_gpu[:] = cp.asarray(freestream_conservative_state(self.freestream, 5))
        self.boundary_ghost_provider = boundary_ghost_provider

        # 掩码只需要压缩空间的面数与已重映射的 group_code
        self._wall_mask_gpu = None
        self._open_mask_gpu = None
        if self.turb_model_gpu is not None:
            import types
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_turbulence_face_masks_gpu
            compact_mesh_stub = types.SimpleNamespace(
                face_connectivity=types.SimpleNamespace(n_faces=self.dist_flat_face.base_flat.n_faces))
            wall_mask_np, open_mask_np = compute_turbulence_face_masks_gpu(compact_mesh_stub,
                                                                           self.boundary_ghost_provider)
            with cp.cuda.Device(self.device_id):
                self._wall_mask_gpu = cp.asarray(wall_mask_np)
                self._open_mask_gpu = cp.asarray(open_mask_np)

    def _finish_setup(self, mode: str) -> None:
        """收敛历史、步数计数、同步与启动日志。"""
        from autoflowcfd.core.fr_solver.convergence_history import ConvergenceHistory

        self.convergence_history = ConvergenceHistory()
        self.iteration = 0
        barrier()
        if is_root():
            n_local, n_halo = self.partition.n_local_cells, self.partition.n_halo
            logger.info(f"MultiGPUDistributedSolver ({mode}) initialized: {self.n_ranks} ranks, "
                        f"{n_local} local cells/rank (+{n_halo} halo)")
            print(f"✅ MultiGPUDistributedSolver Ready ({mode}):")
            print(f"   Ranks: {self.n_ranks}, Cells/rank: {n_local} (+{n_halo} halo)")
            print(f"   GPU device: {self.device_id} per rank")
