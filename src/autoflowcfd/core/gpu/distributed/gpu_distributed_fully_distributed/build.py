"""AutoFlowCFD V2.0 - 从完全分布式包构造多 GPU solver

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_fully_distributed.py`(原 557 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import types



from loguru import logger

from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.mpi import get_rank

from autoflowcfd.core.mpi.comm import barrier

from autoflowcfd.core.mpi.distributed_state import DistributedFRState

from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
from .upload import _upload_wall_geometry_compact


def build_multi_gpu_solver_from_fully_distributed_package(
    cls, package: dict, n_ranks: int,
    device_id=None, rank=None, root_context=None,
):
    """真正的"完全分布式加载"构造入口（见模块文档）。

    Args:
        cls: `MultiGPUDistributedSolver`（由该类的 `from_fully_
            distributed_package` classmethod 传入 `cls`，本函数只是把
            实现拆到独立模块控制单文件行数，与 CPU
            `DistributedFRSolver.from_fully_distributed_package` 同一个
            拆分动机）
        package: `build_fully_distributed_rank_package`（或经
            `distributed_mesh_load_v2` MPI 收发后本 rank 收到的那一份）
            的返回值——与 CPU 路径共用同一份数据结构。
        n_ranks: MPI rank 总数
        device_id: GPU 设备号（None 时按 rank 轮询分配，与主 `__init__`
            同一个默认策略）
        rank: 当前 rank（None 时从 MPI 获取）
        root_context: 仅 root rank 需要非 None——`distributed_mesh_
            load_v2` 返回的第二个值，供 Order Continuation 重新分发用，
            见模块文档。

    Returns:
        MultiGPUDistributedSolver 实例
    """
    from autoflowcfd.core.gpu.array_manager import GPUArrayManager
    from autoflowcfd.core.gpu.gpu_time_integration import GPUTimeIntegrator
    from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    cp = get_cupy()

    turb_model_name = package.get('turb_model_name', 'NONE')
    if turb_model_name not in ('NONE', 'SST', 'DDES', 'IDDES', 'WMLES', 'LES'):
        raise NotImplementedError(
            f"MultiGPUDistributedSolver.from_fully_distributed_package: "
            f"完全分布式加载模式目前只支持 turbulence_model="
            f"'none'/'sst'/'ddes'/'iddes'/'wmles'/'les'，收到的是 "
            f"'{turb_model_name}'。"
        )

    self = cls.__new__(cls)
    self.rank = rank if rank is not None else get_rank()
    self.n_ranks = n_ranks
    self._is_fully_distributed = True
    self._using_distributed_mesh = True
    self.cell_partition = None  # 本 rank 从未持有完整全局网格，见类文档
    self._root_context = root_context

    self.order = int(package['order'])
    self.current_order = self.order
    self.order_continuation_enabled = package.get('order_continuation_enabled', True)

    precompacted_mesh = package['precompacted_mesh']
    self.mesh = precompacted_mesh
    # 每个 rank 本地重新生成算子（纯函数，只依赖 order，见 CPU 版
    # `from_fully_distributed_package` 同一处注释——避免把 FROperators
    # 整个 pickle 发过来的不必要开销）。
    self.ops = generate_fr_operators(package['order'])

    self.partition = package['partition']
    self.dist_flat_face = package['dist_fc']

    # GPU 设备选择：与主 __init__ 同一个默认策略（rank % n_gpus）。
    if device_id is None:
        n_gpus = cp.cuda.runtime.getDeviceCount()
        device_id = self.rank % n_gpus
    self.device_id = device_id
    with cp.cuda.Device(device_id):
        logger.info(f"Rank {self.rank} using GPU device {device_id} (fully-distributed mode)")
    self.array_mgr = GPUArrayManager(device_id=device_id)

    n_sps = precompacted_mesh.n_sps_per_cell
    n_local = self.partition.n_local_cells

    # 分布式面几何（GPU）：`dist_fc`/`precompacted_mesh` 已经是 root 按
    # 本 rank 的 compact 索引空间预先切好的数据，不需要像"传统模式"
    # `_init_distributed_face_geometry` 那样调用 `build_distributed_
    # flat_face`（那需要完整全局网格，本 rank 没有）——只需要把已经算好
    # 的 `dist_flat_face.base_flat` 上传 GPU。
    self.flat_face_gpu = build_gpu_flat_face(self.dist_flat_face.base_flat, device_id)

    # mesh_data：`precompacted_mesh`（`PrecompactedMeshData`）本身**已经**
    # 是 compact 索引空间大小的几何数据（root 预先切好），不像"传统模式"
    # 需要 `_CompactMeshDataView` 再从全局网格里切一次——直接传给
    # `upload_mesh_data` 即可（鸭子类型接口完全兼容，见该类文档）。
    self.mesh_data = self.array_mgr.upload_mesh_data(precompacted_mesh, self.ops)
    self.n_compact = precompacted_mesh.n_cells
    # WMLES 壁面剪应力修正需要的 numpy compact 视图——`precompacted_mesh`
    # 本身就是这个视图（numpy，compact 索引空间），不需要像"传统模式"
    # 那样另外构造 `_CompactMeshDataView`。
    self._compact_mesh_view = precompacted_mesh
    self._perm_gpu = cp.asarray(self.dist_flat_face.perm)
    self._inv_perm_gpu = cp.asarray(self.dist_flat_face.inv_perm)
    self.ops_data = {k: v for k, v in self.mesh_data.items()}

    self.gpu_halo = GPUHaloExchange(self.partition, n_sps=n_sps, n_vars=5, device_id=device_id)
    self.state = DistributedFRState(self.partition, n_sps, 5)

    freestream_pkg = package['freestream']
    rho_inf = freestream_pkg.get('rho_inf', 1.225)
    vel_inf = freestream_pkg.get('vel_inf', 33.33)
    p_inf = freestream_pkg.get('p_inf', 101325.0)
    # `self.freestream`（含 mach_ref）在"传统模式" `__init__` 里是无条件
    # 设置的（AUSM+up Weiss-Smith 预处理在全部 turb_model_name 下都要
    # 读 `self.freestream["mach_ref"]`，不是只有 SST/DDES/IDDES 才需要
    # ——这一点与 CPU `DistributedFRSolver.from_fully_distributed_
    # package` 不同，那边的 `self.freestream` 只在湍流分支里设置，因为
    # CPU 版 mach_ref 是从 `self._local_solver.freestream` 取的，本类
    # 没有等价的 `_local_solver`），这里同样无条件设置，不放在湍流分支
    # 里，否则 NONE/LES/WMLES 会在 compute_inviscid_residual_gpu 里
    # AttributeError。
    self.freestream = {**freestream_pkg, "mach_ref": package['mach_ref']}
    self.mu_molecular = package['mu_molecular']
    self._package_freestream = freestream_pkg
    self._turbulence_intensity = package.get('turbulence_intensity', 0.01)
    self._viscosity_ratio = package.get('viscosity_ratio', 5.0)

    time_scheme_enum = package.get('time_scheme', TimeIntegrationScheme.SSP_RK3)
    time_scheme_str = getattr(time_scheme_enum, 'value', time_scheme_enum)
    self.time_integrator = GPUTimeIntegrator(scheme=time_scheme_str, cfl=1.0)

    # 自适应 CFL 控制器（2026-09-15 补齐真实缺口）：这条路径此前**完全
    # 没有**控制器，而 `_current_cfl()` 的回退是 `self.time_integrator.cfl`
    # ——上面那行给的是 1.0，远超 P>=1 的 SSP-RK3 稳定极限（对流项
    # ~1/(2p+1)，P1 上可用 CFL 实测在 0.03 量级）。也就是说
    # `--multi-gpu --fully-distributed` 会用一个比可用值大一到两个数量级
    # 的 CFL 跑，而同一个类的"传统模式" `__init__` 里控制器一直是有的：
    # 同一个后端的两种模式在最基本的时间步长策略上不一致。
    # `cfl=1.0` 这个占位值本身保留不动——它只是 GPUTimeIntegrator 的构造
    # 参数，真正生效的是控制器给出的 CFL（见 `_current_cfl()`）。
    self._cfl_controller = None
    if str(time_scheme_str) in ("ssp_rk2", "ssp_rk3"):
        from autoflowcfd.core.time_integration.adaptive_cfl import (
            AdaptiveCFLController,
        )
        # None 感知：package 现在总是带这三个键（CLI 未指定时为 None），
        # 不能用 `.get(k, default)`——那会拿到显式的 None。与 CPU 侧
        # `DistributedFRSolver.from_fully_distributed_package` 同一处理。
        # **不再硬编码兜底默认值**（2026-09-17，与 CPU 分布式路径同一次
        # 改动）：此前写死 `cfl_start=0.1, cfl_max=0.5`，控制器默认值一改
        # （同日按直接谱测量与真实网格失效点重定为 0.03/0.06）这条路径
        # 就与单机脱节。只传**非 None** 的键，默认值的单一事实来源是
        # `AdaptiveCFLController.__init__` 的签名。
        _cfl_kw = {k: package[k]
                   for k in ('cfl_start', 'cfl_max', 'cfl_min')
                   if package.get(k) is not None}
        self._cfl_controller = AdaptiveCFLController(**_cfl_kw)
    # `dual_time_steps`：GPUTimeIntegrator 构造函数本身不接受这个参数
    # （与 gpu_distributed.py::step() 的 `getattr(...,'dual_time_steps',5)`
    # 回退设计一致，见该方法调用点），这里显式设置成 package 携带的值。
    self.time_integrator.dual_time_steps = package.get('dual_time_inner_iter', 20)
    self._dual_time_U_prev = None

    self.turb_model_name = turb_model_name
    self.turb_model_gpu = None
    self.turb_halo_gpu = None
    self.ddes_model_gpu = None
    self.iddes_h_max_compact = None
    self.iddes_h_wn_compact = None
    self.des_length_scale_halo_gpu = None
    self.sgs_model_gpu = None
    self._grid_scale_compact = None
    self.wmles_model = None
    self.wall_distance_gpu = None

    if turb_model_name in ('SST', 'DDES', 'IDDES'):
        from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
        nu = self.mu_molecular / max(rho_inf, 1e-10)
        k_inf = 1.5 * (vel_inf * self._turbulence_intensity) ** 2
        nu_t_inf = self._viscosity_ratio * nu
        omega_inf = k_inf / max(nu_t_inf, 1e-30)
        self.turb_model_gpu = GPUTurbulenceSST(
            n_local, n_sps, device_id, k_inf=k_inf, omega_inf=omega_inf
        )
        self.turb_model_gpu.k_max = 0.5 * vel_inf ** 2
        self.turb_model_gpu.omega_max = 1e6
        self.turb_halo_gpu = GPUHaloExchange(self.partition, n_sps=n_sps, n_vars=2, device_id=device_id)
        logger.info(f"Rank {self.rank}: GPU SST model initialized (fully-distributed mode, "
                    f"k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e})")

        _upload_wall_geometry_compact(self, package, cp, device_id)

        if turb_model_name == 'DDES':
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUDDESModel
            self.ddes_model_gpu = GPUDDESModel()
        elif turb_model_name == 'IDDES':
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUIDDESModel
            self.ddes_model_gpu = GPUIDDESModel()

        if self.ddes_model_gpu is not None:
            self.des_length_scale_halo_gpu = GPUHaloExchange(
                self.partition, n_sps=n_sps, n_vars=1, device_id=device_id
            )

    elif turb_model_name == 'LES':
        from autoflowcfd.core.gpu.turbulence.gpu_sgs import GPUWALEModel
        self.sgs_model_gpu = GPUWALEModel()
        cell_volumes_compact = self.mesh_data.get('cell_volumes')
        delta = cp.abs(cell_volumes_compact) ** (1.0 / 3.0)
        self._grid_scale_compact = cp.tile(delta[:, None], (1, n_sps))
        logger.info(f"Rank {self.rank}: GPU LES with WALE SGS model initialized (fully-distributed mode)")

    elif turb_model_name == 'WMLES':
        from autoflowcfd.core.turbulence.wmles import WMLESModel
        self.wmles_model = WMLESModel(nu=self.mu_molecular / max(rho_inf, 1e-10))
        _upload_wall_geometry_compact(self, package, cp, device_id)
        logger.info(f"Rank {self.rank}: GPU-distributed WMLES model initialized (fully-distributed mode)")

    self.filter_func_gpu = None
    self._init_modal_filter_distributed()

    with cp.cuda.Device(device_id):
        self.U_gpu = cp.zeros((n_local, n_sps, 5), dtype=cp.float64)
        self.U_gpu[:, :, 0] = rho_inf
        self.U_gpu[:, :, 1] = rho_inf * vel_inf
        self.U_gpu[:, :, 4] = p_inf / (1.4 - 1.0) + 0.5 * rho_inf * vel_inf ** 2

    # boundary_ghost_provider：root 已经用完整全局网格构造好并把
    # group_code 重映射到本 rank 的 compact 索引空间（见
    # `build_fully_distributed_rank_package` 文档"边界条件"一节），
    # 不需要像"传统模式" `__init__` 那样再调用 `build_boundary_ghost_
    # provider`/重映射一次。
    self.boundary_ghost_provider = package['boundary_ghost_provider']

    self._wall_mask_k_gpu = None
    if self.turb_model_gpu is not None:
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_wall_dirichlet_mask_gpu
        compact_mesh_stub = types.SimpleNamespace(
            face_connectivity=types.SimpleNamespace(n_faces=self.dist_flat_face.base_flat.n_faces)
        )
        wall_mask_np = compute_wall_dirichlet_mask_gpu(compact_mesh_stub, self.boundary_ghost_provider)
        with cp.cuda.Device(device_id):
            self._wall_mask_k_gpu = cp.asarray(wall_mask_np)

    self.residual_history = []
    self.iteration = 0

    barrier()
    if self.rank == 0:
        logger.info(
            f"MultiGPUDistributedSolver (fully-distributed) initialized: {n_ranks} ranks, "
            f"{n_local} local cells/rank (+{self.partition.n_halo} halo)"
        )
        print(f"✅ MultiGPUDistributedSolver Ready (fully-distributed mode: "
              f"only root rank loaded the full mesh):")
        print(f"   Ranks: {n_ranks}, Cells/rank: {n_local} (+{self.partition.n_halo} halo)")
        print(f"   GPU device: {device_id} per rank")

    return self
