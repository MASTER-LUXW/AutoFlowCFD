"""
AutoFlowCFD V2.0 - MultiGPUDistributedSolver "完全分布式加载"构造入口
（2026-09-02，问题清单 #1：多 GPU "完全分布式加载"/内存最优模式此前
从未实现，只支持"传统模式"——每个 rank 独立加载完整全局网格）。

镜像 CPU `DistributedFRSolver.from_fully_distributed_package`（见
`core/mpi/distributed_mesh_loader.py`/`core/mpi/distributed_solver.py`
模块文档）同一套设计：root rank 预先用完整全局网格算好每个 rank 的
紧凑（local+halo 压缩索引空间）数据包（`build_fully_distributed_rank_
package`/`distributed_mesh_load_v2`，两者本身与后端无关，CPU/GPU 共用
同一套 package 构造逻辑），本 rank 只接收这份紧凑包，从未持有、也不
需要持有完整全局网格——这是"完全分布式加载"名副其实的内存优化。

与 CPU 版本的关键区别只在于：紧凑几何/状态数据构造好之后需要一次性
上传到 GPU（`GPUArrayManager.upload_mesh_data`/`GPUFlatFaceGeometry`/
`cp.asarray`），且湍流模型用 GPU 版类（`GPUTurbulenceSST`/`GPUDDESModel`/
`GPUIDDESModel`/`GPUWALEModel`）而不是 CPU 版——除此之外，package 的
构造、字段含义、范围边界（支持 turbulence_model='none'/'sst'/'ddes'/
'iddes'/'wmles'/'les'，DUAL_TIME 已接入）与 CPU 版完全一致，直接复用
同一个 `build_fully_distributed_rank_package`/`distributed_mesh_load_v2`，
不重新实现一遍。

Order Continuation 支持：`redistribute_multi_gpu_fully_distributed_for_
new_order`（本模块）与 CPU 版 `redistribute_fully_distributed_for_new_
order`（`distributed_mesh_loader.py`）同一套协议——root 用持续持有的
`solver._root_context`（`distributed_mesh_load_v2` 返回的第二个值）
重新计算 + 重新分发新阶数的紧凑包，本 rank 用它替换全部 compact 相关
属性（不重新构造 solver 实例本身）。由
`gpu_distributed_order_continuation.py::gpu_interpolate_to_new_order`
按 `solver._is_fully_distributed` 分派到这里。
"""

import types
import pickle

import numpy as np
from loguru import logger

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.mpi import get_rank
from autoflowcfd.core.mpi.comm import barrier, get_comm
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange


def _upload_wall_geometry_compact(solver, package, cp, device_id):
    """从 package 里取出 root 已经按本 rank 的 compact 索引空间算好、
    切好的 wall_distance_compact/iddes_h_max_compact/iddes_h_wn_compact
    并上传 GPU——SST/DDES/IDDES/WMLES 共用同一套逻辑，两处调用点
    （构造 + Order Continuation 重建）复用，避免写两遍。
    """
    wall_distance_compact = package.get('wall_distance_compact')
    if wall_distance_compact is None:
        raise RuntimeError(
            f"Rank {solver.rank}: 完全分布式加载模式下 turbulence_model="
            f"'{solver.turb_model_name}' 需要 wall_distance_compact，但 "
            f"package 里没有——build_fully_distributed_rank_package 应该"
            f"已经算好并放进 package，说明 root 侧构造有缺陷。"
        )
    with cp.cuda.Device(device_id):
        solver.wall_distance_gpu = cp.asarray(wall_distance_compact)

    h_max_compact = package.get('iddes_h_max_compact')
    h_wn_compact = package.get('iddes_h_wn_compact')
    with cp.cuda.Device(device_id):
        solver.iddes_h_max_compact = (
            cp.asarray(h_max_compact) if h_max_compact is not None else None
        )
        solver.iddes_h_wn_compact = (
            cp.asarray(h_wn_compact) if h_wn_compact is not None else None
        )


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
    self.flux_type = package.get('flux_type', 'radau')

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


def redistribute_multi_gpu_fully_distributed_for_new_order(solver, target_p: int) -> None:
    """"完全分布式加载"模式的多 GPU 阶数切换（Order Continuation）。

    与 CPU 版 `distributed_mesh_loader.py::redistribute_fully_
    distributed_for_new_order` 同一套协议（该函数文档的"协议"一节
    逐字适用，这里只重复 GPU 特有的部分）：
    1. 每个 rank 独立在 CPU（numpy）上插值/重置自己的 local U + 湍流场
       （从 GPU 下载 → 插值 → 稍后重新上传，理由与 `gpu_distributed_
       order_continuation.py::gpu_interpolate_to_new_order`"传统模式"
       版本一致：CuPy 数组不支持 `np.einsum`）。
    2. Root：用 `solver._root_context` 持有的完整全局网格重新计算 +
       重新分发新阶数的紧凑包（`build_fully_distributed_rank_package`，
       与初次构造同一个函数，只是阶数/mesh 状态不同）。
    3. 每个 rank：接收新包，把上一步插值/重置好的 local U 写入新
       partition 布局的 state，替换全部 compact 相关 GPU 属性（mesh_data/
       flat_face_gpu/halo 交换器/wall_distance/boundary_ghost_provider
       等），不重新构造 solver 实例本身（保留 turb_model_gpu/
       sgs_model_gpu/wmles_model 对象引用，只替换它们的 k_field/
       omega_field/nu_t 数组——与"传统模式" `gpu_interpolate_to_new_
       order` 同一个设计）。

    只支持通过 `root_context` 非 None 构造的实例（与 CPU 版同一个
    fail-fast 护栏）。
    """
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.fr.quadrature_points import gauss_legendre
    from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence
    from autoflowcfd.core.utils.order_continuation import _build_linear_interp_matrix_3d

    cp = get_cupy()
    old_order = solver.current_order
    if old_order == target_p:
        return

    is_root_rank = get_rank() == 0
    root_context = getattr(solver, '_root_context', None) if is_root_rank else None
    if is_root_rank and root_context is None:
        raise NotImplementedError(
            "redistribute_multi_gpu_fully_distributed_for_new_order: root rank 没有 "
            "_root_context（本实例不是通过 distributed_mesh_load_v2 返回的 "
            "root_context 构造的）——Order Continuation 在'完全分布式加载' "
            "模式下要求 root 持续持有完整全局网格用于重新分发，不支持从 "
            "缺少这份上下文的实例继续爬坡（如实报告，不是假装能用）。"
        )

    n_local = solver.partition.n_local_cells

    # --- 1. 每个 rank 独立插值/重置自己的 local U + 湍流场（CPU 上做，
    # 理由见函数文档）---
    freestream = getattr(solver, '_package_freestream', None) or solver.freestream
    rho_inf = freestream.get('rho_inf', 1.225)
    vel_inf = freestream.get('vel_inf', 33.33)
    p_inf = freestream.get('p_inf', 101325.0)

    new_k_np = new_omega_np = new_nu_t_np = None
    if target_p > old_order:
        old_sps_1d, _ = gauss_legendre(old_order + 1)
        new_sps_1d, _ = gauss_legendre(target_p + 1)
        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)

        old_U_np = cp.asnumpy(solver.U_gpu)
        new_U_np = np.einsum('ab,cbv->cav', W, old_U_np)

        if solver.turb_model_gpu is not None:
            old_k_np = cp.asnumpy(solver.turb_model_gpu.k_field)
            old_omega_np = cp.asnumpy(solver.turb_model_gpu.omega_field)
            new_k_np = np.einsum('ab,cb->ca', W, old_k_np)
            new_omega_np = np.einsum('ab,cb->ca', W, old_omega_np)
            old_nu_t = getattr(solver.turb_model_gpu, 'nu_t', None)
            if old_nu_t is not None:
                old_nu_t_np = cp.asnumpy(old_nu_t)
                if old_nu_t_np.shape[1] == old_U_np.shape[1]:
                    new_nu_t_np = np.einsum('ab,cb->ca', W, old_nu_t_np)
    else:
        new_n_sps = (target_p + 1) ** 3
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2
        new_U_np = np.zeros((n_local, new_n_sps, 5))
        new_U_np[:, :, 0] = rho_inf
        new_U_np[:, :, 1] = rho_inf * vel_inf
        new_U_np[:, :, 4] = rho_inf * e

        if solver.turb_model_gpu is not None:
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            new_k_np = np.ones((n_local, new_n_sps)) * k_inf
            new_omega_np = np.ones((n_local, new_n_sps)) * omega_inf
            new_nu_t_np = np.zeros((n_local, new_n_sps))

    # des_length_scale：与"传统模式"同一处理，清空而不是插值（依赖 nu_t，
    # 要到本阶数第一次 compute_source 调用后才会被重新算出）。
    if solver.ddes_model_gpu is not None and hasattr(solver.turb_model_gpu, 'des_length_scale'):
        solver.turb_model_gpu.des_length_scale = None

    # --- 2. Root 重新计算 + 分发新紧凑包 ---
    comm = get_comm()
    n_ranks = solver.n_ranks

    if is_root_rank:
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package

        mesh = root_context['mesh']
        stale_orders = [o for o in list(mesh._order_geometry_cache) if o != target_p]
        for o in stale_orders:
            del mesh._order_geometry_cache[o]
        mesh.set_order(target_p)
        ops = generate_fr_operators(target_p, flux_point_type=getattr(solver, 'flux_type', 'radau'))
        root_context['ops'] = ops

        turb_model_name = root_context['turb_model_name']
        root_solver_stub = types.SimpleNamespace(
            mesh=mesh, freestream=root_context['freestream'], turb_model_name=turb_model_name,
            wmles_model=(object() if turb_model_name == "WMLES" else None),
        )
        boundary_ghost_provider_global = build_boundary_ghost_provider(
            root_solver_stub, bc_overrides=root_context.get('bc_overrides', {}),
        )
        root_context['boundary_ghost_provider_global'] = boundary_ghost_provider_global

        packages = [
            build_fully_distributed_rank_package(
                mesh, ops, root_context['fc'], root_context['cell_partition'], r, n_ranks,
                boundary_ghost_provider_global, root_context['freestream'],
                root_context['mu_molecular'], root_context['mach_ref'],
                target_p, root_context['enable_viscous'],
                turb_model_name=turb_model_name, wall_node_indices=root_context['wall_node_indices'],
                h_max_global=root_context['h_max_global'], h_wn_global=root_context['h_wn_global'],
                turbulence_intensity=root_context['turbulence_intensity'],
                viscosity_ratio=root_context['viscosity_ratio'],
                time_scheme=root_context.get('time_scheme'),
                dual_time_inner_iter=root_context.get('dual_time_inner_iter', 20),
            )
            for r in range(n_ranks)
        ]
        my_package = packages[0]
        if n_ranks > 1:
            for r in range(1, n_ranks):
                buf = pickle.dumps(packages[r])
                buf_size = np.array([len(buf)], dtype=np.int64)
                comm.Send(buf_size, dest=r, tag=340)
                comm.Send(buf, dest=r, tag=341)
    else:
        buf_size = np.empty(1, dtype=np.int64)
        comm.Recv(buf_size, source=0, tag=340)
        buf = np.empty(int(buf_size[0]), dtype=np.uint8)
        comm.Recv(buf, source=0, tag=341)
        my_package = pickle.loads(buf.tobytes())

    # --- 3. 应用新包：替换 compact 相关属性，保留 turb_model_gpu/
    # sgs_model_gpu/wmles_model 对象本身（只是上一步已经替换过它们的
    # 数组）---
    solver.ops = generate_fr_operators(target_p, flux_point_type=getattr(solver, 'flux_type', 'radau'))
    precompacted_mesh = my_package['precompacted_mesh']
    solver.mesh = precompacted_mesh
    solver.partition = my_package['partition']
    solver.dist_flat_face = my_package['dist_fc']

    new_n_sps = precompacted_mesh.n_sps_per_cell
    solver.flat_face_gpu = build_gpu_flat_face(solver.dist_flat_face.base_flat, solver.device_id)
    solver.mesh_data = solver.array_mgr.upload_mesh_data(precompacted_mesh, solver.ops)
    solver.n_compact = precompacted_mesh.n_cells
    solver._compact_mesh_view = precompacted_mesh
    solver.ops_data = {k: v for k, v in solver.mesh_data.items()}

    with cp.cuda.Device(solver.device_id):
        solver._perm_gpu = cp.asarray(solver.dist_flat_face.perm)
        solver._inv_perm_gpu = cp.asarray(solver.dist_flat_face.inv_perm)
        solver.U_gpu = cp.asarray(new_U_np)

    solver.gpu_halo = GPUHaloExchange(solver.partition, n_sps=new_n_sps, n_vars=5, device_id=solver.device_id)
    solver.state = DistributedFRState(solver.partition, new_n_sps, 5)

    if solver.turb_model_gpu is not None:
        with cp.cuda.Device(solver.device_id):
            solver.turb_model_gpu.k_field = cp.asarray(new_k_np)
            solver.turb_model_gpu.omega_field = cp.asarray(new_omega_np)
            if new_nu_t_np is not None:
                solver.turb_model_gpu.nu_t = cp.asarray(new_nu_t_np)
        solver.turb_halo_gpu = GPUHaloExchange(
            solver.partition, n_sps=new_n_sps, n_vars=2, device_id=solver.device_id
        )
        if solver.ddes_model_gpu is not None:
            solver.des_length_scale_halo_gpu = GPUHaloExchange(
                solver.partition, n_sps=new_n_sps, n_vars=1, device_id=solver.device_id
            )
        _upload_wall_geometry_compact(solver, my_package, cp, solver.device_id)

    if solver.sgs_model_gpu is not None:
        cell_volumes_compact = solver.mesh_data.get('cell_volumes')
        delta = cp.abs(cell_volumes_compact) ** (1.0 / 3.0)
        solver._grid_scale_compact = cp.tile(delta[:, None], (1, new_n_sps))

    if solver.wmles_model is not None:
        wall_distance_compact = my_package.get('wall_distance_compact')
        with cp.cuda.Device(solver.device_id):
            solver.wall_distance_gpu = (
                cp.asarray(wall_distance_compact) if wall_distance_compact is not None else None
            )

    solver.filter_func_gpu = None
    solver._init_modal_filter_distributed()

    solver.boundary_ghost_provider = my_package['boundary_ghost_provider']
    solver._wall_mask_k_gpu = None
    if solver.turb_model_gpu is not None:
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_wall_dirichlet_mask_gpu
        compact_mesh_stub = types.SimpleNamespace(
            face_connectivity=types.SimpleNamespace(n_faces=solver.dist_flat_face.base_flat.n_faces)
        )
        wall_mask_np = compute_wall_dirichlet_mask_gpu(compact_mesh_stub, solver.boundary_ghost_provider)
        with cp.cuda.Device(solver.device_id):
            solver._wall_mask_k_gpu = cp.asarray(wall_mask_np)

    solver.mu_molecular = my_package['mu_molecular']
    solver.freestream = {**my_package['freestream'], "mach_ref": my_package['mach_ref']}
    solver._package_freestream = my_package['freestream']

    if hasattr(solver, '_dual_time_U_prev'):
        solver._dual_time_U_prev = None

    solver.current_order = target_p
