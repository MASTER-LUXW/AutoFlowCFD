"""AutoFlowCFD V2.0 - 从完全分布式 package 构造，以及本 rank 的局部求解器视图

`DistributedFRSolver` 的一组方法（mixin）。

**只含方法，没有状态**：全部属性由 `DistributedFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from typing import Any, Optional
from autoflowcfd.core.mpi.comm import barrier

from .assembly import DistributedRunSettings, canonical_turbulence_model


def _settings_from_package(package: dict) -> DistributedRunSettings:
    """完全分布式加载的运行设置：取自 root 打包的紧凑包（`distributed_mesh_loader/package.py`，键恒有）。"""
    names = ('mu_molecular', 'turbulence_intensity', 'viscosity_ratio', 'sem_num_eddies',
             'artificial_viscosity_enabled', 'artificial_viscosity_alpha', 'wall_treatment', 'time_scheme',
             'dual_time_inner_iter', 'time_accurate', 'adaptive_cfl', 'cfl_start', 'cfl_max', 'cfl_min',
             'low_mach_precond')
    return DistributedRunSettings(
        turb_model_name=canonical_turbulence_model(package['turb_model_name']),
        freestream={**package['freestream'], "mach_ref": package['mach_ref']},
        **{name: package[name] for name in names})


class _DistributedFromPackageMixin:
    """从完全分布式 package 构造，以及本 rank 的局部求解器视图"""

    # 宿主类在 __init__ / 其它 mixin 中提供的属性（仅类型声明）
    _assemble_settings: Any
    _assemble_state: Any
    _assemble_time_stepping: Any
    _block_jacobi_colors_local: Any
    _coupling_colors_local: Any
    _is_fully_distributed: Any
    _root_context: Any
    ddes_model: Any
    dist_flat_face: Any
    iddes_h_max_compact: Any
    iddes_h_wn_compact: Any
    mesh: Any
    ops: Any
    partition: Any
    sgs_model: Any
    solver_kwargs: Any
    turb_model: Any
    turb_model_name: Any
    wall_distance_compact: Any

    @classmethod
    def from_fully_distributed_package(cls, package: dict, n_ranks: int, rank: Optional[int] = None,
                                       root_context: Optional[dict] = None, n_threads: int = -1):
        """真正的"完全分布式加载"构造入口（2026-09-02，见 core/mpi/
        distributed_mesh_loader/fully_distributed.py::distributed_mesh_load_v2/
        build_fully_distributed_rank_package 模块文档）。

        与主构造函数（`__init__`，"传统模式"：每个 rank 独立加载完整
        全局网格）的关键区别：这里的 `package` 是 root rank 预先算好、
        已经按本 rank 的 compact 索引空间切好的紧凑数据（`partition`/
        `dist_fc`/`PrecompactedMeshData`/切好 `group_code` 的边界幽灵态
        提供者），本 rank 从未持有、也不需要持有完整全局网格——这是
        "完全分布式加载"名副其实的内存优化。

        用 `cls.__new__(cls)` 绕开主 `__init__`（分区与面几何构建要求完整全局网格，这条路径下由 root 预先算好随包
        下发）；与网格来源无关的装配（设置、状态、时间推进）与主 `__init__` 同一组方法（`assembly.py`），这里只给出
        几何与湍流几何量的来源。

        全部湍流模型（`core/turbulence/registry.py::SUPPORTED_MODELS`，与单机相同）、全部时间格式、checkpoint 续算与
        Order Continuation 均已接入。换阶由 `DistributedFRSolver._interpolate_to_new_order` 调用
        `redistribute_fully_distributed_for_new_order`：本 rank 只持有紧凑数据，root 用持有的 `_root_context`
        （完整全局网格）重新计算并分发新阶数的紧凑包。

        Args:
            package: `build_fully_distributed_rank_package` 的返回值
                （或 `distributed_mesh_load_v2` 经 MPI 收发后本 rank
                收到的那一份）
            n_ranks: MPI rank 总数
            rank: 当前 rank（默认从 MPI 获取）
            root_context: 仅 root rank（rank 0）需要非 None——
                `distributed_mesh_load_v2` 返回的第二个值（见该函数
                文档"Order Continuation 支持"一节），持有完整全局
                `mesh`/`ops`/`cell_partition`/`face_connectivity`/
                `boundary_ghost_provider_global`/冻结的 root 端配置，
                供后续阶数切换时重新计算+重新分发紧凑包。非 root rank
                永远传 None（它们从未持有、也不需要这份数据）。不提供
                时（None）意味着这个 solver 实例无法执行 Order
                Continuation（目标阶数 >= 1 时 `solve()` 会 fail-fast
                拒绝而不是静默跳过升阶爬坡）。
            n_threads: 本 rank 的 numba 线程数（与单机 FRSolver 同名参数、同一取值规则）。2026-10-20 以前完全分布式
                加载从不设置，numba 按全部核数开线程（每节点多个 rank 时超订），`--threads` 被忽略。

        Returns:
            DistributedFRSolver 实例
        """
        import types
        from autoflowcfd.fr.operators import generate_fr_operators

        from autoflowcfd.core.fr_solver.solver.threads import configure_numba_threads

        configure_numba_threads(n_threads)
        self = cls.__new__(cls)
        settings = _settings_from_package(package)
        self._assemble_settings(rank, n_ranks, package['order'], settings)
        self._is_fully_distributed = True
        self._root_context = root_context

        self.mesh = package['precompacted_mesh']
        # 每个 rank 本地重新生成算子（纯函数，只依赖阶数），不随包 pickle 传输：微分算子与单元数无关，本地重算比
        # 跨进程传输便宜
        self.ops = generate_fr_operators(package['order'])
        self.partition = package['partition']
        # 隐式步的块 Jacobi 着色（全局一致，root 随包下发；非隐式时为 None）
        self._block_jacobi_colors_local = package['cell_colors']
        self._coupling_colors_local = package['cell_colors_d2']
        self.dist_flat_face = package['dist_fc']
        n_sps = self.mesh.n_sps_per_cell
        self._assemble_state(n_sps)
        self.solver_kwargs = {}

        # 湍流几何量（壁距、DES 网格尺度）由 root 在完整网格上算好、切成本 rank 的紧凑索引空间随包下发；湍流模型
        # 对象与全部后端同一个构造处（`build_turbulence_models`）——这里的 `self.mesh` 是 `PrecompactedMeshData`，
        # 没有完整节点坐标，不能像 `init_turbulence_models` 那样现算 DES 网格尺度
        from autoflowcfd.core.fr_solver.turbulence.init import build_turbulence_models
        from autoflowcfd.core.turbulence.registry import has_transport_equations

        self.wall_distance_compact = package['wall_distance_compact']
        self.iddes_h_max_compact = package['iddes_h_max_compact']
        self.iddes_h_wn_compact = package['iddes_h_wn_compact']
        models = build_turbulence_models(self, self.turb_model_name, self.partition.n_local_cells, n_sps)
        self.turb_model, self.ddes_model, self.sgs_model = models
        if has_transport_equations(self.turb_model_name):
            # 壁距交给模型（SST 的 omega 上界按 root 算好的壁距取全局最小、SA 的壁面解点），与传统模式同一函数
            from autoflowcfd.core.mpi.distributed_turbulence import apply_distributed_wall_distance
            apply_distributed_wall_distance(self.turb_model, self.wall_distance_compact, self)

        # 轻量级鸭子类型"local_solver"替身（不构造真正的 FRSolver——那
        # 需要完整全局网格重新生成 sps 几何/差分算子，在这条路径下既
        # 没有数据也没有必要）：`step()` 只读它的 `.config.physics.
        # enable_viscous`/`.mu_molecular`/`.boundary_ghost_provider`/
        # `.freestream["mach_ref"]` 这 4 个属性（已逐行核实
        # distributed_solver.py 全文只有这 4 处 `self.local_solver.`
        # 访问），不需要真正的 FRSolver 实例。
        self._local_solver = types.SimpleNamespace(
            config=types.SimpleNamespace(
                physics=types.SimpleNamespace(enable_viscous=package['enable_viscous'])
            ),
            mu_molecular=package['mu_molecular'],
            boundary_ghost_provider=package['boundary_ghost_provider'],
            freestream={**package['freestream'], "mach_ref": package['mach_ref']},
        )
        self._assemble_time_stepping(settings)

        barrier()
        return self

    @property
    def local_solver(self):
        """延迟初始化本地求解器（避免循环依赖）。

        FRSolver 按 `order` 自己构建算子（不接受外部预构建的 FROperators）。`order` 缺省取 `mesh.order`（构建
        local_mesh 时用的同一个阶数）；`solver_kwargs` 里显式给出的 order（生产路径如此，与 mesh.order 一致）
        优先，避免与显式关键字参数重复传参。
        """
        if self._local_solver is None:
            from autoflowcfd.core.fr_solver.solver import FRSolver
            kwargs = dict(self.solver_kwargs)
            kwargs.setdefault('order', self.mesh.order)
            self._local_solver = FRSolver(self.mesh, **kwargs)

            # 真实 bug 修复（2026-09-02，实现分布式湍流模型时排查发现，
            # 与湍流本身无关——任何使用真实 WALL/INLET/OUTLET/FARFIELD/
            # SYMMETRY 区分的分布式算例都会中招，与 gpu_distributed.py
            # 同一处修复同一个根因）：`FRSolver(self.mesh, ...)` 用
            # `self.mesh`（完整全局网格，"传统模式"下如此）构造的
            # `boundary_ghost_provider.group_code` 是**全局**面编号索引，
            # 但 `distributed_compute_inviscid_residual`/`_viscous_residual`
            # 最终调用 `ghost_provider(f, ...)` 时 `f` 是本 rank 的
            # local+halo**压缩索引空间**面编号——两套编号不是同一个索引
            # 空间的子区间（见 `distributed_flat_face.py::partition.
            # local_faces` 文档）。真实合成网格验证：2-rank 分区，8/12
            # （67%）压缩面会被分配到错误的边界组编码。修复：把
            # `group_code` 重映射到压缩索引空间。
            provider = getattr(self._local_solver, 'boundary_ghost_provider', None)
            if provider is not None:
                from autoflowcfd.boundary.fr_ghost_state import restrict_ghost_provider

                self._local_solver.boundary_ghost_provider = restrict_ghost_provider(
                    provider, self.partition.local_faces)
        return self._local_solver
