"""AutoFlowCFD V2.0 - DistributedFRSolver 主类：传统模式构造

从 `src/autoflowcfd/core/mpi/distributed_solver.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

from typing import Any, Optional
from loguru import logger
from autoflowcfd.core.mpi import is_root
from autoflowcfd.core.mpi.partition import (
    partition_mesh,
    build_distributed_partition,
)
from autoflowcfd.core.mpi.distributed_flat_face import (
    build_distributed_flat_face,
)
from autoflowcfd.core.mpi.comm import barrier
from .assembly import DistributedRunSettings, _DistributedAssemblyMixin, canonical_turbulence_model
from .from_package import _DistributedFromPackageMixin
from autoflowcfd.core.fr_solver.solver.solve_loop import DistributedSolveLoopMixin
from .step import _DistributedStepMixin
from .support import _DistributedSupportMixin


#: 必需的构造参数（物理输入，与单机 FRSolver 的必需关键字参数一致）
_REQUIRED_KWARGS = ('order', 'turb_model_name', 'time_scheme', 'rho_inf', 'vel_inf', 'p_inf', 'mu_molecular')


def _frsolver_parameters():
    """单机 FRSolver 构造的关键字参数（名字 -> inspect.Parameter）：`solver_kwargs` 的合法键与缺省值的唯一来源
    （`solver_kwargs` 原样用于构造本地 FRSolver，见 `local_solver`）。"""
    import inspect

    from autoflowcfd.core.fr_solver.solver import FRSolver

    params = dict(inspect.signature(FRSolver.__init__).parameters)
    for name in ("self", "mesh", "initial_state"):
        params.pop(name)
    return params


def _check_solver_kwargs(solver_kwargs: dict) -> None:
    """`solver_kwargs` 必须是单机 FRSolver 的构造参数：未知键（拼错的名字）报错，必需参数缺失报错。

    2026-10-19（I-109）以前以 `.get(键, 缺省)` 读取：键名拼错（如 'aoa' 而不是 'aoa_deg'）不报错，静默换成缺省值。
    """
    params = _frsolver_parameters()
    unknown = sorted(set(solver_kwargs) - set(params))
    if unknown:
        raise TypeError(f"DistributedFRSolver 收到未知参数 {unknown}（solver_kwargs 是单机 FRSolver 的构造参数）")
    for name in _REQUIRED_KWARGS:
        if name not in solver_kwargs:
            raise TypeError(f"DistributedFRSolver 缺少必需参数 {name!r}（物理输入必须显式给出，见 FRSolver 模块文档）")


def _kwarg(solver_kwargs: dict, name: str):
    """`solver_kwargs[name]`，缺省时取单机 FRSolver 签名里的缺省值（两条后端同一份缺省值）。"""
    import inspect

    if name in solver_kwargs:
        return solver_kwargs[name]
    default = _frsolver_parameters()[name].default
    if default is inspect.Parameter.empty:
        raise TypeError(f"DistributedFRSolver 缺少必需参数 {name!r}")
    return default


def _settings_from_solver_kwargs(solver_kwargs: dict) -> DistributedRunSettings:
    """传统模式的运行设置：取自单机 FRSolver 同名构造参数（缺省值取 FRSolver 签名，`_kwarg`）。"""
    from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref

    rho, vel, p = solver_kwargs['rho_inf'], solver_kwargs['vel_inf'], solver_kwargs['p_inf']
    freestream = {"rho_inf": rho, "vel_inf": vel, "p_inf": p, "mach_ref": resolve_mach_ref(rho, vel, p),
                  "aoa_deg": float(_kwarg(solver_kwargs, "aoa_deg")),
                  "aos_deg": float(_kwarg(solver_kwargs, "aos_deg"))}
    names = ('turbulence_intensity', 'viscosity_ratio', 'sem_num_eddies', 'artificial_viscosity_enabled',
             'artificial_viscosity_alpha', 'wall_treatment', 'dual_time_inner_iter', 'time_accurate', 'adaptive_cfl',
             'cfl_start', 'cfl_max', 'cfl_min', 'low_mach_precond')
    return DistributedRunSettings(
        turb_model_name=canonical_turbulence_model(solver_kwargs['turb_model_name']), freestream=freestream,
        mu_molecular=solver_kwargs['mu_molecular'], time_scheme=solver_kwargs['time_scheme'],
        **{name: _kwarg(solver_kwargs, name) for name in names})


class DistributedFRSolver(_DistributedAssemblyMixin, _DistributedFromPackageMixin, _DistributedStepMixin,
                          DistributedSolveLoopMixin, _DistributedSupportMixin):
    """MPI 域分解分布式 FR 求解器。

    组合 FRSolver（单机残差计算）+ MPI 基础设施（分区、halo 交换、
    全局归约）。每个 rank 独立持有一个 FRSolver 实例（只处理 local cells），
    通过 halo 交换获取邻居信息。

    Attributes:
        partition: 本 rank 的分区信息
        halo_exchange: halo 交换管理器
        state: 分布式状态
        rank: 当前 rank
        n_ranks: 总 rank 数
    """

    # DDES/IDDES 的逐单元网格尺度（全局单元序），`fr_solver/turbulence/init.py::init_turbulence_models` 设置
    _iddes_h_max: Any
    _iddes_h_wn: Any

    def __init__(
        self,
        mesh,
        ops,
        n_ranks: int,
        face_connectivity,
        rank: Optional[int] = None,
        wall_distance_source=None,
        **solver_kwargs,
    ):
        """初始化分布式求解器（"传统模式"：每个 rank 持有完整全局网格，root 用 METIS 分区后广播）。
        只有 root 持有完整网格的"完全分布式加载"走 `from_fully_distributed_package`。

        Args:
            mesh: 完整全局 HighOrderMesh
            ops: FROperators
            n_ranks: MPI rank 总数
            face_connectivity: 全局 FRFaceConnectivity（分区与 halo 探测都在全局面连接关系上做）
            rank: 当前 rank 编号（默认从 MPI 获取）
            wall_distance_source: 壁面距离来源（湍流模型需要时）
            **solver_kwargs: 与单机 FRSolver 同名的构造参数

        Raises:
            NotImplementedError: 湍流模型名无法识别（支持的模型见 `core/turbulence/registry.py`，
                与单机相同）
        """
        _check_solver_kwargs(solver_kwargs)
        # numba 线程数在任何核调用之前设置（与单机 FRSolver 同一规则）；此前要等本地求解器在第一步惰性构造时才设置，
        # 构造期的几何核按全部核数开线程
        from autoflowcfd.core.fr_solver.solver.threads import configure_numba_threads

        configure_numba_threads(_kwarg(solver_kwargs, 'n_threads'))
        settings = _settings_from_solver_kwargs(solver_kwargs)
        self._assemble_settings(rank, n_ranks, solver_kwargs['order'], settings)
        self._is_fully_distributed = False
        self.mesh = mesh
        self.ops = ops

        # root 用 METIS 分区，广播给全部 rank
        if self.rank == 0:
            logger.info(f"Partitioning mesh into {n_ranks} parts (METIS on root)...")
            cell_partition = partition_mesh(face_connectivity, n_ranks, n_cells=mesh.n_cells)
        else:
            cell_partition = None
        if n_ranks > 1:
            from autoflowcfd.core.mpi.comm import bcast_from_root
            cell_partition = bcast_from_root(cell_partition)
        if cell_partition is None:
            raise RuntimeError("root 没有广播单元分区（partition_mesh 的结果）")

        # 换阶时用同一个 cell_partition 重建 partition/dist_flat_face（`mesh.face_connectivity` 是阶数无关的
        # 拓扑信息，见 `high_order_mesh_order.py::set_order`）
        self._oc_cell_partition = cell_partition

        # 分区与 halo 探测必须在**全局、未裁剪**的面连接关系上做：cell_partition 的下标是全局 cell id，
        # 一条面两侧是否属于不同 rank 要用 cell_partition[owner]/cell_partition[neighbor] 判断
        self.partition = build_distributed_partition(face_connectivity, cell_partition, self.rank, n_ranks)

        if is_root():
            logger.info(
                f"Rank {self.rank}: {self.partition.n_local_cells} local cells, "
                f"{self.partition.n_halo} halo cells, "
                f"{len(self.partition.neighbor_ranks)} neighbors"
            )

        # 3. 构建分布式面几何——必须排在下面 HaloExchange/DistributedFRState
        # 构造**之前**：当 cell_partition 非空时，这一步内部会调用
        # extend_halo_for_flux_point_cross_references 原地扩展
        # self.partition 的 halo_cells/send_lists/recv_lists（覆盖 FR
        # Flux Point 多源交叉插值依赖，真实 cube_demo 网格上 16%~19% 的
        # 单元需要这种扩展，不是边缘情况）。HaloExchange 会按*构造时刻*
        # 的 send_lists/recv_lists 大小预分配定长 buffer——如果它先构造、
        # 面几何扩展后发生，buffer 要么大小对不上（形状不匹配崩溃），
        # 要么根本没有新增邻居 rank 的 key（KeyError，或更隐蔽地静默
        # 跳过、留下未初始化的 halo 数据）。第四次评审第二轮复核发现
        # 此前的构造顺序正是反的，这里改为先做面几何（含 halo 扩展），
        # 再构造依赖最终 halo 布局的对象。
        #
        # 传入（全局）cell_partition 以便扩展 halo 层；完全分布式加载的 halo 扩展由 root 在全局网格上预先
        # 完成（`distributed_mesh_loader/package.py`），不经过这里。
        self.dist_flat_face = build_distributed_flat_face(mesh, ops, self.partition, cell_partition=cell_partition)

        # 4. 守恒状态与 halo 交换器（在面几何 / halo 扩展之后构造，理由见上）
        n_sps = mesh.n_sps_per_cell
        self._assemble_state(n_sps)

        # 5. 本地求解器按 solver_kwargs 延迟构造（`local_solver`）
        self.solver_kwargs = solver_kwargs
        self._local_solver = None

        # 6. 湍流模型对象与单机同一个初始化（`init_turbulence_models`：DES 网格尺度按全局网格现算）；产生项渐变计数器
        # 在时间格式确定后由 init_production_ramp 设置（每步重建的湍流适配器把计数器转发回本对象）
        n_local = self.partition.n_local_cells
        from autoflowcfd.core.turbulence.registry import has_transport_equations
        from autoflowcfd.core.fr_solver.turbulence import init_turbulence_models
        init_turbulence_models(self, n_local, n_sps)
        if has_transport_equations(self.turb_model_name):
            # 壁面距离：与单机同一个来源（CLI 由体网格 WALL 边界面构造后以
            # `wall_distance_source=` 传入），在本 rank compact 解点上查询；
            # 换阶重建时用同一个来源重查（distributed_order_continuation）
            from autoflowcfd.core.mpi.distributed_turbulence import (
                apply_distributed_wall_distance, compute_distributed_wall_distance,
            )
            self._wall_distance_source = wall_distance_source
            self.wall_distance_compact = compute_distributed_wall_distance(
                self.dist_flat_face, mesh, wall_distance_source)
            apply_distributed_wall_distance(self.turb_model, self.wall_distance_compact, self)

            # DDES/IDDES：`init_turbulence_models` 已设好 `self.ddes_model` 与**全局**尺寸的
            # `self._iddes_h_max/_iddes_h_wn`（"传统模式"下 self.mesh 是完整全局网格；两者是单元
            # 自身节点坐标的函数，与分区无关），按 compact_global_ids 切片即得 compact 空间版本。
            if self.turb_model_name in ('DDES', 'IDDES'):
                if self.ddes_model is None:
                    raise RuntimeError(
                        "init_turbulence_models 没有为 DDES/IDDES 设置 self.ddes_model（该函数的对应分支被跳过或"
                        "修改），检查 fr_solver/turbulence/init.py::init_turbulence_models 的 DDES/IDDES 分支")
                compact_global_ids = self.dist_flat_face.compact_global_ids
                self.iddes_h_max_compact = self._iddes_h_max[compact_global_ids]
                self.iddes_h_wn_compact = self._iddes_h_wn[compact_global_ids]

        # 7. 时间推进与 CFL 策略
        self._assemble_time_stepping(settings)

        # 8. 同步
        barrier()
