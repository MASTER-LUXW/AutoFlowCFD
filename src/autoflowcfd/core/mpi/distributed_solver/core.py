"""AutoFlowCFD V2.0 - DistributedFRSolver 主类：传统模式构造

从 `src/autoflowcfd/core/mpi/distributed_solver.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

from typing import Optional
from loguru import logger
from autoflowcfd.core.turbulence.wall_model.treatment import build_wall_model, validate_wall_treatment
from autoflowcfd.core.mpi import get_rank, is_root
from autoflowcfd.core.mpi.partition import (
    partition_mesh,
    build_distributed_partition,
)
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.distributed_flat_face import (
    build_distributed_flat_face,
)
from autoflowcfd.core.mpi.comm import barrier
from autoflowcfd.core.time_integration.base import TimeIntegrator, require_distributed_scheme
from .from_package import _DistributedFromPackageMixin
from autoflowcfd.core.fr_solver.solver.solve_loop import DistributedSolveLoopMixin
from .step import _DistributedStepMixin
from .support import _DistributedSupportMixin
from autoflowcfd.core.turbulence.dual_time import reset_dual_time_history


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


class DistributedFRSolver(_DistributedFromPackageMixin, _DistributedStepMixin, DistributedSolveLoopMixin,
                          _DistributedSupportMixin):
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
        turb_model_name = solver_kwargs['turb_model_name']
        turb_model_upper = str(turb_model_name).upper() if turb_model_name is not None else 'NONE'
        from autoflowcfd.core.turbulence.registry import require_supported
        if turb_model_upper != require_supported(turb_model_name):
            raise NotImplementedError(
                f"MPI 分布式求解器收到无法识别的 turbulence_model '{turb_model_name}'。"
            )
        self._turb_model_upper = turb_model_upper
        # 真实 bug 修复（2026-09-02，扩展 DDES/IDDES 分布式支持时发现，
        # 与本次新增功能无关，SST 分布式路径同样中招）：下面 SST/DDES/
        # IDDES 分支调用 `init_turbulence_models(self, n_local, n_sps)`
        # 内部按 `solver.turb_model_name` 分派，但此前这里从未真正
        # 设置过 `self.turb_model_name` 这个属性——`DistributedFRSolver`
        # 构造时只要请求 SST（或本次新增的 DDES/IDDES），必然在这里
        # `AttributeError` 崩溃。此前从未被测试捕捉到是因为
        # `test_distributed_turbulence.py` 只单独测试更底层的
        # `distributed_compute_turbulence_source_and_viscosity`，从未
        # 真正走过 `DistributedFRSolver.__init__` 这条构造路径。
        self.turb_model_name = turb_model_upper

        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        self.mesh = mesh
        self.ops = ops

        # Order Continuation 支持（2026-09-02，见 core/mpi/
        # distributed_order_continuation.py 模块文档）：`self.order`/
        # `self.current_order` 与单机 FRSolver 同一约定
        # （`order`=目标阶数，`current_order`=当前实际所在阶数，两者
        # 在阶数爬升过程中的中间阶段不相等）。
        self.order = int(solver_kwargs['order'])
        self.current_order = self.order
        self.order_continuation_enabled = True      # 与单机 FRSolver 同一缺省，调用方构造后可关闭
        self._is_fully_distributed = False

        # root 用 METIS 分区，广播给全部 rank
        if self.rank == 0:
            logger.info(f"Partitioning mesh into {n_ranks} parts (METIS on root)...")
            cell_partition = partition_mesh(face_connectivity, n_ranks, n_cells=mesh.n_cells)
        else:
            cell_partition = None
        if n_ranks > 1:
            from autoflowcfd.core.mpi.comm import bcast_from_root
            cell_partition = bcast_from_root(cell_partition)

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

        # 4. 初始化分布式状态（在面几何/halo 扩展之后构造）
        n_sps = mesh.n_sps_per_cell
        self.state = DistributedFRState(self.partition, n_sps)
        from autoflowcfd.core.fr_solver.convergence_history import ConvergenceHistory

        self.convergence_history = ConvergenceHistory()
        n_vars = self.state.n_vars
        # 均匀自由流场初始化（真实 bug 修复，2026-09-02，见
        # DistributedFRState.initialize_uniform 文档）：此前这里从未
        # 对新构造的 state 赋初值，conserved state 恒为全零。
        # 初场速度方向必须与边界 Q_free 用同一个来流方向（2026-09-17）
        from autoflowcfd.core.utils.flow_direction import freestream_velocity
        self.rho_inf = solver_kwargs['rho_inf']
        self.vel_inf = solver_kwargs['vel_inf']
        self.p_inf = solver_kwargs['p_inf']
        _v0 = freestream_velocity(
            self.vel_inf,
            _kwarg(solver_kwargs, 'aoa_deg'),
            _kwarg(solver_kwargs, 'aos_deg'))
        self.state.initialize_uniform(
            rho=self.rho_inf,
            u=float(_v0[0]), v=float(_v0[1]), w=float(_v0[2]),
            p=self.p_inf,
        )

        # 5. 初始化 halo 交换器（同样必须在面几何/halo 扩展之后构造，
        # 理由见上）
        self.halo_exchange = HaloExchange(self.partition, n_sps, n_vars)

        # 6. 保存 solver kwargs 用于创建本地求解器
        self.solver_kwargs = solver_kwargs
        self._local_solver = None  # 延迟初始化

        # 6b. SST 湍流模型初始化（2026-09-02，见 core/mpi/
        # distributed_turbulence.py 模块文档）。turb_model 只按
        # n_local_cells 分配（与 self.state 一致，local+halo 的 k/omega
        # 通过独立的 2-var halo 交换器实时获取，不常驻）。复用单机路径
        # 完全相同的 `init_turbulence_models`（鸭子类型：只需要
        # `.turb_model_name`/`.freestream`/`.mu_molecular`/
        # `._turbulence_intensity`/`._viscosity_ratio` 这几个属性，
        # `self` 已经或即将全部满足）。
        n_local = self.partition.n_local_cells
        self.turb_model = None
        self.wall_distance_compact = None
        self.ddes_model = None
        self.iddes_h_max_compact = None
        self.iddes_h_wn_compact = None
        self.wall_model = None
        # LES：WALE 是纯代数 SGS 模型，没有跨步状态，也不需要壁距——单独一个分支，不并入下面输运模型的构造块。
        self.sgs_model = None
        from autoflowcfd.core.turbulence.registry import uses_algebraic_sgs
        if uses_algebraic_sgs(turb_model_upper):
            from autoflowcfd.core.turbulence.sgs import WALEModel
            self.sgs_model = WALEModel()
        # 自由来流条件**无条件**设置（2026-09-18）：单机 FRSolver 的
        # `self.freestream` 从来就是无条件的（solver.py::__init__），这边
        # 却挂在湍流分支里——于是 `turbulence_model=none/les` 的分布式
        # 运行没有 `.freestream`。任何"按来流量级归一化"的消费方都会因此
        # 在这条路径上崩溃或静默退回内置缺省，BJ 越界判据（门控滤波的
        # 默认判据，其绝对地板必须用来流参考量级）正是其中一个。
        # 同一个语义只允许有一个事实来源、且两条后端不能有不同的可用性。
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref
        self.freestream = {
            "rho_inf": self.rho_inf, "vel_inf": self.vel_inf,
            "p_inf": self.p_inf,
            "mach_ref": resolve_mach_ref(self.rho_inf, self.vel_inf, self.p_inf),
            # 见 gpu_solver.py 同一处说明（2026-09-17）
            "aoa_deg": float(_kwarg(solver_kwargs, "aoa_deg")),
            "aos_deg": float(_kwarg(solver_kwargs, "aos_deg")),
        }
        # 决定物理解的参数同样**无条件**设置（2026-09-25）：此前只在
        # SST/DDES/IDDES/WMLES 分支里设，于是 none/les 运行的分布式 checkpoint
        # 按替身缺省值写粘度与 Tu/VR（见 core/utils/checkpoint_physics.py）。
        self.mu_molecular = solver_kwargs['mu_molecular']
        self._turbulence_intensity = _kwarg(solver_kwargs, 'turbulence_intensity')
        self._viscosity_ratio = _kwarg(solver_kwargs, 'viscosity_ratio')
        # 问题单元人工粘性（与单机 FRSolver 同名参数、同一实现；见
        # core/mpi/distributed_artificial_viscosity.py）
        self.artificial_viscosity_enabled = bool(_kwarg(solver_kwargs, 'artificial_viscosity_enabled'))
        self.artificial_viscosity_alpha = float(_kwarg(solver_kwargs, 'artificial_viscosity_alpha'))
        self.wall_treatment = validate_wall_treatment(_kwarg(solver_kwargs, 'wall_treatment'))
        from autoflowcfd.core.turbulence.registry import has_transport_equations
        # 壁面模型：WMLES 或 RANS 自动壁面处理（与单机同一个构造函数）
        self.wall_model = build_wall_model(turb_model_upper, self.wall_treatment)
        if has_transport_equations(turb_model_upper):
            from autoflowcfd.core.fr_solver.turbulence import init_turbulence_models
            init_turbulence_models(self, n_local, n_sps)  # 设置 self.turb_model（SST 族或 SA-neg）；
            # 产项渐变计数器在下面时间格式确定后由 init_production_ramp 设置（每步重建的湍流
            # 适配器把计数器转发回本对象，见 distributed_turbulence.py）。

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
            if turb_model_upper in ('DDES', 'IDDES'):
                if self.ddes_model is None:
                    raise RuntimeError(
                        "init_turbulence_models 没有为 DDES/IDDES 设置 self.ddes_model（该函数的对应分支被跳过或"
                        "修改），检查 fr_solver/turbulence/init.py::init_turbulence_models 的 DDES/IDDES 分支")
                compact_global_ids = self.dist_flat_face.compact_global_ids
                self.iddes_h_max_compact = self._iddes_h_max[compact_global_ids]
                self.iddes_h_wn_compact = self._iddes_h_wn[compact_global_ids]


        # 7. 时间推进器：与单机 FRSolver 同一套 Shu-Osher SSP-RK3 stage
        # 实现（core/time_integration/base.py），保证分布式与单机路径
        # 时间精度一致（见 step() 文档：此前这里是自称"RK3"实际执行单步
        # 前向欧拉的简化实现）。dt 在这里只是占位——真正推进用的步长由
        # step() 每次显式构造的 dt_local 数组决定，不读 self.dt。
        #
        # 真实 bug 修复（2026-09-02，用户明确要求"不允许出现完成度不是
        # 100%的功能点"后排查发现）：此前这里无条件硬编码
        # `scheme=TimeIntegrationScheme.SSP_RK3`，完全忽略调用方通过
        # `solver_kwargs['time_scheme']` 传入的值——DUAL_TIME（真正
        # 时间精度的瞬态仿真，DES/LES 场景应该用的模式）请求了也会被
        # 静默换成稳态收敛加速模式，`step()` 内部原本也只会
        # `_ssp_rk_stage_step`，即便这里读对了 scheme 也无路可走。
        # 现在真正读取 `time_scheme`/`dual_time_inner_iter`，`step()`
        # 相应按 scheme 分派（与单机 `fr_solver/step.py` 同一个设计）。
        time_scheme = require_distributed_scheme(solver_kwargs['time_scheme'])
        dual_time_steps = _kwarg(solver_kwargs, 'dual_time_inner_iter')
        self.time_integrator = TimeIntegrator(
            scheme=time_scheme, dt=1.0, dual_time_steps=dual_time_steps,
        )
        from autoflowcfd.core.fr_solver.turbulence.init import init_production_ramp
        init_production_ramp(self, time_scheme)
        # DUAL_TIME 模式下 BDF2 需要的上一物理时间层状态（None 表示
        # 尚未跑过一个物理步，退化为 BDF1——与单机
        # `solver._dual_time_U_prev` 同一个约定）。
        reset_dual_time_history(self)
        # NEWTON_KRYLOV 跨步状态（构造时置初值，理由见 reset_newton_state 文档）
        from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import reset_newton_state

        reset_newton_state(self)

        # 自适应 CFL + 低马赫数伪时间预处理（2026-09-14 补齐）。
        # 这两个机制此前在分布式路径上都不存在，因为它们都以"存在一个由
        # CFL 数决定的逐单元局部步长"为前提，而这条路径当时用全局固定
        # dt（被记作"已接受的简化"）。局部步长已补齐（见
        # `_compute_distributed_local_time_step`），这两个机制随之接入，
        # 语义与单机 `FRSolver` 完全一致：
        #   * 控制器按**全局**残差范数更新（allreduce 之后的那个值），
        #     所有 rank 因此得到同一个 CFL 数——这是分布式下唯一正确的
        #     做法，按各自的局部残差更新会让各 rank 的 CFL 漂移、
        #     破坏一致性；
        #   * 预处理只在 SSP-RK2/RK3 下启用（DUAL_TIME 的物理时间导数项
        #     需要单独推导 Gamma 的分配方式；时间精确显式推进同样不启用）；
        # CFL 策略（控制器 or 固定 CFL）的唯一事实来源：
        # `time_integration/adaptive_cfl/policy.py::build_cfl_policy`（六个后端
        # 构造点此前各写一份且已分叉，见该模块文档）。
        # 此前这里只对 SSP-RK2/RK3 建控制器，其余方案（DUAL_TIME 的伪
        # 时间步）走 cfl.py 的替身回退 0.1，与单机的 cfl_start 不一致。
        from autoflowcfd.core.time_integration.time_accurate import configure_time_stepping

        configure_time_stepping(
            self, time_scheme, time_accurate=bool(_kwarg(solver_kwargs, 'time_accurate')),
            adaptive_cfl=bool(_kwarg(solver_kwargs, 'adaptive_cfl')),
            cfl_start=_kwarg(solver_kwargs, 'cfl_start'), cfl_max=_kwarg(solver_kwargs, 'cfl_max'),
            cfl_min=_kwarg(solver_kwargs, 'cfl_min'),
            low_mach_precond=_kwarg(solver_kwargs, 'low_mach_precond'))
        # 上一步的涡粘场（local 排列），供下一步的粘性 CFL 限制使用——
        # 与单机 `_get_turbulent_viscosity_field` 读取湍流模型已存字段
        # （即上一步的结果）是同一个时序。
        self._prev_mu_t_local = None

        # 8. 同步
        barrier()
