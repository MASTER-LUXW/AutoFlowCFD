"""AutoFlowCFD V2.0 - 分布式求解器与网格来源无关的装配（传统模式 `__init__` 与完全分布式
`from_fully_distributed_package` 共用）

两条构造路径只在"几何从哪来"上不同：传统模式每个 rank 持有完整全局网格、自己分区并构建本 rank 的面几何，湍流
模型所需的全局几何量（壁距、DES 网格尺度）自己现算；完全分布式加载由 root 预先切好随包下发。来流、决定物理解的
参数、壁面模型、守恒状态与 halo 交换、时间推进与 CFL 策略都与几何来源无关，在这里装配一次。

2026-10-21（I-110）以前两条路径各自设置这几十个属性，历史上多次出现"某属性只在一条路径设置"的崩溃或静默差异
（完全分布式路径的 `freestream` 曾只在湍流分支设置、`--threads` 曾被忽略，多 GPU 完全分布式曾缺
`low_mach_precond_enabled`）。
"""

from typing import Any, NamedTuple, Optional

from autoflowcfd.core.mpi import get_rank
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.time_integration.base import TimeIntegrator, require_distributed_scheme
from autoflowcfd.core.turbulence.dual_time import reset_dual_time_history
from autoflowcfd.core.turbulence.wall_model.treatment import build_wall_model, validate_wall_treatment


class DistributedRunSettings(NamedTuple):
    """与网格来源无关的运行设置（传统模式取自 `solver_kwargs`，完全分布式取自紧凑包）。"""
    turb_model_name: str               # 规范化的大写模型名
    freestream: dict                   # rho_inf / vel_inf / p_inf / mach_ref，及攻角、侧滑角
    mu_molecular: float
    turbulence_intensity: float
    viscosity_ratio: float
    sem_num_eddies: Optional[int]
    artificial_viscosity_enabled: bool
    artificial_viscosity_alpha: float
    wall_treatment: str
    time_scheme: Any
    dual_time_inner_iter: int
    time_accurate: bool
    adaptive_cfl: bool
    cfl_start: Optional[float]
    cfl_max: Optional[float]
    cfl_min: Optional[float]
    low_mach_precond: bool


def canonical_turbulence_model(name) -> str:
    """规范化的大写湍流模型名；不是规范名（拼写错误、未知模型）时报错，而不是静默按层流运行。"""
    from autoflowcfd.core.turbulence.registry import require_supported

    upper = str(name).upper() if name is not None else "NONE"
    if upper != require_supported(name):
        raise NotImplementedError(f"分布式求解器收到无法识别的湍流模型 {name!r}（应为规范化的大写模型名）")
    return upper


class _DistributedAssemblyMixin:
    """`DistributedFRSolver` 的装配方法（只含方法；属性由两条构造路径经这里设置）。"""

    partition: Any
    freestream: Any
    turb_model_name: Any
    turb_model: Any
    ddes_model: Any
    sgs_model: Any
    wall_distance_compact: Any
    iddes_h_max_compact: Any
    iddes_h_wn_compact: Any

    def _assemble_settings(self, rank: Optional[int], n_ranks: int, order: int,
                           settings: DistributedRunSettings) -> None:
        """身份、阶数、来流与决定物理解的参数、壁面模型；湍流模型对象与壁距等几何量先置空，由各路径按几何来源给出。"""
        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        # `order` = 目标阶数，`current_order` = 当前实际所在阶数（与单机 FRSolver 同一约定）
        self.order = int(order)
        self.current_order = self.order
        self.order_continuation_enabled = True      # 与单机 FRSolver 同一缺省，调用方构造后可关闭
        self.turb_model_name = settings.turb_model_name
        self._turb_model_upper = settings.turb_model_name
        self.freestream = dict(settings.freestream)
        self.mu_molecular = settings.mu_molecular
        self._turbulence_intensity = settings.turbulence_intensity
        self._viscosity_ratio = settings.viscosity_ratio
        self._sem_num_eddies = settings.sem_num_eddies
        # 问题单元人工粘性（与单机 FRSolver 同名参数、同一实现；见 core/mpi/distributed_artificial_viscosity.py）
        self.artificial_viscosity_enabled = bool(settings.artificial_viscosity_enabled)
        self.artificial_viscosity_alpha = float(settings.artificial_viscosity_alpha)
        self.wall_treatment = validate_wall_treatment(settings.wall_treatment)
        # 壁面模型：WMLES 或 RANS 自动壁面处理（与单机同一个构造函数）
        self.wall_model = build_wall_model(self.turb_model_name, self.wall_treatment)
        # `step()` 无条件读取这些属性，不适用的为 None
        self.turb_model = None
        self.ddes_model = None
        self.sgs_model = None
        self.wall_distance_compact = None
        self.iddes_h_max_compact = None
        self.iddes_h_wn_compact = None

    def _assemble_state(self, n_sps: int) -> None:
        """本 rank 的守恒状态（均匀来流初值，速度方向取自攻角 / 侧滑角，与边界 Q_free 同一方向）、收敛历史与
        halo 交换器。必须在面几何构建（含 halo 扩展）之后调用：halo 交换器按构造时刻的收发表预分配缓冲区。"""
        from autoflowcfd.core.fr_solver.convergence_history import ConvergenceHistory
        from autoflowcfd.core.utils.flow_direction import direction_from_freestream

        self.state = DistributedFRState(self.partition, n_sps)
        self.convergence_history = ConvergenceHistory()
        velocity = float(self.freestream["vel_inf"]) * direction_from_freestream(self.freestream)
        self.state.initialize_uniform(rho=self.freestream["rho_inf"], u=float(velocity[0]), v=float(velocity[1]),
                                      w=float(velocity[2]), p=self.freestream["p_inf"])
        self.halo_exchange = HaloExchange(self.partition, n_sps, self.state.n_vars)

    def _assemble_time_stepping(self, settings: DistributedRunSettings) -> None:
        """时间推进器（与单机同一套 Shu-Osher SSP-RK / 双时间步 / Newton-Krylov 分派，`step()` 按格式分派）、产生项
        渐变、上一时间层与 Newton 跨步状态、CFL 策略与低马赫预处理（`configure_time_stepping`，各后端唯一来源）。

        CFL 控制器按**全局**残差范数（allreduce 之后）更新，全部 rank 得到同一个 CFL 数。`dt=1.0` 只是占位：
        推进用的步长由 `step()` 每次构造的逐单元局部步长决定。
        """
        from autoflowcfd.core.fr_solver.turbulence.init import init_production_ramp
        from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import reset_newton_state
        from autoflowcfd.core.time_integration.time_accurate import configure_time_stepping

        time_scheme = require_distributed_scheme(settings.time_scheme)
        self.time_integrator = TimeIntegrator(scheme=time_scheme, dt=1.0,
                                              dual_time_steps=settings.dual_time_inner_iter)
        init_production_ramp(self, time_scheme)
        # DUAL_TIME 的 BDF2 上一物理时间层（None：尚未跑过一个物理步，退化为 BDF1，与单机同一约定）
        reset_dual_time_history(self)
        reset_newton_state(self)
        configure_time_stepping(
            self, time_scheme, time_accurate=bool(settings.time_accurate), adaptive_cfl=bool(settings.adaptive_cfl),
            cfl_start=settings.cfl_start, cfl_max=settings.cfl_max, cfl_min=settings.cfl_min,
            low_mach_precond=settings.low_mach_precond)
        # 上一步的涡粘场（local 排列），供下一步的粘性 CFL 限制使用（与单机读取湍流模型已存字段同一时序）
        self._prev_mu_t_local = None
