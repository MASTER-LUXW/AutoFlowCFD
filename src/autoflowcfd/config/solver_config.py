"""求解器配置数据类。

本模块定义了 AutoFlowCFD 求解器的核心配置结构，
包括稳态和瞬态仿真配置。

关键组件:
    - BackendType: 计算后端枚举 (cpu/gpu/auto)
    - TurbulenceModel: 湍流模型枚举
    - TimeIntegrationScheme: 时间积分方案枚举
    - SolverConfig: 基础求解器配置
    - SteadyConfig: 稳态特定配置
    - TransientConfig: 瞬态特定配置

示例:
    >>> from autoflowcfd.config import SteadyConfig, TransientConfig
    >>> steady = SteadyConfig(backend="gpu", order=3, max_iter=5000)
    >>> transient = TransientConfig(dt=1e-4, total_time=0.3)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os


class BackendType(str, Enum):
    """计算后端类型枚举。"""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class TurbulenceModel(str, Enum):
    """湍流模型枚举。

    与求解器真正支持的集合逐一对应（见 fr_solver/turbulence/init.py::
    init_turbulence_models）：NONE/SST/SA/DDES/IDDES/WMLES/LES。2026-09-02 曾删除
    从未实现的 `SA`/`DES` 占位值；2026-10-04 SA-neg 真正实现后（高阶下 SST 的 C0 折点
    使 P2/P3 无法快速收敛，见 core/turbulence/sa 模块文档）重新加入 `SA`。
    """
    NONE = "none"       # 层流 Navier-Stokes（无湍流模型）
    SST_KW = "sst_kw"
    SA = "sa"           # SA-neg（Allmaras, Johnson & Spalart 2012，core/turbulence/sa）
    DDES = "ddes"
    IDDES = "iddes"
    WMLES = "wmles"
    LES = "les"



# **2026-09-18：配置层原本在这里定义了一个独立的同名枚举，已删除。**
#
# 那是一个"同一语义两个事实来源"的典型：配置层的是 `str, Enum`、取值
# `backward_euler/rk2/rk3/ab3`；核心层的是普通 `Enum`、取值
# `forward_euler/ssp_rk2/ssp_rk3/imex_euler/dual_time`。两者取值范围既
# 不相同、也不可互相表达（配置层没有 dual_time/imex_euler，核心层没有
# backward_euler/ab3 —— 后两个在核心层**从未实现**）。
#
# 后果不是理论上的：`api_config.py::api_create_transient_config` 把
# `dual-time` 映到 `BACKWARD_EULER`、把 `imex` 映到 `RK3`（两者都不是
# 所选的方案），未知取值还 `.get(..., RK3)` 静默退回。之所以一直没人
# 发现，是因为这个字段本身未接入求解器 —— 而"未接入"不是让错误映射
# 留在代码里的理由：一旦有人把它接上就是静默跑错格式。
#
# 现在配置层直接复用核心层那个枚举，字符串解析统一走
# `core/time_integration/base.py::scheme_from_name`（唯一那张词汇表，
# 未知取值报错而不是静默换方案）。
from autoflowcfd.core.time_integration.base import DEFAULT_STEADY_TOL  # noqa: E402
from autoflowcfd.core.time_integration.base import (  # noqa: E402
    TimeIntegrationScheme,
)


@dataclass
class SolverConfig:
    """基础求解器配置。

    属性:
        backend: 计算后端 (cpu/gpu/auto)
        order: FR 离散阶数 (1/2/3)，即 C-01 描述的 polynomial_order (P)
            ——沿用 `order` 这个名字而不是另起一个 polynomial_order 字段，
            是因为它已经和 CLI `--order`/API `run_steady(order=...)`/
            `FRSolver(order=...)` 全程统一使用同一个名字；引入第二个
            同义字段只会制造"两个名字哪个才是真的"的新歧义，不是真正
            解决 C-01。
        turbulence: 湍流模型
        gpu_device: GPU 设备 ID（仅 GPU 模式）
        n_threads: CPU 线程数（仅 CPU 模式，auto=检测）
        output_dir: 输出目录路径
        checkpoint_interval: 检查点保存间隔（步数）
        mu_molecular: 分子动力粘度 (Pa*s)，默认 1.8e-5（标准状态下空气）。
            与 core/fr_solver/solver.py::FRSolver 构造参数同名同义——
            粘性残差组装和粘性 CFL 步长必须用同一个值，不能各自硬编码
            （历史教训见该参数在 FRSolver 里的文档）。CLI `solve steady`/
            `solve transient` 的 `--mu-molecular` 选项、或本 YAML 配置的
            `mu_molecular` 键，都改这一个字段。
        phase_max_iter: Order Continuation（`order>=1` 时触发）非最终
            阶段（P0/P1/...，不含目标阶数）各自的最大迭代步数上限。
            None（默认）时保留旧行为——`max_iter // len(orders)` 按阶段
            数机械均分，目标阶数与非最终阶段拿到同一份额，与目标阶数
            本身是否已经收敛毫无关系。传具体值后非最终阶段各自最多跑
            这么多步，**目标阶数改为吃掉这次求解剩余的全部步数**，不再
            随阶段数被稀释——见 `core/utils/order_continuation.py::
            run_order_continuation` 同名参数文档（2026-09-01，用户直接
            指出"不想机械地按 max_iter // len(orders) 判断"）。CLI
            `--phase-max-iter` 选项、或本 YAML 配置的 `phase_max_iter`
            键，都改这一个字段。
        residual_drop_threshold: 同上，仅 Order Continuation 生效，单个
            非最终阶段判定"可以提前升阶"的残差下降倍数，默认 100（降 2
            个数量级），原来硬编码，现在可配置。CLI
            `--residual-drop-threshold` 选项对应本字段。

    示例:
        >>> config = SolverConfig(backend="gpu", order=3)
        >>> print(config.backend)
        'gpu'
    """
    backend: BackendType = BackendType.AUTO
    order: int = 2
    turbulence: TurbulenceModel = TurbulenceModel.SST_KW
    gpu_device: int = 0
    n_threads: int = -1  # -1 表示自动检测
    output_dir: str = "./results"
    checkpoint_interval: int = 100
    turbulence_intensity: float = 0.01  # 来流湍流强度 Tu（默认 1%）
    viscosity_ratio: float = 5.0  # 来流粘性比 VR = nu_t/nu
    mu_molecular: float = 1.8e-5  # 分子动力粘度 (Pa*s)，默认标准状态下空气
    artificial_viscosity_enabled: bool = False  # 问题单元人工粘性（熵残差判据），与 CLI --artificial-viscosity 对应
    artificial_viscosity_alpha: float = 1.0  # 人工粘性强度标定常数，与 CLI --av-alpha 对应
    phase_max_iter: Optional[int] = None  # Order Continuation 非最终阶段最大步数上限，None=旧行为(按阶段数均分)
    residual_drop_threshold: float = 1e2  # Order Continuation 单阶段提前升阶所需的残差下降倍数

    def __post_init__(self):
        """初始化后验证配置。"""
        # 验证阶数
        if self.order not in [1, 2, 3]:
            raise ValueError(f"FR 阶数必须是 1, 2, 或 3，得到 {self.order}")

        # 验证 gpu_device
        if self.gpu_device < 0:
            raise ValueError(f"GPU 设备 ID 必须为非负数，得到 {self.gpu_device}")

        # 验证 n_threads
        if self.n_threads == -1:
            # 自动检测 CPU 核心数
            import multiprocessing
            self.n_threads = multiprocessing.cpu_count()
        elif self.n_threads < 1:
            raise ValueError(f"线程数必须为正数，得到 {self.n_threads}")

        # 验证湍流参数
        if not (0 < self.turbulence_intensity <= 1.0):
            raise ValueError(f"湍流强度 Tu 必须在 (0, 1] 范围内，得到 {self.turbulence_intensity}")
        if self.viscosity_ratio <= 0:
            raise ValueError(f"粘性比 VR 必须为正数，得到 {self.viscosity_ratio}")
        if self.mu_molecular <= 0:
            raise ValueError(f"分子动力粘度 mu_molecular 必须为正数，得到 {self.mu_molecular}")
        if self.artificial_viscosity_alpha <= 0:
            raise ValueError(f"人工粘性强度 artificial_viscosity_alpha 必须为正数，得到 {self.artificial_viscosity_alpha}")


        # **2026-09-18：这里原本 `os.makedirs(self.output_dir)`，已删除。**
        #
        # 那是"构造一个值对象就产生文件系统副作用"——只要有人构造
        # `SteadyConfig()` / `TransientConfig()`（默认 output_dir 是
        # `./results` / `./transient_results`），当前工作目录下就会凭空
        # 出现那个目录。真实后果：**跑一遍 `pytest tests/unit` 就在仓库
        # 根目录留下 `results/`、`transient_results/`、`checkpoints/`**
        # ——用户两次明确要求项目文件夹里不许出现这些目录，根源就在这。
        #
        # 而且它是**冗余**的：真正写输出的两处都自己建目录
        # （`cli/solve/checkpoint_io.py::save_results` 与
        # `cli/solve/steady.py` 的保存分支都有
        # `os.makedirs(output_dir, exist_ok=True)`）。要显式预建请调用
        # 下面的 `ensure_output_dir()`。

    def ensure_output_dir(self) -> str:
        """真正需要写输出时再建目录，返回路径。

        与构造期建目录的区别：**调用者显式表达了"我要写了"这个意图**。
        配置对象本身是纯值对象，构造它不应当碰文件系统。
        """
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir
    
    @property
    def is_gpu(self) -> bool:
        """检查是否使用 GPU 后端。"""
        return self.backend == BackendType.GPU
    
    @property
    def is_cpu(self) -> bool:
        """检查是否使用 CPU 后端。"""
        return self.backend == BackendType.CPU


def cfl_triplet_errors(cfl_init: Optional[float], cfl_max: Optional[float],
                       cfl_min: Optional[float]) -> list:
    """自适应 CFL 三元组的校验（`SteadyConfig` / `TransientConfig` 构造与 `ConfigSchema` 共用）。

    只校验给出的值（None = 由 CFL 律按时间格式取默认，见 `build_cfl_policy`）：都必须为正；给出的两两之间满足
    min <= init <= max。cfl_min > cfl_max 会让控制器的"收缩"分支把 CFL 调高并突破上限（adaptive_cfl 第 11 条），
    cfl_min > cfl_init 会让初值被钳上去，实际跑的不是请求的 CFL。
    """
    named = {"cfl_init": cfl_init, "cfl_max": cfl_max, "cfl_min": cfl_min}
    errors = [f"{k} 必须为正数（CFL），得到 {v}" for k, v in named.items() if v is not None and v <= 0]
    for lo, hi in (("cfl_min", "cfl_init"), ("cfl_init", "cfl_max"), ("cfl_min", "cfl_max")):
        if named[lo] is not None and named[hi] is not None and named[lo] > named[hi]:
            errors.append(f"{lo} ({named[lo]}) 不能超过 {hi} ({named[hi]})（CFL）")
    return errors


def _raise_on_cfl_triplet(config) -> None:
    errors = cfl_triplet_errors(config.cfl_init, config.cfl_max, config.cfl_min)
    if errors:
        raise ValueError("; ".join(errors))


@dataclass
class SteadyConfig(SolverConfig):
    """稳态仿真配置。
    
    继承 SolverConfig 的所有属性并添加稳态特定参数。
    
    属性:
        max_iter: 最大迭代步数
        cfl_init / cfl_max / cfl_min: 自适应 CFL 的初值 / 上限 / 下限。默认 None = 按时间格式取该格式 CFL 律
            签名里的默认值（显式 `AdaptiveCFLController` 与隐式 `SERCFLController` 相差两个数量级，见
            `core/time_integration/adaptive_cfl/policy.py::build_cfl_policy`），与 CLI `--cfl-start/--cfl-max/
            --cfl-min`、`FRSolver.__init__` 同一约定。2026-10-05 以前这里写死显式那一组（0.03/0.06/0.01）：CLI
            与求解器 2026-09-25 已改为 None，配置层这份拷贝被漏掉，经 API 选隐式格式时隐式 CFL 律会被压在
            0.06 的上限下。三者给出时的序关系由 `cfl_triplet_errors` 校验。
        convergence_tol: 相对收敛容差（对应 CLI `--tol`；默认取 `DEFAULT_STEADY_TOL`）
        rho_inf: 自由流密度 (kg/m^3) - 初始条件、入口/远场边界条件和
            Cd/Cl 归一化的单一真实来源，确保三者始终保持一致。
        vel_inf: 自由流速度大小 (m/s)，与 rho_inf 作用相同。
        p_inf: 自由流静压 (Pa)，与 rho_inf 作用相同。

    示例:
        >>> config = SteadyConfig(
        ...     backend="gpu",
        ...     order=3,
        ...     max_iter=5000,
        ...     cfl_init=0.03,
        ...     cfl_max=0.5
        ... )
    """
    max_iter: int = 50
    cfl_init: Optional[float] = None
    cfl_max: Optional[float] = None
    cfl_min: Optional[float] = None
    convergence_tol: float = DEFAULT_STEADY_TOL
    rho_inf: float = 1.225
    vel_inf: float = 33.33
    p_inf: float = 101325.0

    def __post_init__(self):
        """验证稳态配置。"""
        super().__post_init__()

        # 验证迭代次数
        if self.max_iter < 1:
            raise ValueError(f"最大迭代次数必须为正数，得到 {self.max_iter}")

        _raise_on_cfl_triplet(self)

        # 验证收敛容差
        if self.convergence_tol <= 0:
            raise ValueError(f"收敛容差必须为正数，得到 {self.convergence_tol}")

        # 验证自由流条件
        if self.rho_inf <= 0:
            raise ValueError(f"rho_inf 必须为正数，得到 {self.rho_inf}")
        if self.vel_inf <= 0:
            raise ValueError(f"vel_inf 必须为正数，得到 {self.vel_inf}")
        if self.p_inf <= 0:
            raise ValueError(f"p_inf 必须为正数，得到 {self.p_inf}")


@dataclass
class TransientConfig(SolverConfig):
    """瞬态仿真配置。
    
    继承 SolverConfig 的所有属性并添加瞬态特定参数。
    
    属性:
        dt: 时间步长（秒）
        total_time: 总物理时间（秒）
        time_scheme: 时间积分方案
        init_from_checkpoint: 从 checkpoint 初始化（对应 CLI `--init-from`）
        rho_inf, vel_inf, p_inf: 自由流条件，含义和作用与 SteadyConfig 相同
            （初始条件、边界条件和 Cd/Cl 归一化的单一真实来源）。

    示例:
        >>> config = TransientConfig(
        ...     backend="gpu",
        ...     order=3,
        ...     dt=1e-4,
        ...     total_time=0.3,
        ...     time_scheme="dual-time"
        ... )
    """
    dt: float = 1e-4
    total_time: float = 0.1
    time_scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3
    # 自适应 CFL 三元组：`rk3/imex` 下 `step()` 按逐单元局部 CFL 步长推进，控制器是激活的；`dual-time` 档外层不构造
    # 控制器（内层伪时间有自己的步长调节）。默认 None 的含义见 SteadyConfig 文档 cfl_init 一节
    cfl_init: Optional[float] = None
    cfl_max: Optional[float] = None
    cfl_min: Optional[float] = None
    init_from_checkpoint: Optional[str] = None
    rho_inf: float = 1.225
    vel_inf: float = 33.33
    p_inf: float = 101325.0

    def __post_init__(self):
        """验证瞬态配置。"""
        super().__post_init__()

        # 验证时间步长
        if self.dt <= 0:
            raise ValueError(f"时间步长必须为正数，得到 {self.dt}")

        # 验证总时间
        if self.total_time <= 0:
            raise ValueError(f"总时间必须为正数，得到 {self.total_time}")

        _raise_on_cfl_triplet(self)

        # 计算总步数
        self.total_steps = int(self.total_time / self.dt)

        # 验证自由流条件
        if self.rho_inf <= 0:
            raise ValueError(f"rho_inf 必须为正数，得到 {self.rho_inf}")
        if self.vel_inf <= 0:
            raise ValueError(f"vel_inf 必须为正数，得到 {self.vel_inf}")
        if self.p_inf <= 0:
            raise ValueError(f"p_inf 必须为正数，得到 {self.p_inf}")
        if self.total_steps < 1:
            raise ValueError(
                f"总步数必须至少为 1，得到 {self.total_steps} "
                f"(dt={self.dt}, total_time={self.total_time})"
            )
    
    @property
    def n_steps(self) -> int:
        """获取总时间步数。"""
        return self.total_steps


def create_steady_config(**kwargs) -> SteadyConfig:
    """创建带有默认值的稳态配置的工厂函数。
    
    Args:
        **kwargs: 覆盖默认值
        
    Returns:
        SteadyConfig: 配置好的稳态求解器配置
        
    示例:
        >>> config = create_steady_config(backend="gpu", max_iter=10000)
    """
    return SteadyConfig(**kwargs)


def create_transient_config(**kwargs) -> TransientConfig:
    """创建带有默认值的瞬态配置的工厂函数。
    
    Args:
        **kwargs: 覆盖默认值
        
    Returns:
        TransientConfig: 配置好的瞬态求解器配置
        
    示例:
        >>> config = create_transient_config(dt=1e-5, total_time=0.5)
    """
    return TransientConfig(**kwargs)
