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

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal
import os


class BackendType(str, Enum):
    """计算后端类型枚举。"""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class TurbulenceModel(str, Enum):
    """湍流模型枚举。

    与求解器真正支持的集合逐一对应（见 fr_solver/turbulence.py::
    init_turbulence_models）：NONE/SST/DDES/IDDES/WMLES/LES。此前这里
    还有 `SA`（Spalart-Allmaras）/`DES`（非延迟 DES）两个值，但求解器
    从未实现过这两种模型——配置层能表示、真正接入求解器构造时才报错，
    是"信息源"层面的过度承诺；用户确认没有这两种模型的需求后
    （2026-09-02）移除，不再保留这两个从未有对应实现的占位值。
    """
    NONE = "none"       # 层流 Navier-Stokes（无湍流模型）
    SST_KW = "sst_kw"
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
        verbose: 启用详细日志记录
        mu_molecular: 分子动力粘度 (Pa*s)，默认 1.8e-5（标准状态下空气）。
            与 core/fr_solver/solver.py::FRSolver 构造参数同名同义——
            粘性残差组装和粘性 CFL 步长必须用同一个值，不能各自硬编码
            （历史教训见该参数在 FRSolver 里的文档）。CLI `solve steady`/
            `solve transient` 的 `--mu-molecular` 选项、或本 YAML 配置的
            `mu_molecular` 键，都改这一个字段。
        phase_max_iter: Order Continuation（`order>=2` 时触发）非最终
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

    flux_type: FR 修正函数族 + 通量点位置，'radau'（默认，VCJH g2）或
        'gauss'。透传给 `fr/operators.py::generate_fr_operators` 的
        `flux_point_type`，与 CLI `--flux-type` 同一个量。

        **本字段的历史（2026-09-15 更正）**：这里原先写着一整段"本类刻意
        不提供 flux_type 字段，因为 `compute_correction_weights` 的
        `flux_point_type` 形参是彻底的空操作、数值层从未实现第二种修正
        函数族"。那个结论在写下时或许成立，但**早已过时**：
        `fr/matrix_operators.py::compute_correction_weights` 现在有真实
        的 `if flux_point_type == 'gauss': return
        _compute_gauss_correction_derivative(...)` 分支，
        `generate_fr_operators` 也真实分派，CLI 两条 solve 命令都暴露了
        `--flux-type`、`FRSolver.__init__` 有 `flux_type` 形参并存成
        `self.flux_type` 供 Order Continuation 跨阶数复用。于是"配置类
        少一个字段"从"避免假实现"变成了它自己就是一处缺口：用 YAML 配置
        跑的用户拿不到一个 CLI 用户已经能用的真实数值方案。

        同一类过时信息在本项目造成过真实误判（AUSM+up alpha/beta 那条
        "已记录不改动"被后续会话当成当前事实复述，见
        `~/.claude/plans/zippy-painting-balloon.md` 的"信息源更正记录"），
        所以这里不只是加字段，而是把原来的论断连同它为什么过时一起写清。

        注意 `gauss` 档目前只有单机 CPU 路径支持（GPU/多 GPU/MPI 分布式
        的 ops 构造不带 `flux_point_type`），CLI 侧已有显式护栏
        （`solve_steady_command.py`）；`api.py` 的 `run_steady`/
        `run_transient` 同样会在 backend 不支持时显式报错，不静默退回
        radau。

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
    verbose: bool = False
    turbulence_intensity: float = 0.01  # 来流湍流强度 Tu（默认 1%）
    viscosity_ratio: float = 5.0  # 来流粘性比 VR = nu_t/nu
    mu_molecular: float = 1.8e-5  # 分子动力粘度 (Pa*s)，默认标准状态下空气
    flux_type: str = "radau"  # FR 修正函数族：radau（VCJH g2）| gauss
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

        # 验证 flux_type（不静默退回默认——与本项目其余开关同一条约定）
        if self.flux_type not in ("radau", "gauss"):
            raise ValueError(
                f"flux_type 必须是 'radau' 或 'gauss'，得到 {self.flux_type!r}")

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
        # （`cli/solve_checkpoint_io.py::save_results` 与
        # `cli/solve_steady_command.py` 的保存分支都有
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


@dataclass
class SteadyConfig(SolverConfig):
    """稳态仿真配置。
    
    继承 SolverConfig 的所有属性并添加稳态特定参数。
    
    属性:
        max_iter: 最大迭代步数
        cfl_init: 初始 CFL 数（推荐：复杂网格为 0.05-0.1）。**刻意低于 CLI
            `--cfl-start` 的默认 0.1**：79 万单元 cube_demo 上真 P1
            （AFCFD_FILTER_MODE=off、零阶数损失）实测稳定的 CFL 在 0.03
            量级，0.05 更靠近可用区间。不要为了"两边一致"把它调高。
        cfl_max: 最大 CFL 数。**2026-09-15 从 10.0 改为 0.5**：10.0 是
            CLI `--cfl-max` 默认值（0.5）的 20 倍，也是 SSP-RK3 线性稳定
            极限（~1.0）的 10 倍——自适应控制器会真的往那个上限爬（软上限
            只在失败之后才收紧，见 core/time_integration/adaptive_cfl.py
            模块文档第 8 条），于是 YAML 驱动的算例会反复穿越稳定边界。
            同一个物理量在两个配置面上差 20 倍本身就是缺陷。
        cfl_min: 自适应 CFL 下限（2026-09-15 新增，与 CLI `--cfl-min` 对应）。
            此前配置层完全没有这个字段，而控制器默认 0.05 **高于**上面提到
            的实测稳定值 0.03——也就是说通过 YAML 根本到不了那个已验证可用
            的工作点（`cfl_init` 会被下限钳上去，见 adaptive_cfl.py 第 11 条）。

            默认取 **0.01** 而不是跟随控制器的 0.05，有两个理由：(a) 0.05
            会恰好等于 `cfl_init` 的默认 0.05，那样控制器**一步也收缩不了**，
            YAML 驱动的算例就完全失去了向下的保护；(b) 0.05 挡住 0.03 这个
            唯一在真实网格上被 250 步验证过稳定的值。0.01 远低于任何实测值，
            只留作"再低就说明问题本身不对"的兜底。
        convergence_tol: 收敛容差（残差）
        monitor_coefficients: 在迭代期间监控气动系数
        growth_rate: 边界层几何增长率（表面 -> 体网格）
        bl_layers: 可选覆盖项，用于定义在切换到（固定增长率）过渡阶段之前，
            计为精细边界层阶段的层数（参见 mesh_extrusion.extrude_layers 的
            bl_layers 文档）。None（默认）使用 8。过渡阶段本身没有层数上限 - 
            它以固定速率增长，直到达到 max_cell_size。
        min_cell_size: 第一层（近壁）厚度，单位米
        target_cells: 目标总单元数（目前仅由纯挤出体网格路径 consulted；
            基于 tetgen 的混合路径忽略它）
        max_cell_size: 核心区域单元尺寸的可选硬上限（米），
            从边界层的近壁尺寸向外渐变，而不是统一应用。
            None 使核心填充的单元尺寸无界（仅应用 tetgen 自身的形状质量边界，
            因此单元可以 grow 到与粗远场输入面一样大，例如
            稀疏三角化的隧道/入口/出口壁所允许的）。
        rho_inf: 自由流密度 (kg/m^3) - 初始条件、入口/远场边界条件和
            Cd/Cl 归一化的单一真实来源，确保三者始终保持一致。
        vel_inf: 自由流速度大小 (m/s)，与 rho_inf 作用相同。
        p_inf: 自由流静压 (Pa)，与 rho_inf 作用相同。
        use_wall_functions: 在 WALL/GROUND 边界面上启用 Menter 可扩展/自动壁面
            处理（基于对数律），而不是解析到壁面。False（默认）完全保留之前的行为 - 
            即解析梯度壁面剪切力/k/omega 处理，这需要第一个单元的 y+~1 才能准确。
            True 允许较粗的近壁网格（y+ 高达 ~100+）仍能给出具有物理意义的
            皮肤摩擦力和近壁湍流，代价是对数律模型自身的平衡边界层假设在强分离流中
            不如解析梯度准确。默认为关闭，因为这是新的、尚未在实际中广泛使用的物理模型 - 
            请显式选择加入，而不是静默更改现有精细网格案例的结果。

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
    cfl_init: float = 0.03  # 2026-09-17：原为 0.05，与 CLI --cfl-start 对齐
    cfl_max: float = 0.06   # 2026-09-17：原为 0.5，见本类文档 cfl_max 一节
    cfl_min: float = 0.01   # 2026-09-15 新增，见本类文档 cfl_min 一节
    convergence_tol: float = 1e-3
    monitor_coefficients: bool = True
    growth_rate: float = 1.15
    bl_layers: Optional[int] = None
    min_cell_size: float = 0.003
    target_cells: int = 500000
    max_cell_size: Optional[float] = None
    rho_inf: float = 1.225
    vel_inf: float = 33.33
    p_inf: float = 101325.0
    use_wall_functions: bool = False

    def __post_init__(self):
        """验证稳态配置。"""
        super().__post_init__()

        # 验证迭代次数
        if self.max_iter < 1:
            raise ValueError(f"最大迭代次数必须为正数，得到 {self.max_iter}")

        # 验证 CFL 数
        if self.cfl_init <= 0:
            raise ValueError(f"初始 CFL 必须为正数，得到 {self.cfl_init}")
        if self.cfl_max <= 0:
            raise ValueError(f"最大 CFL 必须为正数，得到 {self.cfl_max}")
        if self.cfl_init > self.cfl_max:
            raise ValueError(f"初始 CFL ({self.cfl_init}) 不能超过最大 CFL ({self.cfl_max})")
        # 三者的序关系必须自洽（2026-09-15）：控制器里 cfl_min > cfl_max
        # 会让"收缩"分支把 CFL 调高并突破 cfl_max（adaptive_cfl.py 第 11 条
        # 记录的真实缺陷），配置层应当在更早的地方就拦下这种矛盾配置。
        if self.cfl_min <= 0:
            raise ValueError(f"CFL 下限必须为正数，得到 {self.cfl_min}")
        if self.cfl_min > self.cfl_max:
            raise ValueError(
                f"CFL 下限 ({self.cfl_min}) 不能超过最大 CFL ({self.cfl_max})")
        if self.cfl_min > self.cfl_init:
            raise ValueError(
                f"CFL 下限 ({self.cfl_min}) 不能超过初始 CFL "
                f"({self.cfl_init})——否则控制器会把初始值钳上去，"
                f"实际跑的不是你要的那个 CFL")

        # 验证收敛容差
        if self.convergence_tol <= 0:
            raise ValueError(f"收敛容差必须为正数，得到 {self.convergence_tol}")

        # 验证体网格参数
        if self.growth_rate <= 1.0:
            raise ValueError(f"growth_rate 必须 > 1.0，得到 {self.growth_rate}")
        if self.min_cell_size <= 0:
            raise ValueError(f"min_cell_size 必须为正数，得到 {self.min_cell_size}")
        if self.target_cells < 1:
            raise ValueError(f"target_cells 必须为正数，得到 {self.target_cells}")
        if self.max_cell_size is not None:
            if self.max_cell_size <= 0:
                raise ValueError(f"max_cell_size 必须为正数，得到 {self.max_cell_size}")
            if self.max_cell_size < self.min_cell_size:
                raise ValueError(
                    f"max_cell_size ({self.max_cell_size}) 不能小于 "
                    f"min_cell_size ({self.min_cell_size})"
                )

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
        sample_interval: 数据采样间隔（步数）
        warmup_time: 跳过的预热时间（秒，用于统计）
        init_from_checkpoint: 从稳态检查点初始化
        growth_rate, bl_layers, min_cell_size, target_cells,
            max_cell_size: 体网格生成参数，含义与 SteadyConfig 相同。
        rho_inf, vel_inf, p_inf: 自由流条件，含义和作用与 SteadyConfig 相同
            （初始条件、边界条件和 Cd/Cl 归一化的单一真实来源）。
        use_wall_functions: 在 WALL/GROUND 面上启用 Menter 可扩展/自动壁面处理，
            含义与 SteadyConfig 相同。

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
    # 自适应 CFL 三元组（2026-09-17 新增）。为什么瞬态也需要：
    # `--time-method rk3/imex` 下 `step()` 忽略 dt、按**逐单元局部 CFL
    # 步长**推进（见 core/fr_solver/step.py 的 dt 语义一节），那条路径上
    # 自适应 CFL 控制器是**激活**的；而此前 TransientConfig 没有这三个
    # 字段、`solve transient` 也没有对应 CLI 选项，于是瞬态运行只能吃
    # FRSolver 的构造默认值，配置不出本项目实测稳定的 ~0.03。
    # （`dual-time` 档不构造这个控制器，内层伪时间有自己的逻辑，这三个
    # 字段对它无效——语义与 `solve steady` 完全一致。）
    cfl_init: float = 0.03   # 2026-09-17：与 CLI --cfl-start 对齐
    cfl_max: float = 0.06    # 2026-09-17：原为 0.5，见 SteadyConfig.cfl_max 一节
    cfl_min: float = 0.01
    sample_interval: int = 10
    warmup_time: float = 0.05
    init_from_checkpoint: Optional[str] = None
    growth_rate: float = 1.15
    bl_layers: Optional[int] = None
    min_cell_size: float = 0.003
    target_cells: int = 500000
    max_cell_size: Optional[float] = None
    rho_inf: float = 1.225
    vel_inf: float = 33.33
    p_inf: float = 101325.0
    use_wall_functions: bool = False

    def __post_init__(self):
        """验证瞬态配置。"""
        super().__post_init__()

        # 验证时间步长
        if self.dt <= 0:
            raise ValueError(f"时间步长必须为正数，得到 {self.dt}")

        # 验证总时间
        if self.total_time <= 0:
            raise ValueError(f"总时间必须为正数，得到 {self.total_time}")

        # 验证预热时间
        if self.warmup_time < 0:
            raise ValueError(f"预热时间必须为非负数，得到 {self.warmup_time}")
        if self.warmup_time >= self.total_time:
            raise ValueError(f"预热时间 ({self.warmup_time}) 不能超过总时间 ({self.total_time})")

        # 计算总步数
        self.total_steps = int(self.total_time / self.dt)

        # 验证体网格参数
        if self.growth_rate <= 1.0:
            raise ValueError(f"growth_rate 必须 > 1.0，得到 {self.growth_rate}")
        if self.min_cell_size <= 0:
            raise ValueError(f"min_cell_size 必须为正数，得到 {self.min_cell_size}")
        if self.target_cells < 1:
            raise ValueError(f"target_cells 必须为正数，得到 {self.target_cells}")
        if self.max_cell_size is not None:
            if self.max_cell_size <= 0:
                raise ValueError(f"max_cell_size 必须为正数，得到 {self.max_cell_size}")
            if self.max_cell_size < self.min_cell_size:
                raise ValueError(
                    f"max_cell_size ({self.max_cell_size}) 不能小于 "
                    f"min_cell_size ({self.min_cell_size})"
                )

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
