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
    """湍流模型枚举。"""
    NONE = "none"       # 层流 Navier-Stokes（无湍流模型）
    SST_KW = "sst_kw"
    SA = "sa"
    DES = "des"
    DDES = "ddes"
    LES = "les"


class TimeIntegrationScheme(str, Enum):
    """时间积分方案枚举。"""
    BACKWARD_EULER = "backward_euler"
    RK2 = "rk2"
    RK3 = "rk3"
    AB3 = "ab3"  # Adams-Bashforth 三阶


@dataclass
class SolverConfig:
    """基础求解器配置。
    
    属性:
        backend: 计算后端 (cpu/gpu/auto)
        order: FR 离散阶数 (1/2/3)
        turbulence: 湍流模型
        gpu_device: GPU 设备 ID（仅 GPU 模式）
        n_threads: CPU 线程数（仅 CPU 模式，auto=检测）
        output_dir: 输出目录路径
        checkpoint_interval: 检查点保存间隔（步数）
        verbose: 启用详细日志记录
        
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
        
        # 如果输出目录不存在则创建
        os.makedirs(self.output_dir, exist_ok=True)
    
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
        cfl_init: 初始 CFL 数（推荐：复杂网格为 0.05-0.1）
        cfl_max: 最大 CFL 数
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
        ...     cfl_init=0.1,
        ...     cfl_max=5.0
        ... )
    """
    max_iter: int = 50
    cfl_init: float = 0.05  # 复杂网格的保守默认值（原为 1.0）
    cfl_max: float = 10.0
    convergence_tol: float = 1e-3
    monitor_coefficients: bool = True
    growth_rate: float = 1.15
    bl_layers: Optional[int] = None
    min_cell_size: float = 0.003
    target_cells: int = 500000
    max_cell_size: Optional[float] = None
    rho_inf: float = 1.225
    vel_inf: float = 30.0
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
        ...     time_scheme="backward_euler"
        ... )
    """
    dt: float = 1e-4
    total_time: float = 0.1
    time_scheme: TimeIntegrationScheme = TimeIntegrationScheme.BACKWARD_EULER
    sample_interval: int = 10
    warmup_time: float = 0.05
    init_from_checkpoint: Optional[str] = None
    growth_rate: float = 1.15
    bl_layers: Optional[int] = None
    min_cell_size: float = 0.003
    target_cells: int = 500000
    max_cell_size: Optional[float] = None
    rho_inf: float = 1.225
    vel_inf: float = 30.0
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
