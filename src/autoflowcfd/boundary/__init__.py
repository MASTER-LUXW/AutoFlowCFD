"""边界条件管理模块。

本模块负责边界条件的元数据登记，包括内置类型（velocity_inlet、
pressure_outlet、wall、symmetry、slip_wall）以及自定义 BC 插件。实际
的边界值计算由 `core/bc_handler.py` 独立完成——见
`conditions.py`/`manager.py` 模块文档字符串对这个分层的说明。

Key Components:
    - BoundaryManager: 支持 auto/manual/hybrid 模式的 BC 登记管理器
    - YAMLConfigLoader: YAML 配置文件加载器
    - BoundaryTypeMapper: 自动边界类型映射器
    - 内置 BC: InletBC、OutletBC、WallBC、GroundBC、FarfieldBC、SymmetryBC、BodyBC
    - 通过 register_boundary_condition 装饰器提供的自定义 BC 扩展接口

Example:
    >>> from autoflowcfd.boundary import BoundaryManager
    >>> from autoflowcfd.grid.structures import BoundaryMap
    >>> import numpy as np
    >>>
    >>> # 创建边界映射
    >>> bmap = BoundaryMap(
    ...     groups={"INLET": np.array([0, 1, 2], dtype=np.int32)},
    ...     bc_types={"INLET": "WALL"},
    ... )
    >>>
    >>> # 创建管理器并登记边界条件
    >>> bc_manager = BoundaryManager(bmap)
    >>> bc_manager.auto_configure()  # Auto 模式
    >>> # 或者
    >>> bc_manager.configure_from_yaml("config.yaml")  # Manual 模式
    >>> # 或者
    >>> bc_manager.hybrid_configure("config.yaml")  # Hybrid 模式
"""

from .conditions import (
    BaseBC,
    InletBC,
    OutletBC,
    WallBC,
    GroundBC,
    FarfieldBC,
    SymmetryBC,
    BodyBC,
    register_boundary_condition,
    get_boundary_condition_class,
    create_boundary_condition,
)
from .outlet_bc import OutletCharacteristicBC, OutletSpongeBC
from .manager import BoundaryManager
from .config import (
    YAMLConfigLoader,
    BoundaryTypeMapper,
    ParameterValidator,
    ConfigurationError,
)

__all__ = [
    # Manager
    "BoundaryManager",
    
    # Configuration
    "YAMLConfigLoader",
    "BoundaryTypeMapper",
    "ParameterValidator",
    "ConfigurationError",
    
    # Base class
    "BaseBC",
    
    # Built-in BCs
    "InletBC",
    "OutletBC",
    "WallBC",
    "GroundBC",
    "FarfieldBC",
    "SymmetryBC",
    "BodyBC",
    
    # Advanced outlet BCs
    "OutletCharacteristicBC",
    "OutletSpongeBC",
    
    # Extension mechanism
    "register_boundary_condition",
    "get_boundary_condition_class",
    "create_boundary_condition",
]
