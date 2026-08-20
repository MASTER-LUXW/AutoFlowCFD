"""边界条件管理模块——V1（FVM）时代遗留，当前 V2 FR 求解路径不使用。

**现状说明（V2.0 专家组评审核实）**：本模块（BoundaryManager/
conditions.py/manager.py/config.py/outlet_bc.py/manager_configure.py/
config_validators.py/conditions_advanced.py，合计约 2560 行）此前的
文档字符串声称"实际的边界值计算由 core/bc_handler.py 独立完成"——
`core/bc_handler.py` 在 V2 重构中已被删除，这个分层描述早已失实。
本模块在真实的 FR 求解路径上**零调用点**：无粘/粘性残差的边界条件
统一由 `boundary/fr_ghost_state.py`（弱形式幽灵态，BD-01）+
`boundary/synthetic_inlet.py`（SEM 合成湍流入口，BD-02）提供，两者都
真正接入 `core/fr_solver/boundary.py`——与本文件描述的
InletBC/OutletBC/WallBC 等类完全无关。

本模块目前只保留给测试（tests/unit/test_boundary*.py）覆盖，未在本轮
删除——删除它需要同时处理这些测试，是比"标注状态"更大的独立改动，
留给后续专门的死代码清理会话。二次开发/新代码不应该以本模块为起点，
应参考 `boundary/fr_ghost_state.py`。

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
