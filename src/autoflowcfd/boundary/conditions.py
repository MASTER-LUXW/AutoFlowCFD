"""边界条件实现。

本模块提供 AutoFlowCFD 内置的边界条件类，包括入口、出口、壁面、地面、
远场、对称面和车身边界。

核心组件:
    - BaseBC: 所有边界条件的抽象基类
    - InletBC: 速度/压力入口边界
    - OutletBC: 压力出口边界
    - WallBC: 无滑移壁面边界
    - GroundBC: 移动/静止地面边界
    - FarfieldBC: 自由来流远场边界
    - SymmetryBC: 对称面边界
    - BodyBC: 车身表面（特殊壁面）

⚠️ 现状说明：这些类现在只承担参数校验和元数据登记的角色（构造 +
`validate()`），由 `boundary/manager.py` 的 `BoundaryManager.add_bc()`/
`auto_configure()`/`configure_from_yaml()`/`hybrid_configure()` 实际
使用——这些方法确实是 `solver_steady.py`/`transient_solver_loop.py`
求解主流程调用的（登记边界元数据供查询/导出用）。以前每个类还有一个
`apply()` 方法，会直接往 solution 数组里写边界值——但**从未被生产
求解路径调用过**：实际求解用的是 `core/bc_handler.py` 里完全独立、
向量化实现的边界处理逻辑，那套逻辑读的是这里通过 `add_bc()` 存进
`BoundaryMap` 的名字/类型信息，不会调用这些类的 `apply()`。已经
确认这条 `apply()` 路径是死代码（唯一调用方是同样已删除的
`BoundaryManager.apply_boundary()`/`apply_all()`），且 `SymmetryBC`/
`BodyBC.apply()` 是完全没实现的空 TODO——如果这条路径被误用会静默给出
错误结果，所以整体删除了 `apply()`，只保留元数据登记功能。

Example:
    >>> from autoflowcfd.boundary.conditions import InletBC
    >>> inlet = InletBC(velocity=30.0, pressure=101325.0)
    >>> inlet.validate()
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger


class BaseBC(ABC):
    """所有边界条件的抽象基类。

    所有边界条件实现都必须继承此类并实现要求的方法。

    Attributes:
        bc_type: 边界条件类型标识
        params: 边界条件参数

    Example:
        >>> class MyBC(BaseBC):
        ...     def __init__(self, **kwargs):
        ...         super().__init__("MY_BC", kwargs)
    """

    def __init__(self, bc_type: str, params: Dict[str, Any]):
        """初始化边界条件。

        Args:
            bc_type: 边界条件类型标识
            params: 边界条件参数
        """
        self.bc_type = bc_type
        self.params = params

    @abstractmethod
    def validate(self) -> bool:
        """校验边界条件参数。

        Returns:
            bool: 参数有效则为 True

        Raises:
            ValueError: 参数无效时
        """
        pass

    def get_type(self) -> str:
        """获取边界条件类型。

        Returns:
            str: 边界条件类型标识
        """
        return self.bc_type

    def __repr__(self) -> str:
        """字符串表示。"""
        return f"{self.__class__.__name__}(type={self.bc_type})"


class InletBC(BaseBC):
    """速度/压力入口边界条件。

    指定入口的速度分量和静压。同时支持均匀入口和基于剖面的入口条件
    （元数据层面；实际求解见模块文档字符串）。

    Attributes:
        velocity_x: 入口速度 X 分量 (m/s)
        velocity_y: 入口速度 Y 分量 (m/s)
        velocity_z: 入口速度 Z 分量 (m/s)
        pressure: 入口静压 (Pa)
        turbulence_k: 湍动能 (m²/s²)
        turbulence_omega: 比耗散率 (1/s)

    Example:
        >>> inlet = InletBC(
        ...     velocity_x=30.0,
        ...     velocity_y=0.0,
        ...     velocity_z=0.0,
        ...     pressure=101325.0
        ... )
    """

    def __init__(
        self,
        velocity_x: float = 30.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        pressure: float = 101325.0,
        turbulence_k: float = 0.1,
        turbulence_omega: float = 10.0,
        **kwargs
    ):
        """初始化入口边界条件。

        Args:
            velocity_x: 速度 X 分量 (m/s)
            velocity_y: 速度 Y 分量 (m/s)
            velocity_z: 速度 Z 分量 (m/s)
            pressure: 静压 (Pa)
            turbulence_k: 湍动能 (m²/s²)
            turbulence_omega: 比耗散率 (1/s)
            **kwargs: 额外参数
        """
        params = {
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
            'pressure': pressure,
            'turbulence_k': turbulence_k,
            'turbulence_omega': turbulence_omega,
        }
        params.update(kwargs)
        super().__init__('INLET', params)

    def validate(self) -> bool:
        """校验入口边界条件参数。

        Returns:
            bool: 全部参数有效则为 True

        Raises:
            ValueError: 任一参数无效时
        """
        # 校验速度大小
        vel_mag = np.sqrt(
            self.params['velocity_x']**2 +
            self.params['velocity_y']**2 +
            self.params['velocity_z']**2
        )

        if vel_mag < 0:
            raise ValueError("Velocity magnitude cannot be negative")

        if vel_mag > 340.0:  # 声速近似值
            logger.warning(
                f"Inlet velocity {vel_mag:.2f} m/s is supersonic. "
                f"Ensure compressible flow solver is enabled."
            )

        # 校验压力
        if self.params['pressure'] <= 0:
            raise ValueError(f"Pressure must be positive, got {self.params['pressure']}")

        # 校验湍流量
        if self.params['turbulence_k'] < 0:
            raise ValueError(f"Turbulence k must be non-negative, got {self.params['turbulence_k']}")

        if self.params['turbulence_omega'] <= 0:
            raise ValueError(f"Turbulence omega must be positive, got {self.params['turbulence_omega']}")

        return True


class OutletBC(BaseBC):
    """压力出口边界条件。

    在出口边界指定静压，流动方向由局部解的梯度决定。

    Attributes:
        pressure: 出口静压 (Pa)
        backflow_turbulence_k: 回流时使用的湍动能 (m²/s²)
        backflow_turbulence_omega: 回流时使用的比耗散率 (1/s)

    Example:
        >>> outlet = OutletBC(pressure=101325.0)
    """

    def __init__(
        self,
        pressure: float = 101325.0,
        backflow_turbulence_k: float = 0.1,
        backflow_turbulence_omega: float = 10.0,
        **kwargs
    ):
        """初始化出口边界条件。

        Args:
            pressure: 静压 (Pa)
            backflow_turbulence_k: 回流湍动能
            backflow_turbulence_omega: 回流比耗散率
            **kwargs: 额外参数
        """
        params = {
            'pressure': pressure,
            'backflow_turbulence_k': backflow_turbulence_k,
            'backflow_turbulence_omega': backflow_turbulence_omega,
        }
        params.update(kwargs)
        super().__init__('OUTLET', params)

    def validate(self) -> bool:
        """校验出口边界条件参数。

        Returns:
            bool: 参数有效则为 True

        Raises:
            ValueError: 参数无效时
        """
        if self.params['pressure'] <= 0:
            raise ValueError(f"Pressure must be positive, got {self.params['pressure']}")

        if self.params['backflow_turbulence_k'] < 0:
            raise ValueError(f"Backflow turbulence k must be non-negative")

        if self.params['backflow_turbulence_omega'] <= 0:
            raise ValueError(f"Backflow turbulence omega must be positive")

        return True


class WallBC(BaseBC):
    """无滑移壁面边界条件。

    在固壁上施加无滑移条件 (u=v=w=0)，支持湍流壁面函数（元数据层面；
    实际的壁面函数求解见 core/turbulence_wmles.py 的 WMLESModel）。

    Attributes:
        wall_function: 壁面函数类型（'standard'、'enhanced'、'none'）
        roughness_height: 表面粗糙度高度 (m)
        temperature: 壁面温度 (K)——用于传热

    Example:
        >>> wall = WallBC(wall_function='standard')
    """

    def __init__(
        self,
        wall_function: str = 'standard',
        roughness_height: float = 0.0,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """初始化壁面边界条件。

        Args:
            wall_function: 壁面函数类型（'standard'、'enhanced'、'none'）
            roughness_height: 表面粗糙度高度 (m)
            temperature: 壁面温度 (K)，None 表示绝热
            **kwargs: 额外参数
        """
        if wall_function not in ['standard', 'enhanced', 'none']:
            raise ValueError(
                f"Invalid wall function: {wall_function}. "
                f"Must be 'standard', 'enhanced', or 'none'"
            )

        params = {
            'wall_function': wall_function,
            'roughness_height': roughness_height,
            'temperature': temperature,
        }
        params.update(kwargs)
        super().__init__('WALL', params)

    def validate(self) -> bool:
        """校验壁面边界条件参数。

        Returns:
            bool: 参数有效则为 True

        Raises:
            ValueError: 参数无效时
        """
        if self.params['roughness_height'] < 0:
            raise ValueError(f"Roughness height must be non-negative")

        if self.params['temperature'] is not None and self.params['temperature'] <= 0:
            raise ValueError(f"Temperature must be positive if specified")

        return True



# GroundBC 已拆分到 conditions_advanced.py，重新导出以保持向后兼容
from .conditions_advanced import FarfieldBC, SymmetryBC, BodyBC, GroundBC  # noqa: F401


# 自定义边界条件注册表
_bc_registry: Dict[str, type] = {}


def register_boundary_condition(bc_type: str):
    """注册自定义边界条件类的装饰器。

    Args:
        bc_type: 边界条件类型标识

    Example:
        >>> @register_boundary_condition("CUSTOM_INLET")
        ... class CustomInletBC(BaseBC):
        ...     def validate(self):
        ...         return True
    """
    def decorator(cls: type) -> type:
        if not issubclass(cls, BaseBC):
            raise TypeError(f"{cls.__name__} must inherit from BaseBC")

        _bc_registry[bc_type] = cls
        logger.info(f"Registered custom boundary condition: {bc_type}")
        return cls

    return decorator


def get_boundary_condition_class(bc_type: str) -> type:
    """按类型标识获取边界条件类。

    Args:
        bc_type: 边界条件类型标识

    Returns:
        type: 边界条件类

    Raises:
        KeyError: 边界条件类型未注册时
    """
    # 内置边界条件
    builtin_bcs = {
        'INLET': InletBC,
        'OUTLET': OutletBC,
        'OUTLET_CHARACTERISTIC': None,  # 惰性导入
        'OUTLET_SPONGE': None,  # 惰性导入
        'WALL': WallBC,
        'GROUND': GroundBC,
        'FARFIELD': FarfieldBC,
        'SYMMETRY': SymmetryBC,
        'BODY': BodyBC,
    }

    if bc_type in builtin_bcs:
        if builtin_bcs[bc_type] is None:
            # 高级出口边界条件惰性导入，避免循环依赖
            if bc_type == 'OUTLET_CHARACTERISTIC':
                from .outlet_bc import OutletCharacteristicBC
                return OutletCharacteristicBC
            elif bc_type == 'OUTLET_SPONGE':
                from .outlet_bc import OutletSpongeBC
                return OutletSpongeBC
        return builtin_bcs[bc_type]

    if bc_type in _bc_registry:
        return _bc_registry[bc_type]

    raise KeyError(
        f"Unknown boundary condition type: {bc_type}. "
        f"Available types: {list(builtin_bcs.keys()) + list(_bc_registry.keys())}"
    )


def create_boundary_condition(bc_type: str, **kwargs) -> BaseBC:
    """创建边界条件实例的工厂函数。

    Args:
        bc_type: 边界条件类型标识
        **kwargs: 边界条件参数

    Returns:
        BaseBC: 边界条件实例

    Example:
        >>> bc = create_boundary_condition('INLET', velocity_x=30.0)
    """
    bc_class = get_boundary_condition_class(bc_type)
    return bc_class(**kwargs)
