"""高级边界条件实现。

从 conditions.py 拆出，控制单文件行数。包含远场、对称面和车身边界条件。

核心组件:
    - GroundBC: 移动/静止地面
    - FarfieldBC: 自由来流远场边界
    - SymmetryBC: 对称面边界
    - BodyBC: 车身表面（特殊壁面）
"""

from typing import Optional
from loguru import logger

from .conditions import BaseBC, WallBC


class GroundBC(BaseBC):
    """地面边界条件。

    地面平面的特殊壁面边界，支持移动地面仿真（滚动路面）和静止地面。

    Attributes:
        moving: 地面是否移动（滚动路面）
        velocity_x: 地面速度 X 分量 (m/s)
        velocity_y: 地面速度 Y 分量 (m/s)
        velocity_z: 地面速度 Z 分量 (m/s)

    Example:
        >>> # 静止地面
        >>> ground = GroundBC(moving=False)
        >>>
        >>> # 移动地面（滚动路面，30 m/s）
        >>> ground = GroundBC(moving=True, velocity_x=30.0)
    """

    def __init__(
        self,
        moving: bool = False,
        velocity_x: float = 0.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        **kwargs
    ):
        """初始化地面边界条件。

        Args:
            moving: 地面是否移动
            velocity_x: 地面速度 X 分量 (m/s)
            velocity_y: 地面速度 Y 分量 (m/s)
            velocity_z: 地面速度 Z 分量 (m/s)
            **kwargs: 额外参数
        """
        params = {
            'moving': moving,
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
        }
        params.update(kwargs)
        super().__init__('GROUND', params)

    def validate(self) -> bool:
        """校验地面边界条件参数。

        Returns:
            bool: 参数有效则为 True

        Raises:
            ValueError: 参数无效时
        """
        if not self.params['moving']:
            # 静止地面的速度应为零
            if abs(self.params['velocity_x']) > 1e-6:
                logger.warning(
                    "Stationary ground has non-zero X velocity. "
                    "Setting moving=True or velocity_x=0."
                )

        return True


class FarfieldBC(BaseBC):
    """远场边界条件。

    在远场边界施加自由来流条件，采用基于特征的无反射边界条件（元数据
    层面；实际求解见 core/bc_handler.py 的 _farfield_bc_vectorized）。

    Attributes:
        velocity_x: 自由来流速度 X 分量 (m/s)
        velocity_y: 自由来流速度 Y 分量 (m/s)
        velocity_z: 自由来流速度 Z 分量 (m/s)
        pressure: 自由来流压力 (Pa)
        temperature: 自由来流温度 (K)

    Example:
        >>> farfield = FarfieldBC(
        ...     velocity_x=30.0,
        ...     pressure=101325.0,
        ...     temperature=288.15
        ... )
    """

    def __init__(
        self,
        velocity_x: float = 30.0,
        velocity_y: float = 0.0,
        velocity_z: float = 0.0,
        pressure: float = 101325.0,
        temperature: float = 288.15,
        **kwargs
    ):
        """初始化远场边界条件。

        Args:
            velocity_x: 自由来流速度 X 分量 (m/s)
            velocity_y: 自由来流速度 Y 分量 (m/s)
            velocity_z: 自由来流速度 Z 分量 (m/s)
            pressure: 自由来流压力 (Pa)
            temperature: 自由来流温度 (K)
            **kwargs: 额外参数
        """
        params = {
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'velocity_z': velocity_z,
            'pressure': pressure,
            'temperature': temperature,
        }
        params.update(kwargs)
        super().__init__('FARFIELD', params)

    def validate(self) -> bool:
        """校验远场边界条件参数。

        Returns:
            bool: 参数有效则为 True

        Raises:
            ValueError: 参数无效时
        """
        if self.params['pressure'] <= 0:
            raise ValueError(f"Pressure must be positive")

        if self.params['temperature'] <= 0:
            raise ValueError(f"Temperature must be positive")

        return True


class SymmetryBC(BaseBC):
    """对称面边界条件。

    施加对称条件：法向速度和所有变量的法向梯度均为零（元数据层面；
    实际求解见 core/bc_handler.py 的 _symmetry_bc_vectorized）。

    Example:
        >>> symmetry = SymmetryBC()
    """

    def __init__(self, **kwargs):
        """初始化对称面边界条件。

        Args:
            **kwargs: 额外参数（目前无）
        """
        super().__init__('SYMMETRY', kwargs)

    def validate(self) -> bool:
        """校验对称面边界条件参数。

        Returns:
            bool: 始终为 True（没有需要校验的参数）
        """
        return True


class BodyBC(WallBC):
    """车身表面边界条件。

    车身表面的特殊壁面边界。继承自 WallBC，气动表面可能需要特殊处理。

    Example:
        >>> body = BodyBC(wall_function='enhanced')
    """

    def __init__(
        self,
        wall_function: str = 'enhanced',
        roughness_height: float = 0.0,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """初始化车身边界条件。

        Args:
            wall_function: 壁面函数类型
            roughness_height: 表面粗糙度高度 (m)
            temperature: 壁面温度 (K)
            **kwargs: 额外参数
        """
        super().__init__(
            wall_function=wall_function,
            roughness_height=roughness_height,
            temperature=temperature,
            **kwargs
        )
        self.bc_type = 'BODY'
