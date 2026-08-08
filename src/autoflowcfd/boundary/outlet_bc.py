"""亚/超声速流动的基于特征的出口边界条件。

用特征理论实现无反射边界条件，减少域出口处的虚假波反射。

Key Features:
    - 自动判断亚声速/超声速
    - 亚声速出口的压力松弛
    - 超声速出口的零梯度外推
    - 熵一致的实现

References:
    - Thompson, K.W. "Time-dependent boundary conditions for hyperbolic systems", 1987
    - Poinsot & Lele, "Boundary conditions for direct simulations", 1992

⚠️ 现状说明：本文件的 `apply()` 方法（`OutletCharacteristicBC.apply`/
`OutletSpongeBC.apply` 及其私有帮助方法 `_apply_subsonic_outlet`）已
删除——与 conditions.py 里的其它 BC 类一样，这条路径从未被生产求解
路径调用过，唯一调用方是已删除的 `BoundaryManager.apply_boundary`/
`apply_all`。这两个类仍然保留（构造 + `validate()`，`OutletSpongeBC`
还保留独立、确实被测试直接调用的 `compute_damping_factor()`），只是
不再提供 `apply()`——详见 conditions.py 模块文档字符串里对这个决定的
完整说明。
"""

import numpy as np
from loguru import logger

from .conditions import BaseBC


class OutletCharacteristicBC(BaseBC):
    """基于特征的出口边界条件。

    用 Riemann 不变量理论决定哪些变量应该被指定、哪些应该从内部外推。

    亚声速出流 (Ma < 1)：
        - 一个特征进入计算域 -> 指定压力
        - 其余特征离开 -> 外推速度、密度

    超声速出流 (Ma >= 1)：
        - 全部特征离开 -> 不需要边界条件（纯外推）

    Attributes:
        pressure_ref: 亚声速出口的参考压力 (Pa)
        relaxation_factor: 压力松弛系数 (0-1)
        ma_threshold: 区分亚/超声速的马赫数阈值
    """

    def __init__(
        self,
        pressure_ref: float = 101325.0,
        relaxation_factor: float = 0.1,
        ma_threshold: float = 1.0,
        **kwargs
    ):
        """初始化特征出口边界条件。

        Args:
            pressure_ref: 出口目标静压 (Pa)
            relaxation_factor: 压力施加的力度 (0-1)
                              越低越稳定，越高收敛越快
            ma_threshold: 区分亚/超声速的马赫数阈值
            **kwargs: 传给 BaseBC 的额外参数
        """
        super().__init__('OUTLET_CHARACTERISTIC', kwargs)

        self.pressure_ref = pressure_ref
        self.relaxation_factor = relaxation_factor
        self.ma_threshold = ma_threshold

        logger.info(
            f"OutletCharacteristicBC initialized: p_ref={pressure_ref:.1f} Pa, "
            f"relaxation={relaxation_factor:.2f}"
        )

    def validate(self) -> bool:
        """校验边界条件参数。

        Returns:
            全部参数有效则为 True

        Raises:
            ValueError: 参数无效时
        """
        if not (0 < self.pressure_ref < 1e7):
            raise ValueError(f"Invalid pressure_ref: {self.pressure_ref}")

        if not (0 <= self.relaxation_factor <= 1):
            raise ValueError(f"Invalid relaxation_factor: {self.relaxation_factor}")

        if not (0.1 <= self.ma_threshold <= 2.0):
            raise ValueError(f"Invalid ma_threshold: {self.ma_threshold}")

        return True


class OutletSpongeBC(BaseBC):
    """海绵层（sponge layer）出口边界条件。

    在出口附近的缓冲区里加入人工阻尼，吸收出行波、防止反射。阻尼强度
    从海绵层起点的零逐渐增大到出口处的最大值。

    Attributes:
        damping_strength: 最大阻尼系数 (0-1)
        sponge_fraction: 用作海绵层的计算域长度占比
        coordinate_axis: 施加海绵阻尼的坐标轴 (0=x, 1=y, 2=z)
    """

    def __init__(
        self,
        damping_strength: float = 0.5,
        sponge_fraction: float = 0.1,
        coordinate_axis: int = 0,
        **kwargs
    ):
        """初始化海绵层出口边界条件。

        Args:
            damping_strength: 最大阻尼强度 (0=无阻尼，1=强阻尼)
            sponge_fraction: 用作海绵层的计算域占比
            coordinate_axis: 海绵方向对应的坐标轴
            **kwargs: 额外参数
        """
        super().__init__('OUTLET_SPONGE', kwargs)

        self.damping_strength = damping_strength
        self.sponge_fraction = sponge_fraction
        self.coordinate_axis = coordinate_axis

        logger.info(
            f"OutletSpongeBC initialized: strength={damping_strength:.2f}, "
            f"fraction={sponge_fraction:.2f}, axis={coordinate_axis}"
        )

    def validate(self) -> bool:
        """校验边界条件参数。

        Returns:
            全部参数有效则为 True

        Raises:
            ValueError: 参数无效时
        """
        if not (0 <= self.damping_strength <= 1):
            raise ValueError(f"Invalid damping_strength: {self.damping_strength}")

        if not (0 < self.sponge_fraction < 1):
            raise ValueError(f"Invalid sponge_fraction: {self.sponge_fraction}")

        if self.coordinate_axis not in [0, 1, 2]:
            raise ValueError(f"Invalid coordinate_axis: {self.coordinate_axis}")

        return True

    def compute_damping_factor(
        self,
        cell_center: np.ndarray,
        domain_min: float,
        domain_max: float
    ) -> float:
        """根据位置计算局部阻尼系数。

        阻尼从海绵层起点的 0 二次增长到出口处的 damping_strength。

        Args:
            cell_center: 单元中心坐标
            domain_min: 计算域最小坐标
            domain_max: 计算域最大坐标（出口位置）

        Returns:
            阻尼系数，范围 [0, damping_strength]
        """
        x = cell_center[self.coordinate_axis]

        # 海绵层从距出口这个比例处开始
        sponge_start = domain_min + (1.0 - self.sponge_fraction) * (domain_max - domain_min)

        if x <= sponge_start:
            # 海绵层之外：无阻尼
            return 0.0

        # 海绵层内：二次增长
        normalized_pos = (x - sponge_start) / (domain_max - sponge_start)
        damping = self.damping_strength * normalized_pos ** 2

        return min(damping, self.damping_strength)
