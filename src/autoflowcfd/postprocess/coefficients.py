"""气动系数数据类定义模块。

本模块定义气动系数/气动力的数据容器（AerodynamicCoefficients /
AerodynamicForces），供真正的积分实现 postprocess/fr_coefficients.py
（FR 原生面通量点压力积分，含力矩）复用。

历史说明：本文件原有的 CoefficientCalculator（V1 单元中心压力积分）已在
第三轮整体专家组评审整改中移除——它依赖从未存在的 `GridData.get_face_data()`
且用全场 ρRT 均值压力伪积分，气动系数恒为 0；所有生产入口（CLI `post
coefficients`、API calculate_coefficients）都已改走 FR 原生路径，无调用方。

示例:
    >>> from autoflowcfd.postprocess import AerodynamicCoefficients
    >>> coeffs = AerodynamicCoefficients(Cd=0.28, Cl=0.05)
    >>> print(f"Cd = {coeffs.Cd:.4f}")
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class AerodynamicCoefficients:
    """气动系数数据类。

    Attributes:
        Cd: 阻力系数
        Cl: 升力系数
        Cm: 俯仰力矩系数
        Cs: 侧向力系数
        Cy: 偏航力矩系数
        Cr: 滚转力矩系数
    """
    Cd: float = 0.0
    Cl: float = 0.0
    Cm: float = 0.0
    Cs: float = 0.0
    Cy: float = 0.0
    Cr: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转换成字典。"""
        return {
            'Cd': self.Cd,
            'Cl': self.Cl,
            'Cm': self.Cm,
            'Cs': self.Cs,
            'Cy': self.Cy,
            'Cr': self.Cr
        }

    def __str__(self) -> str:
        """字符串表示。"""
        return (
            f"Aerodynamic Coefficients:\n"
            f"  Cd (Drag):              {self.Cd:.6f}\n"
            f"  Cl (Lift):              {self.Cl:.6f}\n"
            f"  Cm (Pitch Moment):      {self.Cm:.6f}\n"
            f"  Cs (Side Force):        {self.Cs:.6f}\n"
            f"  Cy (Yaw Moment):        {self.Cy:.6f}\n"
            f"  Cr (Roll Moment):       {self.Cr:.6f}"
        )


@dataclass
class AerodynamicForces:
    """气动力与力矩（绝对值）。

    Attributes:
        drag_force: 阻力 (N)
        lift_force: 升力 (N)
        side_force: 侧向力 (N)
        pitch_moment: 俯仰力矩 (N·m)
        yaw_moment: 偏航力矩 (N·m)
        roll_moment: 滚转力矩 (N·m)
    """
    drag_force: float = 0.0
    lift_force: float = 0.0
    side_force: float = 0.0
    pitch_moment: float = 0.0
    yaw_moment: float = 0.0
    roll_moment: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转换成字典。"""
        return {
            'drag_force': self.drag_force,
            'lift_force': self.lift_force,
            'side_force': self.side_force,
            'pitch_moment': self.pitch_moment,
            'yaw_moment': self.yaw_moment,
            'roll_moment': self.roll_moment
        }

    def __str__(self) -> str:
        """字符串表示。"""
        return (
            f"Aerodynamic Forces:\n"
            f"  Drag Force:             {self.drag_force:.2f} N\n"
            f"  Lift Force:             {self.lift_force:.2f} N\n"
            f"  Side Force:             {self.side_force:.2f} N\n"
            f"  Pitch Moment:           {self.pitch_moment:.2f} N·m\n"
            f"  Yaw Moment:             {self.yaw_moment:.2f} N·m\n"
            f"  Roll Moment:            {self.roll_moment:.2f} N·m"
        )
