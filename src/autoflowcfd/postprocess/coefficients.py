"""气动系数计算模块。

本模块通过压力积分从 CFD 仿真结果计算气动系数（Cd、Cl、Cm 等）。

Key Components:
    - CoefficientCalculator: 气动系数主计算器
    - ForceDecomposition: 力与力矩分解工具

Example:
    >>> from autoflowcfd.postprocess import CoefficientCalculator
    >>> calc = CoefficientCalculator(grid_data, solution)
    >>> coeffs = calc.calculate()
    >>> print(f"Cd = {coeffs['Cd']:.4f}")
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger
from dataclasses import dataclass

from ..grid.structures import GridData
from ..core.backend.base import SolutionVector


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


class CoefficientCalculator:
    """气动系数计算器。

    通过对车身表面积分压力和粘性力来计算阻力、升力和力矩系数。

    Attributes:
        grid_data: 网格数据对象
        solution: 流场解向量
        reference_area: 参考面积 (m²)，默认取轿车迎风面积
        reference_length: 参考长度 (m)，默认取车长
        density: 空气密度 (kg/m³)，默认 1.225
        velocity: 自由来流速度 (m/s)，默认 30.0
        dynamic_pressure: 动压 q = 0.5 * rho * V²

    Example:
        >>> calc = CoefficientCalculator(grid_data, solution)
        >>> coeffs = calc.calculate()
        >>> print(f"Cd = {coeffs['Cd']:.4f}")
    """

    def __init__(
        self,
        grid_data: GridData,
        solution: SolutionVector,
        reference_area: float = 2.2,
        reference_length: float = 4.5,
        density: float = 1.225,
        velocity: float = 30.0
    ):
        """初始化系数计算器。

        Args:
            grid_data: 网格数据对象
            solution: 流场解向量
            reference_area: 参考面积（默认轿车迎风面积）
            reference_length: 参考长度（默认车长）
            density: 空气密度
            velocity: 自由来流速度

        Raises:
            ValueError: 参数无效（reference_area <= 0 或 velocity <= 0）
        """
        if reference_area <= 0:
            raise ValueError(f"Reference area must be positive, got {reference_area}")
        if velocity <= 0:
            raise ValueError(f"Velocity must be positive, got {velocity}")
        if reference_length <= 0:
            raise ValueError(f"Reference length must be positive, got {reference_length}")

        self.grid_data = grid_data
        self.solution = solution
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.density = density
        self.velocity = velocity
        self.dynamic_pressure = 0.5 * density * velocity ** 2

        logger.info(
            f"CoefficientCalculator initialized:\n"
            f"  Reference area:     {reference_area:.2f} m²\n"
            f"  Reference length:   {reference_length:.2f} m\n"
            f"  Density:            {density:.3f} kg/m³\n"
            f"  Velocity:           {velocity:.2f} m/s\n"
            f"  Dynamic pressure:   {self.dynamic_pressure:.2f} Pa"
        )

    def calculate(self) -> AerodynamicCoefficients:
        """计算气动系数。

        对全部车身表面积分压力和粘性力，算出无量纲气动系数。

        Returns:
            AerodynamicCoefficients: 无量纲系数

        Example:
            >>> coeffs = calc.calculate()
            >>> print(f"Cd = {coeffs['Cd']:.4f}")
            >>> print(f"Cl = {coeffs['Cl']:.4f}")
        """
        logger.info("Calculating aerodynamic coefficients...")

        # 计算力与力矩
        forces = self.calculate_forces()

        # 转换成无量纲系数
        coeffs = AerodynamicCoefficients(
            Cd=forces.drag_force / (self.dynamic_pressure * self.reference_area),
            Cl=forces.lift_force / (self.dynamic_pressure * self.reference_area),
            Cm=forces.pitch_moment / (self.dynamic_pressure * self.reference_area * self.reference_length),
            Cs=forces.side_force / (self.dynamic_pressure * self.reference_area),
            Cy=forces.yaw_moment / (self.dynamic_pressure * self.reference_area * self.reference_length),
            Cr=forces.roll_moment / (self.dynamic_pressure * self.reference_area * self.reference_length)
        )

        logger.success(f"Aerodynamic coefficients calculated:\n{coeffs}")
        return coeffs

    def calculate_forces(self) -> AerodynamicForces:
        """计算气动力（绝对值）。

        把实际的压力 + 摩擦表面积分委托给
        core.aero_coeffs.AeroCoefficientCalculator——与实际求解器在求解
        过程中报告 Cd/Cl 用的是同一套、经过充分验证的实现，而不是自己
        另外编一个值。这个方法以前无论传入什么 solution/grid，都直接
        返回硬编码的占位数字（"假设 Ahmed body 在 30 m/s 下的典型值"）
        ——这是一个真实的 bug，不是占位符：任何调用它的用户都会拿到一个
        看起来合理、实则完全虚构的答案，且没有任何提示说明这不是真实
        结果。

        侧向力和三个力矩 AeroCoefficientCalculator **不**计算（它只积分
        阻力/升力轴向）——这些量如实报告为 0.0 并记录警告，而不是像以前
        那样凭空编造；阻力/升力现在则是真实值。

        Returns:
            AerodynamicForces: 力 (N) 与力矩 (N·m)

        Example:
            >>> forces = calc.calculate_forces()
            >>> print(f"Drag force: {forces['drag_force']:.1f} N")
        """
        logger.info("Calculating aerodynamic forces via real surface integration...")

        from ..core.fvm_faces import FVMFaceExtractor
        from ..core.aero_coeffs import AeroCoefficientCalculator

        if not hasattr(self.grid_data, "ensure_faces_exist"):
            raise TypeError(
                "calculate_forces() requires a volume mesh (VolumeMeshData) - "
                f"got {type(self.grid_data).__name__}. Real pressure/skin-friction "
                "surface integration needs FVM face connectivity built from actual "
                "3D cells, which a surface-only GridData doesn't have."
            )

        face_extractor = FVMFaceExtractor()
        face_data = self.grid_data.ensure_faces_exist()
        face_extractor.face_connectivity = face_data.connectivity
        face_extractor.face_normals = face_data.normal
        face_extractor.face_areas = face_data.area
        face_extractor.boundary_flags = (face_data.connectivity[:, 1] < 0).astype(np.int32)

        aero_calc = AeroCoefficientCalculator(
            self.grid_data, face_extractor, rho_inf=self.density, vel_inf=self.velocity
        )
        Cd, Cl, _Cd_p, _Cd_f = aero_calc.compute_coefficients(self.solution.data)

        drag_force = Cd * self.dynamic_pressure * self.reference_area
        lift_force = Cl * self.dynamic_pressure * self.reference_area

        logger.warning(
            "Side force and all three moments are not yet implemented "
            "(AeroCoefficientCalculator only integrates drag/lift) - "
            "reporting 0.0 for side_force/pitch_moment/yaw_moment/roll_moment."
        )

        forces = AerodynamicForces(
            drag_force=drag_force,
            lift_force=lift_force,
            side_force=0.0,
            pitch_moment=0.0,
            yaw_moment=0.0,
            roll_moment=0.0,
        )

        logger.info(f"Forces calculated:\n{forces}")
        return forces

    def calculate_by_boundary(
        self,
        boundary_name: str
    ) -> AerodynamicCoefficients:
        """计算指定边界的气动系数。

        对某一个特定边界组（例如 BODY、MIRROR、WHEEL）积分力，算出该
        边界自己的系数。

        Args:
            boundary_name: 边界组名称（例如 'BODY'、'MIRROR'）

        Returns:
            AerodynamicCoefficients: 该边界的系数

        Raises:
            KeyError: 网格中找不到该边界名

        Example:
            >>> body_coeffs = calc.calculate_by_boundary('BODY')
            >>> mirror_coeffs = calc.calculate_by_boundary('MIRROR')
        """
        if boundary_name not in self.grid_data.boundaries.groups:
            raise KeyError(
                f"Boundary '{boundary_name}' not found. "
                f"Available boundaries: {list(self.grid_data.boundaries.groups.keys())}"
            )

        logger.info(f"Calculating coefficients for boundary: {boundary_name}")

        # TODO: 实现按边界的积分——需要按边界组过滤面，只对那部分面积分
        # 占位实现：返回全零系数
        coeffs = AerodynamicCoefficients()

        logger.warning(
            f"Boundary-specific calculation not yet fully implemented. "
            f"Returning zero coefficients for '{boundary_name}'."
        )
        return coeffs
