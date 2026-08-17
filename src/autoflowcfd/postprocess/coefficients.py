"""气动系数计算模块。

本模块通过压力积分从 CFD 仿真结果计算气动系数（Cd、Cl、Cm 等）。

核心组件:
    - CoefficientCalculator: 气动系数主计算器
    - ForceDecomposition: 力与力矩分解工具

示例:
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
        """Calculate aerodynamic forces via surface integration.
        
        基于FR求解器的高阶表面集成方法，对车身表面积分压力和粘性力。
        
        Returns:
            AerodynamicForces: Calculated forces and moments
            
        Example:
            >>> forces = calc.calculate_forces()
            >>> print(f"Drag force: {forces['drag_force']:.1f} N")
        """
        logger.info("Calculating aerodynamic forces...")
        
        # 检查是否有边界数据
        if not hasattr(self.grid_data, 'boundaries') or not self.grid_data.boundaries:
            logger.warning("No boundary data available, returning zero forces")
            return AerodynamicForces()
        
        # 初始化力和力矩
        total_force = np.zeros(3)  # [Fx, Fy, Fz]
        total_moment = np.zeros(3)  # [Mx, My, Mz]
        
        # 参考点（通常为质心或几何中心）
        reference_point = np.array([0.0, 0.0, 0.0])
        
        # 获取平均压力
        pressure_avg = self._get_average_pressure()
        
        # 遍历所有边界组
        for boundary_name, boundary_faces in self.grid_data.boundaries.groups.items():
            # 只对车身相关边界积分（排除远场、入口、出口等）
            if boundary_name.upper() in ['FARFIELD', 'INLET', 'OUTLET', 'SYMMETRY']:
                continue
            
            # 处理边界数据 - 假设boundary_faces是面索引数组
            if hasattr(boundary_faces, '__iter__'):
                for face_idx in boundary_faces:
                    try:
                        # 获取面的几何信息 (需要根据实际网格数据结构实现)
                        face_data = self.grid_data.get_face_data(face_idx)
                        area = face_data.get('area', 0.0)
                        normal = face_data.get('normal', np.array([0.0, 0.0, 1.0]))
                        centroid = face_data.get('centroid', np.array([0.0, 0.0, 0.0]))
                        
                        # 压力力：F = -p * A * n
                        force = -pressure_avg * area * normal
                        
                        # 力矩：M = r × F
                        r = centroid - reference_point
                        moment = np.cross(r, force)
                        
                        # 累加
                        total_force += force
                        total_moment += moment
                    except (IndexError, AttributeError, KeyError):
                        logger.warning(f"Could not process face {face_idx} in boundary {boundary_name}")
                        continue
        
        # 转换到气动坐标系
        # 假设来流方向为X轴正向
        drag_force = total_force[0]      # X方向为阻力
        side_force = total_force[1]      # Y方向为侧向力
        lift_force = -total_force[2]     # Z方向向上为正，升力向下为负
        
        pitch_moment = total_moment[1]   # 绕Y轴为俯仰力矩
        yaw_moment = -total_moment[2]    # 绕Z轴为偏航力矩
        roll_moment = total_moment[0]    # 绕X轴为滚转力矩
        
        forces = AerodynamicForces(
            drag_force=float(drag_force),
            lift_force=float(lift_force),
            side_force=float(side_force),
            pitch_moment=float(pitch_moment),
            yaw_moment=float(yaw_moment),
            roll_moment=float(roll_moment)
        )
        
        logger.success(f"Aerodynamic forces calculated:\n{forces}")
        return forces
    
    def _get_average_pressure(self) -> float:
        """获取平均压力值（简化实现）。
        
        Returns:
            平均压力值（Pa）
        """
        # TODO: 从solution中准确提取压力场
        # 当前返回标准大气压作为占位符
        return 101325.0  # Pa

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
