"""边界类型映射器和参数验证器。

从 config.py 拆出，控制单文件行数。

核心组件:
    - BoundaryTypeMapper: 属性名到边界类型的自动映射
    - ParameterValidator: 边界条件参数物理一致性验证
"""

from typing import List
from loguru import logger


class BoundaryTypeMapper:
    """边界类型自动映射器

    Maps property names to boundary condition types using keyword matching.

    Example:
        >>> mapper = BoundaryTypeMapper()
        >>> bc_type = mapper.map("INLET_BOUNDARY")
        >>> print(bc_type)  # VELOCITY_INLET
    """

    def __init__(self):
        """Initialize mapper with default keyword rules."""
        self.rules = {
            'VELOCITY_INLET': ['INLET', 'INFLOW', 'VELOCITY_INLET'],
            'PRESSURE_OUTLET': ['OUTLET', 'OUTFLOW', 'PRESSURE_OUTLET'],
            'SYMMETRY': ['SYMMETRY', 'SYMM'],
            'SLIP_WALL': ['TUNNEL', 'FARFIELD', 'FAR', 'BOUNDARY'],
            'PERIODIC': ['PERIODIC'],
        }

    def map(self, property_name: str) -> str:
        """Map property name to boundary condition type

        Args:
            property_name: Property name from NAS file

        Returns:
            str: Boundary condition type
        """
        name_upper = property_name.upper()

        for bc_type, keywords in self.rules.items():
            if any(keyword in name_upper for keyword in keywords):
                return bc_type

        # Default to WALL for unmatched properties
        return 'WALL'

    def add_rule(self, bc_type: str, keywords: List[str]) -> None:
        """Add custom mapping rule

        Args:
            bc_type: Boundary condition type
            keywords: List of keywords to match

        Raises:
            ValueError: If bc_type is invalid
        """
        valid_types = [
            'VELOCITY_INLET',
            'PRESSURE_OUTLET',
            'WALL',
            'SYMMETRY',
            'SLIP_WALL'
        ]

        if bc_type not in valid_types:
            raise ValueError(
                f"Invalid boundary type '{bc_type}'. Must be one of {valid_types}"
            )

        self.rules[bc_type] = keywords
        logger.debug(f"Added custom mapping rule: {bc_type} -> {keywords}")


class ParameterValidator:
    """边界条件参数验证器

    Validates boundary condition parameters for physical consistency.

    Example:
        >>> validator = ParameterValidator()
        >>> validator.validate_velocity([30.0, 0.0, 0.0])  # OK
    """

    @staticmethod
    def validate_velocity(velocity: List[float]) -> bool:
        """Validate velocity vector"""
        if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
            raise ValueError("Velocity to be a list of 3 floats")

        try:
            velocity = [float(v) for v in velocity]
        except (TypeError, ValueError):
            raise ValueError("Velocity values must be numeric")

        magnitude = sum(v**2 for v in velocity)**0.5
        if magnitude < 0:
            raise ValueError("Velocity magnitude cannot be negative")

        if magnitude > 500:
            logger.warning(
                f"Very high velocity magnitude: {magnitude:.1f} m/s. "
                f"Ensure this is intended."
            )

        return True

    @staticmethod
    def validate_pressure(pressure: float) -> bool:
        """Validate pressure value"""
        try:
            pressure = float(pressure)
        except (TypeError, ValueError):
            raise ValueError("Pressure must be numeric")

        if pressure < -100000:
            logger.warning(
                f"Very low gauge pressure: {pressure:.0f} Pa. "
                f"This corresponds to near-vacuum conditions."
            )

        return True

    @staticmethod
    def validate_turbulence_intensity(ti: float) -> bool:
        """Validate turbulence intensity"""
        try:
            ti = float(ti)
        except (TypeError, ValueError):
            raise ValueError("Turbulence intensity must be numeric")

        if not (0.0 <= ti <= 1.0):
            raise ValueError(
                f"Turbulence intensity must be between 0.0 and 1.0, got {ti}"
            )

        if ti > 0.2:
            logger.warning(
                f"High turbulence intensity: {ti:.3f}. "
                f"Typical values are 0.01-0.1 for external flows."
            )

        return True

    @staticmethod
    def validate_roughness_height(height: float) -> bool:
        """Validate roughness height"""
        try:
            height = float(height)
        except (TypeError, ValueError):
            raise ValueError("Roughness height must be numeric")

        if height < 0:
            raise ValueError("Roughness height must be non-negative")

        if height > 0.1:
            logger.warning(
                f"Very large roughness height: {height:.3f} m. "
                f"Ensure this is intended."
            )

        return True
