"""边界层生长逻辑。

负责边界层挤出时每层厚度与增长率的计算。为控制行数从 mesh_extrusion.py
拆分出来。
"""

import numpy as np
from loguru import logger

# Hard backstop on total BL layer count.
_MAX_SAFETY_LAYERS = 200


def compute_layer_thickness(
    current_thickness: float,
    growth_rate: float,
    base_thickness: float,
    layer_idx: int,
) -> float:
    """Compute the target CUMULATIVE thickness for the end of the next BL layer.

    Geometric growth from base_thickness: layer 0 has thickness
    base_thickness, layer 1 has base_thickness * growth_rate, etc.

    Args:
        current_thickness: Current cumulative thickness.
        growth_rate: The growth rate to use.
        base_thickness: The first layer's own thickness increment, grown
            geometrically from.
        layer_idx: 0-based exponent for the geometric growth (the BL-local
            layer index).

    Returns:
        The new target cumulative thickness.
    """
    next_layer_thickness = base_thickness * (growth_rate ** layer_idx)
    return current_thickness + next_layer_thickness
