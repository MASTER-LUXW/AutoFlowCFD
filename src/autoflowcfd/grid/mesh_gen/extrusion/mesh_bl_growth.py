"""边界层生长逻辑。

负责边界层挤出时每层厚度与增长率的计算。为控制行数从 mesh_extrusion.py
拆分出来。
"""

# 总 BL 层数的硬底线
_MAX_SAFETY_LAYERS = 200


def compute_layer_thickness(
    current_thickness: float,
    growth_rate: float,
    base_thickness: float,
    layer_idx: int,
) -> float:
    """计算下一个 BL 层结束时的目标累积厚度。

    从 base_thickness 几何增长：第 0 层厚度为 base_thickness，第 1 层为
    base_thickness * growth_rate，等等。

    Args:
        current_thickness: 当前累积厚度
        growth_rate: 使用的增长率
        base_thickness: 第一层自身的厚度增量，几何增长的基础
        layer_idx: 几何增长的 0 基指数（BL 局部层索引）

    Returns:
         新的目标累积厚度
    """
    next_layer_thickness = base_thickness * (growth_rate ** layer_idx)
    return current_thickness + next_layer_thickness
