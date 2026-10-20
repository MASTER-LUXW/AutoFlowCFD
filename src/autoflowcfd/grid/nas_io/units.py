"""NAS 文件坐标单位（唯一实现：面网格解析 NASParser 与体网格解析 parse_volume_mesh_nas 共用，CLI 的 --units 取值
也来自这里）。内部坐标一律为米。"""

from typing import Tuple

from loguru import logger

#: --units 的取值：文件坐标的长度单位，'auto' 按原始包围盒大小判定
UNITS_CHOICES = ("mm", "m", "auto")
#: 各单位到米的换算
SCALE_TO_METERS = {"mm": 1e-3, "m": 1.0}
#: units='auto' 时：原始（未缩放）包围盒最大边长高于此值视为毫米，否则视为米。单车外流计算域通常是几米（车）到
#: 几十米（风洞 / 远场），同一几何以毫米表示会到几千，远在阈值另一侧。
AUTO_UNITS_MM_THRESHOLD = 50.0


def resolve_units(units: str, raw_extent: float, source: str = "") -> Tuple[str, float]:
    """文件坐标单位 -> (实际单位 'mm' / 'm', 换算到米的比例)。

    Args:
        units: 'mm'、'm' 或 'auto'
        raw_extent: 原始坐标包围盒的最大边长
        source: 日志里标注的文件名

    Raises:
        ValueError: units 不是 'mm' / 'm' / 'auto'
    """
    if units not in UNITS_CHOICES:
        raise ValueError(f"units must be one of {UNITS_CHOICES}, got {units!r}")
    resolved = units
    if units == "auto":
        resolved = "mm" if raw_extent > AUTO_UNITS_MM_THRESHOLD else "m"
        logger.info(f"{source} units='auto': raw bounding-box max extent={raw_extent:.4g} "
                    f"{'>' if resolved == 'mm' else '<='} {AUTO_UNITS_MM_THRESHOLD:g} -> {resolved}")
    scale = SCALE_TO_METERS[resolved]
    extent_m = raw_extent * scale
    # 换算后计算域尺寸明显不合理时提醒（不改值）：单位给错是静默的，网格 / 求解会在物理上无意义的几何上进行
    if extent_m < 0.1:
        logger.warning(f"{source} 换算后（units={units!r} -> {resolved}）包围盒最大边长只有 {extent_m:.4g} m，"
                       "远小于汽车外流计算域的常见尺度——文件的实际单位可能不是这个，请检查 --units")
    elif extent_m > 1000.0:
        logger.warning(f"{source} 换算后（units={units!r} -> {resolved}）包围盒最大边长为 {extent_m:.4g} m，"
                       "远大于汽车外流计算域的常见尺度——文件的实际单位可能不是这个，请检查 --units")
    return resolved, scale
