"""tetgen 原始报错到本项目专属排查建议的翻译。

从 mesh_tetgen_core.py 的 fill_core_volume 拆出来 - tetgen 自身的 C 层报
错文本对使用者不友好（不知道该调哪个参数），这里把两类已经在真实 case 上
定位过根因的报错，翻译成带具体排查建议的信息；不认识的报错原样返回 None，
调用方按原始异常继续抛出，不遮盖未知失败模式。
"""

from typing import Optional


def translate_tetgen_failure(exc: RuntimeError) -> Optional[RuntimeError]:
    """把 tetgen 抛出的 RuntimeError 翻译成带排查建议的版本。

    Args:
        exc: tetgen.tetrahedralize 抛出的原始 RuntimeError

    Returns:
        翻译后的 RuntimeError（调用方应该 `raise translated from exc`），
        或者 None（未识别的报错类型 - 调用方应该原样重新抛出 exc 本身，
        不能吞掉或伪造一个更"友好"但可能误导的翻译）。
    """
    message = str(exc).lower()

    if "self-intersection" in message:
        return RuntimeError(
            f"{exc}. The BL outer surface self-intersects at a tight local "
            f"feature (common at small welded contact patches with sharp "
            f"edges). Try fewer/thinner BL layers (--bl-layers, "
            f"--min-cell-size) - naive normal-offset extrusion has no "
            f"per-feature thickness limiting yet, so cumulative BL "
            f"thickness must stay well under the tightest local gap in "
            f"the geometry."
        )

    if "removevertexbyflips" in message or "internal tetgen error" in message:
        # Observed on a real case when Stage B's reactive BL thickness cap
        # (mesh_repair.compute_bl_thickness_limit_override) needs to cap a
        # very large fraction of surface vertices - itself already a
        # symptom of Stage A leaving widespread, not localized, bad cells -
        # producing a boundary facet with enough near-coincident points to
        # exceed tetgen's own numerical robustness limits internally (a
        # tetgen implementation limitation, not a meshing-strategy error on
        # this codebase's side) rather than failing with a clearer
        # diagnostic like the self-intersection case above.
        return RuntimeError(
            f"{exc}. tetgen hit an internal robustness limit - on a case "
            f"seen directly, this followed a very widespread Stage B "
            f"BL-thickness cap (a sign Stage A already found bad cells "
            f"across much of the surface, not just a few corners). Try "
            f"loosening --growth-rate/--min-cell-size/--bl-layers so "
            f"Stage A has fewer bad cells to begin with."
        )

    return None
