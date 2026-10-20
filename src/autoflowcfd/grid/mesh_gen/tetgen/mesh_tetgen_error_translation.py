"""tetgen 原始报错到本项目专属排查建议的翻译。

从 mesh_tetgen_core.py 的 fill_core_volume 拆出来——tetgen 自身的 C 层报
错文本对使用者不友好（不知道该调哪个参数），这里把两类已经在真实案例上
定位过根因的报错，翻译成带具体排查建议的信息；不认识的报错原样返回
None，调用方按原始异常继续抛出，不遮盖未知失败模式。
"""

from typing import Optional


def translate_tetgen_failure(exc: RuntimeError) -> Optional[RuntimeError]:
    """把 tetgen 抛出的 RuntimeError 翻译成带排查建议的版本。

    Args:
        exc: tetgen.tetrahedralize 抛出的原始 RuntimeError

    Returns:
        翻译后的 RuntimeError（调用方应该 `raise translated from exc`），
        或 None（未识别的报错类型——调用方应该原样重新抛出 exc 本身，
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
        # 在真实案例上观察到：BL 挤出在大比例表面顶点上停长（收敛预算
        # 耗尽、节点冻结在后续各层同一坐标）时产生大量近重合点，超过
        # tetgen 自身的内部数值鲁棒性极限（tetgen 的实现限制），而不是
        # 像上面自交情况那样给出更清晰的诊断。
        return RuntimeError(
            f"{exc}. tetgen hit an internal robustness limit - on a case "
            f"seen directly, this followed boundary-layer extrusion stalling "
            f"over a large fraction of the surface (many near-coincident "
            f"points). Try loosening --growth-rate/--min-cell-size/--bl-layers "
            f"so fewer BL columns stall."
        )

    return None
