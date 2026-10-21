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
        # 现行挤出按"与对面表面的局部间隙"限厚（`mesh_tetgen_seam.compute_local_thickness_limit`：累积厚度不超过
        # 间隙的 0.45 倍，间隙在未变形表面上评估、搜索半径为域尺寸的 8%）。它是几何启发式，不是非交叉保证：两侧都在
        # 挤出的强烈弯曲前沿，以及搜索半径之外的对面（大域里的细长缝隙）仍可能相交
        return RuntimeError(
            f"{exc}。边界层外表面在局部紧致特征处自相交（常见于带锐边的小面积焊接接触区、细长缝隙）。"
            f"挤出已按与对面表面的局部间隙限制累积厚度，但该限制在未变形表面上评估：两侧都在挤出的强烈弯曲前沿、"
            f"或间隙对面在搜索半径（域尺寸的 8%）之外时仍可能相交。可减少层数或减小厚度（--bl-layers、"
            f"--first-layer-height / --target-yplus、--growth-rate），或在源几何的该处做小倒角 / 圆角。"
        )

    if "removevertexbyflips" in message or "internal tetgen error" in message:
        # 在真实案例上观察到：BL 挤出在大比例表面顶点上停长（收敛预算
        # 耗尽、节点冻结在后续各层同一坐标）时产生大量近重合点，超过
        # tetgen 自身的内部数值鲁棒性极限（tetgen 的实现限制），而不是
        # 像上面自交情况那样给出更清晰的诊断。
        return RuntimeError(
            f"{exc}。tetgen 触到内部数值鲁棒性极限。实测案例里它发生在边界层挤出于大比例表面顶点上停长之后"
            f"（大量近重合点）。可放宽 --growth-rate / --min-cell-size / --bl-layers，让停长的边界层柱更少。"
        )

    return None
