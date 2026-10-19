"""AutoFlowCFD V2.0 - 湍流量的非有限值替换与如实报告（SST、SA-neg，CPU 与 GPU 共用）。

退化网格上源项、输运或涡粘可能出现 inf-inf；非有限值一旦进入场更新就不可恢复，所以在这些位置替换掉。替换本身
必须报告（2026-10-19，I-48 / I-45：此前各处静默归零，退化单元上的失控对用户不可见）：累计解点数记在
`model.nonfinite_count`，首次出现告警、此后累计数每翻一倍再告警一次（不逐步刷屏）。
"""

from loguru import logger


def replace_nonfinite(model, arr, what: str, xp, fill=0.0):
    """`arr` 的非有限元素换成 `fill` 并计数告警；没有非有限元素时原样返回。

    Args:
        model: 湍流模型（构造时 `nonfinite_count = 0`、`_nonfinite_reported = 0`）
        arr: 数组
        what: 告警里的量名
        xp: `arr` 所在的数组模块
        fill: 替换值（标量或可广播的数组）
    """
    finite = xp.isfinite(arr)
    n_bad = int(arr.size - xp.count_nonzero(finite))
    if n_bad == 0:
        return arr
    model.nonfinite_count += n_bad
    if model.nonfinite_count >= 2 * model._nonfinite_reported:
        model._nonfinite_reported = model.nonfinite_count
        logger.warning(f"湍流 {what} 出现 {n_bad} 个非有限值，已替换（累计 {model.nonfinite_count}）"
                       "——检查退化单元或湍流输运")
    return xp.where(finite, arr, fill)
