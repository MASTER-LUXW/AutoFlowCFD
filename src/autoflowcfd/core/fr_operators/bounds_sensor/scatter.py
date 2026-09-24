"""AutoFlowCFD V2.0 - 按面把邻居极值散射累加到单元（BJ 包络的底层归约）

从 `src/autoflowcfd/core/fr_operators/bounds_sensor.py`(原 506 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


import numpy as np



# 档位解析（`AFCFD_TROUBLED_SENSOR`）已拆到 `troubled_sensor_mode.py`
# （2026-09-19，项目"单文件不超 500 行"规范）。这里 re-export，全仓库
# `from ...bounds_sensor import resolve_troubled_sensor` 不用改。


def _scatter_minmax(xp, nb_max, nb_min, idx, values):
    """`nb_max[idx] = max(nb_max[idx], values)` 与 min 的对偶，**索引可重复**。

    为什么要这个分派层：BJ 判据的邻域包络必须对同一个 owner 单元累积它
    *全部*面邻居的均值，所以是一次索引重复的散射归约。NumPy 用
    `np.maximum.at`，CuPy 没有 ufunc.at，对应的是 `cupyx.scatter_max`/
    `cupyx.scatter_min`（语义完全一致：对重复索引做归约而不是后写覆盖）。

    做成分派层而不是给 GPU 另写一份 `compute_bounds_violation_mask`：这个
    判据要同时服务 CPU 单机 / CPU MPI / 单 GPU / 多 GPU 四条后端，四份
    人工同步的副本在本项目已经反复出过"只改了一份"的真实缺陷（见项目
    记忆 `feedback-prefer-deleting-redundant-code`）。除这三行散射之外，
    整个判据本来就是纯数组运算，NumPy/CuPy 同名同义。

    Raises:
        RuntimeError: 传入的是 CuPy 数组但该版本 cupyx 没有 scatter_max/
            scatter_min。不静默退回逐元素循环——那在 GPU 上是灾难性的
            性能陷阱，而且会让"判据开着"与"判据实际生效"看起来一样。
    """
    if xp is np:
        np.maximum.at(nb_max, idx, values)
        np.minimum.at(nb_min, idx, values)
        return
    import cupyx
    smax = getattr(cupyx, "scatter_max", None)
    smin = getattr(cupyx, "scatter_min", None)
    if smax is None or smin is None:
        raise RuntimeError(
            "BJ 越界判据在 GPU 上需要 cupyx.scatter_max/scatter_min（对重复"
            "索引做归约的散射），当前 CuPy 版本没有提供。请升级 CuPy——"
            "不退回逐元素循环：那在 GPU 上是灾难性的性能陷阱。"
        )
    smax(nb_max, idx, values)
    smin(nb_min, idx, values)
