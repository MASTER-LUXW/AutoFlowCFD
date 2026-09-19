"""AutoFlowCFD V2.0 - troubled-cell 判据档位解析（`AFCFD_TROUBLED_SENSOR`）。

从 `bounds_sensor.py`（原 568 行）拆出（2026-09-19，项目"单文件不超
500 行"规范）：那边是 BJ 型越界判据的**实现**，这里只是档位解析这一
件事（与 `fr/modal_filter.py` 的 `_FILTER_MODE` 解析是成对的默认值，
见下方函数文档）。纯搬家，逻辑未改。
"""

from typing import Optional


def resolve_troubled_sensor(value: Optional[str] = None) -> str:
    """解析 `AFCFD_TROUBLED_SENSOR`：`persson` | `bounds` | `both`。

    **默认 `bounds`（2026-09-17 从 `persson` 改）。**

    为什么改：Persson-Peraire 在 `order=1` 上**原理性退化**（`s0 =
    -4*log10(order)` 在 order=1 时为 0，触发门限成了"顶模态能量占比
    >= 10%"，而 P1 的顶模态就是全部非常数内容），而且它探的是守恒密度
    ——真实解上那一项是光滑的，掩码实测 **0.000%**（同一时刻 `rho_v`/
    `rho_w` 是 98.8%）。所以 `persson` 在 P1 上等于**没有门控**：实测
    `AFCFD_FILTER_MODE=sensor` + `persson` 与 `FILTER_MODE=off` **逐位
    相同**（平板边界层算例 res 1.6658e+05）。

    与 `AFCFD_FILTER_MODE=sensor` 必须**成对**使用（同日一起改默认）：
    只改一个等于把默认值悄悄改成 `off`。三档在 P1 上的实测对照见
    `fr/modal_filter.py` 里 `_FILTER_MODE` 上方那节。

    真实网格上的决定性证据（plate_demo_volume_les，179,237 单元）：
    `legacy` 在 iter 112 发散，而 `sensor`+`bounds` 跑出 216 步残差
    **单调下降 3.5 倍**，Cd 漂移从零曲率的线性 0.0167/步变成负曲率的
    0.0071 -> 0.0042/步。等熵涡精确解上 P1 的收敛阶保住 2.16/2.18
    （设计阶 2）。

    `persson` 保留为合法档：它在 order>=2 上判据本身是有效的，且是复现
    历史结果的唯一途径。

    Raises:
        ValueError: 取值非法（不静默回退，理由同
            `fr_operators/kernels.py::resolve_ausm_precond_mode`：静默回退
            会让一次拼写错误伪装成默认行为、把 A/B 的两条运行悄悄变成
            同一档）。
    """
    import os

    if value is None:
        value = os.environ.get("AFCFD_TROUBLED_SENSOR", "").strip()
        if not value:
            return "bounds"
    key = str(value).strip().lower()
    if key in ("persson", "bounds", "both"):
        return key
    raise ValueError(
        f"AFCFD_TROUBLED_SENSOR 取值非法: {value!r}；"
        f"合法值 ['bounds', 'both', 'persson']"
    )
