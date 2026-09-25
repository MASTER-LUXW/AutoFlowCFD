"""AutoFlowCFD V2.0 - 何时走 Order Continuation（P0 -> ... -> 目标阶数）的唯一判据。

全部后端（单机 CPU、单机 GPU、CPU-MPI 两种加载模式、多 GPU）的 `solve()` 都
调用 `uses_order_continuation`；此前四处各写一个 `self.order >= 2`。

## 目标阶数为 1 也从 P0 起步（2026-09-25）

此前只有目标阶数 >= 2 才爬坡，P1 直接从均匀初场冲击启动。同一版代码、同一网格
（plate_demo，17.9 万单元，SST，隐式 NK）的 A/B：

| 第 20 步（P1）         | 先 P0 再 P1        | 直接 P1 起步             |
|---|---|---|
| 驻点线 Cp_t 最大       | 1.12（物理值约 1） | 3.52                    |
| 温度 T/T_inf           | [0.996, 1.005]     | [0.905, 1.346]，31 个解点偏离 >5% |
| 板边最大速度           | 50 m/s             | 152 m/s（来流 33 m/s）   |
| 平均流残差（P1 第 5~20 步） | 2.3e7 -> 2.0e6 持续下降 | ~5e7 附近停滞 |

直接起步时，锐边与驻点处的初始冲击在 P1 的高阶模态里激起非物理状态，并且
不会自行消退；P0 阶段先把大尺度流场建立起来，再延拓到 P1。

`order_continuation_enabled = False` 仍可显式关闭（稳定边界扫描、固定阶数的
对照测试需要它）。
"""

#: 走 Order Continuation 的最低目标阶数（目标 P0 没有可爬的阶）。
MIN_TARGET_ORDER = 1


def uses_order_continuation(solver) -> bool:
    """该求解器的 `solve()` 是否应分派到逐阶爬坡（见模块文档）。"""
    return (bool(getattr(solver, "order_continuation_enabled", True))
            and int(solver.order) >= MIN_TARGET_ORDER)
