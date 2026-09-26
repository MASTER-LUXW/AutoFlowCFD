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

## 阶段内的升阶 / 收敛判据：湍流必须一起到位（2026-09-26，`PhaseGate`）

此前两套切阶循环只看平均流残差相对本阶段基准的下降倍数（外加至少 20 步）。
湍流产生项斜坡要 50 步才完成，残差基准也是在斜坡完成时才重置——于是隐式 NK
下 P0 在第 21 步、产生项系数只有 0.42 时就满足了判据（基准还是斜坡开始前的
值），直接升阶，湍流从没真正打开过：plate_demo P0 终态与 P1 第 19/39 步，剪切
层与尾迹里 k 仍约为来流值、nu_t/nu 约为来流的 5（正常应为数百到数千），分离
剪切层实际上是层流的，P1 在锐边处局部不稳定、CFL 升不上去。RANS 的"本阶段
已收敛"必须包括湍流方程：

* 有产生项斜坡的湍流模型时，斜坡未完成不升阶、不判收敛；
* 隐式湍流（NK）的残差也要相对本阶段基准（斜坡完成后的第一个值）下降到同样
  倍数；显式格式不跟踪湍流残差，只检查斜坡。
"""

#: 走 Order Continuation 的最低目标阶数（目标 P0 没有可爬的阶）。
MIN_TARGET_ORDER = 1


def uses_order_continuation(solver) -> bool:
    """该求解器的 `solve()` 是否应分派到逐阶爬坡（见模块文档）。"""
    return (bool(getattr(solver, "order_continuation_enabled", True))
            and int(solver.order) >= MIN_TARGET_ORDER)


def turbulence_residual_norm(solver):
    """隐式湍流（NK）最近一步的残差范数；没有隐式湍流时 None。"""
    st = getattr(solver, "_newton_turb_state", None)
    info = st.get("last_info") if st else None
    return None if not info else float(info["res_norm"])


def production_ramp_complete(solver) -> bool:
    """湍流产生项斜坡是否已完成（没有斜坡的模型恒为 True）。"""
    model = getattr(solver, "turb_model", None)
    if model is None:
        model = getattr(solver, "turb_model_gpu", None)
    if model is None or not hasattr(model, "production_factor"):
        return True
    return bool(getattr(solver, "_turb_production_ramp_complete", False))


class PhaseGate:
    """一个 Order Continuation 阶段内的升阶 / 收敛判据（见模块文档）。每个阶段新建一个。"""

    __slots__ = ("turb_baseline",)

    def __init__(self):
        self.turb_baseline = None

    def turbulence_drop(self, solver):
        """隐式湍流残差相对本阶段基准的下降倍数；没有隐式湍流时 None，斜坡未完成时 0。"""
        r = turbulence_residual_norm(solver)
        if r is None:
            return None
        if not production_ramp_complete(solver):
            return 0.0
        if self.turb_baseline is None:
            self.turb_baseline = r
        return self.turb_baseline / max(r, 1e-300)

    def reached(self, solver, mean_drop: float, required: float) -> bool:
        """平均流与湍流是否都已达到 `required` 倍下降（且斜坡已完成）。"""
        if not production_ramp_complete(solver):
            return False
        turb = self.turbulence_drop(solver)
        return mean_drop >= required and (turb is None or turb >= required)
