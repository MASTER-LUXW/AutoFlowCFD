"""双时间步（DUAL_TIME）跨物理步保存的上一时间层。

每个物理步的 BDF 方程由隐式内迭代求解，平均流与带输运方程的湍流模型紧耦合
（`time_integration/implicit/newton/physical_time.py`）。BDF2 需要的上一物理时间层存在求解器上：平均流
`_dual_time_U_prev`，湍流 `_dual_time_turb_prev`（模型的 Newton 未知量：SST 为 k 与 ln omega，SA 为
nu_tilde），随 checkpoint 读写（`core/utils/checkpoint_time.py`）。
"""


def reset_dual_time_history(owner) -> None:
    """清空上一物理时间层（平均流与湍流一起）：构造、换阶、重分发、开始一次新的 `solve()` 时调用。"""
    owner._dual_time_U_prev = None
    owner._dual_time_turb_prev = None
