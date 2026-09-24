"""AutoFlowCFD V2.0 - gpu_viscous 的共享常量。

从 `src/autoflowcfd/core/gpu/residual/gpu_viscous.py` 拆出(2026-09-24)。**唯一事实来源** -- 子模块一律从这里导入,
绝不各自复制一份(那是本项目明令禁止的"同一语义两个事实来源")。
"""


# 罚项常数与 cp/R_AIR 统一从 CPU 侧那一份取（此前本文件各自 `R_AIR = 287.0`
# 与 `_VISCOUS_BOUNDARY_IP_C = 4.0`，罚项公式也整个抄了一遍 —— 本项目已多次
# 因"两份实现只改了一份"出真实缺陷）。


GAMMA = 1.4
