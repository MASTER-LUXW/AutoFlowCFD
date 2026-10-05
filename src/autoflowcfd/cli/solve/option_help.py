"""`solve steady/transient/resume` 共用选项的帮助文字（唯一来源）。

2026-10-05 以前三个命令各写一份，且都已过时："仅单机 CPU 路径支持、其余后端传非默认值会报错"
（Order Continuation 早已四个后端同一个循环）、"默认时目标阶数与非最终阶段拿到同一份额"（2026-09-05
起目标阶数无条件吃掉剩余步数）、"CPU 后端 numba 线程数"（单 GPU 的主机侧 Jacobian 装配同样使用）。
"""

PHASE_MAX_ITER_HELP = (
    "Order Continuation（目标阶数 >= 1 时触发）非最终阶段（P0/P1/...，不含目标阶数）各自的最大迭代步数；"
    "不传时取总步数按阶段数均分的值。目标阶数不受它约束，吃掉剩余的全部步数（满足 --residual-drop-threshold "
    "的非最终阶段可提前升阶）。四个后端（CPU / 单 GPU / CPU MPI / 多 GPU）是同一个循环")

RESIDUAL_DROP_THRESHOLD_HELP = (
    "Order Continuation 非最终阶段提前升阶所需的残差下降倍数（默认 100，即 2 个数量级；湍流方程一起到位"
    "才升阶）。四个后端是同一个循环")

THREADS_HELP = (
    "numba 并行 kernel 使用的线程数（CPU 残差；GPU 后端的主机侧 Jacobian 装配），"
    "默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）")
