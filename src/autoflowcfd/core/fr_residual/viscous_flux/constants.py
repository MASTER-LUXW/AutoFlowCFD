"""AutoFlowCFD V2.0 - viscous_flux 的共享常量。

从 `src/autoflowcfd/core/fr_residual/viscous_flux.py` 拆出(2026-09-24)。**唯一事实来源** -- 子模块一律从这里导入,
绝不各自复制一份(那是本项目明令禁止的"同一语义两个事实来源")。
"""








GAMMA = 1.4

R_AIR = 287.0  # 空气比气体常数 J/(kg*K)

#: 分子 / 湍流普朗特数。残差（CPU 各路径、GPU）与解析 Jacobian 共用这一份。
PRANDTL = 0.72
PRANDTL_TURBULENT = 0.9
