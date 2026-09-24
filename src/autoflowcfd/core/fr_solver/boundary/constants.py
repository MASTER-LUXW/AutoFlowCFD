"""AutoFlowCFD V2.0 - boundary 的共享常量。

从 `src/autoflowcfd/core/fr_solver/boundary.py` 拆出(2026-09-24)。**唯一事实来源** -- 子模块一律从这里导入,
绝不各自复制一份(那是本项目明令禁止的"同一语义两个事实来源")。
"""







# LES/DDES 入口合成湍流默认参数（BD-02）。真实修复（V2.0 专家组盲审
# 发现，2026-08-28）：此前这两个量恒为硬编码常量，没有任何 CLI/配置
# 途径覆盖，跑真实工程案例（不同来流湍流度/涡核密度）只能改源码。现在：
# - 目标雷诺应力改为复用已有的 `--turbulence-intensity`（solver._turbulence_
#   intensity，同一个量本来就用于 RANS 自由来流 k/omega 初值），不再单独
#   维护一个 SEM 专用湍流度——一个旋钮统一表达"来流湍流强度"，
#   见 build_boundary_ghost_provider 内 u_fluct 的计算。这个下面的常量
#   现在只是 solver 没有设置 _turbulence_intensity 属性时（理论上不会
#   发生，FRSolver/GPUFRSolver 构造时恒会设置）的兜底默认值。
# - 涡核数量新增 `--sem-num-eddies` CLI 选项（solver._sem_num_eddies），
#   默认值沿用原来的 200（未指定时保持向后兼容的行为）。
_SEM_DEFAULT_TURBULENCE_INTENSITY = 0.01

_SEM_DEFAULT_NUM_EDDIES = 200

#: BJ 越界判据里"没有 Dirichlet 值"的标记。用 NaN 而不是哨兵数值：任何
#: 有限哨兵都可能与真实边界值撞车，而 NaN 在 `xp.isfinite` 下是无歧义的。
_NO_DIRICHLET = float("nan")
