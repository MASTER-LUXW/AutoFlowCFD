"""AutoFlowCFD V2.0 - boundary 的共享常量。

从 `src/autoflowcfd/core/fr_solver/boundary.py` 拆出(2026-09-24)。**唯一事实来源** -- 子模块一律从这里导入,
绝不各自复制一份(那是本项目明令禁止的"同一语义两个事实来源")。
"""







# LES/DDES 入口合成湍流（BD-02）：目标雷诺应力复用 `--turbulence-intensity`（solver._turbulence_intensity，
# 与 RANS 来流湍流同一个量），涡核数量由 `--sem-num-eddies`（solver._sem_num_eddies）给定。下面是涡核数的
# 默认值（各求解器构造签名的唯一来源）；构造边界提供者时直接读求解器上的两个值、不再兜底（2026-10-04：
# 完全分布式加载的根桩缺这两个属性，兜底让 CLI 设置被静默忽略）。
_SEM_DEFAULT_NUM_EDDIES = 200

#: BJ 越界判据里"没有 Dirichlet 值"的标记。用 NaN 而不是哨兵数值：任何
#: 有限哨兵都可能与真实边界值撞车，而 NaN 在 `xp.isfinite` 下是无歧义的。
_NO_DIRICHLET = float("nan")
