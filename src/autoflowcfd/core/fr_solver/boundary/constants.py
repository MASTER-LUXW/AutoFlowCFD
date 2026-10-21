"""AutoFlowCFD V2.0 - boundary 的共享常量。

从 `src/autoflowcfd/core/fr_solver/boundary.py` 拆出(2026-09-24)。**唯一事实来源** -- 子模块一律从这里导入,
绝不各自复制一份(那是本项目明令禁止的"同一语义两个事实来源")。
"""


# LES/DDES 入口合成湍流（`boundary/synthetic_inlet.py`）：雷诺应力与涡核尺度由来流湍流强度、涡粘比给定（与 RANS
# 来流同一组量），涡核数量由 `--sem-num-eddies`（solver._sem_num_eddies）给定。下面是涡核数的默认值（各求解器
# 构造签名与 CLI 的唯一来源）：None 表示按 Jarrin 密度判据 N = V_B / sigma_min^3 自动取（2026-10-17 以前固定 200，
# 与入口大小、涡核尺度无关）。构造边界提供者时直接读求解器上的值、不兜底（2026-10-04：完全分布式加载的根桩缺
# 这些属性，兜底让 CLI 设置被静默忽略）。
_SEM_DEFAULT_NUM_EDDIES = None

#: BJ 越界判据里"没有 Dirichlet 值"的标记。用 NaN 而不是哨兵数值：任何
#: 有限哨兵都可能与真实边界值撞车，而 NaN 在 `xp.isfinite` 下是无歧义的。
_NO_DIRICHLET = float("nan")
