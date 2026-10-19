"""AutoFlowCFD V2.0 - 块预处理刷新判据的确定性代价模型：一次块装配折合多少次 GMRES 迭代。

`block_jacobi.py` 的刷新判据是"上一步 GMRES 迭代数 > 刚装配完那一步的基线 + 一次装配折合的迭代数"
（多出来的迭代比重装配一次还贵时才重装配）。2026-09-26（ea6b04d）起这个折合数取实测墙钟耗时之比
`装配耗时 / 单次迭代耗时`，后果是同一输入跨进程的 Newton 轨迹不可逐位复现：SST 黄金轨迹的 P2 段在
两个结果之间切换，屏蔽墙钟项后逐位复现（2026-10-04）。按用户决策改为本模块的确定性查表。

## 标定

湍流平板（`tests/validation/_flat_plate_case.py`，3072 棱柱，单机 CPU、生产默认设置）各阶 8 个 NK 步，
解析装配的 `装配耗时 / 单次迭代耗时` 中位数：

| 块 | P1 | P2 | P3 |
|---|---|---|---|
| 平均流，层流 | 14.0 | 13.6 | 22.1 |
| 平均流，与 SA 耦合 / 与 SST 耦合 | 8.9 / 6.1 | 9.1 / 7.2 | 14.2 / 11.3 |
| SA 湍流（1 个未知量） | 2.2 | 2.0 | 7.0 |
| SST 湍流（k、ln omega 2 个未知量） | 2.2 | 3.7 | 16.9 |

耦合时单次迭代含湍流求值，平均流块的比值低于层流；SST 湍流块 P3 的逐点导数代价陡增。表值取上面
数据（耦合平均流取 SA、SST 两者的平均），低于 `REFRESH_SLACK_MIN` 的取下限。四面体槽道（144 单元）
上的比值被每步固定开销主导（单次迭代 3.6 ms），不作标定。plate_demo P1（36 万单元）此前实测装配
9~11 s、迭代约 1 s，与表中耦合平均流 P1 的量级一致。

差分装配（P0 等解析装配不覆盖的离散）的残差求值次数精确已知：一次差分列求值与一次 GMRES 迭代
各含一次整场残差求值，平板 P0 实测比值 36.3 / 50 次求值 = 0.73。

GPU 后端沿用 CPU 标定（块在主机上装配、迭代在设备上做，真实比值更大，刷新会偏勤）；需要在有 GPU 的
机器上按同一方法重新标定。
"""

#: 刷新判据松弛的下限（迭代数）：低于它时 GMRES 迭代数的正常波动就会触发重装配。
REFRESH_SLACK_MIN = 3

#: 差分装配：每次整场残差求值折合的 GMRES 迭代数（见模块文档"标定"）。
FD_EVAL_ITERS = 0.73

#: 解析装配折合的 GMRES 迭代数：`(块类型, 阶数) -> 迭代数`。块类型见 `block_kind`。
ANALYTIC_ASSEMBLY_ITERS = {
    ("mean_laminar", 1): 14, ("mean_laminar", 2): 14, ("mean_laminar", 3): 22,
    ("mean_coupled", 1): 8, ("mean_coupled", 2): 8, ("mean_coupled", 3): 13,
    ("turbulence_1", 0): 3, ("turbulence_1", 1): 3, ("turbulence_1", 2): 3, ("turbulence_1", 3): 7,
    ("turbulence_2", 0): 3, ("turbulence_2", 1): 3, ("turbulence_2", 2): 4, ("turbulence_2", 3): 17,
}


def block_kind(n_var: int, with_turbulence: bool) -> str:
    """块缓存的类型：平均流（层流 / 与湍流耦合）或湍流（按未知量个数）。"""
    if n_var == 5:
        return "mean_coupled" if with_turbulence else "mean_laminar"
    return f"turbulence_{n_var}"


def assembly_cost_iters(kind: str, order: int, analytic: bool, n_residual_evals: int) -> int:
    """一次装配折合的 GMRES 迭代数（刷新判据的松弛，不低于 `REFRESH_SLACK_MIN`）。

    Raises:
        KeyError: 解析装配的 `(块类型, 阶数)` 没有标定值——不静默套用别的值，补标定再用。
    """
    if not analytic:
        return max(REFRESH_SLACK_MIN, int(round(FD_EVAL_ITERS * n_residual_evals)))
    key = (kind, int(order))
    if key not in ANALYTIC_ASSEMBLY_ITERS:
        raise KeyError(f"块预处理刷新判据缺少解析装配 {key} 的标定值（refresh_cost.py 模块文档"
                       f"\"标定\"），请按同一方法实测后补入表中")
    return max(REFRESH_SLACK_MIN, ANALYTIC_ASSEMBLY_ITERS[key])
