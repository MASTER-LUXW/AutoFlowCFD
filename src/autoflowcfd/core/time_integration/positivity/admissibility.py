"""AutoFlowCFD V2.0 - 无网格信息时的可容许性检查：只检查、不修改。

时间积分器也会被脱离网格单独使用（单元测试里直接对抽象数组推进）。那种
场景拿不到单元结构，做不了守恒的正性限制；此前的做法是逐点硬钳（把密度钳到
1e-6、压力钳到 1 Pa、速度钳到 1e4 m/s），它**不守恒**，而且在真实网格上正是
把一个解点的负密度一步放大成 1e53 的那个环节（plate_demo 长程运行第 4134
步）。所以这里不再修改任何值：状态可容许就原样放行，不可容许就明确报错。
"""

from autoflowcfd.core.fr_operators.flux_kernels import GAMMA
from autoflowcfd.core.utils.array_module import array_module


def assert_admissible(U_flat, gamma=GAMMA):
    """`U_flat` `(n, n_vars)` 的每个点都必须密度 > 0、压力 > 0、有限。

    NumPy / CuPy 数组都接受（按 `array_module` 分派，CPU 与 GPU 积分器共用）。
    """
    xp = array_module(U_flat)
    rho = U_flat[:, 0]
    ke = 0.5 * (U_flat[:, 1] ** 2 + U_flat[:, 2] ** 2 + U_flat[:, 3] ** 2) / xp.where(rho > 0, rho, 1.0)
    p = (gamma - 1.0) * (U_flat[:, 4] - ke)
    bad = ~((rho > 0) & (p > 0) & xp.isfinite(rho) & xp.isfinite(p))
    if bool(bad.any()):
        from autoflowcfd.core.fr_solver.residual_diagnostics import SolverDivergedError

        idx = [int(i) for i in xp.where(bad)[0][:8].tolist()]
        raise SolverDivergedError(
            f"{int(bad.sum())} 个点的状态不可容许（密度或压力 <= 0 或非有限），"
            f"例如扁平索引 {idx}。本调用没有提供正性保持限制器（需要网格的单元"
            f"结构才能守恒地限制），不再用逐点硬钳掩盖它。")
    return U_flat
