"""时间推进配置（全部后端唯一来源）与时间精确的显式推进。

## 配置（`configure_time_stepping`）

六个构造点（CPU 单机、CPU 分布式两种加载模式、单 GPU、多 GPU 两种加载模式）共用：CFL 策略
（`adaptive_cfl/policy.py::build_cfl_policy`）、低马赫伪时间预处理开关
（`utils/preconditioning.py::resolve_low_mach_precond`）、时间精确显式模式。此前前两件事各写六份，单 GPU
不接受 `adaptive_cfl`（请求固定 CFL 被静默换成自适应）。

## 时间精确的显式推进（`time_accurate=True`，瞬态命令的 rk3 档）

显式格式的稳态用法是逐单元局部时间步（各单元推进的物理时间不同，只为尽快到达定常解）。时间精确时，每个
物理步 `dt` 切成若干子步，每个子步全场同一步长

    h = min(剩余时间, 全场显式稳定步长的全局最小值)

最后一个子步恰好落在 `t + dt`（各后端的局部步长函数在算出逐单元稳定步长后经 `uniform_time_step` 换成 h，
分布式取跨 rank 最小值；SEM 入口涡核按 h 推进，`sem_time_step`）。这样 `--dt` 仍是输出/统计用的物理
步长，不必为满足显式稳定条件而缩小。同时：

* CFL 固定为 `cfl_start`（自适应控制器按残差下降调 CFL，是稳态加速手段，瞬态里残差不是收敛量）；
* 低马赫伪时间预处理关闭：它推进的是 `dU/dt = -Gamma R`，改变声速并逐点混合守恒分量（周期域上质量/
  总能每 100 步漂移 2.5e-6，关闭后 5e-15，见 tests/validation/test_tgv.py）。

此前瞬态命令默认 rk3 走的是稳态的局部时间步、忽略 `--dt`，结果不是时间精确解；全域时间步只在验证算例里
靠替换求解器方法实现。
"""

import math

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme, scheme_from_name
from autoflowcfd.core.utils.array_module import array_module

#: 时间精确推进可用的格式（显式）；双时间步本身就是时间精确的
TIME_ACCURATE_SCHEMES = (TimeIntegrationScheme.FORWARD_EULER, TimeIntegrationScheme.SSP_RK2,
                         TimeIntegrationScheme.SSP_RK3)

#: 剩余时间相对物理步长小于它即认为已到达 t + dt（浮点舍入）
_ARRIVAL_TOL = 1e-12

#: 一个物理步内子步数的上限：超过说明显式稳定步长与物理步长差了六个数量级（网格或 CFL 设置有误）
_MAX_SUBSTEPS = 1_000_000


def configure_time_stepping(solver, time_scheme, *, time_accurate: bool = False, adaptive_cfl: bool = True,
                            cfl_start=None, cfl_max=None, cfl_min=None, low_mach_precond: bool = True) -> None:
    """设置 `solver._cfl_controller / fixed_cfl_number / low_mach_precond(_enabled) / time_accurate`。

    `low_mach_precond` 是请求值（写进 checkpoint，续算默认沿用），`low_mach_precond_enabled` 是按时间格式
    判定后的生效值（`resolve_low_mach_precond`）。checkpoint 记请求值：瞬态（DUAL_TIME）运行的生效值恒为
    False，若记生效值，从它切换到稳态格式续算时会把用户没有关过的预处理关掉。
    """
    from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
    from autoflowcfd.core.utils.preconditioning import resolve_low_mach_precond

    scheme = scheme_from_name(time_scheme)
    if time_accurate and scheme not in TIME_ACCURATE_SCHEMES:
        raise ValueError(f"时间精确的显式推进只用于显式格式 {[s.value for s in TIME_ACCURATE_SCHEMES]}，"
                         f"收到 {scheme.value}（双时间步本身就是时间精确的）")
    solver.time_accurate = bool(time_accurate)
    solver._ta_remaining = None
    solver._ta_h = None
    solver._cfl_controller, solver.fixed_cfl_number = build_cfl_policy(
        scheme, adaptive=adaptive_cfl and not time_accurate,
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min)
    solver.low_mach_precond = bool(low_mach_precond)
    solver.low_mach_precond_enabled = (not time_accurate) and resolve_low_mach_precond(low_mach_precond, scheme)


def uniform_time_step(solver, result, return_physical_too: bool, reduce_min=None, owned=None):
    """局部步长函数的出口：时间精确子步中把逐单元步长换成全场同一步长 h（并记下 h），否则原样返回。

    Args:
        result: 局部步长函数的返回值（`dt_mean` 或 `(dt_mean, dt_phys)`；时间精确模式下预处理关闭，二者相同）
        reduce_min: 分布式的跨 rank 最小值归约（标量 -> 标量），单进程为 None
        owned: 步长数组里本进程拥有的单元下标（多 GPU 的紧凑空间含 halo 单元，最小值只在拥有的单元上取，
            与单机同一个全场最小值）；None 为全部
    """
    remaining = getattr(solver, "_ta_remaining", None)
    if remaining is None:
        return result
    dt_mean = result[0] if return_physical_too else result
    if solver._ta_h is None:
        xp = array_module(dt_mean)
        stable = float(xp.min(dt_mean if owned is None else dt_mean[owned]))
        if reduce_min is not None:
            stable = float(reduce_min(stable))
        if not stable > 0.0:
            raise RuntimeError(f"全场显式稳定步长非正（{stable}），无法做时间精确推进")
        solver._ta_h = min(float(remaining), stable)
    xp = array_module(dt_mean)
    h = xp.full_like(dt_mean, solver._ta_h)
    return (h, h) if return_physical_too else h


def sem_time_step(solver, dt: float) -> float:
    """SEM 入口涡核本步推进的物理时间：时间精确子步取 h（须在本步局部步长算出之后调用），否则 `dt`。"""
    if getattr(solver, "_ta_remaining", None) is None:
        return dt
    if solver._ta_h is None:
        raise RuntimeError("时间精确子步的步长尚未确定（SEM 推进须在局部步长计算之后）")
    return solver._ta_h


def advance(solver, dt: float) -> float:
    """推进一个物理步 `dt`，返回最后一次 `step` 的残差范数。时间精确模式下切成若干全场同步长的子步。"""
    if not getattr(solver, "time_accurate", False):
        return solver.step(dt)
    dt = float(dt)
    if not dt > 0.0:
        raise ValueError(f"时间精确推进需要正的物理步长，收到 {dt}")
    remaining, res, n = dt, None, 0
    while remaining > _ARRIVAL_TOL * dt:
        if n >= _MAX_SUBSTEPS:
            raise RuntimeError(f"一个物理步 dt={dt:g} 需要超过 {_MAX_SUBSTEPS} 个显式子步，检查网格或 CFL")
        solver._ta_remaining, solver._ta_h = remaining, None
        try:
            res = solver.step(remaining)
        finally:
            solver._ta_remaining = None
        if solver._ta_h is None:
            raise RuntimeError("时间精确子步没有经过局部步长函数（后端未接入 uniform_time_step）")
        remaining -= solver._ta_h
        n += 1
        if res is not None and not math.isfinite(res):
            break
    solver.last_substeps = n
    return res
