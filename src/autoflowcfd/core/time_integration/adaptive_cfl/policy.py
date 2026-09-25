"""AutoFlowCFD V2.0 - CFL 策略的唯一事实来源（全部后端共用）。

两件事：

1. `build_cfl_policy`：构造时决定"用自适应控制器还是固定 CFL"，以及固定
   CFL 取多少；
2. `current_cfl_number`：步长计算时取当前 CFL（控制器 > 固定值 > 替身回退）。

## 为什么要收到一处（2026-09-25）

此前 6 个构造点（CPU 单机、CPU-MPI 两种加载模式、单机 GPU、多 GPU 两种
加载模式）各写一份，而且已经分叉：

* 控制器启用条件：CPU 单机"除 DUAL_TIME 外都启用"；CPU-MPI 与多 GPU
  "只有 SSP-RK2/RK3"；单机 GPU "RK2/RK3/前向 Euler"；
* 没有控制器时的 CFL：CPU 单机取请求的 `cfl_start`（默认 0.03）；CPU-MPI
  走 `cfl.py` 的替身回退 0.1；单机/多 GPU 取构造参数 `cfl`（默认 1.0）。
  于是同一个双时间步算例的伪时间步长在三类后端上差 30 倍，其中 1.0 远超
  P>=1 的显式稳定极限（~0.1 量级）；
* 单机 GPU 与多 GPU"传统模式"还保留着 `cfl_start 缺省退回 cfl`、
  `cfl_max 缺省退回 max(cfl, 0.5)` 两个硬编码兜底 —— 2026-09-17 在 CPU
  两条路径上已经因为同样的兜底让分布式与单机脱节而删掉过一次。

规则取 CPU 单机那一份（自 2026-08-24 起验证最多、有专门回归测试）。
"""

import inspect

from loguru import logger

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme, scheme_from_name

from .controller import AdaptiveCFLController

#: 替身对象（既无控制器也无 `fixed_cfl_number`）的回退 CFL。只为诊断脚本 /
#: 测试替身保留，真实求解器走不到（构造时 `build_cfl_policy` 恰好给出二者之一）。
_STANDIN_FALLBACK_CFL = 0.1


def _controller_default(name: str) -> float:
    """控制器构造参数的默认值 —— 默认值的唯一事实来源是控制器签名本身。"""
    return inspect.signature(AdaptiveCFLController.__init__).parameters[name].default


def build_cfl_policy(time_scheme, adaptive: bool = True, cfl_start=None,
                     cfl_max=None, cfl_min=None):
    """返回 `(controller, fixed_cfl_number)`，二者恰好一个不是 None。

    * DUAL_TIME：外层步长是物理时间步，伪时间内层有自己的接受/拒绝与步长
      调节（`time_integration/dual.py`），外层控制器不启用；伪时间步长用
      固定 CFL；
    * `adaptive=False`：固定 CFL（稳定边界扫描、A/B 对照都靠它）；
    * 其余（SSP-RK、前向 Euler、IMEX、Newton-Krylov）：自适应控制器。

    固定 CFL 取 `cfl_start`（关掉自适应时"初始值"就是全程唯一的值）；未给
    时取控制器 `cfl_start` 的默认值。`None` 的参数一律不传给控制器，默认值
    只有控制器签名这一个来源。
    """
    scheme = scheme_from_name(time_scheme)
    if adaptive and scheme != TimeIntegrationScheme.DUAL_TIME:
        kw = {k: v for k, v in (("cfl_start", cfl_start), ("cfl_max", cfl_max),
                                ("cfl_min", cfl_min)) if v is not None}
        return AdaptiveCFLController(**kw), None
    fixed = cfl_start if cfl_start is not None else _controller_default("cfl_start")
    return None, float(fixed)


def describe_cfl_policy(controller, fixed_cfl_number) -> str:
    """启动日志用的一行描述（影响数值的开关必须在日志里可见）。"""
    if controller is not None:
        return (f"Adaptive CFL: enabled (start={controller.cfl_start}, "
                f"max={controller.cfl_max}, min={controller.cfl_min})")
    return f"Adaptive CFL: disabled (fixed CFL = {fixed_cfl_number:g})"


def current_cfl_number(solver) -> float:
    """当前 CFL：控制器 > `solver.fixed_cfl_number` > 替身回退（打一次警告）。"""
    controller = getattr(solver, "_cfl_controller", None)
    if controller is not None:
        return controller.cfl_number
    fixed = getattr(solver, "fixed_cfl_number", None)
    if fixed is not None:
        return float(fixed)
    if not getattr(solver, "_afcfd_cfl_fallback_warned", False):
        logger.warning(
            "[CFL] 求解器既没有自适应控制器也没有 fixed_cfl_number，回退到 "
            f"{_STANDIN_FALLBACK_CFL}。真实求解器不会走到这一档（构造时 "
            "build_cfl_policy 恰好给出二者之一），走到这里说明调用方是个替身"
            "对象——若它本意是固定 CFL，请显式设 `solver.fixed_cfl_number`，"
            f"否则这个 {_STANDIN_FALLBACK_CFL} 与你请求的值无关。")
        try:
            solver._afcfd_cfl_fallback_warned = True
        except AttributeError:
            pass
    return _STANDIN_FALLBACK_CFL
