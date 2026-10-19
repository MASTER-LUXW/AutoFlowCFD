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

from autoflowcfd.core.time_integration.base import IMPLICIT_SCHEMES, TimeIntegrationScheme, scheme_from_name

from .controller import AdaptiveCFLController
from .ser import SERCFLController

#: 替身对象（既无控制器也无 `fixed_cfl_number`）的回退 CFL。只为诊断脚本 /
#: 测试替身保留，真实求解器走不到（构造时 `build_cfl_policy` 恰好给出二者之一）。
_STANDIN_FALLBACK_CFL = 0.1


#: 双时间步内迭代的伪时间 CFL 起点（每个物理步重新从它起步）。内迭代解的系统对角上已有物理时间项
#: `a0/dt`，伪时间项只是 Newton 被拒时的回退手段：起点取到让它在绝大多数单元上可忽略（最小单元的显式
#: 步长约 1e-7 s，乘 1e4 即 1e-3 s，与常用的物理时间步同量级或更大），不从稳态冷启动用的 5 起步。
DUAL_TIME_PSEUDO_CFL_START = 1.0e4


def controller_class_for(time_scheme):
    """该时间格式用哪一类 CFL 律：Newton-Krylov 稳态与双时间步（内迭代是同一个 Newton-Krylov 步，
    `implicit/newton/physical_time.py`）用 SER（`ser.py` 模块文档写了为什么显式那套控制器在隐式上是错的），
    其余用显式控制器。"""
    if scheme_from_name(time_scheme) in IMPLICIT_SCHEMES:
        return SERCFLController
    return AdaptiveCFLController


def controller_default(name: str, time_scheme) -> float:
    """该格式所用控制器的构造参数默认值 —— 控制器签名，双时间步的伪时间 CFL 起点除外
    （`DUAL_TIME_PSEUDO_CFL_START`）。"""
    if name == "cfl_start" and scheme_from_name(time_scheme) == TimeIntegrationScheme.DUAL_TIME:
        return DUAL_TIME_PSEUDO_CFL_START
    return inspect.signature(controller_class_for(time_scheme).__init__).parameters[name].default


def build_cfl_policy(time_scheme, adaptive: bool = True, cfl_start=None,
                     cfl_max=None, cfl_min=None):
    """返回 `(controller, fixed_cfl_number)`，二者恰好一个不是 None。

    * `adaptive=False`：固定 CFL（稳定边界扫描、A/B 对照都靠它）；
    * Newton-Krylov 稳态：SER 律（`ser.py`）；
    * DUAL_TIME：内迭代的伪时间 CFL 同样用 SER，起点 `DUAL_TIME_PSEUDO_CFL_START`、每个物理步重新起步
      （2026-10-10 以前内迭代是显式的，伪时间步长用固定 CFL 0.03）；
    * 其余（SSP-RK、前向 Euler）：显式自适应控制器（时间精确模式固定 CFL，见 time_accurate.py）。

    固定 CFL 取 `cfl_start`（关掉自适应时"初始值"就是全程唯一的值）；未给
    时取该格式所用控制器 `cfl_start` 的默认值。`None` 的参数一律不传给
    控制器，默认值只有控制器签名这一个来源——显式与隐式的默认值差两个
    数量级（0.03 vs 5），所以 CLI 层不能再写死任何一个。
    """
    scheme = scheme_from_name(time_scheme)
    if adaptive:
        kw = {k: v for k, v in (("cfl_start", cfl_start), ("cfl_max", cfl_max),
                                ("cfl_min", cfl_min)) if v is not None}
        if scheme == TimeIntegrationScheme.DUAL_TIME:
            kw.setdefault("cfl_start", DUAL_TIME_PSEUDO_CFL_START)
        return controller_class_for(scheme)(**kw), None
    fixed = cfl_start if cfl_start is not None else controller_default("cfl_start", scheme)
    return None, float(fixed)


def describe_cfl_policy(controller, fixed_cfl_number) -> str:
    """启动日志用的一行描述（影响数值的开关必须在日志里可见）。"""
    if controller is not None:
        law = "SER" if isinstance(controller, SERCFLController) else "explicit"
        return (f"Adaptive CFL: enabled [{law}] (start={controller.cfl_start}, "
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
