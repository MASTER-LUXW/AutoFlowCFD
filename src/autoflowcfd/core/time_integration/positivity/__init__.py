"""AutoFlowCFD V2.0 - 守恒的正性保持（Zhang–Shu 型限制器）。

## 取代了什么、为什么（2026-09-24）

此前每个 RK stage 之后调用的 `enforce_positivity` 是**逐点硬钳**：

    rho = max(rho, 1e-6); vel = m / rho; |vel| <= 1e4; p >= 1 Pa

它有三个真实缺陷：

1. **不守恒**：钳制凭空改变质量、动量、能量；
2. **是一步爆炸的放大器**：plate_demo 长程运行（P1+SST）里一个解点的密度
   过零后被钳到 1e-6，速度 = 动量/1e-6 被限到 1e4、动量写回 0.01；下一个
   stage 把多项式外插到通量点时再次出现负密度（通量点上没有钳制），一步放大
   到 1e53、随即 NaN（第 4134 步，插桩确定性复现）；
3. **带量纲的下限**：`p_floor = 1 Pa` 按大气压标定，无量纲算例的物理压力会被
   直接钳死 —— 等熵涡验证算例因此不得不改写成国际单位制。

## 现在的做法

* 有网格时（全部求解器）：`limiter.PositivityLimiter`，向**守恒的单元均值**
  收缩，使全部真实解点与面通量点上密度、压力严格为正；单元均值不变（到舍入）；
  正常运行里从不触发，结果与"没有限制器"逐位相同。单元均值本身不可容许时
  报 `SolverDivergedError`（那是真正的发散，不修）。
* 无网格时（积分器被单独使用）：`admissibility.assert_admissible`，只检查、
  不修改。

数值核心与公式见 `zhang_shu.py` 模块文档。
"""

from .admissibility import assert_admissible  # noqa: F401
from .limiter import (  # noqa: F401
    PositivityLimiter,
    build_positivity_limiter,
    build_positivity_limiter_from_arrays,
    get_positivity_limiter,
)

__all__ = [
    "PositivityLimiter",
    "assert_admissible",
    "build_positivity_limiter",
    "build_positivity_limiter_from_arrays",
    "get_positivity_limiter",
]
