"""AutoFlowCFD V2.0 - 自适应 CFL 控制器的字段类型声明

`AdaptiveCFLController` 的参数与运行状态字段（只有类型声明，值由 `controller.py` 的 `__init__` / `reset`
赋值；各字段含义见那里的参数说明）。两个方法 mixin（`gates.py`、`update.py`）继承它，类型检查才能在 mixin
内部看到这些字段。
"""

from __future__ import annotations

from typing import Deque, List, Optional, Tuple


class _CFLState:
    """自适应 CFL 控制器的字段（仅类型声明，见模块文档）"""

    # ---- 参数
    cfl_start: float
    cfl_max: float
    cfl_min: float
    growth_factor: float
    shrink_factor: float
    growth_threshold: float
    shrink_threshold: float
    growth_confirm_steps: int
    cooldown_steps: int
    ramp_steps: int
    crawl_threshold: float
    crawl_factor: float
    crawl_confirm_steps: int
    mild_shrink_factor: float
    trend_window: int
    trend_threshold: float
    trend_factor: float
    mild_shrink_confirm_steps: int
    trend_shrink_threshold: float
    ceiling_backoff: float
    ceiling_release_steps: int
    grow_block_ratio: float
    mild_shrink_block_ratio: float
    fixed_cfl: bool

    # ---- 运行状态
    cfl_number: float
    _prev_residual: float
    _step_count: int
    _consecutive_good: int
    _consecutive_crawl: int
    _consecutive_mild_bad: int
    _cfl_ceiling: Optional[float]
    _steps_since_shrink: int
    _res_best: Optional[float]
    _res_at_ceiling: float
    _steps_since_last_change: int
    _history: List[Tuple[int, float, float]]
    _res_window: Deque[float]
