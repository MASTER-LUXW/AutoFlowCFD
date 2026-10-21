"""
AutoFlowCFD V2.0 - GPU 版时间积分

* `GPUTimeIntegrator`：**不是**独立实现。SSP-RK stage 推进、双时间步、
  全部继承 CPU 的 `time_integration.TimeIntegrator`（它们与数组模块
  无关，CuPy 数组原样走同一份代码）。
局部 CFL 步长与 CPU 同一个函数（`core/fr_solver/cfl_limits.py::local_time_step_limits`）；2026-10-22（I-115）
以前这里有一份逐解点调用的 GPU 重写。

GPU 与 CPU 走同一份积分器、同一个守恒限制器（`time_integration/positivity/`，GPU 上用它的数组模块无关实现），没有 GPU 独立拷贝。
"""

from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS, TimeIntegrator, scheme_from_name


class GPUTimeIntegrator(TimeIntegrator):
    """GPU 后端的时间积分器 —— 数值本体全部继承 `TimeIntegrator`。

    与父类只差一点：`scheme` 接受用户侧字符串（`scheme_from_name` 解析，非法
    取值报错，不静默退回前向 Euler —— 此前 `_SCHEME_TABLE.get(scheme, _EULER)`
    正是那样）。CFL 不归积分器管：控制器 / 固定 CFL 由
    `adaptive_cfl/policy.py::build_cfl_policy` 在求解器上建立（此前这里的
    `cfl=1.0` 是双时间步下的实际 CFL，远超 P>=1 显式稳定极限）。

    `dual_time_steps` 默认值与 CPU 同一个（父类构造参数）。
    """

    def __init__(self, scheme="ssp_rk3", dual_time_steps: int = DEFAULT_DUAL_TIME_STEPS):
        super().__init__(scheme=scheme_from_name(scheme), dual_time_steps=dual_time_steps)
