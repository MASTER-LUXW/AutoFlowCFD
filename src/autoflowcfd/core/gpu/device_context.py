"""AutoFlowCFD V2.0 - GPU 设备上下文与"有 CuPy 就上设备、没有就留在 numpy"。

## 为什么需要这一份

GPU 侧的初始化代码里到处是这个模式：

    cp = get_cupy()
    with cp.cuda.Device(self.device_id):
        x = cp.asarray(x)

它有两个问题：

1. **`get_cupy()` 在没有 CuPy 时返回 `None`**，于是 `cp.cuda` 直接抛
   `AttributeError: 'NoneType' object has no attribute 'cuda'` —— 一个
   和真实原因毫无关系的报错。本项目确实会在没有 CuPy 的机器上构造 GPU
   求解器：多 GPU 分布式的端到端测试就是用 numpy 替身跑通的（本机与 CI
   都没有 CUDA 设备）。
2. 这个模式此前被一个 `except Exception -> warning` 兜底包着，于是上面
   那个 AttributeError 连同**所有真正的硬护栏**一起被吞成一条 warning。
   兜底已于 2026-09-18 删除，这个模块是它留下的缺口的正解：不是把兜底
   加回来，而是让"没有 CuPy"成为一条**明确的、数值上等价的** numpy 路径。

**为什么退回 numpy 而不是跳过**：跳过会让"替身测试通过"与"真实 GPU 上
这段逻辑真的生效"脱钩——那正是本项目反复吃过亏的那类缺陷。退回 numpy
则让替身测试走的是同一条代码路径、同一套判据，只是数组模块不同。
"""

from contextlib import nullcontext
from typing import Any, Callable, Tuple

import numpy as np

from . import get_cupy

__all__ = ["device_transfer", "ascontiguous_like"]


def device_transfer(device_id: int) -> Tuple[Any, Callable[[Any], Any]]:
    """返回 `(device_ctx, to_device)`。

    有 CuPy 时：`device_ctx` 是 `cp.cuda.Device(device_id)`，`to_device`
    把数组搬到该设备（自带设备上下文，所以它在 `with device_ctx` 之外
    调用同样正确 —— 惰性构造的边界表就是那样用的）。

    没有 CuPy 时：`device_ctx` 是 `nullcontext()`，`to_device` 是
    `np.asarray`。

    Args:
        device_id: CUDA 设备号（无 CuPy 时忽略）

    Returns:
        `(device_ctx, to_device)`。`device_ctx` 可以被 `with` 多次进入
        （CuPy 的 `Device` 对象支持重入）。
    """
    cp = get_cupy()
    if cp is None:
        return nullcontext(), np.asarray

    dev = cp.cuda.Device(device_id)

    def to_device(a):
        with cp.cuda.Device(device_id):
            return cp.asarray(a)

    return dev, to_device


def ascontiguous_like(a):
    """`ascontiguousarray`，按 `a` 自己的数组模块分派。

    比 `cp.ascontiguousarray(a)` 安全：后者在 `a` 是 numpy 数组时会把它
    隐式搬上设备（或在没有 CuPy 时直接崩）。
    """
    cp = get_cupy()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.ascontiguousarray(a)
    return np.ascontiguousarray(a)
