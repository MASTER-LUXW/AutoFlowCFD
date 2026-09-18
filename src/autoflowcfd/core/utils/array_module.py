"""AutoFlowCFD V2.0 - NumPy / CuPy 数组模块分派（**唯一**事实来源）。

为什么需要它：本项目有一批**后端无关的纯数组内核**（troubled-cell 判据、
过积分的度量收缩等），它们按"取数组所属模块、再调该模块的同名函数"的
方式写，于是 CPU 与 GPU 共用同一份实现。这比给 GPU 另抄一份要安全得多
——本项目已经反复因为"同一件事有两份实现、只改了一份"出真实缺陷（见
项目记忆 `feedback-prefer-deleting-redundant-code`）。

为什么单独成一个模块：这个判断本身曾经有**两份**实现，且判据不同——
`core/gpu/gpu_overintegration.py` 按 `type(arr).__module__` 前缀判断，
`core/fr_operators/bounds_sensor.py` 按有没有 `__cuda_array_interface__`
判断。两者在 CuPy 与 NumPy 上结论一致，但对第三方数组（例如 numba 的
`DeviceNDArray`——它有 `__cuda_array_interface__` 却**不是** CuPy 数组，
`cupy.ndarray` 的方法在它上面不存在）结论相反：前者返回 NumPy（会在
后续运算里报错，能看见），后者返回 CuPy（会拿 CuPy 函数去操作一个
非 CuPy 对象，行为不可预期）。所以这里统一取**模块前缀**判据，并显式
拒绝"有 CUDA 接口但不是 CuPy"的对象，而不是默默按 NumPy 处理。

不能无条件 `import cupy`：本机与 CI 都没有 CuPy，GPU 模块按"有则用、
无则跳过测试"的方式组织（见 `core/gpu/__init__.py`）。
"""

from typing import Any


def array_module(*arrays: Any):
    """返回这些数组所属的数组模块（`cupy` 或 `numpy`）。

    只要有**任意一个**入参是 CuPy 数组就返回 `cupy`——混合调用的典型
    形态是"场在设备上、面连接索引还在主机上"，此时应当按设备侧处理，
    由内核自己 `xp.asarray` 把主机侧的小数组搬上去。

    Args:
        *arrays: 待判定的数组（`None` 被忽略，便于直接把可选参数传进来）

    Returns:
        `cupy` 模块（任一入参是 CuPy 数组时）或 `numpy` 模块。

    Raises:
        TypeError: 某个入参暴露了 `__cuda_array_interface__` 但不是 CuPy
            数组（典型是 numba 的 `DeviceNDArray`）。不静默按 NumPy 处理
            ——那会把一个设备指针当主机数组读，是难以定位的错误；也不
            当成 CuPy——CuPy 的方法在它上面并不存在。调用方应当先用
            `cupy.asarray` 做零拷贝转换。
    """
    saw_cupy = False
    for a in arrays:
        if a is None:
            continue
        mod = type(a).__module__ or ""
        if mod.startswith("cupy"):
            saw_cupy = True
        elif hasattr(a, "__cuda_array_interface__"):
            raise TypeError(
                f"{type(a).__module__}.{type(a).__name__} 暴露了 "
                f"__cuda_array_interface__ 但不是 CuPy 数组。后端无关的"
                f"数组内核按 CuPy/NumPy 的同名函数写，无法直接操作它——"
                f"请先 cupy.asarray()（对实现了该接口的对象是零拷贝）。"
            )
    if saw_cupy:
        import cupy

        return cupy
    import numpy

    return numpy
