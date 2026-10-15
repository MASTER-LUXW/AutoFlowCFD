# -*- coding: utf-8 -*-
"""GPU 显存查询按 CuPy 官方接口：默认显存池是顶层 `cupy.get_default_memory_pool()`，空闲显存是
`cupy.cuda.runtime.memGetInfo()`（numpy 替身 `_numpy_as_cupy.py` 与真实 CuPy 放在同一位置）。

2026-10-15 以前 `array_manager.py` 调的是不存在的 `cupy.cuda.get_default_memory_pool()`，三条 GPU 路径
（单 GPU、多 GPU、完全分布式）构造即 AttributeError；替身把显存池放在同一个错误位置上，测试一直通过。
"""

from tests.unit._numpy_as_cupy import NumpyAsCupy


def test_array_manager_uses_the_top_level_memory_pool(monkeypatch):
    import autoflowcfd.core.gpu as core_gpu
    import autoflowcfd.core.gpu.array_manager as am
    from tests.unit._gpu_cupy_shim import patch_module_get_cupy
    from tests.unit._patch_pkg import patch_pkg_attr

    shim = NumpyAsCupy()
    patch_module_get_cupy(monkeypatch, [core_gpu, am], shim)
    patch_pkg_attr(monkeypatch, [core_gpu, am], "gpu_available", True)
    mgr = am.GPUArrayManager(device_id=0)
    assert mgr._mem_pool is shim.get_default_memory_pool()
    assert not hasattr(shim.cuda, "get_default_memory_pool")


def test_device_available_memory_counts_free_device_memory_and_pooled_blocks():
    from autoflowcfd.core.utils.system_memory import available_bytes

    shim = NumpyAsCupy()
    free, total = shim.cuda.runtime.memGetInfo()
    assert available_bytes(shim) == free + shim.get_default_memory_pool().free_bytes()
    assert 0 < available_bytes(shim) <= total
