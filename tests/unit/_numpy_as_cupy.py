"""numpy 充当 CuPy 的测试替身（全部 GPU 单元测试共用的唯一实现）。

GPU 生产函数经 `get_cupy()` 拿张量库；把它换成这个替身，同一份 GPU 代码就在没有 CUDA 设备的机器上用 numpy 执行
——跑的是生产函数本身，不是另写一份参考实现。替身只补 numpy 没有的 CuPy 接口：`asnumpy`、`scatter_add`
（CuPy 的 `cupyx.scatter_add` 语义，等价于 `np.add.at`）、顶层的显存池（`cupy.get_default_memory_pool`）与 `cuda`
子模块里的设备、流、运行时查询。每个接口都放在真实 CuPy 里的同一位置：替身放错位置会让生产代码里同样的错误
在测试中通过（2026-10-15 以前显存池放在 `cuda` 下面，`array_manager.py` 的同一个错误调用一直没有暴露）。
"""

import numpy as np


class _MemoryPool:
    """`cupy.cuda.MemoryPool` 用到的查询与释放接口（替身不缓存显存块）。"""

    def used_bytes(self):
        return 0

    def free_bytes(self):
        return 0

    def total_bytes(self):
        return 0

    def free_all_blocks(self):
        pass


_POOL = _MemoryPool()


class NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x, out=None):
        if out is None:
            return np.asarray(x)
        out[...] = x
        return out

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

    @staticmethod
    def get_default_memory_pool():
        return _POOL

    class cuda:
        class Device:
            def __init__(self, device_id=0):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class runtime:
            @staticmethod
            def getDeviceCount():
                return 1

            @staticmethod
            def getDeviceProperties(device_id):
                return {'name': b'FakeGPU', 'totalGlobalMem': 8 * 1024 ** 3}

            @staticmethod
            def memGetInfo():
                """`(空闲, 总量)` 字节，与 `getDeviceProperties` 的 8 GiB 一致。"""
                return 6 * 1024 ** 3, 8 * 1024 ** 3

        class Stream:
            def __init__(self, non_blocking=True):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def synchronize(self):
                pass

        @staticmethod
        def get_current_stream():
            return NumpyAsCupy.cuda.Stream()
