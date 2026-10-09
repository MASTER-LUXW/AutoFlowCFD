"""numpy 充当 CuPy 的测试替身（全部 GPU 单元测试共用的唯一实现）。

GPU 生产函数经 `get_cupy()` 拿张量库；把它换成这个替身，同一份 GPU 代码就在没有 CUDA 设备的机器上用 numpy 执行
——跑的是生产函数本身，不是另写一份参考实现。替身只补 numpy 没有的 CuPy 接口：`asnumpy`、`scatter_add`
（CuPy 的 `cupyx.scatter_add` 语义，等价于 `np.add.at`）与 `cuda` 子模块里的设备、流、显存池空实现。
"""

import numpy as np


class NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)

    def scatter_add(self, a, indices, b):
        np.add.at(a, indices, b)

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
        def get_default_memory_pool():
            class _Pool:
                def used_bytes(self):
                    return 0

                def free_all_blocks(self):
                    pass
            return _Pool()
