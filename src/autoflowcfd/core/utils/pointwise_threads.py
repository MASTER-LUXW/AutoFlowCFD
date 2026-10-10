"""AutoFlowCFD V2.0 - 逐点 numpy 函数按单元分块的多线程求值。

## 为什么需要

湍流模型的逐点函数（SA 的源项、梯度项、扩散系数等，`turbulence/sa/pointwise.py`）只有一份定义，
numpy 与 cupy 共用。numpy 下它们是单线程的整场数组表达式：cube_demo（37.4 万单元，P1）上一次耦合残差
求值 3.57 s，其中 SA 一个标量方程占 1.83 s（五个平均流方程 1.45 s，核是 numba 并行的），仅
`sa_source_terms` 一个函数 0.45 s。

## 做法

不另写一份 numba 核（那是第二个事实来源）。逐点函数对第 0 轴（单元）逐行独立，把输入沿单元切成
线程数那么多块，在线程池里对每块调用**同一个函数**，再按原顺序拼回。numpy 的 ufunc 在执行期间释放
GIL，线程之间真正并行；每个元素经历的运算与整场调用完全相同，结果逐位一致。

`np.errstate` 是线程局部的，工作线程按调用方当时的设置求值（否则调用方屏蔽掉的溢出警告会在工作线程里
重新出现）。单元数少于 `MIN_CELLS_PER_CHUNK` 的两倍、只有一个线程、或数组不在 numpy 上时直接调用。
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import numpy as np

#: 每块至少这么多单元才值得分块（更小时线程调度开销超过收益）
MIN_CELLS_PER_CHUNK = 4096

_pool = None
_pool_size = 0


def _executor(n_threads: int) -> ThreadPoolExecutor:
    """按当前线程数复用的线程池（线程数变了就重建）。"""
    global _pool, _pool_size
    if _pool is None or _pool_size != n_threads:
        if _pool is not None:
            _pool.shutdown(wait=True)
        _pool = ThreadPoolExecutor(max_workers=n_threads, thread_name_prefix="afcfd-pointwise")
        _pool_size = n_threads
    return _pool


def _call(func: Callable, err: dict, chunks, args):
    with np.errstate(**err):
        return func(*chunks, *args)


def map_cell_chunks(func: Callable, arrays: Sequence, *args, xp=np):
    """`func(*arrays, *args)`，`arrays` 沿第 0 轴（单元）分块并行求值。

    Args:
        func: 对第 0 轴逐行独立的函数；返回一个数组或数组的元组，第 0 轴与输入对齐。
        arrays: 第 0 轴长度相同的数组（每块取同一段切片）。
        args: 原样传给每次调用的其余参数（标量、数组模块等）。
        xp: `arrays` 所在的数组模块；不是 numpy 时直接调用（设备上的数组表达式本身就是并行的）。
    """
    if xp is not np:
        return func(*arrays, *args)
    import numba

    n_cells = int(arrays[0].shape[0])
    n_chunks = min(int(numba.get_num_threads()), n_cells // MIN_CELLS_PER_CHUNK)
    if n_chunks < 2:
        return func(*arrays, *args)
    bounds = np.linspace(0, n_cells, n_chunks + 1).astype(np.int64)
    err = np.geterr()
    futures = [_executor(n_chunks).submit(_call, func, err, [a[lo:hi] for a in arrays], args)
               for lo, hi in zip(bounds[:-1], bounds[1:])]
    parts = [f.result() for f in futures]
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate([p[k] for p in parts], axis=0) for k in range(len(parts[0])))
    return np.concatenate(parts, axis=0)


__all__ = ["map_cell_chunks", "MIN_CELLS_PER_CHUNK"]
