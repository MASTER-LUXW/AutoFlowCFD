"""AutoFlowCFD V2.0 - numba 磁盘缓存按核源码版本隔离。

## 为什么需要（2026-09-25 真实事故）

numba 的 `cache=True` 只按**被装饰函数自身所在源文件**的时间戳判断缓存是否
失效。本项目大量核函数在编译期内联别的模块里的 `@njit` 函数（例如全部无粘
残差核都内联 `fr_operators/kernels.py::compute_ausm_up_flux`）——被内联的
函数改了，调用方的源文件没变，缓存**不会失效**，于是继续静默执行旧机器码。

实际踩到：修正 AUSM+up P5 分裂系数后，CPU 残差核仍从旧缓存加载旧通量，
GPU（numpy 替身）路径已用新通量，CPU/GPU 交叉验证当场不一致；换一个空缓存
目录后全部通过。这意味着在此之前任何"改了被内联的核函数、没换缓存目录"的
验证，都可能跑在旧代码上。

## 做法

对包内所有引用 numba 的源文件内容求一个哈希，把缓存放在
`<缓存根>/<哈希>` 下：任何一个核相关源文件改动都会换到新目录、强制全部
重编译（正确性优先于首次编译耗时）。缓存根不放在项目树里（项目约定不在
项目目录里产生输出目录）：用户显式设了 `NUMBA_CACHE_DIR` 时以它为根，否则
用系统的用户级缓存目录。
"""

import hashlib
import os


def kernel_source_digest(package_root: str) -> str:
    """包内所有引用 numba 的源文件（按相对路径排序）的内容哈希，16 位十六进制。"""
    h = hashlib.sha1()
    paths = []
    for dp, dn, fn in os.walk(package_root):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            if f.endswith(".py"):
                paths.append(os.path.join(dp, f))
    for p in paths:
        with open(p, "rb") as fh:
            data = fh.read()
        if b"numba" in data:
            h.update(os.path.relpath(p, package_root).replace(os.sep, "/").encode("utf-8"))
            h.update(b"\0")
            h.update(data)
    return h.hexdigest()[:16]


def _user_cache_root() -> str:
    base = (os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(base, "autoflowcfd", "numba_cache")


def configure_numba_cache_dir() -> str:
    """把 numba 缓存目录设为 `<根>/<核源码哈希>` 并返回它。

    必须在任何带 `@njit(cache=True)` 的模块导入之前调用（包 `__init__` 最前面）；
    numba 若已被导入，同时更新 `numba.config.CACHE_DIR`（dispatcher 在装饰时
    读取它）。
    """
    package_root = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("AFCFD_NUMBA_CACHE_ROOT") or os.environ.get("NUMBA_CACHE_DIR") or _user_cache_root()
    # 记住用户给的根，重复调用（或子进程继承环境）时不会层层嵌套哈希目录
    os.environ["AFCFD_NUMBA_CACHE_ROOT"] = root
    cache_dir = os.path.join(root, kernel_source_digest(package_root))
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    import sys

    if "numba" in sys.modules:
        sys.modules["numba"].config.CACHE_DIR = cache_dir
    return cache_dir
