"""AutoFlowCFD V2.0 - 求解器的线程配置：BLAS 运行时限线程、numba 线程池。

从 `core/fr_solver/solver.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）。
"""

import ctypes
import functools
import glob
import importlib.util
import os

import numba
from loguru import logger

# 进程里可能同时存在的 BLAS 实例：numpy 与 SciPy 的 wheel 各自带一份
# OpenBLAS（numba 在 nopython 里的 `np.dot`/`@` 走 SciPy 那份的 cython_blas）。
# 两份导出符号的前缀不同——numpy 的是 `openblas_*`（64 位整型接口带 `64_`
# 后缀），SciPy 1.13+ 的 `libscipy_openblas` 是 `scipy_openblas_*`。
_BLAS_PACKAGES = ("numpy", "scipy")
_OPENBLAS_SYMBOL_PREFIXES = ("scipy_openblas", "openblas")
_OPENBLAS_SYMBOL_SUFFIXES = ("64_", "")


def _package_lib_dirs(package: str) -> list:
    """`package` 的 wheel 可能放 OpenBLAS 动态库的目录（不 import 该包）。

    三种真实布局：`<site-packages>/<pkg>.libs/`（numpy 2.x / SciPy 的
    Windows 与 manylinux wheel）、`<pkg>/.libs/`、`<pkg>/libs/`（旧布局）。
    按**包名**拼目录而不是 glob `*.libs`：site-packages 里可能躺着 pip
    中断升级留下的 `~umpy.libs` 之类的残留副本，glob 会把这份没人用的
    DLL 加载进进程（2026-09-25 实测本机就有）。
    """
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return []
    pkg_dir = list(spec.submodule_search_locations)[0]
    site_dir = os.path.dirname(pkg_dir)
    return [os.path.join(site_dir, package + ".libs"),
            os.path.join(pkg_dir, ".libs"), os.path.join(pkg_dir, "libs")]


def _openblas_entry(lib):
    """库导出的 `(get_num_threads, set_num_threads)`；不是 OpenBLAS 时返回 None。"""
    for prefix in _OPENBLAS_SYMBOL_PREFIXES:
        for suffix in _OPENBLAS_SYMBOL_SUFFIXES:
            try:
                get = getattr(lib, f"{prefix}_get_num_threads{suffix}")
                set_ = getattr(lib, f"{prefix}_set_num_threads{suffix}")
            except AttributeError:
                continue
            get.argtypes, get.restype = [], ctypes.c_int
            set_.argtypes, set_.restype = [ctypes.c_int], None
            return get, set_
    return None


@functools.lru_cache(maxsize=1)
def _blas_thread_controls() -> tuple:
    """本项目用到的每个 BLAS 实例的 `(路径, get, set)`，进程内只探测一次。

    `CDLL` 一个已加载的路径拿到的是同一个模块句柄，所以 set 作用在
    **正在用的那个实例**上。只认 numpy/SciPy 自带的 OpenBLAS；两者都没有
    时再试 MKL（Intel 发行版 numpy）。一个都找不到返回空元组——此时限线程
    无从谈起、求解照常，`tests/unit/test_blas_thread_limit.py` 钉住本项目
    标准环境下必须找得到。
    """
    controls, seen = [], set()
    for package in _BLAS_PACKAGES:
        for lib_dir in _package_lib_dirs(package):
            for path in sorted(glob.glob(os.path.join(lib_dir, "*openblas*"))):
                if not (path.endswith((".dll", ".so", ".dylib")) or ".so." in path):
                    continue
                key = os.path.normcase(os.path.realpath(path))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    entry = _openblas_entry(ctypes.CDLL(path))
                except OSError:
                    continue
                if entry is not None:
                    controls.append((path,) + entry)
    if not controls:
        try:
            mkl = ctypes.CDLL("mkl_rt")
        except OSError:
            return ()
        get, set_ = mkl.MKL_Get_Max_Threads, mkl.MKL_Set_Num_Threads
        get.argtypes, get.restype = [], ctypes.c_int
        set_.argtypes, set_.restype = [ctypes.c_int], None
        controls.append(("mkl_rt", get, set_))
    return tuple(controls)


class blas_threads_limited:
    """把进程里**每个** BLAS 实例的线程数限制在 `n`（默认 1），退出时逐个
    恢复到进入前的值。

    **为什么求解阶段要限**：计算热点全是 numba `prange` kernel，由 numba
    自己的线程池管；OpenBLAS 同时开 cpu_count 个线程就是 2 倍超额订阅
    （实测数据见 `autoflowcfd/__init__.py` 顶部）。那里在 import numpy
    之前设的 `OPENBLAS_NUM_THREADS` 只在"先 import 本包"时生效，调用方
    先 import 过 numpy 时线程池早已建好，所以需要运行时这一道。

    **为什么必须是"有作用域"的**（2026-09-14 真实 bug）：线程数是进程级
    状态。网格几何（LAPACK 求逆得到的 `inv_jacs`）与 FR 算子构造的结果会随
    BLAS 线程数在最后一位上变化，离散 GCL / 自由流场保持性依赖这些度量量
    之间的精确抵消。第一版在 `FRSolver.__init__` 里永久限制，同进程构造的
    第二个求解器就落在 1 线程下——`tests/validation/test_couette.py` 整文件
    连跑时第三个用例必然失败。所以只在求解循环（`FRSolver.solve` /
    `run_order_continuation`）期间限制。

    **为什么恢复到"进入前的值"而不是 cpu_count**：进入前的值才是构造阶段
    实际用的那个；猜一个默认值会在用户显式设过线程数时把它改掉。

    **为什么每个实例都要限**（2026-09-25）：第一版找到第一个 OpenBLAS 就
    返回，SciPy 那份（`libscipy_openblas`，符号前缀不同）从未被限制——
    实测限制后 numpy 的 gemm 1.7 s（1 线程）、SciPy 的 dgemm 仍是 0.17 s
    （16 线程）。

    `AFCFD_NO_BLAS_THREAD_LIMIT=1` 时完全不动 BLAS（给自己管线程的用户/CI
    留的逃生门）。`n_applied` 是实际限制了的实例数。
    """

    def __init__(self, n: int = 1):
        self._n = int(n)
        self._saved = []

    @property
    def n_applied(self) -> int:
        return len(self._saved)

    def __enter__(self):
        self._saved = []
        if os.environ.get("AFCFD_NO_BLAS_THREAD_LIMIT") != "1":
            for _path, get, set_ in _blas_thread_controls():
                self._saved.append((set_, get()))
                set_(self._n)
        return self

    def __exit__(self, exc_type, exc, tb):
        for set_, previous in reversed(self._saved):
            set_(previous)
        self._saved = []
        return False


def configure_numba_threads(n_threads: int, mesh) -> int:
    """设置 numba 全局线程数（求解器生命周期内只设一次，必须在任何残差
    kernel 被调用之前）并做界面 kernel 私有缓冲区的内存提醒；返回实际生效
    的线程数。取值依据见函数体注释与 `FRSolver.__init__` 的 `n_threads` 文档。
    """
    # numba 全局线程数只在这里设置一次（求解器生命周期内不再修改），
    # 理由见本方法 n_threads 参数文档。必须在任何残差 kernel 被调用
    # 之前设置。-1 解析成 8，不是 os.cpu_count()。
    #
    # 默认值从 4 上调到 8（2026-09-13，用户反馈"每步耗时太长、且对
    # CPU 核数不敏感"后重新实测）：原来的 4 是在体积项/梯度链路还
    # 大量依赖 numpy 批量 matmul（**完全不随线程数并行**，见
    # fr_operators/volume_contract.py 模块文档的性能优化记录）时测出
    # 来的甜点——那时增加线程只会加剧内存带宽争用而没有任何可并行的
    # 新工作，所以 4 以上净倒退。这些链路改成 numba prange kernel 后
    # 重新在同一台 16 核机器、同一份 79 万单元真实网格 P1 状态上实测
    # （BLAS 线程已按上方 `blas_threads_limited` 限制为 1）：
    #   nt=4  inviscid 5.61s viscous 5.35s turb 9.88s -> 约 45.5s/步
    #   nt=8  inviscid 5.28s viscous 5.21s turb 9.59s -> 约 43.8s/步（最优）
    #   nt=12 inviscid 6.09s viscous 7.02s turb 9.40s -> 约 51.4s/步
    #   nt=16 inviscid 7.03s viscous 8.59s turb 9.39s -> 约 59.0s/步
    # 8 之后仍然净倒退，根因不再是"没有可并行的工作"，而是界面项
    # 按图着色**逐色串行调用** kernel（每色一次并行区+同步屏障，见
    # fr_residual/inviscid.py 界面项注释）：线程越多、每色分到的面
    # 越少，屏障与调度开销占比越高，加上本机是 P 核/E 核混合架构、
    # numba prange 是静态均分调度（最慢的 E 核决定每个屏障的时间），
    # 两者叠加。要真正吃满 16 核需要把 scatter 改成"逐面算通量 +
    # 逐单元 gather"的两趟无冲突结构（不需要着色、没有逐色屏障）；
    # 在现有着色结构下，8 是有实测数据支撑的最优默认值（显式 --threads 照常生效）。
    _DEFAULT_N_THREADS = 8
    resolved_n_threads = n_threads if n_threads > 0 else _DEFAULT_N_THREADS
    # 真实健壮性 bug 修复（2026-09-14）：`numba.set_num_threads(n)` 要求
    # n <= numba 线程池上限（`NUMBA_NUM_THREADS`，默认取 cpu_count，但
    # 用户/CI/作业调度器可以把它设成更小的值），否则直接抛
    # `ValueError: The number of threads must be between 1 and N`——
    # 求解器在**构造期**就崩溃，且报错完全看不出与这个环境变量有关。
    # 真实复现：跑对照实验时设了 `NUMBA_NUM_THREADS=6`，而这里的默认
    # 值是 8，两个进程都在构造 FRSolver 时直接异常退出。
    # 现在按线程池上限钳制并在被钳制时明确告知，而不是崩溃。
    _pool_max = int(getattr(numba.config, "NUMBA_NUM_THREADS", resolved_n_threads))
    if resolved_n_threads > _pool_max:
        logger.warning(
            f"n_threads={resolved_n_threads} 超过 numba 线程池上限 "
            f"{_pool_max}（由 NUMBA_NUM_THREADS 或 CPU 核数决定），"
            f"按上限钳制为 {_pool_max}"
        )
        resolved_n_threads = _pool_max
    resolved_n_threads = max(1, resolved_n_threads)
    numba.set_num_threads(resolved_n_threads)

    # 防御性内存检查：两个界面 kernel 各自的私有累加缓冲区峰值约
    # n_threads * n_cells * n_sps * 5 vars * 8 bytes（无粘/粘性两次
    # 调用不会同时存活，见 fr_residual_inviscid_kernel.py 模块文档
    # "多核并行"一节），超过系统总内存一半就提醒用户，不静默跑到
    # OOM。取不到总内存（非 Windows 平台没有对应 ctypes 调用）时
    # 直接跳过，不影响求解——这只是个提醒，不是硬性门禁。
    n_cells_est = getattr(mesh, 'n_cells', 0)
    n_sps_est = getattr(mesh, 'n_sps_per_cell', 8)
    buf_bytes = resolved_n_threads * n_cells_est * n_sps_est * 5 * 8
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_mem = stat.ullTotalPhys
        if total_mem > 0 and buf_bytes > 0.5 * total_mem:
            print(
                f"⚠️  警告：n_threads={resolved_n_threads} 下界面 kernel 私有累加缓冲区峰值约 "
                f"{buf_bytes / 1e9:.1f}GB，超过系统总内存（{total_mem / 1e9:.1f}GB）的一半，"
                f"叠加其他计算环节的内存占用可能导致 OOM。建议用更小的 n_threads。"
            )
    except Exception:
        pass
    return resolved_n_threads
