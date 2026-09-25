"""AutoFlowCFD V2.0 - 求解器的线程配置：BLAS 运行时限线程、numba 线程池。

从 `core/fr_solver/solver.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）。
"""

import os

import numba
from loguru import logger


def _limit_blas_threads(n: int = 1) -> bool:
    """把已加载的 OpenBLAS/MKL 线程数在**运行时**限制为 `n`（默认 1）。

    为什么需要运行时这一道（`autoflowcfd/__init__.py` 已经在 import numpy
    之前设过 `OPENBLAS_NUM_THREADS=1` 等环境变量）：那条路径只在
    "先 import autoflowcfd、再由它间接 import numpy" 时生效。如果调用方
    （交互式会话、第三方脚本、pytest 插件等）在导入本包之前就已经
    import 过 numpy，OpenBLAS 早已按 cpu_count 建好线程池，环境变量
    不再有任何作用——那正是与 numba 线程池 2 倍超额订阅、实测慢 9~11%
    的情形（数据见 `autoflowcfd/__init__.py` 顶部注释）。

    实现用 ctypes 直接调 OpenBLAS 导出的 `openblas_set_num_threads`
    （numpy 的 wheel 里是 64 位整型变体 `openblas_set_num_threads64_`）。
    找不到符号/不是 OpenBLAS 后端时静默跳过——这是纯性能调优，任何
    失败都不应影响求解本身。用户显式设过 `OPENBLAS_NUM_THREADS` 时
    同样跳过，尊重显式配置。

    Returns:
        True 表示确实调到了某个后端的 set_num_threads；False 表示没找到
        可用入口（静默跳过，不影响求解）。返回值供
        `tests/unit/test_blas_thread_limit.py` 断言"这条路径在当前环境里
        真的有效"——不要把它改成 `None`，否则那个自检就失去意义。
    """
    if os.environ.get("AFCFD_NO_BLAS_THREAD_LIMIT") == "1":
        return False
    try:
        import ctypes
        import glob
        import numpy as _np

        # numpy 自带的 BLAS 动态库已经在进程里（numpy import 时加载），
        # 重新 `CDLL` 同一个路径拿到的是同一个已加载模块的句柄，因此
        # 调用它导出的 set_num_threads 会作用在**正在用的那个实例**上。
        #
        # 搜索路径要覆盖三种真实的 wheel 布局（2026-09-13 真实踩坑：
        # 第一版只找了包内的 `.libs`/`libs`，而本机 numpy 2.x Windows
        # wheel 把 dll 放在 **site-packages/numpy.libs/**——numpy 包的
        # *同级*目录，于是 ctypes 路径静默失效、9~11% 的收益并没有真正
        # 拿到。用一个"限制前后测同一个大 gemm 耗时"的探针才发现，光看
        # 代码不会发现——详见本函数末尾的自检说明）：
        #   1) <site-packages>/numpy.libs/          （Windows wheel）
        #   2) <numpy>/.libs/、<numpy>/libs/        （旧布局/部分 Linux wheel）
        #   3) 系统安装的 libopenblas（Linux 发行版包管理器装的）
        np_dir = os.path.dirname(_np.__file__)
        site_dir = os.path.dirname(np_dir)
        patterns = [
            os.path.join(site_dir, "*.libs", "*openblas*"),
            os.path.join(np_dir, ".libs", "*openblas*"),
            os.path.join(np_dir, "libs", "*openblas*"),
        ]
        candidates = []
        for pat in patterns:
            candidates += [f for f in glob.glob(pat)
                           if f.endswith((".dll", ".so", ".dylib")) or ".so." in f]
        for lib_path in candidates:
            try:
                lib = ctypes.CDLL(lib_path)
            except OSError:
                continue
            # 64 位整型接口的 OpenBLAS（numpy 用的就是 openblas64）导出的是
            # 带 `64_` 后缀的符号名；两个都试，取到哪个用哪个。
            for sym in ("openblas_set_num_threads64_", "openblas_set_num_threads"):
                try:
                    fn = getattr(lib, sym)
                except AttributeError:
                    continue
                fn.argtypes = [ctypes.c_int]
                fn.restype = None
                fn(int(n))
                return True
        # MKL 后端（Intel 发行版 numpy）走另一个入口
        try:
            mkl = ctypes.CDLL("mkl_rt")
            mkl.MKL_Set_Num_Threads(ctypes.c_int(int(n)))
            return True
        except OSError:
            pass
        return False
    except Exception:
        # 纯性能调优，任何异常都不应影响求解
        return False


class blas_threads_limited:
    """把 BLAS 线程数限制在 `n`（默认 1）的上下文管理器，退出时恢复。

    **为什么必须是"有作用域"的，而不是构造时设一次就不管**（2026-09-14
    真实 bug 修复）：`_limit_blas_threads` 改的是**进程级**状态。第一版把
    它放在 `FRSolver.__init__` 里，于是一个进程里构造第二个求解器时，
    它的网格几何（LAPACK 求逆得到的 `inv_jacs`）与 FR 算子构造就落在
    "BLAS 只剩 1 线程"的环境下——而这两者的结果会随 BLAS/LAPACK 线程数
    在最后一位上变化，离散 GCL / 自由流场保持性依赖这些度量量之间的
    精确抵消（完整记录见 `autoflowcfd/__init__.py` 顶部）。真实后果：
    `tests/validation/test_couette.py` 单独跑每个用例都过，整文件连跑时
    第三个用例 `test_couette_prism_residual_trend` 必然失败（残差到最后
    一步仍在上升、从未回落）——因为它构造求解器时 BLAS 已被前面的用例
    永久限制成了 1。用 `AFCFD_NO_BLAS_THREAD_LIMIT=1` 关掉限制后整文件
    3 项全过，是这个因果链的决定性验证。

    现在只在**求解循环**（`FRSolver.solve` / `run_order_continuation`）
    期间限制：求解阶段 9~11% 的收益完整保留（那本来就是收益的来源），
    而任何构造/几何/算子生成阶段都仍然拿到多线程 BLAS，进程内前后
    构造的求解器因此得到逐位一致的度量量。
    """

    def __init__(self, n: int = 1):
        self._n = n
        self._applied = False

    def __enter__(self):
        self._applied = _limit_blas_threads(self._n)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._applied:
            # 恢复到包默认值（`autoflowcfd/__init__.py` 把 BLAS 环境变量
            # 设为 cpu_count）；用户显式设过 OPENBLAS_NUM_THREADS 时以它为准。
            import multiprocessing
            try:
                restore = int(os.environ.get("OPENBLAS_NUM_THREADS",
                                             multiprocessing.cpu_count()))
            except ValueError:
                restore = multiprocessing.cpu_count()
            _limit_blas_threads(max(1, restore))
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
    # （BLAS 线程已按下方 `_limit_blas_threads` 限制为 1）：
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
    # 逐单元 gather"的两趟无冲突结构（不需要着色、没有逐色屏障），
    # 是独立的架构改动，不在本次优化范围内；8 是当前实现下有实测
    # 数据支撑的最优默认值。
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
