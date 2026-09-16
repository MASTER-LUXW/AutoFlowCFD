"""
AutoFlowCFD V2.0 - FR 求解器主类 (Final Integration)

本模块整合 FRState, HighOrderMesh, FR Kernels, Turbulence Models 及 Weak BCs。
它是 V2.0 求解器的总控中心，负责协调各模块完成 N-S 方程的高阶离散与求解。

核心功能:
1. 支持多种湍流模型（SST/DDES/IDDES/WMLES/LES）
2. 自动切换RANS/LES模式
3. 完整的时间推进循环
4. 残差监控和收敛判断
"""

import os
import numpy as np
import numba
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from autoflowcfd.core.fr_solver.state import FRState, SolverResult
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.fr.operators import generate_fr_operators, FROperators
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
from autoflowcfd.core.time_integration.base import TimeIntegrator, TimeIntegrationScheme
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual as compute_viscous_residual_ldg

# 导入辅助模块
from autoflowcfd.core.utils import solver_helpers
from autoflowcfd.core.utils import order_continuation
from . import turbulence as fr_solver_turbulence
from . import boundary as fr_solver_boundary
from . import step as fr_solver_step
from .residual_diagnostics import check_residual_finite
from .solver_geometry import _SolverGeometryMixin

# AUSM+up Weiss-Smith 预处理参考马赫数的物理下限（见 __init__ 内 mach_ref
# 钳制处的完整推导/实证标定记录）。低于此值时预处理通量的压差放大系数
# ~1/mach_ref² 会让显式时间推进在任何声学 CFL 步长下失稳。
_MACH_REF_FLOOR = 0.1

# `logger` 本文件自己不直接调用（真实排查过：0 处 `logger.xxx(...)`），
# 但 `step.py::mean_flow_residual` 用 `from autoflowcfd.core.fr_solver.
# solver import logger` 延迟导入它（避免循环依赖，见该行注释）——删掉
# 会导致那个 ImportError（真实复现，2026-08-21，被 tests/validation/
# test_couette.py 抓到）。之前这里是标准库 `logging.getLogger(__name__)`
# （从未 basicConfig 过，root logger 默认无 handler），`step.py` 里唯一
# 用到它的 `logger.error(f"Step failed with error: {e}")` 因此也一直
# 被静默吞掉（该异常处理分支里唯一可见的输出其实是同一段代码紧接着的
# `traceback.print_exc()`，两者容易混淆成"看起来在正常报错"）。改成
# 本代码库统一用的 loguru，两处修复是同一个根因。


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


class FRSolver(_SolverGeometryMixin):
    """
    基于通量重构 (FR) 方法的 N-S 方程求解器。
    
    Attributes:
        mesh: 高阶网格对象
        state: 求解器状态容器
        ops: 预计算的 FR 算子
        bc_handler: 弱边界条件处理器
        turb_model: 湍流模型处理器
    """

    def __init__(self, mesh: 'HighOrderMesh', order: int = 2,
                 turb_model_name: str = "SST", n_vars: int = 5,
                 time_scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
                 initial_state: Optional[FRState] = None,
                 backend: str = "cpu",
                 rho_inf: float = 1.225, vel_inf: float = 33.33, p_inf: float = 101325.0,
                 bc_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                 mu_molecular: float = 1.8e-5,
                 dual_time_inner_iter: int = 20,
                 n_threads: int = -1,
                 adaptive_cfl: bool = True,
                 cfl_start: float = 0.1,
                 cfl_max: float = 0.5,
                 cfl_min: float = 0.05,
                 turbulence_intensity: float = 0.01,
                 viscosity_ratio: float = 5.0,
                 sem_num_eddies: int = 200,
                 flux_type: str = 'radau',
                 artificial_viscosity_enabled: bool = False,
                 artificial_viscosity_alpha: float = 1.0,
                 entropy_stable_volume_enabled: bool = False,
                 low_mach_precond: bool = True):
        """
        初始化 FRSolver。

        Args:
            mesh: HighOrderMesh 类型的高阶网格对象
            order: 多项式阶数
            turb_model_name: 湍流模型名称 ("SST"/"DDES"/"IDDES"/"WMLES"/"LES"/"NONE")
            mu_molecular: 分子动力粘度（默认 1.8e-5 Pa*s，标准状态下空气）。
                此前粘性残差（fr_residual_viscous.py 默认参数）与粘性 CFL
                步长（_compute_local_time_step）各自独立硬编码这个值，没有
                任何受支持的方式一致地设置自定义粘度——两处必须同步改，
                否则 CFL 步长会与真正参与残差组装的粘度脱节（同类问题见
                记忆条目 hardcoded_molecular_viscosity_mismatch）。现在两处
                都从这个构造参数读取。
            n_vars: 守恒变量数量（默认5：rho, rho_u, rho_v, rho_w, rho_e）
            time_scheme: 时间推进方案
            initial_state: 初始状态（用于 Order Continuation）
            backend: 计算后端 ("cpu" 或 "gpu")
            rho_inf, vel_inf, p_inf: 自由来流条件（密度/速度大小[沿+x]/静压），
                用作 FARFIELD 边界的幽灵态、未匹配到边界组的默认边界条件，
                以及 INLET 组未显式覆盖时的默认入口状态
            bc_overrides: 按边界组名称覆盖 BC 类型/参数，例如
                {"inlet": {"type": "INLET", "Q_inlet": [...]}, "car_body": {"type": "WALL"}}；
                未提供的组按 mesh.boundary_bc_types 自动映射
                （WALL/SLIP_WALL->WALL, VELOCITY_INLET->INLET,
                PRESSURE_OUTLET->OUTLET, SYMMETRY->SYMMETRY, 其余->FARFIELD）
            dual_time_inner_iter: DUAL_TIME 方案每个物理步的伪时间内迭代
                次数（仅 time_scheme=DUAL_TIME 时有意义）。此前完全没有
                途径设置，恒为硬编码 3；真实测得默认保守 CFL 起点下 3 次
                内迭代通常远不足以让伪残差收敛到物理时间精度要求的水平
                （见 TimeIntegrator.__init__ 与 step_dual_time 文档）。
            n_threads: numba 并行 kernel（无粘/粘性残差界面项，见
                core/fr_residual_inviscid_kernel.py、
                core/fr_viscous_flux_kernel.py 模块文档"多核并行"一节）
                使用的 CPU 线程数。默认 -1 **不是** `os.cpu_count()`，
                而是解析成 **8**——见下方 `_DEFAULT_N_THREADS` 处的完整
                实测记录（2026-09-13 重新测量：体积项/梯度链路改成 numba
                kernel 且 BLAS 线程限制为 1 之后，79 万单元真实网格 P1
                的甜点从此前的 4 上移到 8，nt=12/16 仍然净倒退，根因是
                界面项按图着色逐色串行调用导致的屏障开销 + 本机 P/E 混合
                核在静态均分调度下的不均衡）。调用方仍可显式传更大或更小
                的 n_threads 覆盖（CLI 见 `--threads`/`-j`）。
                只在这里调用一次 `numba.set_num_threads`——两个界面
                kernel 的 `n_threads` 参数要求调用方紧邻调用前取
                `numba.get_num_threads()`，如果这个全局状态在其他地方
                被并发修改，会破坏该约束（见两个 kernel 模块文档"多核
                并行"一节的坑E）。
            artificial_viscosity_enabled: 是否启用 Persson-Peraire 模态
                传感器 + 局部人工粘性（见 core/fr_operators/
                artificial_viscosity.py 模块文档，ProjectFiles/V2.0/
                7_重大问题修复-求解稳定性.md 五、六节）。默认 False——
                这是 2026-08-29 调查引入的全新、独立的可选能力，不改变
                任何未显式启用它的现有求解路径/测试的行为。启用后在
                `compute_viscous_residual` 里，对模态谱衰减速率超出
                光滑函数理论预期（`1/N^4`）的单元，叠加一个局部人工
                粘性到既有的 `mu_t_field` 通道（复用已验证的 BR1 面
                耦合粘性通量组装，不新建独立扩散残差路径）。
            artificial_viscosity_alpha: 人工粘性强度标定常数（无量纲，
                默认 1.0），只在 artificial_viscosity_enabled=True 时
                有意义，见 compute_persson_peraire_artificial_viscosity
                文档。
            entropy_stable_volume_enabled: 是否在体积项过积分（over-
                integration）分支启用 Chandrashekar (2013) 熵守恒两点
                通量替代逐点通量代入（见 core/fr_residual/inviscid.py::
                compute_inviscid_residual_fr 的 entropy_stable_volume
                参数文档、`8_算法重构-Entropy-Stable_Split-Form通量
                重构-Part1/2.md`）。默认 False——这是 2026-08-30 调查
                引入的全新可选能力，不改变任何未显式启用它的现有求解
                路径/测试的行为。真实决定性测试确认方向一致的真实改善
                （在已经用了过积分的基础上再改善约 2~4 倍），代价是
                体积项计算量从 O(n_fine) 升到 O(n_fine^2)（两点通量
                遍历 SP 对的固有代价），默认关闭以避免无条件拖慢现有
                全部生产用例。
            turbulence_intensity: 来流湍流强度 Tu（默认 0.01 = 1%），用于从
                物理自洽的公式推导 k/omega 初值（工业 RANS 标准做法）。
                外部气动默认 ≤1%，城市道路 3-5%，风洞对标 0.5-2%。
            viscosity_ratio: 来流粘性比 VR = nu_t/nu（默认 5.0），与 Tu 共同
                决定 omega 初值。外部气动推荐 2-10。
            sem_num_eddies: LES/DDES/IDDES 模式下 BD-02 合成湍流入口 (SEM) 的
                涡核数量（默认 200，此前恒为硬编码常量，见
                fr_solver/boundary.py 模块文档 2026-08-28 的修复说明）。
                只在 turb_model_name 为 LES/DDES/IDDES 且未激活 WMLES 时生效。
            flux_type: FR 修正函数族选择，透传给 `fr/operators.py::
                generate_fr_operators` 的 `flux_point_type` 参数（#14 新增，
                见该函数文档）。默认 'radau'（此前唯一被使用过的方案，
                数值行为不变），'gauss' 启用与 Spectral Difference 等价的
                新方案。Order Continuation 跨阶数重建算子时
                （order_continuation.py）会读取 `self.flux_type` 保持
                同一个选择贯穿整个求解过程，不会在阶数切换时静默退回默认值。
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

        self.mesh = mesh
        self.order = order
        self.flux_type = flux_type
        self.artificial_viscosity_enabled = artificial_viscosity_enabled
        self.artificial_viscosity_alpha = artificial_viscosity_alpha
        # **order == 1 时人工粘性是精确的无操作**（2026-09-15 实测确认）。
        # Persson-Peraire 的 ramp 判据是 `s0 = -4*log10(order)`，order=1 时
        # s0 = 0，于是触发门限成了"顶模态能量占全胞 10% 以上"
        # （`S_e >= 10^(s0-kappa) = 0.1`）。而 P1 的"顶模态"**就是全部
        # 非常数模态**，即胞内变化本身——已解析的 P1 物理与混叠在这个
        # 指标下无法区分，传感器几乎不触发。
        #
        # 实测（plate_demo 363,392 单元，order=1，固定 CFL 0.03，其余参数
        # 逐项相同的 A/B）：开与不开 `--artificial-viscosity`，**前 51 步的
        # 残差与 Cd 逐字符完全相同**——人工粘性贡献精确为零。
        #
        # 这一点要紧，因为项目对退化单元残差放大的既定修复路线是
        # "网格质量门 + 耗散"（见 industry_practice_degenerate_cell_gcl
        # 项目记忆），而耗散那一半在**生产阶数**上不可用。order=2/3 的
        # 门限分别是 6.25e-3 / 1.24e-3，才是传感器的正常工作区间。
        #
        # 按本项目"不接受静默无操作"的约定，这里显式警告而不是让用户以为
        # 自己已经打开了一层保护。同源结论见 fr_solver/filter.py 里
        # `AFCFD_FILTER_MODE=sensor` 在 P1 上原理不适用的说明。
        if artificial_viscosity_enabled and order <= 1:
            import warnings
            warnings.warn(
                f"--artificial-viscosity 在 order={order} 上是**无操作**："
                f"Persson-Peraire 判据 s0=-4*log10(order)=0 使触发门限变成"
                f"顶模态能量占比 >= 10%，而 P1 的顶模态就是全部非常数模态，"
                f"传感器几乎不触发（真实网格 A/B 实测前 51 步残差/Cd 逐字符"
                f"相同）。要用人工粘性请在 order>=2 上使用；P1 的退化单元"
                f"问题目前只能靠网格质量门拦。",
                RuntimeWarning, stacklevel=2,
            )
        self.entropy_stable_volume_enabled = entropy_stable_volume_enabled
        if entropy_stable_volume_enabled and backend.lower() == "gpu":
            import warnings
            warnings.warn(
                "entropy_stable_volume_enabled=True 目前只在 CPU 路径实现"
                "（core/fr_residual/inviscid.py::compute_inviscid_residual_fr），"
                "GPU 路径（core/gpu/residual/gpu_inviscid.py）尚未移植，"
                "backend='gpu' 时这个开关不生效，体积项仍走 GPU 原有的"
                "逐点通量代入实现——不是被静默忽略导致错误结果，只是"
                "拿不到这个优化。",
                stacklevel=2,
            )
        self.backend_type = backend.lower()
        
        # 安全地获取网格信息
        n_cells = getattr(mesh, 'n_cells', 0)
        n_sps = getattr(mesh, 'n_sps_per_cell', 8)
        
        # 1. 初始化状态 (S-01)
        if initial_state is not None:
            # 使用提供的初始状态（Order Continuation）
            self.state = initial_state
            print(f"   [OK] Using provided initial state from lower order")
        else:
            # 根据湍流模型确定变量数。必须先 upper()：self.turb_model_name
            # 要到第 5 步才被规范化为大写，这里若直接用构造参数原始大小写
            # 比较，会导致 `turbulence-model sst`（小写，steady CLI 路径）
            # 与 `SST`（大写，transient CLI 路径）判出不同的 n_vars——已用
            # 两条 CLI 路径实测复现（steady 得到 n_vars=5，transient 得到 7）。
            default_n_vars = 7 if turb_model_name.upper() in ["SST", "DDES", "IDDES"] else 5
            self.state = FRState(n_cells, n_sps, default_n_vars)
            # 初场必须与自由来流边界条件一致（都用同一套 rho_inf/vel_inf/p_inf），
            # 否则显式伪时间推进第一步就要吸收一个几个数量级的压力跳跃
            # （旧版本硬编码 rho=1,p=1 的"单位"初场，与真实边界条件的
            # rho_inf~1.2, p_inf~1e5 相差 5 个数量级，显式格式在这种冲击下
            # 数值发散——这不是残差组装的 bug，是初始条件与边界条件不一致
            # 导致的可预见的数值不稳定，工业代码从来不会这样初始化）。
            self.state.initialize_uniform(rho=rho_inf, u=vel_inf, v=0.0, w=0.0, p=p_inf)
        
        # 2. 预计算算子 (G-04)——四面体坍缩坐标基已删除（2026-09-03，
        # 见 fr/operators.py 模块文档），`generate_fr_operators` 不再
        # 接受 `tet_basis_mode` 参数，恒生成 native 四面体算子，与
        # `mesh`（同样恒为 native，见 HighOrderMesh 文档）天然一致，
        # 不再需要从 mesh 读取这个属性来保持两者同步。
        # 低马赫数伪时间预处理（2026-09-14 新增，用户提出"收敛需要数万步"
        # 后的根本性优化）。完整推导/正确性论证见
        # `core/utils/preconditioning.py` 模块末尾"伪时间预处理矩阵 Gamma"
        # 一节；接入点见 `step.py::mean_flow_residual`（残差侧）与
        # `cfl.py::compute_local_time_step`（步长侧）——两者**必须成对启用**。
        #
        # 为什么默认开：本项目的目标工况是汽车外流场，M~0.09（33m/s vs
        # 声速 340m/s）。不做预处理时显式格式的 dt 被声速限制，比对流
        # 时间尺度小约 11 倍，收敛步数因此白付约一个数量级——这正是
        # 用户观察到"需要数万步"的主因之一。预处理后 dt 由预处理波速
        # (|un|+c_precond) 决定，M=0.1 下放大约 5 倍。
        # 不动点不变（det(Gamma)=beta^2>0），收敛解与关闭时是同一个解。
        #
        # DUAL_TIME（真正的非稳态物理时间推进）下强制关闭：那条路径的
        # dt 是有物理时间精度含义的物理步长，不是伪时间步长，预处理的
        # 前提（"只要收敛到 R=0，路径无所谓"）不成立。
        # 只对 SSP-RK2/RK3 这两个"纯稳态伪时间推进"方案启用：
        # * DUAL_TIME 的 dt 是有物理时间精度含义的物理步长，不是伪时间
        #   步长，预处理的前提（"只要收敛到 R=0，路径无所谓"）不成立；
        # * IMEX 把残差**拆成**对流/扩散两半分别显式/隐式处理
        #   （见 step.py 的 convective_residual_only/diffusive_residual_only），
        #   Gamma 作用在拆分后的任一半上都不等价于作用在整体残差上，
        #   需要专门推导如何在两半之间分配预处理——不在本次范围内，
        #   所以这里直接不启用，而不是套一个未经验证的近似。
        # 环境变量 `AFCFD_LOW_MACH_PRECOND=0/1` 可强制关闭/开启，优先于
        # 构造参数——供 A/B 对照实验与现场排查用（"把这个新机制单独关掉
        # 再跑一遍"必须是一条随时可用的路径，不需要改代码）。
        _env = os.environ.get("AFCFD_LOW_MACH_PRECOND")
        _req = bool(low_mach_precond) if _env is None else (_env == "1")
        self.low_mach_precond_enabled = _req and time_scheme in (
            TimeIntegrationScheme.SSP_RK2, TimeIntegrationScheme.SSP_RK3,
        )

        self.ops = generate_fr_operators(order, flux_point_type=flux_type)

        # BLAS 线程数压到 1（性能优化 2026-09-13，见 `_limit_blas_threads`
        # 与 `autoflowcfd/__init__.py` 顶部的完整实测记录）。**位置很关键**：
        # 必须在网格几何（mesh.jacobians，调用方在构造 solver 之前就算好）
        # 与上面这行 FR 算子构造**之后**才限制——这两处的 LAPACK 结果会随
        # 线程数在最后一位上变化，而离散 GCL/自由流场保持性依赖度量量之间
        # 近乎精确的抵消（把限制提前会让棱柱 P2 保持性判据从 8.07e-7 退化到
        # 3.35e-6、真实测试失败）。在这之后限制：残差与全程多线程逐位相同，
        # 同时求解循环拿到 9~11% 的收益。
        # 注意：**不要**在这里调 `_limit_blas_threads()`——那样是进程级、
        # 粘性的，会污染同一进程里后续求解器的几何/算子构造（真实 bug，
        # 见 `blas_threads_limited` 文档记录的 Couette 连跑失败）。
        # 限制只在求解循环内生效，见 `solve()` 里的 `blas_threads_limited`。
        
        # 3. 初始化边界条件 (BD-01) —— 真正参与残差组装的幽灵态边界条件
        # （不再持有未被使用的 FRWeakBC 罚项处理器实例——那是旧版本从未被
        # 求解主循环调用过的死代码路径，真正生效的是下面的
        # boundary_ghost_provider，见 boundary/fr_ghost_state.py）
        #
        # self.turb_model_name 必须先于 boundary_ghost_provider 构造
        # （BD-02：LES/DDES 模式下要给 VELOCITY_INLET 组接入合成湍流
        # 入口，需要在构造 ghost provider 时就知道湍流模型），其余湍流
        # 模型对象（turb_model/sgs_model）留到下面第 5 步再真正初始化。
        #
        # 真实 bug 修复（2026-09-02，排查分布式 WMLES 支持时发现，与
        # 分布式本身无关，是本文件独立的一个真实回归）：上面这句"ghost
        # provider 构造时只用 getattr(...,None) 安全读取 wmles_model，
        # 不依赖它已经存在"是过时的错误理解——`build_boundary_ghost_
        # provider` 用 `getattr(solver,"wmles_model",None) is None` 判断
        # WALL 组是否要切换成 is_no_slip=False（见该函数文档 #9 修复
        # 说明），但 `self.wmles_model` 真正被赋值（`_init_turbulence_
        # models`，下面第 5 步）发生在 `boundary_ghost_provider` 构造
        # **之后**——`getattr` 在属性完全不存在时同样返回 None，不会
        # 报错，但这意味着 `wall_is_no_slip` 恒为 True，#9 修复的
        # "WMLES 假滑移边界"效果自那次修复写下起就从未真正生效过（只是
        # 不崩溃，掩盖了这个事实）。这里提前构造真正的 wmles_model（只
        # 需要 mu_molecular/rho_inf 两个已经是构造参数的标量，不依赖
        # 第 5 步其余状态），下面第 5 步改为不再重复构造（见该处新增的
        # guard）。
        self.turb_model_name = turb_model_name.upper()
        self.wmles_model = None
        if self.turb_model_name == "WMLES":
            from autoflowcfd.core.turbulence.wmles import WMLESModel
            self.wmles_model = WMLESModel(nu=mu_molecular / max(rho_inf, 1e-10))
        # mach_ref：AUSM+up Weiss-Smith 低马赫数预处理（kernels.py::
        # compute_ausm_up_flux）和 CFL 步长估计（cfl.py）共用的同一个
        # 参考马赫数，从真实自由来流条件算一次，不再各处各用一套（2026-
        # 08-14 那次失稳正是因为 CFL 和通量各自假设了不一致的参考值，
        # 见 cfl.py 模块文档"已撤销"一节）。
        #
        # 物理下限钳制（2026-08-26，P2 发散专项修复）：真实来流马赫数低于
        # _MACH_REF_FLOOR 时钳制到下限。AUSM+up 的 Mp 压力扩散项正比于
        # (pR-pL)/(fa*a_half_p²)，其中 fa≈2*mach_ref（滞止面）、
        # a_half_p²=beta2*a²、beta2 下限=1.1*mach_ref²——两者同时随
        # mach_ref 塌缩，压差的有效放大系数 ~1/mach_ref²：Couette 验证
        # 算例（vel_inf=U_wall=0.01 m/s，mach_ref≈2.9e-5）实测残差泛函
        # 对能量扰动的增益高达 ~3.4e13/Pa（mach_ref 扫描证实增益严格正比于
        # 1/mach_ref²），显式 SSP-RK3 在任何声学 CFL 步长下都必然发散。
        # 物理上这个下限对应低马赫数渐近展开的适用边界：真实压力扰动按
        # ρ·U² ~ mach_ref² 缩小才与预处理通量的 1/mach_ref² 放大相互抵消，
        # 低于下限后离散舍入/边界瞬态扰动不再随 M² 缩小，方案转为舍入驱动失稳。
        # 下限值经真实算例实证扫描标定（分两档）：
        # （1）Couette 棱柱算例：0.02 仍发散（iter 6）、0.05 稳定；
        # （2）TGV 三向周期坍缩坐标四面体算例（真实 mach_ref≈0.0874，
        #     网格条件数更差）：0.0874 仍发散（step 5 KE 暴涨至 1e93、
        #     step 6 溢出）、0.1 稳定且动能衰减曲线与历史实测一致。
        # 因此下限取 0.1——恰好等于 kernels.py 历史注释记载的遗留硬编码值，
        # 那次把硬编码改成传入真实值的重构正是这两个算例的共同回归点；
        # 0.1 以上真实马赫数的算例不受影响。
        mach_ref = vel_inf / np.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10))
        mach_ref = max(mach_ref, _MACH_REF_FLOOR)
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf, "mach_ref": mach_ref}
        # Tu/VR/SEM 涡核数设置必须先于 boundary_ghost_provider 构造（与上面
        # turb_model_name 的顺序要求同理，2026-08-28 补充）：
        # build_boundary_ghost_provider 现在会读 solver._turbulence_intensity/
        # _sem_num_eddies 来驱动 BD-02 SEM 入口的目标雷诺应力/涡核数量
        # （见 fr_solver/boundary.py 模块文档），若这几个属性此时还没设置，
        # getattr 的兜底默认值会悄悄生效、用户传入的 --turbulence-intensity/
        # --sem-num-eddies 就会被忽略——必须在 _build_boundary_ghost_provider
        # 调用之前赋值。
        self._turbulence_intensity = turbulence_intensity
        self._viscosity_ratio = viscosity_ratio
        self._sem_num_eddies = sem_num_eddies
        # B-9：保存 bc_overrides 供 Order Continuation 切阶后重建边界幽灵态
        # provider（SEM 入口幽灵态持有构造阶数的 FP 坐标，见 order_continuation.py）。
        self.bc_overrides = bc_overrides or {}
        self.boundary_ghost_provider = self._build_boundary_ghost_provider(self.bc_overrides)
        self.mu_molecular = mu_molecular

        # 4. 初始化计算后端 (B-01)——见 solver_helpers.py::resolve_backend_type 文档。
        self.backend = None
        self.backend_type = solver_helpers.resolve_backend_type(self.backend_type)

        # 5. 初始化湍流模型（self.turb_model_name 已在第 3 步设置；
        # self.wmles_model 若适用也已在第 3 步提前构造好，供
        # boundary_ghost_provider 正确读取，这里不再重置，见第 3 步
        # 的说明与 _init_turbulence_models 里对应的 guard）
        self.turb_model = None
        self.ddes_model = None
        self.sgs_model = None

        self._init_turbulence_models(n_cells, n_sps)
        
        # 6. 初始化时间积分器 (S-05)
        self.time_integrator = TimeIntegrator(scheme=time_scheme, dual_time_steps=dual_time_inner_iter)

        # 6b. 自适应 CFL 控制器（2026-08-24）：
        # 稳态伪时间迭代中根据残差历史自动调节 CFL 数，替代此前硬编码 0.1。
        # 仅对稳态路径（SSP-RK2/RK3）生效；DUAL_TIME 有自己的内层自适应逻辑。
        self._cfl_controller = None
        if adaptive_cfl and time_scheme != TimeIntegrationScheme.DUAL_TIME:
            from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController
            # cfl_start/cfl_max 现在是构造参数（2026-09-07）：此前
            # `AdaptiveCFLController()` 恒用硬编码默认值（0.1/0.3），
            # `SteadyConfig.cfl_init`/`cfl_max` 这两个 config 字段从未
            # 真正接到控制器上——CLI `--cfl-start`/`--cfl-max` 现在直接
            # 透传到这里。cfl_max 默认值同步从 0.3 上调到 0.5（SSP-RK3
            # 线性稳定极限 ~1.0，0.3 对本项目多数网格过于保守；AUSM+up
            # 低马赫预处理激活的算例真实可用上限更低，需要时用
            # `--cfl-max` 显式回调）。
            # cfl_min 同样是构造参数（2026-09-15）：此前五处控制器构造点
            # 全都没有传它，于是恒用控制器默认 0.05。那个值**高于**真 P1
            # （模态滤波器关闭、零阶数损失）在 79 万单元 cube_demo 上实测
            # 稳定的 CFL 0.03——也就是说一个已验证可用的工作点通过 CLI
            # 根本到不了：`--cfl-start 0.03` 会被 cfl_min 钳回 0.05
            # （修复前是第一次收缩时静默跳到 0.05，见 adaptive_cfl.py
            # 模块文档第 11 条），必然发散。下限必须可配。
            self._cfl_controller = AdaptiveCFLController(
                cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            )
            print(f"   Adaptive CFL: enabled (start={self._cfl_controller.cfl_start}, "
                  f"max={self._cfl_controller.cfl_max}, "
                  f"min={self._cfl_controller.cfl_min})")
        else:
            print(f"   Adaptive CFL: disabled")

        # 影响物理/数值的开关必须在启动日志里可见（2026-09-15 引入）：做
        # 人工粘性 A/B 对照时发现，`--artificial-viscosity` 生效与否在
        # 整份日志里**没有任何痕迹**，只能靠翻进程命令行确认——那让
        # "这份日志是哪个配置跑出来的"变成了事后考古。模态滤波档同理，
        # 它直接决定解的多项式阶数有没有被清掉一整阶。
        #
        # 2026-09-16：这三行原先写在上面 `if adaptive_cfl and ...` 的
        # **then 分支里**，所以关掉自适应 CFL（含全部 DUAL_TIME 运行）
        # 时它们一行都不打印——恰好复刻了它本该消除的那个问题。已移到
        # 分支外，与 CFL 档无关。同时补上 AUSM+up 预处理档：它决定壁面
        # 压力是否被推进 P5± 的饱和分支（legacy 档在固壁上有 7.3 倍虚假
        # 超压，见 fr_operators/kernels.py 模块级 PRECOND_* 常量上方的
        # 长注释），是比滤波档更强的物理开关，绝不能只能靠命令行考古。
        from autoflowcfd.core.fr_operators.kernels import (
            ausm_precond_mode_label as _pm_label,
            resolve_ausm_precond_mode as _pm_resolve,
        )
        from autoflowcfd.fr.modal_filter import FILTER_MODE as _fm
        _av = ("enabled (alpha=%g)" % self.artificial_viscosity_alpha
               if self.artificial_viscosity_enabled else "disabled")
        print(f"   Artificial viscosity: {_av}")
        print(f"   Modal filter mode: {_fm}")
        print(f"   AUSM+up precond mode: {_pm_label(_pm_resolve())}")

        # DUAL_TIME 专用：物理时间层 n-1 的解（BDF2 时间导数项需要），
        # None 表示还没有跑过物理步（下一步会退化为 BDF1），见 step()
        # 与 order_continuation.interpolate_to_new_order_checked（阶数
        # 变化后 SPs 布局改变，必须让这份历史失效，否则形状不匹配/物理
        # 上不连续的历史层会被静默用于 BDF2）。
        self._dual_time_U_prev: Optional[np.ndarray] = None
        
        # 7. 壁面距离场（用于DDES/WMLES）
        self.wall_distance = None

        # 8. Order Continuation 状态
        self.current_order = order
        self.order_continuation_enabled = True

        # 9. 收敛历史（真实 bug 修复，V2.0 专家组盲审发现，2026-08-27）：
        # 此前 CPU FRSolver 从不记录逐迭代残差，api.py::get_convergence_history
        # 恒返回硬编码占位符 {"iterations": [], "residuals": []}——GPUFRSolver
        # 一直有这个属性且真正被 solve 循环填充，CPU 版本此前遗漏。solve()/
        # run_order_continuation() 每步迭代后 append，与 GPU 版同一约定。
        self.residual_history: list = []

        print(f"[OK] FRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Turbulence: {turb_model_name}")
        print(f"   Backend: {self.backend_type.upper()}")
        print(f"   Time Scheme: {time_scheme.value}")
        print(f"   Threads: {resolved_n_threads}")

    def _init_turbulence_models(self, n_cells: int, n_sps: int):
        """初始化湍流模型（委托给 fr_solver_turbulence）。"""
        fr_solver_turbulence.init_turbulence_models(self, n_cells, n_sps)
    
    def _build_boundary_ghost_provider(self, bc_overrides: Dict[str, Dict[str, Any]]):
        """构建边界幽灵态提供者 (BD-01)，委托给 fr_solver_boundary。"""
        return fr_solver_boundary.build_boundary_ghost_provider(self, bc_overrides)

    def compute_wall_distance_field(self, mesh_nodes: np.ndarray,
                                   wall_indices: np.ndarray,
                                   connectivity: Optional[np.ndarray] = None,
                                   use_eikonal: bool = False):
        """计算壁面距离场（用于 DDES/WMLES/SST），委托给 fr_solver_turbulence。

        Args:
            mesh_nodes: 全部网格节点坐标
            wall_indices: WALL 边界节点索引
            connectivity: 节点邻接表，use_eikonal=True 时必须提供 - 见
                fr_solver_turbulence.compute_wall_distance_field 自己的文档
            use_eikonal: 是否用 Eikonal 方程（而不是纯欧氏 KD-Tree）求解
        """
        fr_solver_turbulence.compute_wall_distance_field(
            self, mesh_nodes, wall_indices, connectivity=connectivity, use_eikonal=use_eikonal
        )

    def solve(self, max_iter: int = 1000, dt: float = 1e-4, tol: float = 1e-6,
              checkpoint_callback=None,
              phase_max_iter: Optional[int] = None,
              residual_drop_threshold: float = 1e2) -> SolverResult:
        """
        执行稳态/瞬态求解循环。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 相对收敛容差——残差需相对初始值下降 1/tol 倍才算收敛。
                默认 1e-6 表示下降 6 个量级（与 Order Continuation 各阶段
                的 phase_tol 阶数缩放配合：P0 下降 4 级、P1 下降 5 级、P2 下降 6 级）。
                此前为绝对判据 res < tol，对 RMS ~1e8 的流动永远不可达。
            checkpoint_callback: 可选的中间 checkpoint 回调函数，
                签名为 callback(solver, iteration_number)，每步迭代后调用。
                用于在求解过程中定期保存状态到磁盘。
            phase_max_iter: 仅 Order Continuation（`self.order>=2` 时）生效，
                非最终阶段（P0/P1/...）各自的最大迭代步数上限，None 时保留
                旧行为（`max_iter // len(orders)` 按阶段数均分）——见
                `order_continuation.run_order_continuation` 同名参数文档。
            residual_drop_threshold: 仅 Order Continuation 生效，单阶段
                判定"可以提前升阶"的残差下降倍数，默认 1e2（降 2 个数量级），
                原来硬编码，现在可配置。

        Returns:
            SolverResult: 包含收敛状态、最终残差和迭代次数的结果对象
        """
        logger_msg = f"Starting solve loop with {self.time_integrator.scheme.value}"
        if self.turb_model_name != "NONE":
            logger_msg += f", turbulence={self.turb_model_name}"
        print(logger_msg)

        # Order Continuation: 从低阶开始逐步提升精度
        if self.order_continuation_enabled and self.order >= 2:
            # Order Continuation 路径在 `core/utils/order_continuation.py`
            # 里自己套 `blas_threads_limited`（求解循环在那边）。
            return self._solve_with_order_continuation(
                max_iter, dt, tol, checkpoint_callback,
                phase_max_iter=phase_max_iter,
                residual_drop_threshold=residual_drop_threshold,
            )
        
        import time
        converged = False
        final_residual = 1e10
        initial_res = None
        
        # BLAS 线程数只在求解循环期间限制为 1（性能：求解阶段实测快
        # 9~11%；作用域必须是"循环期间"而不是"构造时一次"，理由见
        # `blas_threads_limited` 文档记录的真实 bug）。
        last_finite = None
        with blas_threads_limited(1):
            for i in range(max_iter):
                t_start = time.time()
                res = self.step(dt)
                t_end = time.time()
                final_residual = res
                self.residual_history.append(res)

                # 发散即中止（2026-09-16，真实事故驱动）：此前这条循环
                # 完全没有有限性检查，残差变成 inf/nan 之后照常继续迭代、
                # 照常调用 checkpoint_callback，会把 NaN 状态写进
                # checkpoint 并在收尾时用 NaN 覆盖 final_state.pkl。
                # GPU 单机与多 GPU 路径本来就有这个检查，CPU 路径此前
                # 遗漏——是路径不对等，不是有意设计。检查必须在
                # checkpoint 回调**之前**。
                check_residual_finite(res, i + 1, order=self.order,
                                      last_finite=last_finite)
                last_finite = res

                if initial_res is None:
                    initial_res = res
            
                # 每步打印详细信息（真实功能缺口修复，2026-08-31，用户直接
                # 指出"P0/P1直接运算和P2 order continuation打印的信息应该
                # 一样"）：这条"非 order continuation"常规循环（目标阶数<2，
                # 例如单独求解 P0/P1）此前打印频率（每10步一次）、字段顺序
                # （CFL 在 Time 之前）、前缀（"Iteration N"而非"P{order} Iter
                # N"）、Time 标签（"Time/step"而非"Time"）都与
                # order_continuation.py::run_order_continuation（目标阶数>=2
                # 时走的分阶段路径）不一致——两条路径各自独立发展、从未同步
                # 过格式。这里改成逐字段对齐 order_continuation.py 的格式
                # （以其为准），包括每步都打印、同样的字段顺序与 Cd/Cl/Cs
                # 气动力系数打印。
                drop = initial_res / max(res, 1e-30)
                msg = f"P{self.order} Iter {i+1}: Residual = {res:.6e} | Drop: {drop:.1f}x | Time: {t_end - t_start:.2f}s"
                if self._cfl_controller is not None:
                    msg += f" | CFL={self._cfl_controller.cfl_number:.3f}"
                ref_area = getattr(self, '_reference_area', None)
                if ref_area is not None and ref_area > 0:
                    from autoflowcfd.postprocess.fr_coefficients import compute_forces_pressure_only
                    aero = compute_forces_pressure_only(self, ref_area)
                    msg += f" | Cd={aero['Cd']:.4f} Cl={aero['Cl']:.4f} Cs={aero['Cs']:.4f}"
                # 按方程分别归一化残差 + 最大残差定位（与 order_continuation.py
                # 同一处新增，参照 Fluent scaled residuals / STAR-CCM+ Max
                # 监视器，见 residual_diagnostics.py 模块文档"背景"一节）：
                # 只新增打印，不改变本函数自己的 `tol`/`drop` 收敛判据。
                #
                # 打印频率（2026-09-13 用户反馈修复，与 order_continuation.py
                # 同一处、同一理由）：只在第 1 步和其后每 10 步打印一次，避免
                # 正常运行时每步都刷出这行长诊断信息。
                freestream = getattr(self, 'freestream', None)
                if freestream is not None and hasattr(self.state, 'dU_dt') and (i == 0 or (i + 1) % 10 == 0):
                    from autoflowcfd.core.fr_solver.residual_diagnostics import (
                        compute_scaled_residuals, format_scaled_residual_line,
                    )
                    diag = compute_scaled_residuals(self.state.dU_dt, freestream)
                    msg += " | " + format_scaled_residual_line(diag)
                print(msg)

                # 中间 checkpoint 保存
                if checkpoint_callback is not None:
                    checkpoint_callback(self, i + 1)
                
                # 相对收敛判据：残差相对初始值下降 1/tol 倍
                # tol=1e-6 表示需要下降 6 个量级；tol<=0 表示纯定步数迭代（
                # B-10：transient 命令固定传 tol=0.0，此前 1.0 / tol 在第 2 步
                # 直接 ZeroDivisionError 崩溃），此时不启用收敛判据。
                if i >= 1 and tol > 0.0 and initial_res / max(res, 1e-30) >= 1.0 / tol:
                    converged = True
                    print(f"[OK] Converged at iteration {i+1} with residual {res:.6e} "
                          f"(dropped {initial_res/res:.1e}x)")
                    break
        
        return SolverResult(converged=converged, iterations=i+1, final_residual=final_residual)
    
    def _solve_with_order_continuation(self, max_iter: int, dt: float, tol: float,
                                        checkpoint_callback=None,
                                        phase_max_iter: Optional[int] = None,
                                        residual_drop_threshold: float = 1e2) -> SolverResult:
        """实现 Order Continuation 策略：从P0逐步提升到目标阶数（委托给 order_continuation）。"""
        return order_continuation.run_order_continuation(
            self, max_iter, dt, tol, checkpoint_callback,
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold,
        )

    def _interpolate_to_new_order(self, new_order: int):
        """将解从当前阶数插值到新的阶数（委托给 order_continuation）。"""
        order_continuation.interpolate_to_new_order_checked(self, new_order)

    def step(self, dt: float) -> float:
        """执行一个时间步长 (S-05)。见 fr_solver_step.py::step 文档。"""
        return fr_solver_step.step(self, dt)

    def compute_turbulence_source(self, dt) -> Optional[tuple]:
        """计算湍流模型源项（委托给 fr_solver_turbulence）。dt 可以是标量
        （DUAL_TIME 物理步长）或逐 SP 数组（稳态加速模式的局部 CFL
        步长 dt_local）——见 fr_solver_turbulence.compute_turbulence_source
        文档。"""
        return fr_solver_turbulence.compute_turbulence_source(self, dt)

    def apply_turbulence_corrections(self):
        """应用湍流模型的修正（SGS 涡粘系数），委托给 fr_solver_turbulence。

        WMLES 壁面剪应力**不**在这里施加——它是一个真正的残差贡献项，
        必须在时间积分*之前*参与残差组装才能生效，见
        compute_viscous_residual() 里的调用与该方法文档（T-05 修复：
        此前在这里调用，而这里在 step() 中排在状态更新*之后*，对本步
        毫无影响，架构上不可能生效）。
        """
        fr_solver_turbulence.apply_turbulence_corrections(self)

    def compute_inviscid_residual(self):
        """
        计算无粘残差 (S-02/S-04)。

        真实的曲边/坍缩坐标 FR 离散：体积项用逆变通量 (contravariant flux)
        散度实现（度量项一致，满足自由流场保持性/离散GCL），界面项用基于
        真实单元-面连接关系的 AUSM+up 黎曼求解 + Radau/VCJH 校正函数投影，
        边界面通过 boundary_ghost_provider (BD-01) 构造物理正确的幽灵态。

        取代旧版本"用全场平均态+硬编码法向量冒充相邻单元"的伪校正项——
        详见 core/fr_residual_inviscid.py 模块文档与
        tests/unit/test_fr_residual_inviscid.py 的自由流场保持性验证。
        """
        self.state._update_primitives()

        if not self.state.U.flags['C_CONTIGUOUS']:
            self.state.U = np.ascontiguousarray(self.state.U)

        # GPU 分发 (B-01)：请求 GPU 后端时走 CuPy 加速路径。
        # P0（阶数延续热身阶段）使用 CuPy RawKernel（core/gpu/residual/gpu_p0_inviscid.py）；
        # P>=1 高阶 FR 使用 CuPy 向量化实现（core/gpu/residual/gpu_inviscid.py）。
        # GPU 不可用时自动回退 CPU。
        if self.backend_type == "gpu":
            if self.mesh.n_points_1d == 1:
                from ..gpu.residual.gpu_p0_inviscid import compute_inviscid_residual_p0_cupy
                res_euler = compute_inviscid_residual_p0_cupy(
                    self.state.U, self.mesh,
                    boundary_ghost_provider=self.boundary_ghost_provider,
                    mach_ref=self.freestream["mach_ref"],
                )
            else:
                # P>=1 高阶 FR GPU 路径
                from ..gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu
                res_euler = compute_inviscid_residual_fr_gpu(
                    self.state.U, self.mesh, self.ops,
                    boundary_ghost_provider=self.boundary_ghost_provider,
                    mach_ref=self.freestream["mach_ref"],
                )
        else:
            res_euler = compute_inviscid_residual_fr(
                self.state.U, self.mesh, self.ops,
                boundary_ghost_provider=self.boundary_ghost_provider,
                mach_ref=self.freestream["mach_ref"],
                entropy_stable_volume=self.entropy_stable_volume_enabled,
            )

        if self.state.n_vars > 5:
            # 湍流量 (k, omega) 的对流输运项当前仍由 compute_turbulence_source
            # 单独处理（局部源项积分，不含对流通量），此处只补零占位维度，
            # 不在这里静默引入未经验证的湍流对流项。
            res_full = np.zeros((res_euler.shape[0], res_euler.shape[1], self.state.n_vars))
            res_full[:, :, :5] = res_euler
            return res_full
        return res_euler

    def compute_viscous_residual(self):
        """
        计算粘性残差 (S-03)。

        真实的 BR1 面耦合粘性离散（core/fr_viscous_flux.py），并把湍流模型
        算出的涡粘系数真正耦合进应力张量/热传导（T-01/T-04/T-06 修复：
        此前调用处从不传湍流粘度，粘性通量永远只用分子粘度 1.8e-5，
        SST/DDES/WALE 算出的 nu_t 场只在自身模型内部自用，从未进入
        动量/能量方程的扩散项）。

        Returns:
            viscous_res: 粘性残差（WMLES 激活时已叠加壁面剪应力修正，
                见 solver_helpers.compute_wmles_wall_stress_correction
                文档 T-05 修复说明——必须在这里（残差组装、时间积分之前）
                施加才能真正影响本步的解，而不是像此前那样在状态更新
                之后才计算）
        """
        mu_t_field = self._get_turbulent_viscosity_field()
        res = compute_viscous_residual_ldg(
            self.state.U, self.state.Q, self.ops, self.mesh,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
        )

        if self.wmles_model is not None:
            wall_stress_correction = solver_helpers.compute_wmles_wall_stress_correction(self)
            if wall_stress_correction is not None:
                res = res + wall_stress_correction[..., : res.shape[-1]]

        # 人工粘性的**质量扩散通道**（2026-09-14 补齐）。
        #
        # 此前 `artificial_viscosity.py` 模块文档里如实记录了一条范围
        # 限制、并把它称作"许多实际 DG/FR 实现采用的简化"：Persson &
        # Peraire (2006) 原方法对**全部**守恒变量（含连续性方程）叠加
        # 人工扩散，而本实现只把 epsilon 叠进 `mu_t_field`，于是它只能
        # 通过动量/能量方程既有的粘性应力/热传导通道起作用，密度本身
        # 完全不被扩散（`viscous_physical_flux` 的质量分量 G[...,0]
        # 恒为 0）。用户明确指出本项目不接受简化，这里补上缺的那一项。
        #
        # 实现方式：不改粘性热路径。AV 默认关闭，没有理由为它给所有
        # 运行的 `viscous_physical_flux_batch` 增加参数与分支；而
        # `div(eps*grad(rho))` 正是一个标量扩散算子，直接复用湍流输运
        # 已经验证过的 BR1 面耦合标量扩散装配
        # （`turbulence/transport.py::compute_scalar_diffusion_residual`，
        # 它返回的就是 +div(Gamma*grad(phi))，与这里 dU/dt 的符号约定
        # 一致）。AV 关闭时这段完全不执行，零开销。
        #
        # 守恒性与自由流场保持性：`div(eps*grad(rho))` 是散度形式，
        # 因此严格守恒；均匀流场下 grad(rho)=0，这一项恒为 0，不破坏
        # 自由流场保持性（已用测试钉住，见
        # tests/unit/test_artificial_viscosity_mass_diffusion.py）。
        if getattr(self, "artificial_viscosity_enabled", False):
            res = res + self._artificial_mass_diffusion_residual()

        return res

    def _artificial_mass_diffusion_residual(self) -> np.ndarray:
        """Persson-Peraire 人工粘性作用在连续性方程上的那一项。

        返回形状与粘性残差相同的数组，只有质量分量（索引 0）非零，
        其值为 `+div(epsilon * grad(rho))`（dU/dt 约定）。
        完整动机见 `compute_viscous_residual` 里的调用点注释。
        """
        from autoflowcfd.core.fr_operators.artificial_viscosity import (
            compute_persson_peraire_artificial_viscosity,
        )
        from autoflowcfd.core.turbulence.transport import (
            compute_scalar_diffusion_residual,
        )

        epsilon_av = compute_persson_peraire_artificial_viscosity(
            self, alpha_av=self.artificial_viscosity_alpha
        )
        rho = self.state.U[..., 0]
        d_rho_dt = compute_scalar_diffusion_residual(
            np.ascontiguousarray(rho), np.ascontiguousarray(epsilon_av),
            self.mesh, self.ops,
        )
        out = np.zeros_like(self.state.U[..., : self.state.U.shape[-1]])
        out[..., 0] = d_rho_dt
        return out

    def _get_turbulent_viscosity_field(self) -> Optional[np.ndarray]:
        """汇总当前激活的湍流模型给出的动力涡粘度场 mu_t = rho * nu_t（委托给 fr_solver_turbulence），
        再叠加 Persson-Peraire 人工粘性（若启用）。

        真实 bug 修复（2026-08-29，TGV 真实复现）：人工粘性最初被直接
        加进 `compute_viscous_residual` 里临时拼出的 `mu_t_field`，
        `_compute_local_time_step`（cfl.py）单独调用这个方法算粘性
        CFL 步长时完全看不到这份额外粘度——时间步长仍按"只有分子
        粘度+湍流涡粘"来估算，而实际粘性残差里已经叠加了一份可能
        大出物理粘度一个数量级的人工扩散，显式格式的粘性稳定性条件
        `dt<=C*h^2/mu_eff` 被违反，真实复现：TGV（P2，Re=20 低雷诺数
        算例，物理 mu 已经刻意调得比空气分子粘度大三个数量级）3 步内
        发散。必须让 CFL 计算与粘性残差看到*同一个* `mu_t_field`——
        统一在这个唯一的读取入口叠加，而不是分别在两个消费点各自
        处理（同一类问题见项目记忆 hardcoded_molecular_viscosity_
        mismatch/low_mach_cfl_ausm_inconsistency：任何"物理量在多个
        消费点独立计算/获取"的模式都有两处失去同步的风险）。
        """
        mu_t_field = fr_solver_turbulence.get_turbulent_viscosity_field(self)
        if getattr(self, "artificial_viscosity_enabled", False):
            from autoflowcfd.core.fr_operators.artificial_viscosity import (
                compute_persson_peraire_artificial_viscosity,
            )

            epsilon_av = compute_persson_peraire_artificial_viscosity(
                self, alpha_av=self.artificial_viscosity_alpha
            )
            mu_t_field = epsilon_av if mu_t_field is None else mu_t_field + epsilon_av
        return mu_t_field

    def _compute_gradients(self) -> np.ndarray:
        """
        计算守恒变量的梯度。

        Returns:
            grad_U: 梯度，形状 (n_cells, n_sps, n_vars, 3)
        """
        from autoflowcfd.core.fr_residual.viscous import compute_gradients
        return compute_gradients(self.state.U, self.ops, self.mesh)
    
