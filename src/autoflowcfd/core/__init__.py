"""AutoFlowCFD 核心求解器模块。

提供 AutoFlowCFD 的核心计算组件：FR 离散、湍流模型、求解器后端。

生产求解路径（FRSolver.solve() / TransientSolver.solve()）由
BoundaryConditionHandler、ViscousRANSResidual（fvm_viscous_residual.py，
拆成 fvm_residual_inviscid/viscous/sst.py 三个 mixin）、TimeIntegrator、
FVMFaceExtractor（纯粹作为数据持有者，由 VolumeMeshData 直接填充）和
AeroCoefficientCalculator 构成。RANS-SST 的完整物理（AUSM+up、粘性通量、
SST 涡粘性/源项、Green-Gauss 梯度）已经移植成 Numba（CPU）kernel 并直接
在 `ViscousRANSResidual` 内部 dispatch，CUDA（GPU）版本也已写好、通过
`use_gpu=True`（由 `--backend gpu` 触发）dispatch，但**从未在真实 GPU
硬件上验证过**（开发环境无 GPU）——见 `core/fvm_*_kernels_gpu.py` 各自
的模块文档字符串。`core/backend/`（`create_backend`/`NumbaBackend`/
`CUDABackend`）是一套独立的、更早期的无粘 Euler-only kernel 集合，只在
`FRSolver.__init__`/`TransientSolver.__init__` 里被构造用于硬件可用性
检查和日志提示，不参与实际残差计算——见 `core/backend/cpu_backend.py`
模块文档字符串里对这个历史分层的说明。
"""

from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .transient_result import TransientResult
from .transient_solver_loop import TransientSolver
from .solver_steady import FRSolver, SteadyResult
from .backend import create_backend, get_available_backends
from .backend.base import BackendBase
from .backend.cpu_backend import NumbaBackend
from .backend.gpu_backend import CUDABackend

from .fvm_faces import FVMFaceExtractor
from .bc_handler import BoundaryConditionHandler
from .aero_coeffs import AeroCoefficientCalculator


__all__ = [
    # 时间积分
    "TimeIntegrator",
    "TimeIntegrationScheme",

    # 稳态求解器
    "FRSolver",
    "SteadyResult",

    # 瞬态求解器
    "TransientSolver",
    "TransientResult",

    # Backend
    "create_backend",
    "get_available_backends",
    "BackendBase",
    "NumbaBackend",
    "CUDABackend",

    # FVM 核心模块
    "FVMFaceExtractor",
    "BoundaryConditionHandler",
    "AeroCoefficientCalculator",
]
