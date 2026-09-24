"""
AutoFlowCFD V2.0 - GPU FRSolver

完整的 GPU 版 FR 求解器，对应 core/fr_solver.py。
所有计算在 GPU 上完成，数据常驻显存，只在 I/O 时传输。

设计：
- 与 CPU 版 FRSolver 接口一致（solve/step/compute_*_residual）
- 内部使用 GPUArrayManager 管理 GPU 数据
- 残差计算全部走 CuPy（gpu_inviscid.py / gpu_viscous.py）
- 时间积分走 GPUTimeIntegrator（gpu_time_integration.py）
- 支持 P0 和 P>=1 两种路径
- 支持单 GPU 稳态/伪稳态求解

使用:
    solver = GPUFRSolver(mesh, ops, order=2, device_id=0)
    result = solver.solve(max_iter=1000, dt=1e-4, tol=1e-6)
"""

# 本模块 2026-09-24 按职责拆成子包（项目「单文件不超 500 行」规范）。
# 下面 re-export 全部公开名与测试在用的私有名，所以全仓库
# `from autoflowcfd.core.gpu.solver.gpu_solver import ...` 一个字都不用改。

from .residual import (  # noqa: F401
    _GPUSolverResidualMixin,
)
from .timestep import (  # noqa: F401
    _GPUSolverTimeStepMixin,
)
from .step import (  # noqa: F401
    _GPUSolverStepMixin,
)
from .core import (  # noqa: F401
    GPUFRSolver,
)

__all__ = [
    "GPUFRSolver",
]
