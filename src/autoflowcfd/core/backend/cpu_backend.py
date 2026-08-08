"""CPU backend implementation using Numba JIT compilation.

⚠️ 现状说明：这个模块（`NumbaBackend`/`compute_flux`/`compute_residuals`）
是一套独立的、只实现无粘 Euler 通量的 Numba kernel，有自己的单元测试
（`tests/unit/test_backends.py`），但**不是**生产环境求解器（`FRSolver`/
`TransientSolver`）实际调用的计算路径。真正跑 RANS-SST 物理（AUSM+up、
粘性通量、SST 湍流源项）的 Numba/CUDA kernel 在
`core/fvm_inviscid_kernels.py`、`core/fvm_viscous_kernels.py`、
`core/fvm_sst_kernels.py`、`core/fvm_gradients_kernels.py`（CPU）及其
`*_kernels_gpu.py` 对应文件（GPU），直接被 `fvm_viscous_residual.py`
的 `ViscousRANSResidual` 调用，不经过这里的 `BackendBase`/`create_backend`
体系。

`solver_steady.py`/`transient_solver_loop.py` 仍然会构造
`create_backend(...)` 返回的 backend 对象（`self.backend`），但只用它做
硬件可用性检查/日志，不用它做任何实际通量或残差计算——这是历史遗留的
分层：`BackendBase` 体系最初设计成"计算也走这里"，但完整 RANS-SST 物理
的移植（AUSM+up/黏性/SST/壁面函数）最终选择直接写进
`ViscousRANSResidual` 自己的私有方法，风险更低、改动面更小，不需要先把
`ViscousRANSResidual` 的构造/调用签名重构成通过 `BackendBase` 分发。
"""

import numpy as np
from typing import Dict, Any
from .base import BackendBase
from .cpu_backend_kernels_flux import _compute_flux_kernel, _compute_flux_kernel_muscl
from .cpu_backend_kernels_residual import (
    _compute_residuals_kernel,
    _compute_residuals_kernel_fvm,
    _update_solution_kernel,
    _apply_wall_bc,
    _apply_inlet_bc,
)


class NumbaBackend(BackendBase):
    """带 Numba JIT 并行加速的 CPU backend。

    通过 Numba 的即时编译，用 @njit(parallel=True) 自动多线程加速 CPU
    计算。

    Attributes:
        backend_type: 始终为 'cpu'
        available: Numba 已安装则为 True
        n_threads: 并行线程数
    """

    def __init__(self, n_threads: int = 4):
        """初始化 Numba CPU backend。

        Args:
            n_threads: 并行线程数（默认 4）
        """
        super().__init__()
        self.backend_type = "cpu"
        self.available = True
        self.n_threads = n_threads
        self.device_info = {
            "backend": "Numba CPU",
            "threads": n_threads,
            "parallel": True,
        }

    def initialize(self, n_cells: int, n_nodes: int, n_variables: int = 5) -> None:
        """为 CPU 计算预分配数组。

        Args:
            n_cells: 单元数
            n_nodes: 节点数
            n_variables: 每个单元的变量数
        """
        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.n_variables = n_variables

        # 预分配解和残差数组
        self.solution = np.zeros((n_cells, n_variables), dtype=np.float64)
        self.residuals = np.zeros((n_cells, n_variables), dtype=np.float64)
        self.flux = np.zeros((n_cells, n_variables), dtype=np.float64)

    def compute_flux(
        self,
        solution: np.ndarray,
        cell_connectivity: np.ndarray,
        face_normals: np.ndarray,
        gamma: float = 1.4,
    ) -> np.ndarray:
        """用 Numba 加速的一阶 HLLC kernel 计算通量。

        Args:
            solution: 解向量，形状=(n_cells, n_vars)
            cell_connectivity: 单元连接关系（面到单元），形状=(n_faces, 2)
            face_normals: 面法向量，形状=(n_faces, 3)
            gamma: 比热比

        Returns:
            通量张量，形状=(n_faces, n_vars)

        Note:
            要用二阶（MUSCL 重构）通量，请在外部重构 U_L/U_R（见
            fvm_gradients.py 的 green_gauss_gradient +
            barth_jespersen_limiter，也就是实际 RANS-SST 求解路径用的
            那套）后直接调用 `compute_flux_muscl`——这个方法以前接受
            `use_muscl=True` 并在内部通过 `core.legacy.reconstruction`
            做重构，但那个模块已被删除（一套从未被生产求解路径使用的
            死代码并行实现），本代码库里也从来没有任何地方传过
            `use_muscl=True`，所以那个分支是永远不会被执行到的死代码，
            一旦真的被触发就会抛出 ModuleNotFoundError。
        """
        n_faces = face_normals.shape[0]
        flux = np.zeros((n_faces, self.n_variables), dtype=np.float64)
        flux = _compute_flux_kernel(
            solution, cell_connectivity, face_normals, flux, gamma
        )
        return flux

    def compute_flux_muscl(
        self,
        U_L: np.ndarray,
        U_R: np.ndarray,
        face_normals: np.ndarray,
        gamma: float = 1.4,
    ) -> np.ndarray:
        """从预先重构好的左右状态计算 HLLC 通量。

        当 MUSCL 重构在外部完成时（例如在 solver_steady.py 里）调用此
        方法，避免重新创建 reconstructor。

        Args:
            U_L: 界面左侧状态，形状=(n_faces, n_vars)
            U_R: 界面右侧状态，形状=(n_faces, n_vars)
            face_normals: 面法向量，形状=(n_faces, 3)
            gamma: 比热比

        Returns:
            通量张量，形状=(n_faces, n_vars)
        """
        n_faces = face_normals.shape[0]
        flux = np.zeros((n_faces, self.n_variables), dtype=np.float64)

        # 使用 Numba 加速的 MUSCL 通量 kernel
        flux = _compute_flux_kernel_muscl(
            U_L, U_R, face_normals, flux, gamma
        )

        return flux

    def compute_residuals(
        self,
        solution: np.ndarray,
        flux: np.ndarray,
        cell_volumes: np.ndarray,
        boundary_mask: np.ndarray,
        connectivity: np.ndarray = None,  # 加上 connectivity 以支持正确的 FVM
    ) -> np.ndarray:
        """由通量散度计算残差。

        Args:
            solution: 当前解
            flux: 界面通量
            cell_volumes: 单元体积
            boundary_mask: 边界掩码
            connectivity: 单元连接关系数组 (n_faces, 2)

        Returns:
            残差向量
        """
        residuals = np.zeros_like(solution)

        if connectivity is not None:
            # 用带 connectivity 的正确有限体积法
            residuals = _compute_residuals_kernel_fvm(
                solution, flux, cell_volumes, connectivity, residuals
            )
        else:
            # 退回旧的松弛方法（已弃用）
            residuals = _compute_residuals_kernel(
                solution, flux, cell_volumes, boundary_mask, residuals
            )

        return residuals

    def update_solution(
        self, solution: np.ndarray, residuals: np.ndarray, dt: float, cfl: float
    ) -> np.ndarray:
        """用后向欧拉格式更新解。

        Args:
            solution: 当前解
            residuals: 已算好的残差
            dt: 时间步长
            cfl: CFL 数

        Returns:
            更新后的解
        """
        updated = np.copy(solution)

        updated = _update_solution_kernel(updated, residuals, dt, cfl)

        return updated

    def apply_boundary_conditions(
        self,
        solution: np.ndarray,
        boundary_map: Dict[str, np.ndarray],
        bc_params: Dict[str, Any],
    ) -> np.ndarray:
        """应用边界条件。

        Args:
            solution: 解向量
            boundary_map: 边界到单元的映射
            bc_params: 边界条件参数

        Returns:
            应用边界条件后的解
        """
        # 应用壁面边界条件（无滑移）
        if "WALL" in boundary_map:
            wall_cells = boundary_map["WALL"]
            solution = _apply_wall_bc(solution, wall_cells)

        # 应用入口边界条件
        if "INLET" in boundary_map and "inlet_velocity" in bc_params:
            inlet_cells = boundary_map["INLET"]
            velocity = bc_params["inlet_velocity"]
            solution = _apply_inlet_bc(solution, inlet_cells, velocity)

        return solution

    def synchronize(self) -> None:
        """CPU backend 不需要同步。"""
        pass

    def get_device_info(self) -> Dict[str, Any]:
        """获取 CPU 设备信息。

        Returns:
            设备信息字典
        """
        import platform
        import multiprocessing

        return {
            "backend": "Numba CPU",
            "platform": platform.platform(),
            "cpu_count": multiprocessing.cpu_count(),
            "threads_used": self.n_threads,
            "numba_version": "0.56+",
        }
