"""SolutionVector 数据结构。

此前本文件还定义了 `BackendBase` 抽象基类，供 `cpu_backend.py`
(`NumbaBackend`) / `gpu_backend.py` (`CUDABackend`) 实现——这两个类是
V1 时代 Numba CUDA 方案的遗留骨架，与真正的生产求解路径
（`core/fr_solver/solver.py::FRSolver` / `core/gpu/gpu_solver.py::
GPUFRSolver`，全程 CuPy）完全脱节：`_cuda_flux_kernel` 用"中心平均"
冒充 AUSM+up、`_cuda_residual_kernel` 恒返回 0、GPU 可用性检测用的是
已被项目明确弃用的 `numba.cuda.is_available()`（见
`ProjectFiles/V2.0/7_重大问题修复-GPU大规模并行计算.md` "统一采用
CuPy……移除 Numba CUDA" 的既定决策，此前从未真正执行），且被自己的
单元测试（已删除的 `test_backends.py`）当作真实可用的 GPU 后端做回归
测试，制造"这是一套可用 GPU 后端"的假象（第四次评审发现7）。已确认
`BackendBase`/`NumbaBackend`/`CUDABackend`/`create_backend` 全仓库
零真实调用点（只有已删除的测试用到），第四次评审时一并删除，只保留
真正被生产代码使用的 `SolutionVector`。
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class SolutionVector:
    """解向量数据结构。

    以**守恒**形式存储所有单元的流场解（求解器实际积分的就是这个，
    checkpoint 也是存在 `solution/conserved` 下）：
    - data[:, 0]: rho（密度）
    - data[:, 1:4]: rho*u, rho*v, rho*w（动量）
    - data[:, 4]: rho*E（总能密度）
    - data[:, 5:7]: rho*k, rho*omega（湍流量，可选）

    下面的 get_velocity()/get_pressure()/get_turbulence() 访问器会把
    这些量转换成方法名所承诺的**原始**量（真实速度、静压、k 和 omega）
    ——以前这几个方法直接原样返回未转换的守恒量列（例如所谓的
    "velocity" 其实是动量，"pressure" 其实是总能密度），会给任何调用方
    悄悄地把数值标错好几个数量级。这里保留是为了向后兼容，但要注意
    求解器自己的残差/边界条件代码**不**使用它们——那部分代码是按自己
    的 gamma/下限约定就地推导原始量的（例如见 core/aero_coeffs.py）。

    Attributes:
        data: 解数组，形状=(n_cells, n_variables)
        n_cells: 单元数
        n_variables: 每个单元的变量数
    """
    data: Optional[np.ndarray] = None
    n_cells: int = 0
    n_variables: int = 5

    # 比热比，与求解器全局使用的状态方程一致（例如 core/aero_coeffs.py）。
    GAMMA = 1.4
    _RHO_FLOOR = 1e-10

    def __post_init__(self):
        """若未提供 data，则初始化数组"""
        if self.data is None and self.n_cells > 0:
            self.data = np.zeros((self.n_cells, self.n_variables))

    @property
    def shape(self):
        """获取解数组的形状"""
        if self.data is not None:
            return self.data.shape
        return (0, 0)

    def get_density(self) -> np.ndarray:
        """获取密度场"""
        if self.data is not None and self.data.shape[1] > 0:
            return self.data[:, 0]
        return np.array([])

    def get_velocity(self) -> tuple:
        """获取原始速度分量 (u, v, w)，即动量除以 rho。"""
        if self.data is not None and self.data.shape[1] >= 4:
            rho = np.maximum(self.data[:, 0], self._RHO_FLOOR)
            return (self.data[:, 1] / rho, self.data[:, 2] / rho, self.data[:, 3] / rho)
        return (np.array([]), np.array([]), np.array([]))

    def get_pressure(self) -> np.ndarray:
        """通过理想气体状态方程获取静压，
        p = (gamma-1) * (rho*E - 0.5*rho*|V|^2) —— 而不是直接返回原始的 rho*E 列。"""
        if self.data is not None and self.data.shape[1] >= 5:
            rho = np.maximum(self.data[:, 0], self._RHO_FLOOR)
            rhoE = self.data[:, 4]
            V_sq = (self.data[:, 1]**2 + self.data[:, 2]**2 + self.data[:, 3]**2) / rho**2
            return (self.GAMMA - 1.0) * (rhoE - 0.5 * rho * V_sq)
        return np.array([])

    def get_turbulence(self) -> tuple:
        """获取原始湍流量 (k, omega)，即把守恒形式 (rho*k, rho*omega)
        列除以密度。若该解没有湍流量列，返回两个空数组。"""
        if self.data is not None and self.data.shape[1] >= 7:
            rho = np.maximum(self.data[:, 0], self._RHO_FLOOR)
            return (self.data[:, 5] / rho, self.data[:, 6] / rho)
        return (np.array([]), np.array([]))
