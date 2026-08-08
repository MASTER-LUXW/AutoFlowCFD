"""CPU/GPU 计算的 backend 抽象基类。"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
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


class BackendBase(ABC):
    """求解器 backend 的抽象基类。

    定义所有计算 backend（CPU/Numba、GPU/CUDA）必须实现的接口，为 FR
    求解器提供统一的 API 来对接不同的硬件加速器。

    Attributes:
        backend_type: 类型标识（'cpu' 或 'gpu'）
        available: 当前系统上该 backend 是否可用
        device_info: 硬件信息字典
    """

    def __init__(self):
        """初始化 backend 基类。"""
        self.backend_type = "base"
        self.available = False
        self.device_info: Dict[str, Any] = {}

    @abstractmethod
    def initialize(
        self,
        n_cells: int,
        n_nodes: int,
        n_variables: int = 5
    ) -> None:
        """分配内存并初始化数据结构。

        Args:
            n_cells: 网格单元数
            n_nodes: 网格节点数
            n_variables: 解变量个数（默认 5，对应可压缩流）
        """
        pass

    @abstractmethod
    def compute_flux(
        self,
        solution: np.ndarray,
        cell_connectivity: np.ndarray,
        face_normals: np.ndarray,
        gamma: float = 1.4
    ) -> np.ndarray:
        """计算所有单元界面上的数值通量。

        Args:
            solution: 解向量，形状=(n_cells, n_variables)
            cell_connectivity: 单元连接关系数组
            face_normals: 面法向量
            gamma: 比热比

        Returns:
            界面上的通量张量
        """
        pass

    @abstractmethod
    def compute_residuals(
        self,
        solution: np.ndarray,
        flux: np.ndarray,
        cell_volumes: np.ndarray,
        boundary_mask: np.ndarray
    ) -> np.ndarray:
        """由通量散度计算残差。

        Args:
            solution: 当前解状态
            flux: 已算好的界面通量
            cell_volumes: 单元体积
            boundary_mask: 边界条件掩码

        Returns:
            残差向量，形状=(n_cells, n_variables)
        """
        pass

    @abstractmethod
    def update_solution(
        self,
        solution: np.ndarray,
        residuals: np.ndarray,
        dt: float,
        cfl: float
    ) -> np.ndarray:
        """用时间积分格式更新解。

        Args:
            solution: 当前解
            residuals: 已算好的残差
            dt: 时间步长
            cfl: CFL 数

        Returns:
            更新后的解
        """
        pass

    @abstractmethod
    def apply_boundary_conditions(
        self,
        solution: np.ndarray,
        boundary_map: Dict[str, np.ndarray],
        bc_params: Dict[str, Any]
    ) -> np.ndarray:
        """把边界条件应用到解上。

        Args:
            solution: 解向量
            boundary_map: 边界名到单元索引的映射
            bc_params: 边界条件参数

        Returns:
            应用边界条件后的解
        """
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """同步数据（对 GPU 异步操作很重要）。"""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取硬件设备信息。

        Returns:
            包含设备规格的字典
        """
        pass

    def cleanup(self) -> None:
        """释放已分配的资源。"""
        pass
