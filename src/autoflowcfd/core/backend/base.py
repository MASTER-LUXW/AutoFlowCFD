"""SolutionVector：后处理用的单元平均守恒量与原始量访问器。"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from autoflowcfd.core.thermo import GAMMA as THERMO_GAMMA


@dataclass
class SolutionVector:
    """解向量数据结构。

    以**守恒**形式存储所有单元的流场解（求解器实际积分的就是这个，
    checkpoint 也是存在 `solution/conserved` 下）：
    - data[:, 0]: rho（密度）
    - data[:, 1:4]: rho*u, rho*v, rho*w（动量）
    - data[:, 4]: rho*E（总能密度）
    - 2026-10-05 以前写出的 SST checkpoint 还有 data[:, 5:7]：状态数组的两个历史槽位（从未被更新，
      已删除；湍流量在模型对象上，后处理读 `core/turbulence/output.py` 的单元平均）

    get_velocity()/get_pressure() 把守恒量换成**原始**量（真实速度、静压），供后处理（VTK 导出、瞬态统计）使用；
    求解器自己的残差 / 边界条件代码不经过它们。

    Attributes:
        data: 解数组，形状=(n_cells, n_variables)
        n_cells: 单元数
        n_variables: 每个单元的变量数
    """
    data: Optional[np.ndarray] = None
    n_cells: int = 0
    n_variables: int = 5

    # 比热比（core/thermo.py 唯一定义）
    GAMMA = THERMO_GAMMA
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


