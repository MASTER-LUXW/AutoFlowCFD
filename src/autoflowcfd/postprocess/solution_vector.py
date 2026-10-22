"""SolutionVector：单元积分平均的守恒解，后处理（VTK 导出、瞬态统计）经它取原始量。

2026-10-22（I-93）从 `core/backend/base.py` 移来：它只被后处理使用，与"后端"无关。
"""

import numpy as np

from autoflowcfd.core.thermo import GAMMA

#: 平均流守恒变量数：rho, rho*u, rho*v, rho*w, rho*E
N_CONSERVED = 5


class SolutionVector:
    """单元积分平均的守恒解（求解器积分的就是守恒量，checkpoint 存在 `solution/conserved` 下）。

    Args:
        data: (n_cells, >= 5) 守恒量 rho, rho*u, rho*v, rho*w, rho*E。2026-10-05 以前写出的 SST checkpoint 还有
            第 5、6 列（状态数组的两个历史槽位，从未被更新、已删除；湍流量在模型对象上，后处理读
            `core/turbulence/output.py` 的单元平均），这里只取前 5 列。

    Raises:
        ValueError: `data` 不是二维数组或少于 5 列

    `get_velocity()` / `get_pressure()` 按原值换算，不对密度做下限钳制：密度非正的单元（发散的解）照实给出，
    而不是被一个下限掩盖。
    """

    def __init__(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < N_CONSERVED:
            raise ValueError(f"守恒解应为 (n_cells, >= {N_CONSERVED}) 数组，得到形状 {data.shape}")
        self.data = data[:, :N_CONSERVED]

    @property
    def n_cells(self) -> int:
        return int(self.data.shape[0])

    def get_velocity(self) -> tuple:
        """原始速度分量 (u, v, w)：动量除以密度。"""
        rho = self.data[:, 0]
        return self.data[:, 1] / rho, self.data[:, 2] / rho, self.data[:, 3] / rho

    def get_pressure(self) -> np.ndarray:
        """理想气体静压 p = (gamma - 1) (rho E - |rho V|^2 / (2 rho))。"""
        rho = self.data[:, 0]
        kinetic = 0.5 * np.sum(self.data[:, 1:4] ** 2, axis=1) / rho
        return (GAMMA - 1.0) * (self.data[:, 4] - kinetic)
