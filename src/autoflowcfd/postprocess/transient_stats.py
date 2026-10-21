"""瞬态统计：节点上的时间平均与 RMS 脉动（速度、压力），以及气动系数的时间平均与标准差。

样本是各时刻的单元积分平均解；节点场经混合网格体积加权插值（`_field_utils.cell_to_node`，与低阶 VTK
导出同一实现）。均值与方差用 Welford 在线算法逐样本更新（不保留样本），方差取无偏估计（ddof=1）。

Example:
    >>> stats = TransientStatistics(grid_data)
    >>> for t, sol in samples:
    ...     stats.accumulate(sol, time=t)
    >>> result = stats.compute_statistics()
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from loguru import logger

from ..core.backend.base import SolutionVector
from ._field_utils import cell_to_node


class RunningMoments:
    """Welford 在线均值/方差（标量或数组）。"""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        if self.mean is None:
            self.mean = x.copy()
            self.m2 = np.zeros_like(x)
            return
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        self.m2 = self.m2 + delta * (x - self.mean)

    def variance(self):
        """无偏方差；样本数不足 2 时为 0。"""
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.n - 1)


@dataclass
class TransientResult:
    """瞬态统计结果。

    Attributes:
        mean_fields: 节点时均场 {'velocity_u','velocity_v','velocity_w','pressure'}
        rms_fields: 节点 RMS 场 {'<名>_rms'}
        sampling_time: 首末样本的物理时间差
        num_samples: 样本数
        mean_coefficients / std_coefficients: 气动系数的时均与标准差
    """
    mean_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    rms_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    sampling_time: float = 0.0
    num_samples: int = 0
    mean_coefficients: Dict[str, float] = field(default_factory=dict)
    std_coefficients: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'sampling_time': self.sampling_time,
            'num_samples': self.num_samples,
            'mean_fields': list(self.mean_fields.keys()),
            'rms_fields': list(self.rms_fields.keys()),
            'mean_coefficients': self.mean_coefficients,
            'std_coefficients': self.std_coefficients,
        }


_FIELD_NAMES = ('velocity_u', 'velocity_v', 'velocity_w', 'pressure')
_COEFF_NAMES = ('Cd', 'Cl', 'Cs')


class TransientStatistics:
    """逐样本累积节点时均/RMS 与气动系数统计。

    Args:
        grid_data: 体网格（棱柱在前、四面体在后）或面网格；样本的单元顺序须与之一致
    """

    def __init__(self, grid_data) -> None:
        self.grid_data = grid_data
        self._fields = {name: RunningMoments() for name in _FIELD_NAMES}
        self._coeffs = {name: RunningMoments() for name in _COEFF_NAMES}
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None

    @property
    def n_samples(self) -> int:
        return self._fields['pressure'].n

    @property
    def n_coeff_samples(self) -> int:
        return self._coeffs['Cd'].n

    def accumulate(self, solution: SolutionVector, time: float) -> None:
        """加入一个时刻的单元积分平均解（单元数必须与网格一致）。"""
        n_cells = int(self.grid_data.cell_count)
        if solution is None or solution.data is None or solution.n_cells != n_cells:
            got = None if solution is None or solution.data is None else solution.n_cells
            raise ValueError(f"样本解的单元数 {got} 与网格单元数 {n_cells} 不符")
        u, v, w = solution.get_velocity()
        values = {'velocity_u': u, 'velocity_v': v, 'velocity_w': w, 'pressure': solution.get_pressure()}
        for name, cell_values in values.items():
            self._fields[name].update(cell_to_node(self.grid_data, cell_values))
        if self.t_first is None:
            self.t_first = float(time)
        self.t_last = float(time)

    def accumulate_coefficients(self, Cd: float, Cl: float, Cs: float) -> None:
        """加入一个时刻的气动系数。"""
        for name, value in zip(_COEFF_NAMES, (Cd, Cl, Cs)):
            self._coeffs[name].update(value)

    def get_coefficient_statistics(self) -> Dict[str, Dict[str, float]]:
        """{系数: {'mean','std','n_samples'}}；没有系数样本时为空字典。"""
        if self.n_coeff_samples == 0:
            return {}
        return {name: {'mean': float(m.mean), 'std': float(np.sqrt(max(float(m.variance()), 0.0))),
                       'n_samples': m.n} for name, m in self._coeffs.items()}

    def compute_statistics(self) -> TransientResult:
        if self.n_samples == 0 or self.t_first is None or self.t_last is None:
            raise RuntimeError("还没有累积任何样本")
        coeff = self.get_coefficient_statistics()
        result = TransientResult(
            mean_fields={name: m.mean.copy() for name, m in self._fields.items()},
            rms_fields={f'{name}_rms': np.sqrt(np.maximum(m.variance(), 0.0)) for name, m in self._fields.items()},
            sampling_time=self.t_last - self.t_first,
            num_samples=self.n_samples,
            mean_coefficients={k: v['mean'] for k, v in coeff.items()},
            std_coefficients={k: v['std'] for k, v in coeff.items()},
        )
        logger.info(f"瞬态统计：{result.num_samples} 个样本，采样时长 {result.sampling_time:.4f} s")
        return result
