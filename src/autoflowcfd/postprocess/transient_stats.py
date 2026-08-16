"""瞬态统计后处理模块。

本模块提供瞬态仿真结果的统计分析工具，包括时间平均场和 RMS 脉动。

PSD 频谱分析（`PressurePSD`）原本也在本文件中，因本文件超过 400 行
硬性拆分阈值，已拆分至独立的 pressure_psd.py——`PressurePSD` 与本文件
的 `TransientStatistics`/`TransientResult` 没有相互依赖，是一处干净的
拆分点。`autoflowcfd.postprocess` 包仍统一从 `.pressure_psd` 重新导出
`PressurePSD`，外部调用方无需改动。

Key Components:
    - TransientStatistics: 时间平均、RMS 计算

Example:
    >>> from autoflowcfd.postprocess import TransientStatistics
    >>> stats = TransientStatistics(grid_data)
    >>> stats.accumulate(solution, time=0.1)
    >>> mean_field = stats.compute_mean()
"""

import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass, field

from ..grid.structures import GridData
from ..core.backend.base import SolutionVector
from ._field_utils import cell_to_node


@dataclass
class TransientResult:
    """瞬态统计结果。

    Attributes:
        mean_fields: 时间平均流场
        rms_fields: RMS 脉动场
        sampling_time: 总采样时长（秒）
        num_samples: 已采集的样本数
    """
    mean_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    rms_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    sampling_time: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> Dict:
        """转换成字典（不含数组本身）。"""
        return {
            'sampling_time': self.sampling_time,
            'num_samples': self.num_samples,
            'mean_fields': list(self.mean_fields.keys()),
            'rms_fields': list(self.rms_fields.keys())
        }


class TransientStatistics:
    """瞬态流场统计计算器。

    随时间累积瞬态解，计算：
    - 时间平均场（平均速度、压力）
    - RMS 脉动（u'、v'、w'、p'）
    - 湍动能脉动

    采用滑动窗口方式控制内存占用。

    Attributes:
        grid_data: 网格数据对象
        samples: 解快照列表
        times: 采样时间点
        window_size: 滑动窗口大小（最多保留的样本数）

    Example:
        >>> stats = TransientStatistics(grid_data, window_size=100)
        >>> for i, sol in enumerate(transient_solutions):
        ...     stats.accumulate(sol, time=i*dt)
        >>> result = stats.compute_statistics()
    """

    def __init__(
        self,
        grid_data: GridData,
        window_size: int = 100
    ):
        """初始化瞬态统计计算器。

        Args:
            grid_data: 网格数据对象
            window_size: 最多保留的样本数（滑动窗口）

        Raises:
            ValueError: window_size 无效
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")

        self.grid_data = grid_data
        self.window_size = window_size
        self.samples: List[SolutionVector] = []
        self.times: List[float] = []

        # 在线统计的累加器
        self.n_samples = 0
        self.mean_accumulator: Optional[Dict[str, np.ndarray]] = None
        self.m2_accumulator: Optional[Dict[str, np.ndarray]] = None  # 用于方差

        logger.info(
            f"TransientStatistics initialized:\n"
            f"  Window size: {window_size}\n"
            f"  Nodes:       {grid_data.metadata.node_count}"
        )

    def accumulate(
        self,
        solution: SolutionVector,
        time: float
    ) -> None:
        """累积一个解快照用于统计。

        把解快照加入滑动窗口，并用 Welford 在线算法更新累计统计量。

        Args:
            solution: 流场解向量
            time: 该快照对应的物理时间

        Example:
            >>> stats.accumulate(solution, time=0.1)
        """
        # 加入滑动窗口
        self.samples.append(solution)
        self.times.append(time)

        # 保持窗口大小
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
            self.times.pop(0)

        # 更新在线统计量
        self._update_online_stats(solution)

        self.n_samples += 1

        if self.n_samples % 10 == 0:
            logger.info(
                f"Accumulated {self.n_samples} samples, "
                f"time range: [{self.times[0]:.4f}, {self.times[-1]:.4f}] s"
            )

    def _update_online_stats(self, solution: SolutionVector) -> None:
        """用 Welford 算法更新累计统计量。

        这样可以只扫一遍数据就同时算出均值和方差，不需要把所有样本
        都存下来。

        Args:
            solution: 当前解快照
        """
        n_points = self.grid_data.metadata.node_count

        if solution.data is not None and solution.n_cells > 0:
            u, v, w = solution.get_velocity()
            p = solution.get_pressure()
            conn = np.asarray(self.grid_data.cells.connectivity)
            volumes = getattr(self.grid_data.cells, "volumes", None)

            if solution.n_cells == n_points:
                # 已经是节点分辨率的数据——直接用。
                fields = {'velocity_u': u, 'velocity_v': v, 'velocity_w': w, 'pressure': p}
            else:
                # 单元中心的 FVM 数据——插值到节点（这里以前不管传入的
                # solution 实际是什么，都只是构造全零数组，导致无论仿真
                # 的是什么流动，算出来的 mean/RMS 统计量永远精确地等于
                # 零）。
                fields = {
                    'velocity_u': cell_to_node(conn, u, n_points, volumes=volumes),
                    'velocity_v': cell_to_node(conn, v, n_points, volumes=volumes),
                    'velocity_w': cell_to_node(conn, w, n_points, volumes=volumes),
                    'pressure': cell_to_node(conn, p, n_points, volumes=volumes, fallback=101325.0),
                }
        else:
            logger.warning("Solution data not available for this sample - accumulating zeros.")
            fields = {
                'velocity_u': np.zeros(n_points),
                'velocity_v': np.zeros(n_points),
                'velocity_w': np.zeros(n_points),
                'pressure': np.zeros(n_points),
            }

        if self.mean_accumulator is None:
            # 第一个样本：初始化累加器
            self.mean_accumulator = {k: v.copy() for k, v in fields.items()}
            self.m2_accumulator = {k: np.zeros_like(v) for k, v in fields.items()}
        else:
            # 更新均值和 M2（差值平方和）
            for key in fields:
                delta = fields[key] - self.mean_accumulator[key]
                self.mean_accumulator[key] += delta / self.n_samples
                delta2 = fields[key] - self.mean_accumulator[key]
                self.m2_accumulator[key] += delta * delta2

    def compute_statistics(self) -> TransientResult:
        """计算时间平均和 RMS 统计量。

        Returns:
            TransientResult: 统计结果

        Raises:
            RuntimeError: 尚未累积任何样本

        Example:
            >>> result = stats.compute_statistics()
            >>> print(f"Mean velocity: {result.mean_fields['velocity_u']}")
        """
        if self.n_samples == 0:
            raise RuntimeError("No samples accumulated. Call accumulate() first.")

        logger.info(f"Computing statistics from {self.n_samples} samples...")

        # 计算平均场
        mean_fields = self.mean_accumulator.copy() if self.mean_accumulator else {}

        # 计算 RMS（均方根）脉动
        rms_fields = {}
        if self.m2_accumulator:
            for key in self.m2_accumulator:
                variance = self.m2_accumulator[key] / (self.n_samples - 1)
                rms_fields[f'{key}_rms'] = np.sqrt(variance)

        # 计算总采样时长
        if len(self.times) >= 2:
            sampling_time = self.times[-1] - self.times[0]
        else:
            sampling_time = 0.0

        result = TransientResult(
            mean_fields=mean_fields,
            rms_fields=rms_fields,
            sampling_time=sampling_time,
            num_samples=self.n_samples
        )

        logger.success(
            f"Statistics computed:\n"
            f"  Samples:        {result.num_samples}\n"
            f"  Sampling time:  {result.sampling_time:.4f} s\n"
            f"  Mean fields:    {list(result.mean_fields.keys())}\n"
            f"  RMS fields:     {list(result.rms_fields.keys())}"
        )

        return result

    def get_sampling_info(self) -> Dict:
        """获取采样信息。

        Returns:
            Dict: 采样元数据
        """
        return {
            'total_samples': self.n_samples,
            'window_size': self.window_size,
            'current_samples': len(self.samples),
            'time_range': [self.times[0], self.times[-1]] if self.times else [0.0, 0.0],
            'sampling_duration': self.times[-1] - self.times[0] if len(self.times) >= 2 else 0.0
        }
