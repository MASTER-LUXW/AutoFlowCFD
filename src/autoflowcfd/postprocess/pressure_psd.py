"""压力功率谱密度 (PSD) 分析模块。

本模块从 transient_stats.py 中拆分出来（该文件超过 400 行硬性拆分阈值）：
`PressurePSD` 是一个独立的类，与同文件中的 `TransientStatistics`/
`TransientResult` 没有任何相互依赖，是一处干净、低风险的拆分点。拆分后
`transient_stats.py` 只保留时间平均/RMS 统计逻辑，PSD 频谱分析单独
成文件。对外行为完全不变——`autoflowcfd.postprocess` 包的 `__init__.py`
仍然从此处重新导出 `PressurePSD`，外部调用方无需改动。

Example:
    >>> from autoflowcfd.postprocess import PressurePSD
    >>> psd = PressurePSD(monitor_points=[(0, 0, 0)], dt=1e-4)
    >>> psd.add_sample(time=0.1, pressures=[101325.0])
    >>> freqs, psd_values = psd.compute_psd(point_index=0)
"""

import numpy as np
from typing import Dict, List, Tuple
from loguru import logger


class PressurePSD:
    """压力功率谱密度分析器。

    对指定监测点的压力时间序列做基于 FFT 的频谱分析，识别主导频率。

    Attributes:
        monitor_points: 监测点坐标
        pressure_history: 每个监测点的压力时间序列
        dt: 时间步长

    Example:
        >>> psd = PressurePSD(monitor_points=[(0, 0, 0)], dt=1e-4)
        >>> psd.add_sample(time=0.1, pressures=[101325.0])
        >>> freqs, psd_values = psd.compute_psd(point_index=0)
    """

    def __init__(
        self,
        monitor_points: List[Tuple[float, float, float]],
        dt: float
    ):
        """初始化 PSD 分析器。

        Args:
            monitor_points: 监测点 (x, y, z) 坐标列表
            dt: 时间步长（秒）

        Raises:
            ValueError: dt 无效或 monitor_points 为空
        """
        if not monitor_points:
            raise ValueError("At least one monitor point required")
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")

        self.monitor_points = monitor_points
        self.dt = dt
        self.pressure_history: Dict[int, List[float]] = {i: [] for i in range(len(monitor_points))}
        self.times: List[float] = []

        logger.info(
            f"PressurePSD initialized:\n"
            f"  Monitor points: {len(monitor_points)}\n"
            f"  Time step:      {dt:.2e} s"
        )

    def add_sample(
        self,
        time: float,
        pressures: List[float]
    ) -> None:
        """在各监测点添加一个压力样本。

        Args:
            time: 物理时间
            pressures: 各监测点的压力值

        Raises:
            ValueError: 长度不匹配
        """
        if len(pressures) != len(self.monitor_points):
            raise ValueError(
                f"Expected {len(self.monitor_points)} pressures, "
                f"got {len(pressures)}"
            )

        self.times.append(time)
        for i, p in enumerate(pressures):
            self.pressure_history[i].append(p)

    def compute_psd(
        self,
        point_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算指定监测点的功率谱密度。

        Args:
            point_index: 监测点索引

        Returns:
            Tuple[np.ndarray, np.ndarray]: 频率 (Hz) 和 PSD 值

        Raises:
            IndexError: point_index 无效
            RuntimeError: 样本数不足
        """
        if point_index < 0 or point_index >= len(self.monitor_points):
            raise IndexError(
                f"Point index {point_index} out of range "
                f"[0, {len(self.monitor_points)-1}]"
            )

        history = self.pressure_history[point_index]

        if len(history) < 8:
            raise RuntimeError(
                f"Insufficient samples for PSD: need ≥8, got {len(history)}"
            )

        logger.info(f"Computing PSD for monitor point {point_index}...")

        # 转成 numpy 数组
        signal = np.array(history)

        # 去均值（分析脉动量）
        signal = signal - np.mean(signal)

        # 计算 FFT
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(n, d=self.dt)

        # 计算 PSD（功率谱密度）
        psd = (np.abs(fft_vals) ** 2) / (n * (1 / self.dt))

        # 归一化（单边谱）
        psd[1:] *= 2

        logger.info(
            f"PSD computed:\n"
            f"  Samples:     {n}\n"
            f"  Max freq:    {fft_freqs[-1]:.1f} Hz\n"
            f"  Freq res:    {fft_freqs[1]-fft_freqs[0]:.2f} Hz"
        )

        return fft_freqs, psd

    def find_dominant_frequency(
        self,
        point_index: int,
        min_freq: float = 1.0,
        max_freq: float = 1000.0
    ) -> Tuple[float, float]:
        """在指定频率范围内寻找主导频率。

        Args:
            point_index: 监测点索引
            min_freq: 搜索的最低频率 (Hz)
            max_freq: 搜索的最高频率 (Hz)

        Returns:
            Tuple[float, float]: 主导频率 (Hz) 及其 PSD 值
        """
        freqs, psd = self.compute_psd(point_index)

        # 过滤到指定频率范围
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        filtered_freqs = freqs[mask]
        filtered_psd = psd[mask]

        if len(filtered_freqs) == 0:
            raise ValueError(
                f"No frequencies in range [{min_freq}, {max_freq}] Hz"
            )

        # 找峰值
        peak_idx = np.argmax(filtered_psd)
        dominant_freq = filtered_freqs[peak_idx]
        peak_psd = filtered_psd[peak_idx]

        logger.info(
            f"Dominant frequency at point {point_index}: "
            f"{dominant_freq:.2f} Hz (PSD={peak_psd:.2e})"
        )

        return dominant_freq, peak_psd
