"""压力功率谱密度 (PSD) 分析模块。

本模块从 transient_stats.py 中拆分出来（该文件超过 400 行硬性拆分阈值）：
`PressurePSD` 是一个独立的类，与同文件中的 `TransientStatistics`/
`TransientResult` 没有任何相互依赖。`autoflowcfd.postprocess` 包的 `__init__.py`
从此处重新导出 `PressurePSD`。

谱估计用 Welch 法（`scipy.signal.welch`：分段、加窗、重叠平均，密度谱缩放含窗能量修正，单边谱只倍乘
0 < f < f_N 的频点）。2026-10-20 以前是单段无窗周期图：方差不随样本数下降，偶数长度时 Nyquist 频点也被乘 2，
且只用构造时给的 dt、从不检查记录的时刻是否等间隔。

Example:
    >>> import numpy as np
    >>> from autoflowcfd.postprocess import PressurePSD
    >>> psd = PressurePSD(monitor_points=[(0, 0, 0)])
    >>> for k in range(256):
    ...     psd.add_sample(time=k * 1e-4, pressures=[101325.0 + np.sin(2 * np.pi * 500 * k * 1e-4)])
    >>> freqs, psd_values = psd.compute_psd(point_index=0)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

#: 采样间隔相对中位数的允许偏差（只容许时刻累加的舍入）
SAMPLING_UNIFORMITY_RTOL = 1e-6
#: 少于这么多个样本时用单段 Hann 窗；否则默认约 8 段、50% 重叠
_SINGLE_SEGMENT_BELOW = 64
_MIN_SAMPLES = 8


class PressurePSD:
    """压力功率谱密度分析器：各监测点压力时间序列的 Welch 谱估计与主导频率。

    Attributes:
        monitor_points: 监测点坐标
        pressure_history: 每个监测点的压力时间序列
        times: 样本时刻（必须严格递增且等间隔，采样频率由它给出）
    """

    def __init__(self, monitor_points: List[Tuple[float, float, float]]):
        """
        Args:
            monitor_points: 监测点 (x, y, z) 坐标列表

        Raises:
            ValueError: monitor_points 为空
        """
        if not monitor_points:
            raise ValueError("At least one monitor point required")
        self.monitor_points = monitor_points
        self.pressure_history: Dict[int, List[float]] = {i: [] for i in range(len(monitor_points))}
        self.times: List[float] = []
        logger.info(f"PressurePSD initialized: {len(monitor_points)} monitor point(s)")

    def add_sample(self, time: float, pressures: List[float]) -> None:
        """在各监测点添加一个压力样本。

        Raises:
            ValueError: 长度不匹配，或时刻不晚于上一个样本
        """
        if len(pressures) != len(self.monitor_points):
            raise ValueError(f"Expected {len(self.monitor_points)} pressures, got {len(pressures)}")
        if self.times and not time > self.times[-1]:
            raise ValueError(f"样本时刻必须严格递增：{time} 不晚于上一个样本 {self.times[-1]}")
        self.times.append(float(time))
        for i, p in enumerate(pressures):
            self.pressure_history[i].append(float(p))

    def sampling_interval(self) -> float:
        """由样本时刻给出的采样间隔。

        Raises:
            RuntimeError: 样本少于 2 个
            ValueError: 采样不等间隔（自适应时间步、跳过的样本）——频率轴会错，必须按固定间隔采样
        """
        if len(self.times) < 2:
            raise RuntimeError(f"Need at least 2 samples to determine the sampling interval, got {len(self.times)}")
        steps = np.diff(np.asarray(self.times))
        dt = float(np.median(steps))
        worst = int(np.argmax(np.abs(steps - dt)))
        if abs(steps[worst] - dt) > SAMPLING_UNIFORMITY_RTOL * dt:
            raise ValueError(
                f"PSD 要求等间隔采样：中位间隔 {dt:.6e} s，第 {worst + 1} 个间隔为 {steps[worst]:.6e} s"
                f"（相对偏差 {abs(steps[worst] - dt) / dt:.2e}）。请按固定物理时间间隔保存样本。")
        return dt

    def compute_psd(
        self,
        point_index: int,
        nperseg: Optional[int] = None,
        overlap: float = 0.5,
        window: str = "hann",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """指定监测点压力脉动的单边功率谱密度（Pa^2/Hz），Welch 法。

        Args:
            point_index: 监测点索引
            nperseg: 每段样本数；None 时样本少于 64 个取全长单段，否则取约 8 段（50% 重叠）的段长
            overlap: 相邻段重叠比例，[0, 1)
            window: 窗函数名（scipy.signal.get_window 接受的）

        Returns:
            (频率 Hz, PSD)

        Raises:
            IndexError: point_index 无效
            RuntimeError: 样本少于 8 个
            ValueError: 采样不等间隔或参数无效
        """
        from scipy.signal import welch

        if point_index < 0 or point_index >= len(self.monitor_points):
            raise IndexError(f"Point index {point_index} out of range [0, {len(self.monitor_points) - 1}]")
        signal = np.asarray(self.pressure_history[point_index], dtype=np.float64)
        n = len(signal)
        if n < _MIN_SAMPLES:
            raise RuntimeError(f"Insufficient samples for PSD: need ≥{_MIN_SAMPLES}, got {n}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        if nperseg is None:
            nperseg = n if n < _SINGLE_SEGMENT_BELOW else max(_MIN_SAMPLES, int(n / (1.0 + 7.0 * (1.0 - overlap))))
        if not _MIN_SAMPLES <= nperseg <= n:
            raise ValueError(f"nperseg must be in [{_MIN_SAMPLES}, {n}], got {nperseg}")
        dt = self.sampling_interval()
        freqs, psd = welch(signal, fs=1.0 / dt, window=window, nperseg=nperseg,
                           noverlap=int(round(overlap * nperseg)), detrend="constant", scaling="density")
        logger.info(
            f"PSD for monitor point {point_index}: {n} samples, {nperseg}-sample {window} segments "
            f"({overlap:.0%} overlap), f_max={freqs[-1]:.1f} Hz, df={freqs[1] - freqs[0]:.3g} Hz")
        return freqs, psd

    def find_dominant_frequency(
        self,
        point_index: int,
        min_freq: float = 1.0,
        max_freq: float = 1000.0,
        **psd_options,
    ) -> Tuple[float, float]:
        """在 [min_freq, max_freq] 内 PSD 最大的频率及其 PSD 值（psd_options 原样传给 compute_psd）。

        Raises:
            ValueError: 范围内没有频点
        """
        freqs, psd = self.compute_psd(point_index, **psd_options)
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(mask):
            raise ValueError(f"No frequencies in range [{min_freq}, {max_freq}] Hz")
        peak = int(np.argmax(np.where(mask, psd, -np.inf)))
        logger.info(f"Dominant frequency at point {point_index}: {freqs[peak]:.2f} Hz (PSD={psd[peak]:.2e})")
        return float(freqs[peak]), float(psd[peak])
