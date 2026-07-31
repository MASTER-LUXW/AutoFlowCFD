"""Transient statistics post-processing module.

This module provides tools for statistical analysis of transient simulation results,
including time-averaged fields, RMS fluctuations, and PSD spectral analysis.

Key Components:
    - TransientStatistics: Time-averaging, RMS, PSD calculation
    - PressurePSD: Power spectral density analysis

Example:
    >>> from autoflowcfd.postprocess import TransientStatistics
    >>> stats = TransientStatistics(grid_data)
    >>> stats.accumulate(solution, time=0.1)
    >>> mean_field = stats.compute_mean()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field

from ..grid.structures import GridData
from ..core.backend.base import SolutionVector
from ._field_utils import cell_to_node


@dataclass
class TransientResult:
    """Transient statistics result
    
    Attributes:
        mean_fields: Time-averaged flow fields
        rms_fields: RMS fluctuation fields
        sampling_time: Total sampling time (seconds)
        num_samples: Number of samples collected
    """
    mean_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    rms_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    sampling_time: float = 0.0
    num_samples: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding arrays)"""
        return {
            'sampling_time': self.sampling_time,
            'num_samples': self.num_samples,
            'mean_fields': list(self.mean_fields.keys()),
            'rms_fields': list(self.rms_fields.keys())
        }


class TransientStatistics:
    """Transient flow field statistics calculator
    
    Accumulates transient solutions over time to compute:
    - Time-averaged fields (mean velocity, pressure)
    - RMS fluctuations (u', v', w', p')
    - Turbulent kinetic energy fluctuations
    
    Uses sliding window approach for efficient memory usage.
    
    Attributes:
        grid_data: Grid data object
        samples: List of solution snapshots
        times: Sampling times
        window_size: Sliding window size (max samples to keep)
    
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
        """Initialize transient statistics calculator
        
        Args:
            grid_data: Grid data object
            window_size: Maximum number of samples to retain (sliding window)
            
        Raises:
            ValueError: Invalid window_size
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        
        self.grid_data = grid_data
        self.window_size = window_size
        self.samples: List[SolutionVector] = []
        self.times: List[float] = []
        
        # Accumulators for online statistics
        self.n_samples = 0
        self.mean_accumulator: Optional[Dict[str, np.ndarray]] = None
        self.m2_accumulator: Optional[Dict[str, np.ndarray]] = None  # For variance
        
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
        """Accumulate solution snapshot for statistics
        
        Adds a solution snapshot to the sliding window and updates
        running statistics using Welford's online algorithm.
        
        Args:
            solution: Flow field solution vector
            time: Physical time of this snapshot
            
        Example:
            >>> stats.accumulate(solution, time=0.1)
        """
        # Add to sliding window
        self.samples.append(solution)
        self.times.append(time)
        
        # Enforce window size
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
            self.times.pop(0)
        
        # Update online statistics
        self._update_online_stats(solution)
        
        self.n_samples += 1
        
        if self.n_samples % 10 == 0:
            logger.info(
                f"Accumulated {self.n_samples} samples, "
                f"time range: [{self.times[0]:.4f}, {self.times[-1]:.4f}] s"
            )
    
    def _update_online_stats(self, solution: SolutionVector) -> None:
        """Update running statistics using Welford's algorithm

        This allows computing mean and variance in a single pass
        without storing all samples.

        Args:
            solution: Current solution snapshot
        """
        n_points = self.grid_data.metadata.node_count

        if solution.data is not None and solution.n_cells > 0:
            u, v, w = solution.get_velocity()
            p = solution.get_pressure()
            conn = np.asarray(self.grid_data.cells.connectivity)
            volumes = getattr(self.grid_data.cells, "volumes", None)

            if solution.n_cells == n_points:
                # Already node-resolution data - use directly.
                fields = {'velocity_u': u, 'velocity_v': v, 'velocity_w': w, 'pressure': p}
            else:
                # Cell-centered FVM data - interpolate to nodes (this used
                # to just build all-zero arrays here regardless of the
                # actual solution passed in, so every mean/RMS statistic
                # came out exactly zero no matter what flow was simulated).
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
            # First sample: initialize accumulators
            self.mean_accumulator = {k: v.copy() for k, v in fields.items()}
            self.m2_accumulator = {k: np.zeros_like(v) for k, v in fields.items()}
        else:
            # Update mean and M2 (sum of squared differences)
            for key in fields:
                delta = fields[key] - self.mean_accumulator[key]
                self.mean_accumulator[key] += delta / self.n_samples
                delta2 = fields[key] - self.mean_accumulator[key]
                self.m2_accumulator[key] += delta * delta2
    
    def compute_statistics(self) -> TransientResult:
        """Compute time-averaged and RMS statistics
        
        Returns:
            TransientResult: Statistical results
            
        Raises:
            RuntimeError: No samples accumulated
            
        Example:
            >>> result = stats.compute_statistics()
            >>> print(f"Mean velocity: {result.mean_fields['velocity_u']}")
        """
        if self.n_samples == 0:
            raise RuntimeError("No samples accumulated. Call accumulate() first.")
        
        logger.info(f"Computing statistics from {self.n_samples} samples...")
        
        # Compute mean fields
        mean_fields = self.mean_accumulator.copy() if self.mean_accumulator else {}
        
        # Compute RMS (root mean square) fluctuations
        rms_fields = {}
        if self.m2_accumulator:
            for key in self.m2_accumulator:
                variance = self.m2_accumulator[key] / (self.n_samples - 1)
                rms_fields[f'{key}_rms'] = np.sqrt(variance)
        
        # Calculate total sampling time
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
        """Get sampling information
        
        Returns:
            Dict: Sampling metadata
        """
        return {
            'total_samples': self.n_samples,
            'window_size': self.window_size,
            'current_samples': len(self.samples),
            'time_range': [self.times[0], self.times[-1]] if self.times else [0.0, 0.0],
            'sampling_duration': self.times[-1] - self.times[0] if len(self.times) >= 2 else 0.0
        }


class PressurePSD:
    """Pressure power spectral density analyzer
    
    Performs FFT-based spectral analysis on pressure time series
    at specified monitoring points to identify dominant frequencies.
    
    Attributes:
        monitor_points: Coordinates of monitoring points
        pressure_history: Pressure time series at each point
        dt: Time step size
    
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
        """Initialize PSD analyzer
        
        Args:
            monitor_points: List of (x, y, z) coordinates for monitoring
            dt: Time step size (seconds)
            
        Raises:
            ValueError: Invalid dt or empty monitor_points
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
        """Add pressure sample at monitoring points
        
        Args:
            time: Physical time
            pressures: Pressure values at each monitor point
            
        Raises:
            ValueError: Length mismatch
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
        """Compute power spectral density at specified point
        
        Args:
            point_index: Index of monitor point
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequencies (Hz) and PSD values
            
        Raises:
            IndexError: Invalid point_index
            RuntimeError: Insufficient samples
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
        
        # Convert to numpy array
        signal = np.array(history)
        
        # Remove mean (analyze fluctuations)
        signal = signal - np.mean(signal)
        
        # Compute FFT
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(n, d=self.dt)
        
        # Compute PSD (power spectral density)
        psd = (np.abs(fft_vals) ** 2) / (n * (1 / self.dt))
        
        # Normalize (one-sided spectrum)
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
        """Find dominant frequency in specified range
        
        Args:
            point_index: Index of monitor point
            min_freq: Minimum frequency to search (Hz)
            max_freq: Maximum frequency to search (Hz)
            
        Returns:
            Tuple[float, float]: Dominant frequency (Hz) and PSD value
        """
        freqs, psd = self.compute_psd(point_index)
        
        # Filter to frequency range
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        filtered_freqs = freqs[mask]
        filtered_psd = psd[mask]
        
        if len(filtered_freqs) == 0:
            raise ValueError(
                f"No frequencies in range [{min_freq}, {max_freq}] Hz"
            )
        
        # Find peak
        peak_idx = np.argmax(filtered_psd)
        dominant_freq = filtered_freqs[peak_idx]
        peak_psd = filtered_psd[peak_idx]
        
        logger.info(
            f"Dominant frequency at point {point_index}: "
            f"{dominant_freq:.2f} Hz (PSD={peak_psd:.2e})"
        )
        
        return dominant_freq, peak_psd
