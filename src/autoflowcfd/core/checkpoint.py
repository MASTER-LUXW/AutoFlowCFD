"""Checkpoint management for AutoFlowCFD solver.

This module provides comprehensive checkpoint save/load functionality using HDF5 format,
supporting cross-backend (CPU↔GPU) restart and configuration validation.

Key Components:
    - CheckpointManager: Main checkpoint handler
    - ConservedVariables serialization
    - ConvergenceHistory serialization
    
Example:
    >>> from autoflowcfd.core.checkpoint import CheckpointManager
    >>> manager = CheckpointManager(config)
    >>> manager.save(solution, history, iteration)
    >>> solution, history, iter_num = manager.load("checkpoint.h5")
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.warning("h5py not available. Checkpoint functionality limited.")


class CheckpointManager:
    """Manages simulation checkpoints with HDF5 format.
    
    Attributes:
        config: Solver configuration
        output_dir: Output directory for checkpoints
        checkpoint_interval: Save interval (steps)
        
    Example:
        >>> manager = CheckpointManager(config, output_dir="results/")
        >>> manager.save(solution, history, iteration=100)
        >>> sol, hist, it = manager.load("results/checkpoints/checkpoint_iter_000100.h5")
    """
    
    def __init__(
        self,
        config,
        output_dir: str = "results/",
        checkpoint_interval: int = 100
    ):
        """Initialize checkpoint manager.
        
        Args:
            config: Solver configuration object
            output_dir: Base output directory
            checkpoint_interval: Checkpoint save interval (iterations)
        """
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for checkpoint functionality. "
                "Install with: pip install h5py"
            )
        
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def _compute_config_hash(self) -> str:
        """Compute SHA256 hash of solver configuration.
        
        Returns:
            SHA256 hash string
        """
        config_dict = {
            "mode": getattr(self.config, "mode", "steady"),
            "backend": getattr(self.config, "backend", "cpu"),
            "order": getattr(self.config, "order", 2),
            "turbulence": str(getattr(self.config, "turbulence", "sst_kw")),
            "cfl_initial": getattr(self.config, "cfl_init", 0.1),
            "cfl_max": getattr(self.config, "cfl_max", 5.0),
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def save(
        self,
        solution: np.ndarray,
        history: dict,
        iteration: int,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        """Save checkpoint to HDF5 file.
        
        Args:
            solution: Solution array, shape=(n_cells, n_vars)
            history: Convergence history dict with keys:
                - iterations: List[int]
                - residuals: Dict[str, List[float]]
                - coefficients: Dict[str, List[float]]
                - cfl_history: List[float]
            iteration: Current iteration number
            metadata: Additional metadata (optional)
            
        Returns:
            Path to checkpoint file, or None if failed
            
        Example:
            >>> path = manager.save(solution, history, iteration=500)
            >>> print(f"Checkpoint saved: {path}")
        """
        try:
            # Generate checkpoint filename
            ckpt_filename = f"checkpoint_iter_{iteration:06d}.h5"
            ckpt_path = self.checkpoint_dir / ckpt_filename
            
            logger.debug(f"Saving checkpoint: {ckpt_path}")
            
            with h5py.File(ckpt_path, 'w') as f:
                # === Metadata ===
                meta_group = f.create_group("metadata")
                meta_group.attrs['iteration'] = iteration
                meta_group.attrs['timestamp'] = np.string_(datetime.now().isoformat())
                meta_group.attrs['backend'] = np.string_(getattr(self.config, "backend", "cpu"))
                meta_group.attrs['config_hash'] = np.string_(self._compute_config_hash())
                
                if metadata:
                    for key, value in metadata.items():
                        # Convert string values to bytes for HDF5 compatibility
                        if isinstance(value, str):
                            meta_group.attrs[key] = np.string_(value)
                        else:
                            meta_group.attrs[key] = value
                
                # === Solution ===
                sol_group = f.create_group("solution")
                sol_group.create_dataset("conserved", data=solution)
                sol_group.attrs['shape'] = solution.shape
                sol_group.attrs['dtype'] = np.string_(str(solution.dtype))
                
                # === Convergence History ===
                conv_group = f.create_group("convergence/history")
                
                # Iterations
                if 'iterations' in history:
                    conv_group.create_dataset(
                        "iterations",
                        data=np.array(history['iterations'], dtype=np.int32)
                    )
                
                # Residuals
                if 'residuals' in history:
                    res_group = conv_group.create_group("residuals")
                    for eq_name, values in history['residuals'].items():
                        res_group.create_dataset(
                            eq_name,
                            data=np.array(values, dtype=np.float64)
                        )
                
                # Coefficients
                if 'coefficients' in history:
                    coef_group = conv_group.create_group("coefficients")
                    for coef_name, values in history['coefficients'].items():
                        coef_group.create_dataset(
                            coef_name,
                            data=np.array(values, dtype=np.float64)
                        )
                
                # CFL history
                if 'cfl_history' in history:
                    conv_group.create_dataset(
                        "cfl_history",
                        data=np.array(history['cfl_history'], dtype=np.float64)
                    )
                
                # === Statistics (for transient mode) ===
                if 'statistics' in history:
                    stats_group = f.create_group("statistics")
                    for stat_name, stat_data in history['statistics'].items():
                        if isinstance(stat_data, np.ndarray):
                            stats_group.create_dataset(stat_name, data=stat_data)
                        else:
                            stats_group.attrs[stat_name] = stat_data
            
            logger.info(f"✓ Checkpoint saved: {ckpt_path} ({iteration} iterations)")
            return str(ckpt_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        target_backend: Optional[str] = None
    ) -> Tuple[np.ndarray, dict, int, dict]:
        """Load checkpoint from HDF5 file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            target_backend: Target backend ("cpu" or "gpu"). 
                          If None, uses original backend.
            
        Returns:
            Tuple of (solution, history, iteration, metadata)
            
        Raises:
            FileNotFoundError: Checkpoint file not found
            ValueError: Invalid checkpoint format
            
        Example:
            >>> solution, history, iteration, meta = manager.load("checkpoint.h5")
            >>> print(f"Resumed from iteration {iteration}")
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            with h5py.File(checkpoint_path, 'r') as f:
                # === Load Metadata ===
                meta_group = f["metadata"]
                iteration = int(meta_group.attrs['iteration'])
                
                # Decode byte strings to regular strings
                def decode_attr(value):
                    """Decode HDF5 byte string attributes to Python strings."""
                    if isinstance(value, bytes):
                        return value.decode('utf-8')
                    return value
                
                timestamp = decode_attr(meta_group.attrs['timestamp'])
                original_backend = decode_attr(meta_group.attrs['backend'])
                config_hash = decode_attr(meta_group.attrs['config_hash'])
                
                metadata = {
                    'iteration': iteration,
                    'timestamp': timestamp,
                    'original_backend': original_backend,
                    'config_hash': config_hash,
                }
                
                # Extract additional metadata
                for key in meta_group.attrs.keys():
                    if key not in ['iteration', 'timestamp', 'backend', 'config_hash']:
                        metadata[key] = decode_attr(meta_group.attrs[key])
                
                # === Validate Configuration ===
                current_hash = self._compute_config_hash()
                if config_hash != current_hash:
                    logger.warning(
                        f"⚠ Configuration mismatch!\n"
                        f"  Checkpoint config: {config_hash[:16]}...\n"
                        f"  Current config:    {current_hash[:16]}...\n"
                        f"  This may cause incorrect results."
                    )
                
                # === Load Solution ===
                sol_group = f["solution"]
                solution = sol_group["conserved"][:]
                
                # Backend conversion if needed
                if target_backend and target_backend != original_backend:
                    logger.info(
                        f"Converting solution from {original_backend} to {target_backend}"
                    )
                    # TODO: Implement actual CPU↔GPU conversion
                    # For now, just warn the user
                    logger.warning(
                        "Cross-backend conversion not yet implemented. "
                        "Solution will remain on original backend."
                    )
                
                # === Load Convergence History ===
                conv_group = f["convergence/history"]
                history = {}
                
                # Iterations
                if 'iterations' in conv_group:
                    history['iterations'] = conv_group['iterations'][:].tolist()
                
                # Residuals
                if 'residuals' in conv_group:
                    history['residuals'] = {}
                    for eq_name in conv_group['residuals'].keys():
                        history['residuals'][eq_name] = (
                            conv_group['residuals'][eq_name][:].tolist()
                        )
                
                # Coefficients
                if 'coefficients' in conv_group:
                    history['coefficients'] = {}
                    for coef_name in conv_group['coefficients'].keys():
                        history['coefficients'][coef_name] = (
                            conv_group['coefficients'][coef_name][:].tolist()
                        )
                
                # CFL history
                if 'cfl_history' in conv_group:
                    history['cfl_history'] = conv_group['cfl_history'][:].tolist()
                
                # === Load Statistics (transient mode) ===
                if 'statistics' in f:
                    history['statistics'] = {}
                    stats_group = f['statistics']
                    for stat_name in stats_group.keys():
                        history['statistics'][stat_name] = stats_group[stat_name][:]
                    for attr_name in stats_group.attrs.keys():
                        history['statistics'][attr_name] = stats_group.attrs[attr_name]
            
            logger.success(
                f"✓ Checkpoint loaded: {checkpoint_path}\n"
                f"  Iteration: {iteration}\n"
                f"  Solution shape: {solution.shape}\n"
                f"  History entries: {len(history.get('iterations', []))}"
            )
            
            return solution, history, iteration, metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Checkpoint loading failed: {e}")
    
    def should_save(self, iteration: int) -> bool:
        """Check if checkpoint should be saved at this iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            True if should save checkpoint
        """
        return iteration % self.checkpoint_interval == 0
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint file paths, sorted by iteration
        """
        if not self.checkpoint_dir.exists():
            return []
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_iter_*.h5"))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
    
    def cleanup_old_checkpoints(self, keep_last: int = 3) -> int:
        """Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep
            
        Returns:
            Number of deleted checkpoints
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last:
            return 0
        
        # Delete oldest checkpoints
        to_delete = checkpoints[:-keep_last]
        for ckpt_path in to_delete:
            try:
                ckpt_path.unlink()
                logger.debug(f"Deleted old checkpoint: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {ckpt_path}: {e}")
        
        deleted_count = len(to_delete)
        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count


def resume_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config,
    target_backend: Optional[str] = None
) -> Tuple[np.ndarray, dict, int, dict]:
    """Convenience function to resume from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Solver configuration
        target_backend: Target backend (optional)
        
    Returns:
        Tuple of (solution, history, iteration, metadata)
        
    Example:
        >>> solution, history, iteration, meta = resume_from_checkpoint(
        ...     "results/checkpoints/checkpoint_iter_001000.h5",
        ...     config
        ... )
        >>> solver = FRSolver(grid_data, config, initial_solution=solution)
        >>> result = solver.solve(start_iteration=iteration)
    """
    manager = CheckpointManager(config)
    return manager.load(checkpoint_path, target_backend=target_backend)
