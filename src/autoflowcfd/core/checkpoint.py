"""AutoFlowCFD 求解器的 checkpoint 管理。

本模块基于 HDF5 格式提供完整的 checkpoint 保存/加载功能，支持跨
backend（CPU↔GPU）恢复求解和配置校验。

核心组件:
    - CheckpointManager: 主 checkpoint 处理器
    - ConservedVariables 序列化
    - ConvergenceHistory 序列化

示例:
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
    """用 HDF5 格式管理仿真 checkpoint。

    Attributes:
        config: 求解器配置
        output_dir: checkpoint 的输出目录
        checkpoint_interval: 保存间隔（迭代步数）

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
        """初始化 checkpoint 管理器。

        Args:
            config: 求解器配置对象
            output_dir: 基础输出目录
            checkpoint_interval: checkpoint 保存间隔（迭代数）
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
        """计算求解器配置的 SHA256 哈希。

        Returns:
            SHA256 哈希字符串
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
        metadata: Optional[dict] = None,
        extra_fields: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[str]:
        """把 checkpoint 保存为 HDF5 文件。

        Args:
            solution: 解数组，形状=(n_cells, n_vars)
            history: 收敛历史字典，含以下键：
                - iterations: List[int]
                - residuals: Dict[str, List[float]]
                - coefficients: Dict[str, List[float]]
                - cfl_history: List[float]
            iteration: 当前迭代数
            metadata: 额外元数据（可选）
            extra_fields: 随守恒解一起持久化的额外逐单元求解器派生场，
                例如 {'mu_t': mu_t}——求解器那一步实际算出的湍流涡粘性。
                没有这个，后处理（VTKExporter）就无法还原精确的
                SST 混合值，只能退回到更粗糙的 k/omega 估计。值为 None
                的条目会被跳过（例如调用方因为本次求解关闭了湍流而传
                mu_t=None）。

        Returns:
            checkpoint 文件路径；失败则返回 None

        Example:
            >>> path = manager.save(solution, history, iteration=500)
            >>> print(f"Checkpoint saved: {path}")
        """
        try:
            # 生成 checkpoint 文件名
            ckpt_filename = f"checkpoint_iter_{iteration:06d}.h5"
            ckpt_path = self.checkpoint_dir / ckpt_filename

            logger.debug(f"Saving checkpoint: {ckpt_path}")

            with h5py.File(ckpt_path, 'w') as f:
                # === 元数据 ===
                meta_group = f.create_group("metadata")
                meta_group.attrs['iteration'] = iteration
                meta_group.attrs['timestamp'] = np.string_(datetime.now().isoformat())
                meta_group.attrs['backend'] = np.string_(getattr(self.config, "backend", "cpu"))
                meta_group.attrs['config_hash'] = np.string_(self._compute_config_hash())

                if metadata:
                    for key, value in metadata.items():
                        # 把字符串值转成 bytes 以兼容 HDF5
                        if isinstance(value, str):
                            meta_group.attrs[key] = np.string_(value)
                        else:
                            meta_group.attrs[key] = value

                # === 解 ===
                sol_group = f.create_group("solution")
                sol_group.create_dataset("conserved", data=solution)
                sol_group.attrs['shape'] = solution.shape
                sol_group.attrs['dtype'] = np.string_(str(solution.dtype))

                if extra_fields:
                    for name, arr in extra_fields.items():
                        if arr is not None:
                            sol_group.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

                # === 收敛历史 ===
                conv_group = f.create_group("convergence/history")

                # 迭代数
                if 'iterations' in history:
                    conv_group.create_dataset(
                        "iterations",
                        data=np.array(history['iterations'], dtype=np.int32)
                    )

                # 残差
                if 'residuals' in history:
                    res_group = conv_group.create_group("residuals")
                    for eq_name, values in history['residuals'].items():
                        res_group.create_dataset(
                            eq_name,
                            data=np.array(values, dtype=np.float64)
                        )

                # 系数
                if 'coefficients' in history:
                    coef_group = conv_group.create_group("coefficients")
                    for coef_name, values in history['coefficients'].items():
                        coef_group.create_dataset(
                            coef_name,
                            data=np.array(values, dtype=np.float64)
                        )

                # CFL 历史
                if 'cfl_history' in history:
                    conv_group.create_dataset(
                        "cfl_history",
                        data=np.array(history['cfl_history'], dtype=np.float64)
                    )

                # === 统计量（瞬态模式）===
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
        """从 HDF5 文件加载 checkpoint。

        Args:
            checkpoint_path: checkpoint 文件路径
            target_backend: 目标 backend（"cpu" 或 "gpu"）。
                          若为 None，使用原始 backend。

        Returns:
            (solution, history, iteration, metadata) 元组

        Raises:
            FileNotFoundError: 找不到 checkpoint 文件
            ValueError: checkpoint 格式无效

        Example:
            >>> solution, history, iteration, meta = manager.load("checkpoint.h5")
            >>> print(f"Resumed from iteration {iteration}")
        """
        from .checkpoint_load import load_checkpoint

        return load_checkpoint(self, checkpoint_path, target_backend)

    def should_save(self, iteration: int) -> bool:
        """检查这一迭代步是否应该保存 checkpoint。

        Args:
            iteration: 当前迭代数

        Returns:
            应保存则为 True
        """
        return iteration % self.checkpoint_interval == 0

    def list_checkpoints(self) -> List[Path]:
        """列出所有可用的 checkpoint。

        Returns:
            按迭代数排序的 checkpoint 文件路径列表
        """
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = list(self.checkpoint_dir.glob("checkpoint_iter_*.h5"))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))

        return checkpoints

    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最近的一个 checkpoint。

        Returns:
            最新 checkpoint 的路径；不存在则返回 None
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def cleanup_old_checkpoints(self, keep_last: int = 3) -> int:
        """删除旧的 checkpoint，只保留最近的几个。

        Args:
            keep_last: 保留的最近 checkpoint 数量

        Returns:
            已删除的 checkpoint 数量
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_last:
            return 0

        # 删除最旧的 checkpoint
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
    """从 checkpoint 恢复求解的便捷函数。

    Args:
        checkpoint_path: checkpoint 文件路径
        config: 求解器配置
        target_backend: 目标 backend（可选）

    Returns:
        (solution, history, iteration, metadata) 元组

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
