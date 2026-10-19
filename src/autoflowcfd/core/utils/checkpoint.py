"""AutoFlowCFD 求解器的 checkpoint 写出（HDF5）。

文件结构：

    metadata/   属性：迭代数、时间戳、布局版本（棱柱基 / 解点存储宽度 / 四面体节点集），以及写出端给出的
                重建参数（输入网格、阶数与目标阶数、湍流模型、后端、时间格式、来流物理量……）
    solution/   conserved = 单元积分平均 `(n_cells, n_vars)`；其余数据集是逐解点字段（U_sps、湍流场、
                dual-time 上一时间层……）与逐单元量（tau_accum、湍流单元均值）
    convergence/history/   iterations、residuals/<方程>、coefficients/<系数>、cfl_history

读取见 `checkpoint_load.py::load_checkpoint`（布局校验与旧布局迁移都在那里）。

2026-10-19（I-86）删除：构造参数 `config` 与配置哈希（读取端拿不到写出时的配置，哈希比对恒报"不一致"，
真正决定续算的量已逐项写进 metadata 并在续算时核对）、假的跨后端转换、`h5py` 可选兜底（pyproject 必需依赖）、
零调用的 `should_save` / `list_checkpoints` / `get_latest_checkpoint` / `cleanup_old_checkpoints` /
`resume_from_checkpoint`；写出失败此前被吞掉只返回 None，求解照样往下跑而没有可续算的检查点，现在直接报错。
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
from loguru import logger

from autoflowcfd.fr.warp_blend_nodes import TET_NODE_SET


class CheckpointManager:
    """HDF5 checkpoint 的写出与读取。

    Example:
        >>> manager = CheckpointManager(output_dir="results/")
        >>> path = manager.save(solution, history, iteration=100, metadata={...}, extra_fields={"U_sps": U})
        >>> sol, hist, it, meta = manager.load(path)
    """

    def __init__(self, output_dir: str = "results/", quiet: bool = False):
        """
        Args:
            output_dir: 算例输出目录，checkpoint 写在 `<output_dir>/checkpoints/`（只读取时不用）
            quiet: 静默模式，不打印初始化与写出信息
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        # 构造期不 mkdir，首次 `save()` 时才建（`_ensure_checkpoint_dir`，2026-09-18）：构造一个管理器对象不应
        # 产生文件系统副作用，否则在项目目录里构造它的代码（包括只读取或只验证字段的单元测试）都会凭空留下
        # `checkpoints/`。
        self.quiet = quiet

    def _ensure_checkpoint_dir(self) -> None:
        """首次真正写 checkpoint 时创建目录（见 `__init__` 里的说明）。"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        solution: np.ndarray,
        history: dict,
        iteration: int,
        metadata: Optional[dict] = None,
        extra_fields: Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        """把 checkpoint 写成 `<output_dir>/checkpoints/checkpoint_iter_<iteration>.h5`。

        Args:
            solution: 单元积分平均 `(n_cells, n_vars)`
            history: 收敛历史字典，可含 iterations / residuals（方程名 -> 序列）/ coefficients（系数名 -> 序列）/
                cfl_history / statistics
            iteration: 当前迭代数
            metadata: 写进 metadata 属性的键值（字符串、数值或数值数组）
            extra_fields: 额外数据集（逐解点或逐单元），值为 None 的条目跳过

        Returns:
            checkpoint 文件路径

        Raises:
            OSError 等：写出失败（不吞掉——没有可续算的检查点必须让调用方知道）
        """
        from autoflowcfd.core.utils.checkpoint_load import PRISM_BASIS, SP_LAYOUT

        self._ensure_checkpoint_dir()
        ckpt_path = self.checkpoint_dir / f"checkpoint_iter_{iteration:06d}.h5"
        logger.debug(f"Saving checkpoint: {ckpt_path}")

        with h5py.File(ckpt_path, 'w') as f:
            meta_group = f.create_group("metadata")
            meta_group.attrs['iteration'] = iteration
            meta_group.attrs['timestamp'] = np.string_(datetime.now().isoformat())
            # 布局版本：棱柱基、解点存储宽度、四面体节点集。同一份数组在不同布局下含义不同而形状可能相同，
            # 读取端据此硬失败或精确迁移（`checkpoint_load.py`）。
            meta_group.attrs['prism_basis'] = np.string_(PRISM_BASIS)
            meta_group.attrs['sp_layout'] = np.string_(SP_LAYOUT)
            meta_group.attrs['tet_node_set'] = np.string_(TET_NODE_SET)
            for key, value in (metadata or {}).items():
                meta_group.attrs[key] = np.string_(value) if isinstance(value, str) else value

            sol_group = f.create_group("solution")
            sol_group.create_dataset("conserved", data=solution)
            for name, arr in (extra_fields or {}).items():
                if arr is not None:
                    sol_group.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

            conv_group = f.create_group("convergence/history")
            if 'iterations' in history:
                conv_group.create_dataset("iterations", data=np.array(history['iterations'], dtype=np.int64))
            for group_name in ('residuals', 'coefficients'):
                if group_name in history:
                    group = conv_group.create_group(group_name)
                    for name, values in history[group_name].items():
                        group.create_dataset(name, data=np.array(values, dtype=np.float64))
            if 'cfl_history' in history:
                conv_group.create_dataset("cfl_history", data=np.array(history['cfl_history'], dtype=np.float64))

            if 'statistics' in history:
                stats_group = f.create_group("statistics")
                for stat_name, stat_data in history['statistics'].items():
                    if isinstance(stat_data, np.ndarray):
                        stats_group.create_dataset(stat_name, data=stat_data)
                    else:
                        stats_group.attrs[stat_name] = stat_data

        if not self.quiet:
            logger.info(f"Checkpoint saved: {ckpt_path} ({iteration} iterations)")
        return str(ckpt_path)

    def load(self, checkpoint_path: Union[str, Path]) -> Tuple[np.ndarray, dict, int, dict]:
        """读取 checkpoint，返回 `(solution, history, iteration, metadata)`；见 `checkpoint_load.py::load_checkpoint`。"""
        from .checkpoint_load import load_checkpoint

        return load_checkpoint(checkpoint_path)
