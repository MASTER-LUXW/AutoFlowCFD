"""CheckpointManager.load 实现 (从 checkpoint.py 拆分)。

从 checkpoint.py 拆出来（该文件原有 488 行，超过 400 行硬性拆分
阈值）：`load` 是文件里最长的单个方法（约 190 行），只依赖
`manager._compute_config_hash()` 这一处实例状态，独立成模块函数、
`CheckpointManager` 上保留同名薄委托方法是最干净的拆分点。纯代码
搬移，不改变任何行为。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import h5py
except ImportError:
    h5py = None


def decode_attr(value):
    """把 HDF5 字节串属性解码成 Python 字符串。

    2026-09-20 从 `load_checkpoint` 内部提到模块级：`_check_prism_basis`
    也要用它，两处各写一份就成了两个事实来源。
    """
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _check_prism_basis(meta_group, checkpoint_path) -> None:
    """校验 checkpoint 的棱柱基与当前环境一致，不一致就硬失败。

    **为什么必须硬失败**（2026-09-20）：两条棱柱基的每单元解点布局不同
    —— 原生基 P1 每单元只有前 6 个槽位是自由度，其余是按
    `fr/native_padding.py` 约定复制真实 SP#0 的填充位；坍缩基 8 个全是
    自由度。同一份 `(n_cells, n_sps, n_vars)` 数组在两条基下**含义不同**，
    跨基 resume 会静默重解释它，给出一个看不出异常的错解（不会报形状
    不符：形状恰好相同）。

    与 `config_hash` 那条**刻意不同级别**：配置哈希不一致只是"可能影响
    结果"（例如 CFL 上限变了），而棱柱基不一致是"这份数据的含义变了"。

    `prism_basis` 属性是 2026-09-20 才开始写的。属性缺失时：

    * 当前是 `collapsed` -> 放行（那之前**唯一**的默认值就是 collapsed，
      这是兼容的组合），只记一条 info；
    * 当前是 `native` -> **拒绝**（正是危险组合：默认值改成 native 之后
      去 resume 一份 collapsed 时代的 checkpoint）。
    """
    from autoflowcfd.fr.native_prism.mode import resolve_prism_basis_mode

    current = resolve_prism_basis_mode()
    if 'prism_basis' not in meta_group.attrs:
        if current == "collapsed":
            logger.info(
                "Checkpoint 没有记录棱柱基（2026-09-20 之前写的），当前是 "
                "collapsed —— 那之前唯一的默认值也是 collapsed，组合兼容。")
            return
        raise ValueError(
            f"Checkpoint {checkpoint_path} 没有记录棱柱基（2026-09-20 之前"
            f"写的，那时唯一的默认值是 collapsed），而当前 "
            f"AFCFD_PRISM_BASIS={current!r}。两条基的每单元解点布局不同，"
            f"跨基 resume 会静默给出错解（形状恰好相同、不会报错）。"
            f"要么用 AFCFD_PRISM_BASIS=collapsed 续算这份 checkpoint，"
            f"要么在当前基下从头开始。")

    saved = decode_attr(meta_group.attrs['prism_basis'])
    if saved != current:
        raise ValueError(
            f"Checkpoint {checkpoint_path} 是在棱柱基 {saved!r} 下写的，"
            f"而当前 AFCFD_PRISM_BASIS={current!r}。两条基的每单元解点"
            f"布局不同（原生基有零填充槽位），跨基 resume 会静默重解释"
            f"同一份数组、给出看不出异常的错解。"
            f"请把 AFCFD_PRISM_BASIS 设成 {saved!r} 续算，或在当前基下"
            f"从头开始。")


def load_checkpoint(
    manager,
    checkpoint_path: Union[str, Path],
    target_backend: Optional[str] = None,
) -> Tuple[np.ndarray, dict, int, dict]:
    """从 HDF5 文件加载 checkpoint。

    Args:
        manager: CheckpointManager 实例
        checkpoint_path: checkpoint 文件路径
        target_backend: 目标 backend（"cpu" 或 "gpu"）。
                      若为 None，使用原始 backend。

    Returns:
        (solution, history, iteration, metadata) 元组

    Raises:
        FileNotFoundError: 找不到 checkpoint 文件
        ValueError: checkpoint 格式无效

    Example:
        >>> solution, history, iteration, meta = load_checkpoint(manager, "checkpoint.h5")
        >>> print(f"Resumed from iteration {iteration}")
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        with h5py.File(checkpoint_path, 'r') as f:
            # === 加载元数据 ===
            meta_group = f["metadata"]
            iteration = int(meta_group.attrs['iteration'])

            timestamp = decode_attr(meta_group.attrs['timestamp'])
            original_backend = decode_attr(meta_group.attrs['backend'])
            config_hash = decode_attr(meta_group.attrs['config_hash'])

            metadata = {
                'iteration': iteration,
                'timestamp': timestamp,
                'original_backend': original_backend,
                'config_hash': config_hash,
            }

            # 提取额外的元数据
            for key in meta_group.attrs.keys():
                if key not in ['iteration', 'timestamp', 'backend', 'config_hash']:
                    metadata[key] = decode_attr(meta_group.attrs[key])

            # === 校验棱柱基（硬失败，不是警告）===
            _check_prism_basis(meta_group, checkpoint_path)

            # === 校验配置 ===
            current_hash = manager._compute_config_hash()
            if config_hash != current_hash:
                logger.warning(
                    f"⚠ Configuration mismatch!\n"
                    f"  Checkpoint config: {config_hash[:16]}...\n"
                    f"  Current config:    {current_hash[:16]}...\n"
                    f"  This may cause incorrect results."
                )

            # === 加载解 ===
            sol_group = f["solution"]
            solution = sol_group["conserved"][:]

            # 随守恒解一起保存的任何额外逐单元字段（见 save() 的
            # extra_fields 参数，例如 'mu_t'）——放在
            # metadata['fields'] 下而不是作为第 5 个返回值，这样现有
            # 的 (solution, history, iteration, metadata) 调用方不受
            # 影响；对于在加上这个功能之前写入的 checkpoint，或者
            # save() 从未传过 extra_fields 的情况，这里就不存在。
            extra_field_names = [k for k in sol_group.keys() if k != "conserved"]
            if extra_field_names:
                metadata['fields'] = {name: sol_group[name][:] for name in extra_field_names}

            # 如需要则做 backend 转换
            if target_backend and target_backend != original_backend:
                logger.info(
                    f"Converting solution from {original_backend} to {target_backend}"
                )

                # 实现 CPU↔GPU 转换
                try:
                    if target_backend.lower() == 'gpu':
                        # CPU -> GPU: 将numpy数组转移到GPU
                        try:
                            import cupy as cp
                            # 转换守恒变量
                            if 'conserved' in sol_group:
                                U_cpu = sol_group['conserved'][:]
                                U_gpu = cp.asarray(U_cpu)
                                metadata['conserved'] = U_gpu
                                logger.info("Successfully converted solution from CPU to GPU")

                            # 转换其他场变量
                            for field_name in extra_field_names:
                                if field_name in sol_group:
                                    field_cpu = sol_group[field_name][:]
                                    field_gpu = cp.asarray(field_cpu)
                                    metadata[field_name] = field_gpu

                        except ImportError:
                            logger.warning("CuPy not available, keeping solution on CPU")

                    elif target_backend.lower() == 'cpu':
                        # GPU -> CPU: 将GPU数组转移回CPU
                        try:
                            import cupy as cp
                            # 如果数据已经在GPU上（以cupy数组形式存储）
                            if isinstance(metadata.get('conserved'), cp.ndarray):
                                metadata['conserved'] = cp.asnumpy(metadata['conserved'])
                                logger.info("Successfully converted solution from GPU to CPU")

                            # 转换其他场变量
                            for field_name in extra_field_names:
                                if field_name in metadata and isinstance(metadata[field_name], cp.ndarray):
                                    metadata[field_name] = cp.asnumpy(metadata[field_name])

                        except ImportError:
                            logger.warning("CuPy not available, assuming data is already on CPU")

                    else:
                        logger.warning(f"Unknown target backend: {target_backend}")

                except Exception as e:
                    logger.error(f"Backend conversion failed: {e}")
                    logger.warning("Solution will remain on original backend")

            # === 加载收敛历史 ===
            conv_group = f["convergence/history"]
            history = {}

            # 迭代数
            if 'iterations' in conv_group:
                history['iterations'] = conv_group['iterations'][:].tolist()

            # 残差
            if 'residuals' in conv_group:
                history['residuals'] = {}
                for eq_name in conv_group['residuals'].keys():
                    history['residuals'][eq_name] = (
                        conv_group['residuals'][eq_name][:].tolist()
                    )

            # 系数
            if 'coefficients' in conv_group:
                history['coefficients'] = {}
                for coef_name in conv_group['coefficients'].keys():
                    history['coefficients'][coef_name] = (
                        conv_group['coefficients'][coef_name][:].tolist()
                    )

            # CFL 历史
            if 'cfl_history' in conv_group:
                history['cfl_history'] = conv_group['cfl_history'][:].tolist()

            # === 加载统计量（瞬态模式）===
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
