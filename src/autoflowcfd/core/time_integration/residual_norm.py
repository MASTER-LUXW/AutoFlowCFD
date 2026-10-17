"""监控残差范数（全部后端唯一定义）：真实自由度上的 RMS，经该后端的归约对象做全局归约。

此前三种定义并存：CPU 单机与单 GPU 是 RMS（L2 / sqrt(N)）、CPU 分布式是全局 L2（不除自由度数，打印值比其余
后端大 sqrt(N) 倍，百万自由度即 ~1e3）、多 GPU 是 RMS；三者的分母还都含四面体的零填充槽位（P2 占 63%）。相对
收敛判据本身不受常数因子影响，但阶段基准 `phase_initial_residual` 随 checkpoint 持久化、续算时做种子——跨后端
续算基准差 sqrt(N) 倍，升阶 / 收敛判据会提前触发或永不触发；收敛历史与舍入平台判据跨后端也不可比。

`RESIDUAL_NORM_VERSION` 随阶段基准写进 checkpoint（`order_continuation/checkpoint_state.py`）：旧定义写出的基准
续算时不再使用。
"""

import numpy as np

#: 当前定义的版本号（真实自由度上的 RMS）。旧 checkpoint 没有这个字段
RESIDUAL_NORM_VERSION = 2


def monitor_residual_norm(red, residual, cell_is_prism, order: int) -> float:
    """`residual (n_cells * n_sps, n_vars)` 或 `(n_cells, n_sps, n_vars)` 在真实自由度上的全局 RMS。

    Args:
        red: 归约对象（`implicit/reductions.py::LocalReductions` 或分布式的 `MPIReductions`），数组模块即
            `residual` 所在的模块
        residual: 本进程拥有的单元的残差（分布式为 local 单元、原生排列）
        cell_is_prism: (n_cells,) 主机布尔数组，与 `residual` 的单元顺序一致
        order: 多项式阶数（填充槽位划分）
    """
    from autoflowcfd.fr.native_padding import real_row_mask

    xp = red.xp
    cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
    n_cells = cell_is_prism.size
    flat = residual.reshape(n_cells, -1, residual.shape[-1])
    n_sps = flat.shape[1]
    rows = flat.reshape(n_cells * n_sps, -1)
    mask = real_row_mask(cell_is_prism, n_sps, int(order))
    return red.rms(rows[xp.asarray(mask)])
