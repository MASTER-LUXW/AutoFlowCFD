"""监控残差范数（全部后端唯一定义）：真实自由度上的 RMS，经该后端的归约对象做全局归约。

此前三种定义并存：CPU 单机与单 GPU 是 RMS（L2 / sqrt(N)）、CPU 分布式是全局 L2（不除自由度数，打印值比其余
后端大 sqrt(N) 倍，百万自由度即 ~1e3）、多 GPU 是 RMS；三者的分母还都含四面体的零填充槽位（P2 占 63%）。相对
收敛判据本身不受常数因子影响，但阶段基准 `phase_initial_residual` 随 checkpoint 持久化、续算时做种子——跨后端
续算基准差 sqrt(N) 倍，升阶 / 收敛判据会提前触发或永不触发；收敛历史与舍入平台判据跨后端也不可比。

`RESIDUAL_NORM_VERSION` 随阶段基准写进 checkpoint（`order_continuation/checkpoint_state.py`）：旧定义写出的基准
续算时不再使用。

## 版本 3：逐方程按参考量级归一化（2026-10-19，I-81）

监控残差 = 各方程真实自由度 RMS 除以该方程参考量级（`fr_solver/residual_diagnostics.py::_reference_scales`：
rho_inf、rho_inf |u_inf|、p_inf）后的均方根。版本 2 把五个守恒量的原始残差直接合成一个 RMS：能量方程的量级约为
密度方程的 p_inf / rho_inf ~ 1e5 倍，收敛判据、升阶判据与显式自适应 CFL 实际只看能量方程，其余方程的停滞或发散
被淹没。逐方程 RMS（未归一化）照旧进收敛历史。
"""

from typing import Optional, Tuple

import numpy as np

#: 当前定义的版本号（真实自由度上的 RMS）。旧 checkpoint 没有这个字段
RESIDUAL_NORM_VERSION = 3


def monitor_residuals(red, residual, cell_is_prism, order: int, scales) -> Tuple[float, np.ndarray]:
    """`residual (n_cells * n_sps, n_vars)` 或 `(n_cells, n_sps, n_vars)` 在真实自由度上的全局 RMS：
    `(监控残差, 逐方程 RMS (n_vars,))`。

    监控残差是收敛判据、阶段基准与自适应 CFL 用的那个数：逐方程 RMS 除以 `scales` 后的均方根（定义见模块文档）；
    逐方程 RMS（未归一化）进收敛历史（`core/fr_solver/convergence_history.py`）。每方程的平方和与真实行数一次归约。

    Args:
        red: 归约对象（`implicit/reductions.py::LocalReductions` 或分布式的 `MPIReductions`），数组模块即
            `residual` 所在的模块
        residual: 本进程拥有的单元的残差（分布式为 local 单元、原生排列）
        cell_is_prism: (n_cells,) 主机布尔数组，与 `residual` 的单元顺序一致
        order: 多项式阶数（填充槽位划分）
        scales: `(n_vars,)` 各方程的参考量级（平均流取 `_reference_scales(freestream, n_vars)`；已按尺度归一化的
            量传全 1）
    """
    from autoflowcfd.fr.native_padding import real_row_mask

    xp = red.xp
    cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
    n_cells = cell_is_prism.size
    flat = residual.reshape(n_cells, -1, residual.shape[-1])
    n_sps = flat.shape[1]
    rows = flat.reshape(n_cells * n_sps, -1)
    real = rows[xp.asarray(real_row_mask(cell_is_prism, n_sps, int(order)))]
    sq = xp.sum(real * real, axis=0)
    sums = red.sum_array(np.append(np.asarray(sq.get() if hasattr(sq, "get") else sq), float(real.shape[0])))
    per_equation = np.sqrt(sums[:-1] / sums[-1]) if sums[-1] > 0 else np.zeros(sums.size - 1)
    scaled = per_equation / np.asarray(scales, dtype=np.float64)
    return float(np.sqrt(np.mean(scaled * scaled))), per_equation


def explicit_turbulence_residual(red, model, cell_is_prism, order: int) -> Optional[float]:
    """显式湍流更新这一步的监控残差：`model.last_update_rates`（`turbulence/transported.py::
    advance_transported_fields`，逐未知量除以尺度）在真实自由度上的全局 RMS，参数含义同 `monitor_residuals`。
    取用后清空（下一步没有显式更新时不留旧值）；这一步没有显式更新（隐式路径、没有输运方程的模型）时 None。

    2026-10-19（I-77）以前显式路径的湍流方程没有任何监控残差：监视行与收敛历史只有平均流 5 个方程。
    """
    rates = None if model is None else getattr(model, "last_update_rates", None)
    if rates is None:
        return None
    model.last_update_rates = None
    return monitor_residuals(red, rates, cell_is_prism, order, np.ones(rates.shape[-1]))[0]   # 速率已按尺度归一化


def turbulence_monitor_residual(solver) -> Optional[float]:
    """这一步的湍流监控残差（进监视行与收敛历史）：紧耦合 Newton 步的湍流子系统步前残差范数
    （`_newton_last_info["res_norm_turbulence"]`），否则显式更新的速率 RMS（`explicit_turbulence_residual`，各后端
    在单步里存进 `solver._residual_turbulence`）；都没有时 None。

    只用于监视。显式路径的收敛判据仍只看平均流（`order_continuation/policy.py::PhaseGate` 的湍流条件只接隐式
    子系统残差）：上界截断处的逐点速率在定常解上也不为零，纳入判据会让显式运行判不了收敛。
    """
    info = getattr(solver, "_newton_last_info", None)
    if info and "res_norm_turbulence" in info:
        return float(info["res_norm_turbulence"])
    return getattr(solver, "_residual_turbulence", None)
