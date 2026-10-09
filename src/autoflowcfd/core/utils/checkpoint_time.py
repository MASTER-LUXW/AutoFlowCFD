"""时间精确（dual-time）计算随 checkpoint 持久化的时间状态（单机与分布式写入/恢复共用的唯一实现）。

续算一个 dual-time 瞬态需要两样东西，2026-10-09 以前 checkpoint 一样都不存：

* **物理时间步长 dt。** `solve resume` 一律以 `dt=1e-3` 续算：dual-time 是把物理时间步换成了一个与原算例毫无
  关系的值；按局部伪时间步推进的格式虽不用它推进平均流，合成湍流入口（SEM）每步按 dt 输运涡结构，同样变了。
* **上一物理时间层的解。** BDF2 需要 U^{n-1}；缺了它续算的第一步退化成 BDF1（一阶），时间精度在续算点断一下。

现在 checkpoint 一律写出 `dt`（元数据），dual-time 另写 `U_prev_sps`（字段，形状同 `U_sps`），续算时恢复。

湍流输运方程同样做双时间步（`core/turbulence/dual_time.py`），它的上一时间层 `turb_prev_unknowns`
（`(n_cells, n_sps, n_transported)`，模型的未知量空间：SST 为 k 与 ln omega）一并写出与恢复。
"""

from typing import Optional

import numpy as np

DT_KEY = "dt"
PREVIOUS_LEVEL_FIELD = "U_prev_sps"
TURBULENCE_PREVIOUS_FIELD = "turb_prev_unknowns"


def is_dual_time(solver) -> bool:
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    return solver.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME


def time_metadata(solver) -> dict:
    """{"dt": 本次 `solve()` 的时间步长}；尚未求解过时空。"""
    dt = getattr(solver, "_physical_dt", None)
    return {} if dt is None else {DT_KEY: float(dt)}


def resume_time_step(dt_option, metadata: dict, solver) -> float:
    """续算用的时间步长：命令行 `--dt` > checkpoint 记录的 dt > （早于本字段的 checkpoint）稳态命令的 dt。

    Raises:
        ValueError: dual-time 续算、checkpoint 没有记录 dt 且没有给 `--dt`（物理时间步长不能猜）
    """
    if dt_option is not None:
        return float(dt_option)
    if DT_KEY in metadata:
        return float(metadata[DT_KEY])
    if is_dual_time(solver):
        raise ValueError("checkpoint 没有记录物理时间步长（早于 2026-10-09 写出）：dual-time 续算请用 --dt 给出原算例的"
                         "时间步长")
    from autoflowcfd.core.time_integration.base import STEADY_DT

    return STEADY_DT


def previous_level_rows(solver, n_rows: int) -> Optional[np.ndarray]:
    """上一物理时间层的主机数组 `(n_rows, n_sps, n_vars)`（分布式时 n_rows 为本 rank 的 local 单元数）；
    非 dual-time 或尚未推进过一个物理步时 None。"""
    prev = getattr(solver, "_dual_time_U_prev", None)
    if prev is None or not is_dual_time(solver):
        return None
    host = np.asarray(prev.get() if hasattr(prev, "get") else prev)
    n_sps = int(solver.state.n_sps)
    return np.ascontiguousarray(host.reshape(-1, n_sps, host.shape[-1])[:n_rows])


def restore_previous_level(solver, rows) -> None:
    """由 `(n_rows, n_sps, n_vars)` 恢复上一物理时间层（各后端的 `set_dual_time_history` 负责放到 CPU/GPU 上）。
    只在 dual-time 时恢复：别的格式没有这一层。"""
    if rows is not None and is_dual_time(solver):
        rows = np.asarray(rows)
        solver.set_dual_time_history(rows.reshape(-1, rows.shape[-1]))


def _transport_model(solver):
    """带输运方程的湍流模型（GPU 后端在 `turb_model_gpu`，CPU 后端在 `turb_model`）；没有时 None。"""
    model = getattr(solver, "turb_model_gpu", None)
    if model is None:
        model = getattr(solver, "turb_model", None)
    return model if getattr(model, "TRANSPORTED_FIELDS", ()) else None


def turbulence_previous_rows(solver, n_rows: int) -> Optional[np.ndarray]:
    """湍流上一物理时间层的主机数组 `(n_rows, n_sps, n_transported)`（未知量空间）；非 dual-time、没有输运
    模型或尚未推进过一个物理步时 None。"""
    prev = getattr(solver, "_dual_time_turb_prev", None)
    if prev is None or not is_dual_time(solver):
        return None
    host = [np.asarray(u.get() if hasattr(u, "get") else u) for u in prev]
    return np.ascontiguousarray(np.stack(host, axis=-1)[:n_rows])


def restore_turbulence_previous(solver, rows) -> None:
    """由 `(n_rows, n_sps, n_transported)` 恢复湍流的上一物理时间层（放到模型数组所在的 CPU/GPU 上）。

    只在 dual-time、且 checkpoint 的未知量个数与当前模型一致时恢复：换了湍流模型（例如 SA 稳态 -> DDES 瞬态）
    时输运场本身从来流初值开始，上一时间层没有意义，第一个物理步用 BDF1。"""
    model = _transport_model(solver)
    if rows is None or model is None or not is_dual_time(solver):
        return
    rows = np.asarray(rows)
    current = model.transported_fields()
    if rows.shape[-1] != len(current) or rows.shape[:2] != tuple(current[0].shape):
        return
    first = current[0]
    if type(first).__module__.startswith("cupy"):
        import cupy as xp
    else:
        xp = np
    solver._dual_time_turb_prev = tuple(xp.asarray(np.ascontiguousarray(rows[..., j])) for j in range(rows.shape[-1]))
