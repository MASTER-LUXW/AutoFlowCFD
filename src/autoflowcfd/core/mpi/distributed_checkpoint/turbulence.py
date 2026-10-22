"""AutoFlowCFD V2.0 - 分布式 checkpoint 的湍流场（CPU 分布式与多 GPU 共用）。

写出端把本 rank 的输运场（模型声明的 `TRANSPORTED_FIELDS`，与单机 `write_checkpoint` 同一组字段名）
与涡粘收集成全局数组，另写后处理用的单元平均（`turb_cell_*`，`core/turbulence/output.py`）；读回端
（`solve resume` 与 `solve transient --init-from`）由 root 按 rank 分发出本 rank 的 local 段（`state.py::
distribute_from_root`）交给 `model.restore_transported`（SST 的 omega 可容许性投影、SA-neg 的壁面解点置零都在里面）。

**2026-10-04 修复**：此前分布式 checkpoint 只写平均流 `U_sps`，分布式 resume 后湍流场从来流初值
重新开始；`--init-from` 则从 `U_sps[...,5:7]/rho` 换算 k/omega——那是 SST 状态数组里从未更新的
历史槽位。
"""

import numpy as np

from autoflowcfd.core.utils.array_module import to_host

from .state import gather_global_state


def transported_model(solver):
    """`(model, is_gpu)`：CPU 分布式的 `turb_model` 或多 GPU 的 `turb_model_gpu`；不是输运模型时
    `model` 为 None。"""
    gpu = getattr(solver, "turb_model_gpu", None)
    model = gpu if gpu is not None else getattr(solver, "turb_model", None)
    if not getattr(model, "TRANSPORTED_FIELDS", ()):
        return None, False
    return model, gpu is not None


def _host(a):
    """取回主机的 float64 数组（`to_host`）。"""
    return np.asarray(to_host(a), dtype=np.float64)


def global_cell_is_prism(solver, n_global: int):
    """全局单元的棱柱标志（root 有值）：本 rank 的 local 单元在紧凑排列（"棱柱在前"）里的位置
    小于紧凑棱柱数即为棱柱，逐单元收集——两种分布式模式都适用（完全分布式加载下 `solver.mesh`
    只有本 rank 的紧凑几何，没有全局棱柱数）。集体调用。"""
    n_local = int(solver.partition.n_local_cells)
    dist_fc = solver.dist_flat_face
    is_prism = (np.asarray(dist_fc.inv_perm)[:n_local] < int(dist_fc.base_flat.n_prism)).astype(np.float64)
    out = gather_global_state(is_prism[:, None, None], solver.partition.local_cells, n_global)
    return None if out is None else out[:, 0, 0] > 0.5


def gather_turbulence_fields(solver, n_global: int, W_local):
    """输运场、涡粘与单元积分平均的全局字段（root 返回字典，其余 rank 返回 None）。集体调用。

    Args:
        W_local: 本 rank local 单元的解点积分权重（`fr/native_padding.py::sp_integration_weights`）
    """
    model, _ = transported_model(solver)
    if model is None:
        return {}
    n_local = int(solver.partition.n_local_cells)
    cols = [_host(f)[:n_local] for f in model.transported_fields()]
    cols.append(_host(model.nu_t)[:n_local])
    stacked = gather_global_state(np.stack(cols, axis=-1), solver.partition.local_cells, n_global)

    from autoflowcfd.core.turbulence.output import CHECKPOINT_PREFIX, turbulence_cell_means

    means = turbulence_cell_means(model, W_local)                 # 本 rank 本地积分平均
    keys = list(means)
    # 两次收集都是集体调用（点对点收发），全部 rank 都做完才能让非 root 返回
    gathered = gather_global_state(np.stack([means[k] for k in keys], axis=-1)[:, None, :],
                                   solver.partition.local_cells, n_global)
    if stacked is None or gathered is None:                       # 非 root
        return None
    names = model.TRANSPORTED_FIELDS
    out = {name: np.ascontiguousarray(stacked[..., j]) for j, name in enumerate(names)}
    out["nu_t"] = np.ascontiguousarray(stacked[..., len(names)])
    for j, key in enumerate(keys):
        out[CHECKPOINT_PREFIX + key] = np.ascontiguousarray(gathered[:, 0, j])
    return out


def checkpoint_turbulence_names(solver, available) -> tuple:
    """要从 checkpoint 分发的湍流字段名：模型的输运场与 `nu_t` 中 checkpoint 里有的（缺哪些由
    `restore_turbulence_fields` 判定并记录）；不是输运模型时为空。"""
    model, _ = transported_model(solver)
    if model is None:
        return ()
    return tuple(name for name in (*model.TRANSPORTED_FIELDS, "nu_t") if name in available)


def restore_turbulence_fields(solver, local_fields: dict, metadata: dict, *, source: str) -> bool:
    """由本 rank 的 local 段恢复输运场与涡粘，并按 `metadata` 记录的进度续接产生项渐变（与单机恢复端共用
    `restore_production_ramp`）。`local_fields` 不含输运场（`checkpoint_turbulence_names` 为空）时保留构造时的
    来流初值并返回 False。"""
    from autoflowcfd.core.fr_solver.turbulence.init import restore_production_ramp
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.core.utils.checkpoint_turbulence import missing_turbulence_fields, report_missing_turbulence

    model, is_gpu = transported_model(solver)
    if model is None:
        return False
    names = model.TRANSPORTED_FIELDS
    # 与单机恢复端同一约定（`checkpoint_io/restore.py::restore_turbulence_from_fields`）：缺少的字段记在
    # `solver._missing_turbulence_fields`，续算时由 `check_resumed_turbulence` 拒绝（除非 --reset-turbulence）
    missing, incomplete = missing_turbulence_fields(tuple(names), hasattr(model, "nu_t"), local_fields)
    solver._missing_turbulence_fields = missing
    if is_root():
        report_missing_turbulence(missing, incomplete)
    if incomplete:
        return False
    expect = (int(solver.partition.n_local_cells), int(solver.state.n_sps))
    local = []
    for name in names + (("nu_t",) if "nu_t" in local_fields else ()):
        f = np.asarray(local_fields[name])
        if f.shape != expect:
            raise ValueError(f"Checkpoint 字段 {name} 的本 rank 段形状 {f.shape} 与求解器 {expect} 不匹配")
        local.append(f)
    if is_gpu:
        from autoflowcfd.core.gpu import get_cupy
        cp = get_cupy()
        with cp.cuda.Device(solver.device_id):
            local = [cp.asarray(f) for f in local]
    model.restore_transported(local[:len(names)], source=source)
    if len(local) > len(names):
        model.nu_t = local[len(names)]
    restore_production_ramp(solver, model, metadata)
    return True
