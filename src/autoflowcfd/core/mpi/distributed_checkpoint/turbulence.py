"""AutoFlowCFD V2.0 - 分布式 checkpoint 的湍流场（CPU 分布式与多 GPU 共用）。

写出端把本 rank 的输运场（模型声明的 `TRANSPORTED_FIELDS`，与单机 `write_checkpoint` 同一组字段名）
与涡粘收集成全局数组，另写后处理用的单元平均（`turb_cell_*`，`core/turbulence/output.py`）；读回端
（`solve resume` 与 `solve transient --init-from`）从全局字段切出本 rank 的 local 段交给
`model.restore_transported`（SST 的 omega 可容许性投影、SA-neg 的壁面解点置零都在里面）。

**2026-10-04 修复**：此前分布式 checkpoint 只写平均流 `U_sps`，分布式 resume 后湍流场从来流初值
重新开始；`--init-from` 则从 `U_sps[...,5:7]/rho` 换算 k/omega——那是 SST 状态数组里从未更新的
历史槽位。
"""

import numpy as np

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
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)


def global_cell_is_prism(solver, n_global: int):
    """全局单元的棱柱标志（root 有值）：本 rank 的 local 单元在紧凑排列（"棱柱在前"）里的位置
    小于紧凑棱柱数即为棱柱，逐单元收集——两种分布式模式都适用（完全分布式加载下 `solver.mesh`
    只有本 rank 的紧凑几何，没有全局棱柱数）。集体调用。"""
    n_local = int(solver.partition.n_local_cells)
    dist_fc = solver.dist_flat_face
    is_prism = (np.asarray(dist_fc.inv_perm)[:n_local] < int(dist_fc.base_flat.n_prism)).astype(np.float64)
    out = gather_global_state(is_prism[:, None, None], solver.partition.local_cells, n_global)
    return None if out is None else out[:, 0, 0] > 0.5


def gather_turbulence_fields(solver, n_global: int, order: int, is_prism_global):
    """输运场、涡粘与单元平均的全局字段（root 返回字典，其余 rank 返回 None）。集体调用。"""
    model, _ = transported_model(solver)
    if model is None:
        return {}
    n_local = int(solver.partition.n_local_cells)
    cols = [_host(f)[:n_local] for f in model.transported_fields()]
    cols.append(_host(model.nu_t)[:n_local])
    stacked = gather_global_state(np.stack(cols, axis=-1), solver.partition.local_cells, n_global)
    if stacked is None:
        return None
    names = model.TRANSPORTED_FIELDS
    out = {name: np.ascontiguousarray(stacked[..., j]) for j, name in enumerate(names)}
    out["nu_t"] = np.ascontiguousarray(stacked[..., len(names)])

    from autoflowcfd.fr.native_padding import reduce_rows_over_real_sps
    from autoflowcfd.core.turbulence.output import CHECKPOINT_PREFIX

    for key, field in zip(tuple(model.OUTPUT_FIELD_KEYS) + ("nut",), [out[n] for n in names] + [out["nu_t"]]):
        out[CHECKPOINT_PREFIX + key] = reduce_rows_over_real_sps(field, is_prism_global, order, "mean")
    return out


def restore_turbulence_fields(solver, fields: dict, metadata: dict, *, source: str) -> bool:
    """由全局字段恢复本 rank 的输运场与涡粘，并按 `metadata` 记录的进度续接产生项渐变（与单机恢复端共用
    `restore_production_ramp`）。字段缺失（层流、另一个湍流模型或旧版本 checkpoint）时保留构造时的来流初值
    并返回 False。"""
    from autoflowcfd.core.fr_solver.turbulence.init import restore_production_ramp
    from autoflowcfd.core.mpi import is_root

    model, is_gpu = transported_model(solver)
    if model is None:
        return False
    names = model.TRANSPORTED_FIELDS
    if not all(name in fields for name in names):
        if is_root():
            print(f"   ⚠️  Checkpoint 缺少湍流场 {'/'.join(names)}（层流、另一个湍流模型或旧版本）："
                  "湍流场从来流初值开始。")
        return False
    local_cells = solver.partition.local_cells
    n_sps = int(solver.state.n_sps)
    local = []
    for name in names + (("nu_t",) if "nu_t" in fields else ()):
        f = np.asarray(fields[name])
        if f.shape != (int(solver.partition.n_global_cells), n_sps):
            raise ValueError(f"Checkpoint 字段 {name} 形状 {f.shape} 与求解器的全局形状 "
                             f"({solver.partition.n_global_cells}, {n_sps}) 不匹配")
        local.append(f[local_cells].copy())
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
