"""AutoFlowCFD V2.0 - 湍流 Newton 未知量的步后模态滤波（全部后端、全部输运模型共用的一份算法）。

滤波作用在被多项式表示的量上，即模型声明的 Newton 未知量（SST 为 `k` 与 `w = ln omega`、SA-neg
为 `nu_tilde`，`TransportedTurbulence.newton_unknowns`），写回经模型自己的映射（SST 的 omega 上界
在对数空间施加），随后过一遍正性/上界限制器（滤波矩阵含负权重，输出不天然满足约束）。

为什么需要它（2026-09-12，cube_demo 真实网格 P1 长程测试）与门控档 `AFCFD_FILTER_TURB_GATE` 的
理由见 `fr_solver/filter/scalar.py`。此前 CPU 在 `fr_solver/turbulence/source.py` 里写一份、GPU
在 `GPUTurbulenceSST.filter_fields_gpu` 里写一份（只认 k/omega）；现在两者只注入各自的滤波核
（`fr_solver/filter/scalar.py::CpuFilterKernels`、`gpu/gpu_modal_filter.py::GpuFilterKernels`）。
"""

from typing import Optional


def filter_turbulence_unknowns(model, kernels, n_prism: int, order: Optional[int], ops) -> Optional[float]:
    """对模型的每个 Newton 未知量施加模态滤波并写回（原地改模型的场）。

    滤波矩阵为单位阵（`AFCFD_FILTER_MODE=off`，或 P0 的 1x1 矩阵）时整段跳过
    （`fr_solver/turbulence/init.py::_filter_matrices_are_identity`）。

    Args:
        model: `TransportedTurbulence`（场所在的数组模块与 `kernels.xp` 一致）
        kernels: 后端滤波核（`xp` / `troubled_mask` / `full` / `gated`）
        n_prism: 棱柱单元数（"棱柱在前"排列）
        order: 当前阶数（只有门控档的传感器用；全场滤波档可为 None）
        ops: 带 `filter_prism` / `filter_tet` 的算子

    Returns:
        门控档（sensor）下被标记单元的比例；全场滤波档或跳过时为 None
    """
    from autoflowcfd.core.fr_solver.filter import resolve_turb_filter_gate
    from autoflowcfd.core.fr_solver.turbulence.init import _filter_matrices_are_identity

    if _filter_matrices_are_identity(ops):
        return None

    xp = kernels.xp
    shape = model.transported_fields()[0].shape
    x = model.newton_unknowns(xp)
    columns = [xp.ascontiguousarray(x[:, j]).reshape(shape) for j in range(x.shape[1])]
    frac = None
    if resolve_turb_filter_gate() == "sensor":
        # 传感器量的是多项式表示的光滑度，被表示的是未知量本身
        troubled = kernels.troubled_mask(columns, n_prism, int(order))
        frac = float(xp.mean(troubled))
        columns = [kernels.gated(c, n_prism, ops.filter_prism, ops.filter_tet, troubled) for c in columns]
    else:
        columns = [kernels.full(c, n_prism, ops.filter_prism, ops.filter_tet) for c in columns]
    model.set_newton_unknowns(xp.stack([xp.asarray(c).ravel() for c in columns], axis=1), xp)
    model.apply_positivity_limiter()
    return frac
