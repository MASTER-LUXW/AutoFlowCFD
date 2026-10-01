"""AutoFlowCFD V2.0 - 粘性稳定性步长限制（CPU 与 GPU 后端共用，数组模块无关）。

    dt_visc = 0.25 * CFL * rho * Lc2 / (mu_eff * (1 + 3 c_ip)) / (2p+1)^2,
    Lc2     = 6 V^2 / sum_f A_f^2

各项的推导与实测（各向异性长度尺度、IP 罚项刚性、阶数收紧）见
`cfl.py::compute_local_time_step` 里 `Lc2` 与 `_ip_stiffness` 两处的完整说明。

## 为什么抽出来（2026-10-01）

此前 GPU 有一份自己的粘性限制（`gpu_time_integration.compute_local_cfl_step_gpu`），
停留在 CPU 早已修掉的旧形式（各向同性 `V^(2/3)`、不含罚项刚性）；而且单机 GPU
与多 GPU 两个调用方都**没有传有效粘度**，粘性限制整段不生效。粘性系数一旦远大于
分子粘度（湍流涡粘、问题单元人工粘性可达分子粘度的上千倍），显式推进就会失稳。
两份公式只改一份是本项目反复出现的缺陷，所以只留这一份。
"""

import numpy as np

from autoflowcfd.core.utils.array_module import array_module as _array_module


def _scatter_add(xp, out, idx, val):
    if xp is np:
        np.add.at(out, idx, val)
    else:
        import cupyx
        cupyx.scatter_add(out, idx, val)


def viscous_length_scale_sq(volumes, owner_cell, neighbor_cell, is_boundary, face_areas):
    """逐单元 `Lc2 = 6 V^2 / sum_f A_f^2`（立方体上等于 `V^(2/3)`）。"""
    xp = _array_module(volumes)
    sq = face_areas * face_areas
    s = xp.zeros(volumes.shape[0], dtype=xp.float64)
    _scatter_add(xp, s, owner_cell, sq)
    internal = ~is_boundary
    if bool(xp.any(internal)):
        _scatter_add(xp, s, neighbor_cell[internal], sq[internal])
    return 6.0 * volumes * volumes / xp.maximum(s, 1e-300)


def viscous_time_step_limit(rho, Lc2, mu_eff, cfl: float, poly_order: int):
    """逐点粘性步长限制；`rho`/`mu_eff` 与 `Lc2` 可广播（例如 (n,n_sps) 与 (n,1)）。"""
    from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant

    xp = _array_module(rho)
    order_factor_viscous = 1.0 / (2 * poly_order + 1) ** 2
    ip_stiffness = 1.0 + 3.0 * resolve_viscous_ip_constant(poly_order)
    return (0.25 * cfl * order_factor_viscous * rho * Lc2
            / xp.maximum(mu_eff * ip_stiffness, 1e-30))
