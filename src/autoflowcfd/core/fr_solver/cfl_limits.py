"""AutoFlowCFD V2.0 - 局部 CFL 步长的三项稳定性限制（CPU、CPU MPI、单 GPU、多 GPU 全部后端共用，数组模块无关）

逐解点取三项中最严的一项（推导与实测见 `cfl.py::compute_local_time_step` 文档）：

1. 对流：`dt = CFL / (2p+1) * V / S`，面谱半径 `S[c, sp] = sum_{f in faces(c)} (|u[c, sp] . n_f| + c_s[c, sp]) A_f`
2. 粘性：`cfl_viscous.viscous_time_step_limit`（各向异性长度尺度 + IP 罚项刚性 + 阶数收紧）
3. 几何 / 度量：`dt = CFL * |det J| / (sum_m ||adj(J)[m, :]|| * (|u| + c_s))`，逐解点的有限体积 CFL 类比

`c_s` 为物理声速；启用伪时间低马赫预处理时平均流另算一份，对流项与几何项里的声速换成有效声速
（`preconditioning.preconditioned_sound_speed`，只有 Gamma 真的作用到平均流残差上时才成立），粘性限制与声速无关、
原样复用。

2026-10-22（I-115）以前 GPU 把对流谱半径、几何限制、预处理声速各重写一遍（`gpu_time_integration.
compute_local_cfl_step_gpu`，逐解点循环调用、调用方再取单元最小），只有粘性限制与 CPU 共用。面谱半径在 numpy 上
走逐解点并行的 numba 核，其余数组模块（CuPy）走同一公式的向量化表达式。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.thermo import sound_speed
from autoflowcfd.core.utils.array_module import array_module, scatter_add


@njit(cache=True, parallel=True)
def _face_spectral_sum_kernel(u, v, w, sound, owner, neighbor, is_boundary, unit_normal, area):
    """面谱半径（见模块文档 1）。`u/v/w/sound`: (n_cells, n_sps)；面数组长度 n_faces；返回 (n_cells, n_sps)。
    每个解点一列，列与列之间无耦合，按解点并行。"""
    n_cells, n_sps = sound.shape
    n_faces = owner.shape[0]
    out = np.zeros((n_sps, n_cells))
    for sp in prange(n_sps):
        col = out[sp]
        for f in range(n_faces):
            nx, ny, nz, a_f = unit_normal[f, 0], unit_normal[f, 1], unit_normal[f, 2], area[f]
            o = owner[f]
            col[o] += (abs(u[o, sp] * nx + v[o, sp] * ny + w[o, sp] * nz) + sound[o, sp]) * a_f
            if not is_boundary[f]:
                nb = neighbor[f]
                col[nb] += (abs(u[nb, sp] * nx + v[nb, sp] * ny + w[nb, sp] * nz) + sound[nb, sp]) * a_f
    return np.ascontiguousarray(out.T)


def face_spectral_sum(u, v, w, sound, owner, neighbor, is_boundary, unit_normal, area):
    """面谱半径 `S` (n_cells, n_sps)（模块文档 1）；边界面只累加 owner 侧（neighbor 处的值不读）。"""
    xp = array_module(sound)
    if xp is np:
        f64 = (np.ascontiguousarray(x, dtype=np.float64) for x in (u, v, w, sound))
        return _face_spectral_sum_kernel(*f64, np.ascontiguousarray(owner, dtype=np.int64),
                                         np.ascontiguousarray(neighbor, dtype=np.int64),
                                         np.ascontiguousarray(is_boundary, dtype=np.bool_),
                                         np.ascontiguousarray(unit_normal, dtype=np.float64),
                                         np.ascontiguousarray(area, dtype=np.float64))
    return _face_spectral_sum_vectorized(xp, u, v, w, sound, owner, neighbor, is_boundary, unit_normal, area)


def _face_spectral_sum_vectorized(xp, u, v, w, sound, owner, neighbor, is_boundary, unit_normal, area):
    """`_face_spectral_sum_kernel` 的向量化表达式（全部解点一次完成，重复下标经 scatter_add 累加）。"""
    nx, ny, nz = (unit_normal[:, k:k + 1] for k in range(3))
    out = xp.zeros(sound.shape, dtype=xp.float64)

    def side(cells, sel):
        return (xp.abs(u[cells] * nx[sel] + v[cells] * ny[sel] + w[cells] * nz[sel]) + sound[cells]) * area[sel, None]

    every = xp.arange(owner.shape[0])
    scatter_add(xp, out, owner, side(owner, every))
    internal = xp.nonzero(~is_boundary)[0]
    scatter_add(xp, out, neighbor[internal], side(neighbor[internal], internal))
    return out


def _require_shape(field, shape: tuple, what: str) -> None:
    """逐解点量必须与当前状态同形 `(n_cells, n_sps)`（同一组解点）。"""
    if tuple(field.shape) != tuple(shape):
        raise ValueError(f"CFL 步长计算：{what} 的形状 {tuple(field.shape)} 与当前状态 {tuple(shape)} 不符"
                         "——几何与状态不在同一组解点上（换阶时须同步切换）")


def local_time_step_limits(Q, volumes, owner, neighbor, is_boundary, unit_normal, area, det_jacs, metric_flux_scale,
                           mu_eff, Lc2, cfl: float, poly_order: int, mach_ref=None):
    """逐解点步长 `(dt_physical, dt_mean_flow)`，各 (n_cells, n_sps)（模块文档三项取最严）。

    Args:
        Q: (n_cells, n_sps, >=5) 原始变量 (rho, u, v, w, p)
        volumes: (n_cells,) 单元体积；`Lc2`: (n_cells,) 粘性长度尺度平方（`cfl_viscous.viscous_length_scale_sq`）
        owner, neighbor, is_boundary, unit_normal, area: 逐面连接关系、单位法向与物理面积
        det_jacs, metric_flux_scale, mu_eff: (n_cells, n_sps) 解点 det(J)、`sum_m ||adj(J)[m, :]||`、有效动力粘度
        mach_ref: 给出时按预处理有效声速另算平均流那一份；None 时两份是同一个数组对象
    """
    from autoflowcfd.core.fr_solver.cfl_viscous import viscous_time_step_limit

    xp = array_module(Q)
    rho, u, v, w, p = (Q[..., m] for m in range(5))
    for field, what in ((det_jacs, "网格 det(J)"), (metric_flux_scale, "度量通量尺度"), (mu_eff, "有效粘度")):
        _require_shape(field, rho.shape, what)
    a = sound_speed(xp, rho, p)          # 密度已在状态换算里取下限（与残差同一个换算）
    vel_mag = xp.sqrt(u * u + v * v + w * w)
    advective_scale = cfl / (2 * poly_order + 1) * volumes[:, None]
    abs_det = xp.abs(det_jacs)

    def advective_and_geometric(sound):
        spectral = face_spectral_sum(u, v, w, sound, owner, neighbor, is_boundary, unit_normal, area)
        dt_adv = advective_scale / xp.maximum(spectral, 1e-30)
        dt_geo = cfl * abs_det / xp.maximum(metric_flux_scale * xp.maximum(vel_mag + sound, 1e-10), 1e-300)
        return xp.minimum(dt_adv, dt_geo)

    dt_visc = viscous_time_step_limit(rho, Lc2[:, None], mu_eff, cfl, poly_order)
    dt_physical = xp.minimum(advective_and_geometric(a), dt_visc)
    if mach_ref is None:
        return dt_physical, dt_physical
    from autoflowcfd.core.utils.preconditioning import preconditioned_sound_speed

    dt_mean = xp.minimum(advective_and_geometric(preconditioned_sound_speed(vel_mag, a, float(mach_ref))), dt_visc)
    return dt_physical, dt_mean
