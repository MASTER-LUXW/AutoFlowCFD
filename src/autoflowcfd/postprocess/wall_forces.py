"""壁面力：在壁面通量点上用求解器施加的同一组数值通量积分（与离散守恒一致）。

- 压力部分：AUSM+up 公共无粘通量（本侧外插态 / 边界幽灵态，`face_point_jumps.inviscid_common_flux_point`）的
  动量分量，扣除参考压力 `p_ref n`——物体面不封闭（贴地接触口、半模对称面切开的物体）时
  `∮ p_ref n dA != 0`，绝对压力积分会多出 `p_ref x 开口投影面积`。
- 粘性部分：含内罚项的粘性公共通量（`face_point_jumps.viscous_common_flux_point`，边界"另一侧"梯度按粘性
  边界种类构造）。P0 时单元内梯度为零，壁面摩擦完全来自罚项——与求解器实际施加的一致；只用单元内梯度
  外插会在 P0 得到恒为零的摩阻。
- 作用在物体上的力 = sum_fp w (F_inv - F_vis)[动量] - p_ref sum_fp w a（a 为本侧外向逆变度量行，流体 -> 物体）。
- WMLES：物理无滑移壁面的幽灵态是滑移镜像，粘性公共通量的切向牵引为零，壁面剪应力来自壁面模型——力积分
  用残差里同一个牵引（`core/turbulence/wmles/`），物体受力为 -sum A tau_w（tau_w 为壁面对流体的牵引）。
- 缺省只积分物理无滑移的 WALL 组：滑移壁（隧道壁、远场滑移面）不属于物体；按名字显式指定时可以取任意 WALL 组。

混合拆分面（棱柱四边形侧面一半在边界）上落在边界半区的通量点同样计入：它们的通量由配对的内部面记录
在掩码通量点上施加（`fr/face_flux_points/merge.py`）。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels.viscous_bc import boundary_other_gradients
from autoflowcfd.core.fr_operators.small_dense import extrap_tensor3x3, matvec_small
from autoflowcfd.core.fr_residual.face_point_jumps import inviscid_common_flux_point, viscous_common_flux_point
from autoflowcfd.core.fr_residual.inviscid_kernel import _extrap_matmul
from autoflowcfd.core.fr_operators.face_points import flux_point_positions
from autoflowcfd.core.fr_residual.wall_points import wall_flux_points


@dataclass
class WallSamples:
    """逐壁面通量点的积分量（每行一个通量点）。"""
    force_inviscid: np.ndarray   # (m, 3) w * F_inv[动量] - p_ref * w * a
    force_viscous: np.ndarray    # (m, 3) -w * F_vis[动量]
    position: np.ndarray         # (m, 3) 通量点物理坐标
    group_code: np.ndarray       # (m,) 所属边界组编码


@njit(cache=True, parallel=True)
def _wall_point_fluxes(rec, ghost, fp, owner_cell, owner_face_op, E_all, adj_rows, ref_w, ip_length,
                       Q, Q_ghost, grad_vel, grad_T, mu_t, vbc_kind,
                       mach_ref, precond_mode, mu, Pr, Pr_t, c_ip, p_ref, viscous):
    m = rec.shape[0]
    f_inv = np.zeros((m, 3))
    f_vis = np.zeros((m, 3))
    for k in prange(m):
        f = rec[k]
        g = ghost[k]
        i = fp[k]
        oc = owner_cell[f]
        E = E_all[owner_face_op[f]][i:i + 1]                  # (1, n_sps)
        Q_o = _extrap_matmul(Q[oc], E)[0]
        a = adj_rows[f, i]
        w = ref_w[i]
        F = inviscid_common_flux_point(Q_o, Q_ghost[g, i], a, mach_ref, precond_mode)
        for d in range(3):
            f_inv[k, d] = w * (F[1 + d] - p_ref * a[d])
        if viscous:
            gv_o = extrap_tensor3x3(grad_vel[oc], E)[0]
            gT_o = _extrap_matmul(grad_T[oc], E)[0]
            mut_o = matvec_small(E, mu_t[oc])[0]
            bk = vbc_kind[g]
            gv_x, gT_x = boundary_other_gradients(gv_o, gT_o, a, bk)
            G = viscous_common_flux_point(Q_o, gv_o, gT_o, mut_o, Q_ghost[g, i], gv_x, gT_x, mut_o,
                                          a, ip_length[f], bk, mu, Pr, Pr_t, c_ip)
            for d in range(3):
                f_vis[k, d] = -w * G[1 + d]
    return f_inv, f_vis


def sample_wall(solver, codes: Sequence[int], viscous: bool = True) -> WallSamples:
    """求解器当前状态在 `codes` 组壁面通量点上的力积分量（单机求解器或其主机视图）。"""
    from autoflowcfd.boundary.fr_ghost_state import build_viscous_boundary_kind
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
    from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode
    from autoflowcfd.core.fr_residual.viscous_flux.constants import PRANDTL, PRANDTL_TURBULENT
    from autoflowcfd.core.fr_residual.viscous_flux.pointwise import compute_temperature

    host = solver.host_view() if hasattr(solver, "host_view") else solver
    mesh, ops, provider = host.mesh, host.ops, host.boundary_ghost_provider
    flat = get_flat_face_geometry(mesh, ops)
    rec, ghost, fp, code = wall_flux_points(flat, np.asarray(provider.group_code), codes)
    Q = np.ascontiguousarray(host.state.Q[..., :5])
    n_cells, n_sps = Q.shape[:2]
    # 只为用到的边界面求幽灵态（与残差同一个外插算子与同一个 provider 批量接口）
    used, ghost_pos = np.unique(ghost, return_inverse=True)
    E_g = flat.boundary_extrap_native[flat.owner_face_op[used]]
    Q_ghost = np.ascontiguousarray(provider.ghost_states(
        used, np.einsum('fps,fsv->fpv', E_g, Q[flat.owner_cell[used]]), flat.true_normal[used]))
    if viscous:
        grad_vel = np.ascontiguousarray(compute_physical_gradient(Q, mesh, ops)[:, :, 1:4, :])
        T = compute_temperature(Q)
        grad_T = np.ascontiguousarray(compute_physical_gradient(T[:, :, None], mesh, ops)[:, :, 0, :])
        mu_t = host._get_turbulent_viscosity_field()
        mu_t = np.zeros((n_cells, n_sps)) if mu_t is None else np.ascontiguousarray(mu_t)
    else:
        grad_vel = np.zeros((1, n_sps, 3, 3))
        grad_T = np.zeros((1, n_sps, 3))
        mu_t = np.zeros((1, n_sps))
    vbc_kind = build_viscous_boundary_kind(flat.n_faces, flat.is_boundary, provider)
    vbc_kind = np.ascontiguousarray(vbc_kind[used])
    f_inv, f_vis = _wall_point_fluxes(
        rec, ghost_pos.astype(np.int64), fp, flat.owner_cell, flat.owner_face_op, flat.boundary_extrap_native,
        flat.owner_adj_row_exact, flat.ref_area_weight, flat.ip_length,
        Q, Q_ghost, grad_vel, grad_T, mu_t, vbc_kind,
        float(host.freestream["mach_ref"]), int(resolve_ausm_precond_mode()),
        float(host.mu_molecular), PRANDTL, PRANDTL_TURBULENT, resolve_viscous_ip_constant(int(mesh.order)),
        float(host.freestream["p_inf"]), bool(viscous))
    wmles = getattr(host, "wmles_model", None)
    if viscous and wmles is not None:
        _add_wall_model_traction(f_vis, wmles, host, mesh, flat, provider, rec, fp)
    # 力臂：通量点的物理坐标（唯一来源；P0 每面一个通量点，解点外插只能得到单元中心）
    pos = flux_point_positions(mesh, flat, rec, fp)
    return WallSamples(f_inv, f_vis, pos, code)


def _add_wall_model_traction(f_vis, wmles, host, mesh, flat, provider, rec, fp) -> None:
    """WMLES 壁面模型牵引计入粘性力（残差里同一个牵引；只在壁面模型覆盖的通量点上，即物理无滑移壁面）。"""
    U = np.ascontiguousarray(host.state.U[..., :5])
    m, tau = wmles.traction(np, mesh, flat, provider, U, float(host.mu_molecular))
    if m is None:
        return
    n_fp = int(flat.n_fp)
    key_model = m.face * n_fp + m.fp
    key_sample = rec * n_fp + fp
    order = np.argsort(key_model)
    pos = np.minimum(np.searchsorted(key_model[order], key_sample), len(order) - 1)
    idx = order[pos]
    hit = key_model[idx] == key_sample
    f_vis[hit] -= m.area[idx[hit], None] * tau[idx[hit]]


def wall_group_codes(provider, groups: Optional[Sequence[str]] = None) -> Dict[int, str]:
    """参与积分的组：缺省为全部物理无滑移的 WALL 组（滑移壁不属于物体，模块文档）；`groups` 给出时按名字取
    （必须存在且是 WALL 组，滑移壁也可以显式指定）。"""
    names = dict(getattr(provider, "code_to_name", {}) or {})
    wall = {code: names.get(code, str(code)) for code, cfg in provider.code_to_config.items() if cfg.get("type") == "WALL"}
    if groups is None:
        body = {code: name for code, name in wall.items()
                if provider.code_to_config[code].get("physical_no_slip", True)}
        if not body:
            raise RuntimeError("网格没有物理无滑移的 WALL 边界组，无法积分气动力（可按名字显式指定滑移壁组）")
        return body
    by_name = {name: code for code, name in wall.items()}
    missing = [g for g in groups if g not in by_name]
    if missing:
        raise ValueError(f"组 {missing} 不是 WALL 边界组（可选：{sorted(by_name)}）")
    return {by_name[g]: g for g in groups}
