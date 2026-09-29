"""AutoFlowCFD V2.0 - 标量场的面外插与面校正提升（对流/扩散两条残差共用）。

两侧各在自己的通量点顺序里构造跳变量、按统一符号约定提升，理由与实测见
`face_frames.py` 模块文档（2026-09-26 真实缺陷修复：此前 neighbor 侧用的是
owner 顺序的跳变量，且扩散在 neighbor 侧符号相反）。
"""

import os
from typing import NamedTuple

import numba
import numpy as np

from autoflowcfd.core.fr_operators.volume_contract import contravariant_flux_from_metric

from .face_frames import (
    extrapolate_scalar_pair_kernel,
    face_mass_flux_kernel,
    lift_side_jumps_kernel,
    lift_side_jumps_kernel_colored,
)


def _wall_masks(flat, wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value):
    if wall_dirichlet_zero_face is None:
        wall_dirichlet_zero_face = np.zeros(flat.n_faces, dtype=np.bool_)
    if has_wall_dirichlet_value is None:
        has_wall_dirichlet_value = np.zeros(flat.n_faces, dtype=np.bool_)
    if wall_dirichlet_value_face is None:
        wall_dirichlet_value_face = np.zeros((flat.n_faces, flat.n_fp), dtype=np.float64)
    return wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value


def _extrapolate_scalar_to_faces(
    scalar_sps, flat, ops, mesh, wall_dirichlet_zero_face=None,
    wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
):
    """owner 坐标系下的 `(phi_owner_fp, phi_neighbor_fp)`，各 `(n_faces, n_fp)`。

    owner 侧用自身外插 `boundary_extrap_native[code-6]`，neighbor 侧用
    `neighbor_src`（邻居解点 -> owner 的通量点），都在 **owner 的通量点顺序**里。
    真边界面的 ghost 规则见 `face_frames.extrapolate_scalar_pair_kernel`：

    Args:
        wall_dirichlet_zero_face: (n_faces,) bool，WALL 上 k=0（ghost=-owner）。
        wall_dirichlet_value_face, has_wall_dirichlet_value: 非零 Dirichlet
            目标值（omega 壁面解析式，ghost=2*target-owner），与上一个互斥。
        其余边界：零梯度（ghost=owner）。开放边界的来流值由对流残差按质量通量
        方向另行覆盖（`compute_scalar_convection_residual`）。
    """
    wz, wv, hv = _wall_masks(flat, wall_dirichlet_zero_face, wall_dirichlet_value_face,
                             has_wall_dirichlet_value)
    return extrapolate_scalar_pair_kernel(
        scalar_sps, flat.owner_cell, flat.owner_cube_face, flat.boundary_extrap_native,
        flat.neighbor_src0_cell, flat.neighbor_src0_tpl, flat.neighbor_src0_tid,
        flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
        True, wz, hv, wv, flat.mixed_nb_partner, flat.mixed_nb_mask,
    )


def _extrapolate_scalar_to_faces_neighbor_frame(
    scalar_sps, flat, wall_dirichlet_zero_face=None,
    wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
):
    """neighbor 坐标系下的 `(phi_neighbor_fp, phi_owner_at_neighbor_fp)`。

    neighbor 侧用自身外插、owner 侧用 `owner_src`（owner 解点 -> neighbor 的通量
    点），都在 **neighbor 的通量点顺序**里（与平均流 kernel 的 neighbor-primary
    分支同一结构）。边界面（无 neighbor）两者为 0、不被消费；混合拆分面的边界
    半区按配对边界面的壁面掩码覆盖（该配对面的 owner 就是 neighbor 单元）。
    """
    wz, wv, hv = _wall_masks(flat, wall_dirichlet_zero_face, wall_dirichlet_value_face,
                             has_wall_dirichlet_value)
    return extrapolate_scalar_pair_kernel(
        scalar_sps, flat.neighbor_cell, flat.neighbor_cube_face, flat.boundary_extrap_native,
        flat.owner_src0_cell, flat.owner_src0_tpl, flat.owner_src0_tid,
        flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
        False, wz, hv, wv, flat.mixed_ow_partner, flat.mixed_ow_mask,
    )


def boundary_diffusion_targets(xp, flat, frame, wall_dirichlet_zero_face=None,
                               wall_dirichlet_value_face=None, has_wall_dirichlet_value=None):
    """某一侧坐标系下扩散界面项的边界点分类，返回 `(is_boundary, is_dirichlet, target)`，
    各 `(n_faces, n_fp)`；`xp` 为数组模块（CPU numpy / GPU cupy 共用这一份规则）。

    边界点：owner 坐标系下的真边界面（没有任何 neighbor 来源）全部通量点，以及
    混合拆分面（B-8）配对边界半区的通量点（两个坐标系各自的 `mixed_*_mask`）。
    类型取该边界面（混合面取配对边界面）的壁面掩码：`wall_dirichlet_zero_face`
    为 Dirichlet 目标 0（k），`has_wall_dirichlet_value` 为逐通量点目标值（omega
    壁面解析值），其余为齐次 Neumann。
    """
    n_faces, n_fp = flat.owner_adj_row_exact.shape[:2]
    wz = (xp.zeros(n_faces, dtype=bool) if wall_dirichlet_zero_face is None
          else xp.asarray(wall_dirichlet_zero_face, dtype=bool))
    hv = (xp.zeros(n_faces, dtype=bool) if has_wall_dirichlet_value is None
          else xp.asarray(has_wall_dirichlet_value, dtype=bool))
    wv = (xp.zeros((n_faces, n_fp)) if wall_dirichlet_value_face is None
          else xp.asarray(wall_dirichlet_value_face))
    if frame == "owner":
        true_bnd = (flat.neighbor_src0_cell < 0) & (flat.neighbor_src1_idx < 0)
        partner, mask = flat.mixed_nb_partner, flat.mixed_nb_mask
    else:
        true_bnd = xp.zeros(n_faces, dtype=bool)
        partner, mask = flat.mixed_ow_partner, flat.mixed_ow_mask
    mp = xp.maximum(partner, 0)
    in_mixed = (partner >= 0)[:, None] & mask
    is_bnd = true_bnd[:, None] | in_mixed
    zero = xp.where(in_mixed, wz[mp][:, None], wz[:, None])
    value = xp.where(in_mixed, hv[mp][:, None], hv[:, None]) & ~zero
    target = xp.where(value, xp.where(in_mixed, wv[mp], wv), 0.0)
    return is_bnd, is_bnd & (zero | value), target


def unit_normals(adj_row):
    """`(n_faces, n_fp, 3)` 的 adj 行归一化成该侧的单位外法向。"""
    mag = np.sqrt(np.sum(adj_row * adj_row, axis=-1))
    return adj_row / np.maximum(mag, 1e-300)[..., None]


class ScalarConvectionGeometry(NamedTuple):
    """k/omega 两次标量对流调用共享的、**与标量本身无关**的几何/流场量
    （2026-09-13 性能优化：两次调用的 `rho`/`velocity` 相同，只算一次）。

    - `rho_u_tilde`：逆变质量通量 adj(J) @ (rho*u)，体积项用（标量可以从度量
      乘法里提出来：`adj_j@(rho*u*phi) == phi*(adj_j@(rho*u))`）；
    - `mass_flux` / `mass_flux_neighbor`：两侧坐标系下通量点上的质量通量，见
      `face_frames.face_mass_flux_kernel`。两者都取 **owner 的迹**，于是同一个
      物理点上 `mass_flux_neighbor = -mass_flux`，公共通量单值、上风选择两侧一致。
    """
    rho_u_tilde: np.ndarray          # (n_cells, n_sps, 3)
    mass_flux: np.ndarray            # (n_faces, n_fp)，owner 顺序、owner 外法向
    mass_flux_neighbor: np.ndarray   # (n_faces, n_fp)，neighbor 顺序、neighbor 外法向


def precompute_scalar_convection_geometry(rho, velocity, mesh, ops, flat):
    """算出 k/omega 共享的 `ScalarConvectionGeometry`（见该类文档）。"""
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    rho_u = np.ascontiguousarray(rho[:, :, None] * velocity)          # (n_cells,n_sps,3)
    rho_u_tilde = contravariant_flux_from_metric(det_jacs, inv_jacs, rho_u[..., None])[..., 0]
    mass_flux, mass_flux_neighbor = face_mass_flux_kernel(
        rho_u, flat.owner_cell, flat.owner_cube_face, flat.neighbor_cell, flat.neighbor_cube_face,
        flat.boundary_extrap_native,
        flat.owner_src0_cell, flat.owner_src0_tpl, flat.owner_src0_tid,
        flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
        flat.mixed_ow_partner, flat.mixed_ow_mask,
        np.ascontiguousarray(unit_normals(flat.owner_adj_row_exact)),
        np.ascontiguousarray(unit_normals(flat.neighbor_adj_row_exact)),
    )
    return ScalarConvectionGeometry(rho_u_tilde=rho_u_tilde, mass_flux=mass_flux,
                                    mass_flux_neighbor=mass_flux_neighbor)


def _lift_side_jumps(jump_owner, jump_neighbor, sign, flat, mesh):
    """两侧（各自坐标系下的）物理跳变量提升回解点，返回 `(n_cells, n_sps)`。

    `sign`：对流 -1（`dphi/dt = -div F`）、扩散 +1（`dphi/dt = +div G`），见
    `face_frames.py` 模块文档"提升的符号约定只有一个"。默认图着色（同色面不共享
    单元，直接写共享缓冲）；`AFCFD_USE_COLORING=0` 退回逐线程私有缓冲。
    """
    n_cells = mesh.n_cells
    n_sps = flat.n_sps
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    jump_owner = np.ascontiguousarray(jump_owner)
    jump_neighbor = np.ascontiguousarray(jump_neighbor)
    args = (flat.owner_cell, flat.neighbor_cell, flat.owner_cube_face, flat.neighbor_cube_face,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact, flat.ref_area_weight,
            flat.lift_native, flat.owner_is_primary, flat.neighbor_is_primary, det_jacs)
    if os.environ.get("AFCFD_USE_COLORING", "1") == "1":
        out = np.zeros((n_cells, n_sps))
        for c in range(flat.n_colors):
            face_indices = flat.color_face_indices[c]
            if len(face_indices) == 0:
                continue
            lift_side_jumps_kernel_colored(jump_owner, jump_neighbor, float(sign), *args,
                                           face_indices, out)
        return out
    return lift_side_jumps_kernel(jump_owner, jump_neighbor, float(sign), *args,
                                  numba.get_num_threads())
