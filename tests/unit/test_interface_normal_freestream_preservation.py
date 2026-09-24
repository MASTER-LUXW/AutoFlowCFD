# -*- coding: utf-8 -*-
"""无粘界面项必须对均匀流**精确为零**，不管两侧法向之间夹角多大。

## 钉住的真实缺陷（2026-09-24，plate_demo 长程运行第 4134 步发散的根因）

无粘界面核里 owner / neighbor 两侧各有一个兜底：

    alignment = dir(adj_row) . n_ref        # n_ref = ±true_normal
    if alignment < 0.5:
        dir = n_ref                          # 静默换一个法向
    F_common = AUSM(Q_o, Q_n, dir) * |adj_row|
    F_own    = adj_row . F(Q_side)           # 投影仍用 adj_row
    jump     = F_common - F_own

对均匀流 `F_common = |adj| F(Q).n_ref`、`F_own = |adj| F(Q).dir(adj)`，一旦
走了兜底就 `jump = |adj| F(Q).(n_ref - dir(adj)) != 0`——**凭空注入压力量级
的源项**（低马赫下压力通量远大于动压）。

它是原生基之前的遗留：那时法向由 Lagrange 外插的度量得到，坍缩顶点附近可能
是垃圾方向。原生基下 adj 行是逐通量点的解析精确值，夹角 >60° 说明几何本身
就是那样——plate_demo_volume_les 板锐边处扭曲的棱柱四边形面对两个平面三角形，
11 个面、42 个通量点触发。"全边界取远场"的均匀流里，这些单元的残差是
4~5e6 /s（其余单元 1e-7 量级）；删掉兜底后全场最大 4.0e-7 —— 降 13 个数量级。
长程运行里爆炸的那个单元，正是在这个恒定源项下某个解点密度按恒定斜率下降，
过零后被逐点硬钳，一步放大到 1e53。

粘性核本来就没有这个兜底（法向只取本侧度量），所以粘性残差一直是精确的。

## 判据

在合成混合网格上人为把某个面的一侧度量行**旋转 70°**（cos70 = 0.34，
正好落在旧兜底的触发区间），模拟板锐边那种扭曲面。均匀流下界面修正必须
仍是舍入量级 —— 因为每一侧的 Riemann 通量与本侧投影用的是同一个法向。
"""

import numba
import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode
from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, primitive_to_conserved
from autoflowcfd.core.fr_residual.inviscid_kernel import (
    compute_boundary_ghost_states,
    compute_inviscid_interface_correction_kernel,
)
from autoflowcfd.core.fr_residual.inviscid_kernel_colored import (
    compute_inviscid_interface_correction_kernel_colored,
)
from autoflowcfd.fr.operators import generate_fr_operators

from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

_ROT_DEG = 70.0


def _rotate_rows(rows, deg):
    """把 (n_fp, 3) 的每一行绕一个与它垂直的轴旋转 `deg` 度，模长不变。"""
    out = np.empty_like(rows)
    th = np.deg2rad(deg)
    for i, r in enumerate(rows):
        mag = np.linalg.norm(r)
        u = r / mag
        # 取一个与 u 不平行的向量求垂直轴
        t = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        k = np.cross(u, t)
        k /= np.linalg.norm(k)
        # Rodrigues：k ⊥ u，所以 v = u cos + (k×u) sin
        v = u * np.cos(th) + np.cross(k, u) * np.sin(th)
        out[i] = mag * v
    return out


@pytest.fixture(scope="module")
def case():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    flat = get_flat_face_geometry(mesh, ops)
    Q_inf = np.array([1.225, 30.0, 5.0, -3.0, 101325.0])
    U = np.tile(primitive_to_conserved(Q_inf), (mesh.n_cells, mesh.n_sps_per_cell, 1))
    Q = conserved_to_primitive(U)
    det = np.asarray(mesh.jacobians["det_jacs"]).reshape(mesh.n_cells, mesh.n_sps_per_cell)

    # 挑一个 owner/neighbor 都是 primary 的内部面，把两侧度量行各转 70°
    isb = np.asarray(flat.is_boundary).astype(bool)
    both = (~isb) & np.asarray(flat.owner_is_primary).astype(bool) \
        & np.asarray(flat.neighbor_is_primary).astype(bool)
    f = int(np.where(both)[0][0])
    adj_o = np.array(flat.owner_adj_row_exact, copy=True)
    adj_n = np.array(flat.neighbor_adj_row_exact, copy=True)
    adj_o[f] = _rotate_rows(adj_o[f], _ROT_DEG)
    adj_n[f] = _rotate_rows(adj_n[f], _ROT_DEG)

    # 前提检查：旋转后的方向确实落在旧兜底的触发区间（否则这个测试没有区分度）
    tn = np.asarray(flat.true_normal)[f]
    d_o = adj_o[f] / np.linalg.norm(adj_o[f], axis=1, keepdims=True)
    d_n = adj_n[f] / np.linalg.norm(adj_n[f], axis=1, keepdims=True)
    assert np.all(np.sum(d_o * tn, axis=1) < 0.5)
    assert np.all(np.sum(d_n * (-tn), axis=1) < 0.5)

    from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider
    # 均匀流 + 零梯度幽灵态 = 精确解
    ghost = DefaultGhostProvider()          # 零梯度外插：对均匀流正好是精确解
    adj_j = None
    Q_ghost = compute_boundary_ghost_states(flat, Q, adj_j, ghost)
    return dict(mesh=mesh, flat=flat, Q=Q, det=det, adj_o=adj_o, adj_n=adj_n,
                Q_ghost=Q_ghost, face=f)


def _args_common(c):
    fl = c["flat"]
    return (fl.owner_cell, fl.neighbor_cell, fl.is_boundary,
            fl.owner_is_primary, fl.neighbor_is_primary,
            c["adj_o"], c["adj_n"],
            fl.neighbor_src0_cell, fl.neighbor_src0_mat,
            fl.neighbor_src1_idx, fl.neighbor_src1_cell, fl.neighbor_src1_mat,
            fl.owner_src0_cell, fl.owner_src0_mat,
            fl.owner_src1_idx, fl.owner_src1_cell, fl.owner_src1_mat,
            fl.mixed_nb_partner, fl.mixed_nb_mask,
            fl.mixed_ow_partner, fl.mixed_ow_mask,
            c["Q_ghost"])


def _tail(c):
    fl = c["flat"]
    return (fl.owner_cube_face, fl.neighbor_cube_face, fl.ref_area_weight,
            fl.boundary_extrap_native, fl.lift_native)


def _assert_zero(corr, c):
    scale = np.array([1.225, 1.225 * 30, 1.225 * 30, 1.225 * 30, 101325.0 / 0.4])
    rel = np.abs(corr[..., :5]) / scale
    worst = float(rel.max())
    assert worst < 1e-9, (
        f"均匀流下界面修正不为零（max 相对 {worst:.3e}）—— 两侧法向夹角 "
        f"{_ROT_DEG}° 时 Riemann 通量与本侧投影用了不同的法向，凭空注入源项")


def test_uncolored_kernel_preserves_freestream_under_rotated_metric(case):
    c = case
    corr = compute_inviscid_interface_correction_kernel(
        c["Q"], c["det"], *_args_common(c),
        numba.get_num_threads(), 0.1, resolve_ausm_precond_mode(),
        *_tail(c))
    _assert_zero(corr, c)


def test_colored_kernel_preserves_freestream_under_rotated_metric(case):
    c = case
    fl = c["flat"]
    n_cells, n_sps = c["Q"].shape[:2]
    corr = np.zeros((n_cells, n_sps, 5))
    for k in range(fl.n_colors):
        idx = fl.color_face_indices[k]
        if len(idx) == 0:
            continue
        compute_inviscid_interface_correction_kernel_colored(
            c["Q"], c["det"], *_args_common(c),
            idx, corr, 0.1, resolve_ausm_precond_mode(),
            *_tail(c))
    _assert_zero(corr, c)


def test_gpu_direction_is_the_own_metric_direction():
    """GPU 侧同一件事：方向只取本侧度量行，不再接受任何参考法向。"""
    import inspect

    from autoflowcfd.core.gpu.residual.gpu_inviscid.interface import _ausm_direction

    assert list(inspect.signature(_ausm_direction).parameters) == ["cp", "adjrow"], (
        "_ausm_direction 又带上了参考法向参数 —— 那就是被删掉的兜底")
    rng = np.random.default_rng(3)
    adj = rng.normal(size=(4, 4, 3))
    d, mag = _ausm_direction(np, adj)
    np.testing.assert_allclose(mag, np.linalg.norm(adj, axis=-1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(d, adj / np.linalg.norm(adj, axis=-1, keepdims=True),
                               rtol=0, atol=1e-15)
