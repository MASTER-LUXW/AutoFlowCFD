# -*- coding: utf-8 -*-
"""湍流模型用的速度梯度（`fr_operators/corrected_gradient.py`）。

判据：

1. **P0 不再恒为零**：Couette 流 `u = U y/H`（下壁静止、上壁以 U 运动）在 P0 下的梯度
   就是 Green–Gauss 梯度，`du/dy` 必须是 `U/H`（贴壁单元的公共值取壁面速度）；修复前
   湍流源项用的单元内梯度在 P0 下恒为零，SST 产生项随之为零；
2. **离散散度定理**：提升把面积分原样搬进单元，`sum_cells V g = sum_boundary phi* n dA`
   （P0 的质量矩阵就是单元体积）；
3. **提升算子本身在高阶上正确**：P1/P2 下连续线性场在单元间没有跳变，提升修正梯度必须
   等于精确梯度（直接测核心 `corrected_gradient`）；
4. **阶数策略**：P>=1 返回的必须逐位就是单元内多项式导数（P>=1 加提升会把欠分辨跳变
   放大成伪生成项，槽道 P3 隐式稳态发散，见模块文档）；
5. **GPU 与 CPU 同一份算法**：P0 下 GPU 包装函数（numpy 冒充 cupy）与 CPU 一致。
"""

import numpy as np
import pytest

from autoflowcfd.fr.native_padding import real_row_mask
from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.validation._channel_mesh import build_channel_mesh_prism, build_face_exact_ghost_provider

RHO, U, P = 1.225, 30.0, 101325.0
LX, H, LZ = 0.4, 0.1, 0.08


def _couette_solver(order):
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(order, nx=3, ny=4, nz=2, Lx=LX, H=H, Lz=LZ)
    bc = {n: {"type": "SYMMETRY"} for n in ("z_min", "z_max")}
    bc["wall_bottom"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [0.0, 0.0, 0.0]}
    bc["wall_top"] = {"type": "WALL", "is_no_slip": True, "wall_velocity": [U, 0.0, 0.0]}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, 0.5 * U, 0.0, 0.0, P]}
    solver = FRSolver(mesh=mesh, order=order, rho_inf=RHO, vel_inf=0.5 * U, p_inf=P,
                      mu_molecular=1.8e-5, bc_overrides=bc)
    solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    y = np.asarray(mesh.sps_coords)[..., 1]
    Q = np.asarray(solver.state.Q).copy()
    Q[..., 0], Q[..., 1], Q[..., 2:4], Q[..., 4] = RHO, U * y / H, 0.0, P
    return solver, Q


def _flat(solver):
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
    return get_flat_face_geometry(solver.mesh, solver.ops)


def _source_gradient(solver, Q, ghost_provider):
    from autoflowcfd.core.fr_operators.corrected_gradient import source_velocity_gradient
    flat = _flat(solver)
    return source_velocity_gradient(Q, solver.mesh, solver.ops, flat, ghost_provider), flat


def _lifted_gradient(solver, Q, ghost_provider):
    """直接调用提升核心（与阶数策略无关），CPU 原语与生产包装函数同一组。"""
    from autoflowcfd.core.fr_operators.corrected_gradient import corrected_gradient
    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
    from autoflowcfd.core.turbulence.transport.faces import (
        _extrapolate_scalar_to_faces, _extrapolate_scalar_to_faces_neighbor_frame, _lift_side_jumps,
    )

    mesh, ops, flat = solver.mesh, solver.ops, _flat(solver)
    vel = np.ascontiguousarray(Q[..., 1:4])
    ghost = compute_boundary_ghost_states(flat, np.ascontiguousarray(Q[..., :5]), ghost_provider)[..., 1:4]

    def extrapolate_pair(comp, frame):
        if frame == "owner":
            return _extrapolate_scalar_to_faces(comp, flat, ops, mesh)
        return _extrapolate_scalar_to_faces_neighbor_frame(comp, flat)

    return corrected_gradient(np, vel, compute_physical_gradient(vel, mesh, ops), np.ascontiguousarray(ghost),
                              flat, extrapolate_pair,
                              lambda jo, jn: _lift_side_jumps(jo, jn, +1.0, flat, mesh))


def _real(solver, order):
    mesh = solver.mesh
    return real_row_mask(np.arange(mesh.n_cells) < mesh.n_prism_cells, mesh.n_sps_per_cell, order)


def test_p0_couette_shear_is_green_gauss():
    solver, Q = _couette_solver(0)
    g, flat = _source_gradient(solver, Q, solver.boundary_ghost_provider)
    dudy = g[:, 0, 0, 1]
    # 修复前（单元内梯度）这里恒为 0
    np.testing.assert_allclose(dudy, U / H, rtol=1e-10)
    # 其余分量为零；两端远场的幽灵态是来流（u = U/2），与线性场不符，贴两端的单元
    # 有 du/dx（物理上正确），不参与这条判据
    x_end = np.zeros(solver.mesh.n_cells, dtype=bool)
    bnd = np.nonzero(flat.is_boundary)[0]
    x_end[flat.owner_cell[bnd[np.abs(flat.owner_unit_normal[bnd, 0, 0]) > 0.99]]] = True
    others = g[~x_end, 0].copy()
    others[:, 0, 1] = 0.0
    assert x_end.sum() < solver.mesh.n_cells
    assert np.abs(others).max() < 1e-9 * U / H


def test_p0_discrete_divergence_theorem():
    """`sum_c V_c g_c = sum_boundary A phi* n`（P0，任意非线性场）。"""
    from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states

    solver, Q = _couette_solver(0)
    rng = np.random.default_rng(3)
    Q[..., 1:4] = rng.uniform(-5.0, 5.0, size=Q[..., 1:4].shape)
    g, flat = _source_gradient(solver, Q, solver.boundary_ghost_provider)
    vol = np.asarray(solver.mesh.cell_volumes, dtype=float)
    lhs = np.einsum("c,cvd->vd", vol, g[:, 0])

    ghost = compute_boundary_ghost_states(flat, np.ascontiguousarray(Q[..., :5]), solver.boundary_ghost_provider)
    # 每个单元面恰有一条 primary 记录（提升只作用在它上面），面积分也只按它求和
    bnd = np.nonzero((flat.neighbor_src0_cell < 0) & (flat.neighbor_src1_idx < 0) & flat.owner_is_primary)[0]
    owner_val = Q[flat.owner_cell[bnd], 0, 1:4]                                  # P0 外插即单元值
    star = 0.5 * (owner_val[:, None, :] + ghost[bnd][..., 1:4])                  # (nb, n_fp, 3)
    w = flat.ref_area_weight[None, :] * np.linalg.norm(flat.owner_adj_row_exact[bnd], axis=-1)
    rhs = np.einsum("bf,bfv,bfd->vd", w, star, flat.owner_unit_normal[bnd])
    np.testing.assert_allclose(lhs, rhs, atol=1e-10 * np.abs(rhs).max())


@pytest.mark.parametrize("order", [1, 2])
def test_lifting_is_exact_on_linear_field(order):
    from autoflowcfd.core.fr_residual.inviscid import DefaultGhostProvider

    solver, Q = _couette_solver(order)
    x = np.asarray(solver.mesh.sps_coords)
    A = np.array([[0.3, -1.2, 0.7], [2.1, 0.4, -0.5], [-0.8, 1.6, 0.2]]) * 100.0
    Q[..., 1:4] = 10.0 + x @ A.T
    g = _lifted_gradient(solver, Q, DefaultGhostProvider())     # 幽灵态 = 本侧：边界无跳变
    g = g.reshape(-1, 3, 3)[_real(solver, order)]                 # 原生棱柱填充槽位不是自由度
    np.testing.assert_allclose(g, np.broadcast_to(A, g.shape), atol=1e-9 * np.abs(A).max())


@pytest.mark.parametrize("order", [1, 2])
def test_high_order_uses_cell_gradient(order):
    from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient

    solver, Q = _couette_solver(order)
    rng = np.random.default_rng(7)
    Q[..., 1:4] += rng.uniform(-3.0, 3.0, size=Q[..., 1:4].shape)     # 有跳变的场
    g, _ = _source_gradient(solver, Q, solver.boundary_ghost_provider)
    np.testing.assert_array_equal(g, compute_physical_gradient(np.ascontiguousarray(Q[..., 1:4]),
                                                               solver.mesh, solver.ops))
    # 同一个场上提升修正确实不同——上面的相等不是因为跳变恰好为零
    assert not np.allclose(_lifted_gradient(solver, Q, solver.boundary_ghost_provider), g)


def test_gpu_matches_cpu_p0(monkeypatch):
    import autoflowcfd.core.gpu.residual.gpu_corrected_gradient as gcg_mod
    import autoflowcfd.core.gpu.residual.gpu_gradients as gg_mod
    import autoflowcfd.core.gpu.residual.gpu_inviscid as gi_mod
    import autoflowcfd.core.gpu.residual.gpu_volume_contract as gvc_mod
    import autoflowcfd.core.gpu.turbulence.gpu_scalar_transport as gst_mod
    from tests.unit.test_gpu_solver_turbulence_source import _NumpyAsCupy, _prepare_mesh_ops_data

    shim = _NumpyAsCupy()
    patch_module_get_cupy(monkeypatch, [gg_mod, gi_mod, gvc_mod, gst_mod], shim)
    solver, Q = _couette_solver(0)
    rng = np.random.default_rng(11)
    Q[..., 1:4] += rng.uniform(-3.0, 3.0, size=Q[..., 1:4].shape)
    g_cpu, flat = _source_gradient(solver, Q, solver.boundary_ghost_provider)
    md = _prepare_mesh_ops_data(solver.mesh, solver.ops)
    g_gpu = gcg_mod.source_velocity_gradient_gpu(shim, Q, md, md, flat, flat, solver.boundary_ghost_provider, 0)
    np.testing.assert_allclose(g_gpu, g_cpu, rtol=1e-12, atol=1e-12 * np.abs(g_cpu).max())
