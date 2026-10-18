"""棱柱四边形侧面被拆成两个三角形半区的面上 GPU 残差与 CPU 一致（I-181 / I-182）。

网格（`test_p0_mixed_split_faces`）：混合拆分面（一半挨四面体、一半在边界上，两种拆法）与真多源面（两半各挨
一个四面体），非均匀状态。GPU 走 numpy 替身（`_patch_gpu_modules`），比较的是同一份数据上两套实现的离散，不是硬件。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.unit.test_p0_mixed_split_faces import _mixed_mesh, _provider, _two_tet_mesh

MU, PR = 1.8e-5, 0.72


def _state(mesh, seed=3):
    rng = np.random.default_rng(seed)
    n_sps = mesh.n_sps_per_cell
    Q = np.empty((mesh.n_cells, n_sps, 5))
    Q[..., 0] = 1.2 + 0.05 * rng.standard_normal(Q.shape[:2])
    Q[..., 1:4] = np.array([30.0, 2.0, -1.0]) + 3.0 * rng.standard_normal(Q.shape[:2] + (3,))
    Q[..., 4] = 1.0e5 + 2.0e3 * rng.standard_normal(Q.shape[:2])
    return np.stack([primitive_to_conserved(q) for q in Q.reshape(-1, 5)]).reshape(Q.shape)


def _assert_close(got, ref):
    """逐变量相对误差（以该变量全场最大幅值为尺度；粘性连续方程恒为零的列按绝对值比较）。"""
    scale = np.abs(ref).max(axis=0)
    err = np.abs(got - ref) / np.where(scale > 0.0, scale, 1.0)
    assert err.max() < 1e-10, err.max(axis=0)


def _real_rows(mesh):
    from autoflowcfd.fr.native_padding import real_row_mask

    return real_row_mask(np.arange(mesh.n_cells) < mesh.n_prism_cells, mesh.n_sps_per_cell,
                         mesh.order).reshape(mesh.n_cells, -1)


_MESHES = {"mixed_a": lambda o: _mixed_mesh(o, (0, 1, 4)), "mixed_b": lambda o: _mixed_mesh(o, (0, 4, 3)),
           "two_tets": _two_tet_mesh}


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("kind", sorted(_MESHES))
def test_viscous_residual_matches_cpu_on_mixed_faces(order, kind):
    from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
    from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu

    mesh = _MESHES[kind](order)
    U = _state(mesh)
    prov = _provider(mesh)
    r_cpu = compute_viscous_residual_fr(U, mesh, mesh.operators, MU, PR, boundary_ghost_provider=prov)
    r_gpu = np.asarray(compute_viscous_residual_fr_gpu(U, mesh, mesh.operators, MU, PR, boundary_ghost_provider=prov))
    real = _real_rows(mesh)
    _assert_close(r_gpu[real], r_cpu[real])


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("kind", sorted(_MESHES))
def test_inviscid_residual_matches_cpu_on_mixed_faces(order, kind):
    from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
    from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu

    mesh = _MESHES[kind](order)
    U = _state(mesh)
    prov = _provider(mesh)
    r_cpu = compute_inviscid_residual_fr(U, mesh, mesh.operators, boundary_ghost_provider=prov, mach_ref=0.1)
    r_gpu = np.asarray(compute_inviscid_residual_fr_gpu(U, mesh, mesh.operators, boundary_ghost_provider=prov,
                                                        mach_ref=0.1))
    real = _real_rows(mesh)
    _assert_close(r_gpu[real], r_cpu[real])
