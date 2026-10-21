"""局部 CFL 步长的三项限制只有一份（`core/fr_solver/cfl_limits.py`，I-115）：面谱半径的 numba 核与向量化表达式
（GPU 用）逐位一致；单 GPU 与多 GPU 后端调用同一个函数、整场一次计算（此前逐解点循环调用一份 GPU 重写）。"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.cfl_limits import (
    _face_spectral_sum_vectorized, face_spectral_sum, local_time_step_limits,
)
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _faces(mesh):
    fc = mesh.face_connectivity
    unit = fc.normal / np.linalg.norm(fc.normal, axis=1, keepdims=True)
    return fc.owner_cell, np.where(fc.is_boundary, 0, fc.neighbor_cell), fc.is_boundary, unit, fc.area


def test_vectorized_spectral_sum_matches_kernel():
    mesh = _build_synthetic_mixed_mesh(2)
    rng = np.random.default_rng(3)
    shape = (mesh.n_cells, mesh.n_sps_per_cell)
    u, v, w = (rng.standard_normal(shape) * 30.0 for _ in range(3))
    sound = 340.0 + rng.random(shape)
    faces = _faces(mesh)
    np.testing.assert_allclose(_face_spectral_sum_vectorized(np, u, v, w, sound, *faces),
                               face_spectral_sum(u, v, w, sound, *faces), rtol=1e-14)


@pytest.mark.parametrize("mach_ref", [None, 0.1])
def test_limits_are_the_minimum_of_three(mach_ref):
    """逐解点结果不大于每一项单独的限制，且处处等于其中之一。"""
    from autoflowcfd.core.fr_solver.cfl_viscous import viscous_length_scale_sq, viscous_time_step_limit

    mesh = _build_synthetic_mixed_mesh(1)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0], Q[..., 1], Q[..., 4] = 1.2, 30.0, 1e5
    faces = _faces(mesh)
    vol = mesh.get_all_cell_volumes()
    Lc2 = viscous_length_scale_sq(vol, faces[0], faces[1], faces[2], faces[4])
    det = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    mfs = np.ones((n_cells, n_sps))
    mu = np.full((n_cells, n_sps), 1.8e-5)
    phys, mean = local_time_step_limits(Q, vol, *faces, det, mfs, mu, Lc2, 0.5, 1, mach_ref=mach_ref)
    visc = viscous_time_step_limit(Q[..., 0], Lc2[:, None], mu, 0.5, 1)
    assert np.all(phys <= visc * (1 + 1e-15))
    assert (mean is phys) == (mach_ref is None)
    if mach_ref is not None:
        assert np.all(mean >= phys * (1 - 1e-15))      # 有效声速不大于物理声速


def test_det_shape_mismatch_is_an_error():
    mesh = _build_synthetic_mixed_mesh(1)
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.ones((n_cells, n_sps, 5))
    faces = _faces(mesh)
    with pytest.raises(ValueError, match="det"):
        local_time_step_limits(Q, mesh.get_all_cell_volumes(), *faces, np.ones((n_cells, n_sps + 1)),
                               np.ones((n_cells, n_sps)), np.ones((n_cells, n_sps)), np.ones(n_cells), 0.5, 1)


def test_gpu_backends_call_the_shared_limits():
    from tests.unit._module_source import module_source

    for mod in ("autoflowcfd.core.gpu.solver.gpu_solver", "autoflowcfd.core.gpu.distributed.gpu_distributed"):
        src = module_source(mod)
        assert "local_time_step_limits(" in src and "for sp in range(n_sps)" not in src, mod
    assert "compute_local_cfl_step_gpu" not in module_source("autoflowcfd.core.gpu.gpu_time_integration")
