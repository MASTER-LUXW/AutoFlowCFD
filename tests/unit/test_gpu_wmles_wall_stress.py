"""单 GPU 的 WMLES 壁面剪应力修正（`GPUFRSolver.compute_viscous_residual_gpu` 里的生产调用点）。

GPU 与 CPU 共用同一份数组模块无关的壁面模型（`core/turbulence/wall_model/`），这里在 numpy 替身上跑 GPU 的真实
残差方法（粘性通量核换成返回零，隔离出壁面模型一项），与 CPU 直接求值逐位比对；此前 GPU 路径是把设备数组
下载到主机、调用 CPU 函数再上传的转发层。
"""

import types

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.turbulence.wall_model import EquilibriumWallModel
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit._boundary_physics import ghost_provider
from tests.unit._boundary_groups import tag_wall_cells
from tests.unit._numpy_as_cupy import NumpyAsCupy
from tests.unit._patch_pkg import patch_pkg_attr
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

MU = 1.8e-5


def _case(wall: bool):
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    fc = mesh.face_connectivity
    if wall:
        tag_wall_cells(mesh, [int(fc.owner_cell[np.nonzero(fc.is_boundary)[0][0]])])
    else:
        tag_wall_cells(mesh, [], rest_type="FARFIELD")
    provider = ghost_provider(mesh, {"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0}, "WMLES", wall_modelled=True)
    rng = np.random.default_rng(0)
    Q = np.zeros((mesh.n_cells, mesh.n_sps_per_cell, 5))
    Q[..., 0] = 1.225 * (1.0 + rng.uniform(-0.02, 0.02, Q.shape[:2]))
    Q[..., 1:4] = rng.uniform(-5.0, 5.0, Q.shape[:2] + (3,)) + np.array([30.0, 0.0, 0.0])
    Q[..., 4] = 101325.0
    U = np.stack([primitive_to_conserved(q) for q in Q.reshape(-1, 5)]).reshape(Q.shape)
    return mesh, ops, provider, U


def _gpu_viscous_residual(monkeypatch, mesh, ops, provider, U, model):
    import autoflowcfd.core.gpu.residual.gpu_viscous as gpu_viscous_mod
    import autoflowcfd.core.gpu.solver.gpu_solver.residual as residual_mod
    from autoflowcfd.core.gpu.solver.gpu_solver.residual import _GPUSolverResidualMixin

    monkeypatch.setattr(residual_mod, "get_cupy", lambda: NumpyAsCupy())
    patch_pkg_attr(monkeypatch, gpu_viscous_mod, "compute_viscous_residual_fr_gpu",
                   lambda U, *a, **k: np.zeros_like(U))
    solver = types.SimpleNamespace(
        mesh=mesh, ops=ops, mu_molecular=MU, boundary_ghost_provider=provider, wall_model=model,
        mesh_data=None, ops_data=None, flat_face_gpu=None, device_id=0, U_gpu=U,
    )
    return _GPUSolverResidualMixin.compute_viscous_residual_gpu(solver, U_trial=U)


def test_gpu_residual_path_matches_cpu_model(monkeypatch):
    mesh, ops, provider, U = _case(wall=True)
    expected = EquilibriumWallModel().residual_correction(np, mesh, get_flat_face_geometry(mesh, ops), provider, U, MU)
    assert expected is not None and np.any(expected != 0.0)
    actual = _gpu_viscous_residual(monkeypatch, mesh, ops, provider, U, EquilibriumWallModel())
    np.testing.assert_array_equal(actual, expected)


def test_no_physical_wall_adds_nothing(monkeypatch):
    mesh, ops, provider, U = _case(wall=False)
    assert EquilibriumWallModel().residual_correction(np, mesh, get_flat_face_geometry(mesh, ops), provider, U,
                                                      MU) is None
    actual = _gpu_viscous_residual(monkeypatch, mesh, ops, provider, U, EquilibriumWallModel())
    assert not actual.any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
