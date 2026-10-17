"""单 GPU P0 无粘残差必须在**试探态**上求值。

隐式 Newton 的矩阵自由 Jacobian-向量积与显式 RK 的各级都以 `U_trial` 调用
`compute_inviscid_residual_gpu`。此前 P0 分支取的是 `self.Q_gpu`（当前状态的原始变量），
试探态被忽略：J·v 恒为零、RK 各级退化成同一个残差。

P0 的 CUDA RawKernel 在本机无法执行，这里把它换成记录入参的替身，断言喂给内核的
单元原始变量与边界幽灵态都来自 `U_trial`（与 CPU 侧同一份幽灵态组装逐位一致）。
"""

import numpy as np

from tests.unit.test_gpu_solver_order_continuation import _make_solver, _patch_gpu_modules  # noqa: F401（自动夹具）


def test_p0_inviscid_residual_is_evaluated_at_trial_state(monkeypatch):
    import autoflowcfd.core.gpu.residual.gpu_p0_inviscid as p0_mod
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, primitive_to_conserved
    from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry, _precompute_ghost_states

    solver = _make_solver(0)
    assert solver.mesh.n_points_1d == 1
    n_cells, n_sps = solver.mesh.n_cells, solver.mesh.n_sps_per_cell

    rng = np.random.default_rng(3)
    Q_trial = np.zeros((n_cells, n_sps, 5))
    Q_trial[..., 0] = 1.225 * (1.0 + rng.uniform(-0.1, 0.1, size=(n_cells, 1)))
    Q_trial[..., 1] = 30.0 + rng.uniform(-5.0, 5.0, size=(n_cells, 1))
    Q_trial[..., 2] = rng.uniform(-5.0, 5.0, size=(n_cells, 1))
    Q_trial[..., 3] = rng.uniform(-5.0, 5.0, size=(n_cells, 1))
    Q_trial[..., 4] = 101325.0 * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, 1)))
    U_trial = primitive_to_conserved(Q_trial)
    assert not np.allclose(U_trial, solver.U_gpu[..., :5])

    seen = {}

    def recording_kernel(Q_flat, owner, neighbor, is_bnd, normal, area_w, volumes,
                         Q_ghost, n_cells_, n_faces, mach_ref=0.1):
        seen["Q_flat"] = np.array(Q_flat)
        seen["Q_ghost"] = np.array(Q_ghost)
        return np.zeros((n_cells_, 1, 5))

    monkeypatch.setattr(p0_mod, "compute_inviscid_residual_p0_cupy_gpu_resident", recording_kernel)
    solver.compute_inviscid_residual_gpu(U_trial)

    Q_expected = conserved_to_primitive(U_trial[:, 0, :5])
    np.testing.assert_allclose(seen["Q_flat"], Q_expected, rtol=1e-14, atol=0.0)

    fc = solver.mesh.face_connectivity
    normal, _ = _extract_p0_face_geometry(solver.mesh.face_flux_points, fc, fc.n_faces)
    ghost_expected = _precompute_ghost_states(solver.mesh.face_flux_points, fc, solver.boundary_ghost_provider,
                                              Q_expected, fc.n_faces, normal)
    np.testing.assert_allclose(seen["Q_ghost"], ghost_expected, rtol=1e-14, atol=0.0)


def test_p0_face_arrays_are_cached_per_mesh():
    """P0 面几何只在网格对象变化时重建（此前每次残差求值都重新提取并上传）。"""
    solver = _make_solver(0)
    first = solver._p0_face_arrays_gpu()
    assert solver._p0_face_arrays_gpu() is first
