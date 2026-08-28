"""AutoFlowCFD V2.0 - GPU 版 WMLES 壁面剪应力修正（gpu_turbulence_wmles.py）
接入测试。

`compute_wmles_wall_stress_correction_gpu` 本身不重新实现摩擦速度迭代
求解/坍缩坐标外插——直接复用已验证的 CPU 函数
`core/utils/solver_helpers.py::compute_wmles_wall_stress_correction`
本身，只做一层"GPU 数组下载成 numpy、包装成 CPU 函数期望的 facade、
结果再上传回 GPU"的透传。因此这里验证的重点不是壁面剪应力公式本身
（那部分是 CPU 既有代码，有自己的测试），而是这层 facade/round-trip
是否透明无损：用 numpy 直接模拟"CuPy 数组"（数组本身就是 numpy，
cp.asnumpy/cp.asarray 对 numpy 输入是恒等操作），验证 GPU 入口函数的
输出与直接调用 CPU 函数完全一致。
"""

import types
from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.turbulence.wmles import WMLESModel
from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction
from autoflowcfd.core.gpu.turbulence.gpu_turbulence_wmles import compute_wmles_wall_stress_correction_gpu
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _NumpyAsCupy:
    """把 numpy 数组本身当成"CuPy 数组"：asnumpy/asarray 对 numpy 输入是
    恒等操作，足以验证 facade 的往返逻辑本身是否透明无损。"""

    def asnumpy(self, x):
        return np.asarray(x)

    def asarray(self, x):
        return np.asarray(x)


@pytest.fixture(scope="function")
def mesh_and_ops():
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    return mesh, ops


def _make_wall_mock(mesh):
    """把该网格的前两个边界面标记为 WALL 组，其余保留为未匹配（-1）——
    足以让 compute_wmles_wall_stress_correction 里的
    `tag_boundary_groups_for_mesh` mock 产出至少一个真实 WALL 面。"""
    fc = mesh.face_connectivity
    n_faces = fc.n_faces
    boundary_idx = np.nonzero(fc.is_boundary)[0]
    group_code = np.full(n_faces, -1, dtype=np.int32)
    group_code[boundary_idx[:2]] = 0
    name_to_code = {"wall_group": 0}
    return group_code, name_to_code


class TestGpuWmlesWallStressFacadeMatchesCpu:
    def test_none_when_no_wmles_model(self, mesh_and_ops):
        mesh, ops = mesh_and_ops
        solver = types.SimpleNamespace(
            wmles_model=None, mesh=mesh, ops=ops, wall_distance_gpu=None, U_gpu=None, Q_gpu=None,
        )
        with patch("autoflowcfd.core.gpu.turbulence.gpu_turbulence_wmles.get_cupy", return_value=_NumpyAsCupy()):
            assert compute_wmles_wall_stress_correction_gpu(solver) is None

    def test_gpu_facade_matches_direct_cpu_call(self, mesh_and_ops):
        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell

        rng = np.random.default_rng(0)
        rho_inf, p_inf = 1.225, 101325.0
        Q = np.zeros((n_cells, n_sps, 5))
        Q[:, :, 0] = rho_inf
        Q[:, :, 1:4] = rng.uniform(-5.0, 5.0, size=(n_cells, n_sps, 3))
        Q[:, :, 4] = p_inf
        U = Q.copy()  # 本函数不读 U 的具体守恒量含义，只用其 .shape
        wall_distance = rng.uniform(1e-4, 1e-2, size=(n_cells, n_sps))

        wmles_model = WMLESModel(nu=1.5e-5)
        group_code, name_to_code = _make_wall_mock(mesh)
        mesh.boundary_bc_types = {"wall_group": "WALL"}

        gpu_solver = types.SimpleNamespace(
            wmles_model=wmles_model, mesh=mesh, ops=ops,
            wall_distance_gpu=wall_distance, U_gpu=U, Q_gpu=Q,
        )
        cpu_facade = types.SimpleNamespace(
            wmles_model=wmles_model, mesh=mesh, ops=ops, wall_distance=wall_distance,
            state=types.SimpleNamespace(U=U, Q=Q),
        )

        with patch(
            "autoflowcfd.grid.connectivity.face_connectivity.tag_boundary_groups_for_mesh",
            return_value=(group_code, name_to_code),
        ), patch(
            "autoflowcfd.core.gpu.turbulence.gpu_turbulence_wmles.get_cupy", return_value=_NumpyAsCupy(),
        ):
            expected = compute_wmles_wall_stress_correction(cpu_facade)
            actual = compute_wmles_wall_stress_correction_gpu(gpu_solver)

        assert expected is not None, "test setup must produce at least one real WALL face"
        np.testing.assert_array_equal(actual, expected)

    def test_negative_control_zero_wall_faces_returns_none(self, mesh_and_ops):
        """反向对照：如果没有任何面被标记为 WALL，两条路径都应返回 None
        （不是恰好数值相等的巧合）。"""
        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        Q = np.zeros((n_cells, n_sps, 5))
        Q[:, :, 0] = 1.225
        Q[:, :, 4] = 101325.0
        wall_distance = np.ones((n_cells, n_sps)) * 1e-3
        wmles_model = WMLESModel(nu=1.5e-5)

        fc = mesh.face_connectivity
        group_code = np.full(fc.n_faces, -1, dtype=np.int32)
        name_to_code = {}

        gpu_solver = types.SimpleNamespace(
            wmles_model=wmles_model, mesh=mesh, ops=ops,
            wall_distance_gpu=wall_distance, U_gpu=Q, Q_gpu=Q,
        )
        with patch(
            "autoflowcfd.grid.connectivity.face_connectivity.tag_boundary_groups_for_mesh",
            return_value=(group_code, name_to_code),
        ), patch(
            "autoflowcfd.core.gpu.turbulence.gpu_turbulence_wmles.get_cupy", return_value=_NumpyAsCupy(),
        ):
            assert compute_wmles_wall_stress_correction_gpu(gpu_solver) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
