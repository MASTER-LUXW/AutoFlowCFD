"""AutoFlowCFD V2.0 - FR 粘性残差单元测试 (S-03)。

核心判据：均匀常数流场（零梯度）的粘性残差必须严格为零——牛顿粘性应力
张量和傅里叶热传导对零梯度场恒为零，这是比 BR1 界面耦合本身更基础但
同样严格的判据（任何度量项/界面耦合的符号错误都会破坏这个恒等式）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_synthetic_mixed_mesh(order: int) -> HighOrderMesh:
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array([[5, 6, 7, 8, 9, 10], [5, 7, 11, 8, 10, 12]], dtype=np.int32)

    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
        boundaries=None,
    )
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


class TestViscousResidualUniformFlow:
    @pytest.mark.parametrize("order", [2, 3])
    def test_uniform_flow_gives_zero_viscous_residual(self, order):
        mesh = _build_synthetic_mixed_mesh(order)

        Q_inf = np.array([1.225, 30.0, 5.0, -3.0, 101325.0])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        residual = compute_viscous_residual_fr(U, mesh, mesh.operators, mu=1.8e-5, Pr=0.72)
        rel_res = np.max(np.abs(residual)) / 1e5  # 相对能量通量典型量级归一化

        assert rel_res < 1e-6, f"Uniform-flow viscous residual not zero at P={order}: rel={rel_res:.3e}"

    def test_no_nan_with_turbulent_viscosity(self):
        mesh = _build_synthetic_mixed_mesh(order=2)
        Q_inf = np.array([1.225, 30.0, 5.0, -3.0, 101325.0])
        U_inf = primitive_to_conserved(Q_inf)
        U = np.tile(U_inf, (mesh.n_cells, mesh.n_sps_per_cell, 1))

        mu_t_field = np.ones((mesh.n_cells, mesh.n_sps_per_cell)) * 1e-4
        residual = compute_viscous_residual_fr(U, mesh, mesh.operators, mu=1.8e-5, Pr=0.72, mu_t_field=mu_t_field, Pr_t=0.9)
        assert np.all(np.isfinite(residual))
