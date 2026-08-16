"""AutoFlowCFD V2.0 - 物理空间梯度算子单元测试。

核心判据：对任意非退化四面体/棱柱单元，线性物理函数的梯度必须精确
恢复为其真实常数梯度（这是曲边/坍缩坐标度量项变换是否正确的直接检验）。
"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.fr_gradients import compute_physical_gradient, compute_physical_scalar_gradient
from autoflowcfd.grid.curved_mapping import map_prism_to_physical, map_tet_to_physical
from autoflowcfd.fr.operators import generate_fr_operators, gauss_legendre
from autoflowcfd.grid.high_order_mesh import HighOrderMesh


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_mesh(order):
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0.2, 1, 0], [0.1, 0.3, 1],
            [5, 0, 0], [6, 0, 0], [5, 1, 0], [5.3, 0.2, 1], [6.1, 0.1, 1], [5.2, 1.3, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3]], dtype=np.int32)
    prism_conn = np.array([[4, 5, 6, 7, 8, 9]], dtype=np.int32)
    mock_volume = SimpleNamespace(
        cell_count=2,
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
        boundaries=None,
    )
    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume, build_faces=False)
    return mesh


def test_linear_function_gradient_exact_for_tet_and_prism():
    # P=2 是本项目当前实际生产阶数，要求机器精度；P=3 的四面体坍缩坐标
    # 模态基 Vandermonde 矩阵条件数已明显增长（真实测得 cond(V)~1e9，见
    # fr/collapsed_basis.py::jacobi_polynomial 文档——未做节点优化的
    # 张量积-Duffy 组合在高阶下的已知谱方法限制，不是正确性 bug），
    # 判据相应放宽但仍需明确、有界，如实记录已知数值局限而非静默放宽。
    tolerances = {1: 1e-9, 2: 1e-9, 3: 1e-6}
    for order in [1, 2, 3]:
        mesh = _build_mesh(order)
        a_coef = np.array([2.0, -3.0, 5.0])
        phi = mesh.sps_coords @ a_coef + 7.0  # (n_cells, n_sps)
        grad = compute_physical_scalar_gradient(phi, mesh, mesh.operators)
        max_err = np.max(np.abs(grad - a_coef))
        assert max_err < tolerances[order], f"order={order}: max_err={max_err}"


def test_multi_variable_field_gradient_matches_scalar_case():
    mesh = _build_mesh(order=2)
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 2.0, 0.0])
    phi1 = mesh.sps_coords @ a1
    phi2 = mesh.sps_coords @ a2
    field = np.stack([phi1, phi2], axis=-1)  # (n_cells, n_sps, 2)
    grad = compute_physical_gradient(field, mesh, mesh.operators)  # (n_cells,n_sps,2,3)
    assert np.allclose(grad[:, :, 0, :], a1, atol=1e-9)
    assert np.allclose(grad[:, :, 1, :], a2, atol=1e-9)
