"""AutoFlowCFD V2.0 - 物理空间梯度算子单元测试。

核心判据：对任意非退化四面体/棱柱单元，线性物理函数的梯度必须精确
恢复为其真实常数梯度（这是曲边/坍缩坐标度量项变换是否正确的直接检验）。
"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient, compute_physical_scalar_gradient
from autoflowcfd.grid.curved_mapping.curved_mapping import map_prism_to_physical, map_tet_to_physical
from autoflowcfd.fr.operators import generate_fr_operators, gauss_legendre
from autoflowcfd.fr.native_simplex_basis import build_native_tet_operators
from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


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
    # P=2 是本项目当前实际生产阶数，要求机器精度；P=3 容差沿用同一档
    # （四面体 native 单纯形基在高阶下的条件数特征与坍缩坐标不同，
    # 但同样在此判据下有界，见 native_simplex_basis.py 文档）。
    #
    # 2026-09-03 更正：四面体坍缩坐标基已删除（见 fr/operators.py 模块
    # 文档），四面体单元的梯度输出在"零填充块对角"约定下，填充行
    # （`[n_native:]`）恒为零梯度（`D_native_tet_padded` 的填充行本身
    # 是零，不是真实自由度的物理梯度取值，见 native_tet_padding.py
    # 文档）——这是既有、已验证的设计不变量，不是本次改动引入的新
    # 近似；对比时必须只看真实自由度（`[:n_native]`），不能再要求
    # 填充行也精确等于常数梯度。
    tolerances = {1: 1e-9, 2: 1e-9, 3: 1e-6}
    for order in [1, 2, 3]:
        mesh = _build_mesh(order)
        n_native = build_native_tet_operators(order)[0].shape[0]
        a_coef = np.array([2.0, -3.0, 5.0])
        phi = mesh.sps_coords @ a_coef + 7.0  # (n_cells, n_sps)
        grad = compute_physical_scalar_gradient(phi, mesh, mesh.operators)
        # 单元全局索引约定"棱柱在前、四面体在后"（见 HighOrderMesh 模块
        # 文档）：cell 0 是棱柱（不受本次删除 collapsed 四面体基影响，
        # 全部检查），cell 1 是四面体（只检查真实自由度，填充行恒为零
        # 梯度，见上）。
        max_err_prism = np.max(np.abs(grad[0] - a_coef))
        max_err_tet = np.max(np.abs(grad[1, :n_native] - a_coef))
        max_err = max(max_err_tet, max_err_prism)
        assert max_err < tolerances[order], f"order={order}: max_err={max_err}"


def test_multi_variable_field_gradient_matches_scalar_case():
    mesh = _build_mesh(order=2)
    n_native = build_native_tet_operators(2)[0].shape[0]
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 2.0, 0.0])
    phi1 = mesh.sps_coords @ a1
    phi2 = mesh.sps_coords @ a2
    field = np.stack([phi1, phi2], axis=-1)  # (n_cells, n_sps, 2)
    grad = compute_physical_gradient(field, mesh, mesh.operators)  # (n_cells,n_sps,2,3)
    # 见 test_linear_function_gradient_exact_for_tet_and_prism 同一处
    # 更正说明（"棱柱在前、四面体在后"）：cell 0 是棱柱，全部检查；
    # cell 1 是四面体，只检查真实自由度 [:n_native]。
    assert np.allclose(grad[0, :, 0, :], a1, atol=1e-9)
    assert np.allclose(grad[0, :, 1, :], a2, atol=1e-9)
    assert np.allclose(grad[1, :n_native, 0, :], a1, atol=1e-9)
    assert np.allclose(grad[1, :n_native, 1, :], a2, atol=1e-9)
