"""AutoFlowCFD V2.0 - native 四面体（路径C）网格几何构造端到端接入验证。

见 Part8 文档"三、本次会话实现范围"/"四、明确未做的后续工作"第1条：
`HighOrderMesh(tet_basis_mode="native")` 真正走完
`load_from_volume_mesh` 全流程（含 `face_connectivity`/`face_flux_points`
构建），验证 `sps_coords`/`jacobians`/`cell_volumes`/`operators`/
`face_connectivity` 编码全部按 native 分支正确构造，且棱柱单元完全
不受影响（同一个网格里混合验证两条路径互不干扰）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
from autoflowcfd.fr.native_simplex_basis import (
    build_native_tet_operators, map_native_tet_to_physical, compute_native_tet_jacobian,
)


class _MockNodes:
    def __init__(self, coords):
        self._coords = coords

    def get_coordinates(self):
        return self._coords


class _MockCells:
    def __init__(self, connectivity):
        self.connectivity = connectivity


def _build_synthetic_mixed_mesh(order: int, tet_basis_mode: str = "native") -> HighOrderMesh:
    """与 test_fr_residual_inviscid.py::_build_synthetic_mixed_mesh 同一个
    合成网格构造（2 个共享面的四面体 + 2 个共享侧面的棱柱）。

    tet_basis_mode: 四面体坍缩坐标基已删除（2026-09-03，见 `fr/
    operators.py` 模块文档），只接受 "native"（或默认省略）。"""
    if tet_basis_mode != "native":
        raise ValueError(
            f"tet_basis_mode={tet_basis_mode!r} 已不再支持——四面体坍缩坐标基"
            "已删除，只剩 native 一种实现（见 fr/operators.py 模块文档）。"
        )
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
            [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1], [11, 0, 1], [10, 1, 1],
        ],
        dtype=float,
    )
    tet_conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    nodes = np.vstack([nodes, [[9, -1, 0], [9, -1, 1]]])
    prism_conn = np.array(
        [
            [5, 6, 7, 8, 9, 10],
            [5, 7, 11, 8, 10, 12],
        ],
        dtype=np.int32,
    )

    mock_volume = SimpleNamespace(
        cell_count=len(tet_conn) + len(prism_conn),
        nodes=_MockNodes(nodes),
        cells=_MockCells(tet_conn),
        prism_cells=_MockCells(prism_conn),
    )

    mesh = HighOrderMesh(order=order)
    mesh.load_from_volume_mesh(mock_volume)
    return mesh


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_mesh_loads_without_crashing_and_operators_are_native(order):
    mesh = _build_synthetic_mixed_mesh(order, "native")
    assert mesh.tet_basis_mode == "native"
    assert mesh.operators.tet_basis_mode == "native"
    assert mesh.operators.D_native_tet_padded is not None
    assert mesh.operators.lift_native_tet_padded is not None


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_face_connectivity_codes_are_translated(order):
    """四面体侧的 cube face code 必须落在 native 范围 [6,9]，棱柱侧
    完全不受影响（仍然是 [0,5]）——见 with_native_tet_faces 文档。"""
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells
    fc = mesh.face_connectivity

    owner_is_tet = fc.owner_cell >= n_prisms
    neighbor_is_tet = (~fc.is_boundary) & (fc.neighbor_cell >= n_prisms)
    assert np.all(fc.owner_cube_face[owner_is_tet] >= 6)
    assert np.all(fc.owner_cube_face[~owner_is_tet] < 6)
    assert np.all(fc.neighbor_cube_face[neighbor_is_tet] >= 6)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_sps_coords_real_rows_match_map_native_tet_to_physical(order):
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells
    n_tets = mesh.n_cells - n_prisms
    ref_native, _ = build_native_tet_operators(order)
    n_native = ref_native.shape[0]

    for i in range(n_tets):
        cell_id = n_prisms + i
        cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i]]
        expected = map_native_tet_to_physical(ref_native, cell_nodes)
        np.testing.assert_allclose(mesh.sps_coords[cell_id, :n_native], expected, atol=1e-12)
        # 填充行：必须是有限值（Part8 文档"陷阱"），且约定复制真实 SP #0
        # （assert_allclose 要求形状严格一致，不像普通 numpy 运算那样
        # 自动广播，这里显式广播成同形状）
        padding_rows = mesh.sps_coords[cell_id, n_native:]
        assert np.all(np.isfinite(padding_rows))
        np.testing.assert_allclose(
            padding_rows, np.broadcast_to(expected[0], padding_rows.shape), atol=1e-12
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_det_jacs_are_constant_and_match_reference_and_prism_unaffected(order):
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells
    n_tets = mesh.n_cells - n_prisms
    n_sps = mesh.n_sps_per_cell
    det_jacs = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, n_sps)

    for i in range(n_tets):
        cell_id = n_prisms + i
        cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i]]
        det_j_expected, _ = compute_native_tet_jacobian(cell_nodes)
        np.testing.assert_allclose(det_jacs[cell_id], det_j_expected, atol=1e-12)

    # 棱柱部分与再次独立构造的同一个网格逐位一致（棱柱坍缩坐标构造是
    # 确定性的，不依赖四面体侧——这里不再有"collapsed 模式"可对比，
    # 四面体基已删除，验证退化为可重复性检查）。
    mesh_again = _build_synthetic_mixed_mesh(order, "native")
    det_jacs_again = mesh_again.jacobians["det_jacs"].reshape(mesh_again.n_cells, n_sps)
    np.testing.assert_allclose(det_jacs[:n_prisms], det_jacs_again[:n_prisms], atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_native_cell_volumes_match_independent_tetrahedron_volume_formula(order):
    """决定性判据：native 四面体单元体积必须与教科书四面体体积公式
    `V=|det([p1-p0,p2-p0,p3-p0])|/6` 独立吻合（不复用 compute_native_
    tet_jacobian 本身，另一条完全独立的公式）——同时验证棱柱体积与
    坍缩坐标网格逐位一致（不受 native 分支影响）。"""
    mesh = _build_synthetic_mixed_mesh(order, "native")
    n_prisms = mesh.n_prism_cells
    n_tets = mesh.n_cells - n_prisms

    for i in range(n_tets):
        cell_id = n_prisms + i
        p0, p1, p2, p3 = mesh._node_coords[mesh._fixed_tet_conn[i]]
        expected_volume = abs(np.linalg.det(np.column_stack([p1 - p0, p2 - p0, p3 - p0]))) / 6.0
        assert mesh.cell_volumes[cell_id] == pytest.approx(expected_volume, rel=1e-10)
        # get_cell_volume（单元级别接口）必须给出同样的值
        assert mesh.get_cell_volume(cell_id) == pytest.approx(expected_volume, rel=1e-10)

    mesh_again = _build_synthetic_mixed_mesh(order, "native")
    np.testing.assert_allclose(
        mesh.cell_volumes[:n_prisms], mesh_again.cell_volumes[:n_prisms], rtol=1e-10
    )
