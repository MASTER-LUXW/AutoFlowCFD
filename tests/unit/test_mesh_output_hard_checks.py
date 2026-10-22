"""体网格输出前的硬检查（I-136 / I-151）。

此前网格生成流水线多处"降级继续"：删除退化单元、非流形面保留最大的单元、空腔重铺失败时回退删除，零面积面
"Allowing export for debugging" 照样导出——生成末尾只校验外表面覆盖，不查退化单元与非流形面；外部体网格的边界
映射把匹配不上的外表面单元放进 'UNCLASSIFIED' 并按 WALL 处理，且按单元覆盖写组（角部单元只留最后一个组）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import (
    map_boundaries_by_geometry, map_generated_boundaries,
)
from autoflowcfd.grid.schema.grid_boundaries import BoundaryMap
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from autoflowcfd.grid.validation.cell_validity import degenerate_cells, require_nondegenerate_cells
from tests.unit.test_exterior_faces_by_group import _volume_mesh
from tests.unit.test_generated_boundary_mapping import _cube_with_surface

_PRISM_XYZ = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
_NO_PRISMS = np.empty((0, 6), np.int64)


def test_prism_check_is_orientation_agnostic_and_catches_a_folded_prism():
    """两种朝向的正常棱柱都通过（生成网格里两种朝向各约一半，加载高阶网格时统一）；把一个顶点翻到底面另一侧的
    棱柱三个子四面体体积取模后之和仍为正，只看体积会漏掉，角点行列式异号检出。"""
    prisms = np.array([[0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2]])
    assert degenerate_cells(_PRISM_XYZ, prisms).size == 0
    folded = _PRISM_XYZ.copy()
    folded[4] = [1.0, 0.0, -0.5]
    from autoflowcfd.grid.validation.quality_metrics import compute_prism_volumes
    assert compute_prism_volumes(folded, prisms[:1])[0] > 0.1
    assert degenerate_cells(folded, prisms[:1]).tolist() == [0]


def test_degenerate_cells_are_reported_with_global_mixed_index():
    xyz = np.vstack([_PRISM_XYZ, [[2, 0, 0], [3, 0, 0], [2, 1, 0], [2.5, 0.5, 0.0]]])   # 末一个四面体四点共面
    with pytest.raises(ValueError, match=r"四面体 1 个：1（质心"):
        require_nondegenerate_cells(xyz, np.array([[0, 1, 2, 3, 4, 5]]), np.array([[6, 7, 8, 9]]))
    require_nondegenerate_cells(xyz, np.array([[0, 1, 2, 3, 4, 5]]), np.empty((0, 4), np.int64))


def test_face_shared_by_three_cells_is_an_error():
    nodes, tets, sn, sf, sb = _cube_with_surface()
    with pytest.raises(RuntimeError, match="shared by more than 2 cells"):
        map_generated_boundaries(nodes, _NO_PRISMS, np.vstack([tets, tets[:1]]), sn, sf, sb)


def test_degenerate_face_is_an_error():
    """面积为正但相对退化的面（三点几乎共线，A ~ 1e-15 L^2）。"""
    nodes, tets, sn, sf, sb = _cube_with_surface()
    a, b, c = nodes[tets[0, :3]]
    sliver = len(nodes)
    nodes = np.vstack([nodes, 0.5 * (a + b) + 1e-15 * (c - a)])
    with pytest.raises(ValueError, match="个退化"):
        map_generated_boundaries(nodes, _NO_PRISMS, np.vstack([tets, [[*tets[0, :3], sliver]]]), sn, sf, sb)


def test_flat_cell_on_the_surface_passes_coverage_but_not_the_cell_check():
    """顶点落在外表面三角形上的扁四面体：三个新外部面恰好细分那个三角形，覆盖检查通过——退化只能由单元检查
    （`generate_hybrid_mesh` 输出前的 `require_nondegenerate_cells`）发现。"""
    nodes, tets, sn, sf, sb = _cube_with_surface()
    from tests.unit.test_cavity_weld_conformity import _exterior
    face = np.array(sorted(_exterior(tets)[0])[0])
    flat = len(nodes)
    nodes = np.vstack([nodes, nodes[face].mean(axis=0)])
    tets = np.vstack([tets, [[*face, flat]]])
    map_generated_boundaries(nodes, _NO_PRISMS, tets, sn, sf, sb)
    with pytest.raises(ValueError, match=f"四面体 1 个：{len(tets) - 1}（"):
        require_nondegenerate_cells(nodes, _NO_PRISMS, tets)


def _external(vm, nodes, sb):
    surface_grid = SimpleNamespace(nodes=NodeArray.from_array(vm.surface_mesh["nodes"]),
                                   cells=SimpleNamespace(connectivity=vm.surface_mesh["faces"]), boundaries=sb)
    vm.surface_mesh = None
    vm.metadata.file_format = "external_volume_mesh"
    return surface_grid


def test_external_mapping_keeps_corner_cells_in_every_group():
    """角部单元同时拥有两个组的外部面：两个组都要有它（此前按单元覆盖写，只留最后写入的组）。"""
    vm, nodes, tets = _volume_mesh()
    expected = vm.boundaries.groups
    bm = map_boundaries_by_geometry(vm, _external(vm, nodes, vm.surface_mesh["boundaries"]))
    assert set(bm.groups) == set(expected)
    for name, cells in expected.items():
        assert np.array_equal(np.sort(bm.groups[name]), np.sort(cells))
    assert sum(map(len, bm.groups.values())) > len(np.unique(np.concatenate(list(bm.groups.values()))))


def test_external_mapping_unmatched_faces_are_an_error():
    vm, nodes, tets = _volume_mesh()
    sb = vm.surface_mesh["boundaries"]
    partial = BoundaryMap(groups={k: v for k, v in sb.groups.items() if k != "z_max"},
                          bc_types={k: v for k, v in sb.bc_types.items() if k != "z_max"})
    with pytest.raises(ValueError, match="匹配不到面网格的任何边界组"):
        map_boundaries_by_geometry(vm, _external(vm, nodes, partial))
    vm, nodes, tets = _volume_mesh()
    with pytest.raises(ValueError, match="没有任何边界组"):
        map_boundaries_by_geometry(vm, _external(vm, nodes, BoundaryMap(groups={}, bc_types={})))
