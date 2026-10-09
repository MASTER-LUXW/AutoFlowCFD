"""Stage A 网格质量修复（mesh_gen/mesh_repair.py）的单元测试。"""

import numpy as np

from autoflowcfd.grid.mesh_gen.repair.mesh_repair import smooth_bad_cells, compute_movable_node_mask
from autoflowcfd.grid.mesh_gen.tetgen.mesh_prism_to_tet import orient_tetrahedra
from autoflowcfd.grid.validation.quality_validator import MeshQualityValidator
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from autoflowcfd.grid.mesh_gen.extraction.face_extractor import FaceExtractor


def _bipyramid():
    """方底双棱锥绕中心 C 拆成 8 个四面体——C 是唯一不属于任何边界面的节点
    （其余节点都在外壳上）。一个干净、可手算验证的夹具（没有 Delaunay 近退化），
    用来测 Stage A 的节点资格与光顺。

    返回 (nodes, cells, iC)，cells 朝向正确（体积全部为正）。
    """
    B0, B1, B2, B3 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
    T, D, C = [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]
    nodes = np.array([B0, B1, B2, B3, T, D, C])
    iB0, iB1, iB2, iB3, iT, iD, iC = range(7)

    base_edges = [(iB0, iB1), (iB1, iB2), (iB2, iB3), (iB3, iB0)]
    cells = []
    for (a, b) in base_edges:
        cells.append([iC, a, b, iT])
        cells.append([iC, a, b, iD])
    cells = np.array(cells, dtype=np.int64)
    cells = orient_tetrahedra(nodes, cells).astype(np.int32)
    return nodes, cells, iC


def _bipyramid_core_half():
    """与 _bipyramid() 同一几何，只取"核心"一半（接触 D 极的 4 个四面体）：另一半（接触 T 极）在真实网格里
    是 BL 棱柱，不在被平滑的四面体集合里。赤道面 (iC, Bi, Bi+1) 就是 BL/core 接口，iC 是接口上唯一不在
    物理边界上的节点。"""
    B0, B1, B2, B3 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]
    T, D, C = [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]
    nodes = np.array([B0, B1, B2, B3, T, D, C])
    iB0, iB1, iB2, iB3, iT, iD, iC = range(7)
    base_edges = [(iB0, iB1), (iB1, iB2), (iB2, iB3), (iB3, iB0)]
    cells = np.array([[iC, a, b, iD] for (a, b) in base_edges], dtype=np.int64)
    return nodes, orient_tetrahedra(nodes, cells).astype(np.int32), iC


class TestMovableNodeMask:
    def test_only_interior_node_is_movable(self):
        nodes, cells, iC = _bipyramid()
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells, node_arr)

        movable = compute_movable_node_mask(len(nodes), faces)

        assert np.where(movable)[0].tolist() == [iC]

    def test_bl_core_interface_node_is_pinned(self):
        """接口节点不能被平滑移动（cube_demo 实测：移动它会让核心四面体仍按旧位置构建、与 BL 单元空间重叠）。
        BL 是棱柱、不在四面体集合里，接口面在四面体子网格上是边界面，由边界规则固定。"""
        nodes, cells, iC = _bipyramid_core_half()
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells, node_arr)

        assert not compute_movable_node_mask(len(nodes), faces)[iC]


class TestSmoothBadCells:
    def test_recovers_perturbed_interior_node(self):
        """对唯一的内部节点做一个大但不破坏体积的扰动，应被光顺回几何上正确的
        位置，之后网格应通过 MeshQualityValidator。
        """
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.55, 0.35, 0.0])

        report_before = validator.validate(perturbed, cells, cell_type="tetrahedron")
        assert report_before.passed is False

        new_nodes, bad_mask, actions = smooth_bad_cells(perturbed, cells, validator, max_passes=5)

        report_after = validator.validate(new_nodes, cells, cell_type="tetrahedron")
        assert report_after.passed is True
        assert np.allclose(new_nodes[iC], [0.0, 0.0, 0.0], atol=1e-9)
        assert any("moved" in a for a in actions)

    def test_never_moves_boundary_nodes(self):
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.55, 0.35, 0.0])

        new_nodes, _, _ = smooth_bad_cells(perturbed, cells, validator, max_passes=5)

        non_center = [i for i in range(len(nodes)) if i != iC]
        assert np.allclose(new_nodes[non_center], perturbed[non_center])

    def test_never_moves_bl_core_interface_node(self):
        """经平滑入口的端到端版本：只有接口节点能"修好"的坏单元保持坏，而不是移动接口节点。"""
        nodes, cells, iC = _bipyramid_core_half()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.3, 0.2, -0.1])

        new_nodes, _bad_mask, _actions = smooth_bad_cells(perturbed, cells, validator, max_passes=5)

        assert np.array_equal(new_nodes, perturbed), "iC must not have moved"

    def test_never_introduces_negative_volume(self):
        """即使扰动大到进入时已有单元无效，Stage A 也绝不能让原本有效的单元
        体积变负。
        """
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        perturbed = nodes.copy()
        perturbed[iC] += np.array([0.9, 0.9, 0.0])  # aggressive, may itself invert a cell

        current_volumes = validator._compute_tetrahedron_volumes(perturbed, cells)
        already_bad = current_volumes <= 0

        new_nodes, _, _ = smooth_bad_cells(perturbed, cells, validator, max_passes=5)

        final_volumes = validator._compute_tetrahedron_volumes(new_nodes, cells)
        newly_negative = (final_volumes <= 0) & ~already_bad
        assert not np.any(newly_negative)

    def test_no_op_on_already_good_mesh(self):
        nodes, cells, iC = _bipyramid()
        validator = MeshQualityValidator()

        new_nodes, bad_mask, actions = smooth_bad_cells(nodes, cells, validator, max_passes=5)

        assert np.allclose(new_nodes, nodes)
        assert not np.any(bad_mask)
        assert any("already within thresholds" in a for a in actions)
