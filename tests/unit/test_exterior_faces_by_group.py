"""边界面逐面分组（2026-10-09，`mesh_boundary.exterior_faces_by_group`）。

此前导出端按单元级分组反推边界面：角上单元同时拥有两个组的外部面时出现在两个组里，它的两个面被同时写进两个
组——cube_demo 导出的体网格 39488 个外部面写出了 39804 个 CTRIA3。PSHELL 名称还写成边界条件类型的小写，
两个 WALL 组同名、原始组名丢失。
"""

import re

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import exterior_faces_by_group, map_generated_boundaries
from autoflowcfd.grid.nas_io.nas_export import export_volume_mesh_to_nas
from autoflowcfd.grid.schema.grid_cells import TetrahedralCells
from autoflowcfd.grid.schema.grid_data import VolumeMeshData
from autoflowcfd.grid.schema.grid_metadata import GridMetadata
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from tests.unit.test_cavity_weld_conformity import _exterior
from tests.unit.test_generated_boundary_mapping import _cube_with_surface


def _volume_mesh(with_surface=True):
    nodes, tets, sn, sf, sb = _cube_with_surface()
    groups = map_generated_boundaries(nodes, np.empty((0, 6), np.int64), tets, sn, sf, sb)
    node_arr = NodeArray.from_array(nodes)
    cells = TetrahedralCells(connectivity=tets.astype(np.int32),
                             volumes=TetrahedralCells.compute_volumes(node_arr, tets.astype(np.int32)))
    meta = GridMetadata(node_count=len(nodes), cell_count=len(tets),
                        boundary_groups=list(groups.groups), file_format="hybrid")
    surface = {"nodes": sn, "faces": sf, "boundaries": sb} if with_surface else None
    return VolumeMeshData(nodes=node_arr, cells=cells, boundaries=groups, metadata=meta,
                          prism_cells=None, surface_mesh=surface), nodes, tets


def test_each_exterior_face_lands_in_exactly_one_group():
    vm, nodes, tets = _volume_mesh()
    corner_cells = sum(len(c) for c in vm.boundaries.groups.values()) - len(
        np.unique(np.concatenate(list(vm.boundaries.groups.values()))))
    assert corner_cells > 0                          # 确有同时属于多个组的单元
    by_group = exterior_faces_by_group(vm)
    all_faces = np.vstack(list(by_group.values()))
    assert len(all_faces) == len(_exterior(tets)[0])  # 不重不漏
    assert len({tuple(sorted(f)) for f in all_faces.tolist()}) == len(all_faces)
    for name, faces in by_group.items():
        axis, value = "xyz".index(name[0]), (0.0 if name.endswith("min") else 1.0)
        assert np.allclose(nodes[faces][..., axis], value)   # 每个面都在自己那个立方体面上


def test_ambiguous_cell_groups_without_surface_data_are_rejected():
    vm, _, _ = _volume_mesh(with_surface=False)
    with pytest.raises(ValueError, match="同时属于多个边界组"):
        exterior_faces_by_group(vm)


def test_exported_nas_has_one_ctria3_per_exterior_face_and_group_names(tmp_path):
    vm, _, tets = _volume_mesh()
    out = export_volume_mesh_to_nas(vm, str(tmp_path / "cube.nas"))
    text = open(out).read()
    assert len(re.findall(r"^CTRIA3", text, flags=re.M)) == len(_exterior(tets)[0])
    names = set(re.findall(r"^\$ANSA_NAME_COMMENT;\d+;PSHELL;([^;]+);", text, flags=re.M))
    assert names == set(vm.boundaries.groups)        # 组名，不是边界条件类型
