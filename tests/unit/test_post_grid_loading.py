"""后处理按求解时的体网格加载、边界面片逐面定组（2026-10-09）。

cube_demo 端到端里 `post export-vtk --grid <体网格.nas>` 失败：`_load_grid_data` 把任何 .nas 都当成面网格、现场
重新生成体网格（传入体网格时 tetgen 直接失败；传入面网格时生成的网格单元数与顺序都与解对不上）。边界面片
导出按 owner 单元定组——同时贴着两个边界的角点单元，它的面全部标成后遍历到的那个组；类型按组名另猜。
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.cli.post.helpers import _load_grid_data, _locate_grid_file
from autoflowcfd.postprocess.vtk_export_fields import boundary_zone_ids
from tests.unit.test_grid_convert import _surface_nas, _volume_nas


def test_volume_nas_is_loaded_as_is(tmp_path):
    src, n_tets = _volume_nas(tmp_path)
    grid = _load_grid_data(Path(src))
    assert grid.cell_count == n_tets
    assert grid.surface_mesh is not None            # 边界三角形一并读入，供边界面片定组


def test_surface_mesh_is_rejected_instead_of_regenerating_a_volume_mesh(tmp_path):
    with pytest.raises(ValueError, match="面网格"):
        _load_grid_data(_surface_nas(tmp_path / "surface.nas"))


def test_grid_file_resolution_order(tmp_path):
    src, _ = _volume_nas(tmp_path)
    case = tmp_path / "case"
    case.mkdir()
    # 没有 --grid、算例目录里没有 volume_mesh.pkl：取 checkpoint 记录的求解输入文件
    assert _locate_grid_file(case, None, {"input_file": src}) == Path(src)
    # 显式 --grid 优先
    assert _locate_grid_file(case, "explicit.pkl", {"input_file": src}) == Path("explicit.pkl")
    # 算例目录里的 volume_mesh.pkl 先于 checkpoint 记录
    (case / "volume_mesh.pkl").write_bytes(b"")
    assert _locate_grid_file(case, None, {"input_file": src}) == case / "volume_mesh.pkl"
    (case / "volume_mesh.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="--grid"):
        _locate_grid_file(case, None, {"input_file": str(tmp_path / "missing.nas")})


def test_boundary_patches_are_tagged_per_face_with_recorded_types(tmp_path):
    """立方体每个角上的单元同时贴着三个边界组：逐面定组后每个组恰好覆盖自己那个面（面积 1），类型取网格记录的
    `bc_types`。"""
    src, _ = _volume_nas(tmp_path)
    grid = _load_grid_data(Path(src))
    faces = grid.ensure_faces_exist()
    tri = faces.node_connectivity[faces.get_boundary_face_indices()]
    boundary_id, type_id, id_legend, type_legend = boundary_zone_ids(SimpleNamespace(grid_data=grid), tri)

    names = [entry.split("=", 1)[1] for entry in id_legend]
    assert sorted(names) == ["x_max", "x_min", "y_max", "y_min", "z_max", "z_min"]
    xyz = np.column_stack([grid.nodes.x, grid.nodes.y, grid.nodes.z])
    corners = xyz[tri]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    centre = corners.mean(axis=1)
    for gi, name in enumerate(names):
        mine = boundary_id == gi
        assert area[mine].sum() == pytest.approx(1.0, rel=1e-12), name
        axis, side = "xyz".index(name[0]), (0.0 if name.endswith("min") else 1.0)
        np.testing.assert_allclose(centre[mine, axis], side, atol=1e-12, err_msg=name)

    recorded = grid.surface_mesh["boundaries"].bc_types
    types = [entry.split("=", 1)[1] for entry in type_legend]
    for gi, name in enumerate(names):
        assert {types[t] for t in type_id[boundary_id == gi]} == {str(recorded.get(name, "UNKNOWN"))}


def test_exterior_face_outside_every_group_is_an_error(tmp_path):
    src, _ = _volume_nas(tmp_path)
    grid = _load_grid_data(Path(src))
    with pytest.raises(ValueError, match="不属于任何边界组"):
        boundary_zone_ids(SimpleNamespace(grid_data=grid), np.array([[0, 1, 10**6]]))
