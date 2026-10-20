"""求解命令直接读 .nas 体网格时按 --units 导入，并转存成输出目录下的米制 volume_mesh.pkl（I-171）。

此前求解命令没有 --units，.nas 体网格一律按毫米读入（米制网格被缩小 1000 倍）；续算 / 后处理按 checkpoint 记录的
.nas 重新加载时同样只能按毫米。现在入口处转存成 .pkl，checkpoint、续算与后处理都用这份米制网格。
"""

import pickle

import numpy as np
import pytest

from autoflowcfd.cli.solve.mesh_loader import VOLUME_MESH_CACHE_NAME, prepare_solver_input
from tests.unit.test_cavity_weld_conformity import _cube_tets, _exterior


def _write_volume_and_surface(tmp_path, scale):
    nodes, tets, _ = _cube_tets(2)
    nodes = nodes * scale
    faces = np.array(sorted(_exterior(tets)[0]))
    grid = "".join(f"GRID,{i + 1},,{x:.6f},{y:.6f},{z:.6f}\n" for i, (x, y, z) in enumerate(nodes))
    volume = tmp_path / "v.nas"
    volume.write_text(grid + "".join(f"CTETRA,{k + 1},1,{a + 1},{b + 1},{c + 1},{d + 1}\n"
                                     for k, (a, b, c, d) in enumerate(tets)))
    surface = tmp_path / "s.nas"
    surface.write_text("BEGIN BULK\n$ANSA_NAME_COMMENT;1;PSHELL;wall;;NO;NO;NO;NO;\nPSHELL,1,1,0.001\n" + grid
                       + "".join(f"CTRIA3,{k + 1},1,{a + 1},{b + 1},{c + 1}\n" for k, (a, b, c) in enumerate(faces))
                       + "ENDDATA\n")
    return volume, surface, nodes


@pytest.mark.parametrize("units, to_m", [("m", 1.0), ("mm", 1e-3)])
def test_nas_input_is_imported_with_units_and_cached_in_meters(tmp_path, units, to_m):
    volume, surface, nodes = _write_volume_and_surface(tmp_path, 2.0)          # 2 m（或 2 mm）的立方体
    out = tmp_path / "out"
    new_input, new_surface = prepare_solver_input(str(volume), str(surface), units, str(out))
    assert new_input == str(out / VOLUME_MESH_CACHE_NAME) and new_surface is None
    with open(new_input, "rb") as f:
        cached = pickle.load(f)
    assert cached.boundaries.groups and cached.surface_mesh is not None
    np.testing.assert_allclose(cached.nodes.x.max(), nodes[:, 0].max() * to_m)


def test_pkl_input_is_passed_through(tmp_path):
    assert prepare_solver_input("mesh.pkl", None, "m", str(tmp_path)) == ("mesh.pkl", None)
