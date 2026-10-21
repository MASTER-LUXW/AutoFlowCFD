"""Python API 的网格入口与 CLI 同参（I-192）：load_grid 的坐标单位与边界类型覆盖；generate_volume_mesh 的第一层高度
直接给出或由目标 y+ 推出（与 CLI 同一个函数）。此前 load_grid 没有 units，.nas 一律按毫米读入。"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd import AutoFlowCFDAPI
from autoflowcfd.grid.mesh_gen.extrusion.first_layer import first_layer_height_for_yplus

_CUBE = Path(__file__).resolve().parents[2] / "examples" / "cube_demo" / "cube_demo.nas"


def _extent(grid):
    xyz = np.column_stack([grid.nodes.x, grid.nodes.y, grid.nodes.z])
    return xyz.max(axis=0) - xyz.min(axis=0)


def test_load_grid_honours_the_units():
    """cube_demo.nas 的坐标是毫米（8000 x 3000 x 3000）：缺省与 units="mm" 换算为 8 x 3 x 3 m；units="m" 原样。"""
    api = AutoFlowCFDAPI()
    millimetre = api.load_grid(_CUBE, validate=False)
    np.testing.assert_allclose(_extent(millimetre), [8.0, 3.0, 3.0], rtol=1e-12)
    as_metres = api.load_grid(_CUBE, units="m", validate=False)                       # 此前 API 只能按毫米读
    np.testing.assert_allclose(_extent(as_metres), [8000.0, 3000.0, 3000.0], rtol=1e-12)


def test_load_grid_bc_type_override():
    grid = AutoFlowCFDAPI().load_grid(_CUBE, bc_types={"tunnel": "FARFIELD"}, validate=False)
    assert grid.boundaries.bc_types["tunnel"] == "FARFIELD" and grid.boundaries.bc_types["body"] == "WALL"


def test_generate_volume_mesh_resolves_the_first_layer_like_the_cli():
    api = AutoFlowCFDAPI()
    with patch("autoflowcfd.grid.nas_io.parser_volume.generate_volume_mesh_from_surface") as gen:
        api.generate_volume_mesh("surface", target_yplus=1.0, ref_velocity=40.0, ref_length=1.0, bl_layers=6)
    params = gen.call_args.args[1]
    assert params["bl_layers"] == 6
    assert params["first_layer_height"] == pytest.approx(first_layer_height_for_yplus(1.0, 40.0, 1.0, 1.225, 1.8e-5))
    with pytest.raises(ValueError, match="只能给一个"):
        api.generate_volume_mesh("surface", first_layer_height=1e-5, target_yplus=1.0, ref_velocity=40.0,
                                 ref_length=1.0)
    with pytest.raises(ValueError, match="参考速度"):
        api.generate_volume_mesh("surface", target_yplus=1.0)
