"""Python API 的网格入口与 CLI 同参（I-192）：load_grid 的坐标单位与边界类型覆盖；generate_volume_mesh 的第一层高度
直接给出或由目标 y+ 推出（与 CLI 同一个函数）。此前 load_grid 没有 units，.nas 一律按毫米读入。"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd import AutoFlowCFDAPI
from autoflowcfd.grid.structures import (
    BoundaryMap, GridMetadata, NodeArray, PrismCells, TetrahedralCells, VolumeMeshData,
)
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


def test_get_grid_info_reports_the_boundary_groups():
    """此前对 BoundaryMap 取 len()（没有 __len__）直接 TypeError，组名读的是 GridData 上不存在的属性。"""
    api = AutoFlowCFDAPI()
    info = api.get_grid_info(api.load_grid(_CUBE, validate=False))
    assert info["boundary_count"] == 4 and sorted(info["boundary_groups"]) == ["body", "inlet", "outlet", "tunnel"]
    assert info["node_count"] > 0 and info["cell_count"] > 0


def test_get_volume_mesh_info_counts_tets_and_prisms():
    """此前读 VolumeMeshData 上不存在的 cell_type，调用即 AttributeError。"""
    vm = VolumeMeshData(
        nodes=NodeArray(x=np.arange(7.0), y=np.zeros(7), z=np.zeros(7)),
        cells=TetrahedralCells(connectivity=np.array([[0, 1, 2, 6]], dtype=np.int32), volumes=np.array([1.0])),
        boundaries=BoundaryMap(groups={}, bc_types={}),
        metadata=GridMetadata(node_count=7, cell_count=2, boundary_groups=[], file_format="test"),
        prism_cells=PrismCells(connectivity=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32), volumes=np.array([1.0])),
    )
    assert AutoFlowCFDAPI().get_volume_mesh_info(vm) == {"node_count": 7, "cell_count": 2, "tet_count": 1,
                                                         "prism_count": 1}
