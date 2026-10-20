"""调试导出（`--bl-only` / `--core-only`）不在库函数里结束进程、不在库函数里写文件（I-138 / I-168）。

此前 `_export_partial_mesh_and_exit` 写文件后 `sys.exit(0)`（失败 `sys.exit(1)`），作为 API 调用时调用方进程被
一起结束；写出单位写死为毫米。现在局部网格随 `PartialMeshReady` 带回，由 CLI 按输出单位（与输入面网格同一单位）
写出。
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.grid.mesh_gen.background import mesh_background
from autoflowcfd.grid.mesh_gen.background.mesh_background_merge_utils import (
    PartialMeshReady, _stop_with_partial_mesh,
)
from tests.unit.test_generated_boundary_mapping import _cube_with_surface


def _partial_cube():
    nodes, tets, sn, sf, sb = _cube_with_surface()
    with pytest.raises(PartialMeshReady) as done:
        _stop_with_partial_mesh(nodes, np.empty((0, 6), np.int64), tets, "core-only", sn, sf, sb)
    return done.value, nodes, tets


def test_partial_mesh_is_returned_with_the_signal():
    done, nodes, tets = _partial_cube()
    assert done.label == "core-only"
    assert done.mesh.cells.count == len(tets) and done.mesh.metadata.file_format == "partial"
    np.testing.assert_array_equal(done.mesh.nodes.x, nodes[:, 0])


def test_stalled_bl_corner_is_merged_before_export():
    """BL 挤出停长的角点：上层节点与底点坐标相同、编号不同，侧面三角形面积恰为零。组装前与完整管线一样合并
    重合点（cube_demo --bl-only 实测 5025 个这样的面，此前在面积检查处报错）。"""
    _, _, sn, sf, sb = _cube_with_surface()
    nodes = np.array([[0.2, 0.2, 0.0], [0.6, 0.2, 0.0], [0.2, 0.6, 0.0],
                      [0.2, 0.2, 0.1], [0.6, 0.2, 0.0], [0.2, 0.6, 0.1]])   # w1 停在 v1 处
    with pytest.raises(PartialMeshReady) as done:
        _stop_with_partial_mesh(nodes, np.array([[0, 1, 2, 3, 4, 5]]), np.empty((0, 4), np.int64),
                                "BL-only", sn, sf, sb)
    assert done.value.mesh.prism_cells.count == 1 and done.value.mesh.node_count == 5


def test_generate_hybrid_mesh_passes_the_signal_through(monkeypatch):
    done, _, _ = _partial_cube()

    def stop(*args, **kwargs):
        raise done

    monkeypatch.setattr(mesh_background, "_build_merged_mesh", stop)
    _, _, sn, sf, sb = _cube_with_surface()
    with pytest.raises(PartialMeshReady):
        mesh_background.generate_hybrid_mesh(sn, sf, {}, growth_rate=1.2, min_cell_size=1e-3, surface_boundaries=sb,
                                             export_bl_only=True)


@pytest.mark.parametrize("units, scale", [("m", 1.0), ("mm", 1000.0)])
def test_cli_writes_the_partial_mesh_in_the_input_units(tmp_path, monkeypatch, units, scale):
    """--units m 的面网格：局部网格按米写出（此前恒为毫米，米制往返差 1000 倍）；退出码 0、JSON 状态 partial。"""
    from autoflowcfd.cli.main import cli
    from autoflowcfd.grid.nas_io import parser_core
    from autoflowcfd.grid.nas_io.nas_parser_volume import parse_volume_mesh_nas

    done, nodes, _ = _partial_cube()
    surface = tmp_path / "s.nas"
    surface.write_text("GRID,1,,0.0,0.0,0.0\nGRID,2,,1.0,0.0,0.0\nGRID,3,,0.0,1.0,0.0\nCTRIA3,1,1,1,2,3\n")

    def stop(self, surface_grid, volume_mesh_params=None):
        raise done

    monkeypatch.setattr(parser_core.NASParser, "generate_volume_mesh_from_surface", stop)
    out = str(tmp_path / "partial.nas")
    result = CliRunner().invoke(cli, ["grid", "generate-volume", str(surface), "-o", out, "--core-only",
                                      "--units", units, "--json-output", "--bc-type", "UNCLASSIFIED=WALL"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload["status"] == "partial" and payload["stage"] == "core-only"
    back = parse_volume_mesh_nas(payload["output_file"], units="m")
    np.testing.assert_allclose(np.sort(back.nodes.x), np.sort(nodes[:, 0] * scale), atol=1e-9 * scale)
