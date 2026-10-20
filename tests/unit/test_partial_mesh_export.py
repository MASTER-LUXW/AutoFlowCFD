"""调试导出（`--bl-only` / `--core-only`）不再在库函数里结束进程（I-138）。

此前 `_export_partial_mesh_and_exit` 导出后 `sys.exit(0)`（失败 `sys.exit(1)`），`generate_hybrid_mesh` 另有一处
`sys.exit(0)` 兜底：作为 API 调用时调用方进程被一起结束、绕过上层清理。现在导出完成抛 `PartialMeshExported`，
由 CLI 决定收尾；导出失败原异常照常向上抛。
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.grid.mesh_gen.background import mesh_background
from autoflowcfd.grid.mesh_gen.background.mesh_background_merge_utils import (
    PartialMeshExported, _export_partial_mesh,
)
from tests.unit.test_generated_boundary_mapping import _cube_with_surface


def test_export_writes_the_mesh_and_raises_instead_of_exiting(tmp_path):
    nodes, tets, sn, sf, sb = _cube_with_surface()
    out = tmp_path / "core.nas"
    with pytest.raises(PartialMeshExported) as done:
        _export_partial_mesh(nodes, np.empty((0, 6), np.int64), tets, str(out), "core-only", sn, sf, sb)
    assert done.value.path == str(out) and done.value.label == "core-only"
    assert out.read_text().count("CTETRA") == len(tets)


def test_stalled_bl_corner_is_merged_before_export(tmp_path):
    """BL 挤出停长的角点：上层节点与底点坐标相同、编号不同，侧面三角形面积恰为零。导出前与完整管线一样合并
    重合点（cube_demo --bl-only 实测 5025 个这样的面，此前在面积检查处报错）。"""
    _, _, sn, sf, sb = _cube_with_surface()
    nodes = np.array([[0.2, 0.2, 0.0], [0.6, 0.2, 0.0], [0.2, 0.6, 0.0],
                      [0.2, 0.2, 0.1], [0.6, 0.2, 0.0], [0.2, 0.6, 0.1]])   # w1 停在 v1 处
    out = tmp_path / "bl.nas"
    with pytest.raises(PartialMeshExported):
        _export_partial_mesh(nodes, np.array([[0, 1, 2, 3, 4, 5]]), np.empty((0, 4), np.int64), str(out),
                             "BL-only", sn, sf, sb)
    assert out.read_text().count("CPENTA") == 1


def test_export_failure_propagates_the_original_error(tmp_path):
    nodes, tets, sn, sf, sb = _cube_with_surface()
    with pytest.raises(RuntimeError, match="NAS export failed"):
        _export_partial_mesh(nodes, np.empty((0, 6), np.int64), tets, str(tmp_path / "missing" / "x.nas"),
                             "core-only", sn, sf, sb)


def test_generate_hybrid_mesh_passes_the_signal_through(monkeypatch):
    def stop(*args, **kwargs):
        raise PartialMeshExported("bl.nas", "BL-only")

    monkeypatch.setattr(mesh_background, "_build_merged_mesh", stop)
    nodes, tets, sn, sf, sb = _cube_with_surface()
    with pytest.raises(PartialMeshExported):
        mesh_background.generate_hybrid_mesh(sn, sf, {}, growth_rate=1.2, min_cell_size=1e-3, surface_boundaries=sb,
                                             export_bl_only=True, export_bl_only_path="bl.nas")


def test_cli_reports_the_partial_export_with_exit_code_zero(tmp_path, monkeypatch):
    from autoflowcfd.cli.main import cli
    from autoflowcfd.grid.nas_io import parser_core

    surface = tmp_path / "s.nas"
    surface.write_text("GRID,1,,0.0,0.0,0.0\nGRID,2,,1.0,0.0,0.0\nGRID,3,,0.0,1.0,0.0\nCTRIA3,1,1,1,2,3\n")

    def stop(self, surface_grid, volume_mesh_params=None):
        raise PartialMeshExported(volume_mesh_params["output"], "BL-only")

    monkeypatch.setattr(parser_core.NASParser, "generate_volume_mesh_from_surface", stop)
    out = str(tmp_path / "bl.nas")
    result = CliRunner().invoke(cli, ["grid", "generate-volume", str(surface), "-o", out, "--bl-only",
                                      "--json-output", "--bc-type", "UNCLASSIFIED=WALL"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{"):])
    assert payload == {"command": "grid.generate-volume", "status": "partial", "stage": "BL-only",
                       "output_file": out}
