"""求解命令直接读 .nas 体网格时不修改用户网格；畸变单元的提示指向真实存在的命令（I-171）。

此前求解路径用 import_external_volume_mesh 的默认 repair=True：质量门不过就静默移动用户网格的节点（Stage A 光顺）
再求解；MeshDistortionError 的提示让用户去用不存在的 `autoflowcfd grid check` / `repair`，并称"构造高阶曲边几何"
（几何是直边单元）。
"""

import pytest

from autoflowcfd.cli.solve import mesh_loader
from autoflowcfd.grid.curved_mapping.curved_mapping import MeshDistortionError
from autoflowcfd.grid.validation import solve_gate


def test_solve_path_imports_without_repair(monkeypatch, tmp_path):
    seen = {}

    def fake_import(volume_path, surface_path, **kwargs):
        seen.update(kwargs)
        raise ValueError("stop here")

    import autoflowcfd.grid.mesh_gen.utils.mesh_external_import as ext
    monkeypatch.setattr(ext, "import_external_volume_mesh", fake_import)
    volume = tmp_path / "v.nas"
    volume.write_text("CTETRA\n")
    with pytest.raises(Exception):
        mesh_loader.load_mesh_for_solver(str(volume), 1, surface_mesh=str(volume))
    assert seen.get("repair") is False


def test_distortion_message_points_at_real_commands(monkeypatch):
    class Broken:
        def __init__(self, order):
            pass

        def load_from_volume_mesh(self, volume_data):
            raise MeshDistortionError("cell 7 det<0")

    monkeypatch.setattr(solve_gate, "HighOrderMesh", Broken)
    with pytest.raises(solve_gate.MeshQualityGateError) as err:
        solve_gate.high_order_mesh_for_solver(object(), 1, skip_quality_check=True)
    text = str(err.value)
    assert "grid generate-volume" in text and "grid import-volume" in text
    assert "grid check" not in text and "曲边" not in text
