"""参考面积与气动力积分同一组物体面、resume --output、status 来自命令定义、帮助只写现行语义（I-174）。"""

import re
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from autoflowcfd.cli.solve.aero_coefficients import _compute_reference_area_auto
from autoflowcfd.grid.schema.grid_boundaries import BoundaryMap


def _square(x, size, offset):
    """x = const 平面上边长 `size` 的正方形（两个三角形，外法向 -x），节点编号从 offset 起。"""
    p = np.array([[x, 0, 0], [x, 0, size], [x, size, size], [x, size, 0]], float)
    return p, np.array([[0, 1, 2], [0, 2, 3]]) + offset


def _volume_data(tunnel_type):
    body_p, body_f = _square(0.0, 1.0, 0)
    tunnel_p, tunnel_f = _square(-5.0, 10.0, 4)
    groups = {"body": np.array([0, 1]), "tunnel_wall": np.array([2, 3])}
    bm = BoundaryMap(groups=groups, bc_types={"body": "WALL", "tunnel_wall": tunnel_type})
    return SimpleNamespace(surface_mesh={"nodes": np.vstack([body_p, tunnel_p]), "faces": np.vstack([body_f, tunnel_f]),
                                         "boundaries": bm})


def test_reference_area_uses_the_same_body_groups_as_the_force_integration():
    """'tunnel_wall' 名字含 WALL 而类型是 SLIP_WALL：不是物体面（此前按名字子串算进参考面积，得 101 而非 1）。"""
    assert _compute_reference_area_auto(_volume_data("SLIP_WALL"), direction=[1, 0, 0]) == pytest.approx(1.0)
    assert _compute_reference_area_auto(_volume_data("WALL"), direction=[1, 0, 0]) == pytest.approx(101.0)


def test_override_to_slip_removes_a_group_from_the_reference_area():
    vd = _volume_data("WALL")
    overrides = {"tunnel_wall": {"type": "WALL", "physical_no_slip": False}}
    assert _compute_reference_area_auto(vd, direction=[1, 0, 0], bc_overrides=overrides) == pytest.approx(1.0)


def test_zero_frontal_projection_is_an_error_not_a_silent_fallback():
    """来流平行于物体面：迎风投影为 0（此前静默改用 |投影| 总和 / 2）。"""
    with pytest.raises(ValueError, match="迎风投影面积"):
        _compute_reference_area_auto(_volume_data("SLIP_WALL"), direction=[0, 1, 0])


def test_no_body_group_is_an_error():
    vd = _volume_data("SLIP_WALL")
    vd.surface_mesh["boundaries"].bc_types["body"] = "SLIP_WALL"
    with pytest.raises(ValueError, match="物理无滑移固壁"):
        _compute_reference_area_auto(vd, direction=[1, 0, 0])


def test_resume_has_an_output_option():
    assert "--output" in CliRunner().invoke(cli, ["solve", "resume", "--help"]).output


def test_status_is_generated_from_the_commands():
    out = CliRunner().invoke(cli, ["solve", "status"]).output
    assert "P0 ~ P3" in out and "newton-krylov" in out and "Dual-Time" not in out


@pytest.mark.parametrize("command", ["steady", "transient", "resume"])
def test_help_has_no_development_history(command):
    text = CliRunner().invoke(cli, ["solve", command, "--help"]).output
    assert not re.search(r"20\d\d-\d\d-\d\d|下调|原文案|真实缺口修复", text)
