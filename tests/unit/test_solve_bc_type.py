"""求解命令接受 --bc-type：边界条件类型在导入 .nas 体网格（或改写 .pkl）时应用，写进输出目录的 volume_mesh.pkl（I-197）。

.nas 不记录边界条件类型，读回时按组名重新识别：此前 `grid generate-volume --bc-type` 指定的类型在求解读 .nas 时
丢失，组名有歧义的算例必然报错，而求解命令又没有 --bc-type。
"""

import pickle

import click
import pytest
from click.testing import CliRunner

from autoflowcfd.cli.solve.mesh_loader import VOLUME_MESH_CACHE_NAME, prepare_solver_input
from tests.unit.test_solver_input_units import _write_volume_and_surface


def _ambiguous(tmp_path):
    """唯一的组名 'tunnel_wall' 同时命中 SLIP_WALL 与 WALL：按名字识别不了类型。"""
    volume, surface, _ = _write_volume_and_surface(tmp_path, 2.0)
    surface.write_text(surface.read_text().replace(";PSHELL;wall;", ";PSHELL;tunnel_wall;"))
    return volume, surface


def _cached(out):
    with open(out / VOLUME_MESH_CACHE_NAME, "rb") as f:
        return pickle.load(f)


def test_ambiguous_group_needs_bc_type_and_the_type_reaches_the_cache(tmp_path):
    volume, surface = _ambiguous(tmp_path)
    with pytest.raises(click.ClickException, match="--bc-type"):
        prepare_solver_input(str(volume), str(surface), "m", str(tmp_path / "a"))
    out = tmp_path / "b"
    prepare_solver_input(str(volume), str(surface), "m", str(out), {"tunnel_wall": "SLIP_WALL"})
    assert _cached(out).boundaries.bc_types == {"tunnel_wall": "SLIP_WALL"}


def test_bc_type_on_a_pkl_input_writes_an_overridden_copy(tmp_path):
    volume, surface, _ = _write_volume_and_surface(tmp_path, 2.0)
    first = tmp_path / "first"
    prepare_solver_input(str(volume), str(surface), "m", str(first))
    assert _cached(first).boundaries.bc_types == {"wall": "WALL"}
    second = tmp_path / "second"
    new_input, _ = prepare_solver_input(str(first / VOLUME_MESH_CACHE_NAME), None, "m", str(second),
                                        {"wall": "SLIP_WALL"})
    assert new_input == str(second / VOLUME_MESH_CACHE_NAME)
    assert _cached(second).boundaries.bc_types == {"wall": "SLIP_WALL"}
    assert _cached(first).boundaries.bc_types == {"wall": "WALL"}             # 用户的 .pkl 不改
    with pytest.raises(click.ClickException, match="nosuch"):
        prepare_solver_input(str(first / VOLUME_MESH_CACHE_NAME), None, "m", str(tmp_path / "c"),
                             {"nosuch": "WALL"})


@pytest.mark.parametrize("command", [["solve", "steady"], ["solve", "transient"], ["utils", "benchmark"]])
def test_commands_offer_bc_type(command):
    from autoflowcfd.cli.main import cli

    assert "--bc-type" in CliRunner().invoke(cli, command + ["--help"]).output
