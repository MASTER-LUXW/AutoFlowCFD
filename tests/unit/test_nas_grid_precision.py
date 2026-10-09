"""NAS 节点卡片的精度与大字段格式（2026-10-08）。

导出端此前用 8 字符小字段写坐标：以毫米写出时，量级上千的坐标只剩 3 位小数。cube_demo 实测同一份生成网格的
相邻单元体积比在内存里是 33.20，导出、再导入后变成 222.69。现在导出端写大字段 `GRID*`（10 位有效数字），
解析端支持 `GRID*` + `*` 续行（固定格式与逗号分隔）。
"""

import numpy as np
import pytest

from autoflowcfd.grid.nas_io.nas_export import _write_nodes
from autoflowcfd.grid.nas_io.nas_parser_nodes import parse_nodes_from_nas
from autoflowcfd.grid.schema.grid_nodes import NodeArray


def _parse(path):
    nodes, id_map = parse_nodes_from_nas(str(path))
    return np.column_stack([nodes.x, nodes.y, nodes.z]), id_map


def test_export_import_round_trip_keeps_ten_significant_digits(tmp_path):
    rng = np.random.default_rng(0)
    # 米制坐标，量级与汽车外流域相同（几米），再叠加边界层量级的微小偏移
    xyz = rng.uniform(-6.5, 6.5, size=(500, 3)) + rng.uniform(0, 1e-5, size=(500, 3))
    nodes = NodeArray(x=xyz[:, 0].copy(), y=xyz[:, 1].copy(), z=xyz[:, 2].copy())
    path = tmp_path / "nodes.nas"
    with open(path, "w") as f:
        _write_nodes(f, nodes, scale_factor=1000.0)
    got, id_map = _parse(path)
    assert len(id_map) == 500
    np.testing.assert_allclose(got / 1000.0, xyz, rtol=1e-9, atol=1e-12)
    # 小字段只能做到约 1e-6 m 的绝对误差，这里必须好三个数量级以上
    assert np.abs(got / 1000.0 - xyz).max() < 1e-9


def test_comma_separated_large_field_and_mixed_small_field(tmp_path):
    path = tmp_path / "mixed.nas"
    path.write_text(
        "$ 注释\n"
        "GRID*,1,0,1.2345678901E+03,-2.5E-01\n"
        "*,3.0E+00\n"
        "GRID           2       0     1.0     2.0     3.0\n"
        "GRID*                  3               0 1.000000000E+00 2.000000000E+00\n"
        "*        3.000000000E+00\n")
    got, id_map = _parse(path)
    assert sorted(id_map) == [1, 2, 3]
    np.testing.assert_allclose(got[id_map[1]], [1234.5678901, -0.25, 3.0], rtol=1e-12)
    np.testing.assert_allclose(got[id_map[2]], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(got[id_map[3]], [1.0, 2.0, 3.0])


def test_large_field_without_continuation_is_dropped_not_misread(tmp_path):
    """GRID* 后面跟的不是 `*` 续行（或文件结束）：该节点丢弃，下一张卡照常解析。"""
    path = tmp_path / "broken.nas"
    path.write_text(
        "GRID*                  1               0 1.000000000E+00 2.000000000E+00\n"
        "GRID           2       0     4.0     5.0     6.0\n"
        "GRID*                  3               0 7.000000000E+00 8.000000000E+00\n")
    got, id_map = _parse(path)
    assert sorted(id_map) == [2]
    np.testing.assert_allclose(got[id_map[2]], [4.0, 5.0, 6.0])


def test_each_unparsable_line_is_counted_once(tmp_path):
    """丢弃率按行计：96 张好卡 + 4 张坏卡（4%）不应超过 5% 的阈值。此前坏行在分支内部与外层各计一次，
    这个文件会被算成 8/104 而误判为格式不符。"""
    lines = [f"GRID    {i:>8}       0     1.0     2.0     3.0\n" for i in range(1, 97)]
    lines += ["GRID    xx  yy\n"] * 4
    path = tmp_path / "noisy.nas"
    path.write_text("".join(lines))
    _got, id_map = _parse(path)
    assert len(id_map) == 96


def test_too_many_bad_lines_still_raises(tmp_path):
    from autoflowcfd.grid.nas_io.nas_parser_exceptions import NASParseError

    lines = [f"GRID    {i:>8}       0     1.0     2.0     3.0\n" for i in range(1, 21)]
    lines += ["GRID    xx  yy\n"] * 5
    path = tmp_path / "bad.nas"
    path.write_text("".join(lines))
    with pytest.raises(NASParseError):
        _parse(path)


def _grid(nid, x, y, z):
    return f"GRID    {nid:>8}       0{x:>8}{y:>8}{z:>8}\n"


def test_volume_mesh_export_import_round_trip(tmp_path):
    """体网格导入端（`nas_parser_volume.py`）此前另有一份只认 8 字符小字段的 GRID 解析，导出端改写 `GRID*` 之后，
    `grid generate-volume` 自己写出的体网格被 `solve steady` 拒绝："No GRID cards found"。"""
    from autoflowcfd.grid.nas_io.nas_export import export_volume_mesh_to_nas
    from autoflowcfd.grid.nas_io.nas_parser_volume import parse_volume_mesh_nas

    src = tmp_path / "small.nas"
    pts = [(0, 0, 0), (1000, 0, 0), (0, 1000, 0), (0, 0, 1000), (1000, 0, 1000), (0, 1000, 1000), (0, 0, 2000)]
    src.write_text(
        "".join(_grid(i + 1, f"{x}.", f"{y}.", f"{z}.") for i, (x, y, z) in enumerate(pts))
        + "CPENTA         1       1       1       2       3       4       5       6\n"
        + "CTETRA         2       1       4       5       6       7\n")
    mesh = parse_volume_mesh_nas(str(src), units="mm")
    rng = np.random.default_rng(1)
    mesh.nodes.x[:] += rng.uniform(0, 1e-6, mesh.nodes.x.shape)     # 小字段写不下的偏移

    out = export_volume_mesh_to_nas(mesh, str(tmp_path / "volume.nas"), include_boundaries=False)
    text = open(out).read()
    assert "GRID*" in text
    assert "ANSA" not in text                     # 文件头如实写明来源，不冒用第三方软件名
    back = parse_volume_mesh_nas(out, units="mm")
    assert back.node_count == mesh.node_count
    np.testing.assert_allclose(back.nodes.x, mesh.nodes.x, rtol=0, atol=1e-9)     # 10 位有效数字、1 m 量级
    np.testing.assert_allclose(back.nodes.z, mesh.nodes.z, rtol=0, atol=1e-9)     # 10 位有效数字、1 m 量级
    np.testing.assert_array_equal(back.prism_cells.connectivity, mesh.prism_cells.connectivity)
    np.testing.assert_array_equal(back.cells.connectivity, mesh.cells.connectivity)
