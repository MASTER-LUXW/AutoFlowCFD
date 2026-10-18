"""边界条件类型：按属性名识别、显式指定与完整性检查（`grid/nas_io/boundary_types.py`，I-132）。

修复前：子串匹配（`sym` 命中 `asymmetric`）；`farfield`/`freestream` 被归为滑移壁；没命中的名字与取不到属性名的
面一律默认 WALL；用户没有任何途径指定类型。
"""

import numpy as np
import pytest

from autoflowcfd.grid.nas_io.boundary_types import (
    UNSPECIFIED, BoundaryTypeError, apply_bc_types, detect_boundary_type, parse_bc_type_options,
    require_specified_bc_types)


@pytest.mark.parametrize("name,expected", [
    ("inlet", "VELOCITY_INLET"), ("outlet", "PRESSURE_OUTLET"), ("body", "WALL"), ("ground", "WALL"),
    ("tunnel", "SLIP_WALL"),                                  # 示例算例里的五个名字
    ("Farfield", "FARFIELD"), ("far_field", "FARFIELD"), ("freestream", "FARFIELD"),
    ("sym_plane", "SYMMETRY"), ("SYMMETRY", "SYMMETRY"),
    ("asymmetric_body", "WALL"),                              # 'sym' 不再按子串命中
    ("exit_duct", "PRESSURE_OUTLET"), ("exitwall", UNSPECIFIED),
    ("TUNNEL_WALL", UNSPECIFIED),                             # 同时命中滑移壁与壁面：有歧义
    ("surface", UNSPECIFIED), ("fuselage", UNSPECIFIED),      # 识别不了：不再默认 WALL
    ("车身", "WALL"), ("入口", "VELOCITY_INLET"), ("远场", "FARFIELD"),
])
def test_detect_boundary_type(name, expected):
    assert detect_boundary_type(name) == expected


def test_bc_type_options():
    assert parse_bc_type_options(["fuselage=wall", " far = FARFIELD "]) == {"fuselage": "WALL", "far": "FARFIELD"}
    for bad in (["fuselage"], ["=WALL"], ["x=BOGUS"], ["x=WALL", "x=SYMMETRY"]):
        with pytest.raises(BoundaryTypeError):
            parse_bc_type_options(bad)


def _bmap():
    from autoflowcfd.grid.structures import BoundaryMap

    return BoundaryMap(groups={"inlet": np.array([0]), "fuselage": np.array([1])},
                       bc_types={"inlet": "VELOCITY_INLET", "fuselage": UNSPECIFIED})


def test_unspecified_groups_must_be_given_explicitly():
    bm = _bmap()
    with pytest.raises(BoundaryTypeError, match="fuselage"):
        require_specified_bc_types(bm)
    apply_bc_types(bm, {"fuselage": "WALL"})
    require_specified_bc_types(bm)
    assert bm.bc_types == {"inlet": "VELOCITY_INLET", "fuselage": "WALL"}
    with pytest.raises(BoundaryTypeError, match="不在网格的边界组"):
        apply_bc_types(bm, {"wing": "WALL"})


def _write_surface_nas(path, names):
    """每个名字一个三角形（同一组节点），按 ANSA 约定写属性名。"""
    lines = ["BEGIN BULK\n"]
    for pid, name in enumerate(names, start=1):
        lines.append(f"$ANSA_NAME_COMMENT;{pid};PSHELL;{name};;NO;NO;NO;NO;\n")
        lines.append(f"PSHELL,{pid},1,0.001\n")
    for nid, xyz in enumerate([(0, 0, 0), (1000, 0, 0), (0, 1000, 0), (0, 0, 1000)], start=1):
        lines.append(f"GRID,{nid},,{xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}\n")
    for eid, pid in enumerate(range(1, len(names) + 1), start=1):
        lines.append(f"CTRIA3,{eid},{pid},1,2,{(eid % 2) + 3}\n")
    lines.append("ENDDATA\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_parser_applies_explicit_types_and_volume_generation_requires_them(tmp_path):
    from autoflowcfd.grid.nas_io.parser_core import NASParser
    from autoflowcfd.grid.nas_io.parser_volume import generate_volume_mesh_from_surface

    nas = tmp_path / "s.nas"
    _write_surface_nas(nas, ["inlet", "fuselage"])
    grid = NASParser(str(nas)).parse(skip_validation=True)
    assert grid.boundaries.bc_types["fuselage"] == UNSPECIFIED
    with pytest.raises(BoundaryTypeError, match="fuselage"):
        generate_volume_mesh_from_surface(grid)                      # 在任何生成工作之前就拒绝
    grid = NASParser(str(nas), bc_types={"fuselage": "WALL"}).parse(skip_validation=True)
    assert grid.boundaries.bc_types == {"inlet": "VELOCITY_INLET", "fuselage": "WALL"}
    with pytest.raises(Exception, match="不在网格的边界组"):
        NASParser(str(nas), bc_types={"wing": "WALL"}).parse(skip_validation=True)


def test_import_reads_the_surface_mesh_in_the_same_units(monkeypatch, tmp_path):
    """伴生面网格与体网格按同一长度单位读入（此前面网格恒按默认 'mm'）。"""
    import autoflowcfd.grid.mesh_gen.utils.mesh_external_import as mod
    import autoflowcfd.grid.nas_io.nas_parser_volume as vol
    import autoflowcfd.grid.nas_io.parser_core as core

    seen = {}

    class Stop(Exception):
        pass

    class FakeParser:
        def __init__(self, path, units="mm", bc_types=None):
            seen.update(units=units, bc_types=bc_types)

        def parse(self):
            raise Stop

    monkeypatch.setattr(vol, "parse_volume_mesh_nas", lambda path, units: object())
    monkeypatch.setattr(core, "NASParser", FakeParser)
    with pytest.raises(Stop):
        mod.import_external_volume_mesh("v.nas", "s.nas", units="m", bc_types={"x": "WALL"})
    assert seen == {"units": "m", "bc_types": {"x": "WALL"}}


def test_periodic_is_not_an_assignable_type():
    """周期边界还需要配对组与平移向量，"组名=类型"通道给不出：名字里的 periodic/周期 不识别为周期，
    --bc-type 组=PERIODIC 报错（此前接受它、加载网格时才因缺 paired_with/translation 报错）。"""
    from autoflowcfd.grid.nas_io.boundary_types import (
        UNSPECIFIED, BoundaryTypeError, detect_boundary_type, parse_bc_type_options)

    assert detect_boundary_type("periodic_x") == UNSPECIFIED
    assert detect_boundary_type("周期面") == UNSPECIFIED
    with pytest.raises(BoundaryTypeError, match="不受支持"):
        parse_bc_type_options(["side=PERIODIC"])
