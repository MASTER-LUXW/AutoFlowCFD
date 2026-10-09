"""`autoflowcfd grid convert`（2026-10-09 以前是占位命令，调用即报 "not yet implemented"）。

每种输出都用独立的读取器读回核对：VTK/STL 用 pyvista，CGNS 用 h5py 直接按 CGNS/HDF5 布局读结构、并用 VTK 的
CGNS 读取器读边界面片。体网格用结构化四面体立方体；棱柱的朝向约定由单独的测试逐单元核对（真实 cube_demo
网格的棱柱 + 四面体转换由 CLI 端到端验证覆盖）。
"""

import json

import h5py
import numpy as np
import pytest
import pyvista as pv
from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from autoflowcfd.grid.conversion import load_convertible_mesh, write_cgns, write_stl, write_vtk
from autoflowcfd.grid.nas_io.nas_export import export_volume_mesh_to_nas
from tests.unit.test_exterior_faces_by_group import _volume_mesh


def _surface_nas(path):
    """单位立方体的面网格（mm），六个面各一个 PSHELL 组。"""
    pts = np.array([[x, y, z] for x in (0, 1000) for y in (0, 1000) for z in (0, 1000)], dtype=float)
    quads = {"x_min": (0, 1, 3, 2), "x_max": (4, 6, 7, 5), "y_min": (0, 4, 5, 1),
             "y_max": (2, 3, 7, 6), "z_min": (0, 2, 6, 4), "z_max": (1, 5, 7, 3)}
    lines = [f"GRID    {i + 1:>8}        {p[0]:>8.1f}{p[1]:>8.1f}{p[2]:>8.1f}\n" for i, p in enumerate(pts)]
    eid = 1
    for pid, (name, (a, b, c, d)) in enumerate(quads.items(), start=1):
        lines.append(f"$ANSA_NAME_COMMENT;{pid};PSHELL;{name};;NO;NO;NO;NO;\n")
        lines.append(f"PSHELL  {pid:>8}{pid:>8}     1.0\n")
        for tri in ((a, b, c), (a, c, d)):
            lines.append(f"CTRIA3  {eid:>8}{pid:>8}" + "".join(f"{v + 1:>8}" for v in tri) + "\n")
            eid += 1
    lines.append("ENDDATA\n")
    path.write_text("".join(lines))
    return path


def _volume_nas(tmp_path):
    """单位立方体四面体网格经本项目导出端写成的 .nas（带 CTRIA3 边界面与 PSHELL 组）。"""
    vm, nodes, tets = _volume_mesh()
    out = export_volume_mesh_to_nas(vm, str(tmp_path / "cube.nas"))
    return out, len(tets)


def test_surface_mesh_to_all_formats(tmp_path):
    src = _surface_nas(tmp_path / "surface.nas")
    mesh = load_convertible_mesh(str(src))
    assert not mesh.is_volume and len(mesh.surface_tris) == 12 and len(mesh.group_names) == 6

    write_vtk(mesh, str(tmp_path / "s.vtu"))
    g = pv.read(str(tmp_path / "s.vtu"))
    assert g.n_cells == 12 and abs(g.compute_cell_sizes()["Area"].sum() - 6.0) < 1e-12   # 米制
    assert sorted(set(g.cell_data["group_id"])) == list(range(6))

    write_stl(mesh, str(tmp_path / "s.stl"))
    text = (tmp_path / "s.stl").read_text()
    assert text.count("facet normal") == 12 and text.count("endsolid") == 6

    write_cgns(mesh, str(tmp_path / "s.cgns"))
    with h5py.File(tmp_path / "s.cgns", "r") as f:
        assert list(f["Base/ data"][()]) == [2, 3]                     # 面网格：CellDimension 2
        sections = [k for k, v in f["Base/Zone"].items()
                    if isinstance(v, h5py.Group) and v.attrs["label"].startswith(b"Elements_t")]
        assert sorted(sections) == sorted(mesh.group_names)


def test_volume_mesh_to_all_formats(tmp_path):
    src, n_tets = _volume_nas(tmp_path)
    mesh = load_convertible_mesh(src)
    assert mesh.is_volume and len(mesh.tets) == n_tets
    assert len(mesh.surface_tris) == 108 and sorted(mesh.group_names) == sorted(
        ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"])

    write_vtk(mesh, str(tmp_path / "v.vtu"))
    sizes = pv.read(str(tmp_path / "v.vtu")).compute_cell_sizes()["Volume"]
    assert (sizes > 0).all() and abs(sizes.sum() - 1.0) < 1e-12

    write_stl(mesh, str(tmp_path / "v.stl"))
    assert (tmp_path / "v.stl").read_text().count("facet normal") == 108

    write_cgns(mesh, str(tmp_path / "v.cgns"))
    with h5py.File(tmp_path / "v.cgns", "r") as f:
        zone = f["Base/Zone"]
        assert list(zone[" data"][()].ravel()) == [len(mesh.nodes), n_tets, 0]
        conn = zone["Tetras/ElementConnectivity/ data"][()].reshape(-1, 4) - 1
        xyz = np.column_stack([zone[f"GridCoordinates/Coordinate{a}/ data"][()] for a in "XYZ"])
        p0 = xyz[conn[:, 0]]
        det = np.einsum("ij,ij->i", np.cross(xyz[conn[:, 1]] - p0, xyz[conn[:, 2]] - p0), xyz[conn[:, 3]] - p0)
        assert (det > 0).all()                                         # CGNS TETRA_4：N1N2N3 法向指向 N4
        families = {k: bytes(f["Base"][k]["FamilyBC/ data"][()]).decode() for k in f["Base"]
                    if isinstance(f["Base"][k], h5py.Group) and f["Base"][k].attrs["label"].startswith(b"Family_t")}
        assert families == {name: "BCWall" for name in mesh.group_names}

    from vtkmodules.vtkIOCGNSReader import vtkCGNSReader
    r = vtkCGNSReader()
    r.SetFileName(str(tmp_path / "v.cgns"))
    r.UpdateInformation()
    r.EnableAllBases()
    r.EnableAllFamilies()
    r.SetLoadBndPatch(True)
    r.Update()
    zone = pv.wrap(r.GetOutput())[0][0]
    patches = zone["Patches"]
    assert sorted(patches.get_block_name(i) for i in range(patches.n_blocks)) == sorted(mesh.group_names)
    for i in range(patches.n_blocks):
        assert abs(patches[i].compute_cell_sizes()["Area"].sum() - 1.0) < 1e-12


def test_cgns_penta_follows_the_standard_orientation(tmp_path):
    """CGNS PENTA_6：N1N2N3 的右手法向指向 N4N5N6（SIDS 面表 F4 = N1 N3 N2 朝外；与四面体、金字塔"首个面的
    节点顺序法向朝内"同一约定）。VTK 楔形相反，VTK 的 CGNS 读取器原样映射，读回体积为负是读取器的映射。"""
    from autoflowcfd.grid.conversion.source import ConvertibleMesh

    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
    for prism in ([0, 1, 2, 3, 4, 5], [0, 2, 1, 3, 5, 4]):             # 两种输入朝向
        mesh = ConvertibleMesh(nodes=nodes, tets=np.empty((0, 4), np.int64), prisms=np.array([prism]),
                               surface_tris=np.empty((0, 3), np.int64), surface_group=np.empty(0, np.int64),
                               group_names=[])
        write_cgns(mesh, str(tmp_path / "p.cgns"))
        with h5py.File(tmp_path / "p.cgns", "r") as f:
            c = f["Base/Zone/Prisms/ElementConnectivity/ data"][()] - 1
        n = np.cross(nodes[c[1]] - nodes[c[0]], nodes[c[2]] - nodes[c[0]])
        assert np.dot(n, nodes[c[3:]].mean(axis=0) - nodes[c[:3]].mean(axis=0)) > 0
        write_vtk(mesh, str(tmp_path / "p.vtu"))
        assert pv.read(str(tmp_path / "p.vtu")).compute_cell_sizes()["Volume"][0] > 0


@pytest.mark.parametrize("fmt,ext", [("vtk", ".vtu"), ("stl", ".stl"), ("cgns", ".cgns")])
def test_cli_convert(tmp_path, fmt, ext):
    src = _surface_nas(tmp_path / "surface.nas")
    result = CliRunner().invoke(cli, ["grid", "convert", str(src), "-f", fmt, "--json"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.output[result.output.index("{"):])
    assert info["status"] == "success" and info["boundary_triangles"] == 12
    assert (tmp_path / f"surface{ext}").exists()
