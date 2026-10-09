"""`grid convert` 的 VTK 与 STL 写出。"""

from typing import List

import numpy as np

from .source import ConvertibleMesh


def oriented_tets(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """四面体按 det(p1-p0, p2-p0, p3-p0) > 0 排列（VTK_TETRA 与 CGNS TETRA_4 的共同约定）。"""
    t = np.asarray(tets, dtype=np.int64).copy()
    p0 = nodes[t[:, 0]]
    det = np.einsum('ij,ij->i', np.cross(nodes[t[:, 1]] - p0, nodes[t[:, 2]] - p0), nodes[t[:, 3]] - p0)
    flip = det < 0
    t[flip, 1], t[flip, 2] = t[flip, 2].copy(), t[flip, 1].copy()
    return t


def oriented_prisms(nodes: np.ndarray, prisms: np.ndarray, bottom_normal_toward_top: bool) -> np.ndarray:
    """棱柱 (v0,v1,v2,w0,w1,w2) 按底面三角形右手法向与顶面的相对方向排列。

    VTK_WEDGE：底面 (0,1,2) 的法向背离顶面（`bottom_normal_toward_top=False`）；
    CGNS PENTA_6：底面 (1,2,3) 的法向指向顶面（`True`）。不满足的单元交换 v1/v2 与 w1/w2。
    """
    p = np.asarray(prisms, dtype=np.int64).copy()
    n = np.cross(nodes[p[:, 1]] - nodes[p[:, 0]], nodes[p[:, 2]] - nodes[p[:, 0]])
    up = nodes[p[:, 3:6]].mean(axis=1) - nodes[p[:, 0:3]].mean(axis=1)
    toward = np.einsum('ij,ij->i', n, up) > 0
    flip = toward != bottom_normal_toward_top
    for a, b in ((1, 2), (4, 5)):
        p[flip, a], p[flip, b] = p[flip, b].copy(), p[flip, a].copy()
    return p


def write_vtk(mesh: ConvertibleMesh, path: str) -> None:
    """VTK 非结构网格（扩展名 .vtu 为 XML，.vtk 为旧版格式）。

    体网格写体单元（VTK_TETRA / VTK_WEDGE），单元数据 `cell_kind`（0 四面体、1 棱柱）；面网格写三角形，
    单元数据 `group_id`，字段数据 `group_names` 给出组号对应的组名。
    """
    import pyvista as pv

    if mesh.is_volume:
        tets = oriented_tets(mesh.nodes, mesh.tets)
        prisms = oriented_prisms(mesh.nodes, mesh.prisms, bottom_normal_toward_top=False)
        cells = np.concatenate([
            np.hstack([np.full((len(prisms), 1), 6), prisms]).ravel(),
            np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
        ])
        types = np.concatenate([np.full(len(prisms), pv.CellType.WEDGE), np.full(len(tets), pv.CellType.TETRA)])
        grid = pv.UnstructuredGrid(cells, types.astype(np.uint8), mesh.nodes)
        grid.cell_data["cell_kind"] = np.concatenate([np.ones(len(prisms), np.int32), np.zeros(len(tets), np.int32)])
    else:
        tris = mesh.surface_tris
        cells = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
        grid = pv.UnstructuredGrid(cells, np.full(len(tris), pv.CellType.TRIANGLE, dtype=np.uint8), mesh.nodes)
        grid.cell_data["group_id"] = mesh.surface_group.astype(np.int32)
        grid.field_data["group_names"] = np.array(mesh.group_names)
    grid.save(path, binary=True)


def _solid_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in name) or "unnamed"


def write_stl(mesh: ConvertibleMesh, path: str) -> None:
    """ASCII STL，每个边界组一个 solid（面网格是全部三角形，体网格是外部面）。

    法向按三角形顶点顺序的右手法则计算（STL 约定），退化三角形写零向量。
    """
    tris = mesh.surface_tris
    p0, p1, p2 = mesh.nodes[tris[:, 0]], mesh.nodes[tris[:, 1]], mesh.nodes[tris[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)
    lines: List[str] = []
    for gi, name in enumerate(mesh.group_names):
        idx = np.flatnonzero(mesh.surface_group == gi)
        if len(idx) == 0:
            continue
        solid = _solid_name(name)
        lines.append(f"solid {solid}")
        for i in idx:
            nx, ny, nz = normal[i]
            lines.append(f"  facet normal {nx:.9e} {ny:.9e} {nz:.9e}")
            lines.append("    outer loop")
            for p in (p0[i], p1[i], p2[i]):
                lines.append(f"      vertex {p[0]:.12e} {p[1]:.12e} {p[2]:.12e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {solid}")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
