"""`grid convert` 的 CGNS 写出（CGNS/HDF5 文件布局，h5py 直接写）。

CGNS/HDF5 布局要点（CGNS 标准 "HDF5 File Mapping"）：每个 CGNS 节点是一个 HDF5 组，带 `name`（33 字节）、
`label`（33 字节）、`type`（3 字节：MT/I4/I8/R4/R8/C1）、`flags` 四个属性，数据放在组内名为 " data" 的数据集里；
多维数组按 Fortran 维度顺序存储（HDF5 里维度倒序）。根组另有 " format"、" hdf5version" 两个数据集。

写出内容：一个 CGNSBase_t、一个非结构 Zone_t、GridCoordinates（CoordinateX/Y/Z）。
- 体网格（CellDimension=3）：PENTA_6、TETRA_4 体单元分区，外部面按边界组各一个 TRI_3 分区，ZoneBC_t 下每组一个
  BC_t（PointRange 指向该组的面单元、GridLocation=FaceCenter，BC 类型按面网格识别出的边界条件类型映射）；
- 面网格（CellDimension=2）：每个边界组一个 TRI_3 分区。
"""

import numpy as np

from .source import ConvertibleMesh
from .writers import oriented_prisms, oriented_tets

_ELEMENT_TYPE = {"TRI_3": 5, "TETRA_4": 10, "PENTA_6": 14}
_BC_TYPE = {
    "WALL": "BCWall",
    "SLIP_WALL": "BCWallInviscid",
    "VELOCITY_INLET": "BCInflow",
    "PRESSURE_OUTLET": "BCOutflow",
    "SYMMETRY": "BCSymmetryPlane",
    "FARFIELD": "BCFarfield",
}
_CGNS_VERSION = 4.2


def _chars(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("ascii"), dtype=np.int8)


def _node(parent, name: str, label: str, dtype_code: str, data=None):
    if len(name) > 32:
        raise ValueError(f"CGNS 节点名最长 32 个字符：{name!r}")
    g = parent.create_group(name, track_order=True)
    g.attrs.create("name", np.bytes_(name), dtype="S33")
    g.attrs.create("label", np.bytes_(label), dtype="S33")
    g.attrs.create("type", np.bytes_(dtype_code), dtype="S3")
    g.attrs.create("flags", np.array([1], dtype=np.int32))
    if data is not None:
        g.create_dataset(" data", data=data)
    return g


def _elements(zone, name: str, etype: str, start: int, conn: np.ndarray) -> int:
    sec = _node(zone, name, "Elements_t", "I4", np.array([_ELEMENT_TYPE[etype], 0], dtype=np.int32))
    end = start + len(conn) - 1
    _node(sec, "ElementRange", "IndexRange_t", "I4", np.array([start, end], dtype=np.int32))
    _node(sec, "ElementConnectivity", "DataArray_t", "I4", (np.asarray(conn, dtype=np.int64) + 1).astype(np.int32).ravel())
    return end + 1


def _section_name(name: str, taken: set) -> str:
    base = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in name)[:32] or "unnamed"
    out, k = base, 1
    while out in taken:
        suffix = f"_{k}"
        out = base[:32 - len(suffix)] + suffix
        k += 1
    taken.add(out)
    return out


def write_cgns(mesh: ConvertibleMesh, path: str) -> None:
    import h5py

    n_nodes = len(mesh.nodes)
    if n_nodes >= np.iinfo(np.int32).max:
        raise ValueError("节点数超出 32 位 CGNS 索引范围")
    cell_dim = 3 if mesh.is_volume else 2
    with h5py.File(path, "w", track_order=True) as f:
        f.attrs.create("name", np.bytes_("HDF5 MotherNode"), dtype="S33")
        f.attrs.create("label", np.bytes_("Root Node of HDF5 File"), dtype="S33")
        f.attrs.create("type", np.bytes_("MT"), dtype="S3")
        f.create_dataset(" format", data=_chars("IEEE_LITTLE_32"))
        f.create_dataset(" hdf5version", data=np.frombuffer(
            f"HDF5 Version {h5py.version.hdf5_version}".encode("ascii").ljust(33, b"\0"), dtype=np.int8))
        _node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4", np.array([_CGNS_VERSION], dtype=np.float32))
        base = _node(f, "Base", "CGNSBase_t", "I4", np.array([cell_dim, 3], dtype=np.int32))

        n_cells = len(mesh.prisms) + len(mesh.tets) if mesh.is_volume else len(mesh.surface_tris)
        zone = _node(base, "Zone", "Zone_t", "I4", np.array([[n_nodes], [n_cells], [0]], dtype=np.int32))
        _node(zone, "ZoneType", "ZoneType_t", "C1", _chars("Unstructured"))
        coords = _node(zone, "GridCoordinates", "GridCoordinates_t", "MT")
        for k, axis in enumerate("XYZ"):
            _node(coords, f"Coordinate{axis}", "DataArray_t", "R8", np.ascontiguousarray(mesh.nodes[:, k], dtype=np.float64))

        taken = set()
        start = 1
        if mesh.is_volume:
            if len(mesh.prisms):
                start = _elements(zone, _section_name("Prisms", taken), "PENTA_6", start,
                                  oriented_prisms(mesh.nodes, mesh.prisms, bottom_normal_toward_top=True))
            if len(mesh.tets):
                start = _elements(zone, _section_name("Tetras", taken), "TETRA_4", start,
                                  oriented_tets(mesh.nodes, mesh.tets))
        face_ranges = []
        for gi, name in enumerate(mesh.group_names):
            tris = mesh.surface_tris[mesh.surface_group == gi]
            if len(tris) == 0:
                continue
            first = start
            start = _elements(zone, _section_name(name, taken), "TRI_3", start, tris)
            face_ranges.append((name, first, start - 1))

        if mesh.is_volume and face_ranges:
            # 每个边界组一个 Family_t（FamilyBC_t 记真实 BC 类型），BC_t 写 FamilySpecified 并以 FamilyName
            # 指向它——Pointwise/ICEM 等前处理器的通用写法，下游读取器按族识别边界（VTK 的 CGNS 读取器只认
            # FamilySpecified/BCDirichlet/BCNeumann，其余 BC 类型直接跳过）
            zone_bc = _node(zone, "ZoneBC", "ZoneBC_t", "MT")
            family_taken = set()
            for name, first, last in face_ranges:
                family = _section_name(name, family_taken)
                fam = _node(base, family, "Family_t", "MT")
                _node(fam, "FamilyBC", "FamilyBC_t", "C1",
                      _chars(_BC_TYPE.get(mesh.bc_types.get(name, ""), "BCTypeNull")))
                bc = _node(zone_bc, family, "BC_t", "C1", _chars("FamilySpecified"))
                _node(bc, "PointRange", "IndexRange_t", "I4", np.array([[first], [last]], dtype=np.int32))
                _node(bc, "GridLocation", "GridLocation_t", "C1", _chars("FaceCenter"))
                _node(bc, "FamilyName", "FamilyName_t", "C1", _chars(family))
