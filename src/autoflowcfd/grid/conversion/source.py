"""`grid convert` 的输入：把 .nas（面网格或体网格）读成与输出格式无关的统一表示。"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ConvertibleMesh:
    """单位米。体网格：`tets`/`prisms` 非空，`surface_tris` 是外部面；面网格：只有 `surface_tris`。

    `surface_group[i]` 是第 i 个三角形所属组在 `group_names` 里的下标；`bc_types` 是组名 -> 边界条件类型
    （面网格读入时按组名识别出的，没有就不在字典里）。
    """

    nodes: np.ndarray
    tets: np.ndarray
    prisms: np.ndarray
    surface_tris: np.ndarray
    surface_group: np.ndarray
    group_names: List[str]
    bc_types: Dict[str, str] = field(default_factory=dict)

    @property
    def is_volume(self) -> bool:
        return len(self.tets) + len(self.prisms) > 0


def _has_volume_cards(path: str) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            card = line[:8].split(",")[0].strip().upper()
            if card in ("CTETRA", "CPENTA"):
                return True
    return False


def _has_shell_cards(path: str) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line[:8].split(",")[0].strip().upper() == "CTRIA3":
                return True
    return False


def _flatten_groups(names: List[str], faces_by_group: Dict[str, np.ndarray]):
    tris, group = [], []
    for gi, name in enumerate(names):
        faces = faces_by_group.get(name)
        if faces is None or len(faces) == 0:
            continue
        tris.append(np.asarray(faces, dtype=np.int64))
        group.append(np.full(len(faces), gi, dtype=np.int64))
    if not tris:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.vstack(tris), np.concatenate(group)


def load_volume_mesh_nas(path: str, units: str = "mm"):
    """读体网格 .nas（GRID + CTETRA/CPENTA）为 `VolumeMeshData`，不做任何修复或重新生成（`grid convert` 与
    后处理共用）。文件里同时有 CTRIA3 边界面（本项目导出的体网格、ANSA 体网格导出都有）时，把这些带 PSHELL
    分组的三角形挂成 `surface_mesh`，外部面由此逐面定组（`mesh_boundary.exterior_faces_by_group`）。

    Raises:
        ValueError: 文件里没有 CTETRA/CPENTA（是面网格）
    """
    from ..nas_io.nas_parser_volume import parse_volume_mesh_nas
    from ..nas_io.parser_core import NASParser

    if not _has_volume_cards(path):
        raise ValueError(f"'{path}' 没有 CTETRA/CPENTA 卡片，是面网格而不是体网格")
    vol = parse_volume_mesh_nas(path, units=units)
    if _has_shell_cards(path):
        shell = NASParser(path, units=units).parse()
        vol.surface_mesh = {
            "nodes": np.column_stack([shell.nodes.x, shell.nodes.y, shell.nodes.z]),
            "faces": shell.cells.connectivity,
            "boundaries": shell.boundaries,
        }
    return vol


def load_convertible_mesh(path: str, units: str = "mm") -> ConvertibleMesh:
    """读 .nas 网格。

    - 含 CTETRA/CPENTA：体网格（`load_volume_mesh_nas`）。带 CTRIA3 边界面时外部面按这些三角形逐面定组
      （几何包含判据），定不了组的外部面归 'UNCLASSIFIED'；没有 CTRIA3 时全部外部面归 'exterior'。
    - 否则：面网格（`NASParser`），三角形按 PSHELL 分组，不属于任何组的归 'ungrouped'。

    Args:
        path: .nas 文件
        units: 文件坐标单位（'mm'、'm' 或 'auto'，与 NASParser 同义），结果一律换算到米
    """
    from ..nas_io.parser_core import NASParser

    if _has_volume_cards(path):
        from ..mesh_gen.utils.mesh_boundary import exterior_faces_by_group

        vol = load_volume_mesh_nas(path, units=units)
        nodes = np.column_stack([vol.nodes.x, vol.nodes.y, vol.nodes.z])
        prisms = vol.prism_cells.connectivity if vol.prism_cells is not None else np.empty((0, 6), np.int64)
        tets = vol.cells.connectivity
        bc_types: Dict[str, str] = {}
        surface = getattr(vol, "surface_mesh", None)
        if surface is not None:
            faces_by_group = exterior_faces_by_group(vol)
            bc_types = dict(surface["boundaries"].bc_types)
        else:
            faces = vol.ensure_faces_exist()
            faces_by_group = {"exterior": faces.node_connectivity[faces.get_boundary_face_indices()]}
        names = list(faces_by_group)
        tris, group = _flatten_groups(names, faces_by_group)
        return ConvertibleMesh(nodes=nodes, tets=np.asarray(tets, np.int64), prisms=np.asarray(prisms, np.int64),
                               surface_tris=tris, surface_group=group, group_names=names,
                               bc_types={k: v for k, v in bc_types.items() if k in names})

    grid = NASParser(path, units=units).parse()
    tris = np.asarray(grid.cells.connectivity, dtype=np.int64)
    group = np.full(len(tris), -1, dtype=np.int64)
    names = list(grid.boundaries.groups)
    for gi, name in enumerate(names):
        idx = np.asarray(grid.boundaries.groups[name])
        group[idx[idx < len(tris)]] = gi
    if (group < 0).any():
        names.append("ungrouped")
        group[group < 0] = len(names) - 1
    return ConvertibleMesh(
        nodes=np.column_stack([grid.nodes.x, grid.nodes.y, grid.nodes.z]),
        tets=np.empty((0, 4), np.int64), prisms=np.empty((0, 6), np.int64),
        surface_tris=tris, surface_group=group, group_names=names,
        bc_types={k: v for k, v in grid.boundaries.bc_types.items() if k in names})
