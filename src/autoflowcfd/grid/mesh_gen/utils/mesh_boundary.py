"""边界识别与映射模块。

- `map_generated_boundaries`：本项目自己生成的混合网格（棱柱 + 四面体）——每个外部面落在一个输入面网格三角形上、
  每个三角形被外部面恰好覆盖，否则报错（网格不封闭，或修补改动了输入表面）。
- `map_boundaries_by_geometry`：外部工具生成的体网格——节点与面网格无关，只能按位置近似匹配。
- `match_points_to_surface_groups`：点到面网格边界组的唯一匹配实现（先几何包含、其余按位置近似；
  `map_boundaries_by_geometry`、`exterior_faces_by_group` 与求解期逐面打标签
  `face_connectivity_boundary_tags.tag_boundary_groups_by_geometry` 共用）。
"""

import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...structures import BoundaryMap, GridData, VolumeMeshData

# 几何包含判据：法向距离相对面网格包围盒最大边长、重心坐标的下界、覆盖面积的相对误差。生成管线不移动面网格
# 节点（BL 挤出以面网格节点为棱柱底面、外边界原样使用），外部面落在面网格三角形上只差浮点舍入；
# tetgen 在边界上插 Steiner 点时外部面是面网格三角形的细分，仍然精确落在原三角形上。
_ON_SURFACE_RELATIVE_TOLERANCE = 1e-9
_BARYCENTRIC_TOLERANCE = 1e-9
_AREA_RELATIVE_TOLERANCE = 1e-6
_CANDIDATE_BLOCK = 4_000_000


def _inside(p, a, b, c, normal, nn, dist_tol):
    """点 p（..., 3）是否落在三角形 (a, b, c) 上：到平面距离 <= dist_tol 且投影点的重心坐标都 >= -容差。"""
    signed = np.einsum('...k,...k->...', p - a, normal)
    dist = np.abs(signed) / np.sqrt(nn)
    q = p - (signed / nn)[..., None] * normal
    u = np.einsum('...k,...k->...', np.cross(b - q, c - q), normal) / nn
    v = np.einsum('...k,...k->...', np.cross(c - q, a - q), normal) / nn
    w = 1.0 - u - v
    return (dist <= dist_tol) & (np.minimum(np.minimum(u, v), w) >= -_BARYCENTRIC_TOLERANCE)


def _containing_triangle(points: np.ndarray, tri: np.ndarray, dist_tol: float) -> np.ndarray:
    """每个点所在的面网格三角形下标（点到三角形平面的距离 <= dist_tol 且重心坐标都 >= -容差；找不到为 -1）。

    包含点 p 的三角形 j 满足 |p - c_j| <= r_j（c_j 质心、r_j 质心到顶点的最大距离）。先查最近的 8、64、512 个
    质心（绝大多数点在第一档就找到）；仍没找到的点，候选只可能是 r_j 不小于"第 512 近质心距离"的大三角形，
    只在这些大三角形上按块检查（每块 点数 × 候选数 不超过 _CANDIDATE_BLOCK），内存有界——外部工具生成的网格
    外部面一般不精确落在面网格上，绝大多数点会走到这一步。
    """
    from scipy.spatial import cKDTree

    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    normal = np.cross(b - a, c - a)
    nn = np.einsum('ij,ij->i', normal, normal)
    centroid = tri.mean(axis=1)
    radius = np.linalg.norm(tri - centroid[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centroid)
    result = np.full(len(points), -1, dtype=np.int64)
    todo = np.arange(len(points))
    kth = np.zeros(len(points))
    for k in (8, 64, 512):
        if len(todo) == 0:
            return result
        k = min(k, len(tri))
        d, cand = tree.query(points[todo], k=k)
        cand = np.asarray(cand).reshape(len(todo), k)
        kth[todo] = np.asarray(d).reshape(len(todo), k)[:, -1]
        inside = _inside(points[todo][:, None, :], a[cand], b[cand], c[cand], normal[cand], nn[cand], dist_tol)
        found = inside.any(axis=1)
        result[todo[found]] = cand[found, np.argmax(inside[found], axis=1)]
        todo = todo[~found]
        if k == len(tri):
            return result
    if len(todo) == 0:
        return result
    big = np.flatnonzero(radius >= kth[todo].min())
    if len(big) == 0:
        return result
    block = max(1, _CANDIDATE_BLOCK // len(big))
    for start in range(0, len(todo), block):
        idx = todo[start:start + block]
        p = points[idx][:, None, :]
        near = np.linalg.norm(p - centroid[big][None], axis=2) <= radius[big][None]
        inside = near & _inside(p, a[big][None], b[big][None], c[big][None], normal[big][None], nn[big][None], dist_tol)
        found = inside.any(axis=1)
        result[idx[found]] = big[np.argmax(inside[found], axis=1)]
    return result


def _classify_exterior_faces(nodes, prism_cells, tet_cells, surface_nodes, surface_faces, surface_boundaries):
    """生成网格的外部面逐个找所在的面网格三角形（几何包含）。

    Returns:
        dict：names（组名）、owner（外部面所属单元，全局混合编号）、face_nodes（外部面节点编号）、
        group（外部面所属组下标，不在面网格上为 -1）、host（所在三角形下标或 -1）、corners（外部面三个角点坐标）、
        face_area、tri_area（面网格三角形面积）。
    """
    from ..extraction.face_extractor import FaceExtractor
    from ...schema.grid_nodes import NodeArray

    names: List[str] = list(surface_boundaries.groups.keys())
    tri_idx, tri_group = [], []
    for gi, name in enumerate(names):
        idx = np.asarray(surface_boundaries.groups[name])
        idx = idx[idx < len(surface_faces)]
        tri_idx.append(idx)
        tri_group.append(np.full(len(idx), gi, dtype=np.int64))
    tri_idx = np.concatenate(tri_idx)
    tri_group = np.concatenate(tri_group)
    tri = surface_nodes[surface_faces[tri_idx]]

    faces = FaceExtractor.extract_faces_mixed(
        np.asarray(prism_cells, dtype=np.int32).reshape(-1, 6),
        np.asarray(tet_cells, dtype=np.int32).reshape(-1, 4),
        NodeArray.from_array(nodes),
    )
    bidx = faces.get_boundary_face_indices()
    face_nodes = faces.node_connectivity[bidx].astype(np.int64)
    corners = nodes[face_nodes]
    dist_tol = _ON_SURFACE_RELATIVE_TOLERANCE * float(np.ptp(surface_nodes, axis=0).max())
    host = _containing_triangle(corners.mean(axis=1), tri, dist_tol)
    return dict(
        names=names,
        owner=faces.connectivity[bidx, 0].astype(np.int64),
        face_nodes=face_nodes,
        group=np.where(host >= 0, tri_group[np.maximum(host, 0)], -1),
        host=host,
        corners=corners,
        face_area=0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1),
        tri_area=0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1),
    )


def _groups_from_faces(names, owner, group, bc_source):
    groups, bc_types = {}, {}
    for gi, name in enumerate(names):
        cells = np.unique(owner[group == gi])
        if len(cells) == 0:
            continue
        groups[name] = cells.astype(np.int32)
        if name in bc_source:
            bc_types[name] = bc_source[name]
    return groups, bc_types


def map_generated_boundaries(
    nodes: np.ndarray,
    prism_cells: np.ndarray,
    tet_cells: np.ndarray,
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: 'BoundaryMap',
) -> 'BoundaryMap':
    """本项目生成的体网格（棱柱 + 四面体）的外部面 → 面网格边界组；同时校验外表面恰好覆盖输入表面。

    2026-10-09 以前这里只用四面体求"只出现一次的面"、按节点**编号**三元组去匹配面网格：贴着棱柱顶面的四面体
    面在纯四面体集合里也只出现一次，被当成边界面。cube_demo 实测 24689/40966 个"边界单元"进了 UNCLASSIFIED
    （按 WALL），其中 24635 个根本不拥有外部面；网格缺口处的 86 个外部面 66 个被当成壁面、20 个没有任何分组
    （求解器当远场）。

    外部面在完整混合网格上求（棱柱侧面四边形按面提取的统一对角线拆成三角形）。每个外部面的质心必须落在一个
    面网格三角形上，组取那个三角形的；每个面网格三角形被映射到它的外部面面积之和必须等于它自身的面积
    （不细分时就是一一对应，tetgen 在边界上插点时是细分）。任一项不满足说明体网格外表面与输入表面不同
    （内部缺口、或修补移动了表面点），直接报错——按这份网格给出的边界条件会是错的。

    Returns:
        BoundaryMap：组名 -> 拥有该组外部面的单元（全局混合编号：棱柱 [0,n_prism)、四面体其后）；
        边界条件类型取面网格自己的（没有类型的组不编造默认值，由求解器报错）。

    Raises:
        ValueError: 体网格外表面没有恰好覆盖输入面网格。
    """
    from ...schema.grid_boundaries import BoundaryMap

    ext = _classify_exterior_faces(nodes, prism_cells, tet_cells, surface_nodes, surface_faces, surface_boundaries)
    host, corners = ext["host"], ext["corners"]
    unmatched = np.flatnonzero(host < 0)
    covered = np.bincount(host[host >= 0], weights=ext["face_area"][host >= 0], minlength=len(ext["tri_area"]))
    short = np.flatnonzero(np.abs(covered - ext["tri_area"]) > _AREA_RELATIVE_TOLERANCE * ext["tri_area"])
    if len(unmatched) or len(short):
        parts = []
        if len(unmatched):
            pts = corners[unmatched].reshape(-1, 3)
            parts.append(f"{len(host)} 个外部面中 {len(unmatched)} 个不在任何面网格三角形上"
                         f"（位于包围盒 {pts.min(axis=0)} ~ {pts.max(axis=0)}）")
        if len(short):
            parts.append(f"面网格 {len(ext['tri_area'])} 个三角形中 {len(short)} 个的覆盖面积与自身面积不符")
        raise ValueError("体网格外表面没有恰好覆盖输入面网格：" + "；".join(parts)
                         + "。体网格内部有缺口，或修补移动了表面点——按这份网格给出的边界条件是错的。")

    groups, bc_types = _groups_from_faces(ext["names"], ext["owner"], ext["group"], surface_boundaries.bc_types)
    logger.info(
        f"Generated-mesh boundary mapping: {len(host)} exterior faces cover the {len(ext['tri_area'])} input "
        f"surface triangles exactly, {len(groups)} group(s): {[(n, len(c)) for n, c in groups.items()]}"
    )
    return BoundaryMap(groups=groups, bc_types=bc_types)


def label_partial_mesh_boundaries(
    nodes: np.ndarray,
    prism_cells: np.ndarray,
    tet_cells: np.ndarray,
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: 'BoundaryMap',
) -> 'BoundaryMap':
    """调试用局部网格（`--core-only` 等只导出管线某一段）的边界标注：落在输入面网格上的外部面取所在三角形的组，
    其余外部面（与未导出部分之间的界面）的所属单元归入 'INTERFACE'。局部网格本来就不封闭，不做覆盖校验。"""
    from ...schema.grid_boundaries import BoundaryMap

    ext = _classify_exterior_faces(nodes, prism_cells, tet_cells, surface_nodes, surface_faces, surface_boundaries)
    groups, bc_types = _groups_from_faces(ext["names"], ext["owner"], ext["group"], surface_boundaries.bc_types)
    interface = np.unique(ext["owner"][ext["group"] < 0])
    if len(interface):
        groups["INTERFACE"] = interface.astype(np.int32)
        bc_types["INTERFACE"] = "INTERFACE"
    return BoundaryMap(groups=groups, bc_types=bc_types)


def exterior_faces_by_group(volume_mesh) -> Dict[str, np.ndarray]:
    """体网格每个边界组的外部面（节点编号，(n, 3)），逐面判定归属——不经过单元级分组。

    单元级分组（`BoundaryMap.groups` 是单元列表）在角上有歧义：一个单元同时拥有两个组的外部面时出现在两个
    组里，按单元反推边界面会把它的两个面同时写进两个组（cube_demo 导出的体网格因此多出 316 个重复/错组的
    边界面）。这里按面网格逐面判定，判据随网格来源：

    - 本项目生成的网格（`metadata.file_format == "hybrid"`）：外部面精确落在面网格三角形上，几何包含判据
      （与 `map_generated_boundaries` 同一个分类核心），对不上直接报错；
    - 生成管线某一段的调试局部网格（`"partial"`，`--bl-only`/`--core-only`）：同一判据，不在输入表面上的面
      （与未导出部分之间的界面）归入 'INTERFACE'；
    - 外部导入的网格：表面可能被重新三角化，用导入时同一套位置近似匹配（`match_points_to_surface_groups`，
      容差系数 0.75），匹配不上的面归入 'UNCLASSIFIED'——与导入时 `map_boundaries_by_geometry` 的分组一致；
    - 没有面网格数据：只在单元级分组没有歧义（没有单元同时属于多个组）时按单元分组，否则报错。
    """
    nodes = np.column_stack([volume_mesh.nodes.x, volume_mesh.nodes.y, volume_mesh.nodes.z])
    prism = volume_mesh.prism_cells.connectivity if volume_mesh.prism_cells is not None else np.empty((0, 6), np.int64)
    tets = volume_mesh.cells.connectivity
    surface = getattr(volume_mesh, "surface_mesh", None)
    if surface is not None:
        fmt = getattr(volume_mesh.metadata, "file_format", None)
        if fmt in ("hybrid", "partial"):
            ext = _classify_exterior_faces(nodes, prism, tets, surface["nodes"], surface["faces"], surface["boundaries"])
            off = ext["group"] < 0
            if off.any() and fmt == "hybrid":
                raise ValueError(f"{int(off.sum())} 个外部面不在输入面网格上，无法确定边界组")
            out = {name: ext["face_nodes"][ext["group"] == gi] for gi, name in enumerate(ext["names"])
                   if (ext["group"] == gi).any()}
            if off.any():
                out["INTERFACE"] = ext["face_nodes"][off]
            return out
        faces = volume_mesh.ensure_faces_exist()
        bidx = faces.get_boundary_face_indices()
        face_nodes = faces.node_connectivity[bidx].astype(np.int64)
        names, group = match_points_to_surface_groups(
            nodes[face_nodes].mean(axis=1), surface["nodes"], surface["faces"], surface["boundaries"].groups, 0.75)
        out = {name: face_nodes[group == gi] for gi, name in enumerate(names) if (group == gi).any()}
        if (group < 0).any():
            out["UNCLASSIFIED"] = face_nodes[group < 0]
        return out

    faces = volume_mesh.ensure_faces_exist()
    bidx = faces.get_boundary_face_indices()
    owners = faces.connectivity[bidx, 0]
    face_nodes = faces.node_connectivity[bidx].astype(np.int64)
    membership = np.zeros(volume_mesh.cell_count, dtype=np.int64)
    for cells in volume_mesh.boundaries.groups.values():
        membership[np.asarray(cells)] += 1
    if (membership[owners] > 1).any():
        raise ValueError(
            f"{int((membership[owners] > 1).sum())} 个外部面的所属单元同时属于多个边界组，没有面网格数据无法逐面确定"
            "归属（按单元反推会把这些面同时写进多个组）")
    out = {}
    for name, cells in volume_mesh.boundaries.groups.items():
        in_group = np.zeros(volume_mesh.cell_count, dtype=bool)
        in_group[np.asarray(cells)] = True
        out[name] = face_nodes[in_group[owners]]
    return out


def match_points_to_surface_groups(
    points: np.ndarray,
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    surface_groups: Dict[str, np.ndarray],
    distance_tolerance_factor: float,
) -> Tuple[List[str], np.ndarray]:
    """把点（体网格边界面中心）匹配到面网格边界组，两级：

    1. 点落在某个面网格三角形上（几何包含，`_containing_triangle`）——取该三角形的组，这是精确的事实
       （本项目生成的网格外部面恰好是面网格三角形或其细分）；
    2. 其余点（外部工具生成、表面被重新离散的网格）按位置近似：最近的面网格三角形质心（KD-tree），距离不超过
       该三角形外接半径代理量（质心到顶点的最大距离）× `distance_tolerance_factor` 才算匹配——容差随局部
       网格尺度缩放，细密区域更紧。

    只有第 2 级时，边界被细分过的面（子三角形质心离相邻组的三角形质心可能更近）会归错组——这正是先做第 1 级
    的原因。

    Returns:
        (names, group_index)：组名列表（按 `surface_groups` 顺序）与每个点所属组在其中的下标（未匹配为 -1）。
    """
    from scipy.spatial import cKDTree

    names = list(surface_groups.keys())
    tri_idx, code = [], []
    for gi, name in enumerate(names):
        idx = np.asarray(surface_groups[name])
        idx = idx[idx < len(surface_faces)]
        tri_idx.append(idx)
        code.append(np.full(len(idx), gi, dtype=np.int64))
    group_index = np.full(len(points), -1, dtype=np.int64)
    if not tri_idx or sum(len(t) for t in tri_idx) == 0 or len(points) == 0:
        return names, group_index
    tri = surface_nodes[surface_faces[np.concatenate(tri_idx)]]
    code = np.concatenate(code)

    dist_tol = _ON_SURFACE_RELATIVE_TOLERANCE * float(np.ptp(surface_nodes, axis=0).max())
    host = _containing_triangle(points, tri, dist_tol)
    group_index[host >= 0] = code[host[host >= 0]]

    rest = np.flatnonzero(host < 0)
    if len(rest):
        centroids = tri.mean(axis=1)
        radius = np.linalg.norm(tri - centroids[:, None, :], axis=2).max(axis=1)
        dist, nearest = cKDTree(centroids).query(points[rest])
        matched = dist <= np.maximum(radius[nearest] * distance_tolerance_factor, 1e-12)
        group_index[rest[matched]] = code[nearest[matched]]
    return names, group_index


def map_boundaries_by_geometry(
    volume_mesh: 'VolumeMeshData',
    surface_grid: 'GridData',
    distance_tolerance_factor: float = 0.75,
) -> 'BoundaryMap':
    """将边界组属性分配给外部生成的体网格的外部面，
    通过与伴随表面网格边界组的最近质心几何匹配。

    与 map_generated_boundaries（外部面精确落在面网格三角形上——本项目自身的生成管线
    满足此条件，但对其他工具生成的体网格不成立，例如 ANSA 自身的体导出：
    可能重新三角化表面），此函数按位置近似匹配（match_points_to_surface_groups）：
    `volume_mesh` 的每个外部面都与具有最近面的表面边界组匹配。
    不期望精确重合（体网格生成器可能重新三角化/插入 Steiner 点，
    因此体边界面很少与任何单个原始表面面完全相同）——
    仅通过 `distance_tolerance_factor` 门控接近度，
    使得可疑地远离所有表面边界面的面（例如 tetgen/网格生成器
    错误暴露的内部伪影，或文件对不匹配）落入 'UNCLASSIFIED'
    而非被静默错误地归到几何最近的组。

    Args:
        volume_mesh: 外部解析的体网格（例如
            nas_parser_volume.parse_volume_mesh_nas 的输出）——
            如果面尚未计算则调用 `ensure_faces_exist()`。
        surface_grid: 伴随表面网格（NASParser.parse() 的输出）——
            其 `boundaries.groups` 提供 inlet/outlet/wall/... 组
            用于匹配，其 `bc_types` 对任何匹配的组原样继承。
        distance_tolerance_factor: 体边界面的最近表面边界面质心
            必须在该表面面自身外接半径的这些倍数内才算匹配——
            自动随局部网格密度缩放，而非单一固定绝对距离，
            因为精细区域的表面面小得多（因此需要更紧的容差）。

    Returns:
        BoundaryMap 包含 `volume_mesh` 自身全局混合单元约定中的
        单元索引（棱柱 [0, n_prism)，四面体 [n_prism, n_prism + n_tet)——
        见 face_extractor.extract_faces_mixed 的文档字符串），
        与 map_generated_boundaries 的输出相同的约定。
        未匹配的外部面所属单元进入 'UNCLASSIFIED'（WALL）并告警；求解期逐面打标签
        （tag_boundary_groups_by_geometry）对这些面同样匹配不上，求解器会直接报错。
    """
    from ...schema.grid_boundaries import BoundaryMap

    logger.info("Mapping surface boundaries to external volume mesh by geometry...")

    faces = volume_mesh.ensure_faces_exist()
    boundary_face_idx = faces.get_boundary_face_indices()
    if len(boundary_face_idx) == 0:
        logger.warning("External volume mesh has no exterior faces at all - returning empty BoundaryMap")
        return BoundaryMap(groups={}, bc_types={})

    vol_nodes = np.column_stack([volume_mesh.nodes.x, volume_mesh.nodes.y, volume_mesh.nodes.z])
    vol_face_verts = faces.node_connectivity[boundary_face_idx]
    vol_face_centroids = vol_nodes[vol_face_verts].mean(axis=1)
    vol_face_owner = faces.connectivity[boundary_face_idx, 0]

    surf_nodes = np.column_stack([
        surface_grid.nodes.x, surface_grid.nodes.y, surface_grid.nodes.z
    ])
    names, group_index = match_points_to_surface_groups(
        vol_face_centroids, surf_nodes, surface_grid.cells.connectivity,
        surface_grid.boundaries.groups, distance_tolerance_factor,
    )
    if not names:
        logger.warning(
            "Surface mesh has no boundary groups at all - every external "
            "volume mesh exterior face will fall through to UNCLASSIFIED"
        )
        groups = {'UNCLASSIFIED': np.unique(vol_face_owner).astype(np.int32)}
        bc_types = {'UNCLASSIFIED': 'WALL'}
        return BoundaryMap(groups=groups, bc_types=bc_types)
    matched = group_index >= 0

    volume_cell_to_boundary: Dict[int, str] = {}
    for i in np.flatnonzero(matched):
        volume_cell_to_boundary[int(vol_face_owner[i])] = names[int(group_index[i])]

    unique_owners = np.unique(vol_face_owner)
    n_matched_cells = len(volume_cell_to_boundary)
    n_unmatched = len(unique_owners) - n_matched_cells
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(unique_owners)} exterior-face-owning cells matched no "
            f"surface boundary group within tolerance - placed in an 'UNCLASSIFIED' "
            f"group as WALL instead of being silently dropped from every boundary condition"
        )
        for cell_idx in unique_owners:
            if int(cell_idx) not in volume_cell_to_boundary:
                volume_cell_to_boundary[int(cell_idx)] = 'UNCLASSIFIED'

    groups: Dict[str, list] = {}
    bc_types: Dict[str, str] = {}
    for cell_idx, name in volume_cell_to_boundary.items():
        groups.setdefault(name, []).append(cell_idx)
        if name not in bc_types:
            bc_types[name] = surface_grid.boundaries.bc_types.get(name, 'WALL')

    groups_arr = {name: np.array(idx, dtype=np.int32) for name, idx in groups.items()}
    boundaries = BoundaryMap(groups=groups_arr, bc_types=bc_types)
    logger.info(
        f"Geometric boundary mapping completed: {len(groups_arr)} boundary groups, "
        f"{sum(len(c) for c in groups_arr.values())} total cells "
        f"({n_matched_cells} matched by proximity, {max(n_unmatched, 0)} UNCLASSIFIED)"
    )
    return boundaries
