"""边界识别与映射模块。

从体网格中识别边界面，并把面网格的边界分组映射到体网格单元上。
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...structures import BoundaryMap, GridData, VolumeMeshData


def identify_boundaries_from_surface(
    volume_cells: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: Optional['BoundaryMap'] = None,
    direct_cell_groups: Optional[np.ndarray] = None,
) -> 'BoundaryMap':
    """从体网格中识别边界面并继承表面边界。

    边界面是属于唯一一个单元的面（外部面）。
    此方法将原始表面边界映射到体网格。

    Args:
        volume_cells: 四面体连接关系, shape=(n_cells, 4)
        surface_faces: 带边界信息的原始表面连接关系
        surface_boundaries: 表面网格的可选边界映射
        direct_cell_groups: 可选 (n_cells,) 字符串数组，直接给出每个单元的
            源边界组名（未知时为空字符串），例如来自
            mesh_background.generate_hybrid_mesh 的 BL 挤出面跟踪。
            优先级高于下方的节点索引匹配，
            后者对 BL 挤出组无效（见 map_surface_boundaries）。

    Returns:
        BoundaryMap 对象，包含已识别的边界组
    """
    from ...schema.grid_boundaries import BoundaryMap
    
    logger.info("Identifying boundary conditions from surface mesh...")

    # 提取四面体的面（每个四面体有 4 个三角形面）
    n_tets = len(volume_cells)
    
    # 向量化生成所有四面体面 - 比循环快得多
    logger.info(f"Extracting faces from {n_tets} tetrahedra (vectorized)...")
    
    # 使用高级索引一次性生成所有面
    face_templates = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=np.int64)
    
    # 生成所有面: shape=(n_tets*4, 3)
    tet_faces = volume_cells[:, face_templates].reshape(-1, 3)
    
    # 生成每个面的单元 ID
    tet_face_cell_ids = np.repeat(np.arange(n_tets), 4)
    
    logger.info(f"Generated {len(tet_faces)} total faces")
    
    # 对每个面内的节点排序，得到可比较的规范形式
    tet_faces_sorted = np.sort(tet_faces, axis=1)

    # 找出只出现一次的面（边界面）——全向量化实现
    logger.info("Finding boundary faces (vectorized)...")
    
    # 转换每行为单个 void 类型以便哈希
    face_dtype = np.dtype((np.void, tet_faces_sorted.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(tet_faces_sorted).view(face_dtype).reshape(-1)
    
    # 用 np.unique 计数出现次数
    unique_faces, inverse_indices, counts = np.unique(
        face_voids, 
        return_inverse=True, 
        return_counts=True
    )
    
    # 边界面恰好出现一次
    boundary_face_mask = counts[inverse_indices] == 1
    boundary_faces = tet_faces[boundary_face_mask]
    boundary_cell_indices = tet_face_cell_ids[boundary_face_mask]
    
    logger.info(
        f"Found {len(boundary_faces)} boundary faces on "
        f"{len(np.unique(boundary_cell_indices))} cells"
    )
    
    # 若提供了表面边界，尝试映射到体网格
    if surface_boundaries is not None and len(surface_boundaries.groups) > 0:
        logger.info(
            f"Inheriting {len(surface_boundaries.groups)} boundary groups "
            f"from surface mesh"
        )
        return map_surface_boundaries(
            boundary_faces, boundary_cell_indices,
            surface_faces, surface_boundaries,
            direct_cell_groups=direct_cell_groups,
        )
    
    # 回退：把所有边界单元创建成单个 "wall" 边界组
    groups = {}
    bc_types = {}
    
    if len(boundary_cell_indices) > 0:
        unique_boundary_cells = np.unique(boundary_cell_indices)
        # 转换为 numpy int32 数组（BoundaryMap 要求）
        groups['wall'] = unique_boundary_cells.astype(np.int32)
        bc_types['wall'] = 'WALL'
        logger.info(f"Created 'wall' boundary group with {len(unique_boundary_cells)} cells")
    
    boundaries = BoundaryMap(groups=groups, bc_types=bc_types)
    logger.info(f"Boundary identification completed: {len(groups)} boundary groups")
    
    return boundaries


def map_surface_boundaries(
    boundary_faces: np.ndarray,
    boundary_cell_indices: np.ndarray,
    surface_faces: np.ndarray,
    surface_boundaries: 'BoundaryMap',
    direct_cell_groups: Optional[np.ndarray] = None,
) -> 'BoundaryMap':
    """将表面网格边界映射到体网格边界单元。

    使用基于节点的匹配来识别哪些体边界单元对应哪些表面边界组。
    基于节点的匹配仅对边界面的节点与输入表面网格完全相同的单元有效；
    它无法匹配 BL 挤出后的面，因为挤出会将节点位移到新的坐标/索引。
    对于这些情况，`direct_cell_groups`（在 BL 挤出期间由
    mesh_domain_classify.classify_boundary_groups / mesh_background 构建）
    直接给出每个单元的源分组，优先于节点匹配。

    Args:
        boundary_faces: 体网格边界面, shape=(n_faces, 3)
        boundary_cell_indices: 每个边界面的单元索引
        surface_faces: 原始表面面, shape=(n_surf_faces, 3)
        surface_boundaries: 表面网格边界映射
        direct_cell_groups: 可选 (n_cells,) 字符串数组，未知时为空；
            优先级高于下方的节点匹配

    Returns:
        BoundaryMap 包含继承的边界组
    """
    from ...schema.grid_boundaries import BoundaryMap

    logger.info("Mapping surface boundaries to volume mesh...")

    # 构建表面面节点到边界组的映射
    # 为效率起见，使用以排序节点元组为键的字典
    surface_face_to_boundary = {}
    for boundary_name, cell_indices in surface_boundaries.groups.items():
        for cell_idx in cell_indices:
            if cell_idx < len(surface_faces):
                face_nodes = tuple(sorted(surface_faces[cell_idx]))
                surface_face_to_boundary[face_nodes] = boundary_name

    # 映射体边界面对到表面边界
    volume_cell_to_boundary = {}  # cell_idx -> boundary_name

    n_direct = 0
    if direct_cell_groups is not None:
        for cell_idx in np.unique(boundary_cell_indices):
            if cell_idx < len(direct_cell_groups):
                name = direct_cell_groups[cell_idx]
                if name:
                    volume_cell_to_boundary[cell_idx] = name
                    n_direct += 1
        if n_direct:
            logger.info(
                f"  {n_direct} boundary cells attributed directly from "
                f"BL-extrusion group tracking"
            )

    for i, face in enumerate(boundary_faces):
        cell_idx = boundary_cell_indices[i]
        if cell_idx in volume_cell_to_boundary:
            continue  # 已经被直接归属过（边界层挤出单元）
        face_key = tuple(sorted(face))
        if face_key in surface_face_to_boundary:
            boundary_name = surface_face_to_boundary[face_key]
            volume_cell_to_boundary[cell_idx] = boundary_name

    # 既没被 direct_cell_groups 归属、也没匹配上节点三元组的边界单元，
    # 以前会从所有分组里静默消失——它在网格里仍然有一个外部面，却完全
    # 没有边界条件，下游（求解器的 BC 处理逻辑）也无从知道原因。改为
    # 把这些单元放进一个显式的兜底分组，让"某些单元找不到 BC"的求解器
    # 配置有一个具体、响亮的原因可查。
    unique_boundary_cells = np.unique(boundary_cell_indices)
    unmatched = np.setdiff1d(unique_boundary_cells, np.fromiter(
        volume_cell_to_boundary.keys(), dtype=np.int64, count=len(volume_cell_to_boundary)
    ), assume_unique=True)
    if len(unmatched) > 0:
        logger.warning(
            f"{len(unmatched)}/{len(unique_boundary_cells)} boundary cells matched "
            f"neither BL-extrusion group tracking nor a surface boundary face "
            f"(likely a remeshed/subdivided face whose nodes no longer match the "
            f"original surface) - placed in an 'UNCLASSIFIED' group as WALL "
            f"instead of being silently dropped from every boundary condition"
        )
        for cell_idx in unmatched:
            volume_cell_to_boundary[int(cell_idx)] = 'UNCLASSIFIED'

    # 按边界名称分组单元
    groups = {}
    bc_types = {}

    for cell_idx, boundary_name in volume_cell_to_boundary.items():
        if boundary_name not in groups:
            groups[boundary_name] = []
        groups[boundary_name].append(cell_idx)
        
        # 继承表面的边界类型
        if boundary_name in surface_boundaries.bc_types:
            bc_types[boundary_name] = surface_boundaries.bc_types[boundary_name]
        else:
            bc_types[boundary_name] = 'WALL'  # 默认
    
    # 转换列表为 numpy 数组
    for boundary_name in groups:
        groups[boundary_name] = np.array(groups[boundary_name], dtype=np.int32)
    
    boundaries = BoundaryMap(groups=groups, bc_types=bc_types)
    logger.info(
        f"Surface boundary mapping completed: {len(groups)} boundary groups, "
        f"{sum(len(cells) for cells in groups.values())} total cells"
    )

    return boundaries


def map_boundaries_by_geometry(
    volume_mesh: 'VolumeMeshData',
    surface_grid: 'GridData',
    distance_tolerance_factor: float = 0.75,
) -> 'BoundaryMap':
    """将边界组属性分配给外部生成的体网格的外部面，
    通过与伴随表面网格边界组的最近质心几何匹配。

    与 map_surface_boundaries（节点索引匹配——仅在两个网格共享
    相同节点编号时正确，本项目自身的生成管线满足此条件，
    但对其他工具生成的体网格不成立，例如 ANSA 自身的体导出：
    其节点 ID 与原始表面 .nas 文件完全无关），此函数按位置匹配：
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
        与 map_surface_boundaries 的输出相同的约定。
        未匹配的外部面所属单元进入 'UNCLASSIFIED'（WALL），
        与 map_surface_boundaries 对自身未匹配情况的回退相同。
    """
    from scipy.spatial import cKDTree
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
    surf_faces = surface_grid.cells.connectivity

    surf_centroids_list = []
    surf_radius_list = []
    surf_group_list = []
    for name, face_idx in surface_grid.boundaries.groups.items():
        face_idx = face_idx[face_idx < len(surf_faces)]
        if len(face_idx) == 0:
            continue
        verts = surf_faces[face_idx]
        pts = surf_nodes[verts]
        centroids = pts.mean(axis=1)
        # 外接半径的代理量：形心到自身 3 个顶点中最远那个的距离——一个
        # 廉价但够用的局部尺度估计（不需要精确外接半径，只要与"这个面
        # 有多大"成正比即可）。
        radius = np.linalg.norm(pts - centroids[:, None, :], axis=2).max(axis=1)
        surf_centroids_list.append(centroids)
        surf_radius_list.append(radius)
        surf_group_list.extend([name] * len(face_idx))

    if not surf_centroids_list:
        logger.warning(
            "Surface mesh has no boundary groups at all - every external "
            "volume mesh exterior face will fall through to UNCLASSIFIED"
        )
        groups = {'UNCLASSIFIED': np.unique(vol_face_owner).astype(np.int32)}
        bc_types = {'UNCLASSIFIED': 'WALL'}
        return BoundaryMap(groups=groups, bc_types=bc_types)

    surf_centroids = np.vstack(surf_centroids_list)
    surf_radius = np.concatenate(surf_radius_list)
    surf_group_arr = np.array(surf_group_list, dtype=object)

    tree = cKDTree(surf_centroids)
    dist, nearest_idx = tree.query(vol_face_centroids)
    tolerance = np.maximum(surf_radius[nearest_idx] * distance_tolerance_factor, 1e-12)
    matched = dist <= tolerance

    volume_cell_to_boundary: Dict[int, str] = {}
    for i in np.flatnonzero(matched):
        volume_cell_to_boundary[int(vol_face_owner[i])] = str(surf_group_arr[nearest_idx[i]])

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
