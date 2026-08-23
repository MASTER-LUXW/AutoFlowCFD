"""
AutoFlowCFD - FR 求解器真实单元-面连接关系 (V2.0 Tier-0 基础设施)

本模块是 V2.0 专家评审报告 Tier-0 第2项的实现：为 HighOrderMesh 建立真实的
单元-面拓扑连接关系，取代此前 fr_solver.py 中用「全场单元平均态」冒充相邻
单元、用硬编码 [1,0,0] 冒充界面法向量的伪耦合。

设计思路
--------
1. 复用已存在且经过测试的 `FaceExtractor.extract_faces_mixed`（原本用于
   V1.0 FVM 体网格生成流程的面提取），得到全局面拓扑（owner/邻居 单元、
   面角点全局节点号、法向量、面积、中心）。这是标准的“按排序节点号哈希去重”
   面提取算法，不因为下游是 FR 还是 FVM 而改变，没有必要重新实现一遍。
2. 对每个面，反解出它在 owner（以及 邻居，若为内部面）单元的计算立方体
   坐标系中对应哪一个局部面（a=-1/a=+1/b=-1/b=+1/c=-1/c=+1 之一），
   依据 curved_mapping.py 中已数值验证的 TET_CUBE_FACES / PRISM_CUBE_FACES
   拓扑表。这一步的正确性已用合成四面体对、棱柱对做过匹配验证（零歧义）。
3. 由「局部立方体面」信息，FR 残差组装阶段即可复用已有的、按 1D 方向做
   Lagrange 外插的 SPs->边界插值算子（fr/matrix_operators.py），把体内
   SPs 的解外插到该面的 Flux 点 上，不需要为单纯形重新推导专用的
   插值矩阵。

单元朝向约定：本模块假设传入的 prism_connectivity / tet_connectivity 已经
过 curved_mapping.fix_tet_orientation / fix_prism_orientation 处理，节点
顺序对应正体积；若未处理，立方体面拓扑表的局部索引仍然成立（拓扑关系与
朝向无关），但物理映射本身的 Jacobian 检查会在 HighOrderMesh 阶段报错。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..mesh_gen.extraction.face_extractor import FaceExtractor
from ..structures import NodeArray
from ..curved_mapping.curved_mapping import TET_CUBE_FACES, PRISM_CUBE_FACES

# 立方体面标识 -> 整数编码，供 numpy 数组存储（避免存字符串）
CUBE_FACE_CODES: Dict[str, int] = {"a=-1": 0, "a=+1": 1, "b=-1": 2, "b=+1": 3, "c=-1": 4, "c=+1": 5}
CUBE_FACE_NAMES: List[str] = ["a=-1", "a=+1", "b=-1", "b=+1", "c=-1", "c=+1"]


class FaceTopologyError(RuntimeError):
    """面-立方体面拓扑反解失败（找不到匹配或匹配歧义）。

    这通常意味着传入的单元连接关系与 curved_mapping 假设的节点顺序约定
    不一致（例如未经过朝向修正），或网格本身存在非流形拓扑缺陷。不做静默
    兜底，直接报错定位到具体单元，避免在错误的面拓扑上继续组装残差。
    """


def _resolve_cube_face(
    cell_node_ids: np.ndarray, face_node_ids: np.ndarray, cube_face_table: Dict[str, Tuple[int, ...]]
) -> str:
    """给定单元的局部节点号数组与某个面的全局节点号(3个)，反解该面对应的立方体面标识。"""
    face_set = set(int(n) for n in face_node_ids)
    matches = []
    for key, local_idx in cube_face_table.items():
        candidate = set(int(cell_node_ids[i]) for i in local_idx)
        if len(local_idx) == 3:
            if candidate == face_set:
                matches.append(key)
        else:
            # 棱柱四边形侧面：只有 4 个角点里的某一个对角线三角形会等于 face_set
            if face_set.issubset(candidate):
                matches.append(key)
    if len(matches) != 1:
        raise FaceTopologyError(
            f"Ambiguous or missing cube-face match: face_nodes={face_node_ids}, "
            f"cell_nodes={cell_node_ids}, matches={matches}"
        )
    return matches[0]


@dataclass
class FRFaceConnectivity:
    """FR 求解器使用的面连接关系（在全局单元索引空间中，棱柱在前、四面体在后，
    与 HighOrderMesh.load_from_volume_mesh 的 cell_idx 编号约定一致）。

    属性:
        owner_cell: (n_faces,) int32，owner 单元全局索引
        neighbor_cell: (n_faces,) int32，neighbor 单元全局索引，边界面为 -1
        owner_cube_face: (n_faces,) int32，owner 侧的立方体局部面编码（见 CUBE_FACE_CODES）
        neighbor_cube_face: (n_faces,) int32，neighbor 侧局部面编码，边界面为 -1
        normal: (n_faces, 3) float64，单位法向量，方向由 owner 指向 neighbor（边界面指向域外）
        area: (n_faces,) float64，面的物理面积
        center: (n_faces, 3) float64，面中心物理坐标
        face_node_ids: (n_faces, 3) int32，面角点全局节点号（用于边界组匹配）
        is_boundary: (n_faces,) bool
        face_translation: (n_faces, 3) float64，周期边界配对面的平移向量（见
            pair_periodic_boundary_faces 文档），非周期面恒为零向量。方向
            约定：把 owner 侧面上一点加上这个向量，得到 邻居 侧对应
            周期像点的物理坐标——fr/face_flux_points_merge.py 里定位跨
            单元 Flux 点 时，owner->邻居 方向的搜索目标点要*减去*
            这个向量（因为周期面物理上不重合，不能直接用 owner 的物理坐标
            去 邻居 单元里找，必须先按周期平移量对齐），邻居->owner
            方向则反号（加上这个向量）。
    """

    owner_cell: np.ndarray
    neighbor_cell: np.ndarray
    owner_cube_face: np.ndarray
    neighbor_cube_face: np.ndarray
    normal: np.ndarray
    area: np.ndarray
    center: np.ndarray
    face_node_ids: np.ndarray
    is_boundary: np.ndarray
    face_translation: np.ndarray = None

    def __post_init__(self):
        if self.face_translation is None:
            self.face_translation = np.zeros((self.n_faces, 3), dtype=np.float64)

    @property
    def n_faces(self) -> int:
        return self.owner_cell.shape[0]

    def get_boundary_face_indices(self) -> np.ndarray:
        return np.flatnonzero(self.is_boundary)

    def get_interior_face_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.is_boundary)


def build_face_connectivity(
    prism_connectivity: Optional[np.ndarray],
    tet_connectivity: Optional[np.ndarray],
    nodes: np.ndarray,
) -> FRFaceConnectivity:
    """构建 HighOrderMesh 的真实单元-面连接关系。

    棱柱的四边形侧面会被 FaceExtractor 恒定三角化拆分成 2 个子面记录
    （即使相邻的也是同一个棱柱的单一四边形邻居）；本函数如实返回这些
    原始记录，不做任何去重/合并——正确处理"1 个立方体面对应 1~2 个真实
    相邻单元"这一情形是 fr/face_flux_points_merge.py 的职责（每个
    (cell,立方体面) 分组只让一条记录触发一次自身外插+校正投影，其余
    记录仅贡献跨单元插值信息），不应该在更底层的拓扑构建阶段就丢弃或
    报错——那样反而丢失了"这条记录到底对应四边形哪一半"的信息。

    Args:
        prism_connectivity: (n_prism, 6) int32 或 None，节点顺序 (v0,v1,v2,w0,w1,w2)，
            已经过 fix_prism_orientation 处理
        tet_connectivity: (n_tet, 4) int32 或 None，已经过 fix_tet_orientation 处理
        nodes: (n_nodes, 3) float64 物理坐标

    Returns:
        FRFaceConnectivity，单元全局索引约定：棱柱 [0, n_prism)，
        四面体 [n_prism, n_prism + n_tet)（与 HighOrderMesh.load_from_volume_mesh 一致）
    """
    n_prism = 0 if prism_connectivity is None else len(prism_connectivity)
    n_tet = 0 if tet_connectivity is None else len(tet_connectivity)
    prism_conn = (
        prism_connectivity.astype(np.int32)
        if prism_connectivity is not None
        else np.zeros((0, 6), dtype=np.int32)
    )
    tet_conn = (
        tet_connectivity.astype(np.int32) if tet_connectivity is not None else np.zeros((0, 4), dtype=np.int32)
    )

    node_arr = NodeArray.from_array(nodes)

    logger.info(f"Building FR face connectivity: {n_prism} prisms, {n_tet} tets...")
    face_data = FaceExtractor.extract_faces_mixed(prism_conn, tet_conn, node_arr, strict=True)

    n_faces = face_data.count
    owner_cube_face = np.full(n_faces, -1, dtype=np.int32)
    neighbor_cube_face = np.full(n_faces, -1, dtype=np.int32)

    owner_ids = face_data.connectivity[:, 0]
    neighbor_ids = face_data.connectivity[:, 1]

    def cell_conn(cell_id: int) -> Tuple[np.ndarray, Dict[str, Tuple[int, ...]]]:
        if cell_id < n_prism:
            return prism_conn[cell_id], PRISM_CUBE_FACES
        return tet_conn[cell_id - n_prism], TET_CUBE_FACES

    n_ambiguous = 0
    for i in range(n_faces):
        owner = int(owner_ids[i])
        face_nodes = face_data.node_connectivity[i]
        owner_conn, owner_table = cell_conn(owner)
        try:
            owner_key = _resolve_cube_face(owner_conn, face_nodes, owner_table)
            owner_cube_face[i] = CUBE_FACE_CODES[owner_key]
        except FaceTopologyError as e:
            n_ambiguous += 1
            if n_ambiguous <= 5:
                logger.error(f"Face {i} owner-side topology resolution failed: {e}")
            continue

        neighbor = int(neighbor_ids[i])
        if neighbor >= 0:
            neighbor_conn, neighbor_table = cell_conn(neighbor)
            try:
                neighbor_key = _resolve_cube_face(neighbor_conn, face_nodes, neighbor_table)
                neighbor_cube_face[i] = CUBE_FACE_CODES[neighbor_key]
            except FaceTopologyError as e:
                n_ambiguous += 1
                if n_ambiguous <= 5:
                    logger.error(f"Face {i} neighbor-side topology resolution failed: {e}")

    if n_ambiguous > 0:
        raise FaceTopologyError(
            f"{n_ambiguous}/{n_faces} faces failed cube-face topology resolution. "
            f"This indicates cell node ordering does not match the orientation "
            f"convention assumed by curved_mapping.py (run fix_tet_orientation/"
            f"fix_prism_orientation on all cells before building face connectivity), "
            f"or a non-manifold mesh defect."
        )

    is_boundary = neighbor_ids < 0

    logger.info(
        f"FR face connectivity built: {n_faces} faces "
        f"({np.sum(~is_boundary)} interior, {np.sum(is_boundary)} boundary)"
    )

    return FRFaceConnectivity(
        owner_cell=owner_ids.astype(np.int32),
        neighbor_cell=neighbor_ids.astype(np.int32),
        owner_cube_face=owner_cube_face,
        neighbor_cube_face=neighbor_cube_face,
        normal=face_data.normal,
        area=face_data.area,
        center=face_data.center,
        face_node_ids=face_data.node_connectivity,
        is_boundary=is_boundary,
    )


def tag_boundary_groups(
    face_conn: FRFaceConnectivity, boundary_groups: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """将边界面按所属边界组（WALL/INLET/OUTLET/...）打标签。

    Args:
        face_conn: build_face_connectivity 的输出
        boundary_groups: BoundaryMap.groups，name -> owner 单元全局索引数组
            （沿用 grid/schema/grid_boundaries.py 的既有约定：棱柱在前、
            四面体在后的同一套全局单元索引空间）

    Returns:
        (group_code: (n_faces,) int32，边界面对应组的整数编码，内部面/未匹配为 -1；
         name_to_code: Dict[str,int]，组名到编码的映射)

    真实 bug（已确认，未修复，2026-08-23 用户直接追问"tag_boundary_groups
    修复了吗"促成排查）：按 owner **单元**索引匹配 boundary_groups——若某个
    单元的两个不同边界面分别属于两个不同的组（角单元/边界 patch 交界处），
    这里对该单元只可能命中*一个*组（`np.isin` 按整个 cell_id 集合匹配，
    最后一次循环覆盖前面的），那个单元全部边界面都会被打上同一个组标签，
    即使其中某个面物理上明明属于另一个组。cube_demo 网格的四个边界组
    （tunnel/outlet/inlet/body）互不重叠，未触发此问题，因此当前不是那次
    真实网格 P1 发散的原因（已用实际数据核实排除），但这仍是一个真实、
    未修复的正确性缺陷，触发条件是任意存在角单元的网格。真正的修复是
    `tag_boundary_groups_by_geometry`（按面本身的物理位置几何最近邻匹配，
    不经过单元级别聚合）+ `tag_boundary_groups_for_mesh`（自动选择：有
    `mesh.boundary_surface_mesh` 时用前者，否则回退到本函数）——本函数
    保留作为没有原始面网格数据可用时（例如从 .pkl 加载、或本项目自己
    `grid generate-volume` 的自包含体网格路径）的后备实现，不删除。
    """
    n_faces = face_conn.n_faces
    group_code = np.full(n_faces, -1, dtype=np.int32)
    name_to_code: Dict[str, int] = {}

    boundary_idx = face_conn.get_boundary_face_indices()
    boundary_owners = face_conn.owner_cell[boundary_idx]

    n_unmatched = 0
    for code, (name, cell_ids) in enumerate(boundary_groups.items()):
        name_to_code[name] = code
        cell_id_set = np.asarray(cell_ids)
        mask = np.isin(boundary_owners, cell_id_set)
        group_code[boundary_idx[mask]] = code

    n_unmatched = int(np.sum(group_code[boundary_idx] < 0))
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(boundary_idx)} boundary faces did not match any "
            f"boundary_groups entry (owner cell not found in any group's cell-index "
            f"list) - these faces will not receive a weak BC penalty term unless "
            f"handled by a default/fallback boundary condition."
        )

    return group_code, name_to_code


def tag_boundary_groups_by_geometry(
    face_conn: FRFaceConnectivity,
    surface_mesh: Dict[str, object],
    distance_tolerance_factor: float = 0.75,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """按真实几何最近邻，逐**面**（不经过任何单元级别聚合）匹配边界组标签。

    修复 `tag_boundary_groups` 的单元级别聚合缺陷（见该函数文档）：直接
    对 `face_conn` 自己的每个边界面中心 `center`，用与
    `mesh_gen/utils/mesh_boundary.py::map_boundaries_by_geometry` 完全
    相同的算法（KD-tree 最近邻 + 按匹配到的表面面自身外接半径缩放的
    容差）在原始表面网格的边界组三角面片里找最近邻——同一个单元的两个
    边界面各自独立判定自己的归属，不会互相覆盖。

    Args:
        face_conn: build_face_connectivity 的输出（要打标签的真正对象）
        surface_mesh: VolumeMeshData.surface_mesh 字典（
            `{'nodes': (n,3), 'faces': (m,3) 三角形连接, 'boundaries': BoundaryMap}`，
            见 mesh_external_import.import_external_volume_mesh 的写入处，
            'boundaries'.groups 是原始表面网格的组名->三角面片索引，与
            `map_boundaries_by_geometry` 用的是同一份数据）
        distance_tolerance_factor: 与 map_boundaries_by_geometry 同名参数
            语义一致

    Returns:
        (group_code, name_to_code)，与 tag_boundary_groups 完全相同的
        返回值约定，可直接替换该函数在所有调用点的返回值
    """
    from scipy.spatial import cKDTree

    n_faces = face_conn.n_faces
    group_code = np.full(n_faces, -1, dtype=np.int32)
    name_to_code: Dict[str, int] = {}

    boundary_idx = face_conn.get_boundary_face_indices()
    if len(boundary_idx) == 0:
        return group_code, name_to_code

    surf_nodes = surface_mesh["nodes"]
    surf_faces = surface_mesh["faces"]
    surf_boundaries = surface_mesh["boundaries"]

    surf_centroids_list: List[np.ndarray] = []
    surf_radius_list: List[np.ndarray] = []
    surf_group_code_list: List[np.ndarray] = []
    for code, (name, face_idx) in enumerate(surf_boundaries.groups.items()):
        name_to_code[name] = code
        face_idx = np.asarray(face_idx)
        face_idx = face_idx[face_idx < len(surf_faces)]
        if len(face_idx) == 0:
            continue
        verts = surf_faces[face_idx]
        pts = surf_nodes[verts]
        centroids = pts.mean(axis=1)
        # 外接半径代理量：质心到自己 3 个顶点的最大距离——跟
        # map_boundaries_by_geometry 用同一个近似，不需要精确外接半径，
        # 只需要一个跟"这个面有多大"成正比的局部尺度。
        radius = np.linalg.norm(pts - centroids[:, None, :], axis=2).max(axis=1)
        surf_centroids_list.append(centroids)
        surf_radius_list.append(radius)
        surf_group_code_list.append(np.full(len(face_idx), code, dtype=np.int32))

    if not surf_centroids_list:
        return group_code, name_to_code

    surf_centroids = np.vstack(surf_centroids_list)
    surf_radius = np.concatenate(surf_radius_list)
    surf_group_code = np.concatenate(surf_group_code_list)

    face_centroids = face_conn.center[boundary_idx]
    tree = cKDTree(surf_centroids)
    dist, nearest_idx = tree.query(face_centroids)
    tolerance = np.maximum(surf_radius[nearest_idx] * distance_tolerance_factor, 1e-12)
    matched = dist <= tolerance

    group_code[boundary_idx[matched]] = surf_group_code[nearest_idx[matched]]

    n_unmatched = int(np.sum(~matched))
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(boundary_idx)} boundary faces (face-level geometric "
            f"match) did not match any surface boundary group within tolerance - these "
            f"faces will not receive a weak BC penalty term unless handled by a "
            f"default/fallback boundary condition."
        )

    return group_code, name_to_code


def tag_boundary_groups_for_mesh(mesh, face_conn: Optional[FRFaceConnectivity] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """所有调用点的统一入口：优先用逐面几何匹配，缺少原始面网格数据时
    回退到单元级别匹配。

    真实网格数据（cube_demo）已验证：`mesh.boundary_surface_mesh`（
    HighOrderMesh.load_from_volume_mesh 从 VolumeMeshData.surface_mesh
    透传，input_file 是 .nas 体网格 + --surface-mesh 反推边界的路径才有）
    可用时走 `tag_boundary_groups_by_geometry`；不可用时（例如从 .pkl
    加载——边界分组已经在 pkl 里但原始面网格数据没有一起序列化）回退到
    `tag_boundary_groups`。四处调用点（fr_solver/boundary.py 的 BC 幽灵态
    构建、core/utils/solver_helpers.py 的 WMLES 壁面剪应力、
    postprocess/fr_coefficients.py 的两处气动系数积分）此前各自重复同一段
    `tag_boundary_groups(fc, mesh.boundary_groups or {})`，都受同一个
    单元级别聚合缺陷影响（气动系数积分场景下，被错误纳入/排除的 WALL
    面直接污染 Cd/Cl）——统一到这一个入口，行为对四处调用点完全一致。

    Args:
        mesh: HighOrderMesh 实例
        face_conn: 默认使用 mesh.face_connectivity；单独传入用于尚未把
            结果挂回 mesh 的构造中途场景

    Returns:
        (group_code, name_to_code)，与 tag_boundary_groups 完全相同的
        返回值约定
    """
    fc = face_conn if face_conn is not None else mesh.face_connectivity
    surface_mesh = getattr(mesh, "boundary_surface_mesh", None)
    if surface_mesh is not None:
        return tag_boundary_groups_by_geometry(fc, surface_mesh)
    return tag_boundary_groups(fc, mesh.boundary_groups or {})


def pair_periodic_boundary_faces(
    face_conn: FRFaceConnectivity,
    boundary_groups: Dict[str, np.ndarray],
    group_a: str,
    group_b: str,
    translation: np.ndarray,
    tol_scale: float = 1e-6,
) -> FRFaceConnectivity:
    """把 `group_a`/`group_b` 两组边界面按几何位置一一配对，合并成周期
    内部面（`is_boundary=False`），必须在 `build_face_flux_points` 之前
    调用（配对结果要参与它的 owner/邻居 分组，见模块文档
    face_translation 字段说明）。

    周期面物理上不重合（相差一个平移向量 translation：group_a 上一点
    + translation = group_b 上对应点），FaceExtractor 按共享节点号去重
    时天然找不到这种配对关系（节点号完全不同），只能在这里作为拓扑
    构建的下一步、按几何位置显式配对。

    每一对匹配的 (face_a, face_b) 合并成*一条*内部面记录（复用 face_a
    的记录位置：neighbor_cell/neighbor_cube_face 借用 face_b 的
    owner_cell/owner_cube_face，因为 face_b 在它自己的 owner 单元里
    已经独立解出了正确的局部立方体面），face_b 的记录被丢弃——这与
    普通内部面"一条记录同时描述 owner 和 邻居"的约定完全一致，不
    是"一个物理面两条记录"。

    Args:
        face_conn: build_face_connectivity 的输出（尚未做任何周期配对）
        boundary_groups: BoundaryMap.groups，与 tag_boundary_groups 同源。
            注意：`BoundaryMap.groups` 是按**单元**（owner 单元 全局索引）
            记录组成员关系的（见 grid/schema/grid_boundaries.py），而不是
            按**面**。对于"角点单元"（同一个单元同时有多个边界面分属不同
            边界组，例如一个既贴周期面又贴壁面的单元），仅凭单元 ID 筛选
            会把该单元的其它边界面（如壁面）误当作候选周期面混入——必须
            在下面用面法向量方向做二次几何筛选剔除，见函数体注释。
        group_a, group_b: 待配对的两个边界组名
        translation: (3,) 从 group_a 到 group_b 的平移向量。约定：
            group_a/group_b 所在的边界面必须是与 translation 垂直的平面
            （这是"平移周期性"本身的数学定义——只有法向量与平移方向平行
            的两组平直边界面，才可能通过纯平移一一重合），本函数据此用
            面法向量与 translation 方向的对齐程度识别真正落在该周期面上
            的候选面。
        tol_scale: 匹配容差相对于平移向量模长的比例

    Returns:
        新的 FRFaceConnectivity，group_a/group_b 的边界面已合并为内部面，
        其余面不变；总面数减少（配对成功的每一对减少 1 条记录）

    抛出异常:
        FaceTopologyError: 两组面数量不一致、存在无法在容差内匹配到对应
            点的面、或存在法向量与 translation 方向既不明显平行也不明显
            正交的"暧昧"候选面——都不做静默丢弃/兜底，意味着网格在周期
            方向上不是真正平直共形周期的，必须先修好网格。
    """
    from scipy.spatial import cKDTree

    translation = np.asarray(translation, dtype=np.float64)
    translation_norm = float(np.linalg.norm(translation))
    if translation_norm < 1e-300:
        raise ValueError(f"Periodic translation vector must be nonzero, got {translation.tolist()}")
    translation_dir = translation / translation_norm

    boundary_idx = face_conn.get_boundary_face_indices()
    boundary_owners = face_conn.owner_cell[boundary_idx]

    cand_a = boundary_idx[np.isin(boundary_owners, np.asarray(boundary_groups[group_a]))]
    cand_b = boundary_idx[np.isin(boundary_owners, np.asarray(boundary_groups[group_b]))]

    # 见上方 Args 说明：按 owner 单元筛出的候选面里可能混入了角点单元
    # 的其它边界面（例如同时贴壁面的周期角点单元，其壁面记录也会因为
    # "owner 单元属于 group_a" 被 np.isin 选中）。周期面法向量必须与
    # translation 方向平行，用这个纯几何约束二次过滤：明显不平行（接近
    # 正交）的直接判定为"该单元的其它边界面误入候选集"予以剔除；
    # 既不明显平行也不明显正交的暧昧情况直接报错，不允许靠阈值猜测。
    normal_align_parallel = 1.0 - 1e-6
    normal_align_ambiguous_floor = 0.05

    def _filter_on_periodic_plane(cand_idx: np.ndarray, group_name: str) -> np.ndarray:
        if len(cand_idx) == 0:
            return cand_idx
        align = np.abs(face_conn.normal[cand_idx] @ translation_dir)
        on_plane = align >= normal_align_parallel
        ambiguous = (~on_plane) & (align > normal_align_ambiguous_floor)
        if np.any(ambiguous):
            bad_local = np.flatnonzero(ambiguous)
            raise FaceTopologyError(
                f"Periodic group '{group_name}': {len(bad_local)} candidate boundary "
                f"face(s) have a normal that is neither clearly parallel nor clearly "
                f"orthogonal to the translation direction {translation_dir.tolist()} "
                f"(|cos θ| values: {sorted(np.round(align[bad_local], 4).tolist())}). "
                f"This means the periodic boundary plane is not flat/perpendicular to "
                f"the translation vector, or the mesh has a genuine non-conformity here "
                f"- cannot silently classify these faces as in-plane or not."
            )
        return cand_idx[on_plane]

    idx_a = _filter_on_periodic_plane(cand_a, group_a)
    idx_b = _filter_on_periodic_plane(cand_b, group_b)

    if len(idx_a) != len(idx_b):
        raise FaceTopologyError(
            f"Periodic pairing '{group_a}'<->'{group_b}' face count mismatch: "
            f"{len(idx_a)} vs {len(idx_b)} - mesh is not conforming across the "
            f"periodic planes (must have matching face tessellation on both sides)."
        )

    tol = tol_scale * max(float(np.linalg.norm(translation)), 1.0)
    centers_a_shifted = face_conn.center[idx_a] + translation[np.newaxis, :]
    tree_b = cKDTree(face_conn.center[idx_b])
    dist, match_in_b = tree_b.query(centers_a_shifted, k=1)

    if len(set(match_in_b.tolist())) != len(idx_b) or np.any(dist > tol):
        bad = np.flatnonzero(dist > tol)
        raise FaceTopologyError(
            f"Periodic pairing '{group_a}'<->'{group_b}' failed to geometrically "
            f"match every face within tolerance {tol:.3e} (translation={translation.tolist()}): "
            f"{len(bad)}/{len(idx_a)} faces unmatched or duplicated. Mesh tessellation must be "
            f"identical (node-for-node congruent up to the translation) on both periodic planes."
        )

    faces_b_matched = idx_b[match_in_b]

    owner_cell = face_conn.owner_cell.copy()
    neighbor_cell = face_conn.neighbor_cell.copy()
    owner_cube_face = face_conn.owner_cube_face.copy()
    neighbor_cube_face = face_conn.neighbor_cube_face.copy()
    is_boundary = face_conn.is_boundary.copy()
    face_translation = face_conn.face_translation.copy()

    neighbor_cell[idx_a] = face_conn.owner_cell[faces_b_matched]
    neighbor_cube_face[idx_a] = face_conn.owner_cube_face[faces_b_matched]
    is_boundary[idx_a] = False
    face_translation[idx_a] = translation

    keep_mask = np.ones(face_conn.n_faces, dtype=bool)
    keep_mask[faces_b_matched] = False

    logger.info(
        f"Paired {len(idx_a)} periodic boundary faces ('{group_a}' <-> '{group_b}', "
        f"translation={translation.tolist()}) into interior faces."
    )

    return FRFaceConnectivity(
        owner_cell=owner_cell[keep_mask],
        neighbor_cell=neighbor_cell[keep_mask],
        owner_cube_face=owner_cube_face[keep_mask],
        neighbor_cube_face=neighbor_cube_face[keep_mask],
        normal=face_conn.normal[keep_mask],
        area=face_conn.area[keep_mask],
        center=face_conn.center[keep_mask],
        face_node_ids=face_conn.face_node_ids[keep_mask],
        is_boundary=is_boundary[keep_mask],
        face_translation=face_translation[keep_mask],
    )


def apply_periodic_pairing_from_boundary_map(face_conn: FRFaceConnectivity, boundary_map) -> FRFaceConnectivity:
    """扫描 `boundary_map`（grid/schema/grid_boundaries.py::BoundaryMap）里
    所有 `bc_type=='PERIODIC'` 的边界组，按 `parameters[name]` 里的
    `paired_with`/`translation` 逐对调用 `pair_periodic_boundary_faces`。

    `parameters[name]` 约定（写入方：boundary/config.py 的 YAML 手动/
    混合模式配置合并逻辑，或调用方直接构造 BoundaryMap 时手工填入）：
        {"paired_with": "<另一侧边界组名>", "translation": [tx,ty,tz]}
    只需在配对两侧之一填写（另一侧若也标了 PERIODIC 但没填 参数，
    仍会被从已处理一侧正确配对、跳过重复处理；若两侧都填了，要求
    互相指向对方且平移向量互为相反数，否则报错——避免配置自相矛盾时
    静默按其中一侧为准）。

    Args:
        face_conn: build_face_connectivity 的输出
        boundary_map: 提供 .bc_types / .groups / .get_parameters() 的对象
            （BoundaryMap 实例，或具备同名接口的对象）

    Returns:
        完成全部周期配对后的 FRFaceConnectivity；若没有任何 PERIODIC 组，
        原样返回 face_conn（不做拷贝）
    """
    periodic_names = [name for name, t in boundary_map.bc_types.items() if t == "PERIODIC"]
    if not periodic_names:
        return face_conn

    processed = set()
    for name in periodic_names:
        if name in processed:
            continue
        params = boundary_map.get_parameters(name)
        paired_with = params.get("paired_with")
        translation = params.get("translation")
        if paired_with is None or translation is None:
            raise ValueError(
                f"Boundary group '{name}' has bc_type=PERIODIC but is missing "
                f"'paired_with'/'translation' in its parameters - periodic groups "
                f"must specify both (see apply_periodic_pairing_from_boundary_map docs)."
            )
        if paired_with not in boundary_map.bc_types or boundary_map.bc_types[paired_with] != "PERIODIC":
            raise ValueError(
                f"Boundary group '{name}' is paired with '{paired_with}', but that group "
                f"either does not exist or is not itself tagged bc_type=PERIODIC."
            )
        other_params = boundary_map.get_parameters(paired_with)
        if other_params.get("paired_with") not in (None, name):
            raise ValueError(
                f"Periodic pairing mismatch: '{name}' points to '{paired_with}', but "
                f"'{paired_with}' points to '{other_params.get('paired_with')}' instead of back to '{name}'."
            )
        other_translation = other_params.get("translation")
        if other_translation is not None and not np.allclose(
            np.asarray(other_translation, dtype=np.float64), -np.asarray(translation, dtype=np.float64)
        ):
            raise ValueError(
                f"Periodic pairing translation mismatch between '{name}' ({translation}) "
                f"and '{paired_with}' ({other_translation}) - they must be exact opposites."
            )

        face_conn = pair_periodic_boundary_faces(
            face_conn, boundary_map.groups, name, paired_with, np.asarray(translation, dtype=np.float64)
        )
        processed.add(name)
        processed.add(paired_with)

    return face_conn
