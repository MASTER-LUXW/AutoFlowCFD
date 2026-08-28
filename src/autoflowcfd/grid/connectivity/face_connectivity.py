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


# 以下四个函数（边界组标签打标 + 周期边界配对）已拆分到独立文件以控制
# 单文件行数（本文件此前 675 行，超过 600 行硬性阈值），这里保留原符号名
# 的 re-export，所有现有调用点（fr_solver/boundary.py 等）的导入路径
# `from autoflowcfd.grid.connectivity.face_connectivity import X` 不受影响。
from .face_connectivity_boundary_tags import (  # noqa: E402,F401
    tag_boundary_groups,
    tag_boundary_groups_by_geometry,
    tag_boundary_groups_for_mesh,
)
from .face_connectivity_periodic import (  # noqa: E402,F401
    pair_periodic_boundary_faces,
    apply_periodic_pairing_from_boundary_map,
)
