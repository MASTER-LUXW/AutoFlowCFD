"""
AutoFlowCFD - FR 求解器真实单元-面连接关系 (V2.0 Tier-0 基础设施)

本模块是 V2.0 专家评审报告 Tier-0 第2项的实现：为 HighOrderMesh 建立真实的
单元-面拓扑连接关系，取代此前 fr_solver/ 中用「全场单元平均态」冒充相邻
单元、用硬编码 [1,0,0] 冒充界面法向量的伪耦合。

设计思路
--------
1. 复用已存在且经过测试的 `FaceExtractor.extract_faces_mixed`（原本用于
   V1.0 FVM 体网格生成流程的面提取），得到全局面拓扑（owner/邻居 单元、
   面角点全局节点号、法向量、面积、中心）。这是标准的“按排序节点号哈希去重”
   面提取算法，不因为下游是 FR 还是 FVM 而改变，没有必要重新实现一遍。
2. 对每个面，反解出它是 owner（以及邻居，若为内部面）单元的哪一个真实局部面，直接给出原生面编码
   （四面体 `tet_native_v*`、棱柱 `prism_native_f*`），依据 curved_mapping.py 的 TET_CUBE_FACES /
   PRISM_CUBE_FACES 局部顶点表。按单元类型与局部面批量比较节点（向量化），每个面恰好匹配一个局部面，否则报错。
3. 面编码决定该侧的面外插 / 提升算子（`fr/operators/face_ops.py` 的原生算子整表，按面算子索引取）。

2026-10-19（I-124）以前：反解对每个面做 Python 循环、每个候选局部面构造集合比较（百万面级网格秒到分钟级），
且先产出坍缩坐标编码（0~5）再由 `with_native_face_codes(prism_native=...)` 静态翻译（棱柱开关是坍缩棱柱基删除后
的残留）。

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
from ..schema import NodeArray
from ..curved_mapping.curved_mapping import TET_CUBE_FACES, PRISM_CUBE_FACES

# 立方体面标识 -> 整数编码，供 numpy 数组存储（避免存字符串）。
# tet_native_v0~v3（编码 6~9）是四面体路径C（native basis，不经过坍缩
# 坐标，见 fr/native_tet/basis.py 与
# `8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part6/7.md`）
# 专用的面标识——不是新建一套并行枚举，是在现有 0~5（坍缩坐标 6 个
# 立方体面）基础上追加 4 个新编码（四面体只有 4 个真实面，用"被排除
# 的局部顶点下标 0~3"标识，与 fr/face_flux_points/locate.py::locate_native_tet_face_point 的约定一致）。
# 这个推广让绝大部分消费这套编码的
# 下游代码（法向量/面积、周期边界配对、multi-source 分类、numba flat
# 数组、界面残差核函数）不需要感知"这个面是坍缩坐标面还是 native 面"
# 这个区别，只有真正需要区别对待的两处（点位定位、体积->面插值，见
# fr/face_flux_points/locate.py、fr/face_flux_points/geometry.py::build_cross_interp）
# 才分派，见 Part7 文档第一节完整设计说明。
CUBE_FACE_CODES: Dict[str, int] = {
    "a=-1": 0, "a=+1": 1, "b=-1": 2, "b=+1": 3, "c=-1": 4, "c=+1": 5,
    "tet_native_v0": 6, "tet_native_v1": 7, "tet_native_v2": 8, "tet_native_v3": 9,
    # 原生棱柱的 5 个面（2026-09-18）。编码含义见
    # `fr/native_prism/face.py` 模块文档的表：0/1 是两个三角形封盖、
    # 2/3/4 是三个侧四边形。坍缩参考立方体的 `b=+1` 面是退化面（坍缩成
    # 一条侧棱），原生棱柱里没有对应的面，所以只有 5 个。
    "prism_native_f0": 10, "prism_native_f1": 11, "prism_native_f2": 12,
    "prism_native_f3": 13, "prism_native_f4": 14,
}
CUBE_FACE_NAMES: List[str] = [
    "a=-1", "a=+1", "b=-1", "b=+1", "c=-1", "c=+1",
    "tet_native_v0", "tet_native_v1", "tet_native_v2", "tet_native_v3",
    "prism_native_f0", "prism_native_f1", "prism_native_f2",
    "prism_native_f3", "prism_native_f4",
]

#: 原生编码的起点。判据统一写成 `code >= NATIVE_FACE_CODE_BASE`，不要
#: 再散落 `code >= 6` 这种字面量 —— 加了棱柱编码之后那个字面量的含义
#: 从"是 native 四面体面"变成了"是任意 native 面"，两者在需要分派到
#: **不同**算子组的地方不能混用。
NATIVE_FACE_CODE_BASE: int = CUBE_FACE_CODES["tet_native_v0"]
#: 原生四面体面编码区间 `[6, 10)`，原生棱柱面编码区间 `[10, 15)`。
NATIVE_TET_FACE_CODE_RANGE = (CUBE_FACE_CODES["tet_native_v0"],
                              CUBE_FACE_CODES["prism_native_f0"])
NATIVE_PRISM_FACE_CODE_RANGE = (CUBE_FACE_CODES["prism_native_f0"],
                                CUBE_FACE_CODES["prism_native_f4"] + 1)

# 现有坍缩坐标四面体编码 -> native 编码的翻译表——网格拓扑本身（这个面
# 是四面体的哪个真实几何面）不依赖 tet_basis_mode，`build_face_
# connectivity` 按现有方式识别出的 4 种坍缩坐标 (axis,side) 组合与
# 四面体 4 个真实面是固定的一一映射（见 fr/face_flux_points/locate.py::
# _TET_FIXED_TO_FACE_VERTICES：(0,-1.0)排除顶点1、(0,1.0)排除顶点0、
# (1,-1.0)排除顶点2、(2,-1.0)排除顶点3），因此只需要一次静态查找表
# 翻译，不需要重新做任何几何识别——见本文件 `translate_tet_faces_to_native`。
_COLLAPSED_TO_NATIVE_TET_CODE: Dict[int, int] = {
    CUBE_FACE_CODES["a=-1"]: CUBE_FACE_CODES["tet_native_v1"],
    CUBE_FACE_CODES["a=+1"]: CUBE_FACE_CODES["tet_native_v0"],
    CUBE_FACE_CODES["b=-1"]: CUBE_FACE_CODES["tet_native_v2"],
    CUBE_FACE_CODES["c=-1"]: CUBE_FACE_CODES["tet_native_v3"],
}


def _native_face_tables() -> Dict[str, List[Tuple[Tuple[int, ...], int]]]:
    """每种单元的 `[(局部顶点下标, 原生面编码), ...]`。

    局部顶点表沿用 TET_CUBE_FACES / PRISM_CUBE_FACES（以坍缩坐标面名为键），面名 -> 原生编码：四面体按"被排除的
    局部顶点"（`_COLLAPSED_TO_NATIVE_TET_CODE`），棱柱按 `fr/native_prism/face.py::cube_face_to_native_prism_face`
    （唯一事实来源，形状无关的几何验证在那边）。棱柱的坍缩面 `b=+1` 退化成一条侧棱，不在表里。
    """
    from autoflowcfd.fr.native_prism.face import cube_face_to_native_prism_face

    axis_of = {"a": 0, "b": 1, "c": 2}
    tet = [(tuple(idx), _COLLAPSED_TO_NATIVE_TET_CODE[CUBE_FACE_CODES[key]]) for key, idx in TET_CUBE_FACES.items()]
    prism = []
    for key, idx in PRISM_CUBE_FACES.items():
        fid = cube_face_to_native_prism_face(axis_of[key[0]], 1.0 if key.endswith("+1") else -1.0)
        prism.append((tuple(idx), CUBE_FACE_CODES[f"prism_native_f{fid}"]))
    return {"tet": tet, "prism": prism}


class FaceTopologyError(RuntimeError):
    """面-立方体面拓扑反解失败（找不到匹配或匹配歧义）。

    这通常意味着传入的单元连接关系与 curved_mapping 假设的节点顺序约定
    不一致（例如未经过朝向修正），或网格本身存在非流形拓扑缺陷。不做静默
    兜底，直接报错定位到具体单元，避免在错误的面拓扑上继续组装残差。
    """


def _resolve_native_face_codes(cell_nodes: np.ndarray, face_nodes: np.ndarray,
                               table: List[Tuple[Tuple[int, ...], int]]) -> Tuple[np.ndarray, np.ndarray]:
    """一批面在各自单元上的原生面编码（向量化）。

    Args:
        cell_nodes: (n, n_vertices) 每个面对应单元的全局节点号
        face_nodes: (n, 3) 面的三个全局节点号
        table: `_native_face_tables()` 里该单元类型的表

    Returns:
        (codes (n,) int32，n_matches (n,) int32)：三角形局部面要求三个节点相同；棱柱四边形侧面被三角化拆分，
        面的三个节点落在该侧面四个角点内即匹配。`n_matches != 1` 的面由调用方报错。
    """
    codes = np.full(face_nodes.shape[0], -1, dtype=np.int32)
    n_matches = np.zeros(face_nodes.shape[0], dtype=np.int32)
    for local_idx, native_code in table:
        candidate = cell_nodes[:, list(local_idx)]                               # (n, k)
        inside = np.all((face_nodes[:, :, None] == candidate[:, None, :]).any(axis=2), axis=1)
        codes[inside] = native_code
        n_matches += inside
    return codes, n_matches


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
            周期像点的物理坐标——fr/face_flux_points/merge.py 里定位跨
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
    face_translation: np.ndarray

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
    相邻单元"这一情形是 fr/face_flux_points/merge.py 的职责（每个
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
    owner_ids = face_data.connectivity[:, 0]
    neighbor_ids = face_data.connectivity[:, 1]
    face_node_ids = face_data.node_connectivity
    if face_node_ids is None:
        raise RuntimeError("extract_faces_mixed 没有给出面角点（node_connectivity）")
    face_nodes = np.asarray(face_node_ids, dtype=np.int64)
    tables = _native_face_tables()

    def side_codes(cells: np.ndarray, side: str) -> np.ndarray:
        """`cells` 一侧（-1 为无邻居）的原生面编码；匹配数不为 1 的面报错。"""
        codes = np.full(n_faces, -1, dtype=np.int32)
        for kind, lo, hi, conn in (("prism", 0, n_prism, prism_conn), ("tet", n_prism, n_prism + n_tet, tet_conn)):
            sel = np.flatnonzero((cells >= lo) & (cells < hi))
            if sel.size == 0:
                continue
            c, n_match = _resolve_native_face_codes(conn[cells[sel] - lo], face_nodes[sel], tables[kind])
            bad = sel[n_match != 1]
            if bad.size:
                examples = "; ".join(
                    f"面 {int(f)} 节点 {face_nodes[f].tolist()} 单元 {int(cells[f])} 节点 "
                    f"{conn[int(cells[f]) - lo].tolist()} 匹配数 {int(n_match[np.searchsorted(sel, f)])}"
                    for f in bad[:5])
                raise FaceTopologyError(
                    f"{bad.size}/{n_faces} 个面在 {side} 侧找不到唯一对应的局部面（{kind}）：{examples}。单元节点顺序"
                    f"与 curved_mapping 的朝向约定不一致（先对全部单元做 fix_tet_orientation / fix_prism_orientation），"
                    f"或网格有非流形缺陷")
            codes[sel] = c
        return codes

    owner_cube_face = side_codes(owner_ids, "owner")
    neighbor_cube_face = side_codes(neighbor_ids, "neighbor")

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
        face_node_ids=face_node_ids,
        is_boundary=is_boundary,
        face_translation=np.zeros((n_faces, 3), dtype=np.float64),   # 周期配对之后才有非零平移
    )
