"""
AutoFlowCFD - 周期边界面配对 (从 face_connectivity.py 拆出，控制单文件行数)

把两组周期边界面按几何位置一一配对，合并成周期内部面，供
HighOrderMesh.load_from_volume_mesh 在构建 face_flux_points 之前调用。
"""

from typing import Dict

import numpy as np
from loguru import logger

from .face_connectivity import FaceTopologyError, FRFaceConnectivity


def pair_periodic_boundary_faces(
    face_conn: FRFaceConnectivity,
    boundary_faces: Dict[str, np.ndarray],
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
        boundary_faces: 逐面边界组 `BoundaryMap.faces`（体网格节点编号的三角形），两组的面按节点三元组精确取出
            （2026-10-22 I-130 以前按 owner 单元筛选，角点单元的其它边界面（如壁面）会混进候选，要靠法向二次筛选）。
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

    from .face_connectivity_boundary_tags import face_group_codes

    boundary_idx = face_conn.get_boundary_face_indices()
    codes, _ = face_group_codes(face_conn.face_node_ids[boundary_idx],
                                {group_a: boundary_faces[group_a], group_b: boundary_faces[group_b]})

    # 平移周期性的定义：周期面是与平移向量垂直的平面，两组的每个面法向都必须与平移方向平行
    normal_align_parallel = 1.0 - 1e-6

    def _on_periodic_plane(idx: np.ndarray, group_name: str) -> np.ndarray:
        align = np.abs(face_conn.normal[idx] @ translation_dir)
        off = np.flatnonzero(align < normal_align_parallel)
        if off.size:
            raise FaceTopologyError(
                f"Periodic group '{group_name}': {off.size} face(s) have a normal not parallel to the translation "
                f"direction {translation_dir.tolist()} (|cos θ| values: "
                f"{sorted(np.round(align[off], 4).tolist())[:10]}) - the periodic boundary plane is not "
                f"flat/perpendicular to the translation vector.")
        return idx

    idx_a = _on_periodic_plane(boundary_idx[codes == 0], group_a)
    idx_b = _on_periodic_plane(boundary_idx[codes == 1], group_b)

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

    `parameters[name]` 约定（由直接构造 BoundaryMap 的调用方填入，例如 `tests/validation/_periodic_mesh.py`；
    CLI 不提供周期边界配置）：
        {"paired_with": "<另一侧边界组名>", "translation": [tx,ty,tz]}
    只需在配对两侧之一填写（另一侧若也标了 PERIODIC 但没填 参数，
    仍会被从已处理一侧正确配对、跳过重复处理；若两侧都填了，要求
    互相指向对方且平移向量互为相反数，否则报错——避免配置自相矛盾时
    静默按其中一侧为准）。

    Args:
        face_conn: build_face_connectivity 的输出
        boundary_map: 提供 .bc_types / .faces / .get_parameters() 的对象
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
            face_conn, boundary_map.faces, name, paired_with, np.asarray(translation, dtype=np.float64)
        )
        processed.add(name)
        processed.add(paired_with)

    return face_conn
