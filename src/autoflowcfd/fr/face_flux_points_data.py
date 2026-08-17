"""
AutoFlowCFD - FP 几何组装辅助类和函数

包含 _KernelFaceData（numba kernel 输出的 flat 数组容器）和
棱柱四边形面对角线分类、multi-source 解析等辅助函数。
"""

from typing import List, Tuple

import numpy as np

from autoflowcfd.fr.face_flux_points import (
    ACCEPT_STRICT_REL,
    CUBE_FACE_AXIS_SIDE,
    FaceFluxPointGeometry,
    build_cross_interp,
    cell_info,
    face_ref_grid,
    map_ref_points,
)
from autoflowcfd.grid.curved_mapping.curved_mapping import PRISM_CUBE_FACES
from autoflowcfd.grid.connectivity.face_connectivity import CUBE_FACE_CODES, CUBE_FACE_NAMES, FRFaceConnectivity


class _KernelFaceData:
    """numba kernel 输出的 flat 数组容器，替代 180 万个 FaceFluxPointGeometry 对象。

    残差计算路径（build_flat_face_geometry）直接读取 flat 数组，跳过逐面
    对象创建。后处理代码（fr_coefficients、boundary 等）通过 __getitem__
    按需创建 FaceFluxPointGeometry（仅边界面 ~39K 个，可忽略）。
    """
    __slots__ = (
        'n_faces', 'n_fp', 'n_sps', 'n1d',
        'owner_axis', 'owner_side', 'neighbor_axis', 'neighbor_side',
        'owner_is_primary', 'neighbor_is_primary',
        'true_normal', 'true_area_weight',
        'nb_src0_cell', 'nb_src0_mat', 'nb_src1_idx',
        'ow_src0_cell', 'ow_src0_mat', 'ow_src1_idx',
        'nb_extra_cell', 'nb_extra_mat', 'ow_extra_cell', 'ow_extra_mat',
        '_mesh', '_face_conn', '_sps_1d',
        '_nb_fc', '_nb_resid', '_ow_fc', '_ow_resid',
        '_nb_cell_id', '_ow_cell_id',
        '_is_lower_fp_standard', '_is_lower_fp_flipped',
        '_owner_groups', '_neighbor_groups',
        '_cache',
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)
        self._cache = {}

    def __len__(self):
        return self.n_faces

    def __getitem__(self, f):
        """按需创建 FaceFluxPointGeometry（后处理代码兼容）。"""
        cached = self._cache.get(f)
        if cached is not None:
            return cached
        if f >= self.n_faces:
            raise IndexError(f)
        ffp = self._build_ffp(f)
        self._cache[f] = ffp
        return ffp

    def _build_ffp(self, f):
        """为第 f 个面构建 FaceFluxPointGeometry（按需，仅后处理使用）。"""
        n1d = self.n1d
        n_fp = self.n_fp
        fc = self._face_conn

        oa = int(self.owner_axis[f])
        os_ = float(self.owner_side[f])
        na = int(self.neighbor_axis[f])
        ns_ = float(self.neighbor_side[f])
        tn = self.true_normal[f]
        taw = self.true_area_weight[f]
        op = bool(self.owner_is_primary[f])
        np_ = bool(self.neighbor_is_primary[f])

        if fc.is_boundary[f]:
            return FaceFluxPointGeometry(
                owner_axis=oa, owner_side=os_,
                neighbor_axis=-1, neighbor_side=0.0,
                neighbor_sources=[], owner_sources=[],
                true_normal=tn, true_area_weight=taw,
                owner_is_primary=op, neighbor_is_primary=True,
            )

        # 从 flat 数组直接构建 sources（使用 src0 + src1 紧凑索引）
        nb_sources = []
        c0 = int(self.nb_src0_cell[f])
        if c0 >= 0:
            nb_sources.append((c0, self.nb_src0_mat[f]))
        idx1 = int(self.nb_src1_idx[f])
        if idx1 >= 0:
            nb_sources.append((int(self.nb_extra_cell[idx1]), self.nb_extra_mat[idx1]))

        ow_sources = []
        c0 = int(self.ow_src0_cell[f])
        if c0 >= 0:
            ow_sources.append((c0, self.ow_src0_mat[f]))
        idx1 = int(self.ow_src1_idx[f])
        if idx1 >= 0:
            ow_sources.append((int(self.ow_extra_cell[idx1]), self.ow_extra_mat[idx1]))

        return FaceFluxPointGeometry(
            owner_axis=oa, owner_side=os_,
            neighbor_axis=na, neighbor_side=ns_,
            neighbor_sources=nb_sources, owner_sources=ow_sources,
            true_normal=tn, true_area_weight=taw,
            owner_is_primary=op, neighbor_is_primary=np_,
        )


# 棱柱的 3 个四边形侧面（a=-1,a=+1,b=-1）在立方体面整数编码中的取值——
# 只有这些面会被网格生成器拆分成 2 个三角形子面（c=-1/c=+1 封盖本身就是
# 三角形，b=+1 退化，均不受影响）
_PRISM_QUAD_CODES = {CUBE_FACE_CODES["a=-1"], CUBE_FACE_CODES["a=+1"], CUBE_FACE_CODES["b=-1"]}


def _prism_quad_diagonal_local(cell_node_ids: np.ndarray, quad_local_idx: Tuple[int, ...]) -> Tuple[int, int]:
    """求棱柱某个四边形侧面真正的对角线，与 grid/mesh_gen/face_extraction_kernels.py
    的三角化规则完全一致。

    按 GLOBAL 节点编号对底面三角形 3 个顶点重新排序得到 v0'<v1'<v2'
    （顶面按同一置换得到 w0',w1',w2'），用对角线规则 v0'-w1' / v1'-w2' /
    v0'-w2'。

    Returns:
        (bottom_local_idx, top_local_idx)：对角线两端点的局部存储索引。
    """
    bottom_local = (0, 1, 2)
    top_local = (3, 4, 5)
    bottom_ids = [int(cell_node_ids[i]) for i in bottom_local]
    order = sorted(range(3), key=lambda k: bottom_ids[k])
    v_sorted_local = [bottom_local[order[k]] for k in range(3)]
    w_sorted_local = [top_local[order[k]] for k in range(3)]

    i_bottom_a, i_bottom_b = quad_local_idx[0], quad_local_idx[1]
    edge = {i_bottom_a, i_bottom_b}
    if edge == {v_sorted_local[0], v_sorted_local[1]}:
        return v_sorted_local[0], w_sorted_local[1]
    if edge == {v_sorted_local[1], v_sorted_local[2]}:
        return v_sorted_local[1], w_sorted_local[2]
    if edge == {v_sorted_local[0], v_sorted_local[2]}:
        return v_sorted_local[0], w_sorted_local[2]
    raise RuntimeError(
        f"Quad bottom edge {edge} does not match any base-triangle edge from "
        f"globally-sorted vertices {v_sorted_local} - unexpected prism connectivity."
    )


def _quad_half_sets(cell_node_ids: np.ndarray, quad_local_idx: Tuple[int, ...]) -> Tuple[bool, frozenset, frozenset]:
    """给定棱柱四边形侧面的 4 个局部角点，返回：
    (对角线是否连接第0/2个角点, 下三角局部索引集合, 上三角局部索引集合)。
    """
    i0, i1, i2, i3 = quad_local_idx
    d_bottom, d_top = _prism_quad_diagonal_local(cell_node_ids, quad_local_idx)
    diag = {d_bottom, d_top}
    if diag == {i0, i2}:
        return True, frozenset((i0, i1, i2)), frozenset((i0, i2, i3))
    if diag == {i1, i3}:
        return False, frozenset((i0, i1, i3)), frozenset((i1, i2, i3))
    raise RuntimeError(f"Diagonal {diag} is not a valid quad diagonal of corners {quad_local_idx}")


def _classify_half(
    cell_node_ids: np.ndarray, quad_local_idx: Tuple[int, ...], face_node_ids: np.ndarray
) -> Tuple[str, bool]:
    """判断某条子面记录的 3 个全局节点号对应四边形对角线的哪一侧
    ("lower"/"upper")，以及该四边形的真实对角线是否为标准情形。
    """
    is_standard, lower_local, upper_local = _quad_half_sets(cell_node_ids, quad_local_idx)
    lower_global = frozenset(int(cell_node_ids[i]) for i in lower_local)
    upper_global = frozenset(int(cell_node_ids[i]) for i in upper_local)
    face_set = frozenset(int(x) for x in face_node_ids)
    if face_set == lower_global:
        return "lower", is_standard
    if face_set == upper_global:
        return "upper", is_standard
    raise RuntimeError(
        f"Face node set {set(face_set)} does not match either diagonal half "
        f"({set(lower_global)} / {set(upper_global)}) of quad corners "
        f"{[int(cell_node_ids[i]) for i in quad_local_idx]} - unexpected prism quad-face triangulation."
    )


def _resolve_multi_source(
    face_conn: FRFaceConnectivity,
    mesh,
    n1d: int,
    sps_1d: np.ndarray,
    n_fp: int,
    is_lower_fp_standard: np.ndarray,
    is_lower_fp_flipped: np.ndarray,
    cell_id: int,
    code: int,
    group_faces: List[int],
    role: str,
    precomputed_free_coords: np.ndarray = None,
    precomputed_resid: float = None,
) -> Tuple[List[tuple], float, float]:
    """cell_id 在 group_faces 这组里始终扮演 role 角色。返回该 cell 一侧完整
    原生 Flux Points 网格对应的 sources 列表。group_faces 长度恒为 1 或 2。

    Args:
        precomputed_free_coords: (n_fp, 2) 或 None。numba kernel 预计算的
            Newton 自由坐标。提供时跳过 Newton 迭代。
        precomputed_resid: float 或 None。配套的预计算残差。

    Returns:
        (sources, worst_resid, char_length)
    """
    axis, side = CUBE_FACE_AXIS_SIDE[CUBE_FACE_NAMES[code]]
    is_prism_c, nodes_c = cell_info(mesh, cell_id)
    ref_grid_full = face_ref_grid(n1d, axis, side, sps_1d)
    phys_fp_full = map_ref_points(is_prism_c, ref_grid_full, nodes_c)

    def other_side(gf: int) -> tuple:
        if role == "owner":
            return int(face_conn.neighbor_cell[gf]), int(face_conn.neighbor_cube_face[gf])
        return int(face_conn.owner_cell[gf]), int(face_conn.owner_cube_face[gf])

    def cross_translation(gf: int):
        t = face_conn.face_translation[gf]
        if not np.any(t):
            return None
        return -t if role == "owner" else t

    if len(group_faces) == 1:
        gf = group_faces[0]
        other_cell, other_code = other_side(gf)
        other_axis, other_side_val = CUBE_FACE_AXIS_SIDE[CUBE_FACE_NAMES[other_code]]
        char_length = float(np.sqrt(max(face_conn.area[gf], 1e-300)))
        interp, resid = build_cross_interp(
            mesh, n1d, sps_1d, other_cell, other_axis, other_side_val, phys_fp_full,
            char_length=char_length, translation=cross_translation(gf),
            precomputed_free_coords=precomputed_free_coords,
            precomputed_resid=precomputed_resid,
        )
        return [(other_cell, interp)], resid, char_length

    cell_node_ids = mesh._fixed_prism_conn[cell_id]
    quad_local_idx = PRISM_CUBE_FACES[CUBE_FACE_NAMES[code]]

    sources = []
    worst_resid = 0.0
    worst_char_length = 1.0
    for gi, gf in enumerate(group_faces):
        other_cell, other_code = other_side(gf)
        other_axis, other_side_val = CUBE_FACE_AXIS_SIDE[CUBE_FACE_NAMES[other_code]]
        half, is_standard = _classify_half(cell_node_ids, quad_local_idx, face_conn.face_node_ids[gf])
        is_lower_fp = is_lower_fp_standard if is_standard else is_lower_fp_flipped
        mask = is_lower_fp if half == "lower" else ~is_lower_fp
        sub_pts = phys_fp_full[mask]
        full_matrix = np.zeros((n_fp, n1d**3))
        if sub_pts.shape[0] > 0:
            sub_fc = None
            sub_resid = None
            if gi == 0 and precomputed_free_coords is not None:
                sub_fc = precomputed_free_coords[mask]
                sub_resid = precomputed_resid
            char_length = float(np.sqrt(max(face_conn.area[gf], 1e-300)))
            sub_interp, resid = build_cross_interp(
                mesh, n1d, sps_1d, other_cell, other_axis, other_side_val, sub_pts,
                char_length=char_length, translation=cross_translation(gf),
                precomputed_free_coords=sub_fc,
                precomputed_resid=sub_resid,
            )
            full_matrix[mask] = sub_interp
            if resid / max(char_length, 1e-300) > worst_resid / max(worst_char_length, 1e-300):
                worst_resid, worst_char_length = resid, char_length
        sources.append((other_cell, full_matrix))
    return sources, worst_resid, worst_char_length
