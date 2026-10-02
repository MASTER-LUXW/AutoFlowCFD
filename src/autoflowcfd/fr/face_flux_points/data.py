"""
AutoFlowCFD - FP 几何组装辅助类和函数

包含 _KernelFaceData（numba kernel 输出的 flat 数组容器）和
棱柱四边形面对角线分类、multi-source 解析等辅助函数。
"""

from typing import Tuple

import numpy as np

from .geometry import CUBE_FACE_AXIS_SIDE, FaceFluxPointGeometry
from autoflowcfd.grid.curved_mapping.curved_mapping import PRISM_CUBE_FACES
from autoflowcfd.grid.connectivity.face_connectivity import (
    CUBE_FACE_CODES,
    CUBE_FACE_NAMES,
    NATIVE_PRISM_FACE_CODE_RANGE,
    NATIVE_TET_FACE_CODE_RANGE,
)


def prism_quad_local_idx(code: int) -> Tuple[int, ...]:
    """某个棱柱面编码对应的**四边形侧面局部顶点序号**。

    `PRISM_CUBE_FACES` 只以坍缩立方体面名（`a=-1` 等）为键，而原生档下
    同一批面记录带的是原生编码 [10,15)。两者是**同一个物理面**（原生
    棱柱面的通量点与坍缩立方体面的通量点已验证是同一批物理点、同一
    顺序，见 `fr/native_prism/__init__.py` 模块文档），所以局部顶点序号
    完全相同 —— 这里做一次编码翻译，而不是给 `PRISM_CUBE_FACES` 加 5 个
    重复条目（那张表是几何映射层的定义，不该知道原生面编码）。

    单独一个函数：`build_face_flux_points` 里有三处、本模块里有一处共
    四个调用点原本都写成 `PRISM_CUBE_FACES[CUBE_FACE_NAMES[code]]`，
    原生档下逐个 `KeyError: 'prism_native_f4'`（真实复现：1728 单元
    结构化棱柱网格第一次端到端构造）。四处各自翻译一遍正是本项目反复
    出"只改了一份"缺陷的形态。

    Raises:
        KeyError: 该编码不是棱柱的四边形侧面（三角形封盖与退化面不会、
            也不应该走到这里）。
    """
    code = int(code)
    lo, hi = NATIVE_PRISM_FACE_CODE_RANGE
    if lo <= code < hi:
        from autoflowcfd.fr.native_prism.face import (
            NATIVE_PRISM_FACE_TO_CUBE_FACE,
        )

        axis, side = NATIVE_PRISM_FACE_TO_CUBE_FACE[code - lo]
        name = _AXIS_SIDE_TO_COLLAPSED_NAME[(int(axis), float(side))]
    else:
        name = CUBE_FACE_NAMES[code]
    return PRISM_CUBE_FACES[name]


#: `(axis, side)` -> **坍缩**立方体面名的反查表（只含那 6 个面）。
#: 从 `CUBE_FACE_AXIS_SIDE` 派生，不手抄第二份 —— 原生面在那张表里填的
#: 也是真实 (axis, side)，会与坍缩面撞键，所以这里显式只取前 6 个名字。
_AXIS_SIDE_TO_COLLAPSED_NAME = {
    (int(ax), float(sd)): _n
    for _n, (ax, sd) in CUBE_FACE_AXIS_SIDE.items()
    if CUBE_FACE_CODES[_n] < NATIVE_TET_FACE_CODE_RANGE[0]
}


class _KernelFaceData:
    """numba kernel 输出的 flat 数组容器，替代 180 万个 FaceFluxPointGeometry 对象。

    残差计算路径（build_flat_face_geometry）直接读取 flat 数组，跳过逐面
    对象创建。后处理代码（fr_coefficients、boundary 等）通过 __getitem__
    按需创建 FaceFluxPointGeometry（仅边界面 ~39K 个，可忽略）。

    混合分组字段（B-8，2026-08-25，见 face_flux_points/merge.py 同名段注释）：
    mixed_nb_partner/mixed_ow_partner：(n_faces,) int64，混合面内部记录 ->
    边界子面记录索引（-1 表示普通面）；mixed_nb_mask/mixed_ow_mask：
    (n_faces, n_fp) bool，True 的 FP 落在边界半区（残差 kernel 逐 FP 取幽灵态）；
    mixed_bnd_face：(n_faces,) bool，边界子面记录标志（不参与累加但需算幽灵态）；
    mixed_p0_bnd_frac：(n_faces,) float64，P0 面积占比混合用。

    owner_tri_slot/neighbor_tri_slot：(n_faces,) int64，两侧三角形面的坍缩顶点槽位
    （`fr/triangle_apex.py`；四边形面与边界面 neighbor 侧为 0）。该侧通量点集与顺序、
    面外插/提升算子都按它取；由单元连接关系唯一决定，共享面两侧的点因此物理重合。
    """
    __slots__ = (
        'n_faces', 'n_fp', 'n_sps', 'n1d',
        'owner_axis', 'owner_side', 'neighbor_axis', 'neighbor_side',
        'owner_is_primary', 'neighbor_is_primary',
        'true_normal', 'true_area_weight',
        'owner_adj_row_exact', 'neighbor_adj_row_exact',
        'nb_src0_cell', 'nb_src0_tpl', 'nb_src0_tid', 'nb_src1_idx',
        'ow_src0_cell', 'ow_src0_tpl', 'ow_src0_tid', 'ow_src1_idx',
        'nb_extra_cell', 'nb_extra_mat', 'ow_extra_cell', 'ow_extra_mat',
        'mixed_nb_partner', 'mixed_nb_mask', 'mixed_ow_partner', 'mixed_ow_mask',
        'mixed_bnd_face', 'mixed_p0_bnd_frac',
        'owner_tri_slot', 'neighbor_tri_slot',
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
            nb_sources.append((c0, self.nb_src0_tpl[self.nb_src0_tid[f]]))
        idx1 = int(self.nb_src1_idx[f])
        if idx1 >= 0:
            nb_sources.append((int(self.nb_extra_cell[idx1]), self.nb_extra_mat[idx1]))

        ow_sources = []
        c0 = int(self.ow_src0_cell[f])
        if c0 >= 0:
            ow_sources.append((c0, self.ow_src0_tpl[self.ow_src0_tid[f]]))
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


#: 棱柱的 3 个四边形侧面在立方体面整数编码中的取值 —— 只有这些面会被
#: 网格生成器拆分成 2 个三角形子面（c=-1/c=+1 封盖本身就是三角形，
#: b=+1 退化，均不受影响），因此只有它们会走 multi-source 路径。
#:
#: **必须同时含原生棱柱编码**（2026-09-19）：原生档下同一批物理面记录
#: 的 owner/neighbor cube face 是 f3/f2/f4 = 13/12/14，漏掉它们会让
#: `build_face_flux_points` 的 `owner_groups` 压根不收录这些面，接着在
#: multi-source 分支 `owner_groups[key]` 直接 `KeyError: (0, 14)`
#: （真实复现：1728 单元结构化棱柱网格，原生档第一次端到端构造）。
#:
#: 这是本集合的**唯一**定义；numba 侧的 `face_flux_points_helpers_numba.
#: py::_PQ_CODES` 由它派生（那里需要一个 numba 能用的数组），由
#: `tests/unit/test_native_prism_face.py` 钉住两者不漂移。
_PRISM_QUAD_CODES = {
    CUBE_FACE_CODES["a=-1"], CUBE_FACE_CODES["a=+1"], CUBE_FACE_CODES["b=-1"],
    CUBE_FACE_CODES["prism_native_f2"],   # a=+1 的原生对应面
    CUBE_FACE_CODES["prism_native_f3"],   # a=-1
    CUBE_FACE_CODES["prism_native_f4"],   # b=-1
}


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
