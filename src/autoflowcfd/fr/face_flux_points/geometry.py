"""
AutoFlowCFD - FR 单元-面 Flux Points 几何预计算：核心数值原语 (V2.0 Tier-0)

本模块提供 Flux Points 几何构建所需的底层数值原语：
1. **精确点位定位**：对 owner 的每个 FP 物理坐标，在 neighbor 的对应局部
   面上用 Newton 迭代反解出 neighbor 的精确切向坐标，使曲边映射恰好给出
   该物理点；再用通用 Lagrange 插值算出 neighbor 解在该精确坐标处的取值。
   不假设两侧网格对齐（第一版"最近邻置换"假设在真实网格上被证伪：owner/
   neighbor 各自独立做坍缩坐标离散化，一般不产生物理重合的点集）。

   收敛/接受判据按局部面物理特征尺度（sqrt(面积)）做相对量纲化。棱柱侧面
   是双线性曲面，与相邻四面体的平面三角形共享界面时，两者只在角点严格
   重合，内部会有真实、有界的几何偏差（截断误差量级，非 bug）——严格阈值
   未过但在容忍阈值内的会被接受并记录，超出容忍阈值才视为真正定位失败。

真正把这些原语组装成完整 Flux Points 几何（含棱柱四边形侧面被网格生成器
恒定拆分成 2 个三角形子面、可能对应 1~2 个不同真实相邻单元这一拓扑情形
的正确处理）在 `fr/face_flux_points/merge.py`，避免本文件超过代码规范的
400 行限制。
"""

from typing import Dict

import numpy as np

from .locate import newton_locate_on_prism_face
from ..native_padding import sps_width
from autoflowcfd.grid.connectivity.face_connectivity import (
    CUBE_FACE_NAMES,
    NATIVE_PRISM_FACE_CODE_RANGE,
    NATIVE_TET_FACE_CODE_RANGE,
)

# 每个立方体面标识 -> (被坍缩掉的计算方向索引, 边界取值)。
# tet_native_v0~v3：native 四面体（路径C）没有 (axis,side) 这个概念，
# 这四项复用同一个查找表的存储位置，把"被排除的局部顶点 0~3"直接放进
# axis 槽位、side 槽位固定填 -1.0（只是为了让 face_flux_points/merge.py
# 里 `_CF_AXIS=[v[0] for v in ...values()]`/`_CF_SIDE=[v[1] for v in
# ...values()]` 这两个按字典插入顺序构造的查找表在扩到 10 项后依然
# 给出有效值——下游消费方（core/fr_operators/face_kernels.py::
# boundary_extrap 数组、inviscid_kernel.py
# 的 celltype/axis 索引）按 celltype==2（native tet）时把这个"axis"值
# 重新解释成 excluded_vertex，不是真的把 native 面塞进坍缩坐标的
# (axis,side) 语义里，见 Part7 文档相关小节）。`build_cross_interp`
# 的 native 分支（`target_face_code>=6`）不读这个字典，直接从
# `target_face_code-6` 算 excluded_vertex，这里的扩展只服务于
# `_CF_AXIS`/`_CF_SIDE` 这一条查找路径。
CUBE_FACE_AXIS_SIDE = {
    "a=-1": (0, -1.0),
    "a=+1": (0, 1.0),
    "b=-1": (1, -1.0),
    "b=+1": (1, 1.0),
    "c=-1": (2, -1.0),
    "c=+1": (2, 1.0),
    "tet_native_v0": (0, -1.0),
    "tet_native_v1": (1, -1.0),
    "tet_native_v2": (2, -1.0),
    "tet_native_v3": (3, -1.0),
    # 原生棱柱面（编码 10~14）：这里填的是**真实**的 (axis, side)，不是
    # 四面体那种"axis 槽位复用成 excluded_vertex"的哑值。
    #
    # 理由：原生棱柱面的通量点与坍缩立方体面的通量点已验证是**同一批物理
    # 点、同一顺序**（坍缩面在 (r,s) 里的像恰好是原生的 5 个面：
    # a=-1 -> r=-1 边、a=+1 -> 斜边 r+s=0、b=-1 -> s=-1 边、c=∓1 -> 两个
    # 三角形封盖），所以"在目标单元里定位面点"这一步继续走坍缩的
    # `newton_locate_on_prism_face`，它需要真实的 (axis, side)。
    # 实测等价性（P1~P3、右棱柱与不规则棱柱）：通量点 0.0~7.5e-17、
    # 法向 3.3e-16、物理面积权重 2.4e-15。
    #
    # 与 `NATIVE_PRISM_FACE_TO_CUBE_FACE` 的一致性由
    # `tests/unit/test_native_prism_face.py` 断言（那张表是对应关系的唯一
    # 事实来源、有形状无关的几何验证；这里不 import 它是为了避免模块级
    # 循环导入，改由测试钉住两者不会漂移）。
    "prism_native_f0": (2, -1.0),
    "prism_native_f1": (2, 1.0),
    "prism_native_f2": (0, 1.0),
    "prism_native_f3": (0, -1.0),
    "prism_native_f4": (1, -1.0),
}

ACCEPT_STRICT_REL = 1e-6  # 严格通过阈值：相对局部面特征尺度，供 face_flux_points/merge.py 判断是否需要记录容忍案例

# 原生四面体体积 Vandermonde 的 LU 分解缓存（键为阶数）：用 LU 分解 + lu_solve 而不是显式求逆——显式求逆会把
# Vandermonde 的条件数直接乘进舍入误差（实测用显式逆重构常数场在 P3 残差 2.8e-8，lu_solve 精确到 0.0）。
_V_SPS_LU_CACHE: Dict[tuple, tuple] = {}


def face_ref_grid(n1d: int, axis: int, side: float, sps_1d: np.ndarray) -> np.ndarray:
    other_axes = [a for a in range(3) if a != axis]
    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    pts = np.zeros((n1d * n1d, 3))
    pts[:, axis] = side
    pts[:, other_axes[0]] = g1.ravel()
    pts[:, other_axes[1]] = g2.ravel()
    return pts


def native_tet_face_points_physical(
    n1d: int, excluded_vertex: int, cell_nodes: np.ndarray, sps_1d: np.ndarray
) -> np.ndarray:
    """native 四面体（路径C）某个真实面（对面被排除的局部顶点
    `excluded_vertex` 给定）上，生成 `n1d*n1d` 个物理点位置——数量
    与坍缩坐标方案的 `face_ref_grid` 完全一致（Part7 文档"二·五"节：
    `n_fp` 是全网格统一的单一标量，native 面必须提供同样数量的点，
    不能用原生最小面节点数 `(order+1)(order+2)/2`，否则破坏
    `_KernelFaceData` flat 数组的统一形状假设）。

    复用与棱柱三角形封盖（`map_prism_to_physical` 内部 c=-1/c=+1 分支）
    完全相同的坍缩三角形网格构造（`cube_to_tri_rs`/`tri_barycentric`）
    ——这里坍缩三角形只用来**选取物理点位置**（纯几何采样问题，与
    "用哪套基函数表示体积场"无关，不涉及本次调查关心的对退化轴求导
    病态），是安全的复用，不是重新引入坍缩坐标的病态机制。

    Returns:
        phys: (n1d*n1d, 3)
    """
    from ...grid.curved_mapping.curved_mapping import cube_to_tri_rs, tri_barycentric

    face_vertex_idx = tuple(v for v in range(4) if v != excluded_vertex)
    p_i, p_j, p_k = cell_nodes[face_vertex_idx[0]], cell_nodes[face_vertex_idx[1]], cell_nodes[face_vertex_idx[2]]

    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    r_tri, s_tri = cube_to_tri_rs(g1.ravel(), g2.ravel())
    l1, l2, l3 = tri_barycentric(r_tri, s_tri)
    return l1[:, None] * p_i + l2[:, None] * p_j + l3[:, None] * p_k


def cell_info(mesh, cell_id: int):
    is_prism = cell_id < mesh.n_prism_cells
    node_ids = (
        mesh._fixed_prism_conn[cell_id] if is_prism else mesh._fixed_tet_conn[cell_id - mesh.n_prism_cells]
    )
    return is_prism, mesh._node_coords[node_ids]


def _get_v_sps_lu_native(order: int):
    """native 四面体（路径C）版本体积 Vandermonde LU 缓存——键只用
    `order`（native 没有 n1d/张量积立方体节点这个概念），复用
    `native_tet.basis.build_native_tet_operators` 内部已经对同一组
    体积节点算过的模态取值，避免重复分解。"""
    key = ("tet_native", order)
    cached = _V_SPS_LU_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1]
    from ..native_tet.basis import build_native_tet_operators, restricted_tet_modes, simplex3d_value, rst_to_abc

    ref_rst, _ = build_native_tet_operators(order)
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    modes = restricted_tet_modes(order)
    V_sps = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])
    from scipy.linalg import lu_factor
    lu_piv = lu_factor(V_sps.T)
    _V_SPS_LU_CACHE[key] = (lu_piv, modes)
    return lu_piv, modes


def build_cross_interp(
    mesh,
    n1d: int,
    sps_1d: np.ndarray,
    target_cell: int,
    target_face_code: int,
    source_phys: np.ndarray,
    char_length: float = 1.0,
    translation: np.ndarray = None,
) -> tuple:
    """求 target_cell 的解在给定 source_phys 目标物理点集上的取值算子，
    形状 (n_source_pts, n_sps)（n_sps=sps_width(order)，与 coarse 网格全局体积
    数组宽度一致——native 四面体的有效自由度更少，按 Part6/7"补位对齐"
    原则，插值矩阵多出的列恒为零，不改变全局数组统一形状这个既有假设，
    见下方 native 分支）。target_cell 的固定面由 `target_face_code`
    给定（原生面编码：四面体 6~9、棱柱 10~14，`grid/connectivity/face_connectivity.py`）。

    Args:
        translation: (3,) 或 None。周期边界配对面专用（见
            grid/connectivity/face_connectivity.py::FRFaceConnectivity.face_translation
            文档）——source_phys 是"来源"侧面上的真实物理坐标，但周期面
            物理上不重合，不能直接拿去在 target_cell（"目标"侧、位于
            周期像位置）里定位，必须先减去平移量，把目标点从"来源"侧
            的物理坐标系平移到"目标"侧的物理坐标系。非周期面（绝大多数
            调用）传 None，等价于零平移。
    """
    is_prism, cell_nodes = cell_info(mesh, target_cell)
    is_tet_native = (not is_prism) and (
        NATIVE_TET_FACE_CODE_RANGE[0] <= target_face_code
        < NATIVE_TET_FACE_CODE_RANGE[1])
    # 原生棱柱目标面（编码 [10,15)）：**定位**走坍缩的 Newton（两条基的
    # 几何映射逐位恒等，见 fr/native_prism/__init__.py），只有最后求
    # Vandermonde 时换成原生的 —— 与 numba kernel 同一条逻辑。
    is_prism_native = is_prism and (
        NATIVE_PRISM_FACE_CODE_RANGE[0] <= target_face_code
        < NATIVE_PRISM_FACE_CODE_RANGE[1])

    if is_tet_native:
        excluded_vertex = target_face_code - 6
        from .locate import locate_native_tet_face_point

        search_phys = source_phys if translation is None else source_phys - translation[np.newaxis, :]
        rst, final_resid = locate_native_tet_face_point(
            cell_nodes, excluded_vertex, search_phys, char_length=char_length
        )

        order = n1d - 1
        from ..native_tet.basis import rst_to_abc, simplex3d_value

        a, b, c = rst_to_abc(rst[:, 0], rst[:, 1], rst[:, 2])
        _, modes = _get_v_sps_lu_native(order)
        V_target = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])

        from scipy.linalg import lu_solve
        lu_piv, _ = _get_v_sps_lu_native(order)
        interp_native = lu_solve(lu_piv, V_target.T).T  # (n_pts, n_native_sps)

        n_pts = source_phys.shape[0]
        n_sps = sps_width(n1d - 1)
        interp = np.zeros((n_pts, n_sps))
        interp[:, : interp_native.shape[1]] = interp_native
        return interp, final_resid

    target_axis, target_side = CUBE_FACE_AXIS_SIDE[CUBE_FACE_NAMES[target_face_code]]

    search_phys = source_phys if translation is None else source_phys - translation[np.newaxis, :]
    free_coords, final_resid = newton_locate_on_prism_face(
        cell_nodes, target_axis, target_side, search_phys, char_length=char_length
    )

    other_axes = [a for a in range(3) if a != target_axis]
    n_pts = source_phys.shape[0]
    n_sps = sps_width(n1d - 1)
    order = n1d - 1

    abc = np.zeros((n_pts, 3))
    abc[:, target_axis] = target_side
    abc[:, other_axes[0]] = free_coords[:, 0]
    abc[:, other_axes[1]] = free_coords[:, 1]

    if is_prism_native:
        # 原生棱柱目标：坍缩 Newton 给出的 `abc` **直接**就能求原生棱柱
        # 模态（坐标往返恰好是恒等，实测吻合 9.4e-16 相对）。复用 numba 侧
        # **同一个实现** `native_prism_interp_matrix_nb`（njit 函数同样可以
        # 从纯 Python 调用）—— 不在这里另写一份 Python 版，那会是第二个
        # 要同步的事实来源。
        from autoflowcfd.fr.native_prism.basis import (
            build_native_prism_nodes,
            build_native_prism_vandermonde,
            restricted_prism_modes,
        )
        from autoflowcfd.fr.native_prism.interp_numba import (
            native_prism_interp_matrix_nb,
        )
        from scipy.linalg import lu_factor, lu_solve

        modes_np = restricted_prism_modes(order)
        n_np = len(modes_np)
        V_np, _, _, _ = build_native_prism_vandermonde(
            order, build_native_prism_nodes(order))
        v_inv_np = np.ascontiguousarray(
            lu_solve(lu_factor(V_np.T), np.eye(n_np)).T)
        interp = native_prism_interp_matrix_nb(
            np.ascontiguousarray(abc),
            np.array([m[0] for m in modes_np], dtype=np.int32),
            np.array([m[1] for m in modes_np], dtype=np.int32),
            np.array([m[2] for m in modes_np], dtype=np.int32),
            v_inv_np, n_sps)
        if interp.shape != (n_pts, n_sps):
            raise ValueError(
                f"原生棱柱插值矩阵形状 {interp.shape} 应为 "
                f"({n_pts}, {n_sps})")
        return interp, final_resid

    raise ValueError(
        f"目标面编码 {target_face_code} 不是原生面编码（四面体 [6,10)、棱柱 [10,15)）：坍缩立方体面编码 0~5 随坍缩基"
        f"一并删除，面连接关系必须先翻译成原生面编码")
