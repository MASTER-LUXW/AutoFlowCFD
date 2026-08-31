"""
AutoFlowCFD - FR 单元-面 Flux Points 几何预计算：核心数值原语 (V2.0 Tier-0)

本模块提供 Flux Points 几何构建所需的底层数值原语：
1. 1D 边界外插权重、SPs->FP 张量外插（原生网格对齐场景）。
2. **精确点位定位**：对 owner 的每个 FP 物理坐标，在 neighbor 的对应局部
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
的正确处理）在 `fr/face_flux_points_merge.py`，避免本文件超过代码规范的
400 行限制。
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from autoflowcfd.fr.matrix_operators import compute_interpolation_matrix
from autoflowcfd.fr.collapsed_basis import tet_modal_basis_and_grad, prism_modal_basis_and_grad
from autoflowcfd.fr.face_flux_points_locate import map_ref_points, newton_locate_on_face
from autoflowcfd.grid.connectivity.face_connectivity import CUBE_FACE_NAMES

# 每个立方体面标识 -> (被坍缩掉的计算方向索引, 边界取值)。
# tet_native_v0~v3：native 四面体（路径C）没有 (axis,side) 这个概念，
# 这四项复用同一个查找表的存储位置，把"被排除的局部顶点 0~3"直接放进
# axis 槽位、side 槽位固定填 -1.0（只是为了让 face_flux_points_merge.py
# 里 `_CF_AXIS=[v[0] for v in ...values()]`/`_CF_SIDE=[v[1] for v in
# ...values()]` 这两个按字典插入顺序构造的查找表在扩到 10 项后依然
# 给出有效值——下游消费方（core/fr_operators/face_kernels.py::
# boundary_extrap 数组、inviscid_kernel.py/inviscid_kernel_colored.py
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
}

ACCEPT_STRICT_REL = 1e-6  # 严格通过阈值：相对局部面特征尺度，供 face_flux_points_merge.py 判断是否需要记录容忍案例

# 模态 Vandermonde 矩阵 V_sps 的 LU 分解缓存，键为 (cell_type, n1d)：
# ref_cube_sps（sps_1d 的张量积）与模态基函数定义只依赖单元类型和阶数，
# 与具体是哪个物理单元无关，全网格所有同类型同阶数单元共享同一份分解，
# 缓存避免对每一个面都重新分解一次 n_sps x n_sps 矩阵（真实网格上有
# 数十万个面）。用 LU 分解 + lu_solve，而不是显式求逆
# （V_target @ np.linalg.inv(V_sps)）：V_sps 的条件数随阶数快速增长
# （collapsed_basis.py 文档实测 N=2 时 ~1e5，N=3 时 ~2e9），显式求逆会把
# 这个条件数直接乘进舍入误差——受控数值实验验证：用显式逆重构常数场
# （V_sps^{-1}@ones 应恒为 e0）在 N=3 时残差达 2.8e-8，改用 lu_solve 精确
# 到 0.0（机器精度意义上的恰好相等）。真实网格上曾因此导致均匀自由
# 流场保持性测试在 P=2/P=3 均略微超出容差（P=2: 2.2e-7 vs 容差 1e-7），
# 换成 lu_solve 后消除。
_V_SPS_LU_CACHE: Dict[tuple, tuple] = {}


def _get_v_sps_lu(cell_type: str, n1d: int, sps_1d: np.ndarray):
    key = (cell_type, n1d)
    cached = _V_SPS_LU_CACHE.get(key)
    if cached is not None:
        return cached
    ga, gb, gc = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    a_sps, b_sps, c_sps = ga.ravel(), gb.ravel(), gc.ravel()
    order = n1d - 1
    if cell_type == "tet":
        V_sps, _, _, _ = tet_modal_basis_and_grad(a_sps, b_sps, c_sps, order)
    else:
        V_sps, _, _, _ = prism_modal_basis_and_grad(a_sps, b_sps, c_sps, order)
    from scipy.linalg import lu_factor
    lu_piv = lu_factor(V_sps.T)  # 转置：下面按 V_sps.T @ X = V_target.T 求解
    _V_SPS_LU_CACHE[key] = lu_piv
    return lu_piv


def compute_1d_boundary_weights(sps_1d: np.ndarray) -> Dict[float, np.ndarray]:
    """计算把 1D SPs 上的值外插到边界 x=-1 和 x=+1 的 Lagrange 权重向量。"""
    L = compute_interpolation_matrix(sps_1d, np.array([-1.0, 1.0]))
    return {-1.0: L[0], 1.0: L[1]}


def extrapolate_to_face(data: np.ndarray, n1d: int, axis: int, weights: np.ndarray) -> np.ndarray:
    """把 (n1d^3, ...) 的 SPs 数据沿指定计算方向外插到该方向边界，得到 (n1d^2, ...) 的 FP 数据。"""
    trailing_shape = data.shape[1:]
    reshaped = data.reshape((n1d, n1d, n1d) + trailing_shape)
    moved = np.moveaxis(reshaped, axis, 0)
    fp = np.tensordot(weights, moved, axes=([0], [0]))
    return fp.reshape((n1d * n1d,) + trailing_shape)


@dataclass
class FaceFluxPointGeometry:
    """单个物理面的 Flux Points 几何数据（owner 视角的 FP 网格为准）。

    棱柱的四边形侧面（a=-1/a=+1/b=-1）在网格生成阶段（grid/mesh_gen/
    face_extractor.py）恒定被三角化拆分成 2 个子面记录（哪怕相邻的也是
    同一个棱柱的单一四边形邻居——这是网格生成器保证棱柱与四面体核心区
    比特级保形的既有设计），因此同一个 (owner_cell, owner 立方体面) 可能
    对应 1~2 个不同的真实相邻单元。owner/neighbor 双方各自的贡献
    （自身原生 FP 网格外插 + 校正投影）必须只做一次，不能按拆分出的子面
    记录数重复计入；跨单元求值也必须按每个 FP 物理位置精确匹配到它
    *真正* 所属的那个相邻单元，而不是把整张四边形都指给其中一个。

    因此本结构不再直接存一个 (n_fp, n_sps) 插值矩阵 + 单一 neighbor_cell，
    而是存 sources 列表：每个元素是 (real_cell_id, (n_fp,n_sps) 矩阵)，
    矩阵在不属于该 real_cell 的 FP 行上恒为 0——调用方对列表求和即可得到
    正确的、按物理位置精确来源组装出的界面场，不管背后是 1 个还是 2 个
    真实相邻单元。*_is_primary 标记这条 face_connectivity 记录是否是其
    (cell, 立方体面) 分组里“负责触发一次自身外插+校正投影”的那条——非
    primary 的记录只贡献跨单元插值信息（已经被合并进 primary 记录的
    sources 里），本身不应再触发一次自身贡献，否则等价于重复计入。

    Attributes:
        owner_axis, owner_side: owner 单元被坍缩的计算方向与边界取值
        neighbor_axis, neighbor_side: neighbor 单元侧同上；边界面时为 -1/0.0
        neighbor_sources: List[(neighbor_cell_id, (n_fp,n_sps)矩阵)]，
            对 owner 的 FP 网格各自贡献部分（或全部）行；边界面为空列表
        owner_sources: List[(owner_cell_id, (n_fp,n_sps)矩阵)]，
            对 neighbor 原生 FP 网格各自贡献部分（或全部）行；边界面/
            neighbor_is_primary=False 时为空列表
        true_normal: (n_fp, 3) 真实物理单位法向量（owner->neighbor / 边界面
            指向域外），按 owner 顺序排列
        true_area_weight: (n_fp,) 每个 FP 代表的物理面积权重，按 owner 顺序
        owner_is_primary: 本记录是否负责 owner 侧的自身外插+校正投影
        neighbor_is_primary: 本记录是否负责 neighbor 侧的自身外插+校正投影
            （边界面恒为 True，但此时不产生 neighbor 侧计算）
    """
    __slots__ = (
        'owner_axis', 'owner_side', 'neighbor_axis', 'neighbor_side',
        'neighbor_sources', 'owner_sources', 'true_normal',
        'true_area_weight', 'owner_is_primary', 'neighbor_is_primary',
    )

    owner_axis: int
    owner_side: float
    neighbor_axis: int
    neighbor_side: float
    neighbor_sources: List[tuple]
    owner_sources: List[tuple]
    true_normal: np.ndarray
    true_area_weight: np.ndarray
    owner_is_primary: bool
    neighbor_is_primary: bool


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
    from ..grid.curved_mapping.curved_mapping import cube_to_tri_rs, tri_barycentric

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
    `native_simplex_basis.build_native_tet_operators` 内部已经对同一组
    体积节点算过的模态取值，避免重复分解。"""
    key = ("tet_native", order)
    cached = _V_SPS_LU_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1]
    from .native_simplex_basis import build_native_tet_operators, restricted_tet_modes, simplex3d_value, rst_to_abc

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
    precomputed_free_coords: np.ndarray = None,
    precomputed_resid: float = None,
) -> tuple:
    """求 target_cell 的解在给定 source_phys 目标物理点集上的取值算子，
    形状 (n_source_pts, n_sps)（n_sps=n1d**3，与 coarse 网格全局体积
    数组宽度一致——native 四面体的有效自由度更少，按 Part6/7"补位对齐"
    原则，插值矩阵多出的列恒为零，不改变全局数组统一形状这个既有假设，
    见下方 native 分支）。target_cell 的固定面由 `target_face_code`
    给定（`grid.connectivity.face_connectivity.CUBE_FACE_CODES` 编码：
    0~5 是坍缩坐标 6 个立方体面，6~9 是 native 四面体的 4 个真实面，
    见该模块与 Part7 文档第一节完整设计说明）。

    Args:
        translation: (3,) 或 None。周期边界配对面专用（见
            grid/face_connectivity.py::FRFaceConnectivity.face_translation
            文档）——source_phys 是"来源"侧面上的真实物理坐标，但周期面
            物理上不重合，不能直接拿去在 target_cell（"目标"侧、位于
            周期像位置）里定位，必须先减去平移量，把目标点从"来源"侧
            的物理坐标系平移到"目标"侧的物理坐标系。非周期面（绝大多数
            调用）传 None，等价于零平移。
        precomputed_free_coords: (n_pts, 2) 或 None。numba 并行 kernel
            预计算的 Newton 自由坐标（native 分支目前不支持这个预计算
            路径，传入非 None 会报错——native 定位是解析闭式解，比
            Newton 迭代本身还快，不需要这个性能优化，见 Part7 文档
            "实现顺序建议"未覆盖 numba 层这一如实说明）。
        precomputed_resid: float 或 None。与 precomputed_free_coords 配套
            的预计算残差。
    """
    is_prism, cell_nodes = cell_info(mesh, target_cell)
    is_tet_native = (not is_prism) and target_face_code >= 6

    if is_tet_native:
        if precomputed_free_coords is not None:
            raise NotImplementedError(
                "native 四面体分支暂不支持 precomputed_free_coords（numba 预计算路径）——"
                "见 build_cross_interp 文档，闭式解本身已经足够快，未来如需要可以补上。"
            )
        excluded_vertex = target_face_code - 6
        from .face_flux_points_locate import locate_native_tet_face_point

        search_phys = source_phys if translation is None else source_phys - translation[np.newaxis, :]
        rst, final_resid = locate_native_tet_face_point(
            cell_nodes, excluded_vertex, search_phys, char_length=char_length
        )

        order = n1d - 1
        from .native_simplex_basis import rst_to_abc, simplex3d_value

        a, b, c = rst_to_abc(rst[:, 0], rst[:, 1], rst[:, 2])
        _, modes = _get_v_sps_lu_native(order)
        V_target = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])

        from scipy.linalg import lu_solve
        lu_piv, _ = _get_v_sps_lu_native(order)
        interp_native = lu_solve(lu_piv, V_target.T).T  # (n_pts, n_native_sps)

        n_pts = source_phys.shape[0]
        n_sps = n1d**3
        interp = np.zeros((n_pts, n_sps))
        interp[:, : interp_native.shape[1]] = interp_native
        return interp, final_resid

    target_axis, target_side = CUBE_FACE_AXIS_SIDE[CUBE_FACE_NAMES[target_face_code]]

    if precomputed_free_coords is not None:
        free_coords = precomputed_free_coords
        final_resid = precomputed_resid if precomputed_resid is not None else 0.0
    else:
        search_phys = source_phys if translation is None else source_phys - translation[np.newaxis, :]
        free_coords, final_resid = newton_locate_on_face(
            is_prism, cell_nodes, target_axis, target_side, search_phys, char_length=char_length
        )

    # 用与 fr/collapsed_basis.py::build_collapsed_boundary_extrap（owner
    # 侧自身外插用的同一套算子）一致的坍缩坐标模态基插值，而不是朴素 1D
    # 张量积 Lagrange——此前两侧用不同插值空间，在同一组物理点上最大
    # 相差达 2070（tet c=-1 面，P=2），对光滑场造成约 5% 的伪界面跳跃，
    # 被当作真实间断喂进黎曼求解器，破坏守恒性且凭空注入数值耗散
    # （G-04 数值审计发现）。现在两侧统一用同一个模态 Vandermonde
    # 插值算子 V_target @ V_sps^{-1}，在共形界面上对同一份 SPs 数据
    # 精确给出一致的取值。
    other_axes = [a for a in range(3) if a != target_axis]
    n_pts = source_phys.shape[0]
    n_sps = n1d**3
    order = n1d - 1

    abc = np.zeros((n_pts, 3))
    abc[:, target_axis] = target_side
    abc[:, other_axes[0]] = free_coords[:, 0]
    abc[:, other_axes[1]] = free_coords[:, 1]

    cell_type = "prism" if is_prism else "tet"
    if cell_type == "tet":
        V_target, _, _, _ = tet_modal_basis_and_grad(abc[:, 0], abc[:, 1], abc[:, 2], order)
    else:
        V_target, _, _, _ = prism_modal_basis_and_grad(abc[:, 0], abc[:, 1], abc[:, 2], order)

    from scipy.linalg import lu_solve
    lu_piv = _get_v_sps_lu(cell_type, n1d, sps_1d)
    # interp = V_target @ V_sps^{-1}  <=>  interp.T = V_sps^{-T} @ V_target.T
    #        <=>  solve V_sps.T @ X = V_target.T for X = interp.T
    interp = lu_solve(lu_piv, V_target.T).T
    assert interp.shape == (n_pts, n_sps)
    return interp, final_resid
