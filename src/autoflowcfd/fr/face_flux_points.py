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

from autoflowcfd.fr.matrix_operators import compute_interpolation_matrix, compute_lagrange_weights_batch
from autoflowcfd.grid.curved_mapping import map_prism_to_physical, map_tet_to_physical

# 四面体 (fixed_axis, fixed_val) -> 该真实面 3 个顶点的局部索引（不在面上的
# 第 4 个顶点重心坐标恒为 0），与 curved_mapping.TET_CUBE_FACES 完全一致，
# 供 _tet_exact_locate_on_face 直接在物理空间解重心坐标使用。
_TET_FIXED_TO_FACE_VERTICES = {
    (0, -1.0): (0, 2, 3),
    (0, 1.0): (1, 2, 3),
    (1, -1.0): (0, 1, 3),
    (2, -1.0): (0, 1, 2),
}

# 每个立方体面标识 -> (被坍缩掉的计算方向索引, 边界取值)
CUBE_FACE_AXIS_SIDE = {
    "a=-1": (0, -1.0),
    "a=+1": (0, 1.0),
    "b=-1": (1, -1.0),
    "b=+1": (1, 1.0),
    "c=-1": (2, -1.0),
    "c=+1": (2, 1.0),
}

_NEWTON_MAX_ITER = 50
_NEWTON_TOL_REL = 1e-10  # 收敛判据：相对局部面特征尺度（不是绝对长度单位）
ACCEPT_STRICT_REL = 1e-6  # 严格通过阈值：相对局部面特征尺度，供 face_flux_points_merge.py 判断是否需要记录容忍案例

# 容忍阈值：相对局部面特征尺度，超过则视为真正的定位失败（报错中止，而非
# 静默放行）。取值有真实网格数据支撑，不是拍脑袋：在 cube_demo 网格
# （545597 单元）上直接、独立于 Newton 之外，对全部 322758 个棱柱四边形
# 侧面算了"第4个角点到其余3点所在平面的距离/四边形对角线长度"这个翘曲度
# 量——这是棱柱侧面（双线性曲面）与相邻四面体（平面三角形）只能在角点
# 严格重合、内部存在真实几何偏差的直接度量。结果：99% 的四边形翘曲度
# <2.87%，全网格最大值 11.13%（0 个超过 15%）。Newton 最小二乘解已经是
# 该目标点在双线性曲面上能达到的最优逼近，这个偏差是网格本身的固有几何
# 特征（混合棱柱/四面体网格在等参 FR 框架下的截断误差，类比任何数值
# 离散化都有的截断误差），不是算法缺陷；15% 的阈值留有余量覆盖这一真实
# 分布的同时，仍能可靠地把"真正不相容/断裂的面"（残差应远超此量级，
# 通常是数量级的差异）与"翘曲但合法的面"区分开——超过阈值报错中止而不是
# 静默放宽，接受阈值内的情形也会被完整记录（见 face_flux_points_merge.py
# 的 tolerated 汇总日志），不是简化或掩盖。
_ACCEPT_WARN_REL = 0.15


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


def map_ref_points(is_prism: bool, ref_pts: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    return map_prism_to_physical(ref_pts, cell_nodes) if is_prism else map_tet_to_physical(ref_pts, cell_nodes)


def _tet_exact_locate_on_face(
    cell_nodes: np.ndarray, fixed_axis: int, fixed_val: float, targets_phys: np.ndarray
) -> np.ndarray:
    """四面体某个真实面（fixed_axis=fixed_val）上一批目标物理点的精确点位
    定位——直接在物理空间解重心坐标，不用 Newton 迭代（原因见
    `newton_locate_on_face` 文档）。

    该面由 `curved_mapping.TET_CUBE_FACES` 约定的 3 个局部顶点（第 4 个
    顶点的重心坐标恒为 0）张成的平面唯一确定。物理空间里"3 点确定一个
    平面，求平面内一点的重心坐标"恒为良态的 3×2 最小二乘问题——条件数
    只取决于这 3 点自身构成的三角形的边长比（与三角形本身的几何形状
    绑定，不会像参考坐标系下的雅可比那样被坍缩坐标变换的非线性放大到
    接近奇异），求解后用 `cube_to_tet_rst` 的解析逆（t=c 直接已知；
    s=(1+b)(1-c)/2-1 与 r=-(1+a)(s+t)/2-1 都是关于单个未知量的线性方程，
    顺序回代求解，不需要迭代）换算回立方体坐标里的两个自由方向。

    Returns:
        free_coords: (n_pts, 2)，两个自由方向的坐标（顺序 = other_axes 升序）
    """
    other_axes = [a for a in range(3) if a != fixed_axis]
    key = (fixed_axis, float(fixed_val))
    face_vertex_idx = _TET_FIXED_TO_FACE_VERTICES.get(key)
    if face_vertex_idx is None:
        raise RuntimeError(f"Not a valid tet face: fixed_axis={fixed_axis}, fixed_val={fixed_val}")
    i, j, k = face_vertex_idx

    P_i, P_j, P_k = cell_nodes[i], cell_nodes[j], cell_nodes[k]
    e1 = P_j - P_i
    e2 = P_k - P_i
    rhs = targets_phys - P_i[None, :]  # (n_pts,3)

    # 与 Newton 迭代内部同一套 Jacobi 预条件手法：这里条件数天然良态，
    # 只取决于三角形自身边长比，与坍缩坐标参考系的局部雅可比无关。
    e1_norm = max(float(np.linalg.norm(e1)), 1e-300)
    e2_norm = max(float(np.linalg.norm(e2)), 1e-300)
    e1_hat = e1 / e1_norm
    e2_hat = e2 / e2_norm
    a11 = float(np.dot(e1_hat, e1_hat))
    a12 = float(np.dot(e1_hat, e2_hat))
    a22 = float(np.dot(e2_hat, e2_hat))
    b1 = rhs @ e1_hat
    b2 = rhs @ e2_hat
    det = a11 * a22 - a12 * a12
    det_safe = det if abs(det) > 1e-300 else 1e-300
    alpha = (b1 * a22 - b2 * a12) / det_safe / e1_norm
    beta = (a11 * b2 - a12 * b1) / det_safe / e2_norm

    n_pts = targets_phys.shape[0]
    L = np.zeros((n_pts, 4))
    L[:, i] = 1.0 - alpha - beta
    L[:, j] = alpha
    L[:, k] = beta

    r = 2.0 * L[:, 1] - 1.0
    s = 2.0 * L[:, 2] - 1.0
    t = 2.0 * L[:, 3] - 1.0

    c = t
    b_coord = 2.0 * (s + 1.0) / np.where(np.abs(1.0 - c) > 1e-300, 1.0 - c, 1e-300) - 1.0
    a_coord = -2.0 * (r + 1.0) / np.where(np.abs(s + t) > 1e-300, s + t, 1e-300) - 1.0

    full = {0: a_coord, 1: b_coord, 2: c}
    return np.column_stack([full[other_axes[0]], full[other_axes[1]]])


def newton_locate_on_face(
    is_prism: bool,
    cell_nodes: np.ndarray,
    fixed_axis: int,
    fixed_val: float,
    targets_phys: np.ndarray,
    char_length: float = 1.0,
) -> tuple:
    """在 cell 的某个固定 fixed_axis=fixed_val 局部面上，为一批目标物理点
    (n_pts,3) 精确定位对应的另外两个自由计算坐标，使映射结果与目标重合。

    四面体（is_prism=False）走精确闭式解（`_tet_exact_locate_on_face`）：
    四面体的真实面恒为 3 点确定的平面，直接在物理空间解重心坐标，不需要
    也不应该用 Newton 迭代——`cube_to_tet_rst` 对固定某一轴而言另外两个
    自由方向仍含交叉项（不是仿射的），迭代法在该映射参考坐标局部雅可比
    接近奇异处（两个自由方向映射到物理空间的方向接近平行，真实网格上
    偏斜/细长的四面体会触发）可能被"阻尼线搜索要求残差单调下降"这一
    稳健性措施本身困住，即使目标点物理上精确可达（与该面共面，验证到
    ~1e-17）也可能收敛失败——已在真实网格上复现并确认闭式解可靠收敛到
    机器精度，见 `_tet_exact_locate_on_face` 文档。

    棱柱（is_prism=True）没有解析闭式解（棱柱侧面是真双线性曲面），
    走向量化批量 Newton 迭代（有限差分雅可比）——对该映射在给定面上光滑
    且（对非退化单元）局部可逆，Newton 收敛快（通常 <5 次迭代到机器
    精度），已数值验证，见模块文档。

    Returns:
        (free_coords, final_resid): free_coords 形状 (n_pts, 2)（两个自由
        方向的坐标，顺序 = other_axes 升序）；final_resid 为该批点的最大
        物理残差（绝对长度单位），供调用方按 char_length 分级处理。
    """
    other_axes = [a for a in range(3) if a != fixed_axis]
    n_pts = targets_phys.shape[0]
    scale = max(char_length, 1e-300)

    def full_points(xy: np.ndarray) -> np.ndarray:
        pts = np.zeros((n_pts, 3))
        pts[:, fixed_axis] = fixed_val
        pts[:, other_axes[0]] = xy[:, 0]
        pts[:, other_axes[1]] = xy[:, 1]
        return pts

    if not is_prism:
        x = _tet_exact_locate_on_face(cell_nodes, fixed_axis, fixed_val, targets_phys)
        resid_norm = np.linalg.norm(map_ref_points(is_prism, full_points(x), cell_nodes) - targets_phys, axis=1)
        final_resid = np.max(resid_norm)
        warn_tol = max(1e-9, _ACCEPT_WARN_REL * scale)
        if final_resid > warn_tol:
            raise RuntimeError(
                f"Tet face exact point-location failed to converge: max residual {final_resid:.3e} "
                f"({100 * final_resid / scale:.2f}% of local face scale {scale:.3e}). "
                f"This indicates a genuinely non-conforming mesh face (target point not actually "
                f"coplanar with this tet's face) rather than a numerical precision issue."
            )
        return x, final_resid

    x = np.zeros((n_pts, 2))  # 初值取面中心 (0,0)
    eps = 1e-6
    resid_norm = np.linalg.norm(map_ref_points(is_prism, full_points(x), cell_nodes) - targets_phys, axis=1)
    tol_converge = max(1e-13, _NEWTON_TOL_REL * scale)

    for _ in range(_NEWTON_MAX_ITER):
        if np.max(resid_norm) < tol_converge:
            break

        phys = map_ref_points(is_prism, full_points(x), cell_nodes)
        residual = phys - targets_phys  # (n_pts,3)

        x_p0 = x.copy(); x_p0[:, 0] += eps
        x_p1 = x.copy(); x_p1[:, 1] += eps
        phys_p0 = map_ref_points(is_prism, full_points(x_p0), cell_nodes)
        phys_p1 = map_ref_points(is_prism, full_points(x_p1), cell_nodes)
        J0 = (phys_p0 - phys) / eps  # (n_pts,3) d(phys)/d(x0)
        J1 = (phys_p1 - phys) / eps  # (n_pts,3) d(phys)/d(x1)

        # 逐点最小二乘解 2x2 正规方程 (J^T J) dx = -J^T residual（3方程2未知量，
        # 对确实落在该面上的目标点是相容超定系统，最小二乘解=精确解）。
        # 求解前先把 J0,J1 各自按列归一化（Jacobi 预条件）：偏斜（对角线
        # 方向远长于另一方向）单元的两个参考方向映射到物理空间的尺度可以
        # 相差数倍，直接对原始 J0,J1 形成正规方程会把这个尺度差平方，
        # 显著放大条件数（对四面体面这种严格仿射映射，理论上一步就该精确
        # 收敛，条件数差导致的舍入误差却会把残差顶在 1e-3 量级的精度地板
        # 上、永远到不了收敛判据——已在真实网格的偏斜单元上复现、验证
        # 该修复后收敛到机器精度）。归一化后 a11=a22=1，det=1-a12^2=
        # sin^2(J0,J1 夹角)，只在两个参考方向映射到几乎平行的物理方向时
        # 才会真正病态（这才是物理上无法定位，而不是数值精度问题）。
        J0_norm = np.maximum(np.linalg.norm(J0, axis=1), 1e-300)
        J1_norm = np.maximum(np.linalg.norm(J1, axis=1), 1e-300)
        J0_hat = J0 / J0_norm[:, None]
        J1_hat = J1 / J1_norm[:, None]

        a11 = np.einsum("pi,pi->p", J0_hat, J0_hat)
        a12 = np.einsum("pi,pi->p", J0_hat, J1_hat)
        a22 = np.einsum("pi,pi->p", J1_hat, J1_hat)
        b1 = -np.einsum("pi,pi->p", J0_hat, residual)
        b2 = -np.einsum("pi,pi->p", J1_hat, residual)

        det = a11 * a22 - a12 * a12
        det_safe = np.where(np.abs(det) < 1e-300, 1e-300, det)
        dx0_hat = (b1 * a22 - b2 * a12) / det_safe
        dx1_hat = (a11 * b2 - a12 * b1) / det_safe
        dx = np.column_stack([dx0_hat / J0_norm, dx1_hat / J1_norm])

        # 回溯线搜索（逐点独立）：棱柱侧面是双线性曲面，满步 Newton 对强
        # 非线性/畸变单元可能过冲甚至发散，用简单的按点减半步长直到残差
        # 下降，比放宽收敛容差更能保证物理正确性（不掩盖真正的不相容面）。
        step = np.ones(n_pts)
        x_best, resid_best = x, resid_norm
        for _ls in range(20):
            x_trial = x + step[:, None] * dx
            resid_trial = np.linalg.norm(
                map_ref_points(is_prism, full_points(x_trial), cell_nodes) - targets_phys, axis=1
            )
            improved = resid_trial < resid_best
            x_best = np.where(improved[:, None], x_trial, x_best)
            resid_best = np.where(improved, resid_trial, resid_best)
            if np.all(improved) or np.all(step < 1e-8):
                break
            step = np.where(improved, step, step * 0.5)
        x, resid_norm = x_best, resid_best

    final_resid = np.max(resid_norm)
    warn_tol = max(1e-9, _ACCEPT_WARN_REL * scale)
    if final_resid > warn_tol:
        raise RuntimeError(
            f"Newton face point-location failed to converge: max residual {final_resid:.3e} "
            f"({100 * final_resid / scale:.2f}% of local face scale {scale:.3e}). "
            f"This indicates a genuinely non-conforming mesh face (target point not actually "
            f"on this cell's face) rather than a slow-convergence issue."
        )
    return x, final_resid


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
    而是存 sources 列表：每个元素是 (real_cell_id, (n_fp, n_sps) 矩阵)，
    矩阵在不属于该 real_cell 的 FP 行上恒为 0——调用方对列表求和即可得到
    正确的、按物理位置精确来源组装出的界面场，不管背后是 1 个还是 2 个
    真实相邻单元。*_is_primary 标记这条 face_connectivity 记录是否是其
    (cell, 立方体面) 分组里"负责触发一次自身外插+校正投影"的那条——非
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

    owner_axis: int
    owner_side: float
    neighbor_axis: int
    neighbor_side: float
    neighbor_sources: List[tuple]
    owner_sources: List[tuple]
    true_normal: np.ndarray
    true_area_weight: np.ndarray
    owner_is_primary: bool = True
    neighbor_is_primary: bool = True


def face_ref_grid(n1d: int, axis: int, side: float, sps_1d: np.ndarray) -> np.ndarray:
    other_axes = [a for a in range(3) if a != axis]
    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    pts = np.zeros((n1d * n1d, 3))
    pts[:, axis] = side
    pts[:, other_axes[0]] = g1.ravel()
    pts[:, other_axes[1]] = g2.ravel()
    return pts


def cell_info(mesh, cell_id: int):
    is_prism = cell_id < mesh.n_prism_cells
    node_ids = (
        mesh._fixed_prism_conn[cell_id] if is_prism else mesh._fixed_tet_conn[cell_id - mesh.n_prism_cells]
    )
    return is_prism, mesh._node_coords[node_ids]


def build_cross_interp(
    mesh,
    n1d: int,
    sps_1d: np.ndarray,
    target_cell: int,
    target_axis: int,
    target_side: float,
    source_phys: np.ndarray,
    char_length: float = 1.0,
    translation: np.ndarray = None,
) -> tuple:
    """求 target_cell 的解在给定 source_phys 目标物理点集上的取值算子，
    形状 (n_source_pts, n_sps)。target_cell 的固定面由 (target_axis,target_side) 给定。

    Args:
        translation: (3,) 或 None。周期边界配对面专用（见
            grid/face_connectivity.py::FRFaceConnectivity.face_translation
            文档）——source_phys 是"来源"侧面上的真实物理坐标，但周期面
            物理上不重合，不能直接拿去在 target_cell（"目标"侧、位于
            周期像位置）里定位，必须先减去平移量，把目标点从"来源"侧
            的物理坐标系平移到"目标"侧的物理坐标系。非周期面（绝大多数
            调用）传 None，等价于零平移。
    """
    is_prism, cell_nodes = cell_info(mesh, target_cell)
    search_phys = source_phys if translation is None else source_phys - translation[np.newaxis, :]
    free_coords, final_resid = newton_locate_on_face(
        is_prism, cell_nodes, target_axis, target_side, search_phys, char_length=char_length
    )

    other_axes = [a for a in range(3) if a != target_axis]
    w0 = compute_lagrange_weights_batch(sps_1d, free_coords[:, 0])  # (n_pts, n1d)
    w1 = compute_lagrange_weights_batch(sps_1d, free_coords[:, 1])  # (n_pts, n1d)
    w_fixed = compute_lagrange_weights_batch(sps_1d, np.array([target_side]))[0]  # (n1d,)

    n_pts = source_phys.shape[0]
    n_sps = n1d**3
    # SPs 索引 (i_a,i_b,i_c) 展平顺序：flat = i_a*n1d^2 + i_b*n1d + i_c
    axis_weight = {target_axis: w_fixed[None, :].repeat(n_pts, axis=0), other_axes[0]: w0, other_axes[1]: w1}
    w_a, w_b, w_c = axis_weight[0], axis_weight[1], axis_weight[2]
    # interp[p, ia*n1d^2+ib*n1d+ic] = w_a[p,ia]*w_b[p,ib]*w_c[p,ic]
    tmp = np.einsum("pa,pb->pab", w_a, w_b).reshape(n_pts, n1d * n1d)
    interp = np.einsum("pab,pc->pabc", tmp.reshape(n_pts, n1d, n1d), w_c).reshape(n_pts, n_sps)
    return interp, final_resid
