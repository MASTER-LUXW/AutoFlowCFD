"""
AutoFlowCFD V2.0 - HighOrderMesh 的阶数相关几何构建 (G-01/G-03, CL-02)

从 high_order_mesh.py 拆出来（控制单文件行数，>400 行需拆分的项目
规范）：参考点集生成、（含体积项去混叠用的 fine 网格）Jacobian
批量计算、Order Continuation 的按阶数几何缓存/切换。签名以
`mesh: HighOrderMesh` 为第一参数，HighOrderMesh 上保留同名薄委托
方法，调用方式不变。
"""

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from loguru import logger

from ..curved_mapping.curved_mapping import (
    CurvedMapping,
    map_prism_to_physical,
    map_tet_to_physical,
)
from autoflowcfd.fr.operators import generate_fr_operators

if TYPE_CHECKING:
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh


def generate_reference_cube_sps(mesh: "HighOrderMesh", order: Optional[int] = None) -> np.ndarray:
    """生成计算立方体 [-1,1]^3 内的张量积 Gauss-Legendre SPs 坐标。

    四面体、棱柱共用同一套计算立方体坐标（Duffy 坍缩坐标的计算域）；
    单元类型差异完全体现在 curved_mapping 的物理映射函数中，这里不再
    像旧版本那样对四面体/棱柱分别生成不同的"参考点"。

    Args:
        order: 目标阶数；None 时使用 mesh.order（当前活动阶数）。
            Order Continuation 需要在切换到某个阶数*之前*为该阶数生成
            SPs，此时 mesh.order 还是旧阶数，必须显式传入。
    """
    from autoflowcfd.fr.operators import gauss_legendre

    n_points_1d = (order + 1) if order is not None else mesh.n_points_1d
    sps_1d, _ = gauss_legendre(n_points_1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def compute_jacobians_at_ref_points(
    mesh: "HighOrderMesh", mapper: CurvedMapping, ref_pts: np.ndarray, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """在给定的参考点集 ref_pts 上，对全部单元算一遍精确 Jacobian
    （逐单元循环，共用给同一 CurvedMapping 实例）——从 build_order_geometry
    拆出来，供体积项去混叠（超过-积分）在 FINE 参考点集上复用
    同一套逻辑，而不是另写一份（CurvedMapping.compute_jacobian 对
    tet/prism 走解析精确公式，不依赖构造时传的 顺序，见该类文档，
    因此同一个 mapper 实例可以安全地在不同参考点集上反复调用）。
    """
    from autoflowcfd.core.fr_operators.troubled_cell import compute_scaled_jacobian_quality

    n_prisms = mesh.n_prism_cells
    n_tets = len(mesh._fixed_tet_conn) if mesh._fixed_tet_conn is not None else 0
    n_total = n_prisms + n_tets
    if n_total == 0:
        return None

    # 预分配直写（2026-08-26，P2 专项 OOM 修复）：原实现先把逐单元的
    # (n_ref,)/(n_ref,3,3) 小数组累积进三个 Python 列表（79 万单元→79 万个
    # 独立小数组，仅对象开销就上百 MB），再 np.concatenate——拼接期间小数组
    # 列表与完整拼接副本同时驻留，峰值直接翻倍；P2 过积分点集（64 pts/单元）
    # 下仅 inv_jacs 就 3.6GB/份→峰值 ~7.3GB，叠加尚未清理的旧阶数几何缓存与
    # 求解器状态，79 万单元生产网格 P1→P2 切换时分配失败崩溃（两次独立复现）。
    # 改为预分配单块大数组逐单元就地写入：峰值降为结果单份 + 单元级瞬态，
    # 数值与原实现逐位一致（写入顺序/拼接顺序都是 prism 在前、tet 在后）。
    n_ref = ref_pts.shape[0]
    all_dets = np.empty(n_total * n_ref)
    all_inv_jacs = np.empty((n_total * n_ref, 3, 3))
    all_scaled_quality = np.empty(n_total * n_ref) if want_scaled_quality else None

    def _fill_cell(i, cell_nodes, cell_type):
        phys_pts = (map_prism_to_physical if cell_type == "prism"
                    else map_tet_to_physical)(ref_pts, cell_nodes)
        jac_data = mapper.compute_jacobian(
            phys_pts, cell_id=i, cell_type=cell_type, cell_nodes=cell_nodes, ref_cube_sps=ref_pts
        )
        lo, hi = i * n_ref, (i + 1) * n_ref
        all_dets[lo:hi] = jac_data["det_jacs"].ravel()
        all_inv_jacs[lo:hi] = jac_data["inv_jacs"].reshape(-1, 3, 3)
        if want_scaled_quality:
            all_scaled_quality[lo:hi] = compute_scaled_jacobian_quality(
                jac_data["jacobians"], jac_data["det_jacs"]
            ).ravel()

    if mesh._fixed_prism_conn is not None and n_prisms > 0:
        for i in range(n_prisms):
            _fill_cell(i, mesh._node_coords[mesh._fixed_prism_conn[i]], "prism")

    if mesh._fixed_tet_conn is not None and n_tets > 0:
        for i in range(n_tets):
            _fill_cell(n_prisms + i, mesh._node_coords[mesh._fixed_tet_conn[i]], "tet")

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result


def compute_native_tet_jacobians(
    mesh: "HighOrderMesh", order: int, n_sps_per_cell: int, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """native 四面体（路径C）版本的 Jacobian 构造——见 Part8 文档"一、
    核心不变量：零填充块对角"第3点：直边单元 Jacobian 是每单元一个
    常数（`compute_native_tet_jacobian`，不依赖参考点位置），不像坍缩
    坐标那样需要在 `ref_cube_sps` 上逐点求值；这个常数原样广播填满
    该单元全部 `n_sps_per_cell` 槽位（真实 `n_native` 行与填充行一视
    同仁，因为它们对应同一个 det_j——这也是本设计"填充行必须有限"这个
    强制要求最容易满足的一种情形：native 直边单元根本没有"填充行专属
    的合理值"这个问题，复用同一个真实常数即可，不需要另外编造占位值）。

    `scaled_quality`（诊断量）用与坍缩坐标分支
    `troubled_cell.py::compute_scaled_jacobian_quality` 完全相同的定义
    （`det_j/prod(col_norms(J))`）手算——直边单元只有一个 J，不需要
    调用那个批量/逐点版本。
    """
    from autoflowcfd.fr.native_simplex_basis import compute_native_tet_jacobian

    n_prisms = mesh.n_prism_cells
    n_tets = len(mesh._fixed_tet_conn) if mesh._fixed_tet_conn is not None else 0
    if n_tets == 0:
        return None

    all_dets = np.empty(n_tets * n_sps_per_cell)
    all_inv_jacs = np.empty((n_tets * n_sps_per_cell, 3, 3))
    all_scaled_quality = np.empty(n_tets * n_sps_per_cell) if want_scaled_quality else None

    for i in range(n_tets):
        cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i]]
        det_j, adj_j = compute_native_tet_jacobian(cell_nodes)
        inv_j = adj_j / det_j
        lo, hi = i * n_sps_per_cell, (i + 1) * n_sps_per_cell
        all_dets[lo:hi] = det_j
        all_inv_jacs[lo:hi] = inv_j
        if want_scaled_quality:
            p0, p1, p2, p3 = cell_nodes
            J = 0.5 * np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            col_norms = np.linalg.norm(J, axis=0)
            quality = det_j / max(np.prod(col_norms), 1e-300)
            all_scaled_quality[lo:hi] = quality

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result


def _verify_tet_fine_metric_is_cellwise_constant(
    tet_part: Optional[Dict[str, np.ndarray]], n_sps_per_cell_fine: int
) -> None:
    """校验四面体细点度量在单元内逐槽位完全相同。

    `get_overintegration_context` 的四面体段只取第 0 列再广播到该段自己的
    `n_fine_tet` 宽度——这让四面体的过积分阶数不再受棱柱布局宽度约束
    （P3 因此能取到理想的 over_order=6，去混叠误差 3.37e-3 -> 4.80e-6）。
    这个等价性依赖"直边单元 Jacobian 不依赖参考点位置"，也就是
    `compute_native_tet_jacobians` 的常数广播。

    判据用**逐位相同**而不是容差：那些值本来就是同一次赋值广播出来的，
    任何差异都说明构造方式变了（例如引入曲边四面体后改成逐点求值），
    那时必须显式改掉广播路径，而不是让它悄悄给出"第 0 个细点的度量"。
    """
    if tet_part is None:
        return
    for key in ("det_jacs", "inv_jacs"):
        arr = tet_part.get(key)
        if arr is None:
            continue
        per_cell = arr.reshape((-1, n_sps_per_cell_fine) + arr.shape[1:])
        if per_cell.shape[0] == 0:
            continue
        ref_col = per_cell[:, :1]
        if not np.array_equal(per_cell, np.broadcast_to(ref_col, per_cell.shape)):
            bad = int((per_cell != np.broadcast_to(ref_col, per_cell.shape)).any(
                axis=tuple(range(1, per_cell.ndim))).sum())
            raise ValueError(
                f"四面体细点度量 '{key}' 在 {bad} 个单元内部不是逐槽位常数"
                f"——`get_overintegration_context` 的四面体段取第 0 列广播的"
                f"前提不再成立（典型原因：引入了曲边四面体、Jacobian 改成"
                f"逐点求值）。必须改掉那条广播，不能让它静默只用第 0 个"
                f"细点的度量。"
            )


def _combine_prism_and_tet_jacobians(
    prism_part: Optional[Dict[str, np.ndarray]], tet_part: Optional[Dict[str, np.ndarray]]
) -> Optional[Dict[str, np.ndarray]]:
    """把只算了棱柱部分（native 四面体分支专用，见 `build_order_geometry`）
    的 jacobians 与单独算好的 native 四面体部分按 prism-在前/tet-在后
    的既有顺序拼接——与 `compute_jacobians_at_ref_points` 一次性算全部
    单元时的拼接顺序保持一致，下游（`mesh.jacobians["det_jacs"].reshape
    (n_cells, n_sps)` 等）不需要关心这里是分两段算的。"""
    if prism_part is None:
        return tet_part
    if tet_part is None:
        return prism_part
    result = {}
    for key in prism_part:
        result[key] = np.concatenate([prism_part[key], tet_part[key]])
    return result


def _compute_prism_only_jacobians(
    mesh: "HighOrderMesh", mapper: CurvedMapping, ref_pts: np.ndarray, want_scaled_quality: bool
) -> Optional[Dict[str, np.ndarray]]:
    """`compute_jacobians_at_ref_points` 的棱柱专属子集——四面体部分走
    `compute_native_tet_jacobians`（native 单纯形基），两部分分别算好
    再按 prism-在前/tet-在后拼接，见 `build_order_geometry`。逻辑与
    `compute_jacobians_at_ref_points` 的棱柱分支逐行一致，只是提取出来
    单独调用，不是重新实现一遍。
    """
    from autoflowcfd.core.fr_operators.troubled_cell import compute_scaled_jacobian_quality

    n_prisms = mesh.n_prism_cells
    if mesh._fixed_prism_conn is None or n_prisms == 0:
        return None
    n_ref = ref_pts.shape[0]
    all_dets = np.empty(n_prisms * n_ref)
    all_inv_jacs = np.empty((n_prisms * n_ref, 3, 3))
    all_scaled_quality = np.empty(n_prisms * n_ref) if want_scaled_quality else None

    for i in range(n_prisms):
        cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
        phys_pts = map_prism_to_physical(ref_pts, cell_nodes)
        jac_data = mapper.compute_jacobian(
            phys_pts, cell_id=i, cell_type="prism", cell_nodes=cell_nodes, ref_cube_sps=ref_pts
        )
        lo, hi = i * n_ref, (i + 1) * n_ref
        all_dets[lo:hi] = jac_data["det_jacs"].ravel()
        all_inv_jacs[lo:hi] = jac_data["inv_jacs"].reshape(-1, 3, 3)
        if want_scaled_quality:
            all_scaled_quality[lo:hi] = compute_scaled_jacobian_quality(
                jac_data["jacobians"], jac_data["det_jacs"]
            ).ravel()

    result = {"det_jacs": all_dets, "inv_jacs": all_inv_jacs}
    if want_scaled_quality:
        result["scaled_quality"] = all_scaled_quality
    return result


def build_order_geometry(mesh: "HighOrderMesh", order: int) -> Dict[str, np.ndarray]:
    """在给定阶数下，从已修正朝向的 connectivity/节点坐标重新推导
    SPs 物理坐标与 Jacobian（不依赖 mesh.order/mesh.n_points_1d 的当前值，
    可在切换阶数*之前*安全调用）。

    Order Continuation（CL-02）的核心前提：FR 方法的解自由度与几何量
    必须共享同一组 SPs——只换 FR 微分算子（mesh.operators）而不重新
    推导这里的量，P0/P1 阶段的梯度/残差计算会直接用错误维度的
    Jacobian（真实网格已复现：reshape 到 27 SPs/单元 的 Jacobian 硬套
    1 SP/单元 的状态场，直接崩溃）。

    四面体（native 单纯形基，路径C，Part6/7/8 文档）部分走
    `compute_native_tet_jacobians`/`map_native_tet_to_physical`（直边
    常数 Jacobian + 零填充，见该函数与 Part8 文档），棱柱部分仍走原有
    坍缩坐标路径——两者按 prism-在前/tet-在后拼接
    （`_combine_prism_and_tet_jacobians`）。

    Returns:
        {"sps_coords", "jacobians", "ref_cube_sps", "jacobians_fine",
        "n_sps_per_cell_fine"}
    """
    n_points_1d = order + 1
    n_sps_per_cell = n_points_1d**3
    sps_coords = np.zeros((mesh.n_cells, n_sps_per_cell, 3))
    mapper = CurvedMapping(order)

    ref_cube_sps = generate_reference_cube_sps(mesh, order)
    n_prisms = mesh.n_prism_cells
    n_tets = len(mesh._fixed_tet_conn) if mesh._fixed_tet_conn is not None else 0

    if mesh._fixed_prism_conn is not None and n_prisms > 0:
        for i in range(n_prisms):
            cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
            sps_coords[i] = map_prism_to_physical(ref_cube_sps, cell_nodes)

    if n_tets > 0:
        from autoflowcfd.fr.native_simplex_basis import build_native_tet_operators, map_native_tet_to_physical

        ref_native, _ = build_native_tet_operators(order)
        n_native = ref_native.shape[0]
        for i in range(n_tets):
            cell_nodes = mesh._node_coords[mesh._fixed_tet_conn[i]]
            phys_native = map_native_tet_to_physical(ref_native, cell_nodes)
            sps_coords[n_prisms + i, :n_native] = phys_native
            # 填充行：复制真实 SP #0 的物理坐标（有限、物理上合法的占位值，
            # 见 Part8 文档"一、核心不变量"第4点——不能留 np.zeros 默认值,
            # 那对应原点，可能被后处理/可视化误当成真实几何位置）。
            sps_coords[n_prisms + i, n_native:] = phys_native[0]

    prism_jacobians = (
        _compute_prism_only_jacobians(mesh, mapper, ref_cube_sps, want_scaled_quality=True)
        if n_prisms > 0 else None
    )
    tet_jacobians = compute_native_tet_jacobians(mesh, order, n_sps_per_cell, want_scaled_quality=True)
    jacobians = _combine_prism_and_tet_jacobians(prism_jacobians, tet_jacobians)

    # 体积项去混叠（超过-积分，V2.0 二次评审 Tier 0 #2）用的
    # 细网格几何：过积分阶数 over_order=2*order，与 fr/operators.py::
    # generate_fr_operators 里构造 overint_interp_c2f_*/overint_D_fine_*
    # 用的过积分阶数必须一致（否则 fr_residual_inviscid.py 里插值/
    # 微分/限制三个算子的形状与这里的 jacobians_fine 对不上）。
    # order==0（P0）没有意义（P0 走独立的有限体积残差路径，见
    # fr_residual_inviscid.py::compute_inviscid_residual_fr 的
    # n_points_1d==1 分支），跳过以节省内存/构建时间。
    #
    # native 单纯形基过积分算子已实现（`native_tet_overintegration.py::
    # build_native_tet_overintegration_operators`，Part8 文档"四·七"节）。
    # **四面体的 over_order 自 2026-09-17 起与棱柱不同**（native PKD 基不
    # 受坍缩基条件数上限约束，见 `fr/native_tet_overintegration.py::
    # NATIVE_TET_OVERINTEGRATION_MAX_ORDER`）：P2 是 4 而棱柱是 3、P3 是 5。
    # 这里算的 `n_sps_per_cell_fine` 仍然只由**棱柱**的 over_order 决定，
    # 它同时充当四面体那一段的**布局宽度**上界（`fr/operators.py` 的
    # `resolve_tet_overintegration_order` 会据此把四面体阶数夹到装得下）。
    #
    # `jacobians_fine` 对四面体单元的构造复用 `compute_native_tet_jacobians`
    # （同一个"直边单元常数 Jacobian 广播"函数，只是这里传入 FINE 网格的
    # `n_sps_per_cell_fine` 计数而不是 coarse 的 `n_sps_per_cell`——直边
    # 单元 Jacobian 不依赖参考点位置，广播到多少个槽位都是同一个常数，
    # 不需要为"fine"专门重新推导）。**正是这个"原样广播"让四面体可以只取
    # 前 `n_fine_tet` 列**：那些列上的度量与在真实细点上求值恒等，所以
    # 四面体用更高的 over_order 不需要把这个共用数组加宽（见
    # `core/fr_operators/volume_contract.get_overintegration_context`）。
    jacobians_fine = None
    n_sps_per_cell_fine = 0
    if order >= 1:
        from autoflowcfd.fr.operators import gauss_legendre
        from autoflowcfd.fr.collapsed_basis import (
            OVERINTEGRATION_MAX_ORDER, resolve_overintegration_order_rule,
        )

        # 过积分阶数规则（2026-09-15 起**可切换**，见
        # collapsed_basis.py::resolve_overintegration_order_rule）：
        # `over_order = min(rule*order, OVERINTEGRATION_MAX_ORDER)`，
        # `AFCFD_OVERINT_ORDER_RULE = 2x | 3x`，**默认 2x**（此前已被
        # 长期验证的行为）。
        #
        # `2x` 是为平均流的**二次**非线性（欧拉通量 x 度量项）设计的经验
        # 法则。去混叠此后被接到了两处**三重**乘积上——k/omega 对流体积项
        # `div(adj(J)*rho*u*phi)` 与粘性体积项
        # `div(adj(J)*G(Q,grad_vel,grad_T,mu_t))`，三个一次场的乘积是三次，
        # `over_order=2` 的细网格（二次空间）表示不了它。
        #
        # **`OVERINTEGRATION_MAX_ORDER = 3` 自 2026-09-17 起只约束棱柱**。
        # 那条"放宽到 4 会让 P2 均匀自由流场残差从 1.06e-5 恶化到 5.6e-3、
        # 根因是 D_fine 绝对量级暴涨约 6.3 万倍"的论证只对**坍缩坐标**基
        # 成立，也就是这里算的棱柱；native 四面体实测在 over_order=6 才
        # cond(V)=3856、`max|D|` 从 3 到 6 只长 3.5 倍，已按自己的上限
        # 独立解析（见上方注释与 collapsed_basis.py 该常量上方的说明、
        # tests/unit/test_native_tet_overintegration_conditioning.py）。
        #
        # 所以 `3x` 与 `2x` 对**棱柱**的差别只在 order=1：
        #   order=1: 2x -> over_order 2（细点 27）； 3x -> 3（细点 64）
        #   order>=2: 两者都被 cap 到 3，完全相同
        # （四面体在各阶数上两者都有区别，但那由 `fr/operators.py` 解析，
        # 不影响这里的 `n_sps_per_cell_fine`。）
        #
        # `3x` 在 order=1 的实测精度收益 / 已知代价，以及"为什么默认不改"，
        # 全部记在 `resolve_overintegration_order_rule` 的文档里。
        over_order = min(
            resolve_overintegration_order_rule() * order,
            OVERINTEGRATION_MAX_ORDER)
        n_points_1d_fine = over_order + 1
        n_sps_per_cell_fine = n_points_1d_fine**3
        fine_1d, _ = gauss_legendre(n_points_1d_fine)
        xf, yf, zf = np.meshgrid(fine_1d, fine_1d, fine_1d, indexing="ij")
        ref_cube_sps_fine = np.column_stack([xf.ravel(), yf.ravel(), zf.ravel()])

        prism_jacobians_fine = (
            _compute_prism_only_jacobians(mesh, mapper, ref_cube_sps_fine, want_scaled_quality=False)
            if n_prisms > 0 else None
        )
        tet_jacobians_fine = compute_native_tet_jacobians(
            mesh, order, n_sps_per_cell_fine, want_scaled_quality=False
        )
        # 过积分的四面体段直接取第 0 列广播（见 `core/fr_operators/
        # volume_contract.get_overintegration_context`），前提是"该单元
        # 全部细点槽位的度量完全相同"。这里显式校验，不默默假设——将来
        # 若引入曲边四面体，这条会当场失败而不是静默给出错误度量。
        _verify_tet_fine_metric_is_cellwise_constant(
            tet_jacobians_fine, n_sps_per_cell_fine)
        jacobians_fine = _combine_prism_and_tet_jacobians(prism_jacobians_fine, tet_jacobians_fine)

    return {
        "sps_coords": sps_coords,
        "jacobians": jacobians,
        "ref_cube_sps": ref_cube_sps,
        "jacobians_fine": jacobians_fine,
        "n_sps_per_cell_fine": n_sps_per_cell_fine,
    }


def set_order(mesh: "HighOrderMesh", order: int) -> None:
    """切换网格当前活动的多项式阶数（Order Continuation 专用）。

    SPs 坐标、Jacobian、Flux 点 几何（含 Newton 面点位定位）全部
    随阶数重新推导——这些量不是"复用同一套再插值"就够的，FR 方法要求
    解自由度与几何在同一组 SPs/FPs 上重合。按阶数缓存：目标阶数（网格
    加载时已经构建过）与之前访问过的阶数直接复用缓存，不重复触发昂贵
    的逐面 Newton 点位定位重建。

    Args:
        order: 目标阶数
    """
    if order == mesh._active_order:
        return

    if order not in mesh._order_geometry_cache:
        geom = build_order_geometry(mesh, order)

        # 临时切到新阶数的基础几何量：build_face_flux_points /
        # precompute_cell_face_misalignment 直接读取 mesh.n_points_1d /
        # mesh.jacobians / mesh.operators，必须先落地才能调用。
        mesh.order = order
        mesh.n_points_1d = order + 1
        mesh.n_sps_per_cell = mesh.n_points_1d**3
        mesh.sps_coords = geom["sps_coords"]
        mesh.jacobians = geom["jacobians"]
        mesh._ref_cube_sps = geom["ref_cube_sps"]
        mesh.jacobians_fine = geom["jacobians_fine"]
        mesh.n_sps_per_cell_fine = geom["n_sps_per_cell_fine"]
        mesh.operators = generate_fr_operators(order)

        face_flux_points = None
        cell_face_misalignment = None
        if mesh.face_connectivity is not None:
            from autoflowcfd.fr.face_flux_points_merge import build_face_flux_points

            logger.info(f"Order continuation: building Flux Points geometry for P{order}...")
            face_flux_points = build_face_flux_points(mesh.face_connectivity, mesh)
            mesh.face_flux_points = face_flux_points
            logger.info(f"Order continuation: Flux Points geometry built for P{order}")

            if mesh.jacobians is not None:
                from autoflowcfd.core.fr_operators.troubled_cell import precompute_cell_face_misalignment

                cell_face_misalignment = precompute_cell_face_misalignment(mesh)
                mesh.cell_face_misalignment = cell_face_misalignment

        mesh._order_geometry_cache[order] = {
            "n_points_1d": mesh.n_points_1d,
            "n_sps_per_cell": mesh.n_sps_per_cell,
            "sps_coords": mesh.sps_coords,
            "jacobians": mesh.jacobians,
            "ref_cube_sps": mesh._ref_cube_sps,
            "operators": mesh.operators,
            "face_flux_points": face_flux_points,
            "cell_face_misalignment": cell_face_misalignment,
            "jacobians_fine": mesh.jacobians_fine,
            "n_sps_per_cell_fine": mesh.n_sps_per_cell_fine,
        }

    cached = mesh._order_geometry_cache[order]
    mesh.order = order
    mesh.n_points_1d = cached["n_points_1d"]
    mesh.n_sps_per_cell = cached["n_sps_per_cell"]
    mesh.sps_coords = cached["sps_coords"]
    mesh.jacobians = cached["jacobians"]
    mesh._ref_cube_sps = cached["ref_cube_sps"]
    mesh.operators = cached["operators"]
    mesh.face_flux_points = cached["face_flux_points"]
    mesh.cell_face_misalignment = cached["cell_face_misalignment"]
    mesh.jacobians_fine = cached["jacobians_fine"]
    mesh.n_sps_per_cell_fine = cached["n_sps_per_cell_fine"]
    mesh._active_order = order
