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

# Jacobian 计算函数族已拆到 `order_jacobians.py`（2026-09-19，项目
# "单文件不超 500 行"规范）。这里 import 回来，既有调用点
# （`high_order_mesh.py` 的薄委托方法、测试）不用改。
from .order_jacobians import (  # noqa: F401
    _combine_prism_and_tet_jacobians,
    _compute_prism_only_jacobians,
    _verify_tet_fine_metric_is_cellwise_constant,
    compute_jacobians_at_ref_points,
    compute_native_prism_jacobians,
    compute_native_tet_jacobians,
)


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

    # 棱柱的解点位置与度量按**当前生效的棱柱基**分派（见
    # `fr/native_prism/mode.py`）：原生基的节点不是张量积立方体点，把原生
    # 微分算子套到坍缩节点采样的场上不会报错、只会给出错的导数，所以算子
    # 与几何**必须一起**切换。
    # 棱柱恒为原生基（坍缩档已于 2026-09-23 删除）。保留这个局部名而不是
    # 把下面几处 `if prism_native:` 一并摊平：它在本函数里同时充当"原生档
    # 要额外构造的那几样东西"的作用域标记，摊平会让 coarse/fine 两段共用
    # 的那批 import 失去显式边界。
    prism_native = True

    ref_native_prism = None
    if prism_native:
        # 原生档要用到的三样东西一次导入（下面 coarse 的 sps_coords/
        # Jacobian 与再下面 fine 的细点几何都在同一个 `prism_native`
        # 守卫下使用它们）。
        from autoflowcfd.fr.native_prism.basis import (
            build_native_prism_nodes,
            map_native_prism_to_physical,
        )

        ref_native_prism = build_native_prism_nodes(order)

    if mesh._fixed_prism_conn is not None and n_prisms > 0:
        if prism_native:
            n_native_prism = ref_native_prism.shape[0]
            for i in range(n_prisms):
                cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
                phys = map_native_prism_to_physical(ref_native_prism, cell_nodes)
                sps_coords[i, :n_native_prism] = phys
                # 填充行复制真实 SP #0（有限、物理上合法的占位值，与
                # 四面体那条同一个约定——不能留 np.zeros 默认值，那对应
                # 原点、可能被后处理/可视化误当成真实几何位置）。
                sps_coords[i, n_native_prism:] = phys[0]
        else:
            for i in range(n_prisms):
                cell_nodes = mesh._node_coords[mesh._fixed_prism_conn[i]]
                sps_coords[i] = map_prism_to_physical(ref_cube_sps, cell_nodes)

    if n_tets > 0:
        from autoflowcfd.fr.native_tet.basis import build_native_tet_operators, map_native_tet_to_physical

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

    if n_prisms == 0:
        prism_jacobians = None
    elif prism_native:
        prism_jacobians = compute_native_prism_jacobians(
            mesh, ref_native_prism, n_sps_per_cell, want_scaled_quality=True)
    else:
        prism_jacobians = _compute_prism_only_jacobians(
            mesh, mapper, ref_cube_sps, want_scaled_quality=True)
    tet_jacobians = compute_native_tet_jacobians(mesh, order, n_sps_per_cell, want_scaled_quality=True)
    jacobians = _combine_prism_and_tet_jacobians(prism_jacobians, tet_jacobians)

    # 体积项去混叠（over-integration）用的细网格几何。order==0（P0）没有
    # 意义（P0 走独立的有限体积残差路径，见 `fr_residual/inviscid.py::
    # compute_inviscid_residual_fr` 的 n_points_1d==1 分支），跳过以节省
    # 内存/构建时间。
    #
    # **过积分阶数与细点数一律走 `fr/overintegration_order.py`**
    # （2026-09-19）：此前这里和 `fr/operators.py::generate_fr_operators`
    # 各写一遍同一个 `min(rule*order, cap)`，两处注释都在提醒"必须逐字
    # 一致，否则三个算子的形状与这里的 jacobians_fine 对不上"。靠注释维持
    # 一致在本项目已经出过真实缺陷（滤波档双解析器、CFL 三处硬编码兜底），
    # 而原生/坍缩分档又要再叠一层，所以合并成唯一入口。
    #
    # `n_sps_per_cell_fine` 是 `jacobians_fine` 的**每单元布局宽度**，由
    # 棱柱决定（坍缩档 `(oo+1)^3`、原生档 `(oo+1)^2(oo+2)/2`）。四面体段
    # **不受它约束**：直边四面体的 Jacobian 逐单元常数，
    # `compute_native_tet_jacobians` 把同一个常数写满全部槽位，而过积分
    # 只取第 0 列广播（见 `core/fr_operators/volume_contract.
    # get_overintegration_context`），所以四面体可以用更高的 over_order
    # 而不需要把这个共用数组加宽。
    #
    # 棱柱的细点度量**必须逐点求值**（两档都是）：棱柱即便直边也一般随点
    # 变化，只有顶面是底面纯平移的右棱柱才恒定，不能像四面体那样广播。
    # 原生档下要在**原生**细点上求（原生微分算子配坍缩点采样的场不会报错、
    # 只会给出错的导数），所以两档各自生成自己的参考点集。
    jacobians_fine = None
    n_sps_per_cell_fine = 0
    if order >= 1:
        from autoflowcfd.fr.overintegration_order import (
            prism_n_fine, resolve_prism_overintegration_order,
        )

        over_order = resolve_prism_overintegration_order(order)
        n_sps_per_cell_fine = prism_n_fine(over_order)

        if n_prisms == 0:
            prism_jacobians_fine = None
        elif prism_native:
            ref_fine_prism = build_native_prism_nodes(over_order)
            if ref_fine_prism.shape[0] != n_sps_per_cell_fine:
                raise ValueError(
                    f"原生棱柱细点数不一致：节点生成给出 "
                    f"{ref_fine_prism.shape[0]}，`prism_n_fine` 给出 "
                    f"{n_sps_per_cell_fine}（over_order={over_order}）——"
                    f"后者是 jacobians_fine 的布局宽度，必须相同")
            prism_jacobians_fine = compute_native_prism_jacobians(
                mesh, ref_fine_prism, n_sps_per_cell_fine,
                want_scaled_quality=False)
        else:
            from autoflowcfd.fr.operators import gauss_legendre

            fine_1d, _ = gauss_legendre(over_order + 1)
            xf, yf, zf = np.meshgrid(fine_1d, fine_1d, fine_1d, indexing="ij")
            ref_cube_sps_fine = np.column_stack(
                [xf.ravel(), yf.ravel(), zf.ravel()])
            prism_jacobians_fine = _compute_prism_only_jacobians(
                mesh, mapper, ref_cube_sps_fine, want_scaled_quality=False)

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
            from autoflowcfd.fr.face_flux_points.merge import build_face_flux_points

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
