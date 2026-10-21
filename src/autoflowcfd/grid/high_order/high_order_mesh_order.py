"""
AutoFlowCFD V2.0 - HighOrderMesh 的阶数相关几何构建 (G-01/G-03, CL-02)

参考点集生成、（含体积项去混叠用的 fine 网格）Jacobian
批量计算、Order Continuation 的按阶数几何缓存/切换。签名以
`mesh: HighOrderMesh` 为第一参数，HighOrderMesh 上保留同名薄委托
方法，调用方式不变。
"""
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional

import numpy as np
from loguru import logger

from autoflowcfd.fr.operators import generate_fr_operators

if TYPE_CHECKING:
    from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh

# Jacobian 计算函数族在 `order_jacobians.py`
from .order_jacobians import (
    _combine_prism_and_tet_jacobians,
    build_fine_metrics,
    compute_native_prism_jacobians,
    compute_native_tet_jacobians,
)


class OrderGeometry(NamedTuple):
    """一个阶数下的全部几何量（`build_order_geometry` 的返回值）。"""
    sps_coords: np.ndarray                              # (n_cells, n_sps_per_cell, 3)
    jacobians: Dict[str, np.ndarray]                    # "det_jacs"/"inv_jacs"/"scaled_quality"
    jacobians_fine: Optional[Dict[str, np.ndarray]]     # 过积分细点度量；order==0 时为 None
    n_sps_per_cell_fine: int                            # 棱柱段细点度量的每单元宽度；order==0 时为 0


def build_order_geometry(mesh: "HighOrderMesh", order: int) -> OrderGeometry:
    """在给定阶数下，从已修正朝向的 connectivity/节点坐标重新推导
    SPs 物理坐标与 Jacobian（不依赖 mesh.order/mesh.n_points_1d 的当前值，
    可在切换阶数*之前*安全调用）。

    Order Continuation（CL-02）的核心前提：FR 方法的解自由度与几何量
    必须共享同一组 SPs——只换 FR 微分算子（mesh.operators）而不重新
    推导这里的量，P0/P1 阶段的梯度/残差计算会直接用错误维度的
    Jacobian（真实网格已复现：reshape 到 27 SPs/单元 的 Jacobian 硬套
    1 SP/单元 的状态场，直接崩溃）。

    四面体走 `compute_native_tet_jacobians`/`map_native_tet_to_physical`（直边常数 Jacobian 广播到全部槽位），
    棱柱走 `compute_native_prism_jacobians`/`map_native_prism_to_physical`（原生节点上逐点求值）；两段按
    棱柱在前、四面体在后拼接（`_combine_prism_and_tet_jacobians`）。
    """
    n_sps_per_cell = generate_fr_operators(order).n_sps
    sps_coords = np.zeros((mesh.n_cells, n_sps_per_cell, 3))
    n_prisms = mesh.n_prism_cells
    node_coords = mesh.loaded_node_coords()
    prism_conn, tet_conn = mesh._fixed_prism_conn, mesh._fixed_tet_conn
    n_tets = len(tet_conn) if tet_conn is not None else 0

    # 棱柱的解点位置与度量用原生基（见 `fr/native_prism/__init__.py`）：原生微分算子必须配原生节点上采样的场
    from autoflowcfd.fr.native_prism.basis import build_native_prism_nodes, map_native_prism_to_physical

    ref_native_prism = build_native_prism_nodes(order)
    if prism_conn is not None and n_prisms > 0:
        n_native_prism = ref_native_prism.shape[0]
        for i in range(n_prisms):
            cell_nodes = node_coords[prism_conn[i]]
            phys = map_native_prism_to_physical(ref_native_prism, cell_nodes)
            sps_coords[i, :n_native_prism] = phys
            # 填充行复制真实 SP #0（有限、物理上合法的占位值，不能留 np.zeros——那对应原点）
            sps_coords[i, n_native_prism:] = phys[0]

    if tet_conn is not None and n_tets > 0:
        from autoflowcfd.fr.native_tet.basis import build_native_tet_operators, map_native_tet_to_physical

        ref_native, _ = build_native_tet_operators(order)
        n_native = ref_native.shape[0]
        for i in range(n_tets):
            cell_nodes = node_coords[tet_conn[i]]
            phys_native = map_native_tet_to_physical(ref_native, cell_nodes)
            sps_coords[n_prisms + i, :n_native] = phys_native
            # 填充行：复制真实 SP #0 的物理坐标（有限、物理上合法的占位值，
            # 见 Part8 文档"一、核心不变量"第4点——不能留 np.zeros 默认值,
            # 那对应原点，可能被后处理/可视化误当成真实几何位置）。
            sps_coords[n_prisms + i, n_native:] = phys_native[0]

    prism_jacobians = (compute_native_prism_jacobians(mesh, ref_native_prism, n_sps_per_cell, want_scaled_quality=True)
                       if n_prisms > 0 else None)
    tet_jacobians = compute_native_tet_jacobians(mesh, order, n_sps_per_cell, want_scaled_quality=True)
    jacobians = _combine_prism_and_tet_jacobians(prism_jacobians, tet_jacobians)

    # 体积项去混叠（over-integration）用的细网格几何。order==0（P0）没有
    # 意义（P0 走独立的有限体积残差路径，见 `fr_residual/inviscid.py::
    # compute_inviscid_residual_fr` 的 n_points_1d==1 分支），跳过以节省
    # 内存/构建时间。
    #
    # 过积分阶数与细点数只取自 `fr/overintegration_order.py`（与 `fr/operators/build.py::generate_fr_operators`
    # 同一个来源，三个过积分算子的形状才与这里的 jacobians_fine 对得上）。
    #
    # `n_sps_per_cell_fine` 是**棱柱**段细点度量的每单元宽度 `(oo+1)^2(oo+2)/2`。四面体段逐单元只存一份（直边
    # 四面体的 Jacobian 逐单元常数），过积分按各自的 n_fine 广播（见 `order_jacobians.build_fine_metrics` 与
    # `core/fr_operators/volume_contract.get_overintegration_context`），不占按棱柱细点数展开的内存。
    #
    # 棱柱的细点度量**必须逐点求值**：棱柱即便直边也一般随点变化，只有顶面是底面纯平移的右棱柱才恒定，不能像
    # 四面体那样广播；而且必须在原生棱柱细点上求（原生微分算子配别的点集采样的场不会报错、只会给出错的导数）。
    jacobians_fine = None
    n_sps_per_cell_fine = 0
    if order >= 1:
        from autoflowcfd.fr.overintegration_order import (
            overintegration_order, prism_n_fine,
        )

        over_order = overintegration_order(order)
        n_sps_per_cell_fine = prism_n_fine(over_order)

        if n_prisms == 0:
            prism_jacobians_fine = None
        else:
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

        # 直边四面体的 Jacobian 逐单元常数：每单元只算一份（见 `build_fine_metrics`）
        tet_jacobians_fine = compute_native_tet_jacobians(mesh, order, 1, want_scaled_quality=False)
        from autoflowcfd.fr.triangle_apex import cell_triangle_slots

        jacobians_fine = build_fine_metrics(
            prism_jacobians_fine, tet_jacobians_fine, n_sps_per_cell_fine, n_prisms, n_tets,
            cell_triangle_slots(tet_conn, prism_conn))

    return OrderGeometry(sps_coords=sps_coords, jacobians=jacobians, jacobians_fine=jacobians_fine,
                         n_sps_per_cell_fine=n_sps_per_cell_fine)


_ORDER_FIELDS = ("n_points_1d", "n_sps_per_cell", "sps_coords", "jacobians", "operators", "face_flux_points",
                 "jacobians_fine", "n_sps_per_cell_fine")


def cache_active_order(mesh: "HighOrderMesh") -> None:
    """把网格当前活动阶数的全部阶数相关几何记进按阶数缓存（网格加载与 `set_order` 共用这一份字段清单）。"""
    mesh._order_geometry_cache[mesh.order] = {f: getattr(mesh, f) for f in _ORDER_FIELDS}
    mesh._active_order = mesh.order


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

        # 临时切到新阶数的基础几何量：build_face_flux_points 直接读取 mesh.n_points_1d /
        # mesh.jacobians / mesh.operators，必须先落地才能调用。
        mesh.order = order
        mesh.n_points_1d = order + 1
        mesh.operators = generate_fr_operators(order)
        mesh.n_sps_per_cell = mesh.operators.n_sps
        mesh.sps_coords = geom.sps_coords
        mesh.jacobians = geom.jacobians
        mesh.jacobians_fine = geom.jacobians_fine
        mesh.n_sps_per_cell_fine = geom.n_sps_per_cell_fine

        mesh.face_flux_points = None
        if mesh.face_connectivity is not None:
            from autoflowcfd.fr.face_flux_points.merge import build_face_flux_points

            logger.info(f"Order continuation: building Flux Points geometry for P{order}...")
            mesh.face_flux_points = build_face_flux_points(mesh.face_connectivity, mesh)
            logger.info(f"Order continuation: Flux Points geometry built for P{order}")

        cache_active_order(mesh)
        return

    mesh.order = order
    for field, value in mesh._order_geometry_cache[order].items():
        setattr(mesh, field, value)
    mesh._active_order = order
