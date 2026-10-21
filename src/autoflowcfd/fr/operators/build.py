"""AutoFlowCFD V2.0 - FR 算子构建流程 `generate_fr_operators`（容器定义在 `container.py`）。"""

from ..overintegration_order import overintegration_order, prism_n_fine
from .container import FROperators


def generate_fr_operators(order: int) -> FROperators:
    """
    生成完整的 FR 算子集合。

    Args:
        order: 多项式阶数 P

    棱柱与四面体都是原生基（`fr/native_prism/`、`fr/native_tet/`），界面项是 DG 提升算子（nodal DG），没有
    一维张量积 FR 的校正函数族（VCJH η_p）这一自由度。

    Returns:
        operators: 包含所有预计算算子的 FROperators 对象
    """
    # 解点存储宽度（`fr/native_padding.py::sps_width`，原生棱柱真实自由度）：四面体的真实自由度零填充到这个
    # 宽度（"零填充块对角"不变量）；过积分算子的粗网格轴也填充到这个宽度。
    from autoflowcfd.fr.native_padding import sps_width

    n_sps_global = sps_width(order)

    # 四面体原生单纯形基。
    from ..native_tet.basis import (
        build_native_tet_operators, build_native_tet_boundary_extrap, build_native_tet_lift,
    )
    from ..native_padding import (
        pad_native_matrix_to_global,
    )

    ref_native_tet, D_native_tet = build_native_tet_operators(order)
    n_native_sps_tet = ref_native_tet.shape[0]
    # 面外插 / 提升按（面, 坍缩顶点槽位）全部构造（fr/triangle_apex.py），组装成按面算子
    # 索引排好的整表（face_ops.py，见下方 face_extrap_by_op）
    from ..triangle_apex import N_TRI_SLOTS
    tet_extrap_slots = {(v, sl): build_native_tet_boundary_extrap(order, v, sl)
                        for v in range(4) for sl in range(N_TRI_SLOTS)}
    tet_lift_slots = {(v, sl): build_native_tet_lift(order, v, sl)
                      for v in range(4) for sl in range(N_TRI_SLOTS)}
    D_native_tet_padded = pad_native_matrix_to_global(D_native_tet, n_sps_global, pad_axes=(0, 1))

    # 棱柱原生基（唯一实现，见 fr/native_prism/__init__.py）。
    #
    # 原生 `D` 作用在**原生棱柱节点**上的场；`grid/high_order/high_order_mesh_order.py` 的 `build_order_geometry`
    # 用同一组原生棱柱节点求解点坐标与雅可比。
    from ..native_prism.basis import build_native_prism_operators
    from ..native_prism.face import build_all_native_prism_face_operators

    ref_native_prism, D_native_prism = build_native_prism_operators(order)
    n_native_sps_prism = ref_native_prism.shape[0]
    prism_extrap_slots, prism_lift_slots = build_all_native_prism_face_operators(order)
    D_native_prism_padded = pad_native_matrix_to_global(D_native_prism, n_sps_global, pad_axes=(0, 1))

    # 体积项去混叠（过积分）算子：order==0 时全部为 None（P0 走独立有限体积路径，不需要）。
    #
    # 过积分阶数与细点数一律取自 `fr/overintegration_order.py`（与 `grid/high_order/high_order_mesh_order.py`
    # 的 `build_order_geometry` 同一个来源，算子形状与 `mesh.jacobians_fine` 才对得上）。
    overint_order_prism = 0
    n_fine_prism = 0
    n_fine_tet = 0
    overint_order_tet = 0
    overint_interp_c2f_tet = overint_interp_c2f_prism = None
    overint_D_fine_tet = overint_D_fine_prism = None
    overint_restrict_f2c_tet = overint_restrict_f2c_prism = None
    overint_lifted_div_tet = overint_lifted_div_prism = None
    overint_project_f2c_tet = overint_project_f2c_prism = None
    if order >= 1:
        # 棱柱
        overint_order_prism = overintegration_order(order)
        n_fine_prism = prism_n_fine(overint_order_prism)
        # 细网格轴取**真实**细点数（填充槽位恒为零、零贡献，而 `D_fine` 的收缩是 O(n_fine^2)）；粗网格轴填充到
        # `n_sps_global`（`Q` 数组的存储宽度），理由见 `native_prism/overintegration.py` 模块文档。
        from ..native_prism.overintegration import (
            build_native_prism_l2_projection, build_native_prism_overintegration_operators,
        )

        (_ref_fine_np, _c2f_np, overint_D_fine_prism,
         _f2c_np) = build_native_prism_overintegration_operators(
            order, overint_order_prism)
        if _ref_fine_np.shape[0] != n_fine_prism:
            raise ValueError(
                f"原生棱柱细点数不一致：算子给出 "
                f"{_ref_fine_np.shape[0]}，`prism_n_fine` 给出 "
                f"{n_fine_prism}（order={order}, "
                f"over_order={overint_order_prism}）——两者必须相同，"
                f"后者同时是 mesh.n_sps_per_cell_fine 的布局宽度")
        overint_interp_c2f_prism = pad_native_matrix_to_global(
            _c2f_np, n_sps_global, pad_axes=(1,))
        overint_restrict_f2c_prism = pad_native_matrix_to_global(
            _f2c_np, n_sps_global, pad_axes=(0,))
        # 平均流体积项的细->粗限制是 L2 投影（离散守恒；湍流输运仍用上面的插值，分工见
        # native_tet/overintegration.py 模块文档）
        overint_project_f2c_prism = pad_native_matrix_to_global(
            build_native_prism_l2_projection(order, overint_order_prism), n_sps_global, pad_axes=(0,))

        # 四面体：细网格轴取真实细点数 `(oo+1)(oo+2)(oo+3)/6`；粗网格轴（interp 的列、restrict / project 的行）
        # 按"零填充块对角"填充到 `n_sps_global`（`pad_native_matrix_to_global` 一次只处理同一个 native 长度的
        # 轴集合，所以两个轴分开处理）。
        from ..native_tet.overintegration import (
            build_native_tet_l2_projection,
            build_native_tet_overintegration_operators,
        )

        # 四面体段的细点度量在 `mesh.jacobians_fine` 里逐单元只存一份、使用时广播（直边四面体的 Jacobian 逐单元
        # 为常数），所以 `n_fine_tet` 不受棱柱细点宽度约束。阶数 `2*order` 的依据（实测去混叠误差在
        # `oo = 2*order` 处断崖式下降）见 `fr/native_tet/overintegration.py` 模块文档与
        # `tests/unit/test_overintegration_cap_cost.py`。
        overint_order_tet = overintegration_order(order)

        ref_fine_native, interp_c2f_native, D_fine_native, restrict_f2c_native = (
            build_native_tet_overintegration_operators(order, overint_order_tet)
        )
        # 细网格轴不填充（实测 20000 个四面体单元完整链路：P1 加速 3.04x、P2 加速 4.63x，结果逐位相同或差在舍入）；
        # 粗网格轴必须填充到 n_sps_global：`Q` 数组是 `(n_cells, n_sps_global, 5)` 的填充布局。
        n_fine_tet = ref_fine_native.shape[0]

        # D_fine：两个轴都是细网格轴 -> 完全不填充
        overint_D_fine_tet = D_fine_native
        # interp c2f：列（粗轴）填充到 n_sps_global，行（细轴）保持真实长度
        overint_interp_c2f_tet = pad_native_matrix_to_global(
            interp_c2f_native, n_sps_global, pad_axes=(1,)
        )
        # restrict f2c：行（粗轴）填充到 n_sps_global，列（细轴）保持真实长度
        overint_restrict_f2c_tet = pad_native_matrix_to_global(
            restrict_f2c_native, n_sps_global, pad_axes=(0,)
        )
        overint_project_f2c_tet = pad_native_matrix_to_global(
            build_native_tet_l2_projection(order, overint_order_tet), n_sps_global, pad_axes=(0,))
        # 修正项本侧通量 = 细层通量多项式在面通量点上的法向迹，并进体积算子（离散守恒、
        # 按三角形面坍缩顶点槽位组合各一份，见 fr/face_flux_trace.py）
        from ..face_flux_trace import (
            build_lifted_divergence, build_prism_face_flux_trace, build_tet_face_flux_trace,
        )

        def _padded_lift(table):
            return lambda f, s: pad_native_matrix_to_global(table[(f, s)], n_sps_global, pad_axes=(0,))

        overint_lifted_div_tet = build_lifted_divergence(
            overint_project_f2c_tet, overint_D_fine_tet, _padded_lift(tet_lift_slots),
            build_tet_face_flux_trace(order, overint_order_tet), order, "tet")
        overint_lifted_div_prism = build_lifted_divergence(
            overint_project_f2c_prism, overint_D_fine_prism, _padded_lift(prism_lift_slots),
            build_prism_face_flux_trace(order, overint_order_prism), order, "prism")

    from .face_ops import build_face_op_tables
    face_extrap_by_op, face_lift_by_op = build_face_op_tables(
        tet_extrap_slots, tet_lift_slots, prism_extrap_slots, prism_lift_slots, n_sps_global)

    return FROperators(
        n_sps=n_sps_global,
        overint_order_prism=overint_order_prism,
        overint_n_fine_prism=n_fine_prism,
        overint_n_fine_tet=n_fine_tet,
        overint_order_tet=overint_order_tet,
        overint_interp_c2f_tet=overint_interp_c2f_tet,
        overint_interp_c2f_prism=overint_interp_c2f_prism,
        overint_D_fine_tet=overint_D_fine_tet,
        overint_D_fine_prism=overint_D_fine_prism,
        overint_restrict_f2c_tet=overint_restrict_f2c_tet,
        overint_restrict_f2c_prism=overint_restrict_f2c_prism,
        overint_lifted_div_tet=overint_lifted_div_tet,
        overint_lifted_div_prism=overint_lifted_div_prism,
        D_native_tet=D_native_tet,
        ref_native_tet=ref_native_tet,
        n_native_sps_tet=n_native_sps_tet,
        D_native_tet_padded=D_native_tet_padded,
        D_native_prism=D_native_prism,
        ref_native_prism=ref_native_prism,
        n_native_sps_prism=n_native_sps_prism,
        D_native_prism_padded=D_native_prism_padded,
        face_extrap_by_op=face_extrap_by_op,
        face_lift_by_op=face_lift_by_op,
    )
