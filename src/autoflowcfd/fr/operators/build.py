"""AutoFlowCFD V2.0 - FR 算子构建流程 `generate_fr_operators`。

从 `fr/operators.py`（原 705 行）拆出（2026-09-19，项目"单文件不超
500 行"规范）：容器定义在 `container.py`。纯搬家，未改任何逻辑。
"""

import numpy as np

from ..quadrature_points import gauss_legendre
from ..matrix_operators import (
    compute_diff_matrix_1d,
    compute_diff_matrix_3d,
)
from ..overintegration_order import (
    prism_n_fine,
    resolve_overintegration_order_rule,
    resolve_prism_overintegration_order,
)
from ..modal_filter import build_prism_modal_filter
from .container import FROperators


def generate_fr_operators(order: int) -> FROperators:
    """
    生成完整的 FR 算子集合。

    Args:
        order: 多项式阶数 P

    **不再有 `flux_point_type` 形参**（2026-09-24）：FR 校正函数族
    （VCJH η_p）是一维张量积 FR 形式特有的自由度，而原生单纯形/棱柱基的
    界面项是 DG 提升算子、本身就是 nodal DG。删掉坍缩 1D 分布机制之后，
    这个参数唯一影响的四个产物（`g_left`/`g_right`/`L_interp`/`fps`）
    全部没有消费点，两个取值给出逐位相同的结果 —— 完整论证与两种修正函数
    的公式/文献保留在 `ProjectFiles/V2.0/27_FR修正函数族在原生基下不适用-
    flux-type移除.md`。

    四面体基（2026-09-03 起不再可选，见模块文档）：永远是 native 单纯形基
    （路径C），不再接受 `tet_basis_mode` 参数——坍缩坐标（Duffy 变换）
    四面体基函数构造已删除，不是被绕开。

    Returns:
        operators: 包含所有预计算算子的 FROperators 对象
    """
    # `AFCFD_PRISM_BASIS` 的**唯一校验点**（2026-09-24）：坍缩棱柱基删除
    # 之后，除 checkpoint 存/取之外再没有代码读这个环境变量，于是
    # `AFCFD_PRISM_BASIS=collapsed` 会被**静默忽略**、照原生跑 —— 那正是
    # 本项目明确禁止的静默降级（同一原则见 `fr_solver/filter.py::
    # resolve_filter_mode`、`overintegration_order.py::
    # resolve_native_overintegration_max_order`）。这里调一次纯为它的校验
    # 副作用：非法取值当场报错。每条后端路径（单机 CPU / GPU / CPU MPI /
    # 多 GPU / Order Continuation 重建算子）都要经过本函数，所以这一个点
    # 就够，不需要在各入口重复。
    from ..native_prism.mode import resolve_prism_basis_mode
    resolve_prism_basis_mode()

    n = order + 1  # SPs 数量
    # 全局统一 SPs 宽度（张量积立方体点数）——原生基（四面体/棱柱）的
    # 真实自由度少于它，一律零填充到这个宽度，见 fr/native_padding.py
    # 的"零填充块对角"不变量。定义在这里（而不是下方 3e 段内）是因为
    # 棱柱原生过积分算子（第 6 段）也要用它填充粗网格轴。
    n_sps_global = n ** 3

    # 1. 生成点集
    sps, _ = gauss_legendre(n)

    # 2. 计算一维微分矩阵
    D_1d = compute_diff_matrix_1d(sps)

    # 3. 计算三维微分算子（朴素张量积，只对六面体正确；棱柱见下，
    # 四面体走 native 单纯形基，见下方 3e）。
    D_3d = compute_diff_matrix_3d(D_1d)

    # 3b. `ref_cube_sps` 与 `boundary_extrap_prism` 都已删除
    # （2026-09-24）：前者唯一的用途是喂给后者，后者的全部消费点随坍缩
    # 1D 分布机制一并删除。原生棱柱/四面体的**面通量点定位**刻意复用
    # 坍缩立方体面的参数化（`newton_locate_on_face`，等价性已验证到
    # 7.5e-17），但那条路径自己在 `fr/face_flux_points/geometry.py` 里
    # 调 `prism_modal_basis_and_grad`，不经过这里。
    #
    # 同批删除的还有 `fps`/`L_interp`（SP->FP 插值矩阵与 FP 位置）与
    # `g_left`/`g_right`（修正函数导数）：面通量点的真实位置来自
    # `fr/face_flux_points`（在物理面上 Newton 定位），修正函数族在
    # DG lift 形式下不存在，两组都没有任何消费点。

    # 6. 棱柱体积项去混叠算子（order==0 时 P0 走独立有限体积路径，不
    # 需要）。四面体那份见下方 3e。
    #
    # **过积分阶数与细点数一律走 `fr/overintegration_order.py`**
    # （2026-09-19）：此前这里和 `grid/high_order/high_order_mesh_order.py
    # ::build_order_geometry` 各写一遍同一个 `min(rule*order, cap)`，两处
    # 注释都在提醒"必须逐字一致，否则算子形状与 mesh.jacobians_fine 对不
    # 上"。靠注释维持一致在本项目已经出过真实缺陷（滤波档双解析器、CFL
    # 三处硬编码兜底），而原生/坍缩分档又要再叠一层，所以合并成唯一入口。
    overint_order_prism = 0
    n_fine_prism = 0
    n_fine_tet = 0
    overint_order_tet = 0
    overint_interp_c2f_tet = overint_interp_c2f_prism = None
    overint_D_fine_tet = overint_D_fine_prism = None
    overint_restrict_f2c_tet = overint_restrict_f2c_prism = None
    if order >= 1:
        overint_order_prism = resolve_prism_overintegration_order(order)
        n_fine_prism = prism_n_fine(overint_order_prism)
        # 棱柱恒为原生基（坍缩档已于 2026-09-23 删除）：细网格轴取
        # **真实**细点数、不填充到 `(oo+1)^3` —— 填充槽位恒为零、零
        # 贡献，而 `D_fine` 的收缩是 O(n_fine^2)（实测去掉细轴填充后
        # P1 加速 3.04x、P2 加速 4.63x，最大相对差 1.4e-16）。粗网格
        # 轴仍必须填充到 `n_sps_global`，因为 `Q` 数组是那个宽度的
        # 填充布局。
        # 原生档：细网格轴取**真实**细点数、不填充到 `(oo+1)^3`
        # （填充槽位恒为零、零贡献，而 `D_fine` 的收缩是
        # O(n_fine^2)）；粗网格轴仍必须填充到 `n_sps_global`，因为
        # `Q` 数组是那个宽度的填充布局。与四面体（下方 3e）逐条对应
        # 的同一套做法，理由见 `native_prism/overintegration.py`
        # 模块文档差异（三）。
        from ..native_prism.overintegration import (
            build_native_prism_overintegration_operators,
        )
        from ..native_padding import pad_native_matrix_to_global

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

    # 3e. 四面体 native 单纯形基（路径C）——2026-09-03 起唯一实现，
    # 恒无条件构造（不再有 collapsed 分支可选，见模块文档）。
    from ..native_tet.basis import (
        build_native_tet_operators, build_native_tet_boundary_extrap, build_native_tet_lift,
    )
    from ..native_padding import (
        pad_native_matrix_to_global, pad_native_filter_matrix_to_global,
    )
    from ..native_tet.filter import build_native_tet_modal_filter

    ref_native_tet, D_native_tet = build_native_tet_operators(order)
    n_native_sps_tet = ref_native_tet.shape[0]
    boundary_extrap_native_tet = {
        excluded_vertex: build_native_tet_boundary_extrap(order, excluded_vertex)
        for excluded_vertex in range(4)
    }
    lift_native_tet = {
        excluded_vertex: build_native_tet_lift(order, excluded_vertex)
        for excluded_vertex in range(4)
    }
    D_native_tet_padded = pad_native_matrix_to_global(D_native_tet, n_sps_global, pad_axes=(0, 1))
    lift_native_tet_padded = {
        excluded_vertex: pad_native_matrix_to_global(lift_native_tet[excluded_vertex], n_sps_global, pad_axes=(0,))
        for excluded_vertex in range(4)
    }
    filter_native_tet = build_native_tet_modal_filter(order)
    filter_native_tet_padded = pad_native_filter_matrix_to_global(filter_native_tet, n_sps_global)

    # `D_3d_tet`/`filter_tet` 别名到 native 填充版本（见模块文档"删除
    # collapsed 相关内容"一节）——两者形状恒等（"零填充块对角"设计
    # 从一开始就保证与 D_3d/filter 同构，供任何仍无条件读取这两个
    # 旧字段名的调用点直接得到正确的 native 结果，不需要逐一修改）。
    D_3d_tet = D_native_tet_padded
    filter_tet = filter_native_tet_padded
    # 3f. 棱柱原生基（迁移期，默认关闭）——见 fr/native_prism/mode.py。
    #
    # **必须与几何一起切换**：原生 `D` 作用在**原生棱柱节点**上的场，而
    # 坍缩档的 `sps_coords`/`jacobians` 建在张量积立方体节点上。把原生算子
    # 套到坍缩节点采样的场上没有任何意义（不会报错，只会给出错的导数），
    # 所以 `grid/high_order/high_order_mesh_order.py::build_order_geometry`
    # 读同一个 `AFCFD_PRISM_BASIS` 分派棱柱段的点位与雅可比。
    D_native_prism = None
    ref_native_prism = None
    n_native_sps_prism = None
    boundary_extrap_native_prism = None
    lift_native_prism = None
    D_native_prism_padded = None
    lift_native_prism_padded = None
    filter_native_prism_padded = None
    if True:  # 棱柱恒为原生基（坍缩档 2026-09-23 删除）
        from ..native_prism.basis import (
            build_native_prism_modal_filter,
            build_native_prism_operators,
        )
        from ..native_prism.face import build_all_native_prism_face_operators

        ref_native_prism, D_native_prism = build_native_prism_operators(order)
        n_native_sps_prism = ref_native_prism.shape[0]
        boundary_extrap_native_prism, lift_native_prism = (
            build_all_native_prism_face_operators(order))
        D_native_prism_padded = pad_native_matrix_to_global(
            D_native_prism, n_sps_global, pad_axes=(0, 1))
        lift_native_prism_padded = {
            f: pad_native_matrix_to_global(mat, n_sps_global, pad_axes=(0,))
            for f, mat in lift_native_prism.items()
        }
        filter_native_prism_padded = pad_native_filter_matrix_to_global(
            build_native_prism_modal_filter(order), n_sps_global)
        # 与四面体同一个别名做法：无条件读 `D_3d_prism`/`filter_prism` 的
        # 消费点自动拿到原生结果，形状恒等（零填充块对角设计保证）。
        D_3d_prism = D_native_prism_padded
        filter_prism = filter_native_prism_padded

    # native 单纯形基过积分（去混叠）算子（Part8 文档"四·七"节）：fine
    # 网格宽度沿用与棱柱相同的 (over_order+1)^3（`n_sps_per_cell_fine`，
    # 见 high_order_mesh_order.py::build_order_geometry），native 自己
    # 的 fine 点数 `(over_order+1)(over_order+2)(over_order+3)/6` 严格
    # 更小——D_fine/interp_c2f/restrict_f2c 三者都需要按"零填充块对角"
    # （Part8 文档"一"节）填充到这个全局宽度；interp_c2f/restrict_f2c
    # 的两个轴分别对应不同的 native 长度（fine 轴 vs coarse 轴），
    # `pad_native_matrix_to_global` 一次只处理同一个 native 长度的
    # 轴集合，因此分两步各自填充对应的轴，而不是一次性传两个轴（详见
    # 该函数文档）。
    if order >= 1:
        from ..native_tet.overintegration import (
            NATIVE_TET_OVERINTEGRATION_MAX_ORDER,
            build_native_tet_overintegration_operators,
            resolve_tet_overintegration_order,
        )

        # ===== 四面体过积分阶数与棱柱**解耦**（2026-09-17）=====
        #
        # 此前这里直接复用棱柱的 `overint_order`，也就是让 native 四面体
        # 继承坍缩基那条 `OVERINTEGRATION_MAX_ORDER = 3` 条件数上限
        # （该常量已随坍缩棱柱基于 2026-09-23 一并删除）。
        # 实测证明那条依据对 native PKD 基不成立（见上方 3d 注释），而
        # 上限的代价是量过的（tests/unit/test_overintegration_cap_cost.py，
        # 以 over_order=8 为参照的体积项去混叠相对误差中位数）：
        #
        #     P2  oo=3 -> oo=4    2.38e-2 -> 7.01e-6      3400 倍
        #     P3  oo=3 -> oo=6    9.17e-2 -> 4.92e-6     18600 倍
        #                （P3 的 oo=3 == order，过积分完全无操作）
        #
        # 四面体段的细点度量取自 `mesh.jacobians_fine` 的**第 0 列广播**
        # （直边四面体的 Jacobian 逐单元为常数、被原样广播填满全部槽位，
        # 见 `high_order_mesh_order.compute_native_tet_jacobians` 与
        # `core/fr_operators/volume_contract.get_overintegration_context`），
        # 所以 `n_fine_tet` **不受**棱柱布局宽度 `(oo_prism+1)^3` 约束——
        # P3 的 84 个细点可以超过棱柱的 64 列。实际取到的阶数：
        #
        #     P1  棱柱 oo=2  四面体 oo=2  细点 10   = 理想
        #     P2  棱柱 oo=3  四面体 oo=4  细点 35   = 理想
        #     P3  棱柱 oo=3  四面体 oo=6  细点 84   = 理想
        #
        # P1 的阶数与改动前相同，所以"放开上限"这一项不改变已验证的 P1
        # 生产结果（同批的 `enforce_constant_annihilation` 会在舍入量级上
        # 改动它，见 tests/unit/test_diff_matrix_constant_annihilation.py）。
        overint_order_tet = resolve_tet_overintegration_order(order)
        _oo_tet_ideal = resolve_overintegration_order_rule() * order
        if overint_order_tet < _oo_tet_ideal:
            from loguru import logger
            from ..native_tet.overintegration import (
                resolve_tet_overintegration_max_order,
            )

            # 现在只剩"上限"一条可能夹住它，而默认上限 6 >= 任何 rule*order
            # （rule<=3、order<=2 时）——所以这条只在用户显式调低
            # `AFCFD_TET_OVERINT_MAX_ORDER`、或 order>=3 且 rule=3x 时出现。
            logger.warning(
                f"P{order} 四面体过积分阶数被夹到 {overint_order_tet}"
                f"（理想 {_oo_tet_ideal}）：上限 "
                f"AFCFD_TET_OVERINT_MAX_ORDER="
                f"{resolve_tet_overintegration_max_order()} 在约束（默认 "
                f"{NATIVE_TET_OVERINTEGRATION_MAX_ORDER}）。去混叠精度因此"
                f"低于可达水平——实测误差在 `oo = 2*order` 处断崖式下降，"
                f"低于它基本拿不到去混叠收益（见 tests/unit/"
                f"test_overintegration_cap_cost.py）"
            )

        ref_fine_native, interp_c2f_native, D_fine_native, restrict_f2c_native = (
            build_native_tet_overintegration_operators(order, overint_order_tet)
        )
        # ===== 细网格轴**不再**填充到全局张量积宽度（2026-09-17）=====
        #
        # 此前这三个矩阵的**细网格轴**也被 `pad_native_matrix_to_global`
        # 填到 `(overint_order+1)^3`（与棱柱共用同一个 `n_fine`）。但 native
        # 四面体在 over_order 下只有 `(oo+1)(oo+2)(oo+3)/6` 个真实细点：
        #
        #     P1 (oo=2)   真实 10   填充 27
        #     P2 (oo=3)   真实 20   填充 64
        #
        # 填充槽位恒为零（见 `pad_native_matrix_to_global` 文档"零填充
        # 块对角"不变量），所以它们对结果**零贡献**——但整条过积分链
        # （插值 -> 物理通量 -> 逆变通量 -> 散度 -> 限制）都在这些空点上
        # 白算，而其中 `D_fine` 的收缩是 **O(n_fine^2)**：
        #
        #     P2: 64^2 / 20^2 = 10.2 倍的无效 FLOPs
        #
        # 实测（20000 个四面体单元跑完整链路，与填充版对比）：
        #
        #     P1  加速 3.04x   最大相对差 1.385e-16（舍入）
        #     P2  加速 4.63x   最大相对差 0.000e+00（完全相同）
        #
        # 而四面体在本项目两张真实网格里占约 83% 的单元。
        #
        # **粗网格轴仍然必须填充**：`Q` 数组是 `(n_cells, n_sps_global, 5)`
        # 的填充布局（native 四面体的真实自由度只占前 n_native 个槽位），
        # 所以 interp 的**列**、restrict 的**行**必须是 n_sps_global 宽。
        # 只有细网格轴是本模块内部自己的中间维度，可以取真实长度。
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

    return FROperators(
        D_1d=D_1d,
        D_3d=D_3d,
        D_3d_tet=D_3d_tet,
        D_3d_prism=D_3d_prism,
        filter_tet=filter_tet,
        filter_prism=filter_prism,
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
        D_native_tet=D_native_tet,
        ref_native_tet=ref_native_tet,
        n_native_sps_tet=n_native_sps_tet,
        boundary_extrap_native_tet=boundary_extrap_native_tet,
        lift_native_tet=lift_native_tet,
        D_native_tet_padded=D_native_tet_padded,
        lift_native_tet_padded=lift_native_tet_padded,
        filter_native_tet_padded=filter_native_tet_padded,
        D_native_prism=D_native_prism,
        ref_native_prism=ref_native_prism,
        n_native_sps_prism=n_native_sps_prism,
        boundary_extrap_native_prism=boundary_extrap_native_prism,
        lift_native_prism=lift_native_prism,
        D_native_prism_padded=D_native_prism_padded,
        lift_native_prism_padded=lift_native_prism_padded,
        filter_native_prism_padded=filter_native_prism_padded,
    )


if __name__ == "__main__":
    # 测试算子生成
    order = 2
    ops = generate_fr_operators(order)
    
    print(f"FR Operators for P={order}:")
    print(f"  D_1d shape: {ops.D_1d.shape}")
    print(f"  D_3d shape: {ops.D_3d.shape}")
