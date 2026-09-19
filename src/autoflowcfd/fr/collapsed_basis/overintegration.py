"""AutoFlowCFD V2.0 - 坍缩基体积项去混叠（过积分）算子与阶数上限。

从 `fr/collapsed_basis.py`（原 610 行）拆出（2026-09-19）。纯搬家，逻辑
未改。

`OVERINTEGRATION_MAX_ORDER` 是**坍缩坐标** Vandermonde 的数值条件数极限，
自 2026-09-17 起只约束棱柱的坍缩档；两种**原生**基各有自己的上限，统一
住在 `fr/overintegration_order.py`（跨档的唯一入口）。
"""

import os
from typing import Tuple

import numpy as np

from .modal import (
    build_collapsed_diff_matrices,
    prism_modal_basis_and_grad,
    tet_modal_basis_and_grad,
)


def _gauss_legendre_1d(n: int) -> np.ndarray:
    """n 点 1D Gauss-Legendre 求积点（不需要权重，过积分只用点位）。"""
    from ..quadrature_points import gauss_legendre

    pts, _ = gauss_legendre(n)
    return pts


# 过积分（over-integration）细网格阶数的硬上限。理想去混叠阶数是
# 2*order（二次非线性经验法则），但本模块的模态 Vandermonde 矩阵条件数
# 随阶数爆炸式增长（本文件 jacobi_polynomial 文档实测：N=2 时 cond~1e5，
# N=3 时 ~1e9，N=4 时 ~1e14——接近 float64 ~1e16 动态范围的可用边界）。
# 真实数值实验证实了这个上限过去（V2.0 二次评审 Tier 0 #2）之所以卡在
# 3 的必要性：即使把 build_collapsed_diff_matrices/build_collapsed_
# boundary_extrap 的显式求逆换成 lu_solve 大幅改善了条件数敏感度，
# over_order=4（cond~1e14）用**双精度** LU 在生产阶数 P=2 上仍不稳定：
# 均匀自由流场残差 1.7（应为 ~0），线性剪切流残差 0.56（应为 0）——量级
# 上比不做过积分更差，是真正的数值噪声而非改善。上限设为 3（cond~1e9）
# 后同一组测试稳定给出自由流场残差 1.06e-5、剪切流残差 3.49e-6（后者
# 比不做过积分时的 43~62 倍误差改善约 5~6 个数量级）；但 P=3 下
# over_order=min(2*3,3)=3=order，退化为 fine 点集与 coarse 完全重合
# （interp_c2f/restrict_f2c 退化为恒等矩阵，D_fine=D_3d_tet/prism 本身）
# ——等价于不做过积分，P3 完全拿不到任何去混叠收益。
#
# 2026-08-30（`8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-
# Part1~5.md`）：上面"over_order=4 不稳定"的病根一度被定位为"双精度 LU
# 分解在 cond(V)~1e14 时把条件数直接喂进舍入误差"这一个纯数值线性代数
# 问题——`collapsed_basis_high_precision.py` 用 mpmath 50 位十进制精度
# 重新构造同一套算子验证了这个诊断的一半：Couette 剪切流残差确实真实
# 改善（约 2~3 倍，见 Part4/Part5），且 `D_fine@常数场=0` 这类相对误差
# 判据在 over_order=4 下依然保持在 float64 能达到的最好水平
# （~1e-16 相对误差）。
#
# 但把 OVERINTEGRATION_MAX_ORDER 直接放宽到 4 作为**默认值**上线后，
# 被本代码库已有的"黄金标准判据"—— 均匀自由流场残差必须近似为零
# （tests/unit/test_fr_residual_inviscid.py::TestFreeStreamPreservation）
# ——当场测出真实回归：P=2 均匀流场残差从 over_order=3 时的 1.06e-5
# 恶化到 over_order=4 时的 5.6e-3（约 500 倍变差，远超原有 3e-5 容差）。
# 根因：D_fine 的绝对量级本身随 over_order 从 3 到 4 暴涨约 6.3 万倍
# （6.0e7 -> 3.8e12，与 cond(V) 1e9->1e14 的增长量级一致），即使
# mpmath 构造把**相对**误差稳稳压在 float64 能表示的极限（~1e-16），
# 这个相对误差乘上暴涨后的绝对量级，再经过真实（非常数、非零）的
# 几何度量项收缩求和，会被放大成不可忽略的绝对噪声——这在"残差本该
# 恒为零、任何非零都是纯噪声"的均匀流场检验里被完整暴露；而 Couette
# 剪切流本身有真实的、量级更大的混叠误差需要被"去混叠"，这部分
# 噪声被淹没在更大的（且被过积分实际改善的）信号里，才让 Part4 的
# 决定性测试看起来是纯粹的净改善。**这是一个真实的、此前测试范围
# （只测了 Couette，没有连带重新跑现有黄金标准判据）没有覆盖到的
# 权衡取舍，不是可以无条件默认开启的纯粹提升**——已如实记录在
# `8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part5.md`，
# 上限维持 3 不变。曾经为验证这个诊断写过一版 mpmath 高精度构造
# （collapsed_basis_high_precision.py），构造方法本身确认正确，但
# 上限维持 3 之后它在生产路径里已经完全不可达（`over_order>3` 这个
# 分支永远不会被触发，只有它自己的单元测试在调用）——不留"以后可能
# 用得上"的死代码，已删除；Part5 文档完整保留了这套方法的公式、数据
# 与结论，未来若要重新做"用户显式选择更高 over_order"这个可选项，
# 从那份文档和这段注释就能重新实现，不需要现在占着一份不会被执行
# 的代码。
#
# ===== 2026-09-16：这条上限的适用范围（实测）=====
#
# 上面整条论证是对**坍缩坐标张量积**基做的。坍缩四面体基已于 2026-09-03
# 删除，四面体现在走 native 受限 PKD/Dubiner 基——单纯形上的**正交**基，
# 条件数性质与坍缩坐标完全不在一个量级。实测（
# `tests/unit/test_native_tet_overintegration_conditioning.py` 是该实测的
# 可执行形式）：
#
#     over_order  n_pts   cond(V)     max|D|    D@1/max|D|    D@r - 1
#              3     20   5.637e+01  4.045e+00   9.057e-16   2.998e-15
#              4     35   2.070e+02  6.757e+00   1.376e-14   2.243e-14
#              6     84   3.856e+03  1.420e+01   1.202e-13   6.610e-13
#
# native 在 over_order=6 才 cond(V)=3856（坍缩基 N=4 就约 1e14），
# `max|D|` 从 3 到 6 只长 3.5 倍（坍缩基同区间暴涨约 6.3 万倍）。
# **上面那条数值论证对 native 四面体不适用。**
#
# ===== 2026-09-17：已按单元类型分开，本常量现在**只管棱柱** =====
#
# 上面那段"为什么还没分开放开"曾把原因记成一条架构约束：要让四面体用 4、
# 棱柱用 3，就得把共用的 `mesh.jacobians_fine` 按较大者分配（125 槽 vs
# 64 槽），plate_demo P2 的这块内存会从约 1.86 GB 涨到约 3.63 GB。
#
# **那条论证是错的**（自己的数据推翻的）：`high_order_mesh_order.
# compute_native_tet_jacobians` 对直边四面体只算一个逐单元常数、再把它
# **原样广播**填满该单元全部槽位。所以四面体的细点度量不需要更宽的数组，
# 只需要"够宽"——把前 `n_fine_tet` 列取出来，与在真实细点上求值恒等。
# 真实约束因此只是 `native_tet_n_fine(oo_tet) <= (oo_prism+1)^3`，不需要
# 任何额外内存。
#
# 于是四面体的过积分阶数已独立出去（`native_tet_overintegration.
# NATIVE_TET_OVERINTEGRATION_MAX_ORDER = 6`，env
# `AFCFD_TET_OVERINT_MAX_ORDER`），本常量只约束棱柱。实际取到的阶数：
#
#     P1  棱柱 2  四面体 2（= 理想，与改动前相同）
#     P2  棱柱 3  四面体 4（= 理想；去混叠相对误差 2.38e-2 -> 7.01e-6）
#     P3  棱柱 3  四面体 5（理想 6 需 84 个细点 > 布局 64，被夹）
#
# P3 要完整到 6 仍然需要给四面体单独存一份紧凑的（O(n_cells) 而非
# O(n_cells*n_fine)）细点度量——那是布局层面的独立改动，收益已量化
# （oo=5 -> 6 约 4 倍），不是这条常量的问题。
#
# 放开上限本身会抬高自由流保持性的误差底（`D_fine` 对常数的零化残余随
# 阶数增长），已同批用 `diff_matrix_consistency.enforce_constant_
# annihilation` 消掉，真实网格实测见
# `tests/unit/test_tet_overintegration_cap_raised.py`。
OVERINTEGRATION_MAX_ORDER = 3


def resolve_overintegration_order_rule() -> int:
    """过积分阶数规则的倍率：`AFCFD_OVERINT_ORDER_RULE = 2x | 3x`，默认 `2x`。

    `over_order = min(rule * order, OVERINTEGRATION_MAX_ORDER)`。

    **为什么是可切换的而不是直接改成 3x（2026-09-15）**：`3x` 在 order=1
    上带来很大的精度收益（细点 27 -> 64）——

        k/omega 对流  prism 1.360e-1 -> 2.124e-3（64x）  tet -> 机器零
        粘性体积项    prism 4.615e-3 -> 3.617e-6（1276x）tet -> 机器零

    ——而且自由流场保持性没有回归（order=1 相对残差 4.1101e-11 ->
    4.3839e-11，噪声级；order>=2 因为被 MAX_ORDER=3 卡住、逐位不变）。

    ## 真实网格实测已补齐（2026-09-17）：`3x` 在 P1 上买不到任何东西

    上面那组精度数字是在**合成场/隔离判据**上量的。真实网格 + SST 的
    对照（plate_coarse_volume，168,340 单元，P1+SST，同一初场，预热 4 步
    后计时 8 步，壁面距离场按 CLI 同一路径算好）：

        rule=2x   prism_oo=2 tet_oo=2 n_fine_tet=10   11.996 s/step  res 4.425745e+08
        rule=3x   prism_oo=3 tet_oo=3 n_fine_tet=20   18.967 s/step  res 4.425745e+08

    **残差 7 位有效数字完全相同，而每步贵 1.58 倍。** 也就是说 `oo = 2*order`
    对这条真实算例上的全部项——包括 k/omega 输运与含 mu_t 的粘性通量那两个
    三重乘积——都已经足够。同一结论在平板边界层算例上独立复现过：粘性
    体积项 `AFCFD_VISC_OVERINT=on` 时 `2x` 与 `3x` 的能量分量**逐位相同**
    （见 `core/fr_residual/viscous_flux.py::resolve_viscous_overintegration`）。

    （旧的代价估算——微基准 3.47s -> 14.30s、外推每步 ~1.6x——是在**细网格
    轴还被零填充**的代码上做的，那份填充已于 2026-09-17 去掉（P1 实测整链
    3.04x 加速），所以那个估算已不适用；现在这个 1.58x 是真实网格直接测的。）

    因此默认保持 `2x`，依据从"代价未量化所以保守"升级为"**收益实测为零、
    代价实测 1.58 倍**"。`3x` 保留为受控 A/B 的入口（order>=2 上是否仍然
    为零尚未测，那需要 P2 的真实网格运行）。
    """
    v = os.environ.get("AFCFD_OVERINT_ORDER_RULE", "2x").lower()
    if v not in ("2x", "3x"):
        raise ValueError(
            f"AFCFD_OVERINT_ORDER_RULE={v!r} 不是合法取值（2x | 3x）。"
            f"'2x' 是既有行为（为平均流的二次非线性设计），'3x' 针对标量"
            f"输运/粘性项里的三重乘积，见 resolve_overintegration_order_rule。")
    return 2 if v == "2x" else 3


def build_overintegration_operators(
    cell_type: str, order: int, over_order: int, ref_cube_sps_coarse: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造体积项去混叠（over-integration）三件套：coarse->fine 插值、
    fine 网格上的微分矩阵、fine->coarse 限制。

    背景（V2.0 二次专家评审 Tier 0 #2）：体积项残差 `D_3d_tet/prism @
    (adj(J)*F_phys(Q))` 直接在 coarse SPs（degree=order 的节点表示）上
    做微分——但 `adj(J)*F_phys(Q)` 是 Q 的非线性函数（欧拉通量含
    u_i*u_j、p*u_j 等二次项）与度量项的乘积，其真实多项式次数远高于
    order，直接对它的 degree-order 节点插值多项式求导，等价于先把它
    混叠（alias）到 degree-order 空间再求导——真实数值实验证实：对
    解析残差恒为 0 的线性剪切场 u=30+a*y，P2（生产默认阶数）算出的
    残差达到真值的约 43~62 倍，P1 更是 400+ 倍，只有从未在生产路径上
    使用过的 P3 才勉强正确。这是标准的体积项积分不足（aliasing），
    工程解法是"过积分"：把 Q 插值到更细的求积点上，在细网格上精确
    评估非线性通量和度量项后再求导，最后把结果限制回 coarse SPs——
    不是研究级问题（`5_重大问题修复-Part1.md` 对相关不守恒问题的结论
    "需要 entropy-stable/split-form 研究级重新设计" 针对的是另一个
    机制，见该文档；本机制的标准解法见 Kopriva《Implementing Spectral
    Methods for PDEs》Ch.5 "aliasing and the strong form" 与 Kirby &
    Karniadakis 关于二次非线性去混叠所需求积阶数的经典分析）。

    三个算子都基于同一套模态 Vandermonde 机制（与 build_cross_interp/
    build_collapsed_boundary_extrap 同源）：
    1. Interp_c2f = V_fine_pts_by_coarse_basis @ V_coarse_sps^{-1}
       （用 COARSE 阶数的模态基在 FINE 点上取值——Q 本身次数 <= order，
       这一步是精确插值，不引入混叠）
    2. D_fine：在 FINE 点集上、用 FINE 阶数的模态基构造的微分矩阵
       （即 build_collapsed_diff_matrices 在 over_order 下的结果）——
       微分的是 FINE 点上取值所代表的 degree-over_order 插值多项式，
       更接近真实非线性通量的次数，混叠误差大幅降低
    3. Restrict_f2c = V_coarse_pts_by_fine_basis @ V_fine_sps^{-1}
       （用 FINE 阶数的模态基在 COARSE 点上取值——把微分后的场从细网格
       插值回粗网格 SPs，供残差公式除以 coarse 的 det(J) 使用）

    Args:
        cell_type: "tet" 或 "prism"
        order: 当前求解阶数 P（coarse）
        over_order: 过积分阶数（建议 2*order，二次非线性去混叠的标准
            经验法则；order=0 时不适用，P0 走独立的有限体积路径）
        ref_cube_sps_coarse: coarse SPs 参考坐标 (n_coarse,3)

    Returns:
        (ref_cube_sps_fine, interp_c2f, D_fine, restrict_f2c)：
        ref_cube_sps_fine 形状 (n_fine,3)；interp_c2f 形状
        (n_fine,n_coarse)；D_fine 形状 (n_fine,n_fine,3)；restrict_f2c
        形状 (n_coarse,n_fine)。
    """
    fine_n1d = over_order + 1
    fine_1d = _gauss_legendre_1d(fine_n1d)
    ga, gb, gc = np.meshgrid(fine_1d, fine_1d, fine_1d, indexing="ij")
    ref_cube_sps_fine = np.column_stack([ga.ravel(), gb.ravel(), gc.ravel()])

    basis_fn = tet_modal_basis_and_grad if cell_type == "tet" else prism_modal_basis_and_grad

    V_coarse_sps, _, _, _ = basis_fn(
        ref_cube_sps_coarse[:, 0], ref_cube_sps_coarse[:, 1], ref_cube_sps_coarse[:, 2], order
    )
    V_fine_at_fine, _, _, _ = basis_fn(
        ref_cube_sps_fine[:, 0], ref_cube_sps_fine[:, 1], ref_cube_sps_fine[:, 2], over_order
    )

    # interp_c2f：COARSE 阶数模态基在 FINE 点上取值
    V_coarse_at_fine, _, _, _ = basis_fn(
        ref_cube_sps_fine[:, 0], ref_cube_sps_fine[:, 1], ref_cube_sps_fine[:, 2], order
    )
    from scipy.linalg import lu_factor, lu_solve

    lu_coarse = lu_factor(V_coarse_sps.T)
    interp_c2f = lu_solve(lu_coarse, V_coarse_at_fine.T).T

    # D_fine：FINE 阶数模态基自身的微分矩阵
    D_fine = build_collapsed_diff_matrices(cell_type, over_order, ref_cube_sps_fine)

    # restrict_f2c：FINE 阶数模态基在 COARSE 点上取值
    V_fine_at_coarse, _, _, _ = basis_fn(
        ref_cube_sps_coarse[:, 0], ref_cube_sps_coarse[:, 1], ref_cube_sps_coarse[:, 2], over_order
    )
    lu_fine = lu_factor(V_fine_at_fine.T)
    restrict_f2c = lu_solve(lu_fine, V_fine_at_coarse.T).T

    return ref_cube_sps_fine, interp_c2f, D_fine, restrict_f2c
