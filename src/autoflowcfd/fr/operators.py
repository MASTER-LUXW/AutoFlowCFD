"""
AutoFlowCFD V2.0 - FR 算子生成器主接口

本模块是 Flux Reconstruction 方法算子生成的统一入口，
整合了点集生成、矩阵算子计算等功能。

核心功能：
1. 统一的FR算子生成接口
2. 算子数据容器定义

坍缩坐标四面体基已删除（2026-09-03，用户明确要求"删除 collapsed 相关
内容"，前提是 native 已在同一天的排查中决定性验证覆盖 CPU/GPU 全部
残差链路——含此前被误判为缺失的 over-integration 组合，见
`native_tet_basis_deletion_2026_09_03` 记忆）：四面体不再有
`tet_basis_mode` 选择，永远走 native 单纯形基（Part6/7/8 文档"路径C"，
`fr/native_simplex_basis.py`）——不是"默认改成 native"，是坍缩坐标
（Duffy 变换）四面体基函数构造本身（`collapsed_basis.py::
build_collapsed_diff_matrices("tet",...)`/`build_collapsed_boundary_
extrap("tet",...)`/`modal_filter.py::build_tet_modal_filter`）不再被
调用，这部分计算真正被删除，不是被绕开。棱柱不受影响——棱柱没有
"collapsed vs native"的选择，坍缩坐标是棱柱唯一、正确的构造方式（三棱柱
=三角形×直线，不存在四面体那种坍缩坐标退化轴条件数问题），
`collapsed_basis.py`/`modal_filter.py` 对 "prism" 的调用原样保留。

`D_3d_tet`/`filter_tet` 两个字段现在直接**别名**到对应的 native 填充
版本（`D_native_tet_padded`/`filter_native_tet_padded`，两者形状恒等，
是"零填充块对角"设计从一开始就保证的，见 native_tet_padding.py 文档）；
`boundary_extrap_tet` 保留为占位零矩阵字典（形状与之前一致），只是为了
不用同步修改 `core/fr_operators/face_kernels.py` 里"无条件按 (celltype,
axis,side) 读取 boundary_extrap_tet/prism 拼成统一查找表"这一段代码——
四面体面在 `with_native_tet_faces` 静态翻译（现在总是执行，见
`grid/connectivity/face_connectivity.py`/`grid/high_order/high_order_
mesh.py`）之后恒为 `tet_native_v*` 编码（>=6），这些占位行在生产路径上
永远不会被真正读取（`_native_self_extrap` 等函数按 `is_native` 掩码
丢弃 collapsed 分支的结果，只是 NumPy/CuPy 花式索引要求索引本身合法，
见 gpu_inviscid.py 模块文档"提前无条件求值导致越界"一节的同一原理）。
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from .quadrature_points import gauss_legendre, gauss_lobatto
from .matrix_operators import (
    compute_vandermonde,
    compute_diff_matrix_1d,
    compute_diff_matrix_3d,
    compute_interpolation_matrix,
    compute_correction_weights
)
from .collapsed_basis import (
    build_collapsed_diff_matrices,
    build_collapsed_boundary_extrap,
    build_overintegration_operators,
    OVERINTEGRATION_MAX_ORDER,
    resolve_overintegration_order_rule,
)
from .modal_filter import build_prism_modal_filter


@dataclass
class FROperators:
    """
    FR 算子容器，存储预计算的所有矩阵。

    Attributes:
        D_1d: 一维微分矩阵，形状 (n_pts, n_pts)
        D_3d: 三维微分算子，形状 (n_pts^3, n_pts^3, 3)——朴素张量积 Lagrange
            微分矩阵，只对六面体（无坍缩坐标退化面）正确；四面体/棱柱的
            体积散度/梯度/几何 Jacobian 计算必须改用 D_3d_tet/D_3d_prism。
        D_3d_tet: 四面体体积微分矩阵，形状同 D_3d——**别名到
            `D_native_tet_padded`**（2026-09-03 起，collapsed 四面体基
            已删除，见模块文档），不再是独立构造的坍缩坐标矩阵。
        D_3d_prism: 棱柱专用体积微分矩阵，形状同 D_3d，用坍缩坐标模态基
            （fr/collapsed_basis.py）通过 Vandermonde 矩阵构造，SPs
            位置与 D_3d 完全相同（不改变点集，只改变"如何对这些点上的
            节点值求导"）——原因见该模块文档：朴素张量积基与坍缩坐标
            退化边（棱柱 b=+1）附近真实存在的高阶（度量项、通量的坍缩
            坐标依赖是有理式而非低阶多项式）结构不匹配，插值多项式在
            退化边附近的混叠误差会被同样在该处偏小的真实几何 Jacobian
            放大到灾难量级；坍缩坐标模态基内建与退化因子匹配的结构，
            能显著降低这一混叠误差（真实网格验证：某棱柱单元体积项
            残差从 3.15e-11 降到 6.53e-12，约 5 倍）。棱柱没有 native
            方案可换，这里的坍缩坐标构造不受本次删除影响。
        L_interp: 插值矩阵 (SPs -> FPs)，形状 (n_fps, n_sps)
        g_left, g_right: 左右 Radau/VCJH 校正函数**导数**在各 SP 处的取值，
            形状均为 (n_sps,)（不是校正函数本身的值，见
            matrix_operators.compute_correction_weights 文档说明）
        boundary_extrap_tet: {(axis:int,side:float): (n_fp,n_sps) 全 NaN
            占位矩阵}——**不再是真实的坍缩坐标外插算子**（2026-09-03
            起已删除，见模块文档），只保留字典形状供
            `core/fr_operators/face_kernels.py` 无条件按 (celltype,
            axis,side) 拼表这一段代码不用跟着改；四面体面翻译成
            native 编码后这些占位行在生产路径上不会被真正读取。
        boundary_extrap_prism: {(axis:int,side:float): (n_fp,n_sps)
            矩阵}，棱柱专用体积->边界外插矩阵，用与 D_3d_prism 同一套
            坍缩坐标模态基构造（见 collapsed_basis.build_collapsed_
            boundary_extrap 文档），取代 fr/face_flux_points.py::
            extrapolate_to_face 的朴素 1D 张量积外插——真实网格验证
            发现，朴素外插算出的等效界面法向方向在坍缩坐标退化边附近
            与真实几何法向偏差可达近 30°（仍在现有校验阈值内、不报错，
            但足以在残差公式除以该处真实偏小的 Jacobian 后放大到灾难
            量级），必须换成与体积微分矩阵一致的坍缩坐标模态基外插。
        filter_tet: (n_sps,n_sps) 指数模态滤波矩阵——**别名到
            `filter_native_tet_padded`**（2026-09-03 起，不再是独立
            构造的坍缩坐标滤波器）。
        filter_prism: (n_sps,n_sps) 指数模态滤波矩阵（见
            fr/modal_filter.py 文档）——坍缩坐标节点配置法对高阶模态的
            混叠噪声天然敏感，重复微分（体积散度、粘性梯度+散度两次）
            会把这个噪声逐步放大，真实网格上复现过在几步显式时间推进内
            从机器精度噪声放大到 NaN；每个 RK 阶段结束后对解场施加一次
            滤波是标准谱/DG 方法对策，不影响已解析到的低阶物理精度。
    """
    D_1d: np.ndarray
    D_3d: np.ndarray
    D_3d_tet: np.ndarray = None
    D_3d_prism: np.ndarray = None
    L_interp: np.ndarray = None
    g_left: np.ndarray = None
    g_right: np.ndarray = None
    boundary_extrap_tet: Dict[Tuple[int, float], np.ndarray] = None
    boundary_extrap_prism: Dict[Tuple[int, float], np.ndarray] = None
    filter_tet: np.ndarray = None
    filter_prism: np.ndarray = None
    # 体积项去混叠（over-integration，见 collapsed_basis.build_overintegration_operators
    # 文档）：order==0 时全部为 None（P0 走独立的有限体积残差路径，不需要）。
    overint_order: int = None
    overint_ref_fine: np.ndarray = None
    #: native 四面体过积分的**真实**细点数
    #: `(oo+1)(oo+2)(oo+3)/6`。与棱柱共用的 `mesh.
    #: n_sps_per_cell_fine`（`(oo+1)^3`）**不同**——细网格轴
    #: 不再填充，见 `generate_fr_operators` 里那段说明。
    overint_n_fine_tet: int = 0
    overint_interp_c2f_tet: np.ndarray = None
    overint_interp_c2f_prism: np.ndarray = None
    overint_D_fine_tet: np.ndarray = None
    overint_D_fine_prism: np.ndarray = None
    overint_restrict_f2c_tet: np.ndarray = None
    overint_restrict_f2c_prism: np.ndarray = None
    # 四面体路径C（独立于坍缩坐标，见 fr/native_simplex_basis.py 与
    # `8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part6.md`
    # 阶段0/1）：2026-09-03 起恒为非 None（native 现在是四面体唯一
    # 实现，不再有 collapsed 分支可选，`tet_basis_mode` 字段保留仅为
    # 兼容仍在读取它当"是否 native"判据的旧调用点，取值恒为 "native"）。
    # D_native_tet 形状 (n_native_sps_tet, n_native_sps_tet, 3)，参考
    # 坐标是四面体自己的 (r,s,t) 单纯形，节点数一般少于 (order+1)^3。
    tet_basis_mode: str = "native"
    D_native_tet: np.ndarray = None
    ref_native_tet: np.ndarray = None
    n_native_sps_tet: int = None
    # native 四面体体积->自身面外插矩阵（Part7 阶段2设计文档"二·五"节+
    # `native_simplex_basis.py::build_native_tet_boundary_extrap`），
    # 键是被排除的局部顶点 0~3（与占位用的 boundary_extrap_tet 键
    # 是 (axis,side) 元组不同）。
    boundary_extrap_native_tet: Dict[int, np.ndarray] = None
    # native 四面体 DG 提升算子（`native_simplex_basis.py::
    # build_native_tet_lift` 文档），把面通量跳跃提升成体积节点修正
    # 贡献——是坍缩坐标方案"1D Radau/VCJH 修正函数 + _distribute_point"
    # 对非张量积单纯形基的唯一正确推广（native 基没有"坍缩计算方向"，
    # 1D 修正函数沿某一轴分布这个概念不适用）。键同样是被排除的局部
    # 顶点 0~3。
    lift_native_tet: Dict[int, np.ndarray] = None
    # `D_native_tet`/`lift_native_tet` 零填充到全局统一 SPs 宽度 `n_sps`
    # 之后的版本（`fr/native_tet_padding.py::pad_native_tet_matrix_to_
    # global`，见 Part8 文档"一、核心不变量：零填充块对角"）——生产
    # 残差 kernel（`inviscid.py`/`inviscid_kernel.py` 等）要消费的是
    # 这两个已经填充好的版本，不是上面两个原始（n_native 宽）版本；
    # 保留原始版本是因为部分测试/诊断代码可能只关心真实自由度本身，
    # 不需要每次都从填充版本反推。`D_3d_tet`/`filter_tet` 字段（上方
    # dataclass 开头）现在就是这两者的别名，见模块文档。
    D_native_tet_padded: np.ndarray = None
    lift_native_tet_padded: Dict[int, np.ndarray] = None
    # native 四面体指数模态滤波器（`native_tet_filter.py::build_native_
    # tet_modal_filter`，抑制混叠失稳，见该模块与 fr/modal_filter.py
    # 文档），填充到全局 n_sps 宽度（`native_tet_padding.py::pad_native_
    # tet_filter_matrix_to_global`——填充块是单位矩阵，不是零，与
    # D_native_tet_padded/lift_native_tet_padded 的"零填充"约定不同，
    # 见该函数文档）。
    filter_native_tet_padded: np.ndarray = None

    def get_operators(self) -> Dict[str, np.ndarray]:
        """返回算子字典，兼容旧接口。"""
        return {
            'diff_matrix': self.D_1d,
            'interp_matrix': self.L_interp,
            'g_left': self.g_left,
            'g_right': self.g_right
        }


def generate_fr_operators(order: int, flux_point_type: str = 'radau') -> FROperators:
    """
    生成完整的 FR 算子集合。

    Args:
        order: 多项式阶数 P
        flux_point_type: 修正函数族 + FP 位置选择：
            - 'radau'（默认，此前默认值 'lobatto' 与其行为完全等价，见下）：
              校正函数用 VCJH η_p=0 方案（Huynh 记法 "g_DG"，此前代码误
              标为 "g2"，见 matrix_operators.py 2026-08-28 的命名修正
              说明），FPs 用 Gauss-Lobatto 求积点——这是此前唯一被生产
              路径实际使用过的组合（所有既有调用点都不显式传参，见
              compute_correction_weights 调用处历史行为）。'lobatto' 仍
              作为同义值保留，不破坏任何硬编码传了这个字符串的旧代码。
            - 'gauss'（#14 新增）：校正函数用 VCJH η_p=p/(p+1) 方案，与
              Spectral Difference (SD) 方法等价（matrix_operators.py::
              _compute_gauss_correction_derivative 文档的完整推导/引用），
              配套地把 FPs 改为 SPs 本身 + 两个边界点（标准 SD 做法：
              通量在 SPs 处直接重构，不是在额外的 Lobatto 求积点插值）——
              这个 FP 位置选择与校正函数选择是同一个物理方案的两个方面，
              绑定在同一个 flux_point_type 参数下，不单独暴露。

    四面体基（2026-09-03 起不再可选，见模块文档）：永远是 native 单纯形基
    （路径C），不再接受 `tet_basis_mode` 参数——坍缩坐标（Duffy 变换）
    四面体基函数构造已删除，不是被绕开。

    Returns:
        operators: 包含所有预计算算子的 FROperators 对象
    """
    n = order + 1  # SPs 数量

    # 1. 生成点集
    sps, _ = gauss_legendre(n)

    # 2. 计算一维微分矩阵
    D_1d = compute_diff_matrix_1d(sps)

    # 3. 计算三维微分算子（朴素张量积，只对六面体正确；棱柱见下，
    # 四面体走 native 单纯形基，见下方 3e）。
    D_3d = compute_diff_matrix_3d(D_1d)

    # 3b. 棱柱专用坍缩坐标体积微分矩阵，SPs 与上面完全相同（张量积
    # Gauss-Legendre 点），只是构造 D 用的基不同——见 collapsed_basis.py
    # 与 FROperators.D_3d_prism 文档。棱柱没有 native 方案，坍缩坐标是
    # 唯一、正确的构造方式，不受本次删除 collapsed 四面体基的影响。
    aa, bb, cc = np.meshgrid(sps, sps, sps, indexing="ij")
    ref_cube_sps = np.column_stack([aa.ravel(), bb.ravel(), cc.ravel()])
    D_3d_prism = build_collapsed_diff_matrices("prism", order, ref_cube_sps)

    # 3c. 棱柱专用体积->边界外插矩阵（同一套坍缩坐标模态基），见
    # FROperators.boundary_extrap_prism 文档。
    boundary_extrap_prism = {}
    for axis in range(3):
        for side in (-1.0, 1.0):
            boundary_extrap_prism[(axis, side)] = build_collapsed_boundary_extrap(
                "prism", order, ref_cube_sps, axis, side
            )

    # 3d. 棱柱指数模态滤波矩阵（见 fr/modal_filter.py 文档 与
    # FROperators.filter_prism 文档）。
    filter_prism = build_prism_modal_filter(order, ref_cube_sps)

    # 4. 计算插值矩阵。判据键在 'gauss' 上（而不是此前的 'lobatto'）：
    # 任何非 'gauss' 的取值（'radau'/'lobatto'/历史调用点省略此参数）都
    # 路由到同一个 Gauss-Lobatto FP 分支，保证除新增的 'gauss' 外行为
    # 完全不变——见本函数 flux_point_type 参数文档。
    if flux_point_type == 'gauss':
        fps = np.concatenate([[-1.0], sps, [1.0]])
    else:
        fps, _ = gauss_lobatto(n + 1)

    L_interp = compute_interpolation_matrix(sps, fps)

    # 5. 计算校正权重
    g_left, g_right = compute_correction_weights(n, flux_point_type)

    # 6. 棱柱体积项去混叠算子（order==0 时 P0 走独立有限体积路径，不
    # 需要，见 grid/high_order_mesh.py::_build_order_geometry 里
    # jacobians_fine 同样按 order>=1 跳过的说明——两处的 over_order=
    # 2*order 约定必须一致）。四面体过积分算子见下方 3e（native 单纯形
    # 基专属构造，`build_overintegration_operators("tet",...)` 坍缩
    # 坐标版本已删除，不再先算一遍马上被覆盖）。
    overint_order = None
    overint_ref_fine = None
    n_fine_tet = 0
    overint_interp_c2f_tet = overint_interp_c2f_prism = None
    overint_D_fine_tet = overint_D_fine_prism = None
    overint_restrict_f2c_tet = overint_restrict_f2c_prism = None
    if order >= 1:
        # 必须与 grid/high_order_mesh.py::_build_order_geometry 里
        # jacobians_fine 的 over_order 计算逐字一致（同一个上限常量），
        # 否则 fr_residual_inviscid.py 里插值/微分/限制三个算子的形状
        # 会和 mesh.jacobians_fine 对不上。
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
        # **`OVERINTEGRATION_MAX_ORDER = 3` 的上限不动**，但要注意它的
        # **适用范围**（2026-09-16 实测更正）：那条"放宽到 4 会让 P2 均匀
        # 自由流场残差从 1.06e-5 恶化到 5.6e-3、根因是 D_fine 绝对量级暴涨
        # 约 6.3 万倍"的论证只对**坍缩坐标**基成立（也就是这里的棱柱）。
        # native 四面体实测在 over_order=6 才 cond(V)=3856、`max|D|` 从 3
        # 到 6 只长 3.5 倍，那条数值论证对它不适用；它目前仍受这个上限
        # 约束的真实原因是 `jacobians_fine` 棱柱/四面体共用一个
        # `n_sps_per_cell_fine` 维度这条架构约束。完整说明见
        # collapsed_basis.py 该常量上方的注释与
        # tests/unit/test_native_tet_overintegration_conditioning.py。
        # 所以 `3x` 与 `2x` 的差别**只在 order=1**：
        #   order=1: 2x -> over_order 2（细点 27）； 3x -> 3（细点 64）
        #   order>=2: 两者都被 cap 到 3，完全相同
        #
        # `3x` 在 order=1 的实测精度收益 / 已知代价，以及"为什么默认不改"，
        # 全部记在 `resolve_overintegration_order_rule` 的文档里。
        overint_order = min(
            resolve_overintegration_order_rule() * order,
            OVERINTEGRATION_MAX_ORDER)
        _, overint_interp_c2f_prism, overint_D_fine_prism, overint_restrict_f2c_prism = (
            build_overintegration_operators("prism", order, overint_order, ref_cube_sps)
        )

    # 3e. 四面体 native 单纯形基（路径C）——2026-09-03 起唯一实现，
    # 恒无条件构造（不再有 collapsed 分支可选，见模块文档）。
    from .native_simplex_basis import (
        build_native_tet_operators, build_native_tet_boundary_extrap, build_native_tet_lift,
    )
    from .native_tet_padding import (
        pad_native_tet_matrix_to_global, pad_native_tet_filter_matrix_to_global,
    )
    from .native_tet_filter import build_native_tet_modal_filter

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
    n_sps_global = n ** 3
    D_native_tet_padded = pad_native_tet_matrix_to_global(D_native_tet, n_sps_global, pad_axes=(0, 1))
    lift_native_tet_padded = {
        excluded_vertex: pad_native_tet_matrix_to_global(lift_native_tet[excluded_vertex], n_sps_global, pad_axes=(0,))
        for excluded_vertex in range(4)
    }
    filter_native_tet = build_native_tet_modal_filter(order)
    filter_native_tet_padded = pad_native_tet_filter_matrix_to_global(filter_native_tet, n_sps_global)

    # `D_3d_tet`/`filter_tet` 别名到 native 填充版本（见模块文档"删除
    # collapsed 相关内容"一节）——两者形状恒等（"零填充块对角"设计
    # 从一开始就保证与 D_3d/filter 同构，供任何仍无条件读取这两个
    # 旧字段名的调用点直接得到正确的 native 结果，不需要逐一修改）。
    D_3d_tet = D_native_tet_padded
    filter_tet = filter_native_tet_padded
    # `boundary_extrap_tet` 保留为占位字典（形状与坍缩坐标版本一致：
    # (n_fp,n_sps)=((order+1)**2,(order+1)**3)）——只是为了不用同步修改
    # `core/fr_operators/face_kernels.py` 里"无条件按 (celltype,axis,
    # side) 拼表"那段代码；四面体面经 `with_native_tet_faces` 翻译后
    # 恒为 native 编码，这些占位行在生产路径上不会被真正读取，见模块
    # 文档"提前无条件求值导致越界"一节的同一原理。
    #
    # **占位值从 0 改为 NaN（2026-09-14）**：这是"不变量成立才是死代码"
    # 的典型情形——只要哪天"四面体面恒为 native 编码"这条不变量被破坏
    # （新的面翻译路径、某个绕过 `with_native_tet_faces` 的构造方式），
    # 全零算子会把外插态**静默**算成 0（常数外插本该得常数），残差随之
    # 完全错误，而且不报任何错、也不会触发正性限制器。填 NaN 则会立刻
    # 沿残差传播、被求解器既有的 `np.all(np.isfinite(...))` 检查抓住，
    # 把静默错误变成显式失败。
    # 前提已核实：改成 NaN 后全量测试仍然通过，说明生产路径确实从不
    # 读取这些行（否则 NaN 会立刻让测试失败——这本身就是对该不变量
    # 最直接的一次验证）。
    _n_fp_placeholder = boundary_extrap_prism[(0, -1.0)].shape[0]
    boundary_extrap_tet = {
        (axis, side): np.full((_n_fp_placeholder, n_sps_global), np.nan,
                              dtype=np.float64)
        for axis in range(3) for side in (-1.0, 1.0)
    }

    # native 单纯形基过积分（去混叠）算子（Part8 文档"四·七"节）：fine
    # 网格宽度沿用与棱柱相同的 (over_order+1)^3（`n_sps_per_cell_fine`，
    # 见 high_order_mesh_order.py::build_order_geometry），native 自己
    # 的 fine 点数 `(over_order+1)(over_order+2)(over_order+3)/6` 严格
    # 更小——D_fine/interp_c2f/restrict_f2c 三者都需要按"零填充块对角"
    # （Part8 文档"一"节）填充到这个全局宽度；interp_c2f/restrict_f2c
    # 的两个轴分别对应不同的 native 长度（fine 轴 vs coarse 轴），
    # `pad_native_tet_matrix_to_global` 一次只处理同一个 native 长度的
    # 轴集合，因此分两步各自填充对应的轴，而不是一次性传两个轴（详见
    # 该函数文档）。
    if order >= 1:
        from .native_tet_overintegration import build_native_tet_overintegration_operators

        ref_fine_native, interp_c2f_native, D_fine_native, restrict_f2c_native = (
            build_native_tet_overintegration_operators(order, overint_order)
        )
        # ===== 细网格轴**不再**填充到全局张量积宽度（2026-09-17）=====
        #
        # 此前这三个矩阵的**细网格轴**也被 `pad_native_tet_matrix_to_global`
        # 填到 `(overint_order+1)^3`（与棱柱共用同一个 `n_fine`）。但 native
        # 四面体在 over_order 下只有 `(oo+1)(oo+2)(oo+3)/6` 个真实细点：
        #
        #     P1 (oo=2)   真实 10   填充 27
        #     P2 (oo=3)   真实 20   填充 64
        #
        # 填充槽位恒为零（见 `pad_native_tet_matrix_to_global` 文档"零填充
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

        overint_ref_fine = ref_fine_native
        # D_fine：两个轴都是细网格轴 -> 完全不填充
        overint_D_fine_tet = D_fine_native
        # interp c2f：列（粗轴）填充到 n_sps_global，行（细轴）保持真实长度
        overint_interp_c2f_tet = pad_native_tet_matrix_to_global(
            interp_c2f_native, n_sps_global, pad_axes=(1,)
        )
        # restrict f2c：行（粗轴）填充到 n_sps_global，列（细轴）保持真实长度
        overint_restrict_f2c_tet = pad_native_tet_matrix_to_global(
            restrict_f2c_native, n_sps_global, pad_axes=(0,)
        )

    return FROperators(
        D_1d=D_1d,
        D_3d=D_3d,
        D_3d_tet=D_3d_tet,
        D_3d_prism=D_3d_prism,
        L_interp=L_interp,
        g_left=g_left,
        g_right=g_right,
        boundary_extrap_tet=boundary_extrap_tet,
        boundary_extrap_prism=boundary_extrap_prism,
        filter_tet=filter_tet,
        filter_prism=filter_prism,
        overint_order=overint_order,
        overint_ref_fine=overint_ref_fine,
        overint_n_fine_tet=n_fine_tet,
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
    )


if __name__ == "__main__":
    # 测试算子生成
    order = 2
    ops = generate_fr_operators(order)
    
    print(f"FR Operators for P={order}:")
    print(f"  D_1d shape: {ops.D_1d.shape}")
    print(f"  D_3d shape: {ops.D_3d.shape}")
    print(f"  L_interp shape: {ops.L_interp.shape}")
    print(f"  g_left shape: {ops.g_left.shape}")
    print(f"  g_right shape: {ops.g_right.shape}")
