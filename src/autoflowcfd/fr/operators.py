"""
AutoFlowCFD V2.0 - FR 算子生成器主接口

本模块是 Flux Reconstruction 方法算子生成的统一入口，
整合了点集生成、矩阵算子计算等功能。

核心功能：
1. 统一的FR算子生成接口
2. 算子数据容器定义
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
)
from .modal_filter import build_prism_modal_filter, build_tet_modal_filter


@dataclass
class FROperators:
    """
    FR 算子容器，存储预计算的所有矩阵。

    Attributes:
        D_1d: 一维微分矩阵，形状 (n_pts, n_pts)
        D_3d: 三维微分算子，形状 (n_pts^3, n_pts^3, 3)——朴素张量积 Lagrange
            微分矩阵，只对六面体（无坍缩坐标退化面）正确；四面体/棱柱的
            体积散度/梯度/几何 Jacobian 计算必须改用 D_3d_tet/D_3d_prism。
        D_3d_tet, D_3d_prism: 四面体/棱柱专用体积微分矩阵，形状同 D_3d，
            用坍缩坐标模态基（fr/collapsed_basis.py）通过 Vandermonde
            矩阵构造，SPs 位置与 D_3d 完全相同（不改变点集，只改变"如何
            对这些点上的节点值求导"）——原因见该模块文档：朴素张量积基
            与坍缩坐标退化边（棱柱 b=+1，四面体 b=+1/c=+1）附近真实存在
            的高阶（度量项、通量的坍缩坐标依赖是有理式而非低阶多项式）
            结构不匹配，插值多项式在退化边附近的混叠误差会被同样在该处
            偏小的真实几何 Jacobian 放大到灾难量级；坍缩坐标模态基内建
            与退化因子匹配的结构，能显著降低这一混叠误差（真实网格验证：
            某棱柱单元体积项残差从 3.15e-11 降到 6.53e-12，约 5 倍）。
        L_interp: 插值矩阵 (SPs -> FPs)，形状 (n_fps, n_sps)
        g_left, g_right: 左右 Radau/VCJH 校正函数**导数**在各 SP 处的取值，
            形状均为 (n_sps,)（不是校正函数本身的值，见
            matrix_operators.compute_correction_weights 文档说明）
        boundary_extrap_tet, boundary_extrap_prism: {(axis:int,
            side:float): (n_fp,n_sps) 矩阵}，四面体/棱柱专用体积->边界
            外插矩阵，用与 D_3d_tet/D_3d_prism 同一套坍缩坐标模态基
            构造（见 collapsed_basis.build_collapsed_boundary_extrap
            文档），取代 fr/face_flux_points.py::extrapolate_to_face 的
            朴素 1D 张量积外插——真实网格验证发现，朴素外插算出的等效
            界面法向方向在坍缩坐标退化边附近与真实几何法向偏差可达
            近 30°（仍在现有校验阈值内、不报错，但足以在残差公式除以
            该处真实偏小的 Jacobian 后放大到灾难量级），必须换成与体积
            微分矩阵一致的坍缩坐标模态基外插。
        filter_tet, filter_prism: (n_sps,n_sps) 指数模态滤波矩阵（见
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
    overint_interp_c2f_tet: np.ndarray = None
    overint_interp_c2f_prism: np.ndarray = None
    overint_D_fine_tet: np.ndarray = None
    overint_D_fine_prism: np.ndarray = None
    overint_restrict_f2c_tet: np.ndarray = None
    overint_restrict_f2c_prism: np.ndarray = None
    # 四面体路径C（独立于坍缩坐标，见 fr/native_simplex_basis.py 与
    # `8_算法重构-微分算子对坍缩坐标退化参考轴的病态条件数-Part6.md`
    # 阶段0/1）：只在 tet_basis_mode="native" 时非 None。D_native_tet
    # 形状 (n_native_sps_tet, n_native_sps_tet, 3)，参考坐标是四面体
    # 自己的 (r,s,t) 单纯形，不是 D_3d_tet 的坍缩坐标立方体——两者
    # 节点数一般不同（n_native_sps_tet <= (order+1)^3），消费方不能
    # 直接互换，必须按 tet_basis_mode 分派。
    tet_basis_mode: str = "collapsed"
    D_native_tet: np.ndarray = None
    ref_native_tet: np.ndarray = None
    n_native_sps_tet: int = None
    # native 四面体体积->自身面外插矩阵（Part7 阶段2设计文档"二·五"节+
    # `native_simplex_basis.py::build_native_tet_boundary_extrap`），
    # 键是被排除的局部顶点 0~3（与坍缩坐标的 boundary_extrap_tet 键
    # 是 (axis,side) 元组不同）；只在 tet_basis_mode="native" 时非 None。
    boundary_extrap_native_tet: Dict[int, np.ndarray] = None
    # native 四面体 DG 提升算子（`native_simplex_basis.py::
    # build_native_tet_lift` 文档），把面通量跳跃提升成体积节点修正
    # 贡献——是坍缩坐标方案"1D Radau/VCJH 修正函数 + _distribute_point"
    # 对非张量积单纯形基的唯一正确推广（native 基没有"坍缩计算方向"，
    # 1D 修正函数沿某一轴分布这个概念不适用）。键同样是被排除的局部
    # 顶点 0~3；只在 tet_basis_mode="native" 时非 None。
    lift_native_tet: Dict[int, np.ndarray] = None
    # `D_native_tet`/`lift_native_tet` 零填充到全局统一 SPs 宽度 `n_sps`
    # 之后的版本（`fr/native_tet_padding.py::pad_native_tet_matrix_to_
    # global`，见 Part8 文档"一、核心不变量：零填充块对角"）——生产
    # 残差 kernel（`inviscid.py`/`inviscid_kernel.py` 等）要消费的是
    # 这两个已经填充好的版本，不是上面两个原始（n_native 宽）版本；
    # 保留原始版本是因为部分测试/未来诊断代码可能只关心真实自由度本身，
    # 不需要每次都从填充版本反推。只在 tet_basis_mode="native" 时非 None。
    D_native_tet_padded: np.ndarray = None
    lift_native_tet_padded: Dict[int, np.ndarray] = None
    # native 四面体指数模态滤波器（`native_tet_filter.py::build_native_
    # tet_modal_filter`，抑制混叠失稳，见该模块与 fr/modal_filter.py
    # 文档），填充到全局 n_sps 宽度（`native_tet_padding.py::pad_native_
    # tet_filter_matrix_to_global`——填充块是单位矩阵，不是零，与
    # D_native_tet_padded/lift_native_tet_padded 的"零填充"约定不同，
    # 见该函数文档）。只在 tet_basis_mode="native" 时非 None。
    filter_native_tet_padded: np.ndarray = None

    def get_operators(self) -> Dict[str, np.ndarray]:
        """返回算子字典，兼容旧接口。"""
        return {
            'diff_matrix': self.D_1d,
            'interp_matrix': self.L_interp,
            'g_left': self.g_left,
            'g_right': self.g_right
        }


def generate_fr_operators(order: int, flux_point_type: str = 'radau', tet_basis_mode: str = 'collapsed') -> FROperators:
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
        tet_basis_mode: 四面体体积微分算子选择（Part6 整改计划阶段0/1）：
            - 'collapsed'（默认，行为与此前完全一致）：现有坍缩坐标
              （Duffy变换）方案，`D_3d_tet` 走 `(order+1)^3` 张量积
              立方体族。
            - 'native'：路径C，四面体独立于坍缩坐标构造（见
              `fr/native_simplex_basis.py` 文档），`D_native_tet`/
              `ref_native_tet`/`n_native_sps_tet` 被填充为非 None，
              `D_3d_tet` 等坍缩坐标相关字段仍然照常计算（不是互斥
              关系，只是暂时未被消费——阶段1目前只有体积残差路径
              会读取 native 字段，界面/过积分/滤波器仍然读取坍缩坐标
              字段，属于 Part6 阶段2/3 范围，本参数不改变那些字段）。

    Returns:
        operators: 包含所有预计算算子的 FROperators 对象
    """
    n = order + 1  # SPs 数量
    
    # 1. 生成点集
    sps, _ = gauss_legendre(n)
    
    # 2. 计算一维微分矩阵
    D_1d = compute_diff_matrix_1d(sps)
    
    # 3. 计算三维微分算子（朴素张量积，只对六面体正确；四面体/棱柱见下）
    D_3d = compute_diff_matrix_3d(D_1d)

    # 3b. 四面体/棱柱专用坍缩坐标体积微分矩阵，SPs 与上面完全相同
    # （张量积 Gauss-Legendre 点），只是构造 D 用的基不同——见
    # collapsed_basis.py 与 FROperators.D_3d_tet/D_3d_prism 文档。
    aa, bb, cc = np.meshgrid(sps, sps, sps, indexing="ij")
    ref_cube_sps = np.column_stack([aa.ravel(), bb.ravel(), cc.ravel()])
    D_3d_tet = build_collapsed_diff_matrices("tet", order, ref_cube_sps)
    D_3d_prism = build_collapsed_diff_matrices("prism", order, ref_cube_sps)

    # 3c. 四面体/棱柱专用体积->边界外插矩阵（同一套坍缩坐标模态基），
    # 见 FROperators.boundary_extrap_tet/boundary_extrap_prism 文档。
    boundary_extrap_tet = {}
    boundary_extrap_prism = {}
    for axis in range(3):
        for side in (-1.0, 1.0):
            boundary_extrap_tet[(axis, side)] = build_collapsed_boundary_extrap(
                "tet", order, ref_cube_sps, axis, side
            )
            boundary_extrap_prism[(axis, side)] = build_collapsed_boundary_extrap(
                "prism", order, ref_cube_sps, axis, side
            )

    # 3d. 指数模态滤波矩阵（见 fr/modal_filter.py 文档 与
    # FROperators.filter_tet/filter_prism 文档）。
    filter_tet = build_tet_modal_filter(order, ref_cube_sps)
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

    # 6. 体积项去混叠算子（order==0 时 P0 走独立有限体积路径，不需要，
    # 见 grid/high_order_mesh.py::_build_order_geometry 里 jacobians_fine
    # 同样按 order>=1 跳过的说明——两处的 over_order=2*order 约定必须
    # 一致）。
    overint_order = None
    overint_ref_fine = None
    overint_interp_c2f_tet = overint_interp_c2f_prism = None
    overint_D_fine_tet = overint_D_fine_prism = None
    overint_restrict_f2c_tet = overint_restrict_f2c_prism = None
    if order >= 1:
        # 必须与 grid/high_order_mesh.py::_build_order_geometry 里
        # jacobians_fine 的 over_order 计算逐字一致（同一个上限常量），
        # 否则 fr_residual_inviscid.py 里插值/微分/限制三个算子的形状
        # 会和 mesh.jacobians_fine 对不上。
        overint_order = min(2 * order, OVERINTEGRATION_MAX_ORDER)
        overint_ref_fine, overint_interp_c2f_tet, overint_D_fine_tet, overint_restrict_f2c_tet = (
            build_overintegration_operators("tet", order, overint_order, ref_cube_sps)
        )
        _, overint_interp_c2f_prism, overint_D_fine_prism, overint_restrict_f2c_prism = (
            build_overintegration_operators("prism", order, overint_order, ref_cube_sps)
        )

    # 3e. 四面体路径C（Part6 阶段0/1）：只在显式请求时构造，不影响
    # 'collapsed'（默认）路径的任何既有计算或返回值。
    ref_native_tet = D_native_tet = None
    n_native_sps_tet = None
    boundary_extrap_native_tet = None
    lift_native_tet = None
    D_native_tet_padded = None
    lift_native_tet_padded = None
    filter_native_tet_padded = None
    if tet_basis_mode == "native":
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

        # native 单纯形基过积分（去混叠）算子（Part8 文档"四·七"节）：
        # 直接覆盖上面（第 6 步）已经无条件按坍缩坐标构造好的
        # overint_interp_c2f_tet/overint_D_fine_tet/overint_restrict_
        # f2c_tet——两套算子字段名相同、消费方式相同（inviscid.py 的
        # 过积分分支不需要感知 tet_basis_mode，直接读 ops.overint_*_tet
        # 即可），只是这里换成 native 单纯形基构造。fine 网格宽度沿用
        # 与棱柱相同的 (over_order+1)^3（`n_sps_per_cell_fine`，见
        # high_order_mesh_order.py::build_order_geometry native 分支），
        # native 自己的 fine 点数 `(over_order+1)(over_order+2)
        # (over_order+3)/6` 严格更小——D_fine/interp_c2f/restrict_f2c
        # 三者都需要按"零填充块对角"（Part8 文档"一"节）填充到这个
        # 全局宽度；interp_c2f/restrict_f2c 的两个轴分别对应不同的
        # native 长度（fine 轴 vs coarse 轴），`pad_native_tet_matrix_
        # to_global` 一次只处理同一个 native 长度的轴集合，因此分两步
        # 各自填充对应的轴，而不是一次性传两个轴（详见该函数文档）。
        if order >= 1:
            from .native_tet_overintegration import build_native_tet_overintegration_operators

            ref_fine_native, interp_c2f_native, D_fine_native, restrict_f2c_native = (
                build_native_tet_overintegration_operators(order, overint_order)
            )
            n_fine_global = (overint_order + 1) ** 3

            overint_ref_fine = ref_fine_native
            overint_D_fine_tet = pad_native_tet_matrix_to_global(
                D_fine_native, n_fine_global, pad_axes=(0, 1)
            )
            _interp_col_padded = pad_native_tet_matrix_to_global(
                interp_c2f_native, n_sps_global, pad_axes=(1,)
            )
            overint_interp_c2f_tet = pad_native_tet_matrix_to_global(
                _interp_col_padded, n_fine_global, pad_axes=(0,)
            )
            _restrict_col_padded = pad_native_tet_matrix_to_global(
                restrict_f2c_native, n_fine_global, pad_axes=(1,)
            )
            overint_restrict_f2c_tet = pad_native_tet_matrix_to_global(
                _restrict_col_padded, n_sps_global, pad_axes=(0,)
            )
    elif tet_basis_mode != "collapsed":
        raise ValueError(f"未知 tet_basis_mode: {tet_basis_mode!r}，只接受 'collapsed' 或 'native'")

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
        overint_interp_c2f_tet=overint_interp_c2f_tet,
        overint_interp_c2f_prism=overint_interp_c2f_prism,
        overint_D_fine_tet=overint_D_fine_tet,
        overint_D_fine_prism=overint_D_fine_prism,
        overint_restrict_f2c_tet=overint_restrict_f2c_tet,
        overint_restrict_f2c_prism=overint_restrict_f2c_prism,
        tet_basis_mode=tet_basis_mode,
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
