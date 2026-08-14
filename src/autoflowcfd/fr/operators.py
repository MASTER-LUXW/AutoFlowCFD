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
from .collapsed_basis import build_collapsed_diff_matrices, build_collapsed_boundary_extrap
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
    
    def get_operators(self) -> Dict[str, np.ndarray]:
        """返回算子字典，兼容旧接口。"""
        return {
            'diff_matrix': self.D_1d,
            'interp_matrix': self.L_interp,
            'g_left': self.g_left,
            'g_right': self.g_right
        }


def generate_fr_operators(order: int, flux_point_type: str = 'lobatto') -> FROperators:
    """
    生成完整的 FR 算子集合。
    
    Args:
        order: 多项式阶数 P
        flux_point_type: FP 类型 ('lobatto' 或 'radau')
        
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

    # 4. 计算插值矩阵
    if flux_point_type == 'lobatto':
        fps, _ = gauss_lobatto(n + 1)
    else:
        fps = np.concatenate([[-1.0], sps, [1.0]])
    
    L_interp = compute_interpolation_matrix(sps, fps)
    
    # 5. 计算校正权重
    g_left, g_right = compute_correction_weights(n, flux_point_type)
    
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
