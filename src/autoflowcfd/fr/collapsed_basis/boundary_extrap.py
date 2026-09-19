"""AutoFlowCFD V2.0 - 坍缩基体积 -> 边界面外插算子。

从 `fr/collapsed_basis.py`（原 610 行）拆出（2026-09-19）。纯搬家，逻辑
未改。
"""

import numpy as np

from .modal import prism_modal_basis_and_grad, tet_modal_basis_and_grad


def build_collapsed_boundary_extrap(
    cell_type: str, order: int, ref_cube_sps: np.ndarray, axis: int, side: float
) -> np.ndarray:
    """把体积 SPs 上的节点值外插到某个立方体边界面（fixed axis=side）
    的 Flux Points，用与 build_collapsed_diff_matrices 同一套坍缩坐标
    模态基构造，而不是 fr/face_flux_points.py::extrapolate_to_face 现在
    用的朴素 1D 张量积 Lagrange 外插。

    背景：extrapolate_to_face 对固定轴做 1D Lagrange 边界外插、其余两个
    轴按原生 SP 网格索引直接对应，这个简化对朴素张量积基完全等价于
    "在该点求整张三维张量积插值多项式的值"——但对坍缩坐标单元，度量项
    adj(J) 这类场在退化边附近变化剧烈（真实网格验证：a=+1 面上外插出的
    等效法向方向与真实几何法向偏差最大约 28°，虽然仍在现有 60° 校验
    阈值内、不会报错，但足以在残差公式除以（该处真实偏小的）Jacobian
    后被放大到灾难量级），用只有 3 个内部 Gauss-Legendre 点、不含边界点
    的朴素张量积外插去逼近这种剧烈变化在数学上站不住脚——必须换成与
    体积微分矩阵一致的坍缩坐标模态基外插，让边界取值与体积微分共用同一
    个、真正匹配坍缩坐标退化结构的插值空间。

    Args:
        cell_type: "tet" 或 "prism"
        order: 多项式阶数
        ref_cube_sps: 体积 SPs 参考坐标 (n_sps,3)（与 build_collapsed_
            diff_matrices 用的是同一组点）
        axis: 被固定的坍缩坐标轴（0=a,1=b,2=c）
        side: 该轴的边界取值（-1.0 或 1.0）

    Returns:
        E: (n_fp, n_sps)，E @ field(SPs) 给出 field 在该边界面 Flux
        Points（其余两个轴仍取原生张量积 Gauss-Legendre 网格）处的取值，
        Flux Points 展平顺序与 fr/face_flux_points.py::face_ref_grid
        完全一致（other_axes[0] 为外层、other_axes[1] 为内层）。

    重要更正（V2.0 专家组评审后续复核）：本函数曾一度被怀疑病态到不可用
    （P2 阶 c=-1 面 Lebesgue 常数 4489，vs 朴素张量积基约 2.3），并计划
    替换成朴素张量积 Lagrange 外插。复核发现原测试方法论有误：对照用的
    "真值"是按每轴独立次数构造的多项式（tensor-degree-per-axis），这类
    多项式本来就精确落在朴素张量基的展开空间里、不落在本函数所用坍缩
    模态基（tet_modal_basis_and_grad/prism_modal_basis_and_grad）的展开
    空间里——两者在边界点的取值不一致是两个不同多项式空间的必然结果，
    不是浮点病态的证据；朴素基"精确"只是同义反复（重现了它自己就能精确
    表示的函数）。

    改用坍缩模态基**自己空间内**的随机模态系数正向构造节点值与边界真值
    （V_sps@coeffs、V_fp@coeffs，独立于本函数实现）重新验证，结果与模块
    顶部文档已有的量级声明一致：P2（生产阶数）下 c=-1 面相对误差
    ~8.7e-13，P3（非生产阶数）下 ~1.3e-8，均为可接受的浮点舍入水平，
    不存在灾难性放大。Lebesgue 常数 4489 是最坏情形理论上界（对抗性噪声
    方向），不是典型输入下的实际误差；用它单独判断"病态到需要换基"是
    误导的。**结论：不能换成朴素张量积基**——那是四面体/棱柱物理场在
    坍缩坐标下真正所属的多项式空间（模块顶部文档已详述、且与
    build_collapsed_diff_matrices 共用同一套基），换成朴素基虽然数值上
    "更好看"，实际是把正确的单纯形插值空间换成错误的空间，属于精度倒退，
    不是修复。本函数维持原实现不变。
    """
    n1d = order + 1
    other_axes = [a for a in range(3) if a != axis]
    # ref_cube_sps 是 1D 节点集合 sps_1d 的三维张量积，任一维展开后按
    # 升序取唯一值即可还原 sps_1d（不依赖调用方单独传入）。
    sps_1d = np.unique(ref_cube_sps[:, 0])

    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    fp_pts = np.zeros((n1d * n1d, 3))
    fp_pts[:, axis] = side
    fp_pts[:, other_axes[0]] = g1.ravel()
    fp_pts[:, other_axes[1]] = g2.ravel()

    a_sps, b_sps, c_sps = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    a_fp, b_fp, c_fp = fp_pts[:, 0], fp_pts[:, 1], fp_pts[:, 2]
    if cell_type == "tet":
        V_sps, _, _, _ = tet_modal_basis_and_grad(a_sps, b_sps, c_sps, order)
        V_fp, _, _, _ = tet_modal_basis_and_grad(a_fp, b_fp, c_fp, order)
    elif cell_type == "prism":
        V_sps, _, _, _ = prism_modal_basis_and_grad(a_sps, b_sps, c_sps, order)
        V_fp, _, _, _ = prism_modal_basis_and_grad(a_fp, b_fp, c_fp, order)
    else:
        raise ValueError(f"Unknown cell_type for collapsed boundary extrapolation: {cell_type!r}")

    # 同 build_collapsed_diff_matrices：用 lu_solve 而不是显式求逆，控制
    # V_sps 条件数带来的舍入放大（同一份 V_sps^{-1} 会被同一 (cell_type,
    # order) 的所有单元共享，值得用分解而不是每次都算一次 inv）。
    from scipy.linalg import lu_factor, lu_solve

    lu_piv = lu_factor(V_sps.T)
    return lu_solve(lu_piv, V_fp.T).T
