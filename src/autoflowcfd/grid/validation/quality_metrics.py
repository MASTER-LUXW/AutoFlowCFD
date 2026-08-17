"""纯粹的逐单元几何质量指标计算。

向量化（不含 Python 级逐单元循环）函数，是 MeshQualityValidator
（quality_validator.py）汇总统计背后的原始数组来源——从该模块拆分出来，
让检查/编排逻辑不与这些自包含的几何公式交织在一起。这里每个函数都是
(nodes, cells) 的纯函数：不涉及网格生成或修复的概念，也没有状态。
"""

import numpy as np


def compute_tetrahedron_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个四面体的有符号体积: det(p1-p0, p2-p0, p3-p0) / 6。

    注意: 保留符号（而非取绝对值），这样上游的负体积/反转单元检查
    才有意义；其他地方的幅度统计无论如何只使用正子集。
    """
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    p3 = nodes[cells[:, 3]]

    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0

    return np.einsum('ij,ij->i', v1, np.cross(v2, v3)) / 6.0


def compute_triangle_areas(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个三角形的面积: 0.5 * |cross(p1-p0, p2-p0)|。"""
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]

    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def triangle_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个三角形的边长, shape=(n_cells, 3)。"""
    p0, p1, p2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
    e1 = np.linalg.norm(p1 - p0, axis=1)
    e2 = np.linalg.norm(p2 - p1, axis=1)
    e3 = np.linalg.norm(p0 - p2, axis=1)
    return np.stack([e1, e2, e3], axis=1)


def tetrahedron_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个四面体的全部 6 条边长, shape=(n_cells, 6)。"""
    pts = nodes[cells]  # (n_cells, 4, 3)
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.linalg.norm(pts[:, i] - pts[:, j], axis=1))
    return np.stack(edges, axis=1)


def compute_triangle_aspect_ratios(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """AR = 最长边 / 最短边，对每个三角形（1.0 = 等边）。

    分母下限取三角形自身最长边的一个小比例，而非固定绝对 epsilon——
    见 compute_prism_aspect_ratios 的文档字符串了解原因：网格的边长
    从毫米到米取决于 min_cell_size，因此像 1e-12 这样的常量比任何
    合法边都低好几个数量级，会让近退化（但非零面积）三角形报告
    物理无意义的比值（例如 ~1e10+），淹没其他所有单元的信号。
    """
    edges = triangle_edge_lengths(nodes, cells)
    max_edge = np.max(edges, axis=1)
    min_edge = np.min(edges, axis=1)
    return max_edge / np.maximum(min_edge, max_edge * 1e-6)


def compute_tetrahedron_aspect_ratios(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """AR = 每个四面体全部 6 条边中的最长边 / 最短边。

    分母下限取四面体自身最长边的一个小比例——
    见 compute_triangle_aspect_ratios/compute_prism_aspect_ratios 的文档字符串
    了解为何此处不能用固定绝对 epsilon。
    """
    edges = tetrahedron_edge_lengths(nodes, cells)
    max_edge = np.max(edges, axis=1)
    min_edge = np.min(edges, axis=1)
    return max_edge / np.maximum(min_edge, max_edge * 1e-6)


def compute_triangle_skewness_values(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """通过标准等角偏斜公式计算每个三角形的偏斜度
    （与 Fluent/ANSYS Meshing 报告相同定义），范围 [0, 1]：

        skew = max[ (theta_max - 60) / (180 - 60), (60 - theta_min) / 60 ]

    其中 theta_max/theta_min 是三角形的最大/最小角（度），
    60 度是等边参考角。

    此公式替代了早期公式 `min(max(|angle-60|)/60, 1.0)`，
    该公式在任何角度 >= 120 度时都饱和于恰好 1.0——
    120 度角（一个正常、有效、适度拉长的三角形——例如 BL 棱柱
    在凸角周围挤出扇形展开时的顶部三角形）和 179.99 度角
    （一个真正退化的近零面积碎片）都报告相同的值 1.0，
    无法区分彼此或下面的“退化”（近零边）情况。已在真实案例中
    确认（ProjectFiles Part... 网格质量后续）：一个角度为
    (123, 29, 28) 度的 BL 棱柱顶部——面积 38.5 mm²，远非退化，
    ANSA 自身的质量检查也同意是有效元素——在旧公式下得分
    饱和的 1.0。等角偏斜公式则随角度趋近 0/180 度极限
    连续增长到 1.0（同样的 123 度角现在得分 ~0.53，“中等偏斜”——
    0.95 阈值本项目已经使用，与 Fluent 自身在此相同缩放上的
    “poor/sliver” 截断值匹配，因此阈值无需改变，只需改变
    计算与之比较值的公式）。
    """
    p0, p1, p2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)

    degenerate = (a < 1e-12) | (b < 1e-12) | (c < 1e-12)
    # Guard the law-of-cosines division for degenerate triangles; their
    # skewness is overridden to the worst value (1.0) below regardless.
    safe_b = np.where(degenerate, 1.0, b)
    safe_c = np.where(degenerate, 1.0, c)
    safe_a = np.where(degenerate, 1.0, a)

    cos0 = np.clip((safe_b**2 + safe_c**2 - safe_a**2) / (2 * safe_b * safe_c), -1.0, 1.0)
    cos1 = np.clip((safe_a**2 + safe_c**2 - safe_b**2) / (2 * safe_a * safe_c), -1.0, 1.0)
    angle_0 = np.arccos(cos0)
    angle_1 = np.arccos(cos1)
    angle_2 = np.pi - angle_0 - angle_1

    angles_deg = np.degrees(np.stack([angle_0, angle_1, angle_2], axis=1))
    theta_max = np.max(angles_deg, axis=1)
    theta_min = np.min(angles_deg, axis=1)
    skew_max = (theta_max - 60.0) / (180.0 - 60.0)
    skew_min = (60.0 - theta_min) / 60.0
    skewness = np.clip(np.maximum(skew_max, skew_min), 0.0, 1.0)
    skewness[degenerate] = 1.0

    return skewness


def compute_tetrahedron_skewness_values(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """通过半径比质量度量计算每个四面体的偏斜度：
    1 - 3*r_in/r_circ (0=正四面体, ->1=碎片)。

    标准四面体形状质量度量（与 Verdict/CUBIT 的 TetRadiusRatio
    仅差此 0..1 归一化）。已通过已知案例验证：正四面体 ->
    r_in/r_circ == 1/3 精确（偏斜度=0）；近扁平退化四面体 ->
    偏斜度 ~1。

    r_in = 3V/表面积（标准四面体内径公式）。
    r_circ 通过向量外接半径公式：从一个顶点出发的边向量 a,b,c，
    R = |a|²(b×c) + |b|²(c×a) + |c|²(a×b)（向量和，然后取模）/ (12V)。
    """
    p0, p1, p2, p3 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]], nodes[cells[:, 3]]

    def tri_area(A, B, C):
        return 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)

    # 4 个面，每个面对对一个顶点
    area_opp_p0 = tri_area(p1, p2, p3)
    area_opp_p1 = tri_area(p0, p2, p3)
    area_opp_p2 = tri_area(p0, p1, p3)
    area_opp_p3 = tri_area(p0, p1, p2)
    surface_area = area_opp_p0 + area_opp_p1 + area_opp_p2 + area_opp_p3

    a = p1 - p0
    b = p2 - p0
    c = p3 - p0
    volume = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0

    r_in = 3.0 * volume / np.maximum(surface_area, 1e-300)

    a2 = np.einsum('ij,ij->i', a, a)
    b2 = np.einsum('ij,ij->i', b, b)
    c2 = np.einsum('ij,ij->i', c, c)
    circum_vec = (
        a2[:, None] * np.cross(b, c)
        + b2[:, None] * np.cross(c, a)
        + c2[:, None] * np.cross(a, b)
    )
    r_circ = np.linalg.norm(circum_vec, axis=1) / np.maximum(12.0 * volume, 1e-300)

    radius_ratio = 3.0 * r_in / np.maximum(r_circ, 1e-300)
    skewness = 1.0 - np.clip(radius_ratio, 0.0, 1.0)

    degenerate = volume < 1e-300
    skewness[degenerate] = 1.0

    return skewness


# ---------------------------------------------------------------------------
# 三棱柱（BL 单元）度量。
#
# 连接关系约定, shape=(n_cells, 6): (v0, v1, v2, w0, w1, w2) -
# v0..v2 是底层三角形，w0..w2 是顶层三角形，
# w_i 是 v_i 的挤出对应点（与 mesh_extrusion.py/
# mesh_prism_to_tet.py 已有的层节点对应约定相同——
# w_i 在 v_i “正上方”，而非任意顶点排列）。
# ---------------------------------------------------------------------------

def compute_prism_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个三棱柱的无符号体积，通过精确的 3-四面体分解
    T1=(v0,v1,v2,w2), T2=(v0,v1,w1,w2), T3=(v0,w0,w1,w2)——
    与 mesh_prism_to_tet.convert_layers_to_tetrahedra 使用的
    对角线一致拆分相同，因此此处棱柱的体积始终精确等于
    旧的“拆分为 3 个四面体”表示的体积之和，无论棱柱是否为
    “正”棱柱（平面四边形侧面，无扭转）。

    每个子四面体的贡献取 |有符号体积|：上述 (v0,v1,v2,w2) 风格
    的顶点元组并未单独定向以得到一致符号的结果
    （mesh_prism_to_tet 自身的四面体仅通过单独的
    orient_tetrahedra 步骤获得该保证，而此函数不复制该步骤）——
    已确认，三个子四面体之一在普通非退化棱柱上输出负值。
    取模对体积仍然精确（三个子四面体无重叠地铺满棱柱，
    与各自的索引顺序符号无关），仅意味着此函数——
    不像 compute_tetrahedron_volumes——不能同时用作反转/负体积
    检查；如果需要的话，需要专门的方向测试。
    """
    v0, v1, v2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
    w0, w1, w2 = nodes[cells[:, 3]], nodes[cells[:, 4]], nodes[cells[:, 5]]

    def tet_vol(p0, p1, p2, p3):
        return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    return tet_vol(v0, v1, v2, w2) + tet_vol(v0, v1, w1, w2) + tet_vol(v0, w0, w1, w2)


def prism_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个棱柱的全部 9 条边长, shape=(n_cells, 9)：3 条底边
    + 3 条顶边 + 3 条竖直（近似法向）边，按此顺序。"""
    pts = nodes[cells]  # (n_cells, 6, 3)
    v0, v1, v2 = pts[:, 0], pts[:, 1], pts[:, 2]
    w0, w1, w2 = pts[:, 3], pts[:, 4], pts[:, 5]
    edges = [
        np.linalg.norm(v1 - v0, axis=1), np.linalg.norm(v2 - v1, axis=1), np.linalg.norm(v0 - v2, axis=1),
        np.linalg.norm(w1 - w0, axis=1), np.linalg.norm(w2 - w1, axis=1), np.linalg.norm(w0 - w2, axis=1),
        np.linalg.norm(w0 - v0, axis=1), np.linalg.norm(w1 - v1, axis=1), np.linalg.norm(w2 - v2, axis=1),
    ]
    return np.stack(edges, axis=1)


def compute_prism_aspect_ratios(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """AR = 每个棱柱全部 9 条边中的最长边 / 最短边。

    与四面体不同，此处的高长宽比通常是有意且正确的
    （近壁 BL 棱柱*应该*是薄的：顶部边 ~mm，竖直边
    同样小，但连续层竖直边之间的比值——而非此 per-cell 比值——
    才控制增长率合理性）——这就是为什么验证器对棱柱长宽比
    应用分离的、更宽松的 BL 区域阈值（见 quality_validator.py），
    与对 BL 区域四面体长宽比的处理方式相同。

    分母下限取单元自身最长边的一个小比例，而非固定绝对 epsilon——
    网格的边长从毫米到米取决于 min_cell_size，因此像 1e-12 这样的
    常量比任何合法边都低好几个数量级，根本不提供实际下限。
    这对“坍缩角”棱柱尤为重要（BL 列在恰好一个底顶点处
    增长冻结——见 mesh_prism_to_tet.py / ProjectFiles Part6 Bug 4——
    一个有效、非零体积但有一条真正近零竖直边的单元）：
    用旧的 epsilon 会报告物理无意义的比值（在真实案例上测量：5.11e10），
    淹没质量报告中的其他所有数值。改为相对于单元自身缩放取下限，
    将此类单元的比值截断为 1e6——仍明确标记为错误
    （没有合法的单元需要 6 个数量级的边长跨度），但有界且不误导。
    """
    edges = prism_edge_lengths(nodes, cells)
    max_edge = np.max(edges, axis=1)
    min_edge = np.min(edges, axis=1)
    return max_edge / np.maximum(min_edge, max_edge * 1e-6)


def compute_prism_skewness_values(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个棱柱的偏斜度：max(底面, 顶面) 三角形偏斜度
    （等角偏斜，与 compute_triangle_skewness_values 相同公式）。

    刻意不纳入“竖直度”（3 条竖直边与顶部法向的接近程度，
    即剪切/扭转）——那是一个真正不同的缺陷类别（控制棱柱自身
    侧面的非正交性，而非截面的碎片性），已由现有的基于面的
    正交性检查覆盖（compute_face_diagnostics），它在棱柱的三角化
    侧面上的工作方式与其他任何内部面相同。将两者折叠到一个
    数值会让一个具有完美规则顶部但严重剪切（或反之）的棱柱
    将其最差维度隐藏在另一个的更好值后面。
    """
    bottom = compute_triangle_skewness_values(nodes, cells[:, 0:3])
    top = compute_triangle_skewness_values(nodes, cells[:, 3:6])
    return np.maximum(bottom, top)
