"""直边四面体/棱柱的解析精确雅可比（从 curved_mapping.py 拆分）。

从 curved_mapping.py 中拆分出来（原文件超过 400 行的项目约定上限）：
tet_exact_jacobian / prism_exact_jacobian 是一对纯函数，只依赖 numpy 和
传入的参考坐标/顶点坐标，不与 curved_mapping.py 里的 CurvedMapping 类或
Duffy 坍缩坐标变换函数共享任何模块级状态，是天然独立的一个数学工具单元。
curved_mapping.py 在模块顶层 `from .curved_mapping_exact_jacobian import
tet_exact_jacobian, prism_exact_jacobian` 原样重新导出，任何既有的
`from autoflowcfd.grid.curved_mapping import tet_exact_jacobian` 之类
导入路径不受影响；CurvedMapping.compute_jacobian（含本次会话早些时候修复
的 total_sps UnboundLocalError bug）逻辑完全未改动，只是从这里调用同样
的函数对象。

设计动机说明（原样保留自 curved_mapping.py）：
----------------------------------------------------------------
map_tet_to_physical / map_prism_to_physical 只用顶点节点做重心坐标插值，
是 (a,b,c) 的已知闭式表达式（不是未知的高阶流场，不需要谱微分矩阵近似）。
用固定阶数的谱微分矩阵（哪怕是坍缩坐标专用的 D_3d_tet/D_3d_prism）对这个
闭式映射求导，仍然是对真实（可能是有理函数）度量场的截断/插值近似，会
引入随机误差；对细长偏斜单元（真实网格中棱柱-四面体过渡区常见，边长比
可达 25:1），该误差量级不随 det(J) 一起等比例缩小，导致离散 GCL 恒等式
（见 compute_metric_identity_residual）在这些单元上不能精确成立——真实
网格上实测：谱微分给出的 GCL 残差稳定在 ~1e-14（绝对量级，与单元偏斜、
det(J) 大小无关），当 det(J) 本身只有 ~2e-14 时，相对误差被放大到 18%。

这里改用对闭式映射的解析（符号）求导：tet 情形，物理坐标是参考单纯形
坐标 (r,s,t) 的仿射函数（dx/dr、dx/ds、dx/dt 是与位置无关的常向量），
再用 Duffy 变换 (a,b,c)->(r,s,t) 的闭式雅可比做链式法则；prism 情形同理
（三角形坍缩部分是 (a,b) 的仿射函数，c 方向是精确线性混合）。全程没有
任何插值/截断，只有初等微积分，因此结果精确到浮点舍入误差为止。
已用有限差分数值核对（误差 ~1e-10，与有限差分自身截断误差一致）；用这里
算出的精确 adj(J) 代入 D_3d_tet 做离散散度检验，真实网格最差单元的 GCL
残差从 ~1e-14 降到 ~1e-19，与单元偏斜程度、det(J) 大小无关。
"""

import numpy as np


def tet_exact_jacobian(ref_cube_sps: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """直边四面体的解析精确雅可比 J[:, :, m] = d(phys)/d(xi_m)，m=0,1,2 对应 a,b,c。

    Args:
        ref_cube_sps: 计算立方体坐标 (a,b,c)，形状 (n_pts, 3)
        cell_nodes: 四面体 4 个顶点物理坐标，形状 (4, 3)，顺序需与
            fix_tet_orientation 保证的正体积顺序一致（同 map_tet_to_physical）

    Returns:
        雅可比矩阵，形状 (n_pts, 3, 3)
    """
    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    p0, p1, p2, p3 = cell_nodes
    e1 = (p1 - p0) / 2.0  # dx/dr（常向量，物理坐标对参考四面体坐标是仿射的）
    e2 = (p2 - p0) / 2.0  # dx/ds
    e3 = (p3 - p0) / 2.0  # dx/dt

    # Duffy 变换 (a,b,c)->(r,s,t) 的闭式雅可比（s,t 是 b,c 的多项式）：
    #   t=c, s=(1+b)(1-c)/2-1, r=-(1+a)(s+t)/2-1
    s = (1.0 + b) * (1.0 - c) / 2.0 - 1.0
    t = c

    n = ref_cube_sps.shape[0]
    jac = np.zeros((n, 3, 3))
    coef_a = -(s + t) / 2.0  # dr/da
    jac[:, :, 0] = coef_a[:, None] * e1[None, :]

    coef_b1 = -(1.0 + a) * (1.0 - c) / 4.0  # dr/db
    coef_b2 = (1.0 - c) / 2.0  # ds/db
    jac[:, :, 1] = coef_b1[:, None] * e1[None, :] + coef_b2[:, None] * e2[None, :]

    coef_c1 = -(1.0 + a) * (1.0 - b) / 4.0  # dr/dc
    coef_c2 = -(1.0 + b) / 2.0  # ds/dc （dt/dc=1，贡献 e3）
    jac[:, :, 2] = coef_c1[:, None] * e1[None, :] + coef_c2[:, None] * e2[None, :] + e3[None, :]
    return jac


def prism_exact_jacobian(ref_cube_sps: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """直边棱柱的解析精确雅可比 J[:, :, m] = d(phys)/d(xi_m)，m=0,1,2 对应 a,b,c。

    Args:
        ref_cube_sps: 计算立方体坐标 (a,b,c)，形状 (n_pts, 3)
        cell_nodes: 棱柱 6 个顶点物理坐标，形状 (6, 3)，顺序同
            map_prism_to_physical (v0,v1,v2,w0,w1,w2)

    Returns:
        雅可比矩阵，形状 (n_pts, 3, 3)
    """
    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    p0, p1, p2, p3, p4, p5 = cell_nodes

    # 三角形坍缩部分 bottom(a,b)/top(a,b) 对 a,b 的解析偏导（重心坐标
    # l1,l2,l3 是 (r,s) 的仿射函数，(r,s)=cube_to_tri_rs(a,b) 是 (a,b) 的
    # 多项式：r=(1+a)(1-b)/2-1, s=b）。
    d_bottom_da = ((1.0 - b) / 4.0)[:, None] * (p1 - p0)[None, :]
    d_top_da = ((1.0 - b) / 4.0)[:, None] * (p4 - p3)[None, :]
    d_bottom_db = (
        (-(1.0 - a) / 4.0)[:, None] * p0[None, :]
        + (-(1.0 + a) / 4.0)[:, None] * p1[None, :]
        + 0.5 * p2[None, :]
    )
    d_top_db = (
        (-(1.0 - a) / 4.0)[:, None] * p3[None, :]
        + (-(1.0 + a) / 4.0)[:, None] * p4[None, :]
        + 0.5 * p5[None, :]
    )

    r = (1.0 + a) * (1.0 - b) / 2.0 - 1.0
    s = b
    l1 = -(r + s) / 2.0
    l2 = (1.0 + r) / 2.0
    l3 = (1.0 + s) / 2.0
    bottom = l1[:, None] * p0[None, :] + l2[:, None] * p1[None, :] + l3[:, None] * p2[None, :]
    top = l1[:, None] * p3[None, :] + l2[:, None] * p4[None, :] + l3[:, None] * p5[None, :]

    n = ref_cube_sps.shape[0]
    jac = np.zeros((n, 3, 3))
    jac[:, :, 0] = 0.5 * (1.0 - c)[:, None] * d_bottom_da + 0.5 * (1.0 + c)[:, None] * d_top_da
    jac[:, :, 1] = 0.5 * (1.0 - c)[:, None] * d_bottom_db + 0.5 * (1.0 + c)[:, None] * d_top_db
    jac[:, :, 2] = 0.5 * (top - bottom)  # dx/dc 精确闭式，与 a,b 无关的线性混合
    return jac
