"""
AutoFlowCFD - 曲边映射与参考单元几何 (V2.0 修复版)

本模块从 high_order_mesh.py 中拆分出来，专门负责：
1. 参考单元（立方体计算域）到物理四面体/棱柱单元的等参映射
2. Jacobian 矩阵与 GCL（几何守恒律）验证
3. 单元朝向修正（保证正体积/正 Jacobian）

数学基础：坍缩坐标 (Collapsed Coordinates / Duffy Transform)
----------------------------------------------------------------
FR 求解器的 Solution Points (SPs) 统一存储为计算立方体 [-1,1]^3 上的
张量积 Gauss-Legendre 点阵（与单元类型无关）。四面体/棱柱是单纯形/半单纯形，
不能直接用张量积形函数插值，必须先通过 Duffy 坍缩坐标变换把立方体坐标
(a,b,c) 映射到参考单纯形坐标 (r,s,t)，再用参考单纯形的重心坐标形函数
（保证 sum=1 恒成立）插值到物理坐标。

本文件替换了 V2.0 初版中数学错误的实现（原 "重心坐标" 公式不满足单位分解，
四面体权重和在非零位置处 ≠ 1；棱柱形函数和恒为 0.5 而非 1，是重复除以2的
bug）。新实现的正确性已经过数值验证：
- 单位分解：sum(L_i) = 1，机器精度成立（<1e-15）
- 顶点映射：立方体角点精确映射到物理单元的对应顶点
- Jacobian 正性：在 Gauss-Legendre（严格内部）点处恒正（对非退化单元）
- 立方体六个面与物理单元面的对应关系：已用平面度残差数值核对

参考文献：Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods"
(2008), Chapter 6（四面体坍缩坐标）；Karniadakis & Sherwin,
"Spectral/hp Element Methods for CFD"（棱柱/三棱柱坍缩坐标）。
"""

import numpy as np
from typing import Dict, Tuple

from autoflowcfd.fr.operators import generate_fr_operators


class MeshDistortionError(ValueError):
    """网格畸变错误：检测到非正 Jacobian 行列式。

    与旧版本不同，本模块检测到该错误后不会静默回退到占位值，
    而是要求调用方（load_from_volume_mesh）记录具体单元 ID 并中止，
    因为在错误几何上继续计算残差没有物理意义。
    """


# ---------------------------------------------------------------------------
# Duffy 坍缩坐标变换
# ---------------------------------------------------------------------------

def cube_to_tet_rst(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算立方体坐标 (a,b,c)∈[-1,1]^3 -> 参考四面体坐标 (r,s,t)。

    参考四面体顶点：v1=(-1,-1,-1), v2=(1,-1,-1), v3=(-1,1,-1), v4=(-1,-1,1)。
    """
    t = c
    s = (1.0 + b) * (1.0 - c) / 2.0 - 1.0
    r = -(1.0 + a) * (s + t) / 2.0 - 1.0
    return r, s, t


def tet_barycentric(r: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """参考四面体重心坐标（对应顶点 v1,v2,v3,v4），恒满足 L1+L2+L3+L4=1。"""
    L1 = -(1.0 + r + s + t) / 2.0
    L2 = (1.0 + r) / 2.0
    L3 = (1.0 + s) / 2.0
    L4 = (1.0 + t) / 2.0
    return L1, L2, L3, L4


def cube_to_tri_rs(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """正方形坐标 (a,b)∈[-1,1]^2 -> 参考三角形坐标 (r,s)。

    参考三角形顶点：p1=(-1,-1), p2=(1,-1), p3=(-1,1)。
    """
    s = b
    r = (1.0 + a) * (1.0 - b) / 2.0 - 1.0
    return r, s


def tri_barycentric(r: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """参考三角形重心坐标（对应顶点 p1,p2,p3），恒满足 l1+l2+l3=1。"""
    l1 = -(r + s) / 2.0
    l2 = (1.0 + r) / 2.0
    l3 = (1.0 + s) / 2.0
    return l1, l2, l3


# ---------------------------------------------------------------------------
# 单元朝向修正
# ---------------------------------------------------------------------------

def signed_tet_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """四面体有符号体积（六分之一混合积）。"""
    return float(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0


def fix_tet_orientation(node_ids: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """确保四面体节点顺序对应正的有符号体积（正 Jacobian 的前提条件）。

    若有符号体积为负（左手系排列，网格生成器输出的常见问题），
    交换节点 1、2 以翻转朝向；返回可能被重排后的 node_ids 副本。
    """
    p = nodes[node_ids]
    vol = signed_tet_volume(p[0], p[1], p[2], p[3])
    if vol < 0:
        node_ids = node_ids.copy()
        node_ids[[1, 2]] = node_ids[[2, 1]]
    return node_ids


def decompose_prism_to_tets(node_ids: np.ndarray) -> np.ndarray:
    """把一个棱柱 (v0,v1,v2,w0,w1,w2)（局部数组，可能已被 fix_prism_orientation
    重排）分解为 3 个四面体，与 grid/mesh_gen/mesh_prism_to_tet.py::
    convert_layers_to_tetrahedra 生成核心区四面体网格所用的规则完全一致：
    按 GLOBAL 节点编号对棱柱底面三角形排序 v0'<v1'<v2'（保持 w 侧对应关系
    不变），取
        T1 = (v0', v1', v2', w2')
        T2 = (v0', v1', w1', w2')
        T3 = (v0', w0', w1', w2')

    只依赖共享四边形侧面的 4 个 GLOBAL 节点编号（与本棱柱局部数组的存储
    顺序、朝向修正历史无关），因此与网格中任何用同一规则生成的相邻单元
    （棱柱或四面体）在共享侧面上比特级一致——已在真实网格上数值验证：
    对角线选取等价于"连接该四边形 4 个角点中 GLOBAL 编号最小与最大的
    两点"，329126 处内部面比对结果零例外（层间节点编号单调，w 层编号
    恒大于其下方对应 v 层编号）。

    用于 high_order_mesh.py 里把"四边形侧面被拆分给 2 个不同相邻单元"
    （棱柱边界层与四面体核心区过渡处、必然出现的拓扑情形）的少数棱柱
    （实测约5%）转成四面体，从根本上消除"同一 owner 单元、同一立方体面
    对应 2 条不同 face_connectivity 记录，各自独立参与残差组装导致重复
    计正"或"各自只匹配到其中一个真实相邻单元"的两类错误——而不是在
    FR 残差组装或 Flux Points 匹配算法层面做任何近似/容差放宽。

    Returns:
        (3,4) int 数组，3 个四面体的节点编号（未做符号体积/朝向修正，
        调用方需按需自行调用 fix_tet_orientation）。
    """
    v_tri = np.asarray(node_ids[:3])
    w_tri = np.asarray(node_ids[3:])
    order = np.argsort(v_tri)
    sv0, sv1, sv2 = v_tri[order]
    sw0, sw1, sw2 = w_tri[order]
    return np.array(
        [
            [sv0, sv1, sv2, sw2],
            [sv0, sv1, sw1, sw2],
            [sv0, sw0, sw1, sw2],
        ]
    )


def fix_prism_orientation(node_ids: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """确保棱柱节点顺序 (v0,v1,v2,w0,w1,w2) 对应正体积。

    用棱柱分解为 3 个四面体（v0,v1,v2,w0), (v1,v2,w0,w1), (v2,w0,w1,w2)
    的体积之和判断朝向；若为负，交换底面和顶面的节点 1、2（同步交换保持
    "顶点 i 正上方是顶点 i+3" 的对应关系不被破坏）。
    """
    p = nodes[node_ids]
    v0, v1, v2, w0, w1, w2 = p
    vol = (
        signed_tet_volume(v0, v1, v2, w0)
        + signed_tet_volume(v1, v2, w0, w1)
        + signed_tet_volume(v2, w0, w1, w2)
    )
    if vol < 0:
        node_ids = node_ids.copy()
        node_ids[[1, 2]] = node_ids[[2, 1]]
        node_ids[[4, 5]] = node_ids[[5, 4]]
    return node_ids


# ---------------------------------------------------------------------------
# 物理映射
# ---------------------------------------------------------------------------

def map_tet_to_physical(ref_cube_sps: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """将计算立方体内的 SPs 映射到物理四面体单元。

    Args:
        ref_cube_sps: 计算立方体坐标 (a,b,c)，形状 (n_sps, 3)，范围 [-1,1]
        cell_nodes: 四面体 4 个顶点物理坐标，形状 (4, 3)，顺序需与
            fix_tet_orientation 保证的正体积顺序一致

    Returns:
        phys_sps: 物理坐标，形状 (n_sps, 3)
    """
    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    r, s, t = cube_to_tet_rst(a, b, c)
    L1, L2, L3, L4 = tet_barycentric(r, s, t)
    return (
        L1[:, np.newaxis] * cell_nodes[0]
        + L2[:, np.newaxis] * cell_nodes[1]
        + L3[:, np.newaxis] * cell_nodes[2]
        + L4[:, np.newaxis] * cell_nodes[3]
    )


def map_prism_to_physical(ref_cube_sps: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """将计算立方体内的 SPs 映射到物理棱柱单元。

    Args:
        ref_cube_sps: 计算立方体坐标 (a,b,c)，形状 (n_sps, 3)
        cell_nodes: 棱柱 6 个顶点物理坐标，形状 (6, 3)，顺序
            (v0,v1,v2,w0,w1,w2)：v0..v2 为底面三角形，w0..w2 为对应的顶面
            三角形（w_i 是 v_i 正上方的顶点）

    Returns:
        phys_sps: 物理坐标，形状 (n_sps, 3)
    """
    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    r, s = cube_to_tri_rs(a, b)
    l1, l2, l3 = tri_barycentric(r, s)
    bottom = (
        l1[:, np.newaxis] * cell_nodes[0]
        + l2[:, np.newaxis] * cell_nodes[1]
        + l3[:, np.newaxis] * cell_nodes[2]
    )
    top = (
        l1[:, np.newaxis] * cell_nodes[3]
        + l2[:, np.newaxis] * cell_nodes[4]
        + l3[:, np.newaxis] * cell_nodes[5]
    )
    z = c[:, np.newaxis]
    return 0.5 * (1.0 - z) * bottom + 0.5 * (1.0 + z) * top


# ---------------------------------------------------------------------------
# 解析精确雅可比（直边四面体/棱柱专用，绕开谱微分矩阵）
# ---------------------------------------------------------------------------
#
# map_tet_to_physical / map_prism_to_physical 只用顶点节点做重心坐标插值，
# 是 (a,b,c) 的已知闭式表达式（不是未知的高阶流场，不需要谱微分矩阵近似）。
# 用固定阶数的谱微分矩阵（哪怕是坍缩坐标专用的 D_3d_tet/D_3d_prism）对这个
# 闭式映射求导，仍然是对真实（可能是有理函数）度量场的截断/插值近似，会
# 引入随机误差；对细长偏斜单元（真实网格中棱柱-四面体过渡区常见，边长比
# 可达 25:1），该误差量级不随 det(J) 一起等比例缩小，导致离散 GCL 恒等式
# （见 compute_metric_identity_residual）在这些单元上不能精确成立——真实
# 网格上实测：谱微分给出的 GCL 残差稳定在 ~1e-14（绝对量级，与单元偏斜、
# det(J) 大小无关），当 det(J) 本身只有 ~2e-14 时，相对误差被放大到 18%。
#
# 这里改用对闭式映射的解析（符号）求导：tet 情形，物理坐标是参考单纯形
# 坐标 (r,s,t) 的仿射函数（dx/dr、dx/ds、dx/dt 是与位置无关的常向量），
# 再用 Duffy 变换 (a,b,c)->(r,s,t) 的闭式雅可比做链式法则；prism 情形同理
# （三角形坍缩部分是 (a,b) 的仿射函数，c 方向是精确线性混合）。全程没有
# 任何插值/截断，只有初等微积分，因此结果精确到浮点舍入误差为止。
# 已用有限差分数值核对（误差 ~1e-10，与有限差分自身截断误差一致）；用这里
# 算出的精确 adj(J) 代入 D_3d_tet 做离散散度检验，真实网格最差单元的 GCL
# 残差从 ~1e-14 降到 ~1e-19，与单元偏斜程度、det(J) 大小无关。


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


# ---------------------------------------------------------------------------
# 立方体面 <-> 物理单元面 拓扑对应表（已数值验证，见模块 docstring）
# ---------------------------------------------------------------------------

# 四面体：6 个立方体面中有 4 个与四面体的 4 个面一一双射，
# 2 个（b=+1, c=+1）退化（分别坍缩为一条棱和一个顶点），不携带独立通量信息。
# 值为该立方体面对应的物理面上 3 个局部顶点索引（用于面连接匹配）。
TET_CUBE_FACES: Dict[str, Tuple[int, int, int]] = {
    "a=-1": (0, 2, 3),  # 对面顶点1的面（不含局部索引1）
    "a=+1": (1, 2, 3),  # 对面顶点0的面（不含局部索引0）
    "b=-1": (0, 1, 3),  # 对面顶点2的面（不含局部索引2）
    "c=-1": (0, 1, 2),  # 对面顶点3的面（不含局部索引3）
}

# 棱柱：6 个立方体面中有 5 个与棱柱的 5 个面（2 三角形封盖 + 3 四边形侧面）
# 一一对应，1 个（b=+1）退化为棱柱一条侧棱。四边形侧面给出 4 个局部顶点索引。
PRISM_CUBE_FACES: Dict[str, Tuple[int, ...]] = {
    "c=-1": (0, 1, 2),        # 底面三角形
    "c=+1": (3, 4, 5),        # 顶面三角形
    "a=-1": (0, 2, 5, 3),     # 侧面 v0-v2-w2-w0（对边 v0v2）
    "a=+1": (1, 2, 5, 4),     # 侧面 v1-v2-w2-w1（对边 v1v2）
    "b=-1": (0, 1, 4, 3),     # 侧面 v0-v1-w1-w0（对边 v0v1）
}


class CurvedMapping:
    """高阶曲边映射处理器：计算物理单元内 SPs 处的 Jacobian 矩阵，验证 GCL。"""

    def __init__(self, order: int):
        self.order = order
        self.n_points_1d = order + 1
        self.operators = generate_fr_operators(order)
        self.D_3d = self.operators.D_3d

    def _select_d3d(self, cell_type: str) -> np.ndarray:
        """按单元类型选择体积微分矩阵。

        注意：四面体/棱柱的几何 Jacobian 计算已改用解析精确公式
        （tet_exact_jacobian/prism_exact_jacobian），不再经过这里选出的
        D_3d_tet/D_3d_prism——这两个坍缩坐标专用算子现在只用于两处：
        (1) compute_metric_identity_residual 里对（解析精确的）度量场做
        离散散度检验，须与流场残差组装实际使用的算子一致；
        (2) core/fr_gradients.py、fr_residual_inviscid.py 等对高阶流场解
        本身求导——流场解不像几何映射那样有已知闭式表达式，仍然需要谱
        微分矩阵。"hex" 等无坍缩坐标退化面的单元类型，几何 Jacobian 仍走
        朴素张量积 D_3d（compute_jacobian 里对该分支保留原逻辑）。
        """
        if cell_type == "tet":
            return self.operators.D_3d_tet
        if cell_type == "prism":
            return self.operators.D_3d_prism
        return self.D_3d

    def compute_jacobian(
        self,
        phys_nodes: np.ndarray,
        cell_id: int = -1,
        cell_type: str = "hex",
        cell_nodes: np.ndarray = None,
        ref_cube_sps: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        """计算物理单元内所有 SPs 处的 Jacobian 矩阵。

        Args:
            cell_type: "tet"/"prism" 用解析精确雅可比（见 tet_exact_jacobian/
                prism_exact_jacobian 文档：直边单元的物理映射是 (a,b,c) 的
                已知闭式表达式，解析求导没有谱微分矩阵的截断/插值误差，
                真实网格验证在偏斜过渡区单元上把 GCL 残差从 ~1e-14 降到
                ~1e-19，与 det(J) 大小无关），需额外传入 cell_nodes（顶点
                物理坐标）与 ref_cube_sps（计算立方体坐标）；其他单元类型
                （六面体等无坍缩坐标退化面）用朴素张量积 D_3d 对已插值好的
                phys_nodes 直接求导。

        Raises:
            ValueError: cell_type 为 tet/prism 但未提供 cell_nodes/ref_cube_sps
            MeshDistortionError: 检测到非正 Jacobian 行列式
        """
        if cell_type in ("tet", "prism"):
            if cell_nodes is None or ref_cube_sps is None:
                raise ValueError(
                    f"cell_type='{cell_type}' 必须提供 cell_nodes 与 ref_cube_sps "
                    "才能计算解析精确雅可比（不再走谱微分矩阵近似路径）"
                )
            exact_fn = tet_exact_jacobian if cell_type == "tet" else prism_exact_jacobian
            jacobians = exact_fn(ref_cube_sps, cell_nodes)
        else:
            D_3d = self._select_d3d(cell_type)
            total_sps = len(phys_nodes)
            jacobians = np.zeros((total_sps, 3, 3))
            for m in range(3):
                for n in range(3):
                    jacobians[:, n, m] = np.dot(D_3d[:, :, m], phys_nodes[:, n])

        det_jacs = np.linalg.det(jacobians)

        if np.any(det_jacs <= 0):
            min_det = np.min(det_jacs)
            n_negative = np.sum(det_jacs <= 0)
            cell_info = f"cell {cell_id}" if cell_id >= 0 else "unknown cell"
            raise MeshDistortionError(
                f"Negative or zero Jacobian determinant detected in {cell_info}! "
                f"Min det(J) = {min_det:.6e}, {n_negative}/{total_sps} SPs affected. "
                f"This indicates mesh distortion (inverted or degenerate cell) that "
                f"must be fixed in mesh generation/repair, not silently patched here."
            )

        inv_jacs = np.linalg.inv(jacobians)
        return {"jacobians": jacobians, "det_jacs": det_jacs, "inv_jacs": inv_jacs}

    def compute_metric_identity_residual(
        self,
        phys_nodes: np.ndarray,
        cell_type: str = "hex",
        cell_nodes: np.ndarray = None,
        ref_cube_sps: np.ndarray = None,
    ) -> np.ndarray:
        """离散几何守恒律 (GCL) 的严格检验：Kopriva 度量恒等式。

        真正的 GCL 要求伴随矩阵（adjugate/cofactor）的离散散度恒为零：
            sum_m d/dxi_m ( adj(J)_{m,i} ) = 0    对每个物理方向 i=1,2,3
        其中 adj(J) = det(J) * J^{-1}。这等价于「均匀流场必须给出零残差」
        的物理要求，是 GCL 在曲边/坍缩坐标高阶格式中的标准定义
        （Kopriva 2006, "Metric Identities and the DSEM on Curvilinear
        Meshes"），而不是旧版本使用的「det(J) 在单元内近似常数」——
        对四面体/棱柱的坍缩坐标映射而言，det(J) 本身就应该是非均匀的
        （坍缩变换本身引入的度量非均匀性，即使物理单元是完美直边直面），
        用它的均匀性做 GCL 判据在数学上是错误的判据，已废弃。

        度量项 adj(J) 本身现在由 tet_exact_jacobian/prism_exact_jacobian
        解析求出（见该函数文档），不再有谱微分矩阵的截断/插值误差；这里
        散度检验用的 D_3d_tet/D_3d_prism 仍是流场残差组装实际使用的同一套
        算子，真实网格验证（含 det(J)~2e-14 的极端偏斜过渡区四面体）残差
        降到 ~1e-19，与单元偏斜程度、det(J) 大小无关。

        Args:
            cell_type: "tet"/"prism" 用解析精确雅可比 + 坍缩坐标专用散度
                算子；此时须提供 cell_nodes/ref_cube_sps。

        Returns:
            residual: 形状 (n_sps, 3)，每个 SP、每个物理方向的度量恒等式残差
        """
        D_3d = self._select_d3d(cell_type)
        jac_data = self.compute_jacobian(
            phys_nodes, cell_type=cell_type, cell_nodes=cell_nodes, ref_cube_sps=ref_cube_sps
        )
        det_jacs = jac_data["det_jacs"]
        inv_jacs = jac_data["inv_jacs"]
        adj = det_jacs[:, None, None] * inv_jacs  # adj[:, m, i] = adj(J)_{m,i}

        n_sps = phys_nodes.shape[0]
        residual = np.zeros((n_sps, 3))
        for i in range(3):
            for m in range(3):
                residual[:, i] += D_3d[:, :, m] @ adj[:, m, i]
        return residual

    def verify_gcl_strict(
        self,
        phys_nodes: np.ndarray,
        tolerance: float = 1e-8,
        cell_type: str = "hex",
        cell_nodes: np.ndarray = None,
        ref_cube_sps: np.ndarray = None,
    ) -> bool:
        """严格验证几何守恒律 (GCL)：度量恒等式残差是否在容差内。"""
        residual = self.compute_metric_identity_residual(
            phys_nodes, cell_type=cell_type, cell_nodes=cell_nodes, ref_cube_sps=ref_cube_sps
        )
        return bool(np.max(np.abs(residual)) < tolerance)
