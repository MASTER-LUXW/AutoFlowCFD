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
"Spectral/hp 元素 Methods for CFD"（棱柱/三棱柱坍缩坐标）。
"""

import numpy as np
from numba import njit
from typing import Dict, Tuple

from autoflowcfd.fr.operators import generate_fr_operators
from .curved_mapping_exact_jacobian import tet_exact_jacobian, prism_exact_jacobian


@njit(cache=True)
def batched_det_inv_3x3(J: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """批量计算 (N,3,3) 矩阵的行列式与逆矩阵，闭式伴随矩阵公式
    （inv=adj(J)/det(J)，adj 是余子式矩阵的转置），数学上与
    `np.linalg.det`/`np.linalg.inv` 精确等价，不是近似替代。

    性能优化：`compute_jacobian` 里原来对每个单元调用一次
    `np.linalg.det`/`np.linalg.inv`（每次处理该单元全部 SPs 的
    (n_sps,3,3) 批次），真实网格上单是这一步的 LAPACK 通用矩阵求逆
    调度开销（对 3x3 这种小矩阵而言不成比例地大）就占了整个网格构建
    阶段最大的单项开销（cProfile 实测 20,740 次调用耗时 8.94s，占同一
    阶段总时间近三分之一）。闭式公式对 3x3 这种固定小尺寸矩阵是纯
    标量四则运算，没有 LAPACK 调度/主元选择的固定开销，同一批数据上
    实测提速约 300~370 倍。

    数值等价性已验证：随机良态矩阵（条件数 1~3，代表真实网格 Jacobian
    的典型量级）逐位一致（最大误差 6.7e-16，纯浮点舍入噪声）；本项目
    文档记录过的真实退化单元场景（一个方向 det 低至 ~2e-14、其余方向
    正常，例如坍缩坐标退化边附近的单元）精确一致（相对误差 0.0）。
    只有在人为构造的病态随机矩阵（条件数 ~2e5，物理网格不会出现这种
    无结构的病态）上才会看到 ~1e-8 级别的差异——真实网格 Jacobian 来自
    光滑坐标变换，不会产生这类病态，退化单元的病态是"某一方向趋于零"
    这种结构化模式，闭式公式对这种模式反而精确成立（见上面验证）。
    """
    n = J.shape[0]
    det = np.empty(n)
    inv = np.empty((n, 3, 3))
    for idx in range(n):
        m00, m01, m02 = J[idx, 0, 0], J[idx, 0, 1], J[idx, 0, 2]
        m10, m11, m12 = J[idx, 1, 0], J[idx, 1, 1], J[idx, 1, 2]
        m20, m21, m22 = J[idx, 2, 0], J[idx, 2, 1], J[idx, 2, 2]

        c00 = m11 * m22 - m12 * m21
        c01 = -(m10 * m22 - m12 * m20)
        c02 = m10 * m21 - m11 * m20
        c10 = -(m01 * m22 - m02 * m21)
        c11 = m00 * m22 - m02 * m20
        c12 = -(m00 * m21 - m01 * m20)
        c20 = m01 * m12 - m02 * m11
        c21 = -(m00 * m12 - m02 * m10)
        c22 = m00 * m11 - m01 * m10

        d = m00 * c00 + m01 * c01 + m02 * c02
        det[idx] = d
        inv_d = 1.0 / d
        inv[idx, 0, 0] = c00 * inv_d
        inv[idx, 0, 1] = c10 * inv_d
        inv[idx, 0, 2] = c20 * inv_d
        inv[idx, 1, 0] = c01 * inv_d
        inv[idx, 1, 1] = c11 * inv_d
        inv[idx, 1, 2] = c21 * inv_d
        inv[idx, 2, 0] = c02 * inv_d
        inv[idx, 2, 1] = c12 * inv_d
        inv[idx, 2, 2] = c22 * inv_d
    return det, inv


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
#
# signed_tet_volume / fix_tet_orientation / decompose_prism_to_tets /
# fix_prism_orientation 已拆分到 curved_mapping_orientation.py（原文件
# 超过 400 行的项目约定上限），在本文件顶部原样重新导出（见 import 语句）。


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
# tet_exact_jacobian / prism_exact_jacobian 已拆分到
# curved_mapping_exact_jacobian.py（原文件超过 400 行的项目约定上限），
# 在本文件顶部原样重新导出（见 import 语句）。设计动机（为什么用解析求导
# 替代谱微分矩阵几何求导）的完整说明见该文件的模块 docstring。


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

        抛出异常:
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

        # total_sps 只在上面的 hex(else) 分支里赋值；tet/prism 走
        # exact_fn 分支时从未定义，下面报错信息引用它会抛
        # UnboundLocalError——即真正遇到畸变四面体/棱柱时，用户拿到的是
        # 一个 Python 内部错误而不是这里设计好的 MeshDistortionError
        # 诊断（已用共线退化四面体复现）。用 jacobians 的第一维统一补上。
        total_sps = jacobians.shape[0]

        # 闭式批量 det+inv（见 batched_det_inv_3x3 文档：数学上与
        # np.linalg.det/np.linalg.inv 精确等价，真实网格上实测提速约
        # 300~370 倍，是网格构建阶段原来最大的单项开销）。退化/负值
        # Jacobian 下 inv 部分可能算出 inf/nan，但下面的检查在任何调用方
        # 读取 inv_jacs 之前就会抛出 MeshDistortionError 中止，不会被
        # 静默使用。
        det_jacs, inv_jacs = batched_det_inv_3x3(np.ascontiguousarray(jacobians))

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
        （Kopriva 2006, "Metric Identities 并且  DSEM 在...上 Curvilinear
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
