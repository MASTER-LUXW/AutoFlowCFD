"""AutoFlowCFD V2.0 - native 四面体体积项去混叠（过积分）算子。

去混叠的动机与基无关：欧拉通量×度量项的非线性乘积真实多项式次数远高于求解阶数，直接在 coarse SPs 上微分
等价于先混叠再求导（实测 P2 残差可以是真值的 43~62 倍）。三件套（`native_tet/basis.py` 的受限 PKD 模态 +
Warp&Blend 节点）：
1. `interp_c2f`：**COARSE 阶数**模态基在 FINE 点上取值——Q 本身次数
   <=order，这一步是精确插值，不引入混叠。
2. `D_fine`：**FINE 阶数**原生单纯形微分矩阵（直接复用
   `build_native_tet_operators(over_order)`，不是新公式）。
3. `restrict_f2c`：**FINE 阶数**模态基在 COARSE 点上取值——把微分后的场从细网格
   插值回 coarse SPs（湍流 k-omega 输运用）；平均流用 L2 投影
   `build_native_tet_l2_projection`，见下节。

## 限制：平均流投影、湍流插值（2026-10-01）

旧文档写的理由是"过积分要保留非线性通量的高阶内容，不能截断"。这条理由不成立：残差
只存在于解点上，取值本身就把细层散度压成了 p 次插值——只是用插值而不是投影来压。两者
对精度同阶，但**离散守恒只有投影满足**：全域积分 `Σ_s w_s J_s R_s` 等于解点求积作用在
限制结果上，投影保持参考单元积分（常数在求解空间内），插值对高于 p 次的散度不精确。
封闭对称盒子（光滑非均匀场，P1/P2/P3）的全域质量残差相对量：

    插值  -3.0e-6 / 4.8e-9 / -4.6e-8      投影  -2.8e-13 / 1.4e-13 / 2.4e-13

（密度取常数时插值也到 1e-13：ρu 只有 p 次，散度在求解空间内，两者重合——误差确实来自
2p 次的 ρ·u。）能量还要求修正项用细层通量插值的迹，见 `fr/face_flux_trace.py`。

**湍流 k-omega 输运保留插值**：它按非守恒的标量形式离散（未知量 k 与 ln omega，
修正项本侧通量取点上值），守恒对它无意义；而把它的体积限制也换成投影、修正项仍取点上值
这个组合会失稳——两端远场的槽道 SST P3（`tests/unit/test_implicit_sst_nk.py`）实测 140 步
k 最小 -469 k_inf、残差不降；湍流单独换回插值后 106 步收敛（全部插值 91 步）。

## 过积分阶数

阶数见 `fr/overintegration_order.py::overintegration_order`：`rule*order`（默认 2x）是**与基无关**的去混叠
经验法则（二次非线性的标准做法），实测误差在 `oo = 2*order` 处断崖式下降，低于它基本拿不到收益（P3: oo=5
3.37e-3 vs oo=6 4.80e-6）。不设条件数上限：native 在 over_order=6 才 cond=3856、`max|D|` 从 3 到 6 只长 3.5 倍
（`tests/unit/test_native_tet_overintegration_conditioning.py`）。P1/P2/P3 都取到理想的 `2*order`（2/4/6）。

四面体段不需要更宽的细点度量数组：直边四面体的细点度量是逐单元常数的原样广播，取第 0 列广播到本段自己的
`n_fine_tet` 即可（见 `core/fr_operators/volume_contract.get_overintegration_context`）。高阶 `D_fine` 对常数的
零化残余随阶数增长，由 `fr/diff_matrix_consistency.enforce_constant_annihilation` 消掉（真实网格实测见
`tests/unit/test_tet_overintegration_cap_raised.py`）。
"""

from typing import Tuple

import numpy as np

from .basis import (
    build_native_tet_operators,
    restricted_tet_modes,
    simplex3d_value,
    rst_to_abc,
)


def _native_modal_vandermonde(ref_rst: np.ndarray, modes) -> np.ndarray:
    """给定一批参考坐标 (r,s,t) 和一组 (i,j,k) 模态索引，返回
    Vandermonde 矩阵 (n_pts, n_modes)——`build_native_tet_lift`/
    `build_native_tet_boundary_extrap` 已经在各自函数体内重复写过
    这个三行代码，这里单独提出来给过积分算子复用，避免第四份重复。
    """
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    return np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])


def build_native_tet_overintegration_operators(
    order: int, over_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造 native 四面体体积项去混叠三件套，与
    消费方是 `fr_residual/inviscid.py` 的过积分分支。

    Args:
        order: 当前求解阶数 P（coarse）
        over_order: 过积分阶数（`fr/overintegration_order.py::overintegration_order`，见模块文档）

    Returns:
        (ref_rst_fine, interp_c2f, D_fine, restrict_f2c)：
        `ref_rst_fine` 形状 (n_fine,3)；`interp_c2f` 形状
        (n_fine,n_coarse)；`D_fine` 形状 (n_fine,n_fine,3)；
        `restrict_f2c` 形状 (n_coarse,n_fine)——与原生棱柱同一形状约定，`inviscid.py` 的过积分分支按同一套
        `contract_shared_operator_1axis/2axis` 消费。
    """
    from scipy.linalg import lu_factor, lu_solve

    ref_coarse, _ = build_native_tet_operators(order)
    ref_fine, D_fine = build_native_tet_operators(over_order)

    modes_coarse = restricted_tet_modes(order)
    modes_fine = restricted_tet_modes(over_order)

    V_coarse_sps = _native_modal_vandermonde(ref_coarse, modes_coarse)
    V_coarse_at_fine = _native_modal_vandermonde(ref_fine, modes_coarse)
    V_fine_at_fine = _native_modal_vandermonde(ref_fine, modes_fine)

    lu_coarse = lu_factor(V_coarse_sps.T)
    interp_c2f = lu_solve(lu_coarse, V_coarse_at_fine.T).T

    restrict_f2c = lu_solve(lu_factor(V_fine_at_fine.T),
                            _native_modal_vandermonde(ref_coarse, modes_fine).T).T

    return ref_fine, interp_c2f, D_fine, restrict_f2c


def build_native_tet_l2_projection(order: int, over_order: int) -> np.ndarray:
    """细层（over_order 次）节点值 -> 求解空间（order 次）节点值的参考单元 L2 投影
    `M_c^{-1} ∫ φ_i g`，`(n_coarse, n_fine)`（平均流体积项用，见模块文档）。

    两套节点 Lagrange 基在 Duffy 求积点上取值，模态直接在坍缩坐标 (a,b,c) 上求值。
    被积函数（粗 x 细）至多 order+over_order 次，Duffy 因子再加 3 次，
    n = order+over_order+3 点精确到 2n-1 次，有富余。
    """
    from scipy.linalg import lu_factor, lu_solve

    ref_coarse, _ = build_native_tet_operators(order)
    ref_fine, _ = build_native_tet_operators(over_order)
    modes_coarse, modes_fine = restricted_tet_modes(order), restricted_tet_modes(over_order)
    abc, wq = _tet_duffy_rule(order + over_order + 3)

    def modal(modes):
        return np.column_stack([simplex3d_value(abc[0], abc[1], abc[2], i, j, k) for (i, j, k) in modes])

    L_coarse = lu_solve(lu_factor(_native_modal_vandermonde(ref_coarse, modes_coarse).T), modal(modes_coarse).T).T
    L_fine = lu_solve(lu_factor(_native_modal_vandermonde(ref_fine, modes_fine).T), modal(modes_fine).T).T
    return np.linalg.solve(L_coarse.T @ (wq[:, None] * L_coarse), L_coarse.T @ (wq[:, None] * L_fine))


def _tet_duffy_rule(n: int):
    """参考四面体上的 Duffy 张量 Gauss 求积 `((a,b,c) 坍缩坐标, 权重)`，各 `(n^3,)`。"""
    from ..quadrature_points import gauss_legendre

    x, w = gauss_legendre(n)
    a, b, c = (q.ravel() for q in np.meshgrid(x, x, x, indexing="ij"))
    wa, wb, wc = (q.ravel() for q in np.meshgrid(w, w, w, indexing="ij"))
    return (a, b, c), wa * wb * wc * (1.0 - b) / 2.0 * ((1.0 - c) / 2.0) ** 2


def native_tet_n_fine(over_order: int) -> int:
    """native 四面体在 `over_order` 下的**真实**细点数。"""
    oo = int(over_order)
    return (oo + 1) * (oo + 2) * (oo + 3) // 6
