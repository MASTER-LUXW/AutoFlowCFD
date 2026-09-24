"""AutoFlowCFD V2.0 - Persson-Peraire 传感器的**算子构造与缓存**。

从 `artificial_viscosity.py`（619 行）拆出（2026-09-20，项目"单文件不超
500 行"规范）。纯搬家，逻辑未改；三条基各自为什么必须用自己的模态族与
内积形式，见各构造函数的文档。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from autoflowcfd.fr.quadrature_points import gauss_legendre
from autoflowcfd.core.utils.array_module import array_module as _array_module


# Persson-Peraire 传感器分段过渡宽度（对数尺度），沿用 mirgecom 的默认值。
SENSOR_KAPPA = 1.0

# 传感器计算/人工粘性上限所用的默认变量选择：密度（索引0）——密度处处
# 为正、不像压力/速度那样可能因坐标系/驻点而变号或过零，是最鲁棒的
# 光滑性探测变量，也是 Persson-Peraire 原始论文与多数后续实现的默认选择。
DEFAULT_SENSOR_VAR_INDEX = 0





_native_tet_sensor_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
_native_prism_sensor_cache: Dict[int, Tuple] = {}

#: 传感器算子的**设备侧**副本缓存，键是 (主机缓存键, 数组模块名)。
#: 算子只依赖 (cell_type, order)，与流场状态无关，所以搬上设备一次即可；
#: 不缓存的话每个 RK stage 都会重新 H2D 拷一遍（小矩阵，但是同步点）。
_sensor_operator_xp_cache: Dict[Tuple, Tuple] = {}


def _operators_on(xp, key, host_arrays):
    """把主机侧算子元组搬到 `xp` 所在的设备并缓存。

    `xp is np` 时原样返回——CPU 路径不多一次拷贝，因此改造前后逐位相同。
    """
    if xp is np:
        return host_arrays
    ck = (key, xp.__name__)
    hit = _sensor_operator_xp_cache.get(ck)
    if hit is None:
        hit = tuple(xp.asarray(a) for a in host_arrays)
        _sensor_operator_xp_cache[ck] = hit
    return hit


def _build_native_tet_sensor_operators(order: int):
    """构造/缓存 native 四面体传感器算子：(V_inv, top_mask, n_native)。

    **真实 bug 修复（2026-09-15）**：`compute_persson_peraire_sensor` 的
    `cell_type="tet"` 分支一直用 `collapsed_basis.py::tet_modal_basis_and_grad`
    ——那是**坍缩坐标**族（`(order+1)^3` 个模态、各方向独立取阶数），而
    坍缩坐标四面体基已于 2026-09-03 整体删除，四面体现在唯一的实现是
    native PKD/Dubiner 基（`i+j+k<=order`，order=1 时只有 4 个真实自由度，
    零填充到全局 n_sps=8 宽度）。于是传感器对四面体单元做的是：

      1. 用一个**不是解所在空间**的基去做模态分解；
      2. 把填充 SP 当成真实自由度一起喂进指标——而填充槽位按
         `native_padding.py` 的约定是"初始化时复制真实 SP #0"、
         之后残差行填零、滤波行是单位阵，**永远冻结在初值**。实测
         推进 10 步后填充块与真实 SP#0 已相差 3.4%，也就是说指标里
         混进了一个纯人造的阶跃。

    这处缺陷是 2026-09-03 删除坍缩基那轮审计的漏项（同轮在
    `fr_coefficients.py` 等处已抓到 3 处同类问题）。人工粘性默认关闭
    （`artificial_viscosity_enabled=False`，只能由 `--artificial-viscosity`
    开启），所以它此前没有在默认路径上造成影响。

    修法：四面体走**native 专属**的 Vandermonde。
    `simplex3d_value` 是 Hesthaven & Warburton `Simplex3DP.m` 的移植，
    是参考四面体上的**正交归一**基（`2*sqrt(2)` 前因子正是归一化常数），
    因此 L2 内积**精确等于**模态系数的平方和——不需要像棱柱那条分支
    那样引入求积权重（那边节点与 Gauss-Legendre 求积点重合，用张量积
    权重是精确的；native 节点是 Warp & Blend 节点、不是求积点，用
    GL 权重才是错的）。两条分支的内积算法因此**刻意不同**，都是各自
    精确的形式。

    "最高阶模态"的判据也必须换：单纯形空间里模态的次数是 `i+j+k`，
    不是张量积族的 `max(i,j,k)`。

    Returns:
        (V_inv, top_mask, n_native)：`V_inv` 形状 (n_native,n_native)，
        `top_mask` 形状 (n_native,) 标出 `i+j+k==order` 的模态，
        `n_native = (order+1)(order+2)(order+3)/6`。
    """
    if order in _native_tet_sensor_cache:
        return _native_tet_sensor_cache[order]

    from autoflowcfd.fr.native_tet.basis import (
        build_native_tet_operators, restricted_tet_modes, rst_to_abc,
        simplex3d_value,
    )

    ref_rst, _ = build_native_tet_operators(order)
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    modes = restricted_tet_modes(order)
    n_native = len(modes)
    if ref_rst.shape[0] != n_native:
        raise AssertionError(
            f"native 四面体节点数 {ref_rst.shape[0]} 与模态数 {n_native} 不符——"
            f"Vandermonde 不是方阵，无法求逆")

    V = np.empty((n_native, n_native))
    top_mask = np.zeros(n_native, dtype=bool)
    for m, (i, j, k) in enumerate(modes):
        V[:, m] = simplex3d_value(a, b, c, i, j, k)
        top_mask[m] = (i + j + k) == order

    result = (np.linalg.inv(V), top_mask, n_native)
    _native_tet_sensor_cache[order] = result
    return result


def _build_native_prism_sensor_operators(order: int):
    """构造/缓存 native 棱柱传感器算子：`(V_inv, mass_diag, top_mask, n_native)`。

    **与四面体同一类的真实缺陷（2026-09-20 修复）**：`compute_persson_
    peraire_sensor` 的 `cell_type="prism"` 分支用的是
    `collapsed_basis.py::prism_modal_basis_and_grad` —— 坍缩坐标族的
    `(order+1)^3` 个模态、且把 `(order+1)^3` 个张量积 Gauss 点当成解点。
    原生棱柱基（`AFCFD_PRISM_BASIS=native`，2026-09-20 起是默认）下解是
    `(p+1)^2(p+2)/2` 维的 PKD(三角形)⊗Legendre(挤出) 空间、解点是
    Warp&Blend⊗Gauss 点、其余槽位是**冻结在初值**的零填充。继续用坍缩
    分支等于：(a) 用一个不是解所在空间的基做模态分解；(b) 把冻结的填充
    槽位当成自由度喂进指标。与 2026-09-15 修掉的四面体那处是同一个
    错误类型（那次的完整说明见 `_build_native_tet_sensor_operators`）。

    ## 内积形式：对角质量矩阵，不是节点求积权重

    原生棱柱基在参考棱柱上**正交但不归一**（实测非对角 <6e-15，对角
    在 P1 上是 8/2.667/4/1.333/5.333/1.778 这样一组各不相同的数）。所以
    L2 能量是 `Σ_m M_mm * chat_m^2`，其中 `M_mm = ∫ φ_m^2`。

    * 四面体那条分支能直接用 `Σ chat^2` 是因为 PKD 基**正交归一**；
    * 坍缩那条分支能用张量积 GL 权重是因为解点与求积点重合；
    * 原生棱柱两条都不成立，必须显式带上对角质量。

    `M_mm` 用 Duffy 变换后的张量积 Gauss 求积算（被积函数是多项式，
    点数取 `2*order+4` 时是精确的，不是近似），每个阶数只算一次。

    ## "最高阶模态"的判据

    原生棱柱空间是"三角形次数 <= order"⊗"挤出次数 <= order"的张量积，
    所以最高阶模态的判据是 `max(三角形次数, 挤出次数) == order` ——
    与坍缩族的 `max(i,j,k)==order` 同构，而不是单纯形族的 `i+j+k`。

    Returns:
        `(V_inv, mass_diag, top_mask, n_native)`，前三个形状分别是
        `(n_native,n_native)`、`(n_native,)`、`(n_native,)`。
    """
    if order in _native_prism_sensor_cache:
        return _native_prism_sensor_cache[order]

    from autoflowcfd.fr.native_prism.basis import (
        build_native_prism_nodes, build_native_prism_vandermonde,
        restricted_prism_modes,
    )

    nodes = build_native_prism_nodes(order)
    V, _, _, _ = build_native_prism_vandermonde(order, nodes)
    n_native = V.shape[0]
    if V.shape[0] != V.shape[1]:
        raise AssertionError(
            f"native 棱柱 Vandermonde 不是方阵 {V.shape} —— 节点数与模态数"
            f"必须相同才能求逆")

    # 对角质量 M_mm = ∫ φ_m^2（Duffy 变换后的张量积 Gauss，多项式精确）
    n_q = 2 * order + 4
    a_1d, w_a = gauss_legendre(n_q)
    b_1d, w_b = gauss_legendre(n_q)
    t_1d, w_t = gauss_legendre(n_q)
    a, b, t = np.meshgrid(a_1d, b_1d, t_1d, indexing="ij")
    wa, wb, wt = np.meshgrid(w_a, w_b, w_t, indexing="ij")
    a, b, t = a.ravel(), b.ravel(), t.ravel()
    weight = (wa * wb * wt).ravel() * ((1.0 - b) / 2.0)
    r = (1.0 + a) * (1.0 - b) / 2.0 - 1.0
    V_q, _, _, _ = build_native_prism_vandermonde(
        order, np.stack([r, b, t], axis=1))
    mass_diag = weight @ (V_q * V_q)

    # 最高阶模态掩码：模态排列是"三角形模态外层、挤出模态内层"
    # （见 `build_native_prism_vandermonde`），三角形模态的次数由
    # `restricted_prism_modes` 给出的 (i, j, k) 里的 i+j 决定。
    top_mask = np.zeros(n_native, dtype=bool)
    for m, (i, j, k) in enumerate(restricted_prism_modes(order)):
        top_mask[m] = max(i + j, k) == order

    result = (np.linalg.inv(V), mass_diag, top_mask, n_native)
    _native_prism_sensor_cache[order] = result
    return result
