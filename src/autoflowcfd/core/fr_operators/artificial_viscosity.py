"""
AutoFlowCFD V2.0 - Persson-Peraire 模态传感器 + 局部人工粘性（可选能力）

背景（见 ProjectFiles/V2.0/7_重大问题修复-求解稳定性.md 五、节）：本项目
现有的两层稳定性机制——模态滤波器（fr/modal_filter.py）与残差量级异常
检测（core/fr_operators/troubled_cell.py 机制3）——在 2026-08-29 的
cube_demo 真实网格调查中被证明存在结构性张力：模态滤波器对"中间"模态
的压制是真实的稳定性刚需，放松它会重新引入灾难性失稳；机制3无论是
局部（同单元）还是全局（全网格）参照，都无法可靠区分"合法的强局部
物理"（边界层、驻点、尖角）与"真正的欠分辨率伪影"，因为两者都表现为
"残差量级大"。

行业调研（同一份文档）：开源 FR 参照实现 PyFR 的稳定性工具箱第三层是
基于 Persson & Peraire (2006, "Sub-Cell Shock Capturing for Discontinuous
Galerkin Methods") 传感器的局部人工粘性——这个传感器看的不是残差量级，
而是**解本身的模态谱衰减速率**：把节点值变换到模态系数空间，比较"完整
表示"与"截断掉最高阶模态后的表示"之间的能量占比。光滑函数的最高阶
模态系数按 O(1/N^4) 衰减；如果实测占比明显超过这个理论预期，说明该
单元的解含有这个多项式阶数无法真正解析的高频内容（混叠/欠分辨率的
直接证据），与残差本身是大是小无关——边界层里残差可以很大但完全光滑
（多项式能精确表示），这种单元传感器不会触发；反之一个残差看起来不大
但解本身在振荡的单元，传感器会触发。

本模块是一个独立、默认关闭（opt-in）的新增能力，不修改任何现有稳定性
机制的行为，不影响任何未显式启用它的现有测试/求解路径。启用后，对触发
传感器的单元，把一个局部人工粘性系数叠加进现有粘性残差已经在消费的
`mu_t_field` 通道（core/fr_residual/viscous_flux.py 的 `mu_t_field`
参数）——复用已经过充分验证的 BR1 面耦合粘性通量组装机制，不新建一条
独立的扩散残差路径。

**原先的范围限制已补齐（2026-09-14）**：Persson & Peraire 原始方法对
*全部*守恒变量（含质量/连续性方程）叠加人工扩散项；本实现最初只通过
动量/能量方程既有的粘性应力/热传导通道施加，不直接扩散密度
（`viscous_physical_flux` 的质量分量 G[...,0] 恒为 0），当时把这一点
记作"许多实际 DG/FR 实现采用的简化"。用户明确指出本项目不接受简化，
现已补上缺的那一项：`FRSolver._artificial_mass_diffusion_residual`
把 `+div(epsilon * grad(rho))` 加进连续性方程。

实现选择（为什么不改粘性通量本身）：AV 默认关闭，没有理由为它给所有
运行的 `viscous_physical_flux_batch` 热路径增加参数与分支。而
`div(eps*grad(rho))` 本身就是一个标量扩散算子，直接复用湍流输运已经
验证过的 BR1 面耦合标量扩散装配
（`turbulence/transport.py::compute_scalar_diffusion_residual`，它返回的
就是 `+div(Gamma*grad(phi))`，与 dU/dt 的符号约定一致）。AV 关闭时这段
完全不执行，零开销。

性质：散度形式 -> 严格守恒；均匀流场下 grad(rho)=0 -> 该项恒为 0，
不破坏自由流场保持性。两条都有测试钉住
（tests/unit/test_artificial_viscosity_mass_diffusion.py）。

公式来源：Persson & Peraire (2006) 原始传感器公式 + mirgecom
（Illinois/DOE 现役生产级 DG 代码）文档给出的精确数值实现细节
（s_e=log10(S_e)，分段 sine 过渡，kappa 默认值 1.0）——本模块的具体
数值实现直接对照 mirgecom 文档核实过，不是凭记忆重新推导。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from autoflowcfd.fr.collapsed_basis import prism_modal_basis_and_grad, tet_modal_basis_and_grad
from autoflowcfd.fr.quadrature_points import gauss_legendre

# Persson-Peraire 传感器分段过渡宽度（对数尺度），沿用 mirgecom 的默认值。
SENSOR_KAPPA = 1.0

# 传感器计算/人工粘性上限所用的默认变量选择：密度（索引0）——密度处处
# 为正、不像压力/速度那样可能因坐标系/驻点而变号或过零，是最鲁棒的
# 光滑性探测变量，也是 Persson-Peraire 原始论文与多数后续实现的默认选择。
DEFAULT_SENSOR_VAR_INDEX = 0

_sensor_operator_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _build_sensor_operators(cell_type: str, order: int, ref_cube_sps: np.ndarray):
    """构造/缓存单个 (cell_type, order) 组合的传感器算子三件套：
    (V, V_inv, top_mode_mask_times_quad_weight_diag_equivalent)。

    实际缓存的是 (V, V_inv, quad_weights_3d, top_mask) 四件套，供
    compute_persson_peraire_sensor 每步复用，避免每次残差求值都重新做
    Vandermonde 矩阵求逆（(n_sps,n_sps) 规模，P2=27、P3=64，求逆本身
    不便宜，且与 (cell_type, order) 唯一对应、和流场状态无关，只需算
    一次）。

    order==0 没有"上一阶"可截断，调用方必须在此之前短路处理（返回
    全零传感器），这里不处理 order==0。
    """
    key = (cell_type, order)
    if key in _sensor_operator_cache:
        return _sensor_operator_cache[key]

    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    basis_fn = tet_modal_basis_and_grad if cell_type == "tet" else prism_modal_basis_and_grad
    V, _, _, _ = basis_fn(a, b, c, order)
    V_inv = np.linalg.inv(V)

    n1d = order + 1
    top_mask = np.zeros(n1d ** 3, dtype=bool)
    for i in range(n1d):
        for j in range(n1d):
            for k in range(n1d):
                if max(i, j, k) == order:
                    top_mask[i * n1d * n1d + j * n1d + k] = True

    sps_1d, w_1d = gauss_legendre(n1d)
    quad_weights_3d = np.einsum("i,j,k->ijk", w_1d, w_1d, w_1d).reshape(-1)

    result = (V, V_inv, quad_weights_3d, top_mask)
    _sensor_operator_cache[key] = result
    return result


_native_tet_sensor_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}


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
         `native_tet_padding.py` 的约定是"初始化时复制真实 SP #0"、
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

    from autoflowcfd.fr.native_simplex_basis import (
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


def compute_persson_peraire_sensor_native_tet(
    field_nodal: np.ndarray, order: int
) -> np.ndarray:
    """native 四面体专属的 Persson-Peraire 传感器，`s_e = log10(S_e)`。

    `S_e = sum_{i+j+k==order} chat^2 / sum_all chat^2`，其中 `chat` 是在
    正交归一 PKD/Dubiner 基下的模态系数——正交归一保证这个比值**精确
    等于**"只保留最高次模态的重构"与"完整重构"的 L2 能量比，不需要
    任何求积权重。理由与背景见 `_build_native_tet_sensor_operators`。

    Args:
        field_nodal: (n_cells, n_sps) 待探测的场。只使用**前 n_native 列**
            ——其余列是零填充槽位、按约定冻结在初值，不是自由度，
            混进来会引入纯人造的阶跃。
        order: 当前多项式阶数；order==0 时返回 -inf（无最高阶模态可截断）

    Returns:
        s_e: (n_cells,)
    """
    n_cells = field_nodal.shape[0]
    if order == 0:
        return np.full(n_cells, -np.inf)

    V_inv, top_mask, n_native = _build_native_tet_sensor_operators(order)
    if field_nodal.shape[1] < n_native:
        raise ValueError(
            f"field 每单元只有 {field_nodal.shape[1]} 个解点，少于 native "
            f"四面体 order={order} 所需的 {n_native} 个真实自由度")

    real = np.ascontiguousarray(field_nodal[:, :n_native])
    modal = np.einsum("ij,cj->ci", V_inv, real)            # (n_cells,n_native)
    energy_all = np.einsum("ci,ci->c", modal, modal)
    modal_top = np.where(top_mask[np.newaxis, :], modal, 0.0)
    energy_top = np.einsum("ci,ci->c", modal_top, modal_top)

    S_e = energy_top / np.maximum(energy_all, 1e-300)
    with np.errstate(divide="ignore"):
        return np.log10(np.maximum(S_e, 1e-300))


def compute_persson_peraire_sensor(
    field_nodal: np.ndarray, cell_type: str, order: int, ref_cube_sps: np.ndarray
) -> np.ndarray:
    """计算 Persson-Peraire 模态传感器 s_e = log10(S_e)。

    S_e = <u_trunc, u_trunc>_e / <u, u>_e，其中 u_trunc 是把 u 变换到
    模态系数空间、只保留 max(i,j,k)==order 的最高阶模态（其余模态清零）
    后再变换回节点值——即 u 与"截断掉最高阶模态的 u"之间的差恰好等于
    u_trunc 本身（线性算子的性质：完整重构 - 截断重构 = 只保留被截断
    那部分模态的重构），<.,.>_e 是用节点所在 Gauss-Legendre 求积点权重
    做的离散 L2 内积（张量积三维权重，逐元素与场值相乘再求和，不是
    矩阵乘法意义上的质量矩阵内积，但对配置在求积点上的节点表示，两者
    对多项式被积函数是精确等价的——求积点与解点重合正是配置法的定义）。

    Args:
        field_nodal: (n_cells, n_sps) 待探测的场（通常是密度）
        cell_type: "tet" 或 "prism"
        order: 当前多项式阶数
        ref_cube_sps: (n_sps, 3) 计算立方体参考坐标，与 fr/operators.py
            生成 D_3d_tet/D_3d_prism 用的完全一致

    Returns:
        s_e: (n_cells,) 传感器值（已取 log10），order==0 时返回 -inf
            填充的数组（representing "无穷光滑"，任何下游 kappa 判据
            都不会触发人工粘性，与 order==0 没有可截断的高阶模态这一
            事实一致）
    """
    n_cells = field_nodal.shape[0]
    if order == 0:
        return np.full(n_cells, -np.inf)

    V, V_inv, quad_weights_3d, top_mask = _build_sensor_operators(cell_type, order, ref_cube_sps)

    modal = np.einsum("ij,cj->ci", V_inv, field_nodal)
    modal_top = np.where(top_mask[np.newaxis, :], modal, 0.0)
    diff_nodal = np.einsum("ij,cj->ci", V, modal_top)

    num = np.einsum("cs,s,cs->c", diff_nodal, quad_weights_3d, diff_nodal)
    den = np.einsum("cs,s,cs->c", field_nodal, quad_weights_3d, field_nodal)

    S_e = num / np.maximum(den, 1e-300)
    with np.errstate(divide="ignore"):
        s_e = np.log10(np.maximum(S_e, 1e-300))
    return s_e


def compute_artificial_viscosity_ramp(s_e: np.ndarray, order: int, kappa: float = SENSOR_KAPPA) -> np.ndarray:
    """把传感器值 s_e 映射到 [0,1] 的人工粘性强度斜坡（mirgecom 公式）：

        epsilon = 0                                          if s_e < s0-kappa
        epsilon = 0.5*(1+sin(pi*(s_e-s0)/(2*kappa)))         if s0-kappa <= s_e <= s0+kappa
        epsilon = 1                                          if s_e > s0+kappa

    s0 = -4*log10(order)（光滑函数模态系数按 1/N^4 衰减的理论预期，
    Persson & Peraire 原始论文的判据），order<=0 时无意义（调用方保证
    不会以 order==0 调用本函数，见 compute_persson_peraire_sensor 的
    order==0 短路：s_e=-inf 恒小于任意有限 s0-kappa，ramp 天然为 0，
    这里不需要重复特判）。
    """
    s0 = -4.0 * np.log10(max(order, 1))
    ramp = np.zeros_like(s_e)
    mid = (s_e >= s0 - kappa) & (s_e <= s0 + kappa)
    high = s_e > s0 + kappa
    ramp[mid] = 0.5 * (1.0 + np.sin(np.pi * (s_e[mid] - s0) / (2.0 * kappa)))
    ramp[high] = 1.0
    return ramp


def compute_troubled_cell_mask(
    field_nodal: np.ndarray, order: int, *,
    n_prism: Optional[int] = None,
    cell_is_prism: Optional[np.ndarray] = None,
    kappa: float = SENSOR_KAPPA,
) -> np.ndarray:
    """逐单元判定"该单元的这个标量场欠分辨" —— **纯数组接口**。

    与 `compute_persson_peraire_artificial_viscosity` 用的是同一个传感器
    与同一条 ramp 判据（`compute_persson_peraire_sensor` +
    `compute_artificial_viscosity_ramp`）、同一组参考坐标，只是不乘任何
    人工粘性尺度、只返回布尔掩码，并且**不需要 solver 也不需要 ops**——
    只要 (场, order, 单元类型划分)。

    为什么需要这个接口（2026-09-15）：模态滤波器的传感器门控
    （`core/fr_solver/filter.py`）此前只能靠"把场临时塞进
    `solver.state.U` 再调用 solver 版传感器"来实现，那既是一个副作用
    hack，也让门控**只能在单机 CPU 推进循环里用**（CPU MPI / GPU 那些
    路径没有同构的 solver 对象）。改成纯数组接口后：

    - 平均流可以直接传守恒密度（Persson-Peraire 的 S_e 是能量比值、
      对场的整体缩放不变，所以守恒密度与原始密度给出同一个判据）；
    - **k/omega 同样可以用**——这是关键，`filter_scalar_field` 一直
      直接用 `ops.filter_prism`、完全不经过任何门控，于是湍流场在
      legacy/sensor 两档下都被清掉一整阶（即 k/omega 实际是 P0）。
      那是与平均流同一类的静默降阶，只是换了个场。

    参考坐标按 `gauss_legendre(order+1)` 的张量积重新构造，与
    `compute_persson_peraire_artificial_viscosity` 以及 fr/operators.py
    生成 `D_3d_*`/`filter_*` 用的完全是同一组（`FROperators` 不对外暴露
    它，见那边同一处注释）；棱柱与四面体共用这组坐标，只有模态基
    （`cell_type`）不同。

    Args:
        field_nodal: (n_cells, n_sps) 待探测的标量场
        order: 当前多项式阶数；order==0 时没有可截断的最高阶模态，
            直接返回全 False（与传感器 order==0 短路返回 -inf 一致）
        n_prism: 单机"棱柱在前"排列下的棱柱单元数。与 `cell_is_prism`
            **必须且只能给一个**。
        cell_is_prism: (n_cells,) 布尔数组，True=棱柱。分布式的 local
            排列里棱柱与四面体是**交错**的（见 core/fr_solver/filter.py::
            build_filter_func_by_cell_type 同一处理由），不能用 n_prism
            切片表达，必须走这条。
        kappa: ramp 过渡带宽度，默认与人工粘性同一个 SENSOR_KAPPA

    Returns:
        (n_cells,) 布尔掩码，True = 该单元欠分辨（ramp > 0）
    """
    if (n_prism is None) == (cell_is_prism is None):
        raise ValueError(
            "compute_troubled_cell_mask 需要 n_prism 与 cell_is_prism 中"
            "恰好一个：前者是单机'棱柱在前'排列，后者是分布式交错排列。"
            "同时给或都不给都是调用方对索引空间没有明确认知的信号。")

    n_cells = field_nodal.shape[0]
    mask = np.zeros(n_cells, dtype=np.bool_)
    if order == 0 or n_cells == 0:
        return mask

    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    ref_cube_sps = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    if n_prism is not None:
        groups = ((np.arange(0, n_prism), "prism"),
                  (np.arange(n_prism, n_cells), "tet"))
    else:
        cip = np.asarray(cell_is_prism, dtype=bool)
        if cip.shape != (n_cells,):
            raise ValueError(
                f"cell_is_prism 形状 {cip.shape} 与场的单元数 {n_cells} 不符")
        groups = ((np.flatnonzero(cip), "prism"),
                  (np.flatnonzero(~cip), "tet"))

    for sel, cell_type in groups:
        if sel.size == 0:
            continue
        # 四面体走 native 专属传感器（正交归一 PKD 基 + 只用真实自由度），
        # 棱柱走张量积族 + GL 求积权重——两条分支各自精确，理由见
        # `_build_native_tet_sensor_operators`。
        if cell_type == "tet":
            s_e = compute_persson_peraire_sensor_native_tet(
                np.ascontiguousarray(field_nodal[sel]), order)
        else:
            s_e = compute_persson_peraire_sensor(
                np.ascontiguousarray(field_nodal[sel]), cell_type, order, ref_cube_sps)
        mask[sel] = compute_artificial_viscosity_ramp(s_e, order, kappa) > 0.0
    return mask


def compute_persson_peraire_artificial_viscosity(
    solver, alpha_av: float = 1.0, sensor_var_index: int = DEFAULT_SENSOR_VAR_INDEX, kappa: float = SENSOR_KAPPA
) -> np.ndarray:
    """完整流水线：从 solver 当前状态算出每个单元的人工粘性系数
    （与 mu_molecular/mu_t_field 同一套动力粘度量纲，可直接叠加）。

    人工粘性上限（传感器 ramp=1 时的取值）取
    `alpha_av * rho_local * h_cell * vel_local / order`——量纲分析：
    [kg/m^3]*[m]*[m/s] = [kg/(m*s)] = [Pa*s]，与 mu_molecular 同量纲；
    `rho_local`/`vel_local` 用该单元 SPs 的平均值（局部真实流动尺度，
    不依赖自由来流，驻点/尾迹等偏离自由来流较远的区域也能给出合理
    尺度）；`h_cell = cell_volume^(1/3)`（已有的 `mesh.cell_volumes`，
    不新增几何计算）；除以 order 是标准做法（阶数越高，本就该需要
    越少的人工耗散，见 fr/modal_filter.py 同一套"阶数越高、高频内容
    占比越小"的设计前提）。`alpha_av` 是唯一的自由标定常数（mirgecom
    文档同样把它留给调用方，见模块文档"公式来源"一节）。

    Args:
        solver: FRSolver 实例（读 state.Q、mesh、current_order）
        alpha_av: 人工粘性强度标定常数（无量纲，默认1.0，见上）
        sensor_var_index: Q 的第几个分量做传感器探测（默认0=密度）
        kappa: 传感器过渡带宽度（对数尺度，默认1.0，见 SENSOR_KAPPA）

    Returns:
        epsilon_av: (n_cells, n_sps) 人工粘性系数场（已广播到每个 SP，
            同一单元内取值相同——Persson-Peraire 原始方法是分段常数，
            不在单元内部变化），可直接与 mu_t_field 相加后传给
            compute_viscous_residual_ldg
    """
    mesh = solver.mesh
    Q = solver.state.Q
    n_cells, n_sps = Q.shape[0], Q.shape[1]
    order = solver.current_order if hasattr(solver, "current_order") else solver.order

    epsilon_av = np.zeros((n_cells, n_sps))
    if order == 0:
        return epsilon_av

    # `FROperators` 不对外暴露 ref_cube_sps（只是 generate_fr_operators
    # 内部的局部变量，见 fr/operators.py），按同一套 Gauss-Legendre
    # 张量积规则重新构造——与该函数构造 D_3d_tet/D_3d_prism/filter_tet/
    # filter_prism 用的完全是同一组参考坐标。
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    ref_cube_sps = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    n_prism = mesh.n_prism_cells
    field = Q[:, :, sensor_var_index]
    rho = Q[:, :, 0]
    vel_mag = np.sqrt(Q[:, :, 1] ** 2 + Q[:, :, 2] ** 2 + Q[:, :, 3] ** 2)
    h_cell = np.cbrt(np.maximum(mesh.cell_volumes, 1e-300))
    # 只统计**真实自由度**（2026-09-15 系统性审计）：直接 `.mean(axis=1)`
    # 会把 native 四面体的零填充槽位一起算进去，而那些槽位冻结在初值、
    # 会随推进变馊（实测 10 步后偏差 3.4%，且占一半槽位）。详见
    # fr/native_tet_padding.py::reduce_per_cell_over_real_sps。
    from autoflowcfd.fr.native_tet_padding import reduce_per_cell_over_real_sps
    rho_local = reduce_per_cell_over_real_sps(rho, n_prism, order, 'mean')
    vel_local = reduce_per_cell_over_real_sps(vel_mag, n_prism, order, 'mean')
    epsilon_max = alpha_av * rho_local * h_cell * vel_local / order

    if n_prism > 0:
        s_e = compute_persson_peraire_sensor(field[:n_prism], "prism", order, ref_cube_sps)
        ramp = compute_artificial_viscosity_ramp(s_e, order, kappa)
        epsilon_av[:n_prism, :] = (ramp * epsilon_max[:n_prism])[:, np.newaxis]
    if n_cells > n_prism:
        # native 专属传感器（2026-09-15 真实 bug 修复）：此前这里用坍缩
        # 坐标族的 "tet" 分支，而坍缩四面体基已于 2026-09-03 整体删除，
        # 且把冻结的零填充槽位当成真实自由度一起喂进了指标。详见
        # `_build_native_tet_sensor_operators`。
        s_e = compute_persson_peraire_sensor_native_tet(field[n_prism:], order)
        ramp = compute_artificial_viscosity_ramp(s_e, order, kappa)
        epsilon_av[n_prism:, :] = (ramp * epsilon_max[n_prism:])[:, np.newaxis]

    return epsilon_av
