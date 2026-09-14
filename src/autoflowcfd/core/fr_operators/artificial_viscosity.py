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

from typing import Dict, Tuple

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
    rho_local = rho.mean(axis=1)
    vel_local = vel_mag.mean(axis=1)
    epsilon_max = alpha_av * rho_local * h_cell * vel_local / order

    if n_prism > 0:
        s_e = compute_persson_peraire_sensor(field[:n_prism], "prism", order, ref_cube_sps)
        ramp = compute_artificial_viscosity_ramp(s_e, order, kappa)
        epsilon_av[:n_prism, :] = (ramp * epsilon_max[:n_prism])[:, np.newaxis]
    if n_cells > n_prism:
        s_e = compute_persson_peraire_sensor(field[n_prism:], "tet", order, ref_cube_sps)
        ramp = compute_artificial_viscosity_ramp(s_e, order, kappa)
        epsilon_av[n_prism:, :] = (ramp * epsilon_max[n_prism:])[:, np.newaxis]

    return epsilon_av
