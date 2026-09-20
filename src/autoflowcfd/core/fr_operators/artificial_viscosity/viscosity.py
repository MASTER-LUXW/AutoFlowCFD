"""AutoFlowCFD V2.0 - 人工粘性的斜坡映射、troubled-cell 掩码与完整流水线。

从 `artificial_viscosity.py` 拆出（2026-09-20）。纯搬家，逻辑未改。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from autoflowcfd.fr.collapsed_basis import prism_modal_basis_and_grad, tet_modal_basis_and_grad
from autoflowcfd.fr.quadrature_points import gauss_legendre
from autoflowcfd.fr.native_prism.mode import prism_basis_is_native
from autoflowcfd.core.utils.array_module import array_module as _array_module

from .sensor import (
    compute_persson_peraire_sensor,
    compute_persson_peraire_sensor_native_prism,
    compute_persson_peraire_sensor_native_tet,
)
from .sensor_operators import DEFAULT_SENSOR_VAR_INDEX, SENSOR_KAPPA


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
    xp = _array_module(s_e)
    s0 = -4.0 * np.log10(max(order, 1))     # 标量，与数组模块无关
    ramp = xp.zeros_like(s_e)
    mid = (s_e >= s0 - kappa) & (s_e <= s0 + kappa)
    high = s_e > s0 + kappa
    ramp[mid] = 0.5 * (1.0 + xp.sin(np.pi * (s_e[mid] - s0) / (2.0 * kappa)))
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

    xp = _array_module(field_nodal, cell_is_prism)
    n_cells = field_nodal.shape[0]
    mask = xp.zeros(n_cells, dtype=xp.bool_)
    if order == 0 or n_cells == 0:
        return mask

    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    xx, yy, zz = np.meshgrid(sps_1d, sps_1d, sps_1d, indexing="ij")
    ref_cube_sps = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    if n_prism is not None:
        groups = ((xp.arange(0, n_prism), "prism"),
                  (xp.arange(n_prism, n_cells), "tet"))
    else:
        cip = xp.asarray(cell_is_prism).astype(bool)
        if cip.shape != (n_cells,):
            raise ValueError(
                f"cell_is_prism 形状 {cip.shape} 与场的单元数 {n_cells} 不符")
        groups = ((xp.flatnonzero(cip), "prism"),
                  (xp.flatnonzero(~cip), "tet"))

    for sel, cell_type in groups:
        if sel.size == 0:
            continue
        # 四面体走 native 专属传感器（正交归一 PKD 基 + 只用真实自由度），
        # 棱柱走张量积族 + GL 求积权重——两条分支各自精确，理由见
        # `_build_native_tet_sensor_operators`。
        if cell_type == "tet":
            s_e = compute_persson_peraire_sensor_native_tet(
                xp.ascontiguousarray(field_nodal[sel]), order)
        elif prism_basis_is_native():
            s_e = compute_persson_peraire_sensor_native_prism(
                xp.ascontiguousarray(field_nodal[sel]), order)
        else:
            s_e = compute_persson_peraire_sensor(
                xp.ascontiguousarray(field_nodal[sel]), cell_type, order,
                ref_cube_sps)
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
    # fr/native_padding.py::reduce_per_cell_over_real_sps。
    from autoflowcfd.fr.native_padding import reduce_per_cell_over_real_sps
    rho_local = reduce_per_cell_over_real_sps(rho, n_prism, order, 'mean')
    vel_local = reduce_per_cell_over_real_sps(vel_mag, n_prism, order, 'mean')
    epsilon_max = alpha_av * rho_local * h_cell * vel_local / order

    if n_prism > 0:
        # 棱柱同样按基分派（2026-09-20，与四面体那处同一类缺陷，见
        # `_build_native_prism_sensor_operators`）。
        if prism_basis_is_native():
            s_e = compute_persson_peraire_sensor_native_prism(
                field[:n_prism], order)
        else:
            s_e = compute_persson_peraire_sensor(
                field[:n_prism], "prism", order, ref_cube_sps)
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
