"""AutoFlowCFD V2.0 - 阶数切换时解场/湍流场的插值（延拓）算子"""


import numpy as np

from typing import Any

from loguru import logger
from autoflowcfd.core.time_integration.implicit.newton.physical_time import reset_dual_time_history
from autoflowcfd.fr.native_padding import sps_width


def interpolate_to_new_order(solver: Any, new_order: int):
    """把解从当前阶数延拓到新阶数（Order Continuation 升阶/降阶）。

    延拓算子是旧阶数的模态多项式在新阶数解点上的求值（`fr/order_interp.py`，原生四面体 / 原生棱柱各一份）：升阶时
    旧多项式精确落在新空间里，延拓是精确的（任意阶矩在舍入误差内守恒）。算子只依赖阶数，一次构造、向量化作用到
    全部单元与全部场（U、湍流输运量、涡粘）。

    Args:
        solver: FRSolver 实例
        new_order: 目标多项式阶数
    """
    old_order = solver.current_order
    print(f"  Interpolating solution from P{old_order} to P{new_order}...")

    # 新旧每单元解点数按阶数直接给出（不读 solver.state.n_sps）
    old_n_sps = sps_width(old_order)
    new_n_sps = sps_width(new_order)

    print(f"    Old SPs/cell: {old_n_sps}, New SPs/cell: {new_n_sps}")

    # 如果阶数相同，无需插值
    if old_n_sps == new_n_sps:
        print("    Same order, skipping interpolation")
        return

    # 延拓算子按单元类型取各自原生基的精确延拓（零填充槽位的处理见 `fr/order_interp.py` 模块文档）。
    from autoflowcfd.fr.order_interp import apply_order_interp

    n_prism_cells = int(solver.mesh.n_prism_cells)

    def _lift(field):
        return apply_order_interp(field, n_prism_cells, old_order, new_order)

    # 更新状态——(n_cells, old_n_sps, n_vars) -> (n_cells, new_n_sps, n_vars)
    solver.state.U = _lift(solver.state.U)
    solver.state.n_sps = new_n_sps
    solver.state.Q = np.zeros_like(solver.state.U)
    solver.state._update_primitives()

    # 更新湍流输运场（如果有）——(n_cells, old_n_sps) -> (n_cells, new_n_sps)，在未知量空间延拓
    # （`TransportedTurbulence.mapped_fields`）
    if hasattr(solver.turb_model, 'transported_fields'):
        solver.turb_model.set_transported_fields(solver.turb_model.mapped_fields(_lift, np))

    # nu_t（湍流涡粘系数）同样按每单元 SPs 存储，但不会随 k_field/
    # omega_field 自动变形——它只在 compute_turbulence_source 被调用时
    # 才按当时的 k/omega 重新算出。此前假设"任何读取 nu_t 的代码之前，
    # compute_turbulence_source 总会先跑一遍把它刷新成当前阶数的正确
    # 形状"，所以这里从未插值它；但 `_compute_local_time_step` 的粘性
    # CFL 项如果在 nu_t 刷新之前就先被调用（阶数切换后的第一步），会读到
    # 上一阶数形状的陈旧 nu_t——真实复现：P1->P2 切换后 rho 已是
    # (n_cells,27)、nu_t 还留着 P1 的 (n_cells,8)，两者形状既不相等也
    # 没有一方是 1，相乘直接 ValueError 广播失败。用同一个 W 矩阵一并
    # 插值，与 k_field/omega_field 一致处理。
    if getattr(solver.turb_model, "nu_t", None) is not None and solver.turb_model.nu_t.shape[1] == old_n_sps:
        solver.turb_model.nu_t = _lift(solver.turb_model.nu_t)

    # 壁面距离不在这里处理：它是纯几何量，`set_order` 之后由来源在新解点上重新查询
    # （`run.py` -> `recompute_wall_distance_for_current_order`）。此前这里先插值"让形状对上"，
    # 没有来源时插值结果就被静默保留（2026-09-06 修掉的同类缺陷）。

    # "残差计算之后才更新的缓存量，跨阶数切换后维度过期"：LES/WMLES 的 SGS 涡粘
    # (sgs_model.nu_t) 由 apply_turbulence_corrections 在 step() 末尾算出，
    # 但 compute_viscous_residual（同一步更早）就要读取它——跨阶数切换后
    # 直接清空，get_turbulent_viscosity_field 已经对 None 做了判断（这一
    # 步退化为纯分子粘度，物理上合理的边界处理），下一步 SGS 涡粘会用新
    # 维度重新算出（真实网格已复现：不清空会在 P1->P2 等切换时因形状
    # 不匹配崩溃）。
    if getattr(solver, "sgs_model", None) is not None and hasattr(solver.sgs_model, "nu_t"):
        solver.sgs_model.nu_t = None

    # 更新SPs数量
    solver.state.n_sps = new_n_sps  # 关键修复：确保n_sps属性被正确更新
    solver.current_order = new_order

    # 注意：不在这里更新solver.ops，由调用者负责

    print(f"  ✅ Solution interpolated to P{new_order}")


def interpolate_to_new_order_checked(solver: Any, new_order: int) -> None:
    """interpolate_to_new_order 的带维度校验版本，从 fr_solver/ 拆分
    （对应旧版本 FRSolver._interpolate_to_new_order 方法体）。"""
    interpolate_to_new_order(solver, new_order)

    # 阶数变化后 SPs 每单元数量改变，DUAL_TIME 保存的上一物理时间层历史
    # （若存在）形状不再匹配，且严格来说也不再是同一离散空间下的解，
    # 必须让它失效——否则下一步 BDF2 会静默用一份形状不匹配/物理上不
    # 连续的历史层，而不是干净地退化回 BDF1。
    reset_dual_time_history(solver)

    # NEWTON_KRYLOV 跨步状态随阶数失效（理由见 reset_newton_state 文档）
    from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import reset_newton_state

    reset_newton_state(solver)

    new_n_sps = sps_width(new_order)

    actual_n_sps = solver.state.U.shape[1]
    if actual_n_sps != new_n_sps:
        logger.error(
            f"After interpolation: expected {new_n_sps} SPs but got {actual_n_sps}. "
            f"This indicates a bug in the interpolation routine."
        )
        raise RuntimeError(
            f"State dimension mismatch after Order Continuation: "
            f"expected {new_n_sps} SPs/cell, got {actual_n_sps}"
        )

    logger.info(f"Order Continuation: Successfully interpolated to P{new_order} ({new_n_sps} SPs/cell)")
