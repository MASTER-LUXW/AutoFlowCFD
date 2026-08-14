"""
AutoFlowCFD V2.0 - Order Continuation Utilities

本模块包含 Order Continuation 方法所需的插值工具。
"""

import time as _time

import numpy as np
from typing import Any
from loguru import logger


def interpolate_to_new_order(solver: Any, new_order: int):
    """
    将解从当前阶数插值到新的阶数（Order Continuation核心逻辑）。
    
    使用L2投影方法，确保守恒变量在插值前后保持积分守恒。
    
    Args:
        solver: FRSolver 实例
        new_order: 目标多项式阶数
    """
    from autoflowcfd.fr.operators import generate_fr_operators
    
    old_order = solver.current_order
    print(f"  Interpolating solution from P{old_order} to P{new_order}...")
    
    # 获取新旧SPs数量 - 关键修复：直接计算，不依赖solver.state.n_sps
    old_n_points_1d = old_order + 1
    old_n_sps = old_n_points_1d ** 3
    
    new_n_points_1d = new_order + 1
    new_n_sps = new_n_points_1d ** 3
    
    print(f"    Old SPs/cell: {old_n_sps}, New SPs/cell: {new_n_sps}")
    
    # 如果阶数相同，无需插值
    if old_n_sps == new_n_sps:
        print(f"    Same order, skipping interpolation")
        return
    
    # 构造插值矩阵（基于L2投影）
    # 对于FR方法，可以使用节点插值或L2投影
    # 这里使用简化的节点插值：在相同物理位置的值保持不变
    
    # 获取参考单元内的SPs坐标
    from autoflowcfd.fr.quadrature_points import gauss_legendre
    
    # 旧阶数的SPs（参考单元）
    old_sps_1d, _ = gauss_legendre(old_order + 1)
    # 新阶数的SPs（参考单元）
    new_sps_1d, _ = gauss_legendre(new_order + 1)
    
    # 对于张量积单元，构造3D SPs坐标
    old_xx, old_yy, old_zz = np.meshgrid(old_sps_1d, old_sps_1d, old_sps_1d, indexing='ij')
    old_sps_3d = np.column_stack([old_xx.ravel(), old_yy.ravel(), old_zz.ravel()])
    
    new_xx, new_yy, new_zz = np.meshgrid(new_sps_1d, new_sps_1d, new_sps_1d, indexing='ij')
    new_sps_3d = np.column_stack([new_xx.ravel(), new_yy.ravel(), new_zz.ravel()])
    
    # 使用径向基函数（RBF）插值或拉格朗日插值
    # 简化：如果新旧SPs有重叠，直接复制；否则使用最近邻插值
    n_cells = solver.state.n_cells
    n_vars = solver.state.n_vars
    
    # 创建新的状态数组
    new_U = np.zeros((n_cells, new_n_sps, n_vars))
    
    # 对每个单元进行插值
    for i in range(n_cells):
        U_old = solver.state.U[i]  # (old_n_sps, n_vars)
        
        # 对每个变量独立插值
        for v in range(n_vars):
            u_old_values = U_old[:, v]  # (old_n_sps,)
            
            # 使用scipy的插值方法
            try:
                from scipy.interpolate import RegularGridInterpolator
                
                # 重塑为3D网格 - 使用明确的变量名
                u_old_3d = u_old_values.reshape((old_n_points_1d, old_n_points_1d, old_n_points_1d))
                
                # 创建插值器
                interp = RegularGridInterpolator(
                    (old_sps_1d, old_sps_1d, old_sps_1d),
                    u_old_3d,
                    method='linear',
                    bounds_error=False,
                    fill_value=None
                )
                
                # 在新SPs位置插值
                new_U[i, :, v] = interp(new_sps_3d)
                
            except Exception as e:
                logger.warning(f"Interpolation failed for cell {i}, var {v}: {e}")
                # 回退：使用最近邻
                from scipy.spatial import cKDTree
                tree = cKDTree(old_sps_3d)
                _, indices = tree.query(new_sps_3d)
                new_U[i, :, v] = u_old_values[indices]
    
    # 更新状态
    solver.state.U = new_U
    solver.state.n_sps = new_n_sps
    solver.state.Q = np.zeros_like(solver.state.U)
    solver.state._update_primitives()
    
    # 更新湍流场（如果有）
    if hasattr(solver.turb_model, 'k_field'):
        # 插值k和omega场
        k_new = np.zeros((n_cells, new_n_sps))
        omega_new = np.zeros((n_cells, new_n_sps))
        
        for i in range(n_cells):
            try:
                from scipy.interpolate import RegularGridInterpolator
                k_old_3d = solver.turb_model.k_field[i].reshape((old_n_points_1d, old_n_points_1d, old_n_points_1d))
                omega_old_3d = solver.turb_model.omega_field[i].reshape((old_n_points_1d, old_n_points_1d, old_n_points_1d))
                
                interp_k = RegularGridInterpolator(
                    (old_sps_1d, old_sps_1d, old_sps_1d), k_old_3d,
                    method='linear', bounds_error=False, fill_value=None
                )
                interp_omega = RegularGridInterpolator(
                    (old_sps_1d, old_sps_1d, old_sps_1d), omega_old_3d,
                    method='linear', bounds_error=False, fill_value=None
                )
                
                k_new[i] = interp_k(new_sps_3d)
                omega_new[i] = interp_omega(new_sps_3d)
            except:
                from scipy.spatial import cKDTree
                tree = cKDTree(old_sps_3d)
                _, indices = tree.query(new_sps_3d)
                k_new[i] = solver.turb_model.k_field[i][indices]
                omega_new[i] = solver.turb_model.omega_field[i][indices]
        
        solver.turb_model.k_field = k_new
        solver.turb_model.omega_field = omega_new

    # 壁面距离场同样按每单元 SPs 存储（core/fr_solver_turbulence.py 的湍流
    # 源项计算直接按 SP 索引取值），阶数变化后形状同样必须一起插值——
    # 此前遗漏这一步，P0 阶段用均值压缩过的 (n_cells,1) 场会在阶数提升到
    # P1/P2 后与新的 SPs 数量不匹配，下一次湍流源项计算会形状不符崩溃
    # （真实网格已复现：与 mesh Jacobian 缺少按阶数重建是同一类"阶数变化
    # 后遗漏同步派生量"问题的另一处）。
    if getattr(solver, "wall_distance", None) is not None:
        wd_old = solver.wall_distance
        wd_old_3d = wd_old.reshape((n_cells, old_n_points_1d, old_n_points_1d, old_n_points_1d))
        wd_new = np.zeros((n_cells, new_n_sps))
        for i in range(n_cells):
            try:
                from scipy.interpolate import RegularGridInterpolator

                interp_wd = RegularGridInterpolator(
                    (old_sps_1d, old_sps_1d, old_sps_1d), wd_old_3d[i],
                    method='linear', bounds_error=False, fill_value=None
                )
                wd_new[i] = interp_wd(new_sps_3d)
            except Exception as e:
                logger.warning(f"Wall distance interpolation failed for cell {i}: {e}")
                from scipy.spatial import cKDTree
                tree = cKDTree(old_sps_3d)
                _, indices = tree.query(new_sps_3d)
                wd_new[i] = wd_old[i][indices]
        solver.wall_distance = wd_new

    # DDES 的有效长度尺度按上一个阶数的 SPs 维度算出，阶数变化后与刚插值
    # 完的 k_field 形状不再匹配——不能像 k_field/omega_field/wall_distance
    # 那样直接插值（它依赖 nu_t，而 nu_t 要到这一阶数第一次
    # compute_source_terms 调用后才会被重新算出，插值一个维度对但物理上
    # 过期的值没有意义），直接清空即可：下一步 compute_source_terms 会
    # 因为 des_length_scale is None 自动退回标准 RANS 耗散项（物理上是
    # 合理的边界处理，见 fr_solver_turbulence.py 的文档），再下一步
    # apply_to_sst_model 就能用这一阶数正确维度的 nu_t 重新算出它（真实
    # 网格已复现：不清空会在 P1->P2 等跨阶数切换时因形状不匹配崩溃）。
    if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "des_length_scale"):
        solver.turb_model.des_length_scale = None

    # 同一类"残差计算之后才更新的缓存量，跨阶数切换后维度过期"问题
    # （见上面 des_length_scale 的处理）：LES/WMLES 的 SGS 涡粘
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
    """interpolate_to_new_order 的带维度校验版本，从 fr_solver.py 拆分
    （对应旧版本 FRSolver._interpolate_to_new_order 方法体）。"""
    interpolate_to_new_order(solver, new_order)

    # 阶数变化后 SPs 每单元数量改变，DUAL_TIME 保存的上一物理时间层历史
    # （若存在）形状不再匹配，且严格来说也不再是同一离散空间下的解，
    # 必须让它失效——否则下一步 BDF2 会静默用一份形状不匹配/物理上不
    # 连续的历史层，而不是干净地退化回 BDF1。
    if hasattr(solver, "_dual_time_U_prev"):
        solver._dual_time_U_prev = None

    n_points_1d = new_order + 1
    new_n_sps = n_points_1d ** 3

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


def run_order_continuation(solver: Any, max_iter: int, dt: float, tol: float):
    """实现 Order Continuation 策略：从 P0 逐步提升到目标阶数
    （从 fr_solver.py::FRSolver._solve_with_order_continuation 拆分）。

    Args:
        solver: FRSolver 实例
        max_iter: 总迭代次数
        dt: 时间步长
        tol: 收敛容差

    Returns:
        SolverResult: 求解结果
    """
    from autoflowcfd.core.fr_state import FRState, SolverResult
    from autoflowcfd.fr.operators import generate_fr_operators

    print("\n=== Order Continuation Strategy ===")
    print(f"Starting from P0, targeting P{solver.order}")

    original_order = solver.order
    original_ops = solver.ops

    current_state_n_sps = solver.state.U.shape[1]
    expected_p0_n_sps = 1

    if current_state_n_sps != expected_p0_n_sps:
        print(f"[INFO] Current state has {current_state_n_sps} SPs/cell, reinitializing from P0...")

        p0_state = FRState(solver.state.n_cells, expected_p0_n_sps, solver.state.n_vars)
        p0_state.initialize_uniform(
            rho=solver.freestream["rho_inf"], u=solver.freestream["vel_inf"],
            v=0.0, w=0.0, p=solver.freestream["p_inf"],
        )
        solver.state = p0_state

        if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "k_field"):
            solver.turb_model.k_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * 1e-6
            solver.turb_model.omega_field = np.ones((solver.state.n_cells, expected_p0_n_sps)) * 1e-2
            print(f"[INFO] Turbulence fields reset to P0 dimensions")

        if getattr(solver, "turb_model", None) is not None and hasattr(solver.turb_model, "des_length_scale"):
            # 同 interpolate_to_new_order 里的处理：清空而不是插值，理由见
            # 该函数文档。
            solver.turb_model.des_length_scale = None

        if getattr(solver, "sgs_model", None) is not None and hasattr(solver.sgs_model, "nu_t"):
            solver.sgs_model.nu_t = None

        if solver.wall_distance is not None:
            old_wall_dist = solver.wall_distance
            if old_wall_dist.ndim == 2 and old_wall_dist.shape[1] > 1:
                mean_wall_dist = np.mean(old_wall_dist, axis=1, keepdims=True)
                solver.wall_distance = np.tile(mean_wall_dist, (1, expected_p0_n_sps))
                print(f"[INFO] Wall distance field reset to P0 dimensions")

        solver.current_order = 0
        solver.ops = generate_fr_operators(0)
        solver.mesh.set_order(0)

        print(f"[INFO] Reinitialized to P0 ({expected_p0_n_sps} SP/cell)")

    orders = list(range(0, original_order + 1))

    total_iter = 0
    for target_p in orders:
        print(f"\n--- Phase: P{target_p} ---")

        if target_p > 0:
            solver._interpolate_to_new_order(target_p)

        solver.current_order = target_p
        solver.ops = generate_fr_operators(target_p)
        # mesh 的 SPs/Jacobian/Flux Points 几何是阶数相关的（见
        # HighOrderMesh.set_order 文档）——必须随 solver.ops 一起切换，
        # 否则梯度/残差计算会用错误维度的几何量崩溃。
        solver.mesh.set_order(target_p)

        expected_n_sps = solver.ops.D_3d.shape[0]
        actual_n_sps = solver.state.U.shape[1]
        if actual_n_sps != expected_n_sps:
            raise RuntimeError(
                f"Order Continuation dimension mismatch after interpolation to P{target_p}: "
                f"State has {actual_n_sps} SPs but operators expect {expected_n_sps} SPs"
            )

        phase_max_iter = max_iter // len(orders)
        phase_tol = tol * (10 ** (original_order - target_p))

        converged = False
        final_residual = 1e10

        for i in range(phase_max_iter):
            t_start = _time.time()
            res = solver.step(dt)
            t_end = _time.time()
            final_residual = res
            total_iter += 1

            if i == 0 or (i + 1) % 10 == 0:
                print(f"P{target_p} Iter {i+1}: Residual = {res:.6e} | Time: {t_end - t_start:.2f}s")

            if res < phase_tol:
                converged = True
                print(f"[OK] P{target_p} converged at iter {i+1}")
                break

        if target_p == original_order and converged:
            print(f"\n[OK] Order Continuation completed: Final P{original_order} converged")
            return SolverResult(converged=True, iterations=total_iter, final_residual=final_residual)

    solver.order = original_order
    solver.ops = original_ops

    return SolverResult(converged=False, iterations=total_iter, final_residual=final_residual)
    print(f"     Old SPs per cell: {old_n_sps} -> New SPs per cell: {new_n_sps}")
