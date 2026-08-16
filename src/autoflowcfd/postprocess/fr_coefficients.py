"""
AutoFlowCFD V2.0 - FR 原生气动力系数计算 (P-03 相关，直接在 solve 结束时报告)

V2.0 二次评审发现 `postprocess/coefficients.py::CoefficientCalculator` 是
V1（FVM，`GridData`/`SolutionVector` 单元中心存储）时代的实现，在 V2 的
FR 求解器（`(n_cells,n_sps,n_vars)` 多点存储、`HighOrderMesh`/
`FRFaceConnectivity` 几何）下从未被真正打通：`_get_average_pressure()`
硬编码返回 101325.0（不读解场），`calculate_by_boundary()` 调用不存在的
`grid_data.get_face_data()`，最终导致任何工况下 Cd/Cl 恒为 0（详见
ProjectFiles/V2.0/6_整体专家组二次评审.md 发现23）。

本模块直接在 `FRSolver` 的原生数据（`mesh.face_connectivity`/
`mesh.face_flux_points`/`state.Q`）上重新实现压力+粘性力积分，复用
`core/fr_solver_boundary.py`（BD-01 幽灵态）、`core/solver_helpers.py`
（WMLES 壁面应力修正，T-05 修复）已经验证过的同一套"面 -> 物理量 ->
外插到 Flux Points -> 用面积权重积分"机制，不是重新发明一套近似。
"""

from typing import Optional

import numpy as np
from loguru import logger

from autoflowcfd.postprocess.coefficients import AerodynamicCoefficients


def compute_aerodynamic_coefficients_fr(
    solver,
    reference_area: float,
    reference_length: float = 1.0,
    moment_center: Optional[np.ndarray] = None,
    include_viscous: bool = True,
) -> AerodynamicCoefficients:
    """在 FRSolver 的当前解上直接积分 WALL 边界的压力+粘性力，得到气动系数。

    力的方向约定：来流方向为阻力方向（Cd），来流的 z 方向分量为升力方向
    （Cl，右手系下 x=流向, y=展向, z=法向，与本代码库自由来流约定
    vel_inf 沿 +x 一致，见 FRSolver.freestream）。

    符号推导（详见模块内联注释）：`ffp.true_normal` 是流体域边界面的
    外法向（指向域外，即指向固体内部）。物体表面受到的合力
        F_on_body = ∮ p * true_normal dA - ∮ (tau · true_normal) dA
    压力项已用真实网格前缘驻点处高压方向验证符号正确（高压 + 法向指向
    下游 = 阻力方向为正）。

    Args:
        solver: 已完成求解的 FRSolver 实例
        reference_area: 参考面积 A_ref（m^2），通常是车辆正面投影面积
        reference_length: 参考长度（力矩系数用，本函数当前不计算力矩）
        moment_center: 力矩参考点（当前未使用，保留参数位供后续扩展）
        include_viscous: 是否包含粘性摩擦力贡献（默认 True；WALL 边界的
            粘性梯度在低速无 WMLES 时的物理保真度见
            core/fr_viscous_flux.py 模块文档"已知局限"一节——粘性力可能
            低估，但绝不是无中生有，仍然是用实际解场算出的真实积分量）

    Returns:
        AerodynamicCoefficients（Cd/Cl 已填，Cm/Cs/Cy/Cr 当前为 0，
        本函数未实现力矩积分）
    """
    mesh = solver.mesh
    fc = mesh.face_connectivity
    if fc is None or mesh.face_flux_points is None:
        raise RuntimeError(
            "Mesh has no face_connectivity/face_flux_points - cannot integrate "
            "aerodynamic forces. Call load_from_volume_mesh(build_faces=True) first."
        )

    from autoflowcfd.grid.face_connectivity import tag_boundary_groups

    group_code, name_to_code = tag_boundary_groups(fc, mesh.boundary_groups or {})
    bc_types = mesh.boundary_bc_types or {}
    wall_codes = {code for name, code in name_to_code.items() if bc_types.get(name, "") in ("WALL",)}
    if not wall_codes:
        raise RuntimeError(
            "No WALL boundary group found in this mesh - cannot compute aerodynamic "
            "coefficients (Cd/Cl are only meaningful with at least one solid wall)."
        )
    is_wall_face = np.isin(group_code, list(wall_codes))
    n_wall_faces = int(np.sum(is_wall_face))
    if n_wall_faces == 0:
        raise RuntimeError("WALL boundary group(s) matched but zero faces tagged - check mesh boundary groups.")

    n_prism = mesh.n_prism_cells
    ops = solver.ops
    Q = solver.state.Q
    mu = solver.mu_molecular
    mu_t_field = solver._get_turbulent_viscosity_field()

    def extrap_to_face(cell: int, field: np.ndarray, axis: int, side: float) -> np.ndarray:
        E = ops.boundary_extrap_prism[(axis, side)] if cell < n_prism else ops.boundary_extrap_tet[(axis, side)]
        trailing = field.shape[1:]
        flat = E @ field.reshape(field.shape[0], -1)
        return flat.reshape((E.shape[0],) + trailing)

    force_pressure = np.zeros(3)
    force_viscous = np.zeros(3)

    if include_viscous:
        from autoflowcfd.core.fr_gradients import compute_physical_gradient
        from autoflowcfd.core.fr_viscous_flux import compute_temperature

        grad_Q = compute_physical_gradient(Q, mesh, ops)  # (n_cells,n_sps,5,3)
        grad_vel_full = grad_Q[:, :, 1:4, :]  # (n_cells,n_sps,3,3)

    for f in np.nonzero(is_wall_face)[0]:
        ffp = mesh.face_flux_points[f]
        if not ffp.owner_is_primary:
            continue
        owner_cell = int(fc.owner_cell[f])
        axis, side = ffp.owner_axis, ffp.owner_side

        Q_fp = extrap_to_face(owner_cell, Q[owner_cell], axis, side)  # (n_fp,5)
        p_fp = Q_fp[:, 4]
        normal = ffp.true_normal  # (n_fp,3)
        area_w = ffp.true_area_weight  # (n_fp,)

        force_pressure += np.sum(p_fp[:, None] * normal * area_w[:, None], axis=0)

        if include_viscous:
            gv_fp = extrap_to_face(owner_cell, grad_vel_full[owner_cell], axis, side)  # (n_fp,3,3)
            mu_t_fp = (
                extrap_to_face(owner_cell, mu_t_field[owner_cell][:, None], axis, side)[:, 0]
                if mu_t_field is not None
                else np.zeros(gv_fp.shape[0])
            )
            mu_total = mu + mu_t_fp
            S = 0.5 * (gv_fp + np.swapaxes(gv_fp, -1, -2))
            div_u = gv_fp[:, 0, 0] + gv_fp[:, 1, 1] + gv_fp[:, 2, 2]
            lam = -2.0 / 3.0 * mu_total
            eye3 = np.eye(3)
            tau = 2.0 * mu_total[:, None, None] * S + lam[:, None, None] * div_u[:, None, None] * eye3  # (n_fp,3,3)
            traction = np.einsum("fij,fj->fi", tau, normal)  # (n_fp,3): tau·true_normal
            force_viscous += -np.sum(traction * area_w[:, None], axis=0)

    force_total = force_pressure + force_viscous

    rho_inf = solver.freestream["rho_inf"]
    vel_inf = solver.freestream["vel_inf"]
    q_inf = 0.5 * rho_inf * vel_inf**2
    denom = max(q_inf * reference_area, 1e-300)

    # 来流沿 +x（见 FRSolver.freestream 文档），阻力=流向分量，升力=z向分量
    Cd = float(force_total[0] / denom)
    Cl = float(force_total[2] / denom)
    Cs = float(force_total[1] / denom)

    logger.info(
        f"Aerodynamic force integration: {n_wall_faces} wall faces, "
        f"F_pressure={force_pressure}, F_viscous={force_viscous}, F_total={force_total}"
    )

    return AerodynamicCoefficients(Cd=Cd, Cl=Cl, Cm=0.0, Cs=Cs, Cy=0.0, Cr=0.0)
