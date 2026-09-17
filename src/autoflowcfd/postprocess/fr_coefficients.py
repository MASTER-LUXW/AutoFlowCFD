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
        reference_length: 力矩系数参考长度 L_ref（m），通常是车身长度/轴距，
            力矩系数分母为 q_inf * A_ref * L_ref
        moment_center: 力矩参考点 (3,)，None 时取坐标原点；力矩按
            M = Σ (r × dF) 积分，r 为各 Flux Point 物理坐标与该参考点之差
        include_viscous: 是否包含粘性摩擦力贡献（默认 True；WALL 边界的
            粘性梯度在低速无 WMLES 时的物理保真度见
            core/fr_viscous_flux.py 模块文档"已知局限"一节——粘性力可能
            低估，但绝不是无中生有，仍然是用实际解场算出的真实积分量）

    Returns:
        AerodynamicCoefficients（Cd/Cl/Cs 与力矩系数 Cm/Cy/Cr 均已由真实面积分填入）
    """
    mesh = solver.mesh
    fc = mesh.face_connectivity
    if fc is None or mesh.face_flux_points is None:
        raise RuntimeError(
            "Mesh has no face_connectivity/face_flux_points - cannot integrate "
            "aerodynamic forces. Call load_from_volume_mesh(build_faces=True) first."
        )

    from autoflowcfd.grid.connectivity.face_connectivity import tag_boundary_groups_for_mesh

    group_code, name_to_code = tag_boundary_groups_for_mesh(mesh, fc)
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

    def extrap_to_face(cell: int, field: np.ndarray, axis: int, side: float, oc_code: int) -> np.ndarray:
        """体积场外插到某个 WALL 面的 Flux Points。

        真实 bug 修复（2026-09-03，"delete collapsed"后用真实 cube_demo
        网格端到端 CLI 冒烟测试首次触发——之前的生产缓存网格恰好没有
        WALL 面被四面体单元拥有，掩盖了这个此前就存在的缺口）：四面体
        坍缩坐标基已删除（见 fr/operators.py 模块文档），`ops.boundary_
        extrap_tet` 现在是占位全零字典，键仍是 (axis,side)（axis∈{0,1,2}）
        ——但 `axis`（`ffp.owner_axis`）对 native 四面体面存的是复用槽位
        的 excluded_vertex（可达 3），不在这个占位字典任何合法键里，会
        直接 `KeyError: (3, -1.0)`（真实复现）。改用 `oc_code`（
        `fc.owner_cube_face[f]`，>=6 即 native 真实面，与本项目其余
        所有消费点同一个判据）分派到 `ops.boundary_extrap_native_tet
        [excluded_vertex]`——这个矩阵形状是 (n_fp,n_native)，不是 padded
        到全局 n_sps 宽度的版本，必须先把 `field` 按 `[:n_native]` 切片
        （填充槽位不携带真实自由度，见 native_tet_padding.py 文档），
        再做矩阵乘法，不能直接对全宽度 `field` 求值。
        """
        if oc_code >= 6:
            excluded_vertex = oc_code - 6
            E = ops.boundary_extrap_native_tet[excluded_vertex]  # (n_fp, n_native)
            n_native = E.shape[1]
            trailing = field.shape[1:]
            flat = E @ field[:n_native].reshape(n_native, -1)
            return flat.reshape((E.shape[0],) + trailing)
        E = ops.boundary_extrap_prism[(axis, side)]
        trailing = field.shape[1:]
        flat = E @ field.reshape(field.shape[0], -1)
        return flat.reshape((E.shape[0],) + trailing)

    force_pressure = np.zeros(3)
    force_viscous = np.zeros(3)
    # 力矩与力同循环累加：M = Σ (r × dF)，r = FP 物理坐标 - 力矩参考点。
    # FP 几何本身不存物理坐标，用与流场完全相同的边界外插矩阵作用在
    # mesh.sps_coords（SPs 物理坐标场）上得到，机制与
    # core/fr_solver/boundary.py::_compute_inlet_fp_positions 一致：
    # 外插算子是线性的，对坐标分量和对流场分量是同一个矩阵运算。
    mc = np.zeros(3) if moment_center is None else np.asarray(moment_center, dtype=float).reshape(3)
    moment_pressure = np.zeros(3)
    moment_viscous = np.zeros(3)

    if include_viscous:
        from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
        from autoflowcfd.core.fr_residual.viscous_flux import compute_temperature

        grad_Q = compute_physical_gradient(Q, mesh, ops)  # (n_cells,n_sps,5,3)
        grad_vel_full = grad_Q[:, :, 1:4, :]  # (n_cells,n_sps,3,3)

    for f in np.nonzero(is_wall_face)[0]:
        ffp = mesh.face_flux_points[f]
        if not ffp.owner_is_primary:
            continue
        owner_cell = int(fc.owner_cell[f])
        axis, side = ffp.owner_axis, ffp.owner_side
        oc_code = int(fc.owner_cube_face[f])

        Q_fp = extrap_to_face(owner_cell, Q[owner_cell], axis, side, oc_code)  # (n_fp,5)
        p_fp = Q_fp[:, 4]
        normal = ffp.true_normal  # (n_fp,3)
        area_w = ffp.true_area_weight  # (n_fp,)
        # 本面各 Flux Point 的物理坐标（外插 SPs 坐标场），减去力矩参考点得臂向量
        r_arm = extrap_to_face(owner_cell, mesh.sps_coords[owner_cell], axis, side, oc_code) - mc  # (n_fp,3)

        d_force_p = p_fp[:, None] * normal * area_w[:, None]
        force_pressure += np.sum(d_force_p, axis=0)
        moment_pressure += np.sum(np.cross(r_arm, d_force_p), axis=0)

        if include_viscous:
            gv_fp = extrap_to_face(owner_cell, grad_vel_full[owner_cell], axis, side, oc_code)  # (n_fp,3,3)
            mu_t_fp = (
                extrap_to_face(owner_cell, mu_t_field[owner_cell][:, None], axis, side, oc_code)[:, 0]
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
            d_force_v = -traction * area_w[:, None]
            force_viscous += np.sum(d_force_v, axis=0)
            moment_viscous += np.sum(np.cross(r_arm, d_force_v), axis=0)

    force_total = force_pressure + force_viscous
    moment_total = moment_pressure + moment_viscous

    rho_inf = solver.freestream["rho_inf"]
    vel_inf = solver.freestream["vel_inf"]
    q_inf = 0.5 * rho_inf * vel_inf**2
    denom = max(q_inf * reference_area, 1e-300)
    denom_moment = max(q_inf * reference_area * reference_length, 1e-300)

    # 力按**风轴系**分解（2026-09-17）：此前这里直接取体轴分量
    # （Cd=F[0]/Cl=F[2]/Cs=F[1]），那等价于假定来流恒沿 +x —— 一旦有
    # 攻角，"阻力"就不再是来流方向的分量、"升力"也不再垂直于来流，
    # 升阻比整个失去意义。现在按 core/utils/flow_direction.py::wind_axes
    # 给出的正交三元组投影。
    #
    # aoa=aos=0 时该三元组严格等于单位基 (1,0,0)/(0,1,0)/(0,0,1)，所以
    # 默认路径下与此前**逐位相同**。
    #
    # 力矩仍报在**体轴系**（俯仰 Cm=绕 y、偏航 Cy=绕 z、滚转 Cr=绕 x），
    # 不随攻角旋转——这是气动数据的标准呈现方式，力与力矩混用不同轴系
    # 会让同一份数据在不同攻角下不可比。见该模块文档"力矩留在体轴系"。
    from autoflowcfd.core.utils.flow_direction import wind_axes

    d_hat, s_hat, l_hat = wind_axes(
        float(solver.freestream.get("aoa_deg", 0.0) or 0.0),
        float(solver.freestream.get("aos_deg", 0.0) or 0.0),
    )
    Cd = float(np.dot(force_total, d_hat) / denom)
    Cl = float(np.dot(force_total, l_hat) / denom)
    Cs = float(np.dot(force_total, s_hat) / denom)
    Cm = float(moment_total[1] / denom_moment)
    Cy = float(moment_total[2] / denom_moment)
    Cr = float(moment_total[0] / denom_moment)

    logger.info(
        f"Aerodynamic force integration: {n_wall_faces} wall faces, "
        f"F_pressure={force_pressure}, F_viscous={force_viscous}, F_total={force_total}, "
        f"M_total={moment_total} (center={mc}, L_ref={reference_length})"
    )

    return AerodynamicCoefficients(Cd=Cd, Cl=Cl, Cm=Cm, Cs=Cs, Cy=Cy, Cr=Cr)


def compute_forces_pressure_only(solver, reference_area: float) -> dict:
    """轻量级压力积分——仅做压力面积分，不计算粘性力梯度。

    设计用于每迭代步输出 Cd/Cl/Cs 监控，避免每步都算粘性力梯度（开销大）。
    返回字典而非 AerodynamicCoefficients 对象，方便格式化输出。

    Args:
        solver: FRSolver 实例
        reference_area: 参考面积 (m^2)

    Returns:
        dict: {'Cd': float, 'Cl': float, 'Cs': float}，若计算失败返回全零
    """
    mesh = solver.mesh
    fc = mesh.face_connectivity
    if fc is None or mesh.face_flux_points is None:
        return {'Cd': 0.0, 'Cl': 0.0, 'Cs': 0.0}

    try:
        from autoflowcfd.grid.connectivity.face_connectivity import tag_boundary_groups_for_mesh

        group_code, name_to_code = tag_boundary_groups_for_mesh(mesh, fc)
        bc_types = mesh.boundary_bc_types or {}
        wall_codes = {code for name, code in name_to_code.items()
                      if bc_types.get(name, "") in ("WALL",)}
        if not wall_codes:
            return {'Cd': 0.0, 'Cl': 0.0, 'Cs': 0.0}
        is_wall_face = np.isin(group_code, list(wall_codes))

        ops = solver.ops
        Q = solver.state.Q

        force = np.zeros(3)
        for f in np.nonzero(is_wall_face)[0]:
            ffp = mesh.face_flux_points[f]
            if not ffp.owner_is_primary:
                continue
            owner_cell = int(fc.owner_cell[f])
            axis, side = ffp.owner_axis, ffp.owner_side
            oc_code = int(fc.owner_cube_face[f])

            # 真实 bug 修复（2026-09-03，同一处见 compute_aerodynamic_
            # coefficients_fr::extrap_to_face 文档）：四面体坍缩坐标基
            # 已删除，`axis`（`ffp.owner_axis`）对 native 四面体面存的是
            # 复用槽位的 excluded_vertex（可达 3），不能无条件拿去索引
            # 占位全零的 `ops.boundary_extrap_tet` 字典（只有 axis∈{0,1,2}
            # 的键，越界会直接 KeyError）——按 `oc_code>=6`（native 真实
            # 面）分派到 `ops.boundary_extrap_native_tet[excluded_vertex]`
            # （形状 (n_fp,n_native)，只对 `Q[...,4][:n_native]` 这部分
            # 真实自由度求值，填充槽位不携带真实场值）。
            if oc_code >= 6:
                excluded_vertex = oc_code - 6
                E = ops.boundary_extrap_native_tet[excluded_vertex]  # (n_fp, n_native)
                Q_fp = E @ Q[owner_cell, :E.shape[1], 4]
            else:
                E = ops.boundary_extrap_prism[(axis, side)]
                Q_fp = E @ Q[owner_cell, :, 4]  # pressure only, (n_fp,)
            normal = ffp.true_normal
            area_w = ffp.true_area_weight
            force += np.sum(Q_fp[:, None] * normal * area_w[:, None], axis=0)

        rho_inf = solver.freestream["rho_inf"]
        vel_inf = solver.freestream["vel_inf"]
        denom = max(0.5 * rho_inf * vel_inf**2 * reference_area, 1e-300)

        # 与 compute_aerodynamic_coefficients 同一套风轴系分解（2026-09-17）。
        # 这条轻量路径是**每迭代步**都调的监控，若它仍按体轴取分量、而收尾
        # 的完整积分按风轴投影，同一次运行的逐步 Cd 与最终 Cd 在有攻角时
        # 会对不上——那种不一致比两处都错更难排查。
        from autoflowcfd.core.utils.flow_direction import wind_axes

        d_hat, s_hat, l_hat = wind_axes(
            float(solver.freestream.get("aoa_deg", 0.0) or 0.0),
            float(solver.freestream.get("aos_deg", 0.0) or 0.0),
        )
        return {
            'Cd': float(np.dot(force, d_hat) / denom),
            'Cl': float(np.dot(force, l_hat) / denom),
            'Cs': float(np.dot(force, s_hat) / denom),
        }
    except Exception as e:
        logger.warning(f"compute_forces_pressure_only 计算失败，返回零系数（根因需要排查，不应被忽略）: {e}")
        return {'Cd': 0.0, 'Cl': 0.0, 'Cs': 0.0}
