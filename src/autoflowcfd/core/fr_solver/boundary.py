"""
AutoFlowCFD V2.0 - FRSolver 边界条件配置构建 (从 fr_solver.py 拆分)

把网格边界组信息与用户/默认 BC 参数接到 boundary/fr_ghost_state.py 的
BoundaryGhostStateProvider 接口上，供 core/fr_residual_inviscid.py 使用。
"""

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider, InletSEMGhostState
from autoflowcfd.grid.connectivity.face_connectivity import tag_boundary_groups

# LES/DDES 入口合成湍流默认湍流度（BD-02）：本代码库目前没有暴露专门的
# CLI/配置参数来指定目标雷诺应力张量，用来流速度的 5% 作为各向同性
# 脉动强度——是汽车风洞/道路工况常见的自由来流湍流度量级（典型范围
# 0.1%~1% 风洞、可达 5%+ 的道路自然风），偏保守但物理上合理的默认值，
# 不是任意拍的数字；有真实需求时应改为可配置参数，而不是在这里继续加
# 硬编码分支。
_SEM_DEFAULT_TURBULENCE_INTENSITY = 0.05
_SEM_DEFAULT_NUM_EDDIES = 200


def _compute_inlet_fp_positions(solver, face_conn, is_target_face: np.ndarray) -> Dict[int, np.ndarray]:
    """预计算一组边界面各自 Flux Points 的物理坐标。

    Flux Points 几何（fr/face_flux_points.py::FaceFluxPointGeometry）本身
    不存储物理坐标（只存插值矩阵/法向/面积权重，见 G-01 数值审计发现），
    这里用同一个外插矩阵直接作用在 `mesh.sps_coords`（SPs 的物理坐标场）
    上——外插算子是线性的，对坐标分量和对流场分量做外插是同一个矩阵
    运算，不需要另外实现一套"参考坐标 -> 物理坐标"映射。
    """
    mesh = solver.mesh
    ops = solver.ops
    n_prism = mesh.n_prism_cells
    positions: Dict[int, np.ndarray] = {}

    for f in np.nonzero(is_target_face)[0]:
        ffp = mesh.face_flux_points[f]
        if not ffp.owner_is_primary:
            continue
        owner_cell = int(face_conn.owner_cell[f])
        axis, side = ffp.owner_axis, ffp.owner_side
        E = ops.boundary_extrap_prism[(axis, side)] if owner_cell < n_prism else ops.boundary_extrap_tet[(axis, side)]
        positions[f] = E @ mesh.sps_coords[owner_cell]  # (n_fp, 3)

    return positions


def build_boundary_ghost_provider(solver, bc_overrides: Dict[str, Dict[str, Any]]) -> Optional[BoundaryGhostStateProvider]:
    """构建边界幽灵态提供者 (BD-01)。

    Returns:
        BoundaryGhostStateProvider，若网格没有面连接关系（尚未
        load_from_volume_mesh(build_faces=True)）则返回 None
    """
    if solver.mesh.face_connectivity is None:
        logger.warning(
            "Mesh has no face_connectivity - boundary conditions will NOT be enforced "
            "in the residual. This is only acceptable for isolated unit testing, never "
            "for a real solve."
        )
        return None

    rho_inf, vel_inf, p_inf = solver.freestream["rho_inf"], solver.freestream["vel_inf"], solver.freestream["p_inf"]
    Q_free = [rho_inf, vel_inf, 0.0, 0.0, p_inf]

    boundary_groups = solver.mesh.boundary_groups or {}
    bc_types = solver.mesh.boundary_bc_types or {}

    face_conn = solver.mesh.face_connectivity
    group_code, name_to_code = tag_boundary_groups(face_conn, boundary_groups)

    type_map = {
        "WALL": ("WALL", {"is_no_slip": True}),
        "SLIP_WALL": ("WALL", {"is_no_slip": False}),
        "VELOCITY_INLET": ("INLET", {"Q_inlet": Q_free}),
        "PRESSURE_OUTLET": ("OUTLET", {"p_outlet": p_inf}),
        "SYMMETRY": ("SYMMETRY", {}),
    }

    # BD-02：LES/DDES 模式下给 VELOCITY_INLET 组接入合成湍流入口 (SEM)，
    # 取代常量 Q_inlet——算法本身（boundary/synthetic_inlet.py）已在
    # V2.0 二次评审时修复正确（真实入口几何/雷诺应力 Cholesky 分解/
    # 涡核时间演化），但此前从未被任何边界路径调用。solver._sem_instances
    # 供 FRSolver.step() 每个物理步调用一次 advance()（见该列表的
    # 消费点），不在这里（构造阶段）就调用，因为这里只跑一次。
    use_sem = solver.turb_model_name in ("LES", "DDES") and getattr(solver, "wmles_model", None) is None
    solver._sem_instances = []

    code_to_config: Dict[int, Dict[str, Any]] = {}
    for name, code in name_to_code.items():
        override = bc_overrides.get(name)
        if override is not None:
            code_to_config[code] = override
            continue
        raw_type = bc_types.get(name, "FARFIELD")
        mapped_type, default_params = type_map.get(raw_type, ("FARFIELD", {"Q_free": Q_free}))
        config = {"type": mapped_type, **default_params}

        if mapped_type == "INLET" and use_sem:
            from autoflowcfd.boundary.synthetic_inlet import SyntheticEddyMethod

            is_this_group_face = group_code == code
            positions_by_face = _compute_inlet_fp_positions(solver, face_conn, is_this_group_face)
            if positions_by_face:
                all_positions = np.concatenate(list(positions_by_face.values()), axis=0)
                flow_direction = np.array([vel_inf, 0.0, 0.0])
                # length_scale：入口面法向尺度的量级（用坐标散布估计），
                # 太小涡核影响区退化、太大失去局部湍流结构，取入口面
                # 特征尺度的 1/10 是标准 SEM 实践的经验起点。
                span = np.max(all_positions, axis=0) - np.min(all_positions, axis=0)
                length_scale = max(float(np.max(span)) / 10.0, 1e-3)

                sem = SyntheticEddyMethod(
                    num_eddies=_SEM_DEFAULT_NUM_EDDIES, length_scale=length_scale
                )
                sem.configure_inlet_box(all_positions, flow_direction=flow_direction)

                u_fluct = _SEM_DEFAULT_TURBULENCE_INTENSITY * vel_inf
                reynolds_stress = np.diag([u_fluct**2, u_fluct**2, u_fluct**2])

                sem_ghost = InletSEMGhostState(
                    sem, positions_by_face, Q_mean=np.array(Q_free), reynolds_stress=reynolds_stress
                )
                config["sem"] = sem_ghost
                solver._sem_instances.append(sem)
                logger.info(
                    f"BD-02: Synthetic Eddy Method inlet turbulence enabled for group '{name}' "
                    f"({len(positions_by_face)} faces, {_SEM_DEFAULT_NUM_EDDIES} eddies, "
                    f"length_scale={length_scale:.4g}, u'={u_fluct:.3g} m/s)"
                )

        code_to_config[code] = config

    default_config = {"type": "FARFIELD", "Q_free": Q_free}

    logger.info(
        f"Boundary conditions configured for {len(name_to_code)} group(s): "
        f"{[(name, code_to_config[code]['type']) for name, code in name_to_code.items()]}"
    )

    return BoundaryGhostStateProvider(group_code, code_to_config, default_config)
