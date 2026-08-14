"""
AutoFlowCFD V2.0 - FRSolver 边界条件配置构建 (从 fr_solver.py 拆分)

把网格边界组信息与用户/默认 BC 参数接到 boundary/fr_ghost_state.py 的
BoundaryGhostStateProvider 接口上，供 core/fr_residual_inviscid.py 使用。
"""

from typing import Any, Dict, Optional

from loguru import logger

from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider
from autoflowcfd.grid.face_connectivity import tag_boundary_groups


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

    group_code, name_to_code = tag_boundary_groups(solver.mesh.face_connectivity, boundary_groups)

    type_map = {
        "WALL": ("WALL", {"is_no_slip": True}),
        "SLIP_WALL": ("WALL", {"is_no_slip": False}),
        "VELOCITY_INLET": ("INLET", {"Q_inlet": Q_free}),
        "PRESSURE_OUTLET": ("OUTLET", {"p_outlet": p_inf}),
        "SYMMETRY": ("SYMMETRY", {}),
    }

    code_to_config: Dict[int, Dict[str, Any]] = {}
    for name, code in name_to_code.items():
        override = bc_overrides.get(name)
        if override is not None:
            code_to_config[code] = override
            continue
        raw_type = bc_types.get(name, "FARFIELD")
        mapped_type, default_params = type_map.get(raw_type, ("FARFIELD", {"Q_free": Q_free}))
        code_to_config[code] = {"type": mapped_type, **default_params}

    default_config = {"type": "FARFIELD", "Q_free": Q_free}

    logger.info(
        f"Boundary conditions configured for {len(name_to_code)} group(s): "
        f"{[(name, code_to_config[code]['type']) for name, code in name_to_code.items()]}"
    )

    return BoundaryGhostStateProvider(group_code, code_to_config, default_config)
