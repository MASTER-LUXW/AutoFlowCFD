"""FR 求解器的气动力/力矩系数（壁面通量点上用求解器的数值通量积分，见 `postprocess/wall_forces.py`）。

- 力按**风轴系**分解（`core/utils/flow_direction.py::wind_axes`：阻力沿来流、侧力、升力），攻角/侧滑角为零时
  即体轴单位基；力矩报在**体轴系**（俯仰 Cm 绕 y、偏航 Cy 绕 z、滚转 Cr 绕 x），绕给定力矩参考点，用参考
  长度无量纲化。
- 默认积分全部 WALL 组并逐组打印分量；`groups` 指定时只积分给定组（例如只要车身、不要地面）。
"""

from typing import Dict, Optional, Sequence

import numpy as np
from loguru import logger

from autoflowcfd.postprocess.coefficients import AerodynamicCoefficients
from autoflowcfd.postprocess.wall_forces import sample_wall, wall_group_codes


def _wind_axes(freestream):
    from autoflowcfd.core.utils.flow_direction import wind_axes

    return wind_axes(float(freestream.get("aoa_deg", 0.0) or 0.0), float(freestream.get("aos_deg", 0.0) or 0.0))


def _dynamic_pressure_area(freestream, reference_area: float) -> float:
    if not (reference_area > 0.0):
        raise ValueError(f"参考面积必须为正，收到 {reference_area}")
    return 0.5 * float(freestream["rho_inf"]) * float(freestream["vel_inf"]) ** 2 * reference_area


def compute_aerodynamic_coefficients_fr(
    solver,
    reference_area: float,
    reference_length: float = 1.0,
    moment_center: Optional[Sequence[float]] = None,
    include_viscous: bool = True,
    groups: Optional[Sequence[str]] = None,
) -> AerodynamicCoefficients:
    """气动力与力矩系数。

    Args:
        solver: 单机求解器（或其主机视图）
        reference_area: 参考面积 A_ref (m^2)
        reference_length: 力矩参考长度 L_ref (m)
        moment_center: 力矩参考点 (3,)，缺省为原点
        include_viscous: 是否计入粘性（摩擦）力
        groups: 参与积分的 WALL 组名；缺省为全部 WALL 组
    """
    host = solver.host_view() if hasattr(solver, "host_view") else solver
    code_to_name = wall_group_codes(host.boundary_ghost_provider, groups)
    samples = sample_wall(host, list(code_to_name), viscous=include_viscous)
    if samples.group_code.size == 0:
        raise RuntimeError(f"组 {sorted(code_to_name.values())} 上没有壁面通量点")
    mc = np.zeros(3) if moment_center is None else np.asarray(moment_center, dtype=float).reshape(3)
    arm = samples.position - mc
    f_total = samples.force_inviscid + samples.force_viscous

    for code, name in code_to_name.items():
        sel = samples.group_code == code
        logger.info(f"壁面组 '{name}'：压力力 {samples.force_inviscid[sel].sum(axis=0)}，"
                    f"粘性力 {samples.force_viscous[sel].sum(axis=0)} N")

    force = f_total.sum(axis=0)
    moment = np.cross(arm, f_total).sum(axis=0)
    qa = _dynamic_pressure_area(host.freestream, reference_area)
    if not (reference_length > 0.0):
        raise ValueError(f"参考长度必须为正，收到 {reference_length}")
    d_hat, s_hat, l_hat = _wind_axes(host.freestream)
    logger.info(f"气动力积分：{samples.group_code.size} 个壁面通量点，合力 {force} N，"
                f"力矩 {moment} N·m（参考点 {mc}，L_ref={reference_length} m）")
    return AerodynamicCoefficients(
        Cd=float(force @ d_hat / qa), Cl=float(force @ l_hat / qa), Cs=float(force @ s_hat / qa),
        Cm=float(moment[1] / (qa * reference_length)), Cy=float(moment[2] / (qa * reference_length)),
        Cr=float(moment[0] / (qa * reference_length)),
    )


def pressure_force_coefficients(solver, reference_area: float, groups: Optional[Sequence[str]] = None) -> Dict[str,
                                                                                                               float]:
    """只含压力（无粘数值通量）部分的力系数 {'Cd','Cl','Cs'}，供每步监控：不算速度梯度，开销只在壁面通量点上。"""
    host = solver.host_view() if hasattr(solver, "host_view") else solver
    code_to_name = wall_group_codes(host.boundary_ghost_provider, groups)
    force = sample_wall(host, list(code_to_name), viscous=False).force_inviscid.sum(axis=0)
    qa = _dynamic_pressure_area(host.freestream, reference_area)
    d_hat, s_hat, l_hat = _wind_axes(host.freestream)
    return {'Cd': float(force @ d_hat / qa), 'Cl': float(force @ l_hat / qa), 'Cs': float(force @ s_hat / qa)}
