"""气动系数相关 CLI 辅助函数。

`solve steady` 和 `solve transient` 两个命令都需要：求解结束后打印 Cd/Cl/Cs，
以及在用户未显式指定 --reference-area 时从面网格自动估算参考面积。拆到
独立模块，避免两个命令各自的实现文件相互依赖。
"""

from typing import Optional

import numpy as np
from loguru import logger

# 真实 bug（已修复，2026-08-21，见 cli/solve/commands.py 同一处修复的
# 文档）：此前这里用标准库 logging（从未被本项目 basicConfig 过，
# root logger 默认无 handler），本文件的 6 处 logger 调用（含
# "Auto-computed reference area" 这条 INFO 和"Failed to auto-compute
# reference area"这条 WARNING）全部被静默吞掉，用户在参考面积自动
# 估算失败时看不到任何提示。改成本代码库统一使用的 loguru。


def _compute_reference_area_auto(volume_data, direction=None, bc_overrides=None) -> Optional[float]:
    """面网格上物体面沿**来流方向**的迎风投影面积（未给 --reference-area 时的参考面积）。

    物体面与气动力积分取同一组：物理无滑移固壁（`boundary/wall_kind.py::physical_no_slip_wall_groups`，网格类型
    WALL，被 `bc_overrides` 覆盖的组按覆盖后的配置判断）。2026-10-20 以前按组名子串（BODY/CAR/WALL，排除
    INLET/OUTLET/SYMMETRY）选面——"TUNNEL_WALL" 这类名字含 WALL 而类型是 SLIP_WALL 的组被算进参考面积，而力积分
    不含它；迎风投影为 0 时静默改用"|投影|总和 / 2"；任何异常只打印警告、返回 None。

    Args:
        volume_data: VolumeMeshData（需要 surface_mesh：节点、三角形与带边界类型的边界分组）
        direction: (3,) 来流方向（不必归一化）；None 时取 +x。有攻角时必须传真实方向——按 X 投影会偏大
            1/cos(alpha)，15 度就是 3.5%，直接进 Cd 的分母
        bc_overrides: 求解器的边界条件覆盖（与气动力积分同一份）

    Returns:
        参考面积 (m^2)；体网格不带面网格数据时为 None（无法估算，调用方提示指定 --reference-area）

    Raises:
        ValueError: direction 无效；面网格没有物理无滑移固壁组；物体面沿来流的迎风投影面积不为正
    """
    from autoflowcfd.boundary.wall_kind import physical_no_slip_wall_groups

    if direction is None:
        d = np.array([1.0, 0.0, 0.0])
    else:
        d = np.asarray(direction, dtype=np.float64).ravel()
        if d.shape != (3,):
            raise ValueError(f"direction 必须是 (3,) 向量，收到 {d.shape}")
        nrm = float(np.linalg.norm(d))
        if not np.isfinite(nrm) or nrm < 1e-12:
            raise ValueError(f"direction 的模 {nrm!r} 无效")
        d = d / nrm
    surface_mesh = getattr(volume_data, 'surface_mesh', None)
    if surface_mesh is None:
        logger.info("体网格不带面网格数据，无法自动估算参考面积")
        return None

    nodes, faces, boundaries = surface_mesh['nodes'], surface_mesh['faces'], surface_mesh['boundaries']
    body_groups = sorted(physical_no_slip_wall_groups(boundaries.bc_types, bc_overrides) & set(boundaries.groups))
    if not body_groups:
        raise ValueError(f"面网格没有物理无滑移固壁组（各组类型 {dict(boundaries.bc_types)}），无法估算参考面积；"
                         "请用 --reference-area 给出")
    face_idx = np.concatenate([np.asarray(boundaries.groups[name], dtype=np.int64) for name in body_groups])
    v0, v1, v2 = (nodes[faces[face_idx, k]] for k in range(3))
    area_vectors = 0.5 * np.cross(v1 - v0, v2 - v0)
    along = area_vectors @ d
    ref_area = float(-along[along < 0].sum())            # 迎风面（外法向逆着来流）
    if not (ref_area > 0.0 and np.isfinite(ref_area)):
        raise ValueError(f"物体面 {body_groups} 沿来流方向 {np.round(d, 6).tolist()} 的迎风投影面积为 {ref_area!r}"
                         "（面网格法向朝内或物体面不完整），无法作为参考面积；请检查面网格或用 --reference-area 给出")
    logger.info(f"Auto-computed reference area of {body_groups} (projected along freestream direction "
                f"{np.round(d, 6).tolist()}): {ref_area:.6f} m^2")
    return ref_area


def resolve_reference_area(solver, volume_data, reference_area: Optional[float]) -> Optional[float]:
    """`--reference-area` 未给时沿来流方向自动估算，写到 `solver._reference_area`（每步日志的 Cd/Cl/Cs
    读它）并返回。单机 `solve steady/transient` 与 checkpoint 重建共用（此前三处各写一份，`solve transient`
    单机分支没有这一步：不传 --reference-area 时既无逐步系数也无收尾系数）。

    参考面积必须沿**来流方向**投影（有攻角时按 X 投影会偏大 1/cos(alpha)，15 度就是 3.5%，直接进 Cd 分母）。
    """
    reference_area = reference_area_for(solver.freestream, volume_data, reference_area, solver.bc_overrides)
    solver._reference_area = reference_area
    return reference_area


def reference_area_for(freestream: dict, volume_data, reference_area: Optional[float],
                       bc_overrides: Optional[dict] = None) -> Optional[float]:
    """参考面积：给了就用，没给时沿来流方向由体网格的面网格自动估算（单机、分布式与 API 共用）。"""
    if reference_area is not None:
        return reference_area
    from autoflowcfd.core.utils.flow_direction import direction_from_freestream

    return _compute_reference_area_auto(volume_data, direction=direction_from_freestream(freestream),
                                        bc_overrides=bc_overrides)


def distributed_reference_area(solver, volume_data, reference_area: Optional[float]) -> Optional[float]:
    """root 上的参考面积（其余 rank 返回 None）。`volume_data` 为 None 时取完全分布式加载在 root 上保留的那一份
    （`solver._root_context["volume_data"]`）。"""
    from autoflowcfd.core.mpi import is_root

    if not is_root():
        return None
    if volume_data is None:
        volume_data = solver._root_context["volume_data"]
    # 分布式求解器不接受 bc_overrides（边界条件全部来自网格类型），与其气动力积分一致
    return reference_area_for(solver.freestream, volume_data, reference_area)


def report_distributed_aerodynamic_coefficients(solver, reference_area: Optional[float]) -> None:
    """分布式求解收尾的气动系数（**集体调用**，全部 rank 都要调用；参考面积只需 root 给出）。

    各 rank 的 local 段汇总到 root（`core/mpi/global_view.py`），之后与单机同一个函数。2026-10-09 以前分布式路径
    不报告气动系数。逐步日志里的 Cd/Cl 只在单机打印：分布式每步汇总整场代价过高。
    """
    from autoflowcfd.core.mpi.global_view import gather_global_view

    view = gather_global_view(solver)
    if view is not None:
        _report_aerodynamic_coefficients(view, reference_area, turb_model_name=solver.turb_model_name,
                                         wall_treatment=solver.wall_treatment)


def _report_aerodynamic_coefficients(solver, reference_area: Optional[float], turb_model_name: Optional[str] = None,
                                     wall_treatment: Optional[str] = None) -> None:
    """求解结束时打印壁面 y+ 统计与力系数（全部 WALL 组、风轴系）。力矩系数需要力矩参考点与参考长度，用
    `autoflowcfd post coefficients --moment-center ... --reference-length ...` 给出。

    `turb_model_name`/`wall_treatment` 缺省取 `solver` 的同名属性（单机求解器与 GPU 主机视图）；分布式汇总视图
    没有这两项，由调用方给出（y+ 适用性判据按近壁处理方式，`postprocess/wall_yplus.py`）。
    """
    from autoflowcfd.postprocess.wall_yplus import report_wall_y_plus

    if float(solver.mu_molecular) > 0.0:
        report_wall_y_plus(solver, turb_model_name if turb_model_name is not None else solver.turb_model_name,
                           wall_treatment if wall_treatment is not None else solver.wall_treatment)
    if reference_area is None:
        print("\n(未提供 --reference-area，且体网格不带面网格数据无法自动估算，跳过气动系数)")
        return
    from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr

    coeffs = compute_aerodynamic_coefficients_fr(solver, reference_area=reference_area)
    print(f"\n=== 气动力系数（全部 WALL 组，A_ref={reference_area} m^2）===")
    print(f"   Cd (阻力)  = {coeffs.Cd:.6f}")
    print(f"   Cl (升力)  = {coeffs.Cl:.6f}")
    print(f"   Cs (侧力)  = {coeffs.Cs:.6f}")
    print("   力矩系数请用 `autoflowcfd post coefficients` 指定力矩参考点与参考长度后计算")
