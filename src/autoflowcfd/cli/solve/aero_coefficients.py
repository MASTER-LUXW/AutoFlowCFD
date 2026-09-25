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


def _compute_reference_area_auto(volume_data, direction=None) -> Optional[float]:
    """从面网格自动计算参考面积（**沿来流方向**的正投影面积）。

    当用户未指定 --reference-area 时调用，从保存的原始面网格数据计算
    车身迎风面的投影面积，作为气动力系数的参考面积。

    Args:
        volume_data: VolumeMeshData 对象，应包含 surface_mesh 属性
        direction: (3,) 来流单位方向。None 时取 +x —— 与此前把投影方向
            硬编码成 X 的行为**逐位相同**，所以零攻角路径结果不变。
            有攻角时必须传真实方向：迎风投影面积是"垂直于**来流**的
            投影"，按 X 投影会系统性偏大（偏差因子 1/cos(alpha)，
            15 度攻角就是 3.5%，直接进 Cd 的分母）。

    Returns:
        参考面积 (m^2)，计算失败返回 None
    """
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
        logger.debug("Auto reference area: surface_mesh is None")
        return None

    try:
        surface_nodes = surface_mesh.get('nodes')
        surface_faces = surface_mesh.get('faces')
        surface_boundaries = surface_mesh.get('boundaries')

        if surface_nodes is None or surface_faces is None or surface_boundaries is None:
            logger.debug(f"Auto reference area: missing data - nodes={surface_nodes is not None}, faces={surface_faces is not None}, boundaries={surface_boundaries is not None}")
            return None

        # 获取面网格边界名称
        all_boundary_names = list(surface_boundaries.boundary_names)
        logger.debug(f"Auto reference area: surface mesh boundaries = {all_boundary_names}")

        # 查找车身边界面（BODY/CAR/WALL，排除 INLET/OUTLET/SYMMETRY 等）
        body_boundary_names = [
            name for name in all_boundary_names
            if ('BODY' in name.upper() or 'CAR' in name.upper() or 'WALL' in name.upper())
            and 'INLET' not in name.upper() and 'OUTLET' not in name.upper() and 'SYMMETRY' not in name.upper()
        ]

        if not body_boundary_names:
            logger.debug(f"Auto reference area: no body boundary found in {all_boundary_names}")
            return None

        # 收集车身面索引
        body_face_indices = []
        for boundary_name in body_boundary_names:
            face_indices = surface_boundaries.get_cell_indices(boundary_name)
            body_face_indices.extend(face_indices)

        if len(body_face_indices) == 0:
            return None

        body_face_indices = np.array(body_face_indices, dtype=np.int64)

        # 取车身面的节点坐标
        v0 = surface_nodes[surface_faces[body_face_indices, 0]]
        v1 = surface_nodes[surface_faces[body_face_indices, 1]]
        v2 = surface_nodes[surface_faces[body_face_indices, 2]]

        # 计算面法向量和面积
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        areas = 0.5 * np.linalg.norm(normals, axis=1)

        # 归一化法向量
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        unit_normals = normals / norms

        # 沿**来流方向**的投影面积（迎风面：法向与来流方向的点积 < 0）
        d_component = unit_normals @ d
        upstream_mask = d_component < 0
        projected_areas = -d_component[upstream_mask] * areas[upstream_mask]
        ref_area = np.sum(projected_areas)

        if ref_area <= 0 or not np.isfinite(ref_area):
            # 兜底：用绝对投影除以 2（适用于对称车身）
            projected_areas_all = np.abs(d_component) * areas
            ref_area = np.sum(projected_areas_all) / 2.0

        if ref_area > 0 and np.isfinite(ref_area):
            logger.info(
                f"Auto-computed reference area (projected along freestream "
                f"direction {np.round(d, 6).tolist()}): {ref_area:.6f} m^2")
            return float(ref_area)

        return None

    except Exception as e:
        logger.warning(f"Failed to auto-compute reference area: {e}")
        return None


def _report_aerodynamic_coefficients(solver, reference_area: Optional[float]) -> None:
    """求解结束后直接在当前 FRSolver 状态上积分并打印 Cd/Cl。

    不经过 checkpoint/post 命令组的往返（那条路径此前完全打不通，见
    postprocess/fr_coefficients.py 模块文档），直接用求解器仍在内存里的
    mesh+state 计算——这是让 CLI 验收标准"Cd/Cl 非零、符号正确"能够
    兑现的最短路径。reference_area 未提供时跳过（不猜一个可能误导的
    默认值）。
    """
    if reference_area is None or reference_area <= 0:
        print("\n(未提供 --reference-area，跳过气动系数计算；如需 Cd/Cl 请指定参考面积)")
        return
    try:
        from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr

        coeffs = compute_aerodynamic_coefficients_fr(solver, reference_area=reference_area)
        print(f"\n=== Aerodynamic Coefficients (reference_area={reference_area} m^2) ===")
        print(f"   Cd (drag)  = {coeffs.Cd:.6f}")
        print(f"   Cl (lift)  = {coeffs.Cl:.6f}")
        print(f"   Cs (side)  = {coeffs.Cs:.6f}")
        # 力矩系数基于默认参考长度 1.0 m 与原点的力矩参考点（函数默认值）
        print(f"   Cm (pitch) = {coeffs.Cm:.6f}")
        print(f"   Cy (yaw)   = {coeffs.Cy:.6f}")
        print(f"   Cr (roll)  = {coeffs.Cr:.6f}")
    except Exception as e:
        print(f"\n⚠️  Aerodynamic coefficient calculation failed: {e}")
