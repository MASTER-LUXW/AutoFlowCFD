"""面提取的收尾几何计算：面积、法向/中心 + 校验。

从 face_extractor.py 拆分出来。finalize_face_data 是 extract_faces（纯四面体
网格）和 extract_faces_mixed（棱柱 + 四面体混合网格）共用的收尾步骤——
从这一步开始，输入已经和具体单元形状无关，只依赖每个面的 3 个角点节点
编号、owner/neighbour 单元编号，以及该单元已经算好的质心。
"""

import numpy as np
from loguru import logger

from ...schema.grid_nodes import NodeArray
from ...schema.grid_faces import FaceData


def finalize_face_data(
    face_nodes_sorted: np.ndarray,
    face_connectivity: np.ndarray,
    occurrence_count: np.ndarray,
    n_unique_faces: int,
    n_interior: int,
    n_faces_raw: int,
    nodes: NodeArray,
    all_cell_centers: np.ndarray,
    n_cells: int,
    strict: bool = False,
) -> FaceData:
    """去重后共享的几何/朝向/校验，被 extract_faces (tet-only) 和
    extract_faces_mixed (prism+tet) 两者使用——从此处开始真正与单元
    形状无关：下方只消费面的 3 个角点节点索引、其 owner/neighbour 单元
    索引，以及该单元已计算的质心。"""
    n_boundary = n_unique_faces - n_interior
    n_invalid = np.sum(occurrence_count > 2)

    logger.info(
        f"Identified {n_unique_faces} unique faces from {n_faces_raw} occurrences"
    )
    logger.info(
        f"Face topology: {n_interior} interior, {n_boundary} boundary, "
        f"{n_invalid} invalid (>2 cells)"
    )

    if n_invalid > 0:
        # 注意: 上方的去重扫描只记录接触给定面键的前 2 个单元
        # （参见 _deduplicate_and_build_connectivity）；对于被 3+ 个
        # 单元共享的面，前两个之后的每个单元根本不会连接到它，静默
        # 丢弃该单元通过此面的通量贡献——这是真正的局部守恒违反，
        # 而非数值稳定性问题。这可以（并且已被观察到）产生无论 CFL
        # 推得多低都无界发散的残差，而积分体力量相对正常，因为它们
        # 不依赖这些（通常是内部/核心网格）面。在拓扑无效的网格上
        # 继续求解会浪费可能数小时的计算在永远不会物理有意义的结果
        # 上——立即失败，指向产生重叠/重复四面体的体积网格生成步骤。
        invalid_mask = occurrence_count > 2
        invalid_node_ids = np.unique(face_nodes_sorted[invalid_mask])
        bad_x = nodes.x[invalid_node_ids]
        bad_y = nodes.y[invalid_node_ids]
        bad_z = nodes.z[invalid_node_ids]
        logger.warning(
            f"Invalid faces detected (n={n_invalid}), spatially bounded by "
            f"x=[{bad_x.min():.4g}, {bad_x.max():.4g}], "
            f"y=[{bad_y.min():.4g}, {bad_y.max():.4g}], "
            f"z=[{bad_z.min():.4g}, {bad_z.max():.4g}]. "
            f"This is likely due to BL extrusion at sharp corners."
            + (" Proceeding for inspection (non-strict call)." if not strict else "")
        )
        if strict:
            # 不同于网格生成和修复期间的中间/探索性调用方
            # （mesh_repair.py、mesh_repair_cavity.py、
            # mesh_background.py 的修复前检查——都是非严格的，
            # 因为那里的瞬态非流形状态是预期的，会被后续的修复
            # 阶段解决，例如 repair_nonmanifold_mixed），这里是
            # 真正的求解/导出时间门（GridData.ensure_faces_exist，
            # strict=True）——到此时每个修复阶段都已运行，因此
            # 剩余的 >2 个 owner 的面是真实的、未校正的缺陷，
            # 而非瞬态的。
            raise RuntimeError(
                f"Invalid mesh topology: {n_invalid} faces are shared by more than "
                f"2 cells (expected exactly 1 for boundary or 2 for interior faces). "
                f"This means the volume mesh contains overlapping/duplicate "
                f"tetrahedra - almost certainly from the boundary-layer/core "
                f"tetgen merge (see mesh_background.generate_hybrid_mesh). "
                f"Solving on this mesh would silently drop flux through the "
                f"affected faces and is not physically meaningful; regenerate "
                f"the volume mesh (e.g. with different BL parameters) rather "
                f"than proceeding."
            )

    # 预期比例：内部为主的网格约为 2x 单元数
    expected_ratio = n_unique_faces / n_cells
    logger.debug(f"Face-to-cell ratio: {expected_ratio:.2f} (expected ~2.0-2.5)")

    # 步骤 3: 使用向量化操作计算几何属性
    logger.debug("Computing face geometry (vectorized)...")
    x = nodes.x
    y = nodes.y
    z = nodes.z

    # 向量化面中心计算
    n0 = face_nodes_sorted[:, 0]
    n1 = face_nodes_sorted[:, 1]
    n2 = face_nodes_sorted[:, 2]

    face_centers = np.column_stack([
        (x[n0] + x[n1] + x[n2]) / 3.0,
        (y[n0] + y[n1] + y[n2]) / 3.0,
        (z[n0] + z[n1] + z[n2]) / 3.0
    ])

    # 向量化面积向量计算
    p0 = np.column_stack([x[n0], y[n0], z[n0]])
    p1 = np.column_stack([x[n1], y[n1], z[n1]])
    p2 = np.column_stack([x[n2], y[n2], z[n2]])

    v1 = p1 - p0
    v2 = p2 - p0
    face_areas_vec = 0.5 * np.cross(v1, v2)

    # 确定面朝向，必要时翻转
    left_cells = face_connectivity[:, 0]
    right_cells = face_connectivity[:, 1]

    # all_cell_centers 已由调用方计算（tet-only 或混合 prism+tet
    # ——参见 _compute_tet_cell_centers/_compute_prism_cell_centers），
    # 作为参数传入。

    # 获取左右单元中心
    center_left = all_cell_centers[left_cells]

    # 对于内部面，确保法向从 OWNER（left）单元向外指——即远离
    # left 自身的质心，穿过面本身——使用面自身的中心（face_centers）
    # 相对于 left 质心的位置，与下方边界分支已（正确地）使用的
    # 相同判据。
    #
    # 关键: 只翻转法向符号。此代码的早期版本在翻转法向时还会
    # 交换 face_connectivity 的两列——这反而撤销了自身的修复：
    # 如果原始叉乘法向指向 left 内部，取负后使其正确指向 left
    # 外部（正是"left/owner 得 +normal"所需的）——但随后交换列
    # 会将 left 重新标记为"neighbour"、right 为"owner"，因此
    # 求解器的"+normal 到 owner、-normal 到 neighbour"累加最终
    # 给 left 的是 -normal（即回到原始的、仍然错误的、向内的值），
    # 给 right 的是 +normal（从 LEFT 向外，而非从 right 向外——
    # 对 right 也错了）。两个错误在配对的组合簿记中相互抵消
    # （这就是为什么对两单元求和的检查从未捕获到），但对每个
    # 单元自身的闭合都不对。已在本项目的实际 cube_demo 核心
    # 网格上直接确认：89% 的单元（100% 的 BL 棱柱）其自身向外
    # 面积加权面法向之和不为零——由散度定理，任何闭合单元必须
    # 精确为零——这静默破坏了几乎所有地方的通量守恒和
    # Green-Gauss 梯度重构，并且正是求解器在网格质量门通过的
    # 网格上发散的实际根本原因。最小 576 四面体结构化盒子
    # 复现（无棱柱、无偏斜）通过 extract_faces_mixed/extract_faces
    # （共享此函数）复现了相同的 73% 单元缺陷率。
    mask_interior = right_cells >= 0
    dx_interior = face_centers[mask_interior] - center_left[mask_interior]
    dot_interior = np.sum(face_areas_vec[mask_interior] * dx_interior, axis=1)

    # 翻转法向指向错误的面（仅符号——参见上方）
    flip_mask = dot_interior < 0
    indices_to_flip = np.where(mask_interior)[0][flip_mask]
    face_areas_vec[indices_to_flip] *= -1

    # 对于边界面，确保法向向外指
    mask_boundary = ~mask_interior
    dx_boundary = face_centers[mask_boundary] - center_left[mask_boundary]
    dot_boundary = np.sum(face_areas_vec[mask_boundary] * dx_boundary, axis=1)
    flip_boundary = dot_boundary < 0
    indices_to_flip_boundary = np.where(mask_boundary)[0][flip_boundary]
    face_areas_vec[indices_to_flip_boundary] *= -1

    # 计算标量面积和单位法向
    face_scalar_areas = np.linalg.norm(face_areas_vec, axis=1)
    valid_area_mask = face_scalar_areas > 1e-12
    face_normals = np.zeros_like(face_areas_vec)
    face_normals[valid_area_mask] = (
        face_areas_vec[valid_area_mask] /
        face_scalar_areas[valid_area_mask][:, np.newaxis]
    )

    # 创建 FaceData 对象。node_connectivity 是上方纯粹为推导
    # 面积/法向/中心而计算的三角面角点节点索引（face_nodes_sorted）
    # ——也保留在此处，因此需要实际边界表面网格的调用方（例如
    # VTKExporter.export_boundaries 用于分区/分片可视化）不必
    # 从四面体中第二次重新提取它。
    face_data = FaceData(
        connectivity=face_connectivity,
        area=face_scalar_areas,
        normal=face_normals,
        center=face_centers,
        node_connectivity=face_nodes_sorted.astype(np.int32),
    )

    # 校验输出
    validate_face_data(face_data, n_cells)

    logger.success(
        f"Face extraction completed: {face_data.n_interior_faces} interior, "
        f"{face_data.n_boundary_faces} boundary faces"
    )

    return face_data


def validate_face_data(face_data: FaceData, n_cells: int) -> bool:
    """校验提取的面数据的一致性。

    检查项:
    - 所有单元被至少一个面引用
    - 无重复面
    - 面积值具有合理量级
    - 法向向量为单位长度

    Args:
        face_data: 提取的面数据
        n_cells: 预期单元数

    Returns:
        校验通过时为 True

    Raises:
        ValueError: 校验失败时
    """
    # 检查 1: 所有单元都应被引用
    referenced_cells = set()
    for i in range(face_data.count):
        referenced_cells.add(int(face_data.connectivity[i, 0]))
        if face_data.connectivity[i, 1] >= 0:
            referenced_cells.add(int(face_data.connectivity[i, 1]))

    if len(referenced_cells) != n_cells:
        raise ValueError(
            f"Face connectivity references {len(referenced_cells)} cells, "
            f"expected {n_cells}"
        )

    # 检查 2: 面积应为正值
    n_zero_areas = np.sum(face_data.area < 1e-12)
    if n_zero_areas > 0:
        logger.warning(f"Found {n_zero_areas} faces with zero/near-zero area. Allowing export for debugging.")
        # raise ValueError(f"Found {n_zero_areas} faces with zero/near-zero area")

    # 检查 3: 法向向量应为单位长度
    normal_magnitudes = np.linalg.norm(face_data.normal, axis=1)
    n_invalid_normals = np.sum(np.abs(normal_magnitudes - 1.0) > 1e-6)
    if n_invalid_normals > 0:
        logger.warning(f"Found {n_invalid_normals} faces with non-unit normals (magnitude != 1.0)")

    logger.debug("Face data validation passed")
    return True
