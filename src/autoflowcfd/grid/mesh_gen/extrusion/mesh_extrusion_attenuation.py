"""BL 挤出用的两种尖锐特征衰减启发式。

从 mesh_extrusion.py 拆分出来：`_compute_sharp_angle_attenuation`（按节点
自身最尖锐的二面角直接衰减）和 `_compute_edge_distance_field`（按到最近
尖锐边的欧氏距离衰减）。extrude_layers 取两者的逐节点最小值合并使用——
单独任何一个都不足以在稀疏网格化的圆角处可靠地衰减。
"""

from typing import Optional

import numpy as np

# _compute_edge_distance_field 自身衰减的地板，应用于尖锐边顶点自身
# （距离 == 0）。该函数之前完全没有地板（简单的 `dists / (2*char_length)`，
# 裁剪到 [0, 1]），所以任何实际在尖锐边上的节点衰减到精确的 0——通过
# np.minimum 与 _compute_sharp_angle_attenuation 组合，这意味着跟踪物体
# 每条尖锐边的整个节点缝在每层 BL 中几乎不挤出，无论 bl_layers 或
# growth_rate：不是逐渐衰减，而是 BL 覆盖的近乎完全局部坍缩，恰好在汽车
# CFD 最需要良好近壁分辨率的地方（特征线、扰流板/后视镜/底板边——所有
# 分离倾向特征）。0.2 匹配 _compute_sharp_angle_attenuation 自身对简单
# 90 度边的值（其 0.2-1.0 线性斜坡在 90-150 度二面角范围内从精确的 0.2
# 开始），所以两个机制现在在边自身一致，而不是距离场通过它们的 np.minimum
# 组合默默覆盖角度场自身考虑的地板。
MIN_EDGE_DISTANCE_ATTENUATION = 0.2


def _compute_sharp_angle_attenuation(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    normal_faces: Optional[np.ndarray] = None,
    sharp_angle_threshold: float = 45.0,
) -> np.ndarray:
    """基于局部二面角计算衰减（ANSA 风格）。

    尖角处的节点（例如 90 度边）的挤出厚度将衰减以防止自交叉和畸变。

    Args:
        nodes: 表面节点
        faces: 表面连接（拓扑）
        normals: 面法向
        normal_faces: 对应法向的面子集
        sharp_angle_threshold: 低于此值的角度（偏离平坦 180）被认为是尖锐的

    Returns:
        attenuation: [0, 1] 数组。0 = 无挤出（尖角），1 = 完全挤出。
    """
    n_nodes = len(nodes)
    detect_faces = normal_faces if normal_faces is not None else faces

    # 1. 计算每条边的二面角
    edge_map = {}  # (min_v, max_v) -> list of face indices
    for i, face in enumerate(detect_faces):
        for j in range(3):
            v1, v2 = int(face[j]), int(face[(j + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(i)

    # 每节点跟踪最尖锐的接触边作为最小 dot(n1,n2)
    # （dot=1.0 <-> 法向平行 <-> 平坦延续；dot 值进一步低于 1.0
    # 意味着两个相邻面的法向发散更多，即更尖锐的折叠）。初始化为
    # 1.0（平坦）而不是哨兵值，所以不接触合格 2 面边的节点（孤立/
    # 补丁边界顶点）安全地默认为“平滑”，而不是需要单独处理。
    node_min_cos_angle = np.full(n_nodes, 1.0)

    for (v1, v2), face_indices in edge_map.items():
        if len(face_indices) >= 2:
            # 使用前两个相邻面估计二面角
            n1 = normals[face_indices[0]]
            n2 = normals[face_indices[1]]
            cos_angle = np.dot(n1, n2)  # 1.0 = flat (normals parallel)

            # MIN，不是 max：我们要节点的最尖锐接触边（最小点积/最发散的
            # 法向对）控制其衰减——即使只接触一个尖锐边的节点也应该衰减，
            # 无论它还接触多少平坦边。（从早期版本修复，那个版本在这里取
            # max()，尽管变量自己的名称和这个注释说“min”——那个 bug 意味着
            # 节点仅当接触它的每条边都尖锐时才衰减，在真实几何上几乎从不，
            # 所以这个衰减默默地无效。）
            node_min_cos_angle[v1] = min(node_min_cos_angle[v1], cos_angle)
            node_min_cos_angle[v2] = min(node_min_cos_angle[v2], cos_angle)

    # 2. 将法向到法向角度转换为传统表面二面角（180 度 = 平坦延续，
    # 90 度 = 直角折叠，更尖锐的折叠进一步减小），然后应用下面的平滑/
    # 尖锐阈值，这些阈值用该约定编写。单独 arccos(dot(n1,n2)) 是法向
    # 之间的角度，方向相反（平坦约 0 度，更尖锐更大）——混淆两者是早期
    # 版本中第二个独立的 bug：平坦区域（dot~1，法向角度~0）满足
    # “angle < sharp_limit”并几乎 everywhere 衰减到 0.1，而不仅在真正的
    # 尖锐特征处。
    normal_angle_rad = np.arccos(np.clip(node_min_cos_angle, -1.0, 1.0))
    dihedral_rad = np.pi - normal_angle_rad

    # 定义“尖锐度”范围（用二面角术语）
    smooth_limit = np.radians(150)  # 150 degrees: treated as flat
    sharp_limit = np.radians(90)    # 90 degrees: treated as fully sharp

    attenuation = np.ones(n_nodes)

    # 尖锐区域掩码
    sharp_mask = dihedral_rad < smooth_limit
    if np.any(sharp_mask):
        # 在 sharp_limit (0.2) 和 smooth_limit (1.0) 之间线性插值
        # 这创建从边的平滑过渡
        t = (dihedral_rad[sharp_mask] - sharp_limit) / (smooth_limit - sharp_limit)
        attenuation[sharp_mask] = 0.2 + 0.8 * np.clip(t, 0, 1)

    # 对于非常尖锐的角（< 90 度），保持最小厚度以避免零体积单元
    # 但防止大挤出
    very_sharp_mask = dihedral_rad < sharp_limit
    attenuation[very_sharp_mask] = 0.1

    return attenuation


def _compute_edge_distance_field(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    angle_threshold: float = 45.0,
    normal_faces: Optional[np.ndarray] = None,
    min_feature_radius: float = 0.0,
) -> np.ndarray:
    """计算从每个节点到最近尖锐边的距离场。

    Args:
        nodes: 表面节点，形状=(n_nodes, 3)
        faces: 表面连接，形状=(n_faces, 3) - 用于拓扑
        normals: 面法向，形状=(n_normal_faces, 3) - 必须匹配 `normal_faces`
        angle_threshold: 认为边尖锐的二面角阈值（度）
        normal_faces: 可选 `faces` 子集，`normals` 对应它。如果 None，
                      假设 `normals` 匹配 `faces`。
        min_feature_radius: 与 mesh_corner_split.split_sharp_corners 同名
            参数相同的曲率半径过滤（米）——法向偏差超过 angle_threshold 的边，
            如果其自身几何（弦长 + 偏差角）隐含的局部曲率半径达到或超过
            此值，仍不视为尖锐特征。0.0（默认）保留原始纯角度判据，向后
            兼容未显式传入这个新参数的调用方。

            V2.0 专项攻关记录（cube_demo BL 质量campaign 第八轮）：这个
            过滤器此前只存在于 split_sharp_corners（Multiple-Normals 拆分
            判据），本函数完全没有——cube_demo 实测车身圆角（真实半径
            7.6mm）密集三角化后单个面片间法向偏差普遍 45°~46°，恰好越过
            默认 angle_threshold=45°，但这只是圆角的正常离散化，不是真正
            的尖锐折痕（split_sharp_corners 用同一份判据 + 曲率半径过滤
            正确识别出这一点，因此对 cube_demo 零拆分）。本函数在没有这层
            过滤的情况下把这些"伪尖锐边"误判为真正的尖锐特征，实测导致
            衰减场沿车身整条棱边被压到地板值 0.2（177 个节点，覆盖某条棱
            边 x 方向 0.01~0.49 的几乎全长），直接造成硬停止机制在这些
            节点上过早耗尽预算、边界层层数大幅收缩——这正是"圆角处处
            平滑却没有生成完整层数"这个现象的根因。
    """
    n_nodes = len(nodes)

    # 如果提供则使用 normal_faces 进行边检测，否则使用 faces
    detect_faces = normal_faces if normal_faces is not None else faces

    # 1. 基于面法向识别尖锐边
    # 对每条边，检查相邻面之间的角度
    edge_map = {}  # (min_v, max_v) -> list of face indices (into detect_faces)

    for i, face in enumerate(detect_faces):
        for j in range(3):
            v1, v2 = int(face[j]), int(face[(j + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(i)

    if min_feature_radius > 0.0:
        from ..utils.mesh_corner_split_geometry import _implied_edge_radius

    sharp_edges = set()
    for (v1, v2), face_indices in edge_map.items():
        if len(face_indices) >= 2:
            # 计算二面角
            n1 = normals[face_indices[0]]
            n2 = normals[face_indices[1]]
            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle = np.degrees(angle_rad)

            # 如果角度与 180（平坦）显著不同则为尖锐
            # 对于立方体，我们期望 90 度角
            is_sharp = angle > angle_threshold and angle < (180 - angle_threshold)
            # 见 min_feature_radius 自己的文档：与 split_sharp_corners 相同的
            # "这是真正的尖锐折痕，还是只是一段曲率半径够大的平滑曲面被
            # 密集三角化后单个面片角度恰好越过阈值"区分——用弦长+偏差角
            # 隐含的局部曲率半径判断，不是纯角度判据能区分的。
            if is_sharp and min_feature_radius > 0.0:
                edge_length = float(np.linalg.norm(nodes[v1] - nodes[v2]))
                implied_radius = _implied_edge_radius(edge_length, angle_rad)
                if implied_radius >= min_feature_radius:
                    is_sharp = False
            if is_sharp:
                sharp_edges.add((v1, v2))

    if not sharp_edges:
        return np.ones(n_nodes)

    # 2. 计算从每个节点到最近尖锐边的距离
    # 为简单起见，使用到参与尖锐边的最近顶点的距离
    sharp_vertex_set = set()
    for v1, v2 in sharp_edges:
        sharp_vertex_set.add(v1)
        sharp_vertex_set.add(v2)

    sharp_vertices = nodes[list(sharp_vertex_set)]

    # 使用 KDTree 进行高效最近邻搜索
    from scipy.spatial import cKDTree
    tree = cKDTree(sharp_vertices)
    dists, _ = tree.query(nodes, k=1)

    # 3. 使用平滑阶跃函数将距离转换为衰减
    # 特征长度尺度：表面平均边长
    edge_lengths = []
    for v1, v2 in list(sharp_edges)[:100]: # 样本 for performance
        edge_lengths.append(np.linalg.norm(nodes[v1] - nodes[v2]))
    char_length = np.mean(edge_lengths) if edge_lengths else 0.01

    # 平滑衰减：距离 0 处为 MIN_EDGE_DISTANCE_ATTENUATION
    # （见该常量自己的注释了解为什么存在这个地板），在距离 > 2*char_length
    # 时攀升到 1。
    ramp = np.clip(dists / (2.0 * char_length), 0.0, 1.0)
    attenuation = MIN_EDGE_DISTANCE_ATTENUATION + (1.0 - MIN_EDGE_DISTANCE_ATTENUATION) * ramp

    return attenuation
