"""生成后体网格修复阶段 A：质量门控平滑。

阶段 A（本模块主入口 `smooth_bad_cells`）：质量门控的拉普拉斯平滑，只
作用于不在任何物理边界面（车身/隧道/进口/出口）上、也不在 BL/core 交界
面上的节点——覆盖 BL 内部层节点和 core 区域 tetgen 的 Steiner 点，因为
这两类节点既不承载物理几何含义，也不像交界面那样对已生成的相邻几何有
支撑作用（见 compute_movable_node_mask 自己的文档字符串）。

阶段 B（BL 侧厚度封顶）和阶段 B'（局部 cavity 重新铺网）在
mesh_repair_cavity.py 里，纯粹为了控制本文件行数才拆出去——在本文件底部
重新转出，让现有调用方（`from .mesh_repair import smooth_bad_cells,
compute_bl_thickness_limit_override, remesh_core_cavity`）不受影响。完整
的阶段 B/B' 说明见 mesh_repair_cavity.py 自己的模块文档字符串，包括
阶段 B 之前一个基于 core 侧区域的对应方案为什么被移除（实际效果是净负面
的——tetgen 按区域细化会向外泄漏，见 mesh_background.py 自己的历史记录），
以及阶段 B' 为什么现在也覆盖 BL 单元，而不只是纯 core 单元。
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...schema.grid_faces import FaceData
    from ...validation.quality_validator import MeshQualityValidator


def compute_movable_node_mask(
    n_nodes: int, faces: 'FaceData', n_bl_cells: Optional[int] = None,
) -> np.ndarray:
    """阶段 A 可安全移动的节点：不属于任何边界面，且（当给出 `n_bl_cells` 时）
    不在 BL/core 接口任一侧。

    接口最初也被视为可自由移动，理由是它是"合并后的内部网格接缝，
    而非物理边界"——这没错，但不完整：不同于普通内部节点，接口节点
    对两块独立构建且从未相互协调的已 finalized 几何起承载作用——
    BL 侧的挤出（mesh_front_collision.py 的反应式检查已保证边是自洽的）
    和 tetgen 的核心填充，后者将接口的位置作为固定的 PLC 边界约束
    并据此三角化整个核心体积。在平滑期间移动接口节点会改善接触它的
    BL 单元，但留下另一侧的核心四面体仍按节点的旧位置构建——已直接在
    cube_demo 上确认为真实、可复现的缺陷：BL 单元和核心单元完全没有
    共享节点（真正不相邻）却在空间上重叠，每个单独案例都可追溯到
    正好是这个不匹配。排除这些节点是保守修复——一些形状错误源于
    接口节点位置的 BL 单元可能不被平滑，但 CRITICAL 级别的重叠比
    HIGH 级别的偏斜度/正交性警告更严重，阶段 B'/C 仍可在不触碰
    接口的情况下处理后者。

    Args:
        n_nodes: 总节点数
        faces: 当前 (nodes, cells) 几何的 FaceData
        n_bl_cells: 可选——如果给出，单元索引 [0, n_bl_cells) 被视为
            BL 来源，其余为核心来源（与 mesh_background._build_merged_mesh
            自身的约定一致：BL 单元在前，核心单元追加在后）。None（默认）
            完全跳过接口排除——仅对没有任何 BL 区域的调用方安全。
    """
    if faces.node_connectivity is None:
        raise ValueError(
            "faces.node_connectivity is required (see FaceExtractor.extract_faces) "
            "to determine which nodes lie on a physical boundary"
        )
    boundary_face_idx = faces.get_boundary_face_indices()
    boundary_nodes = np.unique(faces.node_connectivity[boundary_face_idx].ravel())
    movable = np.ones(n_nodes, dtype=bool)
    movable[boundary_nodes] = False

    if n_bl_cells is not None:
        owner = faces.connectivity[:, 0]
        neighbor = faces.connectivity[:, 1]
        interior = neighbor >= 0
        crosses_interface = interior & ((owner < n_bl_cells) != (neighbor < n_bl_cells))
        interface_face_idx = np.flatnonzero(crosses_interface)
        if len(interface_face_idx):
            interface_nodes = np.unique(faces.node_connectivity[interface_face_idx].ravel())
            movable[interface_nodes] = False

    return movable


def _bad_cell_mask(
    validator: 'MeshQualityValidator',
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: 'FaceData',
    extra_bad_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """哪些单元触发了偏斜度/正交性/相邻体积比阈值——与 MeshQualityValidator
    门控的相同三项检查，按单元评估而非聚合。

    Args:
        extra_bad_mask: 可选 (n_cells,) bool 数组，或入结果——例如
            与不同、不相邻单元的物理重叠中涉及的单元（mesh_overlap_check.py）。
            重叠是此遍开始前的网格的静态、一次性计算事实（每次平滑传递都
            重新计算宽相空间搜索会不必要地昂贵——参见 smooth_bad_cells 自身
            文档）；在此折叠意味着早期重叠检查标记的单元在阶段 A 运行期间
            持续被视为合法平滑候选，且——如果仍在 smooth_bad_cells 返回的
            掩码中为坏——也是合法的阶段 B' 空腔重铺候选，与其他任何坏单元
            相同。
    """
    n_cells = len(cells)
    bad = np.zeros(n_cells, dtype=bool)

    skew = validator.compute_cell_skewness(nodes, cells)
    bad |= skew > validator.thresholds['max_skewness']

    diag = validator.compute_face_diagnostics(nodes, cells, faces)
    if len(diag['angle_deg']) > 0:
        face_bad = (
            (diag['angle_deg'] > validator.thresholds['max_orthogonality_angle'])
            | (diag['volume_ratio'] > validator.thresholds['max_adjacent_volume_ratio'])
        )
        bad[diag['owner'][face_bad]] = True
        bad[diag['neighbor'][face_bad]] = True

    if extra_bad_mask is not None:
        bad |= extra_bad_mask

    return bad


def _node_target_positions(nodes: np.ndarray, cells: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """每个节点的入射单元质心的加权平均（经典"smart Laplacian"平滑目标）
    ——向量化 scatter-sum，与 mesh_extrusion.py 的逐节点法向平均相同模式。"""
    centroids = nodes[cells].mean(axis=1)
    n_nodes = len(nodes)
    node_sum = np.zeros((n_nodes, 3))
    node_weight = np.zeros(n_nodes)
    flat_nodes = cells.ravel()
    flat_centroids = np.repeat(centroids, cells.shape[1], axis=0)
    flat_weights = np.repeat(np.maximum(weights, 1e-300), cells.shape[1])
    np.add.at(node_sum, flat_nodes, flat_centroids * flat_weights[:, None])
    np.add.at(node_weight, flat_nodes, flat_weights)

    target = nodes.copy()
    has_weight = node_weight > 0
    target[has_weight] = node_sum[has_weight] / node_weight[has_weight, None]
    return target


def smooth_bad_cells(
    nodes: np.ndarray,
    cells: np.ndarray,
    validator: 'MeshQualityValidator',
    max_passes: int = 5,
    initial_faces: Optional['FaceData'] = None,
    extra_bad_mask: Optional[np.ndarray] = None,
    n_bl_cells: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """阶段 A：对偏斜/非正交/体积不匹配单元的质量门控拉普拉斯平滑，
    限制为可移动（非边界）节点。

    每遍：识别坏单元 -> 收集其可移动节点 -> 为每个提议体积加权质心目标
    位置 -> 同时应用所有提议移动 -> 如果这引入任何负体积单元，将位移
    减半并重试（最多 4 次减半）而非丢弃整遍——smart Laplacian 平滑的
    标准"松弛 + 行搜索阻尼"简化：比严格的逐节点接受/拒绝方案便宜
    （后者需要在每个单节点移动后重新验证整个受影响邻域），同时保持
    其核心安全保证——一遍仅在其引入零个负体积单元时才被提交。

    Args:
        nodes: (n_nodes, 3) float64，不被修改——返回新数组
        cells: (n_cells, 4) int32 四面体连接关系
        validator: MeshQualityValidator 实例（重用于其阈值配置和
            逐单元/逐面诊断方法）
        max_passes: 无论结果如何，达到此遍数后停止
        initial_faces: 可选，已从此确切 (nodes, cells) 对提取的 FaceData
            （例如由刚在同一未移动几何上运行自身修复前 validate() 的
            调用方传入）——仅用于第 0 遍，代替从头重新提取，因为第 0 遍
            评估的是任何节点移动前的网格。在大型网格上的实际节省：面
            提取是非平凡成本，且常见情况是第 0 遍根本没找到安全移动
            （无需平滑），此时这是阶段 A 否则会相对于调用方自身修复前
            检查冗余重复的唯一提取。
        extra_bad_mask: 可选 (n_cells,) bool 数组，无论本遍偏斜/正交/
            体积比检查说什么，都始终视为坏并处理的额外单元——参见
            _bad_cell_mask 自身文档（例如 mesh_overlap_check.py 标记的
            单元）。全程针对原始单元索引评估——安全因为此函数从不
            添加/删除单元，只移动节点位置。
        n_bl_cells: 可选——单元索引 [0, n_bl_cells) 为 BL 来源，其余为
            核心来源（参见 compute_movable_node_mask 自身文档字符串了解
            为何这将 BL/core 接口排除在平滑之外）。None（默认）使接口
            可移动——仅对没有任何 BL 区域的调用方正确。

    Returns:
        (new_nodes, bad_cell_mask_after, action_log) - bad_cell_mask_after
        在最终几何上重新评估（如果网格开始时完全没有坏单元因此没有运行
        任何遍，则为空数组）。
    """
    from ...schema.grid_nodes import NodeArray
    from ..extraction.face_extractor import FaceExtractor

    nodes = nodes.copy()
    actions: List[str] = []
    bad_mask = np.zeros(len(cells), dtype=bool)

    for pass_idx in range(max_passes):
        if pass_idx == 0 and initial_faces is not None:
            faces = initial_faces
        else:
            node_arr = NodeArray.from_array(nodes)
            faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)
        movable_mask = compute_movable_node_mask(len(nodes), faces, n_bl_cells)

        bad_mask = _bad_cell_mask(validator, nodes, cells, faces, extra_bad_mask=extra_bad_mask)
        if not np.any(bad_mask):
            if pass_idx == 0:
                actions.append("Stage A: mesh already within thresholds, no smoothing needed")
            break

        candidate_mask = np.zeros(len(nodes), dtype=bool)
        candidate_mask[cells[bad_mask].ravel()] = True
        candidate_mask &= movable_mask

        n_bad = int(np.sum(bad_mask))
        if not np.any(candidate_mask):
            actions.append(
                f"Stage A pass {pass_idx + 1}: {n_bad} bad cells remain, but none of "
                f"their nodes are movable (all on a physical boundary) - stopping"
            )
            break

        current_volumes = validator._compute_tetrahedron_volumes(nodes, cells)
        target = _node_target_positions(nodes, cells, np.abs(current_volumes))

        # 安全判据是"永远不将当前有效单元变为负"，
        # 而非"保证网格级零负单元"——后者在此遍之前某处
        # （例如某处已退化的单元未被此移动触及）已使网格处于
        # 该状态时永远无法满足，那会阻止阶段 A 修复任何东西，
        # 包括它能合法改善的单元。
        already_bad = current_volumes <= 0

        relax = 1.0
        accepted = False
        nodes_trial = nodes
        for _damp_iter in range(4):
            nodes_trial = nodes.copy()
            nodes_trial[candidate_mask] = (
                nodes[candidate_mask] + relax * (target[candidate_mask] - nodes[candidate_mask])
            )
            trial_volumes = validator._compute_tetrahedron_volumes(nodes_trial, cells)
            newly_negative = (trial_volumes <= 0) & ~already_bad
            if not np.any(newly_negative):
                accepted = True
                break
            relax *= 0.5

        if not accepted:
            actions.append(
                f"Stage A pass {pass_idx + 1}: no safe move found for {int(np.sum(candidate_mask))} "
                f"candidate nodes even after damping - stopping"
            )
            break

        n_moved = int(np.sum(candidate_mask))
        nodes = nodes_trial
        actions.append(
            f"Stage A pass {pass_idx + 1}: {n_bad} bad cells -> moved {n_moved} nodes "
            f"(relax={relax:.3f})"
        )

    else:
        actions.append(f"Stage A: reached max_passes={max_passes} limit")

    return nodes, bad_mask, actions


# 阶段 B / B' - 参见 mesh_repair_bl_thickness.py / mesh_repair_cavity.py。
# 在此重新导出，使 `from .mesh_repair import ...` 对现有调用方
# （mesh_background.py、tests/unit/test_mesh_repair.py）继续有效，无需修改。
from .mesh_repair_bl_thickness import compute_bl_thickness_limit_override  # noqa: E402
from .mesh_repair_cavity import remesh_core_cavity  # noqa: E402
