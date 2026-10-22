"""分层挤出（`mesh_extrusion.extrude_layers`）-> 边界层三棱柱单元（`convert_layers_to_prisms`），以及四面体统一
定向（`orient_tetrahedra`）。"""

from typing import List

import numpy as np
from loguru import logger

from .mesh_tetgen_postprocess import DEGENERATE_VOLUME_FRACTION


def orient_tetrahedra(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """翻转倒置的四面体，使每个单元具有正 signed 体积。

    Signed volume = det(p1-p0, p2-p0, p3-p0) / 6。交换两个顶点会
    翻转符号，所以倒置的单元被就地修复。精确退化（零体积）的单元
    无法修复，会被报告。

    Args:
        nodes: 节点坐标，shape=(n_nodes, 3)
        tets: 四面体连接关系，shape=(n_tets, 4)

    Returns:
        所有 signed 体积 >= 0 的连接关系。
    """
    p0 = nodes[tets[:, 0]]
    p1 = nodes[tets[:, 1]]
    p2 = nodes[tets[:, 2]]
    p3 = nodes[tets[:, 3]]
    det = np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))

    inverted = det < 0.0
    n_inv = int(np.count_nonzero(inverted))
    if n_inv:
        # 交换最后两个顶点以恢复正方向。
        tets[inverted, 2], tets[inverted, 3] = (
            tets[inverted, 3].copy(), tets[inverted, 2].copy()
        )
        logger.info(f"Re-oriented {n_inv} inverted tetrahedra")

    n_degen = int(np.count_nonzero(np.abs(det) < 1e-20))
    if n_degen:
        logger.warning(
            f"{n_degen} degenerate (zero-volume) tetrahedra detected; these "
            f"cannot be fixed by re-orientation and indicate collapsed layers"
        )

    return tets


def convert_layers_to_prisms(
    all_nodes: np.ndarray,
    layer_connectivity: List[np.ndarray],
    base_faces: np.ndarray,
    min_cell_size: float,
) -> np.ndarray:
    """将分层挤出转换为三角棱柱单元。

    每个（层，底面）对发出一个棱柱，顶点按全局节点索引排序（v0<v1<v2，w_i 为上一层对应顶点）。侧面四边形的拆分
    只依赖它自己的四个全局编号（`mesh_gen/utils/prism_split.py`），相邻棱柱、BL / 核心界面处相邻的四面体因此
    自动协调，不需要跨单元协调。

    Args:
        all_nodes: 所有层的节点，shape=(total_nodes, 3)
        layer_connectivity: 每个挤出**步**一个条目（n_layers 个节点层对应 n_layers - 1 个条目：
            `extrude_layers` 的节点数组从 layer-0 块开始、每步追加一块）。只使用其长度——每层共享 base_faces
            的局部面拓扑，仅按 nodes_per_layer 偏移。
        base_faces: 原始表面面，shape=(n_faces, 3)
        min_cell_size: 目标单元尺寸（米），退化体积丢弃阈值 `(min_cell_size**3) * DEGENERATE_VOLUME_FRACTION`
            （为什么按尺寸缩放见该常量）。

    Returns:
        prisms：棱柱连接关系，shape=(n_prisms, 6)，格式 (v0,v1,v2,w0,w1,w2)
        （n_prisms 可能少于 n_base_faces*(n_layers-1)——体积相对于上述退化体积
        阈值可忽略的棱柱（无论是因 taper_scale 为 0 将整个底面
        坍缩到零厚度，还是——见该阈值自身的注释——整体面停滞
        到接近但不精确为零的位置）都被丢弃。只有一条垂直边
        接近零而另外两条仍正常增长的棱柱（真正的楔形，不是停滞）
        不会被触碰——其体积仍然与正常棱柱相当，无论 min_cell_size
        如何都远高于此阈值，这是构造保证的（丢弃它会在外边界上
        撕出真实的洞，与完全坍缩的情况不同：本项目自身的 Part12
        历史充满了因过于激进地删除单元而导致的这类缺陷）。完全
        坍缩的丢弃留下的坐标重复但索引不同的 seam 由调用方的
        重合点合并传递清理，与等效的四面体情况相同。
    """
    # +1：layer_connectivity 每个挤出步一个条目，不是每个节点层（少数一层会让 nodes_per_layer 算大，把一层的
    # 顶点连到几层之外，曾在真实案例上产生横跨全域约 14 m^3 的单元）。
    n_layers = len(layer_connectivity) + 1
    n_base_faces = len(base_faces)

    if n_layers < 2:
        raise ValueError("Need at least 2 layers to create volume")

    n_total_nodes = len(all_nodes)
    nodes_per_layer = n_total_nodes // n_layers

    logger.info(f"Converting {n_layers-1} layer pairs to {n_base_faces} boundary-layer prism(s) each...")

    sorted_base = np.sort(base_faces, axis=1)  # (n_faces, 3) -> v0<v1<v2, same per layer

    n_prisms = n_base_faces * (n_layers - 1)
    prisms = np.empty((n_prisms, 6), dtype=np.int64)

    prism_idx = 0
    for layer_idx in range(n_layers - 1):
        off_lo = layer_idx * nodes_per_layer
        off_hi = (layer_idx + 1) * nodes_per_layer
        sl = slice(prism_idx, prism_idx + n_base_faces)
        prisms[sl, 0] = off_lo + sorted_base[:, 0]
        prisms[sl, 1] = off_lo + sorted_base[:, 1]
        prisms[sl, 2] = off_lo + sorted_base[:, 2]
        prisms[sl, 3] = off_hi + sorted_base[:, 0]
        prisms[sl, 4] = off_hi + sorted_base[:, 1]
        prisms[sl, 5] = off_hi + sorted_base[:, 2]
        prism_idx += n_base_faces

    # 丢弃整个底面完全坍缩的棱柱：阈值为何按 min_cell_size 缩放而不用固定的浮点噪声量，见
    # DEGENERATE_VOLUME_FRACTION 的注释；为何按体积（而不是逐条棱）判断才能保住真正的楔形棱柱，见本函数
    # Returns 文档。
    from ...validation.quality_metrics import compute_prism_volumes
    volumes = compute_prism_volumes(all_nodes, prisms)
    degenerate_threshold = (min_cell_size ** 3) * DEGENERATE_VOLUME_FRACTION
    drop = volumes < degenerate_threshold

    n_dropped = int(np.count_nonzero(drop))
    if n_dropped:
        logger.info(
            f"Dropped {n_dropped} fully-collapsed prisms (volume < "
            f"{degenerate_threshold:.3e} m^3, an entire base face stalled)"
        )
        keep = ~drop
        prisms = prisms[keep]

    logger.info(f"Total prisms generated: {len(prisms)}")
    return prisms
