"""把分层棱柱网格（来自 mesh_extrusion.extrude_layers）转换成保形的四面体网格。

从 mesh_extrusion.py 拆出（该文件保留分层生成循环
extrude_layers/extrude_single_layer），纯粹为了让两个文件都控制在 450
行以内；两个模块彼此没有依赖关系。
"""

import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

# 与 mesh_background.py 自身退化四面体清理使用的相同的"相对于单元大小"
# 约定（`(min_cell_size ** 3) * 1e-6`）——在此复用，以便一个完全坍缩的
# 棱柱（整个底面的 taper_scale/budget 停滞到接近 0，而不仅是浮点精确的 0）
# 也被识别为退化，而不仅是旧的固定 1e-20 阈值捕获的精确零情况。故意仍然
# 比真正的楔形棱柱体积小很多个数量级（一条垂直边接近零，另外两条正常）——
# 见 convert_layers_to_prisms 的 Returns 文档了解为何这个区分很重要
# （楔形是真实几何，丢弃它会撕出洞；完全坍缩的棱柱不贡献任何东西）。
DEGENERATE_VOLUME_FRACTION = 1e-6


def convert_layers_to_tetrahedra(
    all_nodes: np.ndarray,
    layer_connectivity: List[np.ndarray],
    base_faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将分层棱柱网格转换为保形的四面体网格。

    每两个相邻层之间的三角棱柱被分割为 3 个四面体。分割方式
    使得相邻棱柱在每个共享四边形面的对角线上保持一致，这正是
    结果网格保形（每个内部面恰好被两个单元共享）的原因。盲目
    应用的固定模板不具备这个性质，会产生悬挂面，有限体积求解器
    会将它们误认为边界面。

    规则：按全局节点索引排序三个底面顶点，v0 < v1 < v2，令 w_i
    为下一层上对应的顶点。输出：

        T1 = (v0, v1, v2, w2)
        T2 = (v0, v1, w1, w2)
        T3 = (v0, w0, w1, w2)

    这在三个四边形面上诱导的对角线分别是 v0-w1、v1-w2 和 v0-w2，
    即始终是"低索引底顶点到高索引顶顶点"。该规则仅取决于共享边
    的两个顶点，所以共享一条边的两个棱柱必然选择相同的对角线。

    四面体还被定向为正 signed 体积，然后任何 signed 体积解析为零
    （不仅是小——见下方"丢弃的四面体"）的四面体从输出中完全移除。

    Args:
        all_nodes: 所有层的节点，shape=(total_nodes, 3)
        layer_connectivity: 每个挤出步一个条目（n_layers 个节点层
            对应 n_layers - 1 个条目），与本项目其他地方从
            extrude_layers 生成的连接关系列表推导节点层数的约定
            一致（见 mesh_background_merge._build_merged_mesh 自身的
            `n_layers = len(bl_layer_conn) + 1` 及其文档字符串了解
            原因——extrude_layers.all_layer_nodes 从 layer-0 块开始
            然后每步追加一个块，所以节点数组总是比步数多一层）。
            只使用此列表的长度（其实际的每层面索引内容从不被
            读取——每层共享 base_faces 自身的局部面拓扑，只是按
            nodes_per_layer 偏移）。
        base_faces: 原始表面面，shape=(n_faces, 3)

    Returns:
        (tetrahedra, face_of_tet)：四面体连接关系，shape=(n_tets, 4)
        ——n_tets 可能少于 n_base_faces*(n_layers-1)*3（见下方"丢弃
        的四面体"）；face_of_tet，shape=(n_tets,)，将每个幸存的
        四面体映射回其 base_faces 行索引。之前假设固定平铺
        （n_tets_per_face = n_tets // n_base_faces，单元 i ->
        base_faces[i % n_base_faces]）的调用方现在必须改用
        face_of_tet，因为单元可能被丢弃。

        丢弃的四面体：精确（到浮点噪声，|det| < 1e-20）零体积——
        例如一个棱柱因 seam taper_scale 为 0 而完全坍缩
        （mesh_extrusion.extrude_layers 的 taper_scale），将节点在
        每层固定在其原始位置，所以两个相同层位置之间的棱柱厚度
        为零。这样的四面体的唯一非退化面总是仅与该棱柱自身的
        其他 2 个四面体内部共享（T1 的真实面就是 T2 自身的某个
        面——棱柱自身的内部分裂对角线），从不与外部邻居共享，
        所以丢弃它不丢失真实几何，也不会孤立另一个棱柱仍期望
        匹配的面（经验验证：合并网格自身的保形检查——每个内部
        面恰好被 2 个单元共享——在丢弃后仍然通过）。
    """
    # +1：layer_connectivity 每个挤出步一个条目，不是每个节点层——
    # 见本函数自身的 `layer_connectivity` 文档。在此直接使用
    # len(layer_connectivity)（本函数在此修复前的做法）会恰好少数
    # 一层，进而使 nodes_per_layer = n_total_nodes // n_layers 计算出
    # 比实际每层节点数更大的值——下面的每个 off_lo/off_hi 都会落在
    # 错误的绝对节点索引上，静默将一个层的顶点连接到几层之外
    # 同局部索引的顶点（已直接确认：在真实案例上产生了顶点来自
    # 近壁过渡区域和域自身的入口/出口/远场壁的四面体——一个跨域
    # 全长约 14 m^3 的四面体，而且这不是罕见的异常值：该案例上
    # 约一半的过渡阶段四面体超过了合理的体积上限）。下面的棱柱
    # 对应函数（convert_layers_to_prisms）有相同的修复；其在
    # mesh_background_merge.py 中的调用方用一个临时切片时 +1 来
    # 补偿，掩盖了那里的同一个 bug，但本函数的调用方没有类似的
    # 补偿。
    n_layers = len(layer_connectivity) + 1
    n_base_faces = len(base_faces)

    if n_layers < 2:
        raise ValueError("Need at least 2 layers to create volume")

    n_total_nodes = len(all_nodes)
    nodes_per_layer = n_total_nodes // n_layers

    logger.info(f"Converting {n_layers-1} layer pairs to conformal tetrahedra...")

    # Sort each base triangle's vertices by global index once; the relative
    # order is identical on every layer (index = base + layer*nodes_per_layer),
    # so one sort is valid for the whole stack.
    sorted_base = np.sort(base_faces, axis=1)          # (n_faces, 3) -> v0<v1<v2

    n_tets = n_base_faces * (n_layers - 1) * 3
    tetrahedra = np.empty((n_tets, 4), dtype=np.int64)
    face_of_tet = np.empty(n_tets, dtype=np.int64)
    face_range = np.arange(n_base_faces)

    tet_idx = 0
    for layer_idx in range(n_layers - 1):
        off_lo = layer_idx * nodes_per_layer
        off_hi = (layer_idx + 1) * nodes_per_layer

        v0 = off_lo + sorted_base[:, 0]
        v1 = off_lo + sorted_base[:, 1]
        v2 = off_lo + sorted_base[:, 2]
        w0 = off_hi + sorted_base[:, 0]
        w1 = off_hi + sorted_base[:, 1]
        w2 = off_hi + sorted_base[:, 2]

        for quad in ((v0, v1, v2, w2),
                     (v0, v1, w1, w2),
                     (v0, w0, w1, w2)):
            sl = slice(tet_idx, tet_idx + n_base_faces)
            tetrahedra[sl, 0] = quad[0]
            tetrahedra[sl, 1] = quad[1]
            tetrahedra[sl, 2] = quad[2]
            tetrahedra[sl, 3] = quad[3]
            face_of_tet[sl] = face_range
            tet_idx += n_base_faces

    # Enforce positive signed volume (swap two vertices where inverted) so that
    # downstream code can rely on orientation instead of taking |det|.
    tetrahedra = orient_tetrahedra(all_nodes, tetrahedra)

    # Drop degenerate/near-degenerate connector-artifact tets (see "Dropped
    # tets" above) - recomputed post-orientation since orient_tetrahedra
    # only flips sign, never changes magnitude.
    p0 = all_nodes[tetrahedra[:, 0]]
    p1 = all_nodes[tetrahedra[:, 1]]
    p2 = all_nodes[tetrahedra[:, 2]]
    p3 = all_nodes[tetrahedra[:, 3]]
    e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
    det = np.einsum('ij,ij->i', e1, np.cross(e2, e3))
    drop = np.abs(det) < 1e-20

    n_dropped = int(np.count_nonzero(drop))
    if n_dropped:
        logger.info(
            f"Dropped {n_dropped} exactly-zero-volume tetrahedra "
            f"(see this function's own docstring)"
        )
        keep = ~drop
        tetrahedra = tetrahedra[keep]
        face_of_tet = face_of_tet[keep]

    logger.info(f"Total tetrahedra generated: {len(tetrahedra)}")
    return tetrahedra, face_of_tet


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
    min_cell_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """将分层棱柱网格转换为真正的三角棱柱单元——
    convert_layers_to_tetrahedra 的真棱柱对应版本，保留在同一模块
   是因为两者共享完全相同的每层节点对应关系记录（只有最终发出的
    单元形状不同）。

    每个（层，底面）对发出一个棱柱，使用与旧的四面体路径相同的
    排序顶点约定（按全局节点索引 v0<v1<v2，w_i 为上一层对应的
    顶点）来保证对角线一致性——见 PrismCells 和
    face_extractor.extract_faces_mixed 的文档字符串了解为何这使得
    一个棱柱的 8 个边界与 convert_layers_to_tetrahedra 的 3-四面体
    分割在相同 slab 中产生的结果位级相同，因此自动与相邻的棱柱
    （或在 BL/core 界面处，相邻的 core 四面体）保形，而本函数
    不需要任何跨单元协调，只需每个棱柱独立应用的相同全局索引排序。

    Args:
        all_nodes: 所有层的节点，shape=(total_nodes, 3)
        layer_connectivity: 每个挤出步一个条目，不是每个节点层——
            见 convert_layers_to_tetrahedra 自身的 `layer_connectivity`
            文档了解完整解释（本函数有相同的差一 bug，直到应用了
            相同的修复）。只使用其长度——每层共享 base_faces 自身
            的局部面拓扑，仅按 nodes_per_layer 偏移。
        base_faces: 原始表面面，shape=(n_faces, 3)
        min_cell_size: 可选的目标单元尺寸（米），用于将退化体积
            丢弃阈值缩放到 `(min_cell_size**3) * DEGENERATE_VOLUME_FRACTION`
            而不是固定的 1e-20——见该常量自身的注释了解为何简单的
            浮点噪声 epsilon 会漏掉一个完全坍缩（所有 3 条垂直边约 0，
            整个底面停滞）但不到精确浮点零的棱柱。None（默认值）
            保持旧的固定 1e-20 阈值，用于没有大小参考的调用方。

    Returns:
        (prisms, face_of_prism)：棱柱连接关系，shape=(n_prisms, 6)，
        格式 (v0,v1,v2,w0,w1,w2)；face_of_prism，shape=(n_prisms,)，
        将每个幸存的棱柱映射回其 base_faces 行索引（n_prisms 可能
        少于 n_base_faces*(n_layers-1)——体积相对于上述退化体积
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
    # +1：见 convert_layers_to_tetrahedra 的相同修复/注释——
    # layer_connectivity 每个挤出步一个条目，不是每个节点层。
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
    face_of_prism = np.empty(n_prisms, dtype=np.int64)
    face_range = np.arange(n_base_faces)

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
        face_of_prism[sl] = face_range
        prism_idx += n_base_faces

    # Drop fully-collapsed (whole-base-face) prisms - see
    # DEGENERATE_VOLUME_FRACTION's own comment for why this threshold is
    # scaled to min_cell_size rather than a fixed float-noise epsilon, and
    # this function's own Returns doc for why a volume-based (not per-
    # edge) check is what keeps a genuine wedge prism safe from being
    # dropped.
    from ...validation.quality_metrics import compute_prism_volumes
    volumes = compute_prism_volumes(all_nodes, prisms)
    degenerate_threshold = (
        (min_cell_size ** 3) * DEGENERATE_VOLUME_FRACTION if min_cell_size is not None
        else 1e-20
    )
    drop = volumes < degenerate_threshold

    n_dropped = int(np.count_nonzero(drop))
    if n_dropped:
        logger.info(
            f"Dropped {n_dropped} fully-collapsed prisms (volume < "
            f"{degenerate_threshold:.3e} m^3, an entire base face stalled)"
        )
        keep = ~drop
        prisms = prisms[keep]
        face_of_prism = face_of_prism[keep]

    logger.info(f"Total prisms generated: {len(prisms)}")
    return prisms, face_of_prism
