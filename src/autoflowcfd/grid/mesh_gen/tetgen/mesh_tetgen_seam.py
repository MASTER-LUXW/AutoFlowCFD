"""tetgen 核心域填充：BL 缝合处（seam）过渡与局部厚度限制。

从 mesh_tetgen_core.py 拆分出来，专门负责两类问题：BL 挤出区域和
core-仅 区域交界处（seam）的平滑过渡缩放，以及两侧 BL 前沿相向生长时
基于几何间隙的局部厚度上限（避免穿透）。供 mesh_background_merge.py 在
生成 core 填充所需的 PLC 边界之前调用。
"""

import numpy as np
from scipy.sparse import coo_matrix
from loguru import logger


def build_seam_taper_scale(
    n_nodes: int,
    extrude_faces: np.ndarray,
    core_faces: np.ndarray,
    taper_rings: int = 100,
) -> np.ndarray:
    """计算每个节点的 [0, 1] BL 挤出缩放，在与 core-only 面共享的
    seam 处平滑衰减到零（例如地平面与隧道壁交汇处）。

    将 seam 硬钉到精确零位移（本函数的早期版本）本身是不够的：
    每个接触被钉节点的三角形在外 BL 层会崩溃到接近零面积（3 个
    顶点中有 2 个被冻结，第三个移动全部 BL 厚度），给 tetgen 一
    个沿整个 seam 周长有退化/接近零面积面的边界表面——这在真实
    汽车几何上可靠地崩溃了 tetgen 的原生四面体化。在足够多的
    网格连接性环上平滑递增缩放可以保持 seam 附近每个面的顶点
    在可比较的位移范围内，避免该退化同时仍然保证精确保形
    （缩放恰好为 0，不仅是很小，就在 seam 本身）。

    默认 100 个环是故意宽松的，不是紧密的局部估计：在真实汽车
    几何上，seam 可能穿过一个小但几何上紧密的特征（例如车身
    地板与地面的焊接接触斑，边缘接近 90 度只有几毫米长）——
    经验验证窄过渡（约 4 个环）仍然在那里产生自交的 BL 表面，
    而加宽它就解决了，不需要单独的局部特征大小分析。

    Args:
        n_nodes: 共享节点数组中的总节点数
        extrude_faces: 将被 BL 挤出的面
        core_faces: 作为外 PLC 边界一部分不变使用的面
        taper_rings: 缩放从 0（seam 处）递增到 1（不受影响的内部）
            的网格连接性跳数

    Returns:
        float 数组，[0, 1]，shape=(n_nodes,)
    """
    scale = np.ones(n_nodes, dtype=np.float64)
    if len(extrude_faces) == 0 or len(core_faces) == 0:
        return scale

    extrude_node_idx = np.unique(extrude_faces)
    core_node_idx = np.unique(core_faces)

    in_extrude = np.zeros(n_nodes, dtype=bool)
    in_extrude[extrude_node_idx] = True
    in_core = np.zeros(n_nodes, dtype=bool)
    in_core[core_node_idx] = True
    seam_nodes = np.flatnonzero(in_extrude & in_core)

    logger.info(f"Seam nodes (shared between extruded and core-only faces): {len(seam_nodes)}")
    if len(seam_nodes) == 0:
        return scale

    # 多源无权最短路径（跳数），从每个 seam 节点出发，
    # 限制在可挤出面图上（只有该区域的节点实际移动，所以只有它的
    # 连接性对过渡很重要）。
    edges = np.vstack([extrude_faces[:, [0, 1]], extrude_faces[:, [1, 2]], extrude_faces[:, [2, 0]]])
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))

    from scipy.sparse.csgraph import dijkstra
    hop_dist = dijkstra(graph, indices=seam_nodes, unweighted=True, min_only=True)

    t = np.clip(hop_dist / taper_rings, 0.0, 1.0)
    # 普通线性斜坡，不是以前用的 smoothstep (3t^2 - 2t^3)。
    # Smoothstep 在 t=0（seam 本身）处斜率为零是构造决定的——
    # 在 t=1 处故意如此（平滑混合到未过渡的内部，缩放在那里保持
    # 精确 1.0），但同样的 t=0 平坦意味着 taper_rings=100 中的
    # 前几十个环保持在约 10% 缩放以下（解 3t^2-2t^3=0.1 得 t~=0.196，
    # 即约 20 个环）——在 cube_demo 上直接确认：约 12-14k 个 BL 棱柱
    # 有接近零高度的垂直边，集中在这个 seam 附近的平坦带，不是真实
    # 缺陷而是这个过渡函数自身设计的形状。线性具有常量斜率
    # 1/taper_rings，实际上比 smoothstep 的峰值斜率 1.5/taper_rings
    # （在 t=0.5 处达到）还要低——所以它在任何地方都"不比 smoothstep
    # 更激进"，只是消除了人为平坦的起点，那个起点把太多环集中在
    # 接近零的带中。它放弃的唯一东西是 t=1 处的零斜率混合（线性
    # 以 1/taper_rings 的斜率不连续地遇到未过渡的 scale=1.0 平台，
    # smoothstep 在那里没有不连续）——一个有界的、小的
    # （taper_rings=100）不连续，正好在过渡区自身的边缘，不是在
    # 本函数文档字符串警告的紧密特征 seam 处。
    linear = t
    # 从任何 seam 节点不可达的节点（不通过挤出面图连接，例如不相关的
    # 嵌入壳）保持 scale=1。
    unreachable = ~np.isfinite(hop_dist)
    linear[unreachable] = 1.0

    scale = linear
    logger.info(
        f"BL taper applied over {taper_rings} connectivity rings from the seam "
        f"({int(np.sum(scale < 1.0))} nodes affected)"
    )
    return scale


def compute_local_thickness_limit(
    nodes: np.ndarray,
    extrude_faces: np.ndarray,
    extrude_node_idx: np.ndarray,
    domain_size: float,
    safety_factor: float = 0.45,
    angle_threshold_deg: float = 60.0,
    search_radius_fraction: float = 0.08,
) -> np.ndarray:
    """将每个可挤出节点的*累积* BL 厚度上限限制到其与最近相对表面
    的局部几何间隙的一个比例，这样两个 BL 前沿在紧密特征（例如车身
    地板离地面几厘米）上相向生长时会在穿透前停止——而不是以均匀速率
    生长然后依赖 repair_nonmanifold_cells 来清理结果重叠。

    间隙在未变形的（layer-0）表面上测量：对每个节点，搜索在
    `domain_size * search_radius_fraction` 内的附近表面节点，只保留
    那些大致在节点自身 outward 法向"前方"的（在 `angle_threshold_deg`
    内）——这区分了真正的相对间隙和节点自身的紧邻网格（它们总是
    空间上很近，只是因为局部网格分辨率，不是真实间隙，而且大致在
    面内而不是在法向前方）。最近合格点的距离就是局部间隙；
    `safety_factor`（< 0.5）为相对表面的 BL 生长留出裕量。

    这是几何启发式，不是非交叉的形式证明：它在未变形表面上评估
    一次，所以一个强烈弯曲的前端，其真实最近点在两侧都挤出时可能
    移动，原则上仍可能比估计的更快收敛。它是大幅减少交叉发生频率，
    不是保证——repair_nonmanifold_cells 仍然作为安全网保留。

    Args:
        nodes: (n_nodes, 3) 所有表面节点坐标（整个表面，不仅是
            可挤出子集——限制某个壁 BL 增长的最近特征可能是另一个
            壁）
        extrude_faces: (m, 3) 可 BL 挤出的面，仅用于计算每个节点的
            outward（挤出）法向
        extrude_node_idx: 实际将被挤出的节点索引
        domain_size: 域整体特征长度（包围盒对角线），同时约束搜索
            半径和回退上限
        safety_factor: 原始间隙距离的保留比例
        angle_threshold_deg: "法向前方"锥的半角，用于区分相对间隙和
            同片网格邻居
        search_radius_fraction: domain_size 的比例，用作 KDTree 球查询
            半径。BL 增长目标只达到约 domain_size 的 2%（见
            extrude_layers 的 bl_target_thickness）；extrude_layers 另外
            无条件硬停在 domain_size 的 40%，无论本函数计算什么，所以
            那个极端情况已被覆盖，不需要这个搜索达到那么远。先前默认
            值（0.4，即与那个无关硬停*相同*的 40%）使查询球在典型
            外气动域上经常包围很大比例的整体表面网格，把本应是局部
            邻居搜索变成接近暴力搜索——在大网格上每个可挤出节点有
            多分钟（或更差）的运行时间风险。目标值的适度倍数（默认
            8%）在正常情况上有舒适的裕量，同时搜索体积和因此的典型
            候选数量减少约 (0.4/0.08)^3 = 125 倍。

    Returns:
        (n_nodes,) float 数组：每个节点的最大累积 BL 厚度（米），
        未找到附近相对特征处为 np.inf
    """
    from scipy.spatial import cKDTree

    n_nodes = len(nodes)
    limit = np.full(n_nodes, np.inf, dtype=np.float64)
    if len(extrude_node_idx) == 0:
        return limit

    face_normals = _face_normals(nodes, extrude_faces)
    avg_normal = _average_node_normals(n_nodes, extrude_faces, face_normals)

    search_radius = domain_size * search_radius_fraction
    cos_threshold = np.cos(np.radians(angle_threshold_deg))

    tree = cKDTree(nodes)
    query_points = nodes[extrude_node_idx]

    # 分块查询和处理，而不是一次全部，并且对每个块内的候选做
    # 向量化的角度/距离测试，而不是 Python 级别的每节点循环。在精细
    # 表面网格上，域尺度的 search_radius 可以在每次查询中包围数万个
    # 同片候选（角度测试会丢弃几乎所有——它们是面内邻居，不是真正的
    # 相对特征）——一次性实现所有查询的完整候选列表（之前未分块的
    # 行为）内存随 n_queries * avg_candidates 缩放，在约 25k 表面节点
    # 的案例上达到多 GB 瞬态使用，随后的每节点 Python 循环（34k+
    # 次迭代，每次做几个小 numpy 调用）主导了运行时间——每次调用
    # 100+ 秒，每次 BL 挤出尝试都重复。这里产生数值相同的结果
    # （相同半径、相同角度测试、相同最近前方点选择）——只改变了工作
    # 的批处理方式。
    chunk_size = 200
    n_capped = 0
    min_cap_seen = np.inf

    for start in range(0, len(query_points), chunk_size):
        end = min(start + chunk_size, len(query_points))
        chunk_node_idx = extrude_node_idx[start:end]
        neighbor_lists = tree.query_ball_point(
            query_points[start:end], r=search_radius, workers=-1
        )
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )
        if counts.sum() == 0:
            continue

        row_idx = np.repeat(np.arange(len(chunk_node_idx)), counts)
        flat_candidates = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )

        node_idx_per_row = chunk_node_idx[row_idx]
        d = nodes[flat_candidates] - nodes[node_idx_per_row]
        dist = np.linalg.norm(d, axis=1)
        real = dist > 1e-9
        safe_dist = np.where(real, dist, 1.0)
        cosang = np.einsum('ij,ij->i', d, avg_normal[node_idx_per_row]) / safe_dist
        ahead = real & (cosang > cos_threshold)
        if not np.any(ahead):
            continue

        dist_masked = np.where(ahead, dist, np.inf)
        seg_min = np.full(len(chunk_node_idx), np.inf)
        np.minimum.at(seg_min, row_idx, dist_masked)

        has_match = np.isfinite(seg_min)
        if np.any(has_match):
            capped_vals = seg_min[has_match] * safety_factor
            limit[chunk_node_idx[has_match]] = capped_vals
            n_capped += int(has_match.sum())
            min_cap_seen = min(min_cap_seen, float(capped_vals.min()))

    if n_capped:
        logger.info(
            f"Local BL thickness limiting: {n_capped} nodes capped by a "
            f"nearby facing feature (min cap {min_cap_seen:.4e} m)"
        )
        limit = _smooth_thickness_limit(limit, extrude_faces)
    return limit


def _smooth_thickness_limit(
    limit: np.ndarray, extrude_faces: np.ndarray,
    max_ratio: float = 1.3, max_iterations: int = 50,
) -> np.ndarray:
    """将每个被限制节点的 thickness_limit 向外传播到整个网格，使得
    任意两个边相邻的可挤出节点差异不超过 `max_ratio`——与本项目的
    BL/过渡增长率已经使用的相同平滑分级原则（约 1.2-1.3 每步，
    通用 CFD 网格实践），在这里应用于 CAP 字段。

    没有这个，每个节点的上限是独立设置的（见上方
    compute_local_thickness_limit），与其网格邻居零协调——一个被附近
    相对特征紧密限制的节点（例如车身地板到地面）可以紧挨着一个完全
    未限制的邻居，在相同表面上距离几分之一毫米。extrude_layers 将该
    上限作为硬每节点停止执行，所以 BL 外表面最终以被冻结节点和
    其仍在增长的邻居之间的突然局部台阶结束——正是这种尖锐的局部
    跳跃已被本项目反复发现会在该 seam 处产生退化（接近零体积）
    单元。那些退化单元然后在后处理期间被无条件丢弃，不做修复尝试
    （没有东西将现在的空区域标记为"坏"，不像存在但低质量的单元
    那样），在最终体积网格中留下真正未网格化的间隙。已直接在真实
    案例上确认为真实效果：约 45-50% 的表面节点被厚度限制（一个有
    许多紧密相对特征靠近的案例），每次生成尝试丢弃数十万个退化
    单元。

    实现为向量化的 Bellman-Ford 风格松弛：反复将 `own_limit *
    max_ratio` 传播到每个网格边邻居，保持最小值，直到没有值变化
    （或达到 max_iterations——从极端（亚毫米）上限到典型远场目标
    大小只需要约 log(target/cap)/log(max_ratio) 跳，对本项目自身
    最小/max_cell_size 范围的任何现实组合都远低于 50）。一个没有
    相对特征上限的节点（np.inf）只会被这个拉低——它永远不能将真正
    被限制的邻居的更紧值推高。
    """
    if not np.any(np.isfinite(limit)):
        return limit

    edges = np.vstack([
        extrude_faces[:, [0, 1]], extrude_faces[:, [1, 2]], extrude_faces[:, [2, 0]],
    ])
    a, b = edges[:, 0], edges[:, 1]

    for _ in range(max_iterations):
        updated = limit.copy()
        np.minimum.at(updated, b, limit[a] * max_ratio)
        np.minimum.at(updated, a, limit[b] * max_ratio)
        if np.array_equal(updated, limit):
            break
        limit = updated

    return limit


def _face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    return normals / norms


def _average_node_normals(n_nodes: int, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    sums = np.zeros((n_nodes, 3), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.int64)
    flat_nodes = faces.ravel()
    np.add.at(sums, flat_nodes, np.repeat(face_normals, 3, axis=0))
    np.add.at(counts, flat_nodes, 1)

    mask = counts > 0
    avg = np.zeros_like(sums)
    avg[mask] = sums[mask] / counts[mask, np.newaxis]
    norms = np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-10)
    avg[mask] = avg[mask] / norms[mask]
    return avg
