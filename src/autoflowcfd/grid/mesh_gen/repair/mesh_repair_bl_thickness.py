"""阶段 B（BL 侧）：定向的 BL 厚度封顶再生成参数。

compute_bl_thickness_limit_override 是一个纯函数，把阶段 A 平滑之后仍然
不达标的一批 BL 区域单元，转换成一个定向再生成参数——在特定表面顶点上的
局部 BL 厚度上限——交给调用方（mesh_background.generate_hybrid_mesh）喂进
第二次定向再生成。之所以安全，正是因为它反馈回的是同一条已经验证正确的
生成路径，而不是自己手搓一个局部/部分重新铺网的实现。

之前 core 侧有一个对应方案（基于区域的局部细化），已经因为实际效果净负面
而被移除：当同一个连通体积里还有一个域级别的分级区域在起作用时，tetgen
按区域细化不会把自己限制在新增区域的小范围局部 footprint 内，于是几个小的
局部修复区域就可能把整个 core 填充规模成倍吹大，而实际质量并没有改善
（见 mesh_background.py 自己的历史记录）。阶段 B'（mesh_repair_cavity.py）
才是 core 侧那种情况真正的局部重铺网替代方案；这里的 BL 侧厚度封顶没有
同类失效模式（它只会局部缩短 BL 层，从不扩张任何东西），所以保持不变。

从 mesh_repair.py 拆出，纯粹为了控制文件行数——在 mesh_repair.py 底部
重新转出（见该文件末尾），让现有调用方不受影响。
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


def compute_bl_thickness_limit_override(
    bad_cell_mask: np.ndarray,
    n_bl_cells: int,
    cells: np.ndarray,
    n_surface_nodes: int,
    cap_thickness: float,
    existing_thickness_limit: Optional[np.ndarray] = None,
    nodes_per_layer: Optional[int] = None,
    node_original_vertex: Optional[np.ndarray] = None,
    local_surface_faces: Optional[np.ndarray] = None,
    taper_rings: int = 3,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """阶段 B，BL 侧：对于仍在 BL 区域内的残留坏单元（单元索引 < n_bl_cells），
    将其节点追溯回播种其 BL 柱的原始表面顶点（BL 节点的全局索引为
    `layer_idx * nodes_per_layer + local_index`，因此 `node_idx %
    nodes_per_layer` 恢复 `local_index`，与层无关），并将那里的累积 BL
    厚度封顶到 `cap_thickness`（约 2-3 层厚度）——强制挤出在正好涉及
    坏单元的顶点处提前停止生长，其他所有地方不受影响。

    Args:
        nodes_per_layer: 实际的每层节点步长——默认为 n_surface_nodes。
            保留为显式参数（而非始终假设 n_surface_nodes），使未来层步长
            合法不同于 n_surface_nodes 的 BL 生成路径不会像此函数的早期
            版本那样静默恢复几乎每个节点的错误局部索引（该 bug 将真正
            局部的坏单元簇膨胀为"25577 表面顶点中的 21888"封顶，随后
            喂入 tetgen 内部鲁棒性崩溃——参见 mesh_tetgen_core.fill_core_
            volume 的 removevertexbyflips 处理）。
        node_original_vertex: 可选 (nodes_per_layer,) 数组，将局部索引
            （取模后）映射回其原始（n_surface_nodes 空间）顶点，用于
            nodes_per_layer 不同于 n_surface_nodes 的情况。None（默认）
            等价于恒等映射，在 nodes_per_layer == n_surface_nodes（正常
            情况）时正确。
        local_surface_faces: 可选 (m, 3) 表面面连接关系
            （mesh_background._build_merged_mesh 自身的 extrude_faces，
            与 node_original_vertex 处于相同的局部/nodes_per_layer 索引
            空间）——给定后，封顶从原始涉及的局部节点向外平滑锥度超过
            taper_rings 个网格连接跳数（与 mesh_tetgen_core.build_seam_
            taper_scale 相同的跳数 + smoothstep 技术），而非在正好涉及的
            顶点处作为硬崖施加。硬崖有在封顶节点与其未封顶网格邻居之间
            的边界处产生严重退化/高长细比单元的风险（两个直接相邻节点
            在几乎相同的 (x, y) 处，一个冻结在近 layer-0 高度，另一个
            以完整速率生长）——锥度将该高度不匹配分散到多跳而非集中在
            一个面上。None（默认）保持先前硬崖行为不变——在
            local_surface_faces 不可用时安全（例如根本没有 BL 区域），
            只是不锥度。
        taper_rings: 给定 local_surface_faces 时锥度的跳数宽度——刻意
            小（不同于 build_seam_taper_scale 的默认 100，后者存在是为了
            在真正紧凑的真实几何特征的接缝上存活）：此锥度只需平滑输出
            一个连接宽度不匹配，不保证大接缝上的无交叉。

    Returns:
        (thickness_limit_array 或无需处理时为 None，受影响的表面顶点索引)
        ——数组大小为 (n_surface_nodes,)，通过逐元素 np.minimum 合并到
        已有数组（已在为紧凑特征封顶计算自身 thickness_limit 的调用方
        应组合两者，而非用一个替换另一个）。
    """
    bl_bad = np.flatnonzero(bad_cell_mask[:n_bl_cells])
    if len(bl_bad) == 0:
        return None, []

    stride = nodes_per_layer if nodes_per_layer is not None else n_surface_nodes
    bad_nodes = np.unique(cells[bl_bad].ravel())
    local_idx = np.unique(bad_nodes % stride)
    surface_verts = np.unique(node_original_vertex[local_idx]) if node_original_vertex is not None else local_idx

    limit = (
        existing_thickness_limit.copy()
        if existing_thickness_limit is not None
        else np.full(n_surface_nodes, np.inf)
    )

    if local_surface_faces is not None and len(local_surface_faces):
        # 从原始涉及的局部节点向外锥度，基于分裂后网格的
        # 连接关系（直接捕获任何新连接邻接——参见本函数自身文档）
        # 而非硬崖。与 build_seam_taper_scale 相同的跳数 + smoothstep
        # 技术；在 hop 0 处这简化为正好 cap_thickness，因此它取代了旧的
        # 硬封顶行为，而非需要为种子节点本身单独一步。
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        edges = np.vstack([
            local_surface_faces[:, [0, 1]],
            local_surface_faces[:, [1, 2]],
            local_surface_faces[:, [2, 0]],
        ])
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(stride, stride))
        hop_dist = dijkstra(graph, indices=local_idx, unweighted=True, min_only=True)

        t = np.clip(hop_dist / taper_rings, 0.0, 1.0)
        smoothstep = t * t * (3.0 - 2.0 * t)
        reachable = np.isfinite(hop_dist)

        tapered_local = np.full(stride, np.inf)
        tapered_local[reachable] = cap_thickness / np.maximum(1.0 - smoothstep[reachable], 1e-9)
        tapered_local[reachable & (t >= 1.0)] = np.inf

        orig_idx = node_original_vertex if node_original_vertex is not None else np.arange(stride)
        tapered_by_vertex = np.full(n_surface_nodes, np.inf)
        np.minimum.at(tapered_by_vertex, orig_idx, tapered_local)

        limit = np.minimum(limit, tapered_by_vertex)
        n_tapered = int(np.sum(np.isfinite(tapered_by_vertex)))
        logger.info(
            f"Stage B (BL side): capping cumulative BL thickness to {cap_thickness:.6f}m "
            f"at {len(surface_verts)} surface vertices implicated in {len(bl_bad)} residual bad "
            f"cells, tapered over {taper_rings} rings ({n_tapered} vertices affected in total)"
        )
    else:
        limit[surface_verts] = np.minimum(limit[surface_verts], cap_thickness)
        logger.info(
            f"Stage B (BL side): capping cumulative BL thickness to {cap_thickness:.6f}m "
            f"at {len(surface_verts)} surface vertices implicated in {len(bl_bad)} residual bad cells"
        )

    return limit, surface_verts.tolist()
