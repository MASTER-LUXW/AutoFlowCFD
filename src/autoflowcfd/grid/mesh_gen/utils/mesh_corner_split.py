"""BL 挤出前的尖锐拐角顶点拆分。

概述：把接触到 3 个及以上曲面片的硬边/拐角顶点，按每个曲面片各复制一份、
沿各自真实法向偏移，再用 bevel/cap 三角形把裂开的缝隙缝合——单一平均法向
无法正确表达 valence-3+ 拐角，容易在挤出时自相交。下面详细说明原因与
实现细节（cap 扇形必须按真实几何环绕顺序连接，否则会产生扭曲的连接面）。

extrude_single_layer 的 per-node 平均法向（mesh_layer_step.py）
及其斜接补偿模型能正确处理单一两补丁尖锐边（沿一个混合方向的
固定补偿因子），但无法表示真正的 valence-3+ 拐角——三个或更多
曲面片交汇于一点——而不自交风险：没有单一混合方向能同时正确
偏移三个独立平面，并且 mesh_front_collision.py 的反应式冻结会
将违规节点回退并永久停止它们，在运行的剩余时间内产生退化
（零体积、丢弃）单元。已在 cube_demo（一个真正的长方体）上
确认：冻结从第一个 BL 层开始，恰好位于长方体自身的边/拐角
相邻节点处，并在几层内级联影响大部分表面。

split_sharp_corners 采用真实 BL 网格生成器（Pointwise 的 T-Rex、ANSA）
对此的替代方案：将硬边/拐角顶点复制为每个接触的曲面片一份，
沿各自真实的（未混合的）法向偏移每个副本，然后用额外的“倒角”
三角形沿每条硬边缝合缝隙，加上在 3+ 曲面片交汇的 valence-3+
拐角处的“帽盖”三角形扇形（两补丁边不需要帽盖——一个倒角四边形
已经完全封闭）。每个部分（补丁 A 的偏移、补丁 B 的偏移、
连接它们的平面倒角/帽盖）都不能在凸特征处折叠，
这与单一混合法向斜接估计不同。

帽盖的正确性要求：帽盖扇形必须按曲面片围绕顶点的真实几何
循环顺序连接，从局部硬边邻接关系推导（每个曲面片在简单闭合
扇形中恰好与 2 个其他曲面片相邻）。任意（例如补丁 ID）顺序的
扇形仅对 k==3 安全（任意 3 个点无论顺序如何都形成一个有效三角形）
——对 k>3 它可能直接连接几何不相邻的曲面片，产生严重扭曲/过大的
连接器（一旦副本从第 0 层开始分离就已确认：一个早期的顺序无关版本
的此函数在一个真实的、非长方体的、具有多个 k>3 顶点的物体上
导致了 140 万对网格重叠爆炸）。每当 k>=3 顶点的局部拓扑不是
简单闭合扇形（非流形输入，或顶点位于网格自身曲面片结构比
普通星形更缠绕的地方）时，此模块完全不拆分该顶点——
回退到拆分前的单一平均法向行为——而非冒顺序错误的连接器
或实际未封闭缝隙的风险。

在整个挤出面补丁边界上的顶点（属于仅有一个邻接面的边的一部分——
与未挤出（例如仅核心）边界组的接缝）正常拆分但从不加盖：
其曲面片不形成闭合扇形（开放扇形的两端独立延伸到边界，
无需连接），并且 mesh_tetgen_core.build_seam_taper_scale 已经
通过将位移逐渐衰减到零来单独处理该接缝。
"""

import numpy as np
from typing import Tuple
from loguru import logger

from .mesh_corner_split_geometry import _face_normals, _implied_edge_radius, _unique_edges

FEATURE_ANGLE_THRESHOLD_RAD = np.deg2rad(20.0)


def split_sharp_corners(
    nodes: np.ndarray,
    faces: np.ndarray,
    threshold: float = FEATURE_ANGLE_THRESHOLD_RAD,
    min_feature_radius: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 `faces` 的每个尖锐拐角/硬边顶点拆分为每个平滑补丁一个节点副本，
    添加倒角/帽盖三角形以保持结果水密。

    Args:
        nodes: 完整节点数组（此函数仅追加行，所有现有索引保持有效），
            shape=(n_nodes, 3)
        faces: 要拆分的三角形连接关系, shape=(n_faces, 3)——
            自包含的子网格（例如 classify_boundary_groups 的
            extrude_faces），而非整个表面
        threshold: 二面角（弧度）超过此值的边为“硬边”，
            其两侧面被视为不同补丁，除非以其他方式连接
        min_feature_radius: 二面角超过 `threshold` 的边如果其自身
            几何暗示的局部曲率半径大于等于此值（米），则仍不视为硬边——
            见 _implied_edge_radius 的文档字符串了解为何这能区分
            真正的尖锐（近零半径）CAD 折痕与普通曲面（圆角、
            圆角拐角），后者只是相对于自身真实半径欠细分。
            刻意是单边启发式，非保证：真正精细的输入网格会让
            单纯的 `threshold` 检查正确区分两者，但朴素（平面、
            非曲面拟合）细分根本不会减小粗曲面的 per-facet 角度
            （平三角形的子三角形仍精确共面——已确认：cube_demo
            上 3 轮细分完全不改变补丁计数，同时因额外小三角形
            挤在同一个紧拐角处而使自交明显恶化），
            因此没有真正的曲面拟合重采样（本项目没有）就无法走这条路。
            0.0（默认）保留单纯的角度检查行为。

    Returns:
        new_nodes: 追加了副本行的节点数组, shape=(n_nodes + n_copies, 3)
            ——每个副本从与原始顶点相同的位置开始
            （仅后续 BL 挤出层使副本分离）
        topology_faces: faces.copy() 并将顶点索引重映射到修正的
            per-补丁副本，然后追加新的倒角/帽盖三角形，
            shape=(n_faces + n_extra, 3)
        real_face_mask: bool, shape=(len(topology_faces),) - 前 n_faces 行
            为 True（原始重映射三角形——用于 per-node 法向平均），
            追加的倒角/帽盖行为 False（纯连接关系，从不贡献
            法向平均——它们的角点节点已从真实补丁的面获得正确法向）
        orig_of_node: int64, shape=(len(new_nodes),) - 将每个节点
            （原始和副本）映射回输入 `nodes` 数组中的原始顶点索引，
            用于同样方式扩展其他 per-原始-顶点数组
            （taper_scale、thickness_limit）
        bevel_source_face: int64, shape=(n_extra,) - 对每个追加行，
            它应从哪个原始 `faces` 行（0 起始）继承边界组属性
    """
    n_nodes = len(nodes)
    n_faces = len(faces)
    if n_faces == 0:
        return (
            nodes.copy(), faces.copy(), np.ones(0, dtype=bool),
            np.arange(n_nodes), np.zeros(0, dtype=np.int64),
        )

    face_normals = _face_normals(nodes, faces)

    parent = np.arange(n_faces)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    boundary_edge_verts = []  # (v0, v1) with only 1 adjacent face
    hard_edge_list = []  # (v0, v1, fa, fb) with 2 adjacent faces, angle > threshold

    for v0, v1, fidx in _unique_edges(faces):
        if len(fidx) == 1:
            boundary_edge_verts.append((int(v0), int(v1)))
            continue
        if len(fidx) != 2:
            continue  # non-manifold edge - leave unioned-apart (safest: treat as hard, no bevel)
        fa, fb = int(fidx[0]), int(fidx[1])
        cosang = np.clip(np.dot(face_normals[fa], face_normals[fb]), -1.0, 1.0)
        angle = np.arccos(cosang)
        is_hard = angle > threshold
        # An edge that crosses the plain angle threshold is still treated
        # as an ordinary smooth edge if its own geometry implies a local
        # curvature radius at or above min_feature_radius - see this
        # function's own min_feature_radius docstring for why (and its
        # documented limits).
        if is_hard and min_feature_radius > 0.0:
            edge_length = float(np.linalg.norm(nodes[v0] - nodes[v1]))
            implied_radius = _implied_edge_radius(edge_length, angle)
            if implied_radius >= min_feature_radius:
                is_hard = False
        if is_hard:
            hard_edge_list.append((v0, v1, fa, fb))
        else:
            union(fa, fb)

    patch_id_raw = np.array([find(f) for f in range(n_faces)], dtype=np.int64)
    _, patch_id = np.unique(patch_id_raw, return_inverse=True)
    n_patches = int(patch_id.max()) + 1 if n_faces else 0

    boundary_verts = set()
    for v0, v1 in boundary_edge_verts:
        boundary_verts.add(v0)
        boundary_verts.add(v1)

    hard_edges = np.array(hard_edge_list, dtype=np.int64).reshape(-1, 4)
    he_pa = patch_id[hard_edges[:, 2]] if len(hard_edges) else np.zeros(0, dtype=np.int64)
    he_pb = patch_id[hard_edges[:, 3]] if len(hard_edges) else np.zeros(0, dtype=np.int64)
    he_differ = he_pa != he_pb
    hard_edges = hard_edges[he_differ]
    he_pa, he_pb = he_pa[he_differ], he_pb[he_differ]

    vertex_patch_adj: dict = {}
    for i in range(len(hard_edges)):
        v0i, v1i, pai, pbi = int(hard_edges[i, 0]), int(hard_edges[i, 1]), int(he_pa[i]), int(he_pb[i])
        for v in (v0i, v1i):
            d = vertex_patch_adj.setdefault(v, {})
            d.setdefault(pai, set()).add(pbi)
            d.setdefault(pbi, set()).add(pai)

    # --- 初步的（顶点，补丁）分组，使用原始 patch_id，
    # 以决定 per-顶点拆分是否安全（见模块文档字符串）。
    flat_vert = faces.ravel().astype(np.int64)
    flat_patch = np.repeat(patch_id, 3)
    key0 = flat_vert * n_patches + flat_patch
    uk0, _ = np.unique(key0, return_inverse=True)
    uk0_vertex = uk0 // n_patches
    uk0_patch = uk0 % n_patches

    group_starts = np.flatnonzero(np.concatenate([[True], uk0_vertex[1:] != uk0_vertex[:-1]]))
    group_ends = np.append(group_starts[1:], len(uk0_vertex))

    force_single = np.zeros(n_nodes, dtype=bool)
    vertex_cyclic_order: dict = {}
    n_skipped_irregular = 0

    for gs, ge in zip(group_starts, group_ends):
        k = ge - gs
        if k < 3:
            continue
        v = int(uk0_vertex[gs])
        if v in boundary_verts:
            continue  # open fan - split is fine, just never capped (below)
        patches_here = uk0_patch[gs:ge].tolist()
        patches_set = set(patches_here)
        adj = vertex_patch_adj.get(v)
        regular = adj is not None
        if regular:
            for p in patches_here:
                neigh = adj.get(p, set())
                if len(neigh) != 2 or not neigh.issubset(patches_set):
                    regular = False
                    break
        if regular:
            cyclic = [patches_here[0]]
            prev, cur = None, patches_here[0]
            ok = True
            for _ in range(k - 1):
                neigh = list(adj[cur])
                nxt = neigh[0] if neigh[0] != prev else neigh[1]
                if nxt in cyclic:
                    ok = False
                    break
                cyclic.append(nxt)
                prev, cur = cur, nxt
            if ok and len(cyclic) == k and patches_here[0] in adj[cyclic[-1]]:
                vertex_cyclic_order[v] = cyclic
            else:
                regular = False
        if not regular:
            force_single[v] = True
            n_skipped_irregular += 1

    if n_skipped_irregular:
        logger.warning(
            f"Sharp-corner splitting: {n_skipped_irregular} valence-3+ vertex/vertices "
            f"had irregular local patch topology (not a simple closed fan) - left "
            f"unsplit (falls back to the pre-split averaged-normal behaviour there) "
            f"rather than risk an incorrectly-ordered connector"
        )

    # --- 最终的（顶点，补丁）分组：force_single 顶点将其所有
    # 面角折叠为单一合成补丁（0）——即该顶点完全不拆分，
    # 无论有多少真实补丁接触它。
    final_flat_patch = np.where(force_single[flat_vert], 0, flat_patch)
    key = flat_vert * n_patches + final_flat_patch
    unique_keys, inverse = np.unique(key, return_inverse=True)
    uk_vertex = unique_keys // n_patches

    is_first_for_vertex = np.ones(len(unique_keys), dtype=bool)
    is_first_for_vertex[1:] = uk_vertex[1:] != uk_vertex[:-1]

    new_node_index = np.empty(len(unique_keys), dtype=np.int64)
    new_node_index[is_first_for_vertex] = uk_vertex[is_first_for_vertex]
    n_extra_copies = int(np.sum(~is_first_for_vertex))
    new_node_index[~is_first_for_vertex] = n_nodes + np.arange(n_extra_copies)

    orig_of_node = np.concatenate([
        np.arange(n_nodes, dtype=np.int64),
        uk_vertex[~is_first_for_vertex],
    ])
    new_nodes = np.vstack([nodes, nodes[uk_vertex[~is_first_for_vertex]]])

    topology_faces_real = new_node_index[inverse].reshape(n_faces, 3)

    n_split_vertices = int(np.sum(np.bincount(uk_vertex, minlength=n_nodes) > 1))
    if n_extra_copies:
        logger.info(
            f"Sharp-corner splitting: {n_split_vertices} vertices split into "
            f"{n_extra_copies} extra copies ({n_patches} smooth patches total)"
        )

    def lookup_copy(vertex_arr: np.ndarray, patch_arr: np.ndarray) -> np.ndarray:
        eff_patch = np.where(force_single[vertex_arr], 0, patch_arr)
        k = vertex_arr.astype(np.int64) * n_patches + eff_patch.astype(np.int64)
        idx = np.searchsorted(unique_keys, k)
        idx = np.clip(idx, 0, len(unique_keys) - 1)
        return new_node_index[idx]

    # --- 沿每条硬边的倒角条，其两侧面落在真正不同的补丁中
    # （硬边通过网格上其他地方的某条路径被合并回去的——
    # 两端已经共享副本——无间隙，无需倒角）。force_single 端点
    # 将结果四边形的一侧折叠为一个点（其两个三角形在该侧
    # 共享 2 个相同顶点）——作为退化三角形在下方过滤掉。
    bevel_tris = []
    bevel_source = []
    if len(hard_edges):
        v0, v1, fa, fb = hard_edges[:, 0], hard_edges[:, 1], hard_edges[:, 2], hard_edges[:, 3]
        c_v0_a = lookup_copy(v0, he_pa)
        c_v1_a = lookup_copy(v1, he_pa)
        c_v0_b = lookup_copy(v0, he_pb)
        c_v1_b = lookup_copy(v1, he_pb)
        bevel_tris.append(np.stack([c_v0_a, c_v1_a, c_v1_b], axis=1))
        bevel_source.append(fa)
        bevel_tris.append(np.stack([c_v0_a, c_v1_b, c_v0_b], axis=1))
        bevel_source.append(fa)

    # --- 拐角帽盖：为上方计算的每个规则 valence-3+ 顶点的
    # 真实循环顺序进行扇形三角化。force_single 对每个这样的 v
    # 按构造为假（仅在规则拆分适用时添加到
    # vertex_cyclic_order），因此 lookup_copy 将每个 (v, patch) 对
    # 解析到其真实的 per-补丁副本，而非折叠的单一索引。
    for v, cyclic in vertex_cyclic_order.items():
        v_arr = np.full(len(cyclic), v, dtype=np.int64)
        p_arr = np.array(cyclic, dtype=np.int64)
        copies = lookup_copy(v_arr, p_arr)
        apex = int(copies[0])
        for i in range(1, len(cyclic) - 1):
            bevel_tris.append(np.array([[apex, int(copies[i]), int(copies[i + 1])]], dtype=np.int64))
            bevel_source.append(np.array([-1], dtype=np.int64))  # filled in below

    if bevel_tris:
        extra_faces = np.vstack(bevel_tris)
        bevel_source_face = np.concatenate(bevel_source)

        # 丢弃退化行（force_single 端点在其他wise拆分的边上
        # 将其倒角四边形的一侧折叠为重复顶点——见上方倒角条注释）。
        degenerate = (
            (extra_faces[:, 0] == extra_faces[:, 1]) |
            (extra_faces[:, 1] == extra_faces[:, 2]) |
            (extra_faces[:, 0] == extra_faces[:, 2])
        )
        if np.any(degenerate):
            extra_faces = extra_faces[~degenerate]
            bevel_source_face = bevel_source_face[~degenerate]

        # 拐角帽盖行以占位符 -1 源面追加
        # （帽盖三角形不“属于”任何单个原始面）——
        # 回退到恰好接触其 apex 顶点的任何面，
        # 使边界组继承仍能解析到正确的组名，
        # 而非在 -1 索引上崩溃。
        need_fallback = bevel_source_face < 0
        if np.any(need_fallback):
            vertex_to_any_face = np.full(n_nodes + n_extra_copies, -1, dtype=np.int64)
            vertex_to_any_face[faces[:, 0]] = np.arange(n_faces)
            vertex_to_any_face[faces[:, 1]] = np.arange(n_faces)
            vertex_to_any_face[faces[:, 2]] = np.arange(n_faces)
            apex_nodes = extra_faces[need_fallback, 0]
            apex_orig = orig_of_node[apex_nodes]
            resolved = vertex_to_any_face[apex_orig]
            resolved = np.where(resolved < 0, 0, resolved)
            bevel_source_face[need_fallback] = resolved
    else:
        extra_faces = np.zeros((0, 3), dtype=np.int64)
        bevel_source_face = np.zeros(0, dtype=np.int64)

    topology_faces = np.vstack([topology_faces_real, extra_faces]).astype(np.int64)
    real_face_mask = np.zeros(len(topology_faces), dtype=bool)
    real_face_mask[:n_faces] = True

    if len(extra_faces):
        logger.info(
            f"Sharp-corner splitting: added {len(extra_faces)} bevel/cap "
            f"triangle(s) to keep the split surface watertight"
        )

    return new_nodes, topology_faces, real_face_mask, orig_of_node, bevel_source_face
