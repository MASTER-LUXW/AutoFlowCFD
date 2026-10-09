"""阶段 B'：生成后体网格修复的局部空腔重铺。

remesh_core_cavity 仅对仍然不合格的单元（经过阶段 A 平滑和阶段 B 的
BL 厚度上限，均在 mesh_repair.py / mesh_repair_bl_thickness.py 中）加上
一圈良好邻居单元的缓冲区进行局部重新四面体化，而不是微调节点（阶段 A）
或重新生成整个网格。提取空腔自身的边界（用 `n_buffer_rings` 圈良好单元
做缓冲，使空腔的新边界位于已经良好的区域中，而非穿过已退化的单元），
仅将这个小的封闭壳体交给独立的 tetgen 调用（nobisect=True，无竞争区域/
体积约束——与本包核心填充在所有其他地方使用的同一默认、已验证可靠的
tetgen 用法相同），然后将结果拼回被移除单元的位置。这在结构上不可能像
旧的核心侧区域方法那样向外泄漏细化（见 mesh_repair_bl_thickness.py 的
历史说明），因为空腔自己的 tetgen 调用完全看不到域的其余部分——没有东西
可以泄漏进去。对替换单元有严格的质量改进门控——如果局部重铺实际上没有
帮助，则保留原始单元，算例继续进入下一个修复阶段。

作用域（核心单元 vs. 接触自身壁面的 BL 单元）记录在 remesh_core_cavity
自己的文档字符串中，此处不重复。

从 mesh_repair.py 拆分纯粹为了控制文件大小——通过 mesh_repair.py 底部
重新转出，让现有调用方继续不受影响地使用。

进一步拆分：cavity 扩张/边界提取/质量评分的共享工具在
mesh_repair_cavity_shared.py；patch_nonmanifold_cavity（非流形修补，与
remesh_core_cavity 共用同一套局部重新四面体化技巧，但用途不同）在
mesh_repair_nonmanifold_patch.py。两者都在本文件重新转出，外部代码一律
仍从 `mesh_repair_cavity` 导入即可。

remesh_core_cavity 自身体积过大（超过 400 行上限），其中"逐簇尝试局部
重新四面体化"的循环体（占了函数体的大半）进一步拆到了
mesh_repair_cavity_cluster_attempt.py 的 _attempt_cavity_retile_clusters，
本文件里的 remesh_core_cavity 只保留候选簇的连通分量计算和最终拼接。
"""

from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

from .mesh_repair_cavity_shared import (
    _CAVITY_FACE_TEMPLATES,
)
from .mesh_repair_cavity_cluster_attempt import _attempt_cavity_retile_clusters
from .mesh_repair_nonmanifold_patch import patch_nonmanifold_cavity

if TYPE_CHECKING:
    from ...schema.grid_faces import FaceData
    from ...validation.quality_validator import MeshQualityValidator

__all__ = [
    '_CAVITY_FACE_TEMPLATES',
    'remesh_core_cavity',
    'patch_nonmanifold_cavity',
]


def remesh_core_cavity(
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: 'FaceData',
    bad_cell_mask: np.ndarray,
    validator: 'MeshQualityValidator',
    n_buffer_rings: int = 1,
    max_cavity_cells: int = 20_000,
    max_clusters_attempted: int = 15_000,
    max_seconds: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """阶段 B'：对仍然不合格的单元（加上良好邻居缓冲区）按其自身固定边界
    局部重新四面体化，而非微调节点（阶段 A）或重新生成整个网格（阶段 B，
    BL 侧）。参见本模块文档字符串了解为何这在结构上避免了旧核心侧区域
    方法的失效模式。

    最初仅限核心单元；在真实情况测量表明 BL/角点相邻的失效模式是
    本功能旨在补充的（厚度封顶，阶段 B）完全无法触及锐凸边单元后，
    扩展到也覆盖 BL 单元（包括接触其挤出来源壁面的——见下方）：
    阶段 A 拒绝移动壁面节点（正确——它是物理几何），阶段 B 只缩短
    BL 柱，无法重新铺网已生成的形状。直接的局部重铺可以。

    作用域：全部单元都合格，包括接触物理外边界面的单元。空腔的边界面（含其中的物理外边界面）在局部
    重铺中被逐字保留——tetgen 不带区域调用时 nobisect=True，接受前还要通过 `retile_is_conformal`
    逐面核对——外表面不变，最终的 map_generated_boundaries（外部面按几何包含关系对应输入面网格）照常给出
    边界组。2026-10-09 以前接触外边界的核心单元被排除，理由是"可能携带 tetgen 面标记"；面标记机制已删除，
    这条排除随之删除。

    Args:
        nodes, cells: 完整合并网格（阶段 A 后）
        faces: 已从此确切 (nodes, cells) 对提取的 FaceData（调用方
            自身的修复前或阶段 A 输出提取）
        bad_cell_mask: (n_cells,) bool，哪些单元在阶段 A 后仍不合格
        validator: 重用于其逐单元偏斜度/面诊断方法和阈值，以门控
            接受（见下方）
        n_buffer_rings: 提取边界前用多少圈面邻接良好单元填充每个空腔
        max_cavity_cells: 安全上限——这么大的空腔表明坏区域太广泛，
            "局部"补丁不再有意义（且 tetgen 成本不再像廉价局部操作）；
            跳过而非尝试
        max_clusters_attempted: 本次调用将尝试的独立空腔簇总数上限，
            无论存在多少候选簇。每个簇是其自身独立的 fill_core_volume
            （tetgen）调用——单独便宜，但 bad_cell_mask 在从广泛几何
            缺陷喂入时可能合法包含数万分散、断开的小/单簇（已直接
            观察到：mesh_overlap_check.py 在锐角密集的真实网格上标记
            单元——参见 mesh_background.py 自身将重叠单元折叠进
            bad_cell_mask 的注释）。顺序尝试所有簇，每个都有真实（虽小）
            的每调用开销，正是使此阶段在该情况下看似挂起的原因，而非
            任何单个簇慢。超过此上限的剩余坏单元保持原样，留给下一个
            运行的修复阶段（阶段 B 的 BL 厚度封顶，或阶段 C 的全局
            回退）——与此函数中每个其他上限一致（大小、质量门控拒绝）：
            优雅回退，从不是硬失败。从早期 2,000 提高到 15,000，在
            真实锐边密集情况（cube_demo）产生 9,013 候选簇且旧上限
            静默留下 7,013 个完全未尝试之后，远在质量门控拒绝问题
            （见下方）进入视野之前——离线网格生成已有分钟可花在此处，
            且每个附加簇尝试都便宜。
        max_seconds: 本次调用自身总挂钟时间的安全上限，在簇间检查
            （不中断已在进行中的）。纯簇数上限在簇大小/tetgen 难度
            差异很大时本身不能约束成本；这是该情况的第二道独立安全网。
            与 max_clusters_attempted 一起从早期 90 秒提高，原因相同。

    Returns:
        (new_nodes, new_cells, new_bad_cell_mask, action_log) - 如果未找到合格空腔或每个候选空腔都未通过接受
        门控则全部不变（非副本）。new_bad_cell_mask 将 bad_cell_mask
        跨应用于 new_cells 的相同单元移除/插入向前传递（每个新插入
        单元标记为 good=False，因为它已通过本函数自身的接受门控）——
        调用方的下一个修复阶段可直接使用它，代替从头重新验证整个网格。
    """
    actions: List[str] = []
    n_cells = len(cells)

    seed = bad_cell_mask
    if not np.any(seed):
        actions.append("Stage B': no bad cells - skipping")
        return nodes, cells, bad_cell_mask, actions

    interior_mask = faces.connectivity[:, 1] >= 0
    owner = faces.connectivity[interior_mask, 0]
    neighbor = faces.connectivity[interior_mask, 1]

    # 每个连通簇的合格种子单元成为其自身独立的空腔——
    # 跨越多个不相关坏口袋的单一组合空腔会（a）无理由超过
    # max_cavity_cells 的风险，且（b）不必要地对两个不相关口袋
    # 之间的良好单元重新四面体化。
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    seed_idx = np.flatnonzero(seed)
    seed_pos = -np.ones(n_cells, dtype=np.int64)
    seed_pos[seed_idx] = np.arange(len(seed_idx))
    edge_mask = seed[owner] & seed[neighbor]
    rows = seed_pos[owner[edge_mask]]
    cols = seed_pos[neighbor[edge_mask]]
    graph = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(seed_idx), len(seed_idx))
    )
    n_clusters, labels = connected_components(graph, directed=False)

    # 逐簇尝试局部重新四面体化的循环体拆到了
    # mesh_repair_cavity_cluster_attempt._attempt_cavity_retile_clusters
    # （原文件超过 400 行上限）——两阶段设计（先只读原始数组决定每个簇的
    # 取舍，最后才一次性拼接进新网格）的原因见本文件模块文档字符串顶部，
    # 这里不重复。
    accepted, claimed, n_skipped_size, n_rejected, n_failed, n_skipped_budget = (
        _attempt_cavity_retile_clusters(
            nodes, cells, bad_cell_mask, validator,
            seed_idx, labels, n_clusters,
            owner, neighbor,
            n_buffer_rings, max_cavity_cells, max_clusters_attempted, max_seconds,
        )
    )

    from ..tetgen.mesh_tetgen_core import repair_nonmanifold_cells

    if not accepted:
        actions.append(
            f"Stage B': {n_clusters} candidate cavity cluster(s) found, "
            f"none accepted (skipped_size={n_skipped_size}, rejected={n_rejected}, "
            f"failed={n_failed}, skipped_budget={n_skipped_budget})"
        )
        logger.info(
            f"Stage B': 0/{n_clusters} cavity cluster(s) remeshed "
            f"(skipped_size={n_skipped_size}, rejected={n_rejected}, "
            f"failed={n_failed}, skipped_budget={n_skipped_budget})"
        )
        return nodes, cells, bad_cell_mask, actions

    keep_mask = ~claimed
    new_nodes_parts = [nodes]
    new_cells_parts = [cells[keep_mask]]
    new_bad_parts = [bad_cell_mask[keep_mask]]
    interior_start = len(nodes)

    for res in accepted:
        n_boundary_pts = res['n_boundary_pts']
        global_pts = res['global_pts']
        retiled_nodes = res['retiled_nodes']
        retiled_tets = res['retiled_tets']

        def _remap(local_idx: np.ndarray, _global_pts=global_pts, _n_boundary=n_boundary_pts,
                   _offset=interior_start) -> np.ndarray:
            is_boundary = local_idx < _n_boundary
            out = np.empty_like(local_idx)
            out[is_boundary] = _global_pts[local_idx[is_boundary]]
            out[~is_boundary] = _offset + (local_idx[~is_boundary] - _n_boundary)
            return out

        new_tets_global = _remap(retiled_tets.ravel()).reshape(-1, 4).astype(cells.dtype)
        new_interior_nodes = retiled_nodes[n_boundary_pts:]

        new_nodes_parts.append(new_interior_nodes)
        new_cells_parts.append(new_tets_global)
        new_bad_parts.append(np.zeros(len(new_tets_global), dtype=bool))
        interior_start += len(new_interior_nodes)

        actions.append(
            f"Stage B': cavity of {len(res['cavity_idx'])} cells "
            f"({res['old_bad']} bad) -> retiled into {len(new_tets_global)} cells "
            f"({res['bad_new']} bad), {len(new_interior_nodes)} new interior point(s)"
        )

    new_nodes = np.vstack(new_nodes_parts)
    new_cells = np.vstack(new_cells_parts)
    new_bad_cell_mask = np.concatenate(new_bad_parts)

    # generate_hybrid_mesh 运行同样的检查一次，正好在初始
    # _build_merged_mesh 输出之后，但之后不再运行——因此本函数
    # 自身拼接引入的非流形重叠（例如两个接受的空腔的重铺在
    # 共享边界处恰好产生重叠四面体，与 repair_nonmanifold_cells
    # 自身文档字符串已记录为存在原因的同类缺陷）在调用方的下一个
    # FaceExtractor.extract_faces 调用之前未被捕获——该调用不仅
    # 会警告（该调用在非严格模式下已容忍 >2 单元面），还会在
    # 更严格的单独检查上硬崩溃：某单元最终完全不被任何面引用
    # （其 4 个面都是某个其他单元的"额外" >2 单元出现），
    # validate_face_data 无论严格性都视为致命。已直接确认，
    # 非理论：真实运行正好碰到这个（"Face connectivity references
    # N-6 cells, expected N"），在阶段 B' 迭代记录
    # "204 invalid (>2 cells)" 面之后。运行本函数自身调用方
    # 已对初始网格信任的相同清理使该不变量在本函数自身变异
    # 后也保持为真，而非留给下一次调用（致命地）发现。
    # 先尝试局部重铺，与 patch_nonmanifold_cavity 自身文档
    # 字符串相同原理：此处简单的"保留最大、丢弃其余"删除
    # 本身被发现留下真实孔洞——已直接确认，非理论，在其他
    # 两个 repair_nonmanifold_cells 调用点（mesh_background.py，
    # 两者都已修补）被证明不是真实 cube_demo 运行剩余
    # 0.147 m^3 缺口（原始 0.189 m^3）的来源之后——它追溯
    # 到正好是这个块。
    keep = repair_nonmanifold_cells(new_nodes, new_cells)
    if not keep.all():
        new_nodes, new_cells, new_bad_cell_mask = patch_nonmanifold_cavity(
            new_nodes, new_cells, keep,
            bad_cell_mask=new_bad_cell_mask,
        )
        keep = repair_nonmanifold_cells(new_nodes, new_cells)
        if not keep.all():
            n_removed = int(np.size(keep) - np.count_nonzero(keep))
            actions.append(f"Stage B': removed {n_removed} non-manifold cell(s) introduced by cavity splicing")
            new_cells = new_cells[keep]
            new_bad_cell_mask = new_bad_cell_mask[keep]

    logger.info(
        f"Stage B': {len(accepted)}/{n_clusters} cavity cluster(s) remeshed "
        f"(skipped_size={n_skipped_size}, rejected={n_rejected}, "
        f"failed={n_failed}, skipped_budget={n_skipped_budget})"
    )

    return new_nodes, new_cells, new_bad_cell_mask, actions
