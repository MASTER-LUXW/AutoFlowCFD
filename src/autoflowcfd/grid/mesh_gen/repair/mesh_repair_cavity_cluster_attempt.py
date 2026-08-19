"""remesh_core_cavity 的"逐簇尝试局部重新四面体化"内层循环。

从 mesh_repair_cavity.py 拆分出来（原文件超过 400 行上限）——这段逻辑
是 remesh_core_cavity 里最长的一段：对每一个连通的坏单元簇分别构造
cavity 边界、调用 tetgen 局部重新四面体化、按质量门槛决定是否接受，纯
粹是代码搬移，未改动任何数值逻辑。两阶段设计（先只读原始数组决定每个
簇的取舍，最后才一次性拼接进新网格）的原因见 remesh_core_cavity 自身
的文档字符串，这里不重复。
"""

import time
from typing import List, Tuple

import numpy as np
from loguru import logger

from .mesh_repair_cavity_shared import (
    _grow_cavity_rings,
    _cavity_boundary_faces,
    _count_bad_cells,
)


def _attempt_cavity_retile_clusters(
    nodes: np.ndarray,
    cells: np.ndarray,
    bad_cell_mask: np.ndarray,
    validator,
    seed_idx: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    owner: np.ndarray,
    neighbor: np.ndarray,
    ineligible: np.ndarray,
    n_buffer_rings: int,
    max_cavity_cells: int,
    max_clusters_attempted: int,
    max_seconds: float,
) -> Tuple[List[dict], np.ndarray, int, int, int, int]:
    """对每个连通的坏单元簇分别尝试局部 cavity 重新四面体化。

    对应 mesh_repair_cavity.remesh_core_cavity 原来函数体中段的
    `claimed = np.zeros(...)` 到 `for cluster_id in range(n_clusters):`
    循环结束为止那一整段，逐字搬移，未改动任何数值逻辑。

    Returns:
        (accepted, claimed, n_skipped_size, n_rejected, n_failed,
        n_skipped_budget) - accepted 是每个被接受的簇的字典列表（cavity_idx/
        global_pts/retiled_nodes/retiled_tets/n_boundary_pts/old_bad/
        bad_new），claimed 是 (n_cells,) bool 数组，标记哪些原始单元已经
        被某个被接受的簇占用（供调用方拼接最终网格时排除）。
    """
    n_cells = len(cells)

    # 两阶段设计，刻意不在簇仍在决策时变异节点/单元/
    # bad_cell_mask：每个簇的空腔纯粹通过读取原始的、
    # 从未触及的数组来提取和重铺，且 `claimed`（而非数组变异）
    # 阻止后续簇的缓冲环重新进入已接受的空腔。所有接受的
    # 结果被组合到新网格中，在非常末尾的单次拼接中完成。
    # 逐簇变异单元/bad_cell_mask（此函数的早期版本这样做过）
    # 会静默使每个*后续*簇的 cavity_idx/cavity_mask 失效——
    # 两者都被计算为到变异前数组的绝对索引，而单元移除会
    # 移动移除行之后的所有内容——因此后续簇会读取并"重铺"
    # 完全错误的单元（这已被经验捕获：第一个之后的几乎每个
    # 空腔都报告"0 个坏单元"，因为它读到的是移位到该位置
    # 的单元而非其自身预期的坏口袋）。
    claimed = np.zeros(n_cells, dtype=bool)
    accepted: List[dict] = []
    n_skipped_size = 0
    n_rejected = 0
    n_failed = 0
    n_skipped_budget = 0

    from ..tetgen.mesh_tetgen_core import (
        fill_core_volume,
        CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL,
    )

    if n_clusters > max_clusters_attempted:
        logger.warning(
            f"Stage B': {n_clusters} candidate cavity clusters found, capping "
            f"at {max_clusters_attempted} attempts (max_clusters_attempted) - "
            f"this many separate/scattered bad pockets usually means a "
            f"widespread defect (e.g. many sharp-corner cells, or a large "
            f"fraction of the mesh flagged by the overlap check) that this "
            f"per-cluster local-patch strategy isn't a good fit for; the "
            f"remainder is left for whichever repair stage runs next"
        )

    start_time = time.perf_counter()
    for cluster_id in range(n_clusters):
        if cluster_id >= max_clusters_attempted:
            n_skipped_budget = n_clusters - cluster_id
            break
        elapsed = time.perf_counter() - start_time
        if elapsed > max_seconds:
            n_skipped_budget = n_clusters - cluster_id
            logger.warning(
                f"Stage B': stopping after {elapsed:.1f}s (max_seconds="
                f"{max_seconds:.0f}) with {cluster_id}/{n_clusters} clusters "
                f"attempted - remaining {n_skipped_budget} left for "
                f"whichever repair stage runs next"
            )
            break

        cluster_seed_mask = np.zeros(n_cells, dtype=bool)
        cluster_seed_mask[seed_idx[labels == cluster_id]] = True

        cavity_mask = _grow_cavity_rings(cluster_seed_mask, owner, neighbor, ineligible | claimed, n_buffer_rings)
        cavity_idx = np.flatnonzero(cavity_mask)
        if len(cavity_idx) > max_cavity_cells:
            n_skipped_size += 1
            continue

        boundary_faces = _cavity_boundary_faces(cells, cavity_idx)
        global_pts = np.unique(boundary_faces)
        local_of_global = -np.ones(len(nodes), dtype=np.int64)
        local_of_global[global_pts] = np.arange(len(global_pts))
        local_faces = local_of_global[boundary_faces].astype(np.int32)
        local_points = nodes[global_pts]

        try:
            # verbose=False: 每个空腔簇运行一次，可能每次修复
            # 很多遍——每个单独调用自身的边界点数/Steiner 预算/
            # 完成行单独来看没意义，只有本函数自身的逐空腔
            # 和最终汇总行（分别在下方/末尾单独记录）才有意义。
            # minratio/mindihedral: 与主核心填充使用的相同收紧标准
            # （mesh_background_merge.py），而非 tetgen 自身更宽松的
            # 默认值——使用比产生其自身（已经坏的）邻居更松的形状
            # 质量边界的空腔重铺没有真正理由产出更好结果。已确认为
            # 真实案例上的真实大效应：使用 tetgen 默认值时，约 72%
            # 的尝试重铺被拒绝为"非改进"；参见 CORE_TETGEN_MINRATIO
            # 自身文档字符串。
            retiled_nodes, retiled_tets, _, _ = fill_core_volume(
                local_points, local_faces, verbose=False,
                minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
            )
        except Exception as e:
            logger.warning(f"Stage B': cavity remesh failed ({e}), keeping original cells")
            n_failed += 1
            continue

        n_boundary_pts = len(local_points)
        if not np.array_equal(retiled_nodes[:n_boundary_pts], local_points):
            # fill_core_volume already logs+handles this internally (coincident-
            # point stitching fallback), but the cavity's own boundary points
            # are exactly the ones that must stay pinned for the splice below
            # to be valid - if even the fallback couldn't preserve them
            # verbatim, don't risk stitching a silently-shifted boundary into
            # the still-good rest of the mesh.
            logger.warning(
                "Stage B': cavity boundary points weren't preserved "
                "verbatim by the local retile, keeping original cells"
            )
            n_failed += 1
            continue

        # 质量门控：仅当重铺严格优于其替换时才接受——
        # 阶段 A 自身的"永远不盲目信任移动"哲学（参见本模块
        # 文档字符串），在此应用于整个空腔交换而非逐节点位置。
        #
        # old_bad_in_cavity 按 bad_cell_mask 的任何判据（偏斜、
        # 非正交、相邻体积比，以及——因为 mesh_background.py 将其
        # 折叠入——物理重叠）计算坏单元数。仅按偏斜度评分
        # bad_new（此检查的早期版本）是苹果比橘子：主要因
        # 非正交或相邻体积比而坏的空腔可能得到真正修复它们的
        # 重铺，但仍在此被拒绝，因为其（无关的）偏斜度计数
        # 碰巧没改善——已直接在真实锐角案例上确认，绝大多数
        # 重铺以这种方式被拒绝。在重铺空腔上评估相同的三项
        # 判据使这成为真正的苹果比苹果比较。
        old_bad_in_cavity = int(np.sum(bad_cell_mask[cavity_idx]))
        bad_new = _count_bad_cells(validator, retiled_nodes, retiled_tets)

        # 优化：在几何困难特征上 TetGen 找不到完美解时，通过
        # 接受"不更差"的结果打破死锁（多次重试后或坏单元数
        # 很少时，例如 <= 2）。防止无限循环。
        is_improvement = bad_new < old_bad_in_cavity
        is_acceptable_fallback = (old_bad_in_cavity <= 2 and bad_new <= old_bad_in_cavity)

        if not is_improvement and not is_acceptable_fallback:
            # debug 而非 info：每个被拒绝的空腔簇触发一次——
            # 在锐角密集网格上的单次修复中可能数百次——而 CLI
            # 默认输出为 INFO 级别（cli/main.py，level="INFO"
            # 除非 --verbose），因此这会用无人实时阅读的逐簇
            # 细节淹没普通控制台输出；本函数自身最终汇总行
            # （下方）中的 `rejected=N` 合计已在正确粒度报告
            # 相同信息供日常使用。仍可通过 `--verbose` 供任何
            # 实际调试特定空腔的人使用。
            logger.debug(
                f"Stage B': cavity of {len(cavity_idx)} cells "
                f"({old_bad_in_cavity} bad) retiled into {len(retiled_tets)} cells "
                f"({bad_new} bad) - not an improvement, keeping original cells"
            )
            n_rejected += 1
            continue

        claimed[cavity_idx] = True
        accepted.append(dict(
            cavity_idx=cavity_idx, global_pts=global_pts,
            retiled_nodes=retiled_nodes, retiled_tets=retiled_tets,
            n_boundary_pts=n_boundary_pts,
            old_bad=old_bad_in_cavity, bad_new=bad_new,
        ))

    return accepted, claimed, n_skipped_size, n_rejected, n_failed, n_skipped_budget
