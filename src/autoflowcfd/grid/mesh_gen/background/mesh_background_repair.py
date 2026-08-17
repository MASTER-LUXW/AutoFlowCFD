"""generate_hybrid_mesh 用到的、重复出现两次的非流形面修复 + 兜底逻辑。
generate_hybrid_mesh（mesh_background.py）在 seam 合并前、seam 合并后各跑一遍几乎一样的
"repair_nonmanifold_cells -> 不行就局部重铺（patch_nonmanifold_cavity）
-> 还不行就放大缓冲环再试一次 -> 实在不行就兜底删除"流程，抽成这一个共享函数，避免同一
段逻辑维护两份。"""

from typing import Tuple

import numpy as np
from loguru import logger

from ..tetgen.mesh_tetgen_core import repair_nonmanifold_cells
from ..repair.mesh_repair_cavity import patch_nonmanifold_cavity


def repair_nonmanifold_tets_with_escalation(
    nodes: np.ndarray,
    cells: np.ndarray,
    cell_groups: np.ndarray,
    n_bl_cells: int,
    context_suffix: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """修复非流形四面体：先局部重铺（填补简单"保留最大、丢弃其余"修复在额外
    单元来自两个不同区域在锐角处合法相遇而非真正重复时会留下的孔洞——参见
    patch_nonmanifold_cavity 自身文档字符串了解真实测量情况，0.189 m^3 的
    缺失体积，正是此功能的动机），在最终回退到简单删除之前先升级到更大缓冲环
    一次（会留下真实的孔洞——已直接确认：在真实 cube_demo 运行中，此处的
    无条件删除产生了断开的、仅四面体的"幻影"边界壳，包围着真正空的空间）。

    Args:
        nodes, cells: 当前合并网格
        cell_groups: (n_cells,) 与 cells 平行的字符串数组
        n_bl_cells: BL 单元占据 cells[:n_bl_cells]
        context_suffix: 追加到最终兜底删除警告消息中，用于在日志中区分
            seam 合并前和 seam 合并后的调用点

    Returns:
        (nodes, cells, cell_groups, n_bl_cells, changed) - changed 为 True
        当且仅当实际有东西被修复/删除，调用方据此知道是否需要重新计算单元体积。
    """
    nonmanifold_keep = repair_nonmanifold_cells(nodes, cells)
    if nonmanifold_keep.all():
        return nodes, cells, cell_groups, n_bl_cells, False

    nodes, cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
        nodes, cells, nonmanifold_keep, cell_groups, n_bl_cells,
    )
    # patch_nonmanifold_cavity 在无法安全修补时会返回输入不变
    # （仍为非流形）——在此情况下重新运行简单的保留掩码检查，与
    # 此修复存在之前相同，因此它无法修复的缺陷仍会被清理而非留在
    # 网格中。
    nonmanifold_keep = repair_nonmanifold_cells(nodes, cells)
    # 默认 n_buffer_rings=1 尝试无法重铺的簇通常只是需要更大、
    # 定义更好的局部边界，而非因为不可修复——在回退到简单删除之前
    # 先升级一次。
    if not nonmanifold_keep.all():
        nodes, cells, cell_groups, n_bl_cells, _ = patch_nonmanifold_cavity(
            nodes, cells, nonmanifold_keep, cell_groups, n_bl_cells,
            n_buffer_rings=4, max_cavity_cells=15_000,
        )
        nonmanifold_keep = repair_nonmanifold_cells(nodes, cells)
    if not nonmanifold_keep.all():
        n_deleted = int((~nonmanifold_keep).sum())
        del_pts = nodes[np.unique(cells[~nonmanifold_keep])]
        logger.warning(
            f"Non-manifold tet repair{context_suffix}: {n_deleted} cell(s) still unpatched "
            f"after retry with a larger buffer ring - deleting as a last resort "
            f"(bbox min={del_pts.min(axis=0)}, max={del_pts.max(axis=0)}); this "
            f"leaves a real gap at that location, not just missing volume"
        )
        n_bl_cells = int(np.sum(nonmanifold_keep[:n_bl_cells]))
        cells = cells[nonmanifold_keep]
        cell_groups = cell_groups[nonmanifold_keep]

    return nodes, cells, cell_groups, n_bl_cells, True
