"""棱柱->四面体的降级拆分 (从 mesh_repair_nonmanifold_mixed.py 拆分)。

从 mesh_repair_nonmanifold_mixed.py 拆出来（该文件原有 440 行，超过
400 行硬性拆分阈值）：`_split_prisms_to_tets` + `demote_invalid_prisms_to_tets`
只互相依赖、不依赖同文件里 `patch_nonmanifold_cavity_mixed` 的任何状态，
是清晰的拆分边界（`patch_nonmanifold_cavity_mixed` 反过来从这里导入
`_split_prisms_to_tets`，因为它也要用同一套拆分规则）。纯代码搬移，
不改变任何行为。
"""

from typing import Tuple

import numpy as np
from loguru import logger


# 棱柱 (v0,v1,v2,w0,w1,w2) 正好拆分为这 3 个四面体——与
# convert_layers_to_tetrahedra 使用的相同对角线一致性规则，
# 因此此处棱柱的边界面与为该板层产生的完全位相同，
# 因此与仍邻接此补丁的任何未拆分邻居（棱柱或四面体）自动
# 共形——前提是 v0<v1<v2 按全局节点索引（底面三角形自身的
# 顶点排序，相同的行排列带到顶面使 w_i 仍在 v_i "上方"——
# convert_layers_to_prisms 首次构建棱柱时的自身约定）。
#
# 该前提在下游节点重映射后不成立：mesh_background.
# generate_hybrid_mesh 在构建棱柱后多次调用
# _dedupe_coincident_points（seam 合并，最终防御遍），
# 每次都可能将节点的全局索引重新分配给其重合点组的
# 任意代表——该重映射中没有任何东西保持"v0 的新索引 <
# v1 的新索引 < v2 的新索引"，仅因为它对旧索引成立。
# 已直接确认，非理论：在未重新排序的真实重映射后棱柱上
# 调用此函数在临时诊断脚本中产生了约 23,000 个幻影"非流形"
# 面组——face_extractor.repair_nonmanifold_mixed 自身的
# _build_prism_face_occurrences（每次调用都重新排序，参见
# 其自身文档字符串）在完全相同的网格上找到零个，证明
# 约 23,000 完全是此函数缺失排序的产物，而非真实缺陷。
# 此处无条件排序（便宜，无论调用方输入是否已排序都正确）
# 既是修复也是未来的安全默认值。
def _split_prisms_to_tets(prisms: np.ndarray) -> np.ndarray:
    bottom = prisms[:, 0:3]
    top = prisms[:, 3:6]
    order = np.argsort(bottom, axis=1)
    row_idx = np.arange(len(prisms))[:, None]
    sb = bottom[row_idx, order]
    st = top[row_idx, order]
    v0, v1, v2 = sb[:, 0], sb[:, 1], sb[:, 2]
    w0, w1, w2 = st[:, 0], st[:, 1], st[:, 2]
    return np.concatenate([
        np.stack([v0, v1, v2, w2], axis=1),
        np.stack([v0, v1, w1, w2], axis=1),
        np.stack([v0, w0, w1, w2], axis=1),
    ], axis=0)


def demote_invalid_prisms_to_tets(
    prism_cells: np.ndarray,
    bl_cell_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """保证没有导出的 CPENTA 引用同一节点两次。

    "折叠角"棱柱（增长恰好在某个底顶点冻结，v_i == w_i——参见
    quality_metrics.compute_prism_aspect_ratios 自身文档字符串）
    按本项目自身的容差是有效的非零体积单元，但作为 CPENTA 记录
    它在 6 个槽中的两个里重复了一个 GRID id——按 Nastran 自身
    定义是格式错误的元素，不仅是质量问题。已直接对照真实
    cube_demo 导出确认：ANSA 21.0.1 拒绝了约 21,000 条此类
    CPENTA 记录（"invalid node combination"），每条对应一个
    折叠角棱柱，这正是导入网格中出现报告"空"补丁的原因——
    而非本项目早期追踪的小四面体体积缺口（参见 ProjectFiles
    Part10 P39）。

    patch_nonmanifold_cavity_mixed（mesh_background.py 中的
    长细比修复遍，已通过 prism_ar <= 500.0 将每个这样的棱柱
    路由通过它）是尽力 tetgen 重铺，静默留下没有簇被"接受"的
    空腔不变——且 tetgen 在从近零体积几何构建的空腔上可靠
    失败或被跳过，正好是折叠角棱柱边界的情况。已直接确认：
    在同一真实导出上，100% 的约 21,000 个标记棱柱仍然呈现、
    未修补、带有原始重复节点 id，尽管基于长细比的补丁已运行。
    此函数是确定性兜底，无失败模式：折叠棱柱通过相同
    对角线一致性规则拆分为正好 3 个四面体，本模块其他
    地方都使用该规则（_split_prisms_to_tets），其中正好
    重复引用该节点两次的那个是退化的并被丢弃；另外 2 个
    是覆盖相同体积的普通、有效、非退化四面体——纯算术，
    不会像 tetgen 调用那样失败。

    Args:
        prism_cells: (n_prism, 6) 棱柱连接关系
        bl_cell_groups: (n_prism,) 与 prism_cells 平行的字符串数组——
            每个降级棱柱的组名称直接由其存活的四面体继承
            （作为 `cell_groups`/`direct_cell_groups`），因此
            棱柱所属的壁面边界组不会丢失。

    Returns:
        (new_prism_cells, new_bl_cell_groups, extra_tets, extra_tet_groups)
        ——无需降级时 extra_tets/extra_tet_groups 为空数组（非 None），
        因此调用方可以始终无条件地 np.vstack/np.concatenate 它们到
        merged_cells/cell_groups 上。
    """
    empty_tets = np.empty((0, 4), dtype=prism_cells.dtype)
    empty_groups = np.empty((0,), dtype=object)
    if len(prism_cells) == 0:
        return prism_cells, bl_cell_groups, empty_tets, empty_groups

    has_dup = np.zeros(len(prism_cells), dtype=bool)
    for i in range(6):
        for j in range(i + 1, 6):
            has_dup |= prism_cells[:, i] == prism_cells[:, j]

    if not has_dup.any():
        return prism_cells, bl_cell_groups, empty_tets, empty_groups

    bad_idx = np.flatnonzero(has_dup)
    split_tets = _split_prisms_to_tets(prism_cells[bad_idx])  # (3*n_bad, 4), block layout: all T1s, then T2s, then T3s
    degenerate = (
        (split_tets[:, 0] == split_tets[:, 1]) | (split_tets[:, 0] == split_tets[:, 2]) |
        (split_tets[:, 0] == split_tets[:, 3]) | (split_tets[:, 1] == split_tets[:, 2]) |
        (split_tets[:, 1] == split_tets[:, 3]) | (split_tets[:, 2] == split_tets[:, 3])
    )
    valid_tets = split_tets[~degenerate]
    # np.tile (not np.repeat) matches _split_prisms_to_tets' block layout -
    # row r of the (3*n_bad,4) output belongs to source prism bad_idx[r % n_bad].
    source_idx = np.tile(bad_idx, 3)[~degenerate]

    logger.warning(
        f"{len(bad_idx)} prism(s) with a duplicate node id among their own 6 "
        f"vertices (collapsed-corner, invalid as a CPENTA record) - demoting "
        f"to {len(valid_tets)} plain tet(s), the deterministic fallback for "
        f"whatever the tetgen-based aspect-ratio patch above did not resolve"
    )

    keep_mask = ~has_dup
    return (
        prism_cells[keep_mask],
        bl_cell_groups[keep_mask],
        valid_tets.astype(prism_cells.dtype),
        bl_cell_groups[source_idx],
    )
