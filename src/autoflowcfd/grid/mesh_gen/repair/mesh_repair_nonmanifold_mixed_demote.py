"""棱柱 -> 四面体降级：导出前把引用同一节点两次的棱柱（折叠角）换成等体积的四面体（`demote_invalid_prisms_to_tets`）。"""

from typing import Tuple

import numpy as np
from loguru import logger


# 棱柱 -> 3 个四面体的拆分（块排列）：唯一实现在 mesh_gen/utils/prism_split.py（对角线只依赖侧面四边形自己
# 的全局编号，节点重映射后照样协调）。
from ..utils.prism_split import split_prisms_to_tets as _split_prisms_to_tets  # noqa: E402


def _split_collapsed_corner_to_2_tets(prisms: np.ndarray, corner: int) -> np.ndarray:
    """"折叠角"棱柱（恰好一个顶点 v_c == w_c，无其他重复）的精确 2-四面体
    分解——取代通用 3-way 拆分再丢弃退化件的做法。

    几何：棱柱的 3 个侧面里，与折叠角相邻的 2 个侧面本身也随之退化为
    三角形（因为它们各自的 4 个角里有一对已重合），只有"对面"（不
    触碰折叠角的那个侧面）仍是一个完好的四边形 quad = v_a,v_b,w_b,w_a
    （沿用棱柱自身"v_i,v_{i+1},w_{i+1},w_i"侧面缠绕约定）。折叠角本身
    的顶点 apex = v_c(=w_c) 与这个 quad 一起，精确构成一个四棱锥
    （quad 底面 + apex 顶点）——这正是折叠棱柱退化后的真实立体形状，
    5 个不同顶点，不是 6 个。

    四棱锥沿 quad 的一条对角线（v_a—w_b）恰好精确、无残留地分解为 2 个
    四面体。

    与通用 3 路拆分丢弃退化候选后剩下的 2 个四面体逐节点集合相同（只差顶点顺序，绕向由调用方
    `orient_tetrahedra` 统一），已用符号推导与 cube_demo 实测（全部质量指标逐位不变）核实；这里用显式公式代替
    "恰好丢对"的隐式正确性，并省去第 3 个候选的构造。（曾怀疑折叠角降级是 cube_demo 相邻体积比 336.57 的 sliver
    来源，已证伪，见 ProjectFiles/V2.0/40_网格阶段D-BL与核心界面体积比修复-排查记录.md。）

    Args:
        prisms: (n, 6) 棱柱连接关系，每行恰好满足 prisms[:,corner] ==
            prisms[:,corner+3] 且无其他重复顶点对（调用方保证）。
        corner: 0/1/2，折叠角在棱柱自身编号中的索引。

    Returns:
        (2*n, 4) 四面体连接关系，block 布局：前 n 行是每个输入棱柱的
        第一个子四面体，后 n 行是第二个（与 _split_prisms_to_tets 的
        block 布局约定一致，调用方可用同一种 np.tile(idx, 2) 方式
        推导来源）。
    """
    a, b = [k for k in range(3) if k != corner]
    v_a, v_b = prisms[:, a], prisms[:, b]
    w_a, w_b = prisms[:, a + 3], prisms[:, b + 3]
    apex = prisms[:, corner]  # == prisms[:, corner+3]，调用方已保证
    # quad 顶点按 v_a -> v_b -> w_b -> w_a 的原侧面缠绕顺序，对角线取
    # v_a—w_b：三角形 (v_a,v_b,w_b) 和 (v_a,w_b,w_a)，各自加 apex。
    tet1 = np.stack([v_a, v_b, w_b, apex], axis=1)
    tet2 = np.stack([v_a, w_b, w_a, apex], axis=1)
    return np.concatenate([tet1, tet2], axis=0)


def demote_invalid_prisms_to_tets(
    prism_cells: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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

    折叠棱柱本身覆盖的是真实、非零的物理体积（BL 挤出过程里，该角
    的增长提前冻结，不是几何缺陷或重复单元）——因此正确的处理方式
    是把它转换成等体积的四面体表示，而不是直接删除（删除会在域内
    留下一个没有任何单元覆盖的真实空洞，比退化单元更严重：面提取
    会在那里产生虚假的内部边界，或者干脆让相邻单元的通量计算漏掉
    这部分体积，是真实的守恒违反）。

    分两种情况处理：
    1. **精确的单角折叠**（v_c == w_c 恰好一对，无其他重复）：用
       `_split_collapsed_corner_to_2_tets` 直接构造 2 个四面体（四棱锥的标准分解，体积 100% 覆盖，与通用拆分
       的结果相同，见该函数文档）。这是绝大多数真实折叠角棱柱的情况。
    2. **更复杂/非预期的重复模式**（多对重复，或重复不在 v_i/w_i
       竖直对上）：退回旧有的通用 3-way 拆分 + 丢弃恰好退化的那个
       子四面体——通用公式对这类不常见形状仍然正确（纯算术，不会
       像 tetgen 调用那样失败）；这类情况按本模块的历史实测记录
       （ProjectFiles Part10 P39）远少于情况 1。

    Args:
        prism_cells: (n_prism, 6) 棱柱连接关系

    Returns:
        (new_prism_cells, extra_tets)——无需降级时 extra_tets 为空数组（非 None），
        因此调用方可以始终无条件地 np.vstack 到 merged_cells 上。
    """
    empty_tets = np.empty((0, 4), dtype=prism_cells.dtype)
    if len(prism_cells) == 0:
        return prism_cells, empty_tets

    # 竖直对（折叠角的标志）与全部 15 对的重复计数分开统计，用于区分
    # "干净的单角折叠"（情况 1）与其他重复模式（情况 2）。
    vertical_match = np.stack(
        [prism_cells[:, k] == prism_cells[:, k + 3] for k in range(3)], axis=1
    )  # (n, 3)
    total_dup_pairs = np.zeros(len(prism_cells), dtype=np.int32)
    for i in range(6):
        for j in range(i + 1, 6):
            total_dup_pairs += (prism_cells[:, i] == prism_cells[:, j]).astype(np.int32)

    has_dup = total_dup_pairs > 0
    if not has_dup.any():
        return prism_cells, empty_tets

    n_vertical = vertical_match.sum(axis=1)
    clean_single_collapse = has_dup & (total_dup_pairs == 1) & (n_vertical == 1)
    messy = has_dup & ~clean_single_collapse

    tet_parts = []

    if clean_single_collapse.any():
        collapsed_corner = np.argmax(vertical_match, axis=1)  # 仅在 clean 行上有意义
        for c in range(3):
            rows = np.flatnonzero(clean_single_collapse & (collapsed_corner == c))
            if len(rows) == 0:
                continue
            two_tets = _split_collapsed_corner_to_2_tets(prism_cells[rows], c)
            tet_parts.append(two_tets)
        logger.warning(
            f"{int(clean_single_collapse.sum())} prism(s) with a single collapsed "
            f"corner (v_i == w_i, invalid as a CPENTA record) - demoting to "
            f"{2 * int(clean_single_collapse.sum())} plain tet(s) via exact "
            f"quad-pyramid decomposition (no discarded degenerate piece, "
            f"volumes stay balanced - see _split_collapsed_corner_to_2_tets docstring)"
        )

    if messy.any():
        bad_idx = np.flatnonzero(messy)
        split_tets = _split_prisms_to_tets(prism_cells[bad_idx])  # (3*n_bad,4), block layout
        degenerate = (
            (split_tets[:, 0] == split_tets[:, 1]) | (split_tets[:, 0] == split_tets[:, 2]) |
            (split_tets[:, 0] == split_tets[:, 3]) | (split_tets[:, 1] == split_tets[:, 2]) |
            (split_tets[:, 1] == split_tets[:, 3]) | (split_tets[:, 2] == split_tets[:, 3])
        )
        valid_tets = split_tets[~degenerate]
        logger.warning(
            f"{len(bad_idx)} prism(s) with an unexpected duplicate-vertex pattern "
            f"(not a single clean corner collapse) - falling back to generic "
            f"3-way split + discard-the-degenerate-one, demoting to "
            f"{len(valid_tets)} plain tet(s)"
        )
        tet_parts.append(valid_tets)

    extra_tets = np.concatenate(tet_parts, axis=0) if tet_parts else empty_tets

    keep_mask = ~has_dup
    return prism_cells[keep_mask], extra_tets.astype(prism_cells.dtype)
