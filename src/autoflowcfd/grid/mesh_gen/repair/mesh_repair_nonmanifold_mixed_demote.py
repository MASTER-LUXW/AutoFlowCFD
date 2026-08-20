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

    **重要更正（诚实记录一次证伪，避免下一轮重复走这条已证明无效的
    路）**：最初怀疑旧版"通用 3-way 拆分、丢弃恰好重复引用同一节点的
    第 3 个候选"会产生病态的不均衡子四面体对（一个几乎占满、一个
    近零体积），是本函数存在的动机。但已用符号推导+数值核对严格证明：
    对单角折叠的全部 3 种情形，通用拆分丢弃退化候选后剩下的 2 个
    候选，与本函数直接构造的 2 个四棱锥子四面体，是**逐节点集合完全
    相同**的两个四面体（只是同一四面体的 4 个顶点写入顺序可能不同，
    不影响形状/体积，只影响绕向符号，而绕向本来就会被调用方
    `orient_tetrahedra` 统一处理）——即通用拆分对这个特定简并情形早已
    是精确解，不存在实际的体积不均衡 bug。已在 cube_demo 上实测验证：
    换用本函数前后，全部质量指标（含 adjacent_volume_ratio_max）
    逐位不变。

    因此本函数相对旧版通用拆分**不改变任何输出网格几何**，价值仅在于
    ①省去构造并丢弃第 3 个候选的计算 ②用推导式的显式公式取代"凑巧
    discard 对了"的隐式正确性，可读性更好、更不容易在未来被误改坏。
    cube_demo 上实测确认的 336.57 相邻体积比根因**不是**折叠角棱柱
    降级——已用逐面追踪定位到具体的最坏单元后确认它不触碰任何折叠角
    棱柱产生的四面体，是另一个尚未定位的、独立的近退化 sliver 来源
    （详见 mesh_repair_interface.py 模块文档"已知局限"一节的后续
    分析）。

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

    折叠棱柱本身覆盖的是真实、非零的物理体积（BL 挤出过程里，该角
    的增长提前冻结，不是几何缺陷或重复单元）——因此正确的处理方式
    是把它转换成等体积的四面体表示，而不是直接删除（删除会在域内
    留下一个没有任何单元覆盖的真实空洞，比退化单元更严重：面提取
    会在那里产生虚假的内部边界，或者干脆让相邻单元的通量计算漏掉
    这部分体积，是真实的守恒违反）。

    分两种情况处理：
    1. **精确的单角折叠**（v_c == w_c 恰好一对，无其他重复）：用
       `_split_collapsed_corner_to_2_tets` 直接构造 2 个四面体（四棱锥
       的标准分解，体积 100% 覆盖）。**注意**：已证明这与旧版通用
       3-way 拆分 + 丢弃退化候选，对这个特定简并情形产生逐节点相同的
       结果（见 `_split_collapsed_corner_to_2_tets` 文档字符串"重要
       更正"一节）——本路径不改变任何输出几何，只是更清晰/省一次
       无用计算。这是绝大多数真实折叠角棱柱的情况。
    2. **更复杂/非预期的重复模式**（多对重复，或重复不在 v_i/w_i
       竖直对上）：退回旧有的通用 3-way 拆分 + 丢弃恰好退化的那个
       子四面体——通用公式对这类不常见形状仍然正确（纯算术，不会
       像 tetgen 调用那样失败）；这类情况按本模块的历史实测记录
       （ProjectFiles Part10 P39）远少于情况 1。

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
        return prism_cells, bl_cell_groups, empty_tets, empty_groups

    n_vertical = vertical_match.sum(axis=1)
    clean_single_collapse = has_dup & (total_dup_pairs == 1) & (n_vertical == 1)
    messy = has_dup & ~clean_single_collapse

    tet_parts = []
    group_parts = []

    if clean_single_collapse.any():
        collapsed_corner = np.argmax(vertical_match, axis=1)  # 仅在 clean 行上有意义
        for c in range(3):
            rows = np.flatnonzero(clean_single_collapse & (collapsed_corner == c))
            if len(rows) == 0:
                continue
            two_tets = _split_collapsed_corner_to_2_tets(prism_cells[rows], c)
            tet_parts.append(two_tets)
            group_parts.append(np.tile(bl_cell_groups[rows], 2))
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
        source_idx = np.tile(bad_idx, 3)[~degenerate]
        logger.warning(
            f"{len(bad_idx)} prism(s) with an unexpected duplicate-vertex pattern "
            f"(not a single clean corner collapse) - falling back to generic "
            f"3-way split + discard-the-degenerate-one, demoting to "
            f"{len(valid_tets)} plain tet(s)"
        )
        tet_parts.append(valid_tets)
        group_parts.append(bl_cell_groups[source_idx])

    extra_tets = np.concatenate(tet_parts, axis=0) if tet_parts else empty_tets
    extra_groups = np.concatenate(group_parts, axis=0) if group_parts else empty_groups

    keep_mask = ~has_dup
    return (
        prism_cells[keep_mask],
        bl_cell_groups[keep_mask],
        extra_tets.astype(prism_cells.dtype),
        extra_groups,
    )
