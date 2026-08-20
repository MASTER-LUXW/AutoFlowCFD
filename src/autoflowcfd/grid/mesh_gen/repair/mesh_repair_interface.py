"""阶段 D：BL/core 界面相邻体积比定向修复。

## 背景（V2.0 专家组评审新发现的根因，取代此前 16+ 轮 BL 挤出参数调优）

`generate_hybrid_mesh`（mesh_background.py）里阶段 A（`run_stage_a_repair`）
和阶段 B/B'（`run_stage_b_repair`/`remesh_core_cavity`）的坏单元判据，
全程只作用于 `merged_cells`——这个数组**只是核心 tetgen 填充的四面体**，
从不包含棱柱（见 `mesh_background_merge_with_bl.py` 的返回签名：
`prism_cells`/`tet_cells` 是两个独立数组，`generate_hybrid_mesh` 里
`n_bl_cells` 绑定的是恒为 0 的 `n_transition_cells`，不是棱柱数量）。

而全网格唯一一次在**完整混合面图**（棱柱+四面体，`FaceExtractor.
extract_faces_mixed` 的全局单元索引：棱柱 `[0,n_prism)`、四面体
`[n_prism,n_prism+n_tet)`）上计算 `adjacent_volume_ratio_max` 的地方，
是 `quality_validator_mixed.py::validate_mixed_mesh`——只在 CLI 最终
质量报告里被调用一次，纯诊断，从不反馈给任何修复阶段。

结论：BL 最后一层棱柱与紧邻核心四面体之间的体积级差，从未被阶段
A/B/B' 的 badness 判据看到过——无论怎么调 BL 挤出参数（衰减曲线、
硬停止预算、CORE_FILL_VOLUME_CAP_FRACTION 等），这类缺陷都不会被
触及。这正是 cube_demo 相邻体积比长期停留在 336.57、对 16+ 轮 BL
挤出侧调优毫无反应的真实原因（详见 V2.0 专家组三次评审的网格生成
审计报告）。

## 本模块的修复方式与已知局限（务必先读，避免重复走过的弯路）

复用阶段 B' 的局部空腔重铺机制（`remesh_core_cavity`），坏单元判据
换成基于**完整混合面图**的判据：只要一个四面体与其相邻棱柱之间的
体积比超过质量门阈值，就标记为坏单元，交给 `remesh_core_cavity` 在
四面体一侧局部重新四面体化——棱柱侧（BL 层结构/厚度/节点）在整个
过程中不做任何改动。

**开发过程中的两个真实教训（已用 cube_demo 实测数据定位，供下一轮
接手者参考，不要重复验证）：**

1. `remesh_core_cavity` 自身基于**纯四面体子图**的 `touches_physical_
   boundary` 判定，会把"接触本函数刚标记为坏单元的棱柱接口"误判为
   "接触物理外部边界"（纯四面体视角下二者都表现为"这个面没有 tet
   邻居"，无法区分），导致全部候选单元被判定 ineligible（已实测复现：
   81 个候选全部被拒绝，"all touch an out-of-scope core boundary"）。
   修复方式：改用**混合面图**的边界面判定（`mixed_faces.
   get_boundary_face_indices()`，棱柱-四面体接口在混合图里正确显示
   为内部面）逐一识别真正接触物理外部边界的四面体，调用
   `remesh_core_cavity` 时传 `n_bl_cells=len(merged_cells)` 让它自身
   基于纯四面体子图的（对本场景系统性假阳性的）排除逻辑对整个数组
   失效，只依赖上面更准确的判定。

2. **更根本的发现**：用 cube_demo 实测数据核实这 81 个违规单元的
   体积比方向后，确认全部 100% 是"棱柱更大、四面体是近零体积
   sliver"（v_tet~5e-10 m^3 vs v_prism~1.8e-7 m^3，比值336.57），
   不是"四面体过大"。这类 sliver 与"2,677 个因坍缩角棱柱（自身6个
   顶点中有重复节点id）被降级为普通四面体"（见
   mesh_repair_nonmanifold_mixed_demote.py）在数量级和产生机制上高度
   吻合——退化的近重复节点直接产生近零体积四面体，不是网格粗细/
   过渡梯度问题。这意味着：
   - 曾尝试过的"质心细分把过大的违规单元压小"（第一版实现，已废弃）
     从根上就点错了方向——sliver 本来就是全场景里体积最小的单元之一，
     细分只会产生更小的子单元，不可能让比值变小，实测 81/81 全部
     "reached max subdivision depth without meeting target volume"
     （因为它们的体积从第一次检查就已经小于目标阈值，根本不会触发
     细分分支）。
   - 当前保留的空腔重铺方式方向正确（重铺可以把 sliver 替换成与其
     周围点云间距匹配、形状合格的新四面体，体积通常会显著增大），
     且实测确实会成功重铺相当一部分候选（例如 cube_demo 上 14 个
     空腔簇被接受，多个从"N 个坏单元"变为"0 个坏单元"）。**但
     `remesh_core_cavity` 的接受/拒绝门是 `MeshQualityValidator` 的
     通用逐单元形状指标（偏斜度/长宽比），不知道"必须把体积压到
     相邻这个具体棱柱的 N 倍以内"这个目标**——重铺后单元形状变好，
     体积未必真的向目标收敛到低于阈值，实测最终 `adjacent_volume_
     ratio_max` 在 cube_demo 上仍保持 336.57 不变（最坏的两个 sliver
     未被消除）。

3. **对第 2 条的证伪与更精确的定位（同一轮会话内完成，不要重复
   排查）**：最初怀疑"坍缩角棱柱降级"是这批 sliver 的根因，据此把
   `mesh_repair_nonmanifold_mixed_demote.py::demote_invalid_prisms_to_tets`
   的通用 3-way 拆分换成了专门的四棱锥精确 2-四面体分解
   （`_split_collapsed_corner_to_2_tets`）。**但已用符号推导严格证明
   两种拆分对单角折叠情形产生逐节点相同的输出**（该函数自身文档
   字符串"重要更正"一节有完整推导）——换用后 cube_demo 的全部质量
   指标（含 336.57）逐位不变，证实这条假设是错的：坍缩角棱柱降级
   从未真正产生体积不均衡的 sliver，通用拆分对这个情形本就是精确解。
   进一步用逐面追踪定位了 336.57 对应的具体最坏单元（owner 全局索引
   73812 即某棱柱，neighbor 全局索引 367385 即某四面体，v_tet=
   5.38e-10 vs v_prism=1.81e-7），核实它**不触碰任何物理外部边界**
   （对本阶段合格），但其四面体的 4 个顶点呈现两对几乎重合、仅在
   BL 挤出法向（z 分量）上有约 1.7e-3 差距的坐标——这不是坍缩角棱柱
   降级的产物，是一个当时尚未定位来源的独立近退化 sliver。

4. **对第 3 条"尚未定位来源"的完整溯源（后续会话完成，用插桩追踪
   脚本复现真实生产管线，未改任何生产逻辑）**：`split_sharp_corners`
   已直接排除。真正根因在 BL 挤出的**逐顶点独立**反应式冻结机制
   （`mesh_layer_step.py` 硬停止 + `mesh_front_collision.py::
   clamp_budget_for_convergence`/`freeze_self_colliding_nodes`）——
   同一张表面三角形的 3 个角点里，2 个因自碰撞预判在 layer1→layer2
   被砍掉约 60% budget，layer4 即耗尽硬冻结；第 3 个角点未被同样触发，
   一直长到 layer6-8。这在**孤立看每个顶点**时都是"正确"决策，但结果
   是同一张三角形的 BL 柱高被撕裂成不连续的——tetgen 核心填充只看
   最外层 outer_nodes slice 看不到这个撕裂，真正把它连接起来产生
   sliver 的正是本模块的跨类型非流形缝合修复（阶段 D 之前的
   `mesh_repair_nonmanifold_mixed.py::patch_nonmanifold_cavity_mixed`）：
   它只管拓扑连通、不管缝合产生的单元质量，用现有节点桥接这个撕裂
   缺口，桥出来的四面体两条边恰好是冻结前最后一步的~1.73mm 短边。
   下面第 a/b 两条路径的方向判断被这次追踪证实成立，且现在有了明确
   的第三条路径 c。

5. **第 19 轮攻关：路径 a 实测证伪（已回退），路径 b 实测有效（已实施）**：
   路径 a（挤出源头做三角形/前沿一致冻结）实现了 `enforce_triangle_
   height_consistency`（`mesh_front_collision.py`），在整个可测试阈值
   区间（三角形角点相对已冻结兄弟的高度领先量 0.3~2.0 倍层厚）内实测
   要么零效果（阈值过松，从未真正触发），要么让相邻单元体积比从
   336.57 恶化到 6939~12046（20-35 倍，阈值过紧时把大量正常、非撕裂
   的BL列也误判为需要提前冻结），要么直接导致 tetgen 拓扑崩溃（"3
   faces shared by more than 2 cells"）——从未观测到任何改善区间，已
   用 `git checkout --` 完整回退，`mesh_extrusion.py`/`mesh_front_
   collision.py` 不含这部分改动。**结论：三角形高度一致性耦合这个
   方向在这套挤出机制上不可行**，不要在未换一套完全不同设计思路前
   重新尝试同类耦合冻结。

   路径 b 实测有效，已实施（`mesh_repair_cavity_shared.py::_weld_
   near_coincident_boundary_points` + `patch_nonmanifold_cavity_mixed`
   里的事后 `_count_bad_cells` 质量门，见两处各自的文档字符串）：
   **单纯"重铺完再拒绝"式的质量门本身不够**——tetgen 做约束 Delaunay
   必须精确尊重给定边界点集，撕裂留下的近重合点对本身在边界里，
   任何重铺结果都绕不开在这两点间产生退化单元；必须先按空腔局部
   特征尺度焊接这类近重合点对，再在焊接后的结果上加事后质量门做
   双保险。真实 cube_demo A/B 验证：相邻单元体积比 336.57→33.20
   （改善约 10 倍），偏斜度 0.9955→0.9746、非正交角 85.02°→81.34°
   同步改善，纵横比/负体积计数/总体积守恒均不变差，54 项 mesh_gen
   相关单元测试全部通过。**质量门仍未整体 PASSED**（33.20 仍超过
   5.0 阈值）——剩余的最坏单元不再是本模块缝合阶段制造的（这条根因
   链已堵住），大概率是 BL 挤出本身或 core tetgen 分级过渡的独立
   残余缺陷，需要新一轮单独排查，不在这轮范围内。

   路径 c（给 `remesh_core_cavity` 加目标体积/局部特征尺寸参数）
   未实施——路径 b 已经把这个具体 sliver 类别堵住，路径 c 针对的是
   "形状合格但体积未必达标"这类更泛化的场景（见本节第 2 条），仍然
   是未来可选的独立方向，不是本次目标 case 必需。

即便如此，本模块已经是一个真实的净改进（不是零效果）：它是全项目
第一次让任何修复阶段能够"看见"BL/core 界面并对其采取行动——过去
16+ 轮专项攻关全部作用于 BL 挤出侧参数，从未触及此类根因；本模块
把能被现有空腔重铺工具处理的那部分（形状合格但体积未必达标的过渡
情形）实际修好了。第 19 轮（本节第 5 条）进一步把"最极端的退化
sliver"这个当时未覆盖的残余部分也堵住了大半——cube_demo 上相邻单元
体积比从 336.57 降到 33.20，虽然仍未达到质量门 5.0 的阈值。

这是一个新增的收尾阶段（阶段 D），在阶段 A/B/B' 与最终装配之间、
`prism_cells`/`merged_cells` 都已定型之后运行，独立于阶段 A/B/B'
内部逻辑，不改动它们的任何既有行为。
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_nodes import NodeArray
    from ...validation.quality_validator import MeshQualityValidator


def run_stage_d_interface_repair(
    merged_nodes: np.ndarray,
    prism_cells: np.ndarray,
    merged_cells: np.ndarray,
    cell_groups: np.ndarray,
    nodes_obj: 'NodeArray',
    validator: 'MeshQualityValidator',
    max_rounds: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 'NodeArray', bool, List[str]]:
    """阶段 D：基于完整混合面图的 BL/core 界面相邻体积比定向修复。

    Args:
        merged_nodes: 节点坐标数组（阶段 A/B/B' 之后）。
        prism_cells: 棱柱连接关系（本函数只读，从不修改/重铺）。
        merged_cells: 四面体连接关系（阶段 A/B/B' 之后，本函数唯一可能
            修改的单元数组）。
        cell_groups: 与 merged_cells 平行的边界组标签。
        nodes_obj: merged_nodes 对应的 NodeArray（供面提取/体积计算用，
            调用方已构造好，避免本函数重复构造）。
        validator: 质量校验器实例，复用其 `max_adjacent_volume_ratio`
            阈值，与最终质量报告的判据完全一致（不引入新的独立阈值）。
        max_rounds: 最多重复几轮（修复一批界面单元可能在其新生成的邻接
            关系上暴露新的、原本被旧单元形状掩盖的违规——用有限轮次
            收敛，而非一次性假设修复完全，也避免在退化输入上死循环）。

    Returns:
        (merged_nodes, merged_cells, cell_groups, nodes_obj, changed, actions)
        —— merged_nodes/nodes_obj 原样透传（本阶段从不新增/移动节点属于
        棱柱一侧；`remesh_core_cavity` 可能为四面体一侧插入新的内部点，
        因而 merged_nodes/nodes_obj 在有实际修复发生时会被替换为更新
        后的版本）。

    已知局限：只能修好"形状合格但体积未必达标"的过渡情形；无法消除
    真正退化的近零体积 sliver 四面体（`remesh_core_cavity` 的接受门是
    通用形状指标，不是体积匹配目标）——见模块文档"已知局限"一节。
    """
    from ...schema.grid_cells import TetrahedralCells, PrismCells
    from ...schema.grid_nodes import NodeArray
    from ..extraction.face_extractor import FaceExtractor
    from .mesh_repair_cavity import remesh_core_cavity

    actions: List[str] = []
    changed = False
    n_prism = len(prism_cells)
    threshold = validator.thresholds.get('max_adjacent_volume_ratio', 5.0)

    if n_prism == 0 or len(merged_cells) == 0:
        return merged_nodes, merged_cells, cell_groups, nodes_obj, changed, actions

    for round_idx in range(max_rounds):
        tet_volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells.astype(np.int32))
        prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells.astype(np.int32))
        cell_volumes = np.concatenate([prism_volumes, tet_volumes])

        mixed_faces = FaceExtractor.extract_faces_mixed(
            prism_cells.astype(np.int64), merged_cells.astype(np.int64), nodes_obj
        )
        conn = mixed_faces.connectivity
        interior_mask = conn[:, 1] >= 0
        owner = conn[interior_mask, 0]
        neighbor = conn[interior_mask, 1]

        # 只关心跨越 BL/core 边界的面：一侧是棱柱（全局索引 < n_prism），
        # 另一侧是四面体（全局索引 >= n_prism）。
        owner_is_prism = owner < n_prism
        neighbor_is_prism = neighbor < n_prism
        crosses_interface = owner_is_prism != neighbor_is_prism
        if not np.any(crosses_interface):
            break

        io = owner[crosses_interface]
        jn = neighbor[crosses_interface]
        v_i = cell_volumes[io]
        v_j = cell_volumes[jn]
        ratio = np.maximum(v_i, v_j) / np.maximum(np.minimum(v_i, v_j), 1e-300)
        bad_face = ratio > threshold
        if not np.any(bad_face):
            break

        # 取每条违规界面面上的四面体一侧全局索引，转回局部（四面体数组
        # 内）索引：全局索引减 n_prism。
        tet_side_global = np.where(owner_is_prism[crosses_interface][bad_face],
                                    jn[bad_face], io[bad_face])
        bad_tet_local = np.unique(tet_side_global - n_prism)
        bad_tet_local = bad_tet_local[(bad_tet_local >= 0) & (bad_tet_local < len(merged_cells))]
        if len(bad_tet_local) == 0:
            break

        # 真正的物理外部边界面（入口/出口/隧道/远场——可能携带
        # remesh_core_cavity 不处理的 tetgen 面标记/区域归属，见其自身
        # 文档字符串"作用域"一节）必须用**混合面图**上的边界判定来识别，
        # 见本模块文档"已知局限"第 1 条——纯四面体子图无法区分"真正的
        # 物理外部边界"和"邻居是棱柱（BL/core 接口）"。
        true_boundary_idx = mixed_faces.get_boundary_face_indices()
        true_boundary_owner = mixed_faces.connectivity[true_boundary_idx, 0]
        tets_on_real_boundary = np.unique(
            true_boundary_owner[true_boundary_owner >= n_prism] - n_prism
        )
        bad_tet_local = np.setdiff1d(bad_tet_local, tets_on_real_boundary, assume_unique=False)
        if len(bad_tet_local) == 0:
            actions.append(
                "Stage D: all interface-violating tet(s) also touch a real exterior "
                "boundary face - out of scope (may carry facet markers), skipping"
            )
            break

        bad_cell_mask = np.zeros(len(merged_cells), dtype=bool)
        bad_cell_mask[bad_tet_local] = True

        logger.info(
            f"Stage D (round {round_idx + 1}/{max_rounds}): {len(bad_tet_local)} core tet(s) "
            f"exceed adjacent-volume-ratio threshold {threshold:.2f} against a neighbouring "
            f"BL prism (full mixed-mesh face graph, not visible to stage A/B/B') - "
            f"attempting local cavity retiling on the tet side only"
        )

        tet_faces = FaceExtractor.extract_faces(merged_cells.astype(np.int32), nodes_obj)
        # n_bl_cells=len(merged_cells)：见本模块文档"已知局限"第 1 条，
        # 让 remesh_core_cavity 自身基于纯四面体子图的 touches_physical_
        # boundary 排除对整个数组失效，只依赖上面更准确的混合面图判定。
        new_nodes, new_cells, new_groups, _new_bad_mask, cavity_actions = remesh_core_cavity(
            merged_nodes, merged_cells, cell_groups, len(merged_cells), tet_faces, bad_cell_mask, validator,
        )
        actions.extend(f"Stage D: {a}" for a in cavity_actions)

        if new_cells is merged_cells:
            # remesh_core_cavity 在"没有任何候选空腔被接受"时，按本项目
            # 同类函数的统一约定（见 patch_nonmanifold_cavity_mixed 文档
            # 字符串"返回未修改的（非副本）原始数组"），原样返回*同一个*
            # cells 对象（不拷贝）——身份相等是唯一精确的"这一轮真的什么
            # 都没接受"判据。
            #
            # 之前这里用的是 `len(new_cells) == len(merged_cells) and
            # np.array_equal(new_nodes, merged_nodes)`：对本函数的典型
            # 场景是错的——大多数被接受的空腔重铺是"N 个单元 -> N 个
            # 单元、0 个新增内部点"（cavity 的边界点集不变，只是内部重新
            # 三角化成質量更好的单元），这种情况下单元*总数*和*节点数组*
            # 都不变，但 `new_cells` 里单元引用节点的方式（connectivity）
            # 已经改了——旧判据只看数量和节点坐标，完全看不到 connectivity
            # 的变化，把这类真实生效的改进误判成"无变化"而整体丢弃。已用
            # 真实 cube_demo 数据证实：round 1 的 15 个候选空腔里有多个
            # 从"2 bad -> 0 bad"这样明确改善（remesh_core_cavity 自己的
            # 逐 cavity 日志可见），但外层这个判据仍然判定"no cavity
            # candidate improved quality"并直接 break，连一次真正的赋值
            # `merged_cells = new_cells` 都没有执行过——Stage D 对这个
            # case（本模块文档"已知局限"里说的、cavity 形状可以被局部重铺
            # 修好的过渡情形）实际上从未真正生效，是本函数自身的一个独立
            # bug，不是 remesh_core_cavity 质量门控的设计限制。
            actions.append("Stage D: no cavity candidate improved quality - stopping")
            break

        merged_nodes = new_nodes
        merged_cells = new_cells
        cell_groups = new_groups
        nodes_obj = NodeArray.from_array(merged_nodes)
        changed = True

    return merged_nodes, merged_cells, cell_groups, nodes_obj, changed, actions
