"""generate_hybrid_mesh 在阶段 A/B/B' 结束之后、最终装配之前的
"混合网格（棱柱 + 四面体）收尾修补"阶段：跨类型非流形面拼接、BL 棱柱长细比
局部重铺、collapsed-corner 棱柱降级为四面体。

从 mesh_background.py 拆分出来（原文件超过 400 行上限），逐字搬运
generate_hybrid_mesh 原来紧接着"Final defensive pass: merge coincident
点并修复 non-manifold"之后的那一整段，未改动任何数值逻辑——
三个子步骤共享同一组滚动状态（merged_nodes/prism_cells/merged_cells/
bl_cell_groups/cell_groups/nodes_obj/mesh_changed_by_repair），因此作为
一个整体一起搬运，而不是拆成三个更小的函数。
"""

import numpy as np
from typing import Tuple
from loguru import logger


def _repair_mixed_mesh_post_stage_c(
    merged_nodes: np.ndarray,
    prism_cells: np.ndarray,
    merged_cells: np.ndarray,
    bl_cell_groups: np.ndarray,
    cell_groups: np.ndarray,
    nodes_obj,
    mesh_changed_by_repair: bool,
    min_cell_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, bool]:
    """跨棱柱 + 四面体的非流形面修补、BL 棱柱长细比修补、collapsed-corner
    棱柱降级为四面体——见本模块文档字符串。逐字对应
    mesh_background.generate_hybrid_mesh 原来这一段代码，未改动任何数值
    逻辑（新增的收尾去重步骤除外，见函数末尾）。

    Args:
        min_cell_size: 与 generate_hybrid_mesh 同名参数一致，仅用于
            函数末尾新增的去重后退化体积过滤阈值
            （(min_cell_size**3)*1e-6，与 generate_hybrid_mesh 自身
            对 _build_merged_mesh 直接输出的同名过滤完全一致）。

    Returns:
        (merged_nodes, prism_cells, merged_cells, bl_cell_groups,
        cell_groups, nodes_obj, mesh_changed_by_repair) - 与传入参数一一
        对应，反映本阶段可能施加的任意次原地重建。
    """
    # 延迟导入，避免循环导入（约定见本项目 core/fr_solver_cfl.py 等）。
    from ...schema.grid_nodes import NodeArray
    from ...schema.grid_cells import TetrahedralCells
    from ..extraction.face_extractor import repair_nonmanifold_mixed
    from ..tetgen.mesh_prism_to_tet import orient_tetrahedra
    from ..tetgen.mesh_tetgen_core import _dedupe_coincident_points
    from ..repair.mesh_repair_nonmanifold_mixed import patch_nonmanifold_cavity_mixed, demote_invalid_prisms_to_tets

    # 跨混合网格的非流形检查——先尝试局部重铺
    # （与上方仅四面体的 patch 相同原理：简单的"保留最大、丢弃其余"
    # 修复在额外单元来自两个不同区域在锐角处合法相遇而非真正重复时会
    # 留下孔洞；这就是 cube_demo 上实测 0.189 m^3 缺口的剩余部分——
    # 之前的仅四面体 patch 已修复了它能修的部分后仍缺 0.147 m^3
    # ——被追溯到的位置，因为此检查运行在阶段 A/B/C 之后的完整
    # 棱柱 + 四面体网格上，且此前没有自身的补丁）。
    if len(prism_cells):
        prism_keep_mm, tet_keep_mm = repair_nonmanifold_mixed(nodes_obj, prism_cells, merged_cells.astype(np.int64))
        if not prism_keep_mm.all() or not tet_keep_mm.all():
            merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups = patch_nonmanifold_cavity_mixed(
                merged_nodes, prism_cells, merged_cells.astype(np.int64),
                prism_keep_mm, tet_keep_mm, bl_cell_groups, cell_groups,
            )
            nodes_obj = NodeArray.from_array(merged_nodes)
            prism_keep_mm, tet_keep_mm = repair_nonmanifold_mixed(nodes_obj, prism_cells, merged_cells)

            # 默认 n_buffer_rings=1 尝试失败的簇（tetgen 异常，或其自身
            # 重铺结果不比原始好——参见 patch_nonmanifold_cavity_mixed 自身
            # 的逐簇循环）并不意味着缺陷不可修复，只是该空腔的边界太紧/
            # 形状太奇怪使 tetgen 无法工作。用大得多的缓冲环升级
            # （拉入更多周围好单元，给 tetgen 更好的边界定义），然后才
            # 回退到删除——下方无条件删除会留下真实的孔洞（已直接确认：
            # 在真实 cube_demo 运行中，此精确回退删除约 48-65 个失败簇
            # 的四面体产生了断开的、仅四面体的"幻影"边界壳，包围着
            # 尾流区域中真正空的空间——不仅是缺失体积，而且是外部查看器
            # 如 ANSA 可以走进去的孔洞，因为周围存活单元的新暴露面闭合
            # 成自身自洽的小流形，甚至通过了水密性开放边检查）。
            if not prism_keep_mm.all() or not tet_keep_mm.all():
                merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups = patch_nonmanifold_cavity_mixed(
                    merged_nodes, prism_cells, merged_cells.astype(np.int64),
                    prism_keep_mm, tet_keep_mm, bl_cell_groups, cell_groups,
                    n_buffer_rings=4, max_cavity_cells=15_000,
                )
                nodes_obj = NodeArray.from_array(merged_nodes)
                prism_keep_mm, tet_keep_mm = repair_nonmanifold_mixed(nodes_obj, prism_cells, merged_cells)

            if not prism_keep_mm.all() or not tet_keep_mm.all():
                n_prism_del = int((~prism_keep_mm).sum())
                n_tet_del = int((~tet_keep_mm).sum())
                # 记录发生在哪里，不仅是多少——单纯计数无法判断此运行的
                # 删除是少数分散碎片（无害）还是如真实运行中测量的大块
                # 连续口袋（真实孔洞）。
                del_pts = []
                if n_tet_del:
                    del_pts.append(merged_nodes[np.unique(merged_cells[~tet_keep_mm])])
                if n_prism_del:
                    del_pts.append(merged_nodes[np.unique(prism_cells[~prism_keep_mm])])
                if del_pts:
                    bbox = np.vstack(del_pts)
                    logger.warning(
                        f"Non-manifold mixed-cavity patch: {n_prism_del} prism(s) + "
                        f"{n_tet_del} tet(s) still unpatched after retry with a larger "
                        f"buffer ring - deleting as a last resort (bbox min={bbox.min(axis=0)}, "
                        f"max={bbox.max(axis=0)}); this leaves a real gap at that location, "
                        f"not just missing volume"
                    )
                prism_cells = prism_cells[prism_keep_mm]
                bl_cell_groups = bl_cell_groups[prism_keep_mm]
                merged_cells = merged_cells[tet_keep_mm]
                cell_groups = cell_groups[tet_keep_mm]
            mesh_changed_by_repair = True

    # BL 棱柱长细比修补：上方阶段 A/B/B' 只操作 merged_cells（过渡/核心
    # 四面体）——prism_cells 从未被它们中的任何一个触及，因此严重细长的
    # "collapsed-corner"棱柱（BL 柱的增长恰好在某个底顶点冻结——参见
    # quality_metrics.compute_prism_aspect_ratios 自身文档字符串，
    # "ProjectFiles Part6 Bug 4"，有效的非零体积单元，非生成错误）
    # 目前完全没有修补路径并无条件存活，无论多极端（已直接测量：
    # 最大 BL 长细比钉在该函数自身的 1e6 报告上限，即最小边小于
    # 单元自身最长边的百万分之一）。复用上方非流形修复使用的完全
    # 相同的局部空腔修补机制——那里的种子条件只是"此单元被标记为
    # 删除"，坏长细比保留掩码与非流形掩码同样满足；返回的重铺用
    # 普通四面体替换折叠棱柱，四面体可以表示任意薄的角点而不会
    # 有棱柱固定顶盖/侧面四边形拓扑在冻结柱上强加的极端长细比伪影。
    if len(prism_cells):
        from ...validation.quality_metrics import compute_prism_aspect_ratios
        prism_ar = compute_prism_aspect_ratios(merged_nodes, prism_cells)
        # 刻意比质量报告自身的 bl_max_aspect_ratio=50 阈值宽松得多
        # （普通 BL 单元本就应该是细长的——参见 compute_prism_aspect_ratios
        # 自身文档字符串）——此遍只针对局部重铺实际能改善的真正
        # 折叠/退化离群值，而非每个只是拉伸但没问题的 BL 单元。
        ar_keep = prism_ar <= 500.0
        if not ar_keep.all():
            n_bad_ar = int((~ar_keep).sum())
            logger.warning(
                f"{n_bad_ar} BL prism(s) with extreme aspect ratio "
                f"(collapsed-corner columns, max={float(prism_ar.max()):.3g}) - "
                f"attempting local cavity patch"
            )
            tet_keep_allones = np.ones(len(merged_cells), dtype=bool)
            # 保持默认 n_buffer_rings=1。V2.0 专家组评审在追查 cube_demo
            # 336.57 相邻体积比根因时，怀疑默认 1 圈缓冲把这里的空腔固定
            # 边界卡得太紧，试过把它调大：n_buffer_rings=4 和 =2 都是
            # 真实崩溃（不是"指标没改善"，是硬性 ValueError——"Face
            # connectivity references N cells, expected N+2"），且两次
            # 崩溃的具体数字完全相同，与两个独立、真实、已验证的中间
            # 修复（本函数末尾新增的收尾去重；mesh_repair_cavity_shared.
            # _cavity_boundary_faces 补上了此前遗漏的退化面过滤）叠加
            # 后依然逐位复现同样的崩溃——说明这两个修复虽然本身是正确
            # 的独立改进（保留），但都不是这次崩溃的根因。诊断已经做到
            # 能在具体坐标层面复现异常模式的程度（见本文件 git 历史/
            # 会话记录），但触发这次崩溃的确切机制本轮未能定位。在
            # 找到真正根因之前，调大 n_buffer_rings 对这个调用点是已
            # 验证的不安全操作，不要在未确认修复的情况下重新尝试。
            merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups = patch_nonmanifold_cavity_mixed(
                merged_nodes, prism_cells, merged_cells.astype(np.int64),
                ar_keep, tet_keep_allones, bl_cell_groups, cell_groups,
            )
            # 成功的 patch 会将新内部节点追加到 merged_nodes——
            # nodes_obj（在此块之前构建）必须从可能更大的数组重建，
            # 再让下游索引到它，否则引用这些新节点之一的单元会索引
            # 超过过时数组的末尾。已直接确认，非理论：此精确间隙
            # 在簇足够大实际需要新内部点时崩溃了紧接着的下一行
            # （TetrahedralCells.compute_volumes）。
            nodes_obj = NodeArray.from_array(merged_nodes)
            mesh_changed_by_repair = True

    # 上方基于 tetgen 的长细比补丁无法修复的情况的确定性兜底：
    # 任何仍在自身 6 个顶点中引用同一节点两次的棱柱都是格式错误的
    # CPENTA 记录（不仅是低质量——外部工具会验证并直接拒绝它；已
    # 直接对照真实 ANSA 21.0.1 导入确认，其以"invalid node combination"
    # 拒绝了约 21,000 条此类记录，每条对应上方长细比补丁因 tetgen
    # 无法重铺近零体积空腔而保持不变的折叠角棱柱）。纯算术，不会
    # 像 tetgen 补丁那样失败，因此必须作为最终不变量检查无条件运行，
    # 而非仅在上方长细比补丁报告剩余失败时。
    if len(prism_cells):
        prism_cells, bl_cell_groups, extra_tets, extra_tet_groups = demote_invalid_prisms_to_tets(
            prism_cells, bl_cell_groups
        )
        if len(extra_tets):
            # _split_prisms_to_tets 的固定模板假设格式良好棱柱的底/顶
            # 缠绕方向；折叠角棱柱的近零几何可能翻转该近退化情况，
            # 因此显式重新定向而非信任模板——与第 ~93 行在首次构建
            # merged_cells 后已应用的相同约定。
            extra_tets = orient_tetrahedra(merged_nodes, extra_tets.astype(np.int64))
            merged_cells = np.vstack([merged_cells.astype(np.int64), extra_tets])
            cell_groups = np.concatenate([cell_groups, extra_tet_groups])
            mesh_changed_by_repair = True

    # 收尾去重：子步骤 1/2 都通过 patch_nonmanifold_cavity_mixed 调用
    # 本地 tetgen 生成新的内部（Steiner）点——已实测确认（V2.0 专家组
    # 评审，把子步骤 2 的 n_buffer_rings 从默认 1 试调到 2/4 时复现）
    # 这类局部 tetgen 调用可能产生与已有边界点/彼此在数值上重合但索引
    # 不同的新点。这与 _dedupe_coincident_points 自身文档字符串"两个
    # 调用场景"一节描述的 BL 挤出多顶点同层冻结产生重合点是完全同一类
    # 失效模式（"静默的拓扑撕裂...不会立刻崩溃，上游没有任何地方能
    # 捕获它"），只是触发源不同——这里是 tetgen 的 Steiner 点插入，
    # 不是 BL 挤出。已用独立诊断脚本定位：n_buffer_rings=2 时产生了
    # 2910 组"同一个排序后三角形被 2 个以上单元引用"的拓扑异常，逐一
    # 追踪发现是同一个四面体自己的 4 个面（理论上互不相同）因为其
    # 4 个顶点里有 2 个在数值上重合而坍缩成同一个面——即该四面体本身
    # 已退化，只是退化方式（重合点用不同索引表示）绕开了此前只按
    # "同一索引出现两次"判定退化的检查。不去重的后果是这类退化单元
    # 混进最终网格，在下游 validate_face_data 触发硬性崩溃（"Face
    # connectivity references N cells, expected N+2"）。
    #
    # 本函数从不修改本已进入本函数前就存在的节点坐标（只有子步骤 1/2
    # 会追加新节点），因此这一步在整个函数体只需跑一次，放在末尾对
    # 全部子步骤累积的新增点统一处理，而不必在每个子步骤后单独跑。
    n_nodes_before_dedupe = len(merged_nodes)
    merged_nodes, merged_cells, dedupe_remap = _dedupe_coincident_points(
        merged_nodes, merged_cells.astype(np.int64)
    )
    if len(merged_nodes) != n_nodes_before_dedupe:
        merged_cells = merged_cells.astype(np.int64)
        prism_cells = dedupe_remap[prism_cells]
        nodes_obj = NodeArray.from_array(merged_nodes)
        mesh_changed_by_repair = True

        # 去重可能让原本形状健康、只是顶点恰好数值重合的四面体，在
        # 合并后真正退化为（近似）零体积——与 generate_hybrid_mesh 自身
        # 对 _build_merged_mesh 直接输出、以及对它自己的两次
        # _dedupe_coincident_points 调用之后完全相同的过滤（同一个
        # 阈值公式），保持前后一致。
        post_dedupe_volumes = TetrahedralCells.compute_volumes(
            nodes_obj, merged_cells.astype(np.int32)
        )
        degenerate_threshold = (min_cell_size ** 3) * 1e-6
        valid_mask = post_dedupe_volumes > degenerate_threshold
        n_newly_degenerate = int(np.sum(~valid_mask))
        if n_newly_degenerate > 0:
            logger.warning(
                f"Post-stage-c coincident-point merge: {n_newly_degenerate} "
                f"tet(s) became degenerate after merging Steiner points "
                f"introduced by the non-manifold/aspect-ratio cavity patches "
                f"above - removing them (leaves a small local gap rather than "
                f"a downstream hard crash)"
            )
            merged_cells = merged_cells[valid_mask]
            cell_groups = cell_groups[valid_mask]

    return merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups, nodes_obj, mesh_changed_by_repair
