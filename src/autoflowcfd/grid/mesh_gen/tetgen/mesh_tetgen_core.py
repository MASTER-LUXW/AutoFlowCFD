"""基于 tetgen 的域核心约束四面体化。

用 tetgen 填充由分段线性复形（PLC）封闭的体积——边界层（BL）外表面
加上未修改的外壳面（入口/出口/隧道/对称类边界）——取代旧版的任意填充
边界盒 + 笛卡尔背景网格。PLC 在构造上恰好是输入网格已描述的封闭表面，
因此结果永远不会超出真实域。

本文件只保留 fill_core_volume 本体和相关常量。拆分出去的部分：
    - mesh_tetgen_seam.py：BL/core 交界（seam）过渡缩放、局部厚度限制
    - mesh_tetgen_postprocess.py：填充后清理（重合点合并、超大四面体细分、
      非流形修复、边界分组反推）
    - mesh_tetgen_seeding.py：Steiner 点预算估算、远场背景点播种
下面统一从这三个文件转出，外部代码一律仍从 `mesh_tetgen_core` 导入即可，
不需要关心内部是怎么拆的。
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from .mesh_tetgen_seam import (
    build_seam_taper_scale,
    compute_local_thickness_limit,
)
from .mesh_tetgen_postprocess import (
    _dedupe_coincident_points,
    subdivide_oversized_tetrahedra,
    repair_nonmanifold_cells,
    attribute_cells_from_trifaces,
)
from .mesh_tetgen_seeding import (
    estimate_steinerleft,
    generate_core_background_points,
)
from .mesh_tetgen_input_prep import prepare_plc_input
from .mesh_tetgen_error_translation import translate_tetgen_failure

__all__ = [
    'CORE_TETGEN_MINRATIO',
    'CORE_TETGEN_MINDIHEDRAL',
    'CORE_VOLUME_CAP_FRACTION',
    'CORE_TETGEN_OPT_ITERATIONS',
    'build_seam_taper_scale',
    'compute_local_thickness_limit',
    'subdivide_oversized_tetrahedra',
    'repair_nonmanifold_cells',
    'attribute_cells_from_trifaces',
    'estimate_steinerleft',
    'generate_core_background_points',
    'prepare_plc_input',
    'translate_tetgen_failure',
    'fill_core_volume',
]

# 核心填充 tetgen 质量/grading 参数，所有调用 fill_core_volume 的地方
# 都使用本项目自己的收紧标准，而不是 tetgen 的出厂默认值
# （minratio~2.0, mindihedral~0 实际无约束）。最初只在
# mesh_background_merge.py（核心填充的主调用方）里定义——搬到这里，
# 是因为 fill_core_volume 的每个调用方都已经 import 这个最低层模块，
# 这样 mesh_repair_cavity.py 的 Stage B'（局部型腔重铺）也能用同样的
# 标准来做自己的、规模小得多的 fill_core_volume 调用，而不是静默退回
# tetgen 更宽松的默认值。那个不一致是实际测量到的问题，不是理论上的：
# 在真实案例上，Stage B' 有约 72% 的型腔重铺尝试被判定为"没有改善"，
# 而重铺本身也没有理由真的比原始（已经很差的）型腔更好——因为它用的
# 形状质量界限比当初生成那些型腔邻居时还要宽松。
CORE_TETGEN_MINRATIO = 1.15  # was 1.4; tetgen default ~2.0 (lower = stricter)
CORE_TETGEN_MINDIHEDRAL = 15.0  # unchanged - dihedral wasn't the implicated metric
CORE_VOLUME_CAP_FRACTION = 0.08  # was 0.15, of max_cell_size**3

# tetgen 精化完成后、边/面翻转+光顺的局部优化遍数（对应 tetgen 手册的 -O
# 开关；Python 绑定里的 opt_iterations，默认 3）。这个优化阶段完全在已有
# 点集上做纯拓扑操作（不插入新点），因此和 nobisect（-Y，边界点集固定不
# 变）正交、不会重新触发 nobisect 原本要规避的"tetgen 在复杂 BL 表面上挂起"
# 问题 - 只是让它在同一批点上多尝试几轮翻转/光顺来消除退化单元。
#
# 曾经考虑过直接传 insertaddpoints=True（tetgen 手册 -i 开关）当作"消除
# sliver 的插点开关"，核对 Python 绑定的 tetrahedralize 文档字符串后确认
# 这个理解是错的：-i 的实际含义是"插入调用方另外提供的一批点"，需要额外传
# 一份点列表，不传点列表时不是通用的内部质量插点机制 - 已放弃这个方向，
# 改为只调这里的、含义可以从参数本身（就是"遍数"）直接确认、不依赖对
# tetgen C++ 内部位掩码语义猜测的安全参数。
CORE_TETGEN_OPT_ITERATIONS = 6  # tetgen 默认 3


# 从目标边长到 tetgen maxvolume 上限的粗略换算
# （正四面体 volume/edge^3 ≈ 0.118；Delaunay 精化出的四面体更不规则，
# tetgen 自身的区域上限实际也不严格——所以这里故意取宽一些，不求精确）。
_VOLUME_SHAPE_FACTOR = 0.15


# 注意：本模块早期版本用过嵌套二十面体区域（build_graded_regions/
# _generate_icosphere）从壁面向外做 core 填充的分级网格。已废弃——
# tetgen 的 per-region variable-volume 精化在多个区域共享一个 Steiner
# 预算、互相竞争时，不能可靠地各自收敛到自己的目标（见 fill_core_volume
# 的 `regions` 参数文档）——改为 mesh_background.py 直接构建的单一平面
# 区域。删除而非留着不引用，避免有人在缺少上述背景的情况下把它重新接回去。

def fill_core_volume(
    points: np.ndarray,
    faces: np.ndarray,
    minratio: float = 1.4,
    mindihedral: float = 15.0,
    holes: Optional[List[np.ndarray]] = None,
    regions: Optional[List[Tuple[np.ndarray, int, float]]] = None,
    face_markers: Optional[np.ndarray] = None,
    background_points: Optional[np.ndarray] = None,
    verbose: bool = True,
    force_preserve_boundary: bool = False,
    allow_boundary_bisect: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """对封闭 PLC 围成的体积做约束四面体化。

    Args:
        points: (n_points, 3) float64 PLC 顶点
        faces: (n_faces, 3) int32 PLC 三角面（封闭、水密）
        minratio: 最大半径-边比质量上限（tetgen 约定；
            越小质量越高，1.0 是正四面体）
        mindihedral: 最小二面角质量上限（度）
        allow_boundary_bisect: 显式强制 nobisect=False，即使
            `regions` 未设置（默认的无 `regions` 行为是 nobisect=True，
            见本函数下方 nobisect 注释）。当给定边界只是一个可能不
            完全合法的估计（例如近似自交的代理面）时使用，此时优先
            利用 tetgen 自身的边界恢复鲁棒性处理（Steiner 点插入、
            重合点消解）而非硬失败——调用方必须把返回的边界当作权威
            结果（通过本函数的 `conformal` 检查/回退），而不是假设
            输入 `points` 原样存活为精确前缀。当两者同时设置时优先
            于此参数（设计上后请求的、更具体的意图优先，两者本身
            互相矛盾）。
        force_preserve_boundary: 即使设置了 `regions` 也强制 tetgen
            的 `-Y` 开关（nobisect=True）——普通行为（见本函数下方
            nobisect 注释）允许区域分级仅通过同时允许 tetgen 在给定
            边界本身上插入 Steiner 点来实现，当调用方不需要该边界
            在其他地方被精确保留时这没问题。当此处的边界也被另一
            个独立的四面体化用作固定输入且必须精确匹配时才设置此
            参数——例如 mesh_background_merge._build_merged_mesh 中
            的"填充而非挤出"过渡区域策略，同一个估计的 core 侧表面
            被交给本调用（作为其外边界）和一个独立的过渡间隙填充
            （作为其内边界）：如果任一调用让 tetgen 独立细分该共享
            表面，两个网格就不再一致，拼接会撕裂。分级仍然正常
            工作——基于区域的内部精化（regionattrib/varvolume）只在
            四面体内部插入点，从不在边界上，所以 -Y 不会抑制它
            （已直接验证：近/远四面体体积比不受影响——见
            mesh_tetgen_core.py 自身关于 nobisect+regions 的历史注释，
            已修正，之前因为另一个耦合 bug 看起来像 -Y 破坏了分级）。
        holes: 每个 PLC 内隔离的嵌入实体内部各一个严格内部的点
            （mesh_domain_classify.find_point_inside_closed_shell）。
            没有这些点，tetgen 无法知道一个内部封闭面包围的是实体
            而不是另一个约束——它会在周围填充流体区域以及该实体
            自身的（BL 挤出的）内部，产生与已占据该型腔的 BL 棱柱
            重叠的伪四面体。
        regions: (seed_point, region_id, maxvolume) 元组列表（由调用方
            mesh_background.py 构建），用于按分级层限制最大单元尺寸。
            注意：tetgen 自身的背景网格尺寸控制（`bgmesh`/`metric`，
            tetgen 0.8.4）在此未使用——在当前环境和包版本下无论怎么
            设置都会段错误（在简单 cube 上可复现，对应一个未解决的
            上游问题，该路径无测试覆盖）——改用基于区域的分级：已
            验证稳定，只是不如前者平滑连续。

            传入 `regions` 会关闭 `nobisect`：在粗糙的远场边界面上
            （例如稀疏三角化的隧道/入口/出口壁面）强制执行最大单元
            尺寸需要允许 tetgen 细分该面本身——开启 `nobisect`（默认
            值，无 `regions` 时），任何接触域外边界的区域可以证明其
            体积上限完全无效（已验证：在边界相邻区域上开启和关闭上
            限输出完全相同），因为 `nobisect` 禁止在边界面上或附近
            插入点，这也阻止了对边界相邻单元的体积分裂，不仅仅是
            面本身。
        face_markers: (n_faces,) int32，每个输入面一个标记，与 `regions`
            配套使用——边界归属机制（mesh_background.py）不能再按节点
            索引把细分后的边界面对回其源分组（nobisect=False 意味着
            那些索引在输入中不再原样存在），所以改用 tetgen 自身的
            facet 标记，每个被标记面的子面都继承该标记，通过本函数
            的第 3/4 个返回值返回。
        background_points: (q, 3) 可选的额外自由点，不被 `faces` 任何
            行引用，在 tetgen 运行前拼接到 `points` 末尾（见上方
            `generate_core_background_points` 关于如何为稀疏远场逃逸
            四面体问题构建这些点）。tetgen 接受自由（非 facet）点作为
            普通输入顶点并原样纳入其初始 Delaunay 步骤——在合成的
            cube-PLC+3-内部点测试中直接确认，3 个点都出现在输出节点
            数组的精确输入坐标处，并被 60/102 个输出四面体引用。对于
            不传此参数的现有调用方，保持 None（不变默认值）。
        verbose: 是否在 INFO 级别记录本调用的例行进度（边界点/面计数、
            Steiner 预算、完成信息）（默认值，与先前行为完全一致）。
            False 完全去掉这些行（不仅仅是降级到 DEBUG——本项目的
            默认 loguru sink 显示 DEBUG 及以上，所以单纯降级不会实际
            减少可见输出）——用于在循环中做大量小调用的调用方
            （mesh_repair.remesh_core_cavity，每个修复的型腔一次调用），
            其中每次单独调用自身的进度没有意义，只有调用方自身的
            摘要有意义。警告（非保形边界、自相交）无论此标志如何
            都保持在正常级别——它们指示调用方需要看到的问题，不是
            例行进度。

    Returns:
        (nodes, tets, trifaces, triface_markers)：nodes shape=(n, 3)
        float64（输入点原样保留为前 len(points) 行，即使在 subdivision
        下也是如此——经验验证，tetgen 只追加新点，从不重排/替换现有点），
        tets shape=(m, 4) int64。trifaces/triface_markers 除非
        `face_markers` 被给定否则为 None，否则为四面体化后的边界三角
        （shape=(p, 3) int64，索引到 `nodes`）及其继承的标记
        （shape=(p,) int32）。
    """
    import tetgen

    points, faces, face_markers = prepare_plc_input(points, faces, background_points, face_markers)

    # 稍微放宽质量约束以确保在复杂 BL 表面上收敛。
    effective_minratio = max(1.1, minratio - 0.2)
    effective_mindihedral = max(5.0, mindihedral - 10.0)

    # nobisect=True（无 regions 时）曾在这里无条件使用，用于绕过 tetgen
    # 在本项目自身 BL 外表面上真实挂起的问题——但当时那个表面来自
    # mesh_corner_split.py 的角分裂/倒角构造，自身存在真实缺陷（见
    # mesh_corner_split.py 和 mesh_layer_step.py 的文档字符串——价-3+
    # 角处理在本项目自身的后续工作 P27/P28 中已重建，见 ProjectFiles
    # 3-3 Part8 报告）。`regions`（max_cell_size）未设置时，nobisect=True
    # 仍然无条件强制（该情况行为不变）。设置 `regions` 时，现在允许
    # nobisect 关闭——接触域外边界的 max_cell_size 区域要生效必须如此
    # （见本函数 `regions` 参数文档）——而现在 BL 外表面已是 P27/P28
    # 修复过的几何，不是导致原始挂起的那个。
    #
    # 曾尝试无条件强制为 True（所有调用方、所有区域）来修复一个确认
    # 存在的缺陷（22,830 个 BL/过渡外界面中有 726-882 个在 nobisect=False
    # 下被 tetgen 细分，界面处真实的三角化不匹配）——直接验证 -Y 确实
    # 消除了该细分（之后 0/22,830）且不禁用 max_cell_size 分级（近/远
    # 四面体体积比仍约 15,000x）——但实际报告的缺陷（166 条锐角处
    # X-交点边界边、尾流区域一个断开的约 24,000 面伪边界壳）完全
    # 没有变化，因为被保护的挤出过渡阶段外表面从来就不是 tetgen
    # 不同意的东西。已撤销作为全局默认；同一个 -Y 机制现在以
    # OPT-IN 的 `force_preserve_boundary` 参数形式存在（见其上方文档
    # 字符串），用于确实需要它的特定情况：本调用给出的边界同时被
    # 其他地方独立用作固定输入（mesh_background_merge 的"填充而非
    # 挤出"过渡策略）。
    force_nobisect = ((not bool(regions)) or force_preserve_boundary) and not allow_boundary_bisect
    log = logger.info if verbose else (lambda *_a, **_k: None)

    log(
        f"Tetrahedralizing core volume: {len(points)} boundary points, "
        f"{len(faces)} boundary faces (tetgen, nobisect={force_nobisect}, "
        f"minratio={effective_minratio:.1f}, mindihedral={effective_mindihedral:.1f})..."
    )

    if face_markers is not None:
        tgen = tetgen.TetGen(points, faces, np.ascontiguousarray(face_markers, dtype=np.int32))
    else:
        tgen = tetgen.TetGen(points, faces)
    if holes:
        for hole_pt in holes:
            tgen.add_hole(hole_pt)
        log(f"Marked {len(holes)} tetgen hole seed(s) for isolated embedded solids")
    # 只要有 regions 就注册，与 force_nobisect 无关（见下方
    # regionattrib/varvolume 注释了解为何 -Y 不与之冲突）。
    if regions:
        for seed_pt, region_id, maxvol in regions:
            tgen.add_region(region_id, seed_pt, maxvol)
        log(f"Marked {len(regions)} graded max-cell-size region(s)")

    steinerleft = estimate_steinerleft(points, regions)
    # Optimization: For sharp-corner models, increase the Steiner point budget
    steinerleft = max(steinerleft, 500_000)
    log(f"Steiner-point budget: {steinerleft:,}")

    try:
        nodes, elems, _attr, _markers = tgen.tetrahedralize(
            plc=True, nobisect=force_nobisect, quality=True,
            minratio=effective_minratio, mindihedral=effective_mindihedral,
            # 仅取决于 `regions`，不取决于 force_nobisect——基于区域
            # 的内部精化只在四面体内部插入 Steiner 点（从不在边界上），
            # 所以无论 nobisect 为何为 True（完全没有 `regions`，或
            # force_preserve_boundary 的 opt-in——见该参数文档字符串
            # 了解为何本行早期版本与 `not force_nobisect` AND 在一起
            # 会在 nobisect 因无关原因被强制为 True 时静默破坏分级），
            # 都与 -Y 正交。
            regionattrib=bool(regions),
            varvolume=bool(regions),
            steinerleft=steinerleft,
            # 精化完成后的边/面翻转+光顺优化遍数 - 见 CORE_TETGEN_OPT_
            # ITERATIONS 自己的注释：这是纯拓扑操作（不插入新点），跟
            # nobisect 正交，多跑几轮只会让已经精化出的点集更彻底地消除
            # 退化单元，不会重新触发 nobisect 原本要规避的挂起问题。
            opt_iterations=CORE_TETGEN_OPT_ITERATIONS,
            # 曾不顾此函数自身的 `verbose` 参数硬编码为 True——
            # 意味着每个调用方都无条件收到 tetgen 自身的原始 C 层
            # 控制台输出（内存池大小、各阶段进度、Steiner 点计数……），
            # 即使 mesh_repair_cavity.remesh_core_cavity 的 `verbose=False`
            # 调用（每个型腔簇一次，每次修复可能有数百次）——正是
            # `verbose=False` 本应抑制但无法抑制的控制台刷屏，因为它
            # 只控制了本函数的 log() 调用，从未控制 tetgen 的原生输出。
            verbose=verbose,
        )
    except RuntimeError as e:
        translated = translate_tetgen_failure(e)
        if translated is not None:
            raise translated from e
        raise

    trifaces = None
    triface_markers = None
    if face_markers is not None:
        trifaces = tgen.trifaces.astype(np.int64)
        triface_markers = tgen.triface_markers.astype(np.int32)

    n_input = len(points)
    conformal = nodes.shape[0] >= n_input and np.array_equal(nodes[:n_input], points)

    if not conformal:
        logger.warning(
            "tetgen did not preserve all boundary points verbatim "
            "(likely near-duplicate/degenerate input facets); "
            "falling back to coincident-point stitching"
        )
        nodes, elems, remap = _dedupe_coincident_points(nodes, elems)
        if trifaces is not None:
            # trifaces 是在去重前的索引空间（即上面那行操作前的 `nodes`
            # 数组）中从 tgen.trifaces 读取的。不做 remap 的话，会与
            # 现在重新编号的 nodes/elems 失去同步——mesh_background.
            # attribute_cells_from_trifaces 按排序节点三元组匹配
            # trifaces 和 core_tets，所以过时的索引空间会让该匹配在
            # 此回退和 face_markers（即 max_cell_size）同时激活时静默
            # 漏掉或错误归属边界单元。
            trifaces = remap[trifaces]

    log(f"Core tetrahedralization complete: {len(nodes)} nodes, {len(elems)} tets")

    return nodes.astype(np.float64), elems.astype(np.int64), trifaces, triface_markers
