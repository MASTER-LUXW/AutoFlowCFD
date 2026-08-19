"""边界层前沿挤出的逐层自碰撞防护。

extrude_single_layer 的斜接补偿（mesh_layer_step.py 的 MITER_LIMIT）和
mesh_tetgen_core.compute_local_thickness_limit 的先验锥角预算都能减少挤出
前沿自折叠的频率，但都不能*保证*避免：斜接是按未变形表面一次性计算的
固定逐节点缩放，厚度限制预算是基于同一未变形几何的静态全程估计——
两者都不会查看即将生成的层的实际当前几何。尖锐凹曲线（当厚度超过局部
曲率半径时偏移线收敛）或价≥3的角点（三个或更多面片交汇，而非斜接补偿
建模的简单两面片边）无论参数如何都可能折叠——已直接确认：cube_demo 在
多种层数/增长率组合下仍显示数百个重叠单元。

两个互补机制弥补了这一缺陷，镜像了推进前沿方法（如 Pointwise 的 T-Rex）
的处理方式——两者均独立于 growth_rate、bl_layers 或任何其它挤出参数：

  clamp_budget_for_convergence - 在每层挤出之前，测量当前候选非相邻
      面对之间的距离，并将每个参与节点的剩余寿命位移限制到其当前
      距离的至多一半。这是主要防线，即使对于单对行为良好的会聚前沿
      也至关重要：纯事后检查会"穿孔"——两个干净的、不相交的层末快照
      的扫掠棱柱仍可能在中间完全重叠，如果单步相对于剩余间隙过大
      （已直接确认：本项目的测试套件就在两个平坦对面片在少量几何增长
      层上闭合紧密间隙时捕获了这个问题——见
      test_mesh_front_collision.py）。每层从真实几何重新计算（而非一次性
      未变形表面估计），意味着上限仅随前沿接近而收紧，使它们收敛到
      剩余间隙中点附近，永远不会越过。

  freeze_self_colliding_nodes - 每层挤出后，检查层实际生成的几何
      是否存在真正的自交叉，分两种方式，然后回滚并永久冻结仅涉及的
      节点：
        (a) find_self_colliding_faces - 新层与自身比较
            （同一快照，等同于单对穿孔检查但一次覆盖整个网格）。
        (b) find_cross_state_colliding_faces - 新层与前一层比较——
            捕获一个快速推进的面在同一时间步中扫过一个不同、较慢/
            已冻结邻居在该步开始时仍占据的空间，(a) 无法看到这一点
            （它比较的两个快照各自都不是自交叉的），而
            clamp_budget_for_convergence 自身的一阶/瞬时近似也不能
            保证预测到足够大步长的交叉时刻（已直接确认：在 cube_demo
            上发现了横跨大部分 BL 堆栈深度的真实案例，(a) 完全遗漏）。
      两者都是对预步 clamp 的成对、广相位半径限定搜索未覆盖部分的
      兜底——纵深防御，不是主要机制。

见 mesh_extrusion.py 的 extrude_layers 了解这两个函数如何与已有的
remaining_budget 机制组合（收紧/冻结节点实际上就是降低/归零该节点的
remaining_budget）。

底层的广相位候选对搜索 + 精确三角形相交/跨状态检测（find_self_colliding_
面 / find_cross_state_colliding_faces）拆到了同目录
mesh_front_collision_detect.py，本文件只保留事前裁剪
（clamp_budget_for_convergence）和事后冻结（freeze_self_colliding_nodes）
这两个真正对外使用的入口。
"""

import numpy as np
from loguru import logger

from ...validation.overlap_geometry import triangle_triangle_intersect, triangle_triangle_min_distance
from .mesh_front_collision_detect import (
    _face_geometry,
    _iter_candidate_pairs,
    find_self_colliding_faces,
    find_cross_state_colliding_faces,
)

# clamp_budget_for_convergence 将会聚对的每一侧限制到其当前距离的
# 这个比例，不是精确的 0.5。数学上任何 <= 0.5 的比例已经保证间隙不会
# 变负（见该函数自己的文档字符串）；严格小于 0.5 还额外保证它永远不会
# 完全闭合到精确的 0，当单层的步长耗尽节点的预算时（常见情况——见
# mesh_prism_to_tet.py 中的“Dropped tets”，并且已通过本项目的测试套件
# 直接确认：用精确的 0.5，两个完美对称的对面前沿会收敛到完全重合的重复
# 几何，这是 tetgen 可能拒绝的退化 PLC 输入（与已在真实案例中见过的
# “vertices are coplanar”类错误相同，见 ProjectFiles Part5），即使
# 结果零体积四面体本身会被下游的丢弃逻辑正确处理）。0.45 在 0.5 边界
# 以下留有足够的裕量，而不会实质改变收敛所需的层数。
CONVERGENCE_SAFETY_FRACTION = 0.45

# clamp_budget_for_convergence 必须只限制实际在会聚的候选对——仅仅
# 靠近是不够的。仅按接近度限制已被直接确认（在 cube_demo 上，一个普通
# 立方体）是严重缺陷，不是过度保守的启发式：跨越立方体自身凸边的每对小
# 三角形（特征附近的正常、形状正确的网格细化，不是缺陷）从第一层起就
# 坐在彼此几个面宽度内，完全像真正会聚的对一样——没有方向过滤器，
# clamp_budget_for_convergence 无法区分真实接近和真实会聚，会冻结沿立方体
# 几乎每条边的节点，产生的网格重叠单元数比未修复基线多 131 倍
# （132,260 vs. 1,004），当大量退化/近重复冻结几何到达 tetgen 时。
#
# 简单的 dot(normal_a, normal_b) < 0（“法向彼此指向”）测试也被尝试过，
# 同样是错的，只是针对更窄、更隐蔽的一类情况：一个尖锐的凸楔形（薄翼/
# 刀片，例如机翼后缘）有近相反的面法向，纯粹因为楔角多尖锐（已直接验证：
# 对称 10 度楔形给出 dot=-0.98），但其两个表面在向外挤出时确实是发散的，
# 与任何其他凸特征一样——材料的薄度与偏移表面的移动方向无关。真正重要
# 的不是两个法向的绝对方向，而是沿它们移动是否缩小两个候选面之间的距离：
# 对于质心分离向量 d = centroid_b - centroid_a 和法向差 dn = normal_b -
# normal_a，两个面沿各自法向推进时平方分离度的瞬时变化率正比于
# dot(d, dn)——当且仅当该值为负时才是会聚的（已直接验证所有四种重要
# 情况：对面板 -0.1、凸 90 度边 +0.25、凹 90 度缺口 -0.25、尖锐凸楔形
# +0.35——每个都具有物理预期的符号，包括简单法向测试搞反的楔形情况）。
#
# 阈值本身不是精确的 0.0，是一个小的负数容差——V2.0 专项攻关记录（cube_demo
# BL 质量campaign 第九轮）：cube_demo 上直接插桩追踪发现，car body 自身完全
# 平坦的背面（同一个平面上、彼此不相邻的三角形对，法向理论上应精确相同）
# 触发了本应只针对真正会聚特征的收紧——不是几何问题，是浮点噪声：不同
# 三角形算叉积得到的法向量理论上完全相同，实际因为不同顶点顺序/浮点舍入
# 有约 1e-30 量级的噪声，dot(d,dn) 因此落在约 1e-32~1e-30 量级的一个随机
# 符号的极小值上——严格 `< 0.0` 会把恰好取到负号的这批也当成真会聚。真实
# 会聚案例的量级（已在本模块自己的文档里验证过）是 0.1~0.35，比这个噪声
# floor 大 30 个数量级以上，所以容差可以设置得远比任何真实信号更严格、
# 同时仍然比浮点噪声宽松很多个数量级——不是在"保守"和"精确"之间做权衡。
CONVERGING_CLOSING_RATE_THRESHOLD = -1e-9


def clamp_budget_for_convergence(
    nodes: np.ndarray,
    faces: np.ndarray,
    remaining_budget: np.ndarray,
    search_multiplier: float = 3.0,
    chunk_size: int = 2000,
) -> None:
    """收紧 `remaining_budget`，使得没有节点在所有剩余层合计中前进超过
    其当前到最近实际会聚的非相邻面距离的 CONVERGENCE_SAFETY_FRACTION
    （0.45）（相对闭合速率低于 CONVERGING_CLOSING_RATE_THRESHOLD——见
    该常量自己的注释了解为什么这个过滤是必需的，不是可选的）。在挤出
    一层之前调用，在该层的起始 (`current_nodes`) 几何上——见模块文档
    字符串了解为什么纯事后检查本身不够。

    每个找到的候选对将双方的预算收紧到至多其当前距离的约 45%——
    对称的，所以如果双方都花完全部配额直线向对方移动，它们会在中点
    之前停下，保持严格正的间隙（永远不会精确为 0，永远不会穿越——见
    CONVERGENCE_SAFETY_FRACTION 自己的注释了解为什么严格停在中点之前
    很重要）。一个触及多个会聚对的节点（例如价≥3 的凹角，三个前沿
    交汇）最终受所有对的最小值限定，因为每个成对限定通过单次向量化
    scatter-min（np.minimum.at）独立执行——无论节点还参与多少其他对，
    修正都是保证的。每层从网格的实际当前几何重新计算（非一次性估计），
    所以当两个前沿接近时这只会进一步收紧，使它们收敛到——永远不会越过——
    彼此。

    在 `current_nodes` 上已经（精确）相交的候选对在实践中不应发生——
    `current_nodes` 总是前一层的已接受、通过归纳无碰撞的结果——但通过
    直接钳制到零来防御性处理，而不是调用 triangle_triangle_min_distance，
    后者（按其自己的文档字符串）仅对非相交对有意义。

    Args:
        nodes: (n_nodes, 3) 当前（步前）节点位置
        faces: (n_faces, 3) 三角形连接（int）
        remaining_budget: (n_nodes,) float，米，原地修改，
            只会降低，不会升高
        search_multiplier: 广相位 KD-tree 查询半径，作为每个面
            自身 sqrt(area) 的倍数——比 find_self_colliding_faces
            的默认值大，因为这个还必须捕获仅靠近但尚未接触的对
        chunk_size: 每个 KD-tree 批次处理的面数
    """
    n_faces = len(faces)
    if n_faces == 0:
        return

    tri, centroids, face_size, normal = _face_geometry(nodes, faces)
    budget_before = remaining_budget.copy()

    for row_idx, col_idx in _iter_candidate_pairs(
        faces, centroids, face_size, search_multiplier, chunk_size
    ):
        # 仅实际会聚的对——沿它们的自身法向移动会缩小它们之间的距离——
        # 才会被限制；见 CONVERGING_CLOSING_RATE_THRESHOLD 自己的注释
        # 了解为什么必须是相对闭合速率，而不是仅看法向是否彼此指向。
        # 在（更昂贵的）精确几何测试之前应用，既跳过被排除的对的工作，
        # 也因为这是使本函数安全可用的关键（见模块文档字符串/本函数自己的名称）。
        d_vec = centroids[col_idx] - centroids[row_idx]
        n_diff = normal[col_idx] - normal[row_idx]
        closing_rate = np.einsum('ij,ij->i', d_vec, n_diff)
        converging = closing_rate < CONVERGING_CLOSING_RATE_THRESHOLD
        if not np.any(converging):
            continue
        row_idx, col_idx = row_idx[converging], col_idx[converging]

        a_nodes, b_nodes = tri[row_idx], tri[col_idx]
        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )

        safe_budget = np.zeros(len(row_idx), dtype=np.float64)
        safe = ~intersects
        if np.any(safe):
            dists = triangle_triangle_min_distance(
                a_nodes[safe, 0], a_nodes[safe, 1], a_nodes[safe, 2],
                b_nodes[safe, 0], b_nodes[safe, 1], b_nodes[safe, 2],
            )
            safe_budget[safe] = CONVERGENCE_SAFETY_FRACTION * dists
        # 相交对保持 safe_budget == 0：直接钳制到零。

        pair_nodes = np.concatenate([faces[row_idx], faces[col_idx]], axis=1)  # (M, 6)
        pair_budget = np.repeat(safe_budget, 6).reshape(-1, 6)
        np.minimum.at(remaining_budget, pair_nodes.ravel(), pair_budget.ravel())

    # 之前是完全静默的——尽管是模块文档字符串描述的主要（主动）防线，
    # 却在真实运行的控制台输出中完全不可见：一个完全通过这个机制收敛的网格
    # （remaining_budget 在任何实际碰撞发生前就被收紧到约 0）从下游的
    # freeze_self_colliding_nodes 产生零个自交叉警告，看起来好像没有任何
    # 东西约束了那里的增长——已在 cube_demo 上直接确认，这是约 25,000 个
    # 被丢弃（预算完全耗尽）的 BL 棱柱的唯一原因，而没有其他机制显示任何
    # 参与的证据。
    tightened = remaining_budget < budget_before
    n_tightened = int(np.sum(tightened))
    if n_tightened:
        n_exhausted = int(np.sum(remaining_budget[tightened] <= 0.0))
        logger.info(
            f"Convergence budget clamp: {n_tightened} node(s) tightened this "
            f"layer ({n_exhausted} fully exhausted, remaining_budget=0 - that "
            f"column's front is done growing), min remaining "
            f"{float(np.min(remaining_budget[tightened])):.4e} m"
        )


def freeze_self_colliding_nodes(
    new_nodes: np.ndarray,
    current_nodes: np.ndarray,
    faces: np.ndarray,
    remaining_budget: np.ndarray,
    max_iterations: int = 5,
) -> np.ndarray:
    """回滚并永久冻结自交叉面上的每个节点，原地修改 `new_nodes`
    和 `remaining_budget`。

    `current_nodes` 是前一层的已接受几何——通过归纳它本身是无碰撞的，
    因为相同的检查在它作为“新”层时也运行过。将有问题的节点回滚到其
    `current_nodes` 位置因此只能将该节点的面返回到已知无碰撞的状态；
    不会使任何东西变得更糟。冻结就是 `remaining_budget = 0`：
    extrude_single_layer 已经将每个节点的每层位移钳制到其剩余预算
    （见 mesh_extrusion.py），所以冻结的节点在运行剩余时间内完全停止
    移动——T-Rex 的“局部终止，其他地方继续”语义，其他节点不受影响。

    迭代（受 `max_iterations` 限定），因为消除一对的碰撞偶尔会留下一个
    仍在移动的邻居与否则不会出问题的东西交叉——只有在其邻居回滚后才成为
    问题（级联）。每次迭代要么冻结至少一个额外节点，要么找不到并停止，
    所以这总是自行终止的；上限只限定最坏情况下的每层成本——超出上限的
    未解决级联仍会在下一层自己的调用中重新检查，并且无论如何会被最终的
    网格宽重叠验证（mesh_overlap_check.py，在四面体化后运行）捕获。

    Args:
        new_nodes: 本层的试探节点位置——为新冻结的节点原地修改
        current_nodes: 前一层（已接受）的节点位置
        faces: (n_faces, 3) 三角形连接，两层挤出用的同一个
        remaining_budget: 每节点剩余挤出预算（米），原地修改，
            新冻结的节点设为 0
        max_iterations: 级联解决上限（见上方）

    Returns:
        int64 数组，本次调用中冻结的节点索引（无则为空）
    """
    frozen = np.zeros(len(new_nodes), dtype=bool)
    total_frozen_count = 0

    for i in range(max_iterations):
        colliding_faces = find_self_colliding_faces(new_nodes, faces)
        # 还检查快速推进的面在同一时间步中扫过不同、较慢/已冻结
        # 邻居的领地——见 find_cross_state_colliding_faces 自己的文档
        # 字符串了解为什么上面的同快照检查本身看不到这个。标记对的
        # 两侧都被冻结（不仅是“攻击者”），匹配同层检查的双方全部节点
        # 策略——更简单的推理，永远不会更安全。
        cross_faces = find_cross_state_colliding_faces(new_nodes, current_nodes, faces)
        colliding_faces = np.union1d(colliding_faces, cross_faces)
        if len(colliding_faces) == 0:
            break

        guilty = np.unique(faces[colliding_faces].ravel())
        guilty = guilty[remaining_budget[guilty] > 0]
        if len(guilty) == 0:
            break

        new_nodes[guilty] = current_nodes[guilty]
        remaining_budget[guilty] = 0.0
        frozen[guilty] = True
        total_frozen_count += len(guilty)
        
        if i == 0:
            logger.warning(f"Detected {len(colliding_faces)} self-intersecting faces in BL layer. "
                           f"Freezing {len(guilty)} nodes to prevent invalid geometry.")
        elif len(guilty) > 0:
            logger.debug(f"Cascade resolution: freezing {len(guilty)} additional nodes.")

    if total_frozen_count > 0:
        logger.info(f"Total nodes frozen in this BL layer: {total_frozen_count}")

    return np.flatnonzero(frozen)
