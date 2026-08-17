"""单层挤出步骤：法向平均与斜接（miter-join）补偿。

从 mesh_extrusion.py 拆出（该文件保留多层编排循环 extrude_layers），纯粹
为了让两个文件都控制在 450 行以内；extrude_layers 是本模块唯一调用方。
"""

import numpy as np
from typing import Optional
from loguru import logger

# MITER_LIMIT 限制偏移向量的大小，方式与 SVG/矢量图形描边的"斜接连接"
# 相同（那里的默认 stroke-miterlimit 是 4）——超过限制时顶点会回退到
# 更短但仍正确方向的偏移，而不是让它爆炸，这在近反射/针状特征处会发生
# （cos(半角) -> 0，对于单个 2 面片边；extrude_single_layer 中的最小二乘
# 矩阵在一般 N 面片情况下接近奇异）。
#
# 之前是 3.0（"比 SVG 的 4 稍微更保守"），直到直接在真实特征附近测量
# 这个补偿在 uncapped 时导致的实际失败模式：轴对齐盒角的自然（未裁剪）
# 补偿因子约为 1.73x，舒适地在旧的 3.0 上限以下——所以 3.0 在那里实际上
# 从不约束任何东西。那个 1.73x 的每层因子，跨每层的累积高度复合，产生的
# 四面体体积比同一角的典型近壁单元大达约 50 倍（仍然形状良好——偏斜度
# 好——只是大得多，这是 mesh_repair.py 中的形状质量修复阶段都不检查更不
# 修复的网格尺寸均匀性缺陷）。将上限降低到 1.2（直接在真实立方体上测试，
# 最坏现实情况：真正的 3 面轴对齐角）将最坏情况体积削减约 41%，偏斜度
# 没有退化（近壁最大偏斜度实际上略微改善，0.899 -> 0.880），也没有新的
# 退化/负体积单元。没有降低到 1.0（完全没有补偿，在该特定案例上测量更好）
# 是为了保持对这个机制原始目的的裕量——防止在比简单 90 度盒角更尖锐、
# 更锐利的特征处不成比例地薄的层，这个特定测试案例没有测试到。
MITER_LIMIT = 1.2

# 当节点的 remaining_budget（mesh_front_collision.clamp_budget_for_
# convergence 的运行累积上限——见 extrude_layers）已经小于本层名义厚度
# 请求的量时，只消耗剩余量的这个比例，而不是一次性消耗全部。简单的
# min(nominal, remaining_budget)（之前的行为）在下一次层的名义请求超过
# 剩余量时精确命中 0——通常是立即，因为层厚度几何增长（每层约 1.2-1.5
# 倍上一层），而 remaining_budget 按定义已经很小（否则不会触发这个路径）
# ——为该节点后续的每层产生完全重合（零体积）棱柱。已在 cube_demo 上直接
# 确认：这是约 25,000 个被丢弃 BL 棱柱的全部原因，而其他碰撞机制
# （freeze_self_colliding_nodes、compute_local_thickness_limit）都没有
# 参与的任何证据。几何衰减（消耗剩余量的固定比例，不是固定量）意味着
# 受约束节点的高度在多层上渐近地接近其真实极限——它只会产生消失地薄，
# 永远不会精确为零体积的层，所以下游不需要丢弃任何东西。0.5（每次绑定
# 约束时减半剩余间隙）反映了本项目其他地方自己的 CONVERGENCE_SAFETY_
# FRACTION 风格推理（mesh_front_collision.py），而不是一字不差地复用那个
# 特定常量，因为它控制的是不同的量（那里是空间安全裕量，这里是时间衰减率）。
BUDGET_TAPER_FRACTION = 0.5

# 纯几何衰减（BUDGET_TAPER_FRACTION 单独）永远不会精确命中 0，但
# "永不精确为 0"不等于"可用"——已在 cube_demo 上直接确认：用这种方式
# 消除所有被丢弃的棱柱产生了一长串数值上消失的幸存者（最小 3.3e-8 m，
# 即 33 纳米，约 6% 的 BL 棱柱低于 0.1mm），真实求解器没有现实用途
# （相对于该单元几毫米的占地，长宽比达数百万），而且可以说比干净丢弃更糟——
# 对任何下游质量门控或求解器都是垃圾输入，不仅是日志行中更小的数字。
# 低于当前层自身名义厚度的这个比例时，tapered 花费被视为完全耗尽（捕捉
# 到 0，与旧的硬停止行为相同），而不是继续缩小——是地板，不是猜测：它
# 为仍在邻居自身单元尺寸合理范围内的每个节点保持平滑过渡，只有当继续
# 会产生下游阶段无法做任何有用事情的单元时才恢复干净停止。
#
# 0.05（第一个尝试的值）仍然让幸存者小到 0.073mm 通过——复合效应：
# 节点可以跨多个连续层过渡，直到单层的地板检查捕获它（每层自身的地板
# 相对于该层自身更大的名义厚度，但幸存者继续按 BUDGET_TAPER_FRACTION
# 缩小），留下 0.1mm 以下的小但真实的尾部（cube_demo 上约 0.47% 的 BL
# 棱柱）。直接跨多个值测量：0.1 是完全清除 0.1mm 尾部的最小值（最小
# 幸存者 0.130mm，0 个单元低于 0.1mm），以适度额外干净丢弃为代价
# （20,828 vs 0.05 的 13,807，两者仍远低于过渡前的 24,691 基线）。
# 0.3 清除更宽的裕量（最小幸存者 0.260mm）但将丢弃单元推到 32,213——
# 比完全不过渡更糟——所以更高并不简单地更安全；0.1 被保留为测量更好的
# 权衡，不是最保守可用的。
MIN_TAPER_FRACTION_OF_NOMINAL = 0.1


def extrude_single_layer(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    thickness: float,
    taper_scale: 'Optional[np.ndarray]' = None,
    remaining_budget: 'Optional[np.ndarray]' = None,
    miter_decay: float = 1.0,
) -> np.ndarray:
    """挤出一层节点。

    对每个节点，计算最小二乘偏移向量 `d`（对单位厚度）同时满足每个相邻面
    自身的目标距离：`d = argmin sum_i (n_i . d - 1)^2`，对该节点的相邻面
    法向 `n_i`，通过 `(sum_i n_i n_i^T) d = sum_i n_i`（3x3 线性系统，通过
    np.linalg.pinv 批量求解，所以秩亏情况——平坦节点、简单 2 面片边——
    用相同公式处理，不是特殊情况）。

    这是"斜接连接"向量图形描边在尖锐路径角用的直接推广（1/cos(半角)，
    仍沿简单平均法向平分线）——已证明对任意角的 2 面片边约简到精确那个
    公式，对平坦/单面片节点约简到精确简单法向（见本项目自己的验证脚本，
    不在此重现——两者都是最小二乘解的闭合恒等式，不仅经验上相似）。这个
    求解解决完整 3x3 系统而不是旧情况，无论是旧的简单平均然后缩放方法还是
    mesh_corner_split 自身的顶点分割都没有完全解决的情况：价≥3 的角，
    3+ 面片交汇于一点。混合平均方向在那里通常甚至不在任何单个面的正确
    偏移线上，更不用说所有面了——"没有单个混合方向能同时正确偏移三个
    独立平面"（mesh_corner_split 自己的模块文档字符串）。最小二乘求解不是
    方向混合——对 3D 中 3 个独立法向，它是同时满足所有 3 个目标距离的
    精确点（3 个方程，3 个未知数），对超过 3 个则是最接近可达的妥协。
    已直接确认：对 3 个互相垂直的法向（轴对齐盒角）这精确求解到 (1,1,1)，
    大小 sqrt(3)——与本模块自己的 MITER_LIMIT 注释已记录为从第一原理的
    正确值相同的"自然（未裁剪）补偿因子约 1.73x"，现在通过通用公式达到
    而不是仅作为特殊情况参考量已知。

    只限制求解向量的大小（到 MITER_LIMIT，与旧标量上限相同的理由——
    近反射/针状特性可以使系统接近奇异）——从不改变方向，总是精确是最小二
    乘求解产生的方向。

    这个补偿和可选的 remaining_budget 上限都只看未变形表面/节点自身的
    局部邻域——两者都不能看到它们产生的前沿是否真的在某处折叠到网格的
    其他地方。mesh_extrusion.extrude_layers 在调用这个函数后检查真实结果
    并反应式地冻结（remaining_budget = 0）任何确实这样做的节点，通过
    mesh_front_collision.freeze_self_colliding_nodes——见那个模块的文档
    字符串了解为什么静态、未变形表面估计本身不能保证无重叠几何。

    Args:
        nodes: 当前层节点，形状=(n_nodes, 3)
        faces: 面连接，形状=(n_faces, 3)
        normals: 面法向，形状=(n_faces, 3)
        thickness: 挤出距离（米）
        taper_scale: 可选 float 数组，[0, 1]，形状=(n_nodes,)，
            缩放每个节点的位移（见 extrude_layers）
        remaining_budget: 可选 float 数组，米，形状=(n_nodes,)，
            原地修改——名义位移超过剩余预算的节点只花费
            BUDGET_TAPER_FRACTION 的剩余量（渐近过渡，永远不会
            一步精确耗尽——见该常量自己的注释），减去它实际花费的
            相同量（见 extrude_layers 的 thickness_limit）
        miter_decay: 将每节点偏移向量混合向简单（无权重、大小 1）
            平均法向——1.0（默认）保持完整计算偏移向量不变，0.0
            完全禁用本层的补偿。extrude_layers 在过渡阶段降低这个
            （见其自己的文档字符串了解为什么）：偏移向量是每节点固定的
            （只依赖未变形表面的局部特性角，从不对变形几何重新计算），
            所以尖角节点的累积高度跨多层最终与其平坦区域邻居的累积高度
            成常量比例。那个比例在 BL 尺度上无害（小绝对间隙），但一旦
            过渡阶段挤出到远场目标大小，相同比例应用到更大的绝对高度
            会在网格相邻节点之间打开大的绝对间隙——已直接在真实案例上
            确认：尖角列和其平坦区域邻居之间 90mm+ 的绝对高度不匹配，
            精确集中在真正退化的过渡单元和最终网格边界面无法匹配任何
            真实边界组的大部分。完整补偿在整个 BL 阶段本身仍然应用，
            那里最重要（近壁单元形状质量）并且累积高度无论如何保持小。

    Returns:
        挤出后的新节点位置，形状=(n_nodes, 3)
    """
    n_nodes = len(nodes)
    new_nodes = nodes.copy()

    # 使用向量化操作构建节点到面映射
    node_normal_count = np.zeros(n_nodes, dtype=np.int64)
    flat_nodes = faces.ravel()
    repeated_normals = np.repeat(normals, 3, axis=0)
    np.add.at(node_normal_count, flat_nodes, 1)
    mask = node_normal_count > 0

    # 简单（无权重）平均法向，大小 1——miter_decay=0 的"无补偿"回退，
    # 也是下面最小二乘求解对平坦节点验证的参考方向。
    node_normal_sum = np.zeros((n_nodes, 3))
    np.add.at(node_normal_sum, flat_nodes, repeated_normals)
    plain_avg_normal = np.zeros_like(node_normal_sum)
    plain_avg_normal[mask] = node_normal_sum[mask] / node_normal_count[mask, np.newaxis]
    plain_norms = np.maximum(np.linalg.norm(plain_avg_normal, axis=1, keepdims=True), 1e-10)
    plain_avg_normal = plain_avg_normal / plain_norms

    # 单位厚度的最小二乘偏移向量：每节点求解 (sum_i n_i n_i^T) d = sum_i n_i。
    # 通过 np.linalg.pinv 批量处理（用 SVD 以与一般情况相同的方式处理
    # 秩亏的平坦/2 面片情况，不需要为它们单独分支）。
    outer = np.einsum('ki,kj->kij', repeated_normals, repeated_normals)
    A = np.zeros((n_nodes, 3, 3))
    np.add.at(A, flat_nodes, outer)
    A_pinv = np.linalg.pinv(A)
    offset_vec = np.einsum('kij,kj->ki', A_pinv, node_normal_sum)

    # 只限制大小（从不改变方向）——见 MITER_LIMIT 自己的注释。
    offset_mag = np.linalg.norm(offset_vec, axis=1, keepdims=True)
    safe_mag = np.maximum(offset_mag, 1e-10)
    capped_mag = np.minimum(offset_mag, MITER_LIMIT)
    offset_vec = offset_vec / safe_mag * capped_mag

    if miter_decay != 1.0:
        offset_vec = plain_avg_normal + (offset_vec - plain_avg_normal) * miter_decay

    # 下面的 avg_normals/miter_scale 保持其原始名称和含义，作为单位方向
    # 和单独的标量大小，所以函数的其余部分（taper_scale/remaining_budget
    # 应用）不变。
    miter_scale = np.maximum(np.linalg.norm(offset_vec, axis=1), 1e-10)
    avg_normals = offset_vec / miter_scale[:, np.newaxis]

    if taper_scale is not None:
        node_thickness = thickness * taper_scale * miter_scale
    else:
        node_thickness = thickness * miter_scale

    if remaining_budget is not None:
        # 见 BUDGET_TAPER_FRACTION 自己的注释：剩余预算已经比本层名义
        # 请求更紧的节点只花费剩余量的一部分，渐近过渡而不是在一步中
        # 精确耗尽到 0。预算充裕的节点（常见情况）完全不受影响——
        # node_thickness 在那里已经等于其自身的名义请求。
        tight = remaining_budget < node_thickness
        tapered = remaining_budget * BUDGET_TAPER_FRACTION
        # 见 MIN_TAPER_FRACTION_OF_NOMINAL 自己的注释：干净停止而不是
        # 继续过渡到数值上无意义的碎片，当即使过渡后的花费相对于本层
        # 自身尺度也可以忽略时。
        floor = thickness * MIN_TAPER_FRACTION_OF_NOMINAL
        exhausted = tight & (tapered < floor)
        node_thickness = np.where(tight & ~exhausted, tapered, node_thickness)
        node_thickness = np.where(exhausted, 0.0, node_thickness)
        remaining_budget -= node_thickness
        remaining_budget = np.where(exhausted, 0.0, remaining_budget)

    displacement = node_thickness[:, np.newaxis] * avg_normals

    # 挤出所有节点
    logger.info(f"Extruding layer with thickness={thickness:.6f}...")
    new_nodes[mask] += displacement[mask]

    return new_nodes
