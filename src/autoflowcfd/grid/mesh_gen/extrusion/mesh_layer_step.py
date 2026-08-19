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
# convergence 的运行累积上限，或 freeze_self_colliding_nodes 的反应式
# 冻结——见 extrude_layers）已经小于本层名义厚度请求的量时，本层及之后
# 所有层此节点位移精确为 0，硬停止在当前（已完成的）层位置——不再有
# 渐近摊薄的中间阶段（V2.0 专项攻关记录：cube_demo BL 质量campaign 第五轮，
# 曾经在这里用 BUDGET_TAPER_FRACTION/MIN_TAPER_FRACTION_OF_NOMINAL 做
# 渐近摊薄，换来的是一长串"消失地薄但非零"的层，是核心区 tetgen 填充在
# BL/核心交界处生出极端 sliver 四面体的直接原因；改回硬停止后 cube_demo
# 上相邻单元体积比从 1431.89 降到 421.11。参照 ANSA 官方文档描述的
# Collapse 机制："从外层往内逐层减少层数，上层节点收缩回下层节点，棱柱
# 退化为面"——即精确回退，不是渐进逼近）。产生的"完全坍缩"棱柱由
# mesh_prism_to_tet.convert_layers_to_prisms 自身已有的
# DEGENERATE_VOLUME_FRACTION 检查干净丢弃，该检查的文档字符串已经论证过
# 这是安全的：丢弃的棱柱唯一非退化面只与自身内部四面体共享，从不留下
# 真实几何空洞。


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
            原地修改——名义位移超过剩余预算的节点本层位移硬停止为 0
            （见下方实现自己的注释了解为什么不是渐近摊薄），减去它
            实际花费的相同量（见 extrude_layers 的 thickness_limit）
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
        # 硬停止，不摊薄——见本文件顶部这段注释自己的完整说明。
        tight = remaining_budget < node_thickness
        node_thickness = np.where(tight, 0.0, node_thickness)
        # 原地修改（`-=`，不是重新赋值给局部变量名）——本函数自己的
        # Args 文档承诺 remaining_budget 是调用方数组的原地更新，重新
        # 绑定局部名字到 np.where(...) 的新数组不会传播回调用方持有的
        # 那个数组对象，会让冻结状态在下一层调用时静默丢失（已在这版
        # 硬停止实现里直接踩到过：debug 输出显示 remaining_budget 在
        # 调用前后完全不变，corner 节点因此从不真正冻结，一路长到
        # 第 8 层）。
        remaining_budget -= node_thickness

    displacement = node_thickness[:, np.newaxis] * avg_normals

    # 挤出所有节点
    logger.info(f"Extruding layer with thickness={thickness:.6f}...")
    new_nodes[mask] += displacement[mask]

    return new_nodes
