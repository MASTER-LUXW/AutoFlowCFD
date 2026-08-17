"""tetgen 核心域填充：Steiner 点预算估算与远场背景点播种。

从 mesh_tetgen_core.py 拆分出来，是 fill_core_volume 调用前的两个准备
步骤：估算给 tetgen 的 Steiner 点预算（steinerleft），以及在空旷远场
预先播撒一批背景点，避免 tetgen 初始 Delaunay 阶段连出跨越整个域的巨大
四面体。
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


def estimate_steinerleft(
    points: np.ndarray,
    regions: Optional[List[Tuple[np.ndarray, int, float]]],
) -> int:
    """估算 Steiner 点预算（tetgen 的 `steinerleft`），足够宽松以
    满足请求的区域，按实际问题大小缩放而不是固定常量。
   
    tetgen 默认的 steinerleft=100000 是全局上限，整个网格共享——
    当区域的 maxvolume 目标远低于 PLC 的自然（无约束）四面体大小
    时，它可能在目标到处达到之前就耗尽，静默地在最后精化的角落
   留下一长串超大单元（直接测量：5.5x3x3 m 域限制到 0.05 m，
    固定 300,000 预算留下 6-10% 的单元超过目标 1.5 倍，最坏情况
    约 5-6 倍）。
   
    全域分级区域（只要设置了 max_cell_size 就存在——见
    mesh_background._build_merged_mesh）总是此处传入的所有区域中
    maxvol 最大的，所以 bbox_volume / coarsest_maxvol 估计它单独
    需要多少单元来填充核心——那个数量使得本函数在恰好只有一个
    区域时行为不变（与原始单区域公式相同，而且——随着 Stage B
    的 core-边局部修复区域被移除，见 mesh_repair.py 的模块文档
    字符串——`regions` 现在实际最多包含 1 个条目）。
   
    下面的 `n_extra_regions` 处理在当前用法中是死的，但保留而不
    是特殊处理掉，以防未来调用方合法地传入多个区域：用多个区域
    中最小的 maxvol 来除整个 bbox（本函数的早期版本，用
    `min(maxvol for ...)`）会严重高估，只要其中一个是小的局部
    补丁而不是域范围的目标——在真实案例上直接观察到，Stage B
    的现在已移除的 core 区域，估计约 178 亿个目标大小的四面体，
    而该域的单区域核心填充收敛在约 120 万个四面体。注意这个估计
    只是建议性的，不是硬约束：tetgen 被确认收敛到完全相同的实际
    四面体计数，无论 steinerleft 是（有 bug 的）膨胀值还是本函数
    修正后的值——实际的 5 倍核心填充爆炸（1.2M -> 6.1M 四面体）
    结果是一个独立的、仍未解决的 tetgen 多区域精化行为（见
    mesh_repair.py），不是这个预算数量实际导致的。
   
    Args:
        points: PLC 边界点，shape=(n, 3)——只用于其包围盒体积
        regions: (seed_point, region_id, maxvol) 元组，或 None/空
            表示无约束（nobisect=True）填充
   
    Returns:
        steinerleft，限制在 [300_000, 20_000_000]——或当没有活跃区域
        时为 100_000（tetgen 自身默认值）。
    """
    if not regions:
        return 100_000

    bbox_volume = float(np.prod(np.max(points, axis=0) - np.min(points, axis=0)))
    coarsest_maxvol = max(maxvol for _, _, maxvol in regions)
    estimated_tets = bbox_volume / max(coarsest_maxvol, 1e-30)

    n_extra_regions = len(regions) - 1
    extra_tets = n_extra_regions * 200_000

    logger.info(
        f"Steiner-point budget estimate: ~{estimated_tets:,.0f} domain-wide target-sized tets"
        + (f" + {n_extra_regions} local repair region(s) x 200,000" if n_extra_regions else "")
    )
    return int(np.clip((estimated_tets + extra_tets) * 3.0, 300_000, 20_000_000))


def generate_core_background_points(
    plc_points: np.ndarray,
    plc_faces: np.ndarray,
    target_edge_length: float,
    grid_spacing_factor: float = 2.5,
    clearance_factor: float = 3.0,
) -> np.ndarray:
    """在稀疏远场中预撒一批粗糙的背景点网格，作为
    `background_points` 传给 `fill_core_volume`，这样 tetgen 的初始
    Delaunay 四面体化已经有点分散在空旷的远场空间中，而不仅依赖
    PLC 自身的边界点。

    本函数针对的根本原因：只有边界点作为输入时，tetgen 的首遍
    Delaunay 步骤可能将遥远的边界点（例如入口到出口，跨越真正
    空旷的空间）连接成一个巨大的初始四面体；其自身的第二遍
    质量/体积精化应该将它们分裂回区域的 max_cell_size 目标——
    但在真实案例上发现，至少留下一个这样的四面体（14.15 m^3，
    见 mesh_background_merge.py 自身历史了解该发现过程）完全未
    精化，无论 volume_cap_fraction 是放松还是收紧，或者区域有
    一个种子还是约 27 个分散种子都完全一样——都没改变那个单元。
    第一遍就已经存在的点不会被后来的精化遍"漏掉"——这绕开了
    对第二遍能到达远场的依赖，至少在本函数的（粗糙）间距下。

    两个过滤器保持候选网格不会适得其反：
      (a) 与现有 PLC 表面的间隙（`clearance_factor * target_edge_length`，
          通过 KDTree 检查最近 PLC 点）——靠近 BL 外表面或精细的
          core-only 壁，现有网格已经足够精细，在那里挤入背景点
          反而有产生退化薄片的风险；
      (b) 真正在封闭 PLC 体积内部（射线投射奇偶检验，复用
          mesh_domain_classify 自身的向量射线/三角相交例程）——
          PLC 外的点会违反 tetgen 的假设（每个输入点在其面围成
          的区域内），而对于非凸域（此处的真实可能性——车身自身
          的孔在箱形隧道中凿出凹面），单纯的包围盒网格不能保证。

    Args:
        plc_points: (n, 3) 完整 PLC 边界点集（BL 外表面 + core-only
            面）——与 `fill_core_volume` 自身接收的 `points` 相同
        plc_faces: (m, 3) 完整 PLC 边界三角，封闭且水密——与
            `fill_core_volume` 自身接收的 `faces` 相同数组
        target_edge_length: 远场分级目标（max_cell_size），这个网格
            不需要比它更精细
        grid_spacing_factor: 背景网格间距，作为 target_edge_length
            的倍数。故意比目标本身更粗糙——这是种子网格，用于
            打破否则巨大的初始四面体，不是区域自身体积精化的
            替代，后者仍然在它之上运行
        clearance_factor: 与最近 PLC 点的最小允许距离，作为
            target_edge_length 的倍数

    Returns:
        (k, 3) float64 背景点，如果域太小（相对于 target_edge_length）
        没有任何网格单元能通过两个过滤器则 k 可能为 0
    """
    from scipy.spatial import cKDTree
    from .mesh_domain_classify import _ray_triangle_intersect_count

    if target_edge_length <= 0.0 or len(plc_points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    bbox_min = plc_points.min(axis=0)
    bbox_max = plc_points.max(axis=0)
    spacing = target_edge_length * grid_spacing_factor

    axes = [
        np.arange(bbox_min[i] + spacing * 0.5, bbox_max[i], spacing)
        for i in range(3)
    ]
    if any(len(a) == 0 for a in axes):
        return np.empty((0, 3), dtype=np.float64)

    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    candidates = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    tree = cKDTree(plc_points)
    clearance = target_edge_length * clearance_factor
    dist, _ = tree.query(candidates, k=1, workers=-1)
    candidates = candidates[dist >= clearance]
    if len(candidates) == 0:
        logger.info("Core background-point seeding: 0 candidates cleared the PLC-clearance filter")
        return np.empty((0, 3), dtype=np.float64)

    v0 = plc_points[plc_faces[:, 0]]
    v1 = plc_points[plc_faces[:, 1]]
    v2 = plc_points[plc_faces[:, 2]]
    direction = np.array([1.0, 0.0, 0.0])
    inside_mask = np.zeros(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        hits = _ray_triangle_intersect_count(candidates[i], direction, v0, v1, v2)
        inside_mask[i] = (hits % 2) == 1

    result = candidates[inside_mask].astype(np.float64)
    logger.info(
        f"Core background-point seeding: {len(result)}/{len(candidates)} inside-domain "
        f"candidates kept (grid spacing={spacing:.3f}m, clearance={clearance:.3f}m)"
    )
    return result
