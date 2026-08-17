"""
AutoFlowCFD V2.0 - 壁面距离场计算 (T-03)

本模块提供高效的壁面距离 $d_w$ 计算方法，支撑 DDES/IDDES 模型。

核心功能:
1. 基于 KD-Tree 的快速壁面距离查询
2. Eikonal 方程近似求解（基于网格拓扑的单源最短路径松弛，非 FMM——
   见 solve_eikonal_approximation 文档里的命名更正说明）
3. 支持大规模网格的高效计算
"""

import numpy as np
from typing import Optional, Tuple


def compute_wall_distance_kdtree(mesh_nodes: np.ndarray, 
                                 wall_indices: np.ndarray,
                                 batch_size: int = 10000) -> np.ndarray:
    """
    使用 KD-Tree 加速计算网格节点到最近壁面的欧氏距离。
    
    相比暴力搜索 O(N*M)，KD-Tree 将复杂度降至 O(N*log(M))，
    适合处理百万级节点的工业网格。
    
    Args:
        mesh_nodes: 所有网格节点的坐标，形状 (N, 3)
        wall_indices: 壁面节点的索引数组
        batch_size: 批处理大小（避免内存溢出）
        
    Returns:
        distances: 每个节点到壁面的最小距离，形状 (N,)
        
    Raises:
        ImportError: 如果 scipy 未安装
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("Warning: scipy not available, falling back to simple method")
        return _compute_wall_distance_simple(mesh_nodes, wall_indices)
    
    # 提取壁面节点坐标
    wall_nodes = mesh_nodes[wall_indices]
    
    # 构建 KD-Tree
    tree = cKDTree(wall_nodes)
    
    # 批处理查询以避免内存问题
    n_total = len(mesh_nodes)
    distances = np.empty(n_total)
    
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_points = mesh_nodes[start:end]
        
        # query 返回 (distance, index)
        dist_batch, _ = tree.query(batch_points, k=1)
        distances[start:end] = dist_batch
    
    return distances


def _compute_wall_distance_simple(mesh_nodes: np.ndarray, 
                                  wall_indices: np.ndarray) -> np.ndarray:
    """
    简化版壁面距离计算（无 KD-Tree 依赖）。
    
    适用于小规模网格或 scipy 不可用的情况。
    
    Args:
        mesh_nodes: 所有网格节点的坐标
        wall_indices: 壁面节点的索引数组
        
    Returns:
        distances: 每个节点到壁面的最小距离
    """
    wall_nodes = mesh_nodes[wall_indices]
    
    # 对于大规模网格，采用分块计算减少内存占用
    n_total = len(mesh_nodes)
    n_wall = len(wall_nodes)
    distances = np.full(n_total, np.inf)
    
    # 如果壁面节点太多，也进行分块
    wall_batch_size = 5000
    
    for i in range(0, n_total, 1000):
        end_i = min(i + 1000, n_total)
        node_batch = mesh_nodes[i:end_i]
        
        for j in range(0, n_wall, wall_batch_size):
            end_j = min(j + wall_batch_size, n_wall)
            wall_batch = wall_nodes[j:end_j]
            
            # 计算当前批次的距离
            # 使用 broadcasting: (n_nodes, 1, 3) - (1, n_wall, 3)
            diff = node_batch[:, np.newaxis, :] - wall_batch[np.newaxis, :, :]
            dist_batch = np.linalg.norm(diff, axis=2)
            
            # 更新最小值
            min_dist_batch = np.min(dist_batch, axis=1)
            distances[i:end_i] = np.minimum(distances[i:end_i], min_dist_batch)
    
    return distances


def solve_eikonal_approximation(mesh_nodes: np.ndarray,
                                wall_indices: np.ndarray,
                                connectivity: np.ndarray,
                                max_iter: Optional[int] = None) -> np.ndarray:
    """
    基于网格拓扑的单边松弛 Dijkstra 最短路径，近似求解 Eikonal 方程 |∇d| = 1。

    命名/文档更正：此前这里自称 "Fast Marching Method (FMM)"，但实现是
    经典单源最短路径松弛（每次只用一条边的欧氏长度更新一个邻居的距离），
    不是 FMM——真正的 FMM 需要在每个被 accept 的节点上，用其已 accept
    的多个邻居联合求解局部 Eikonal 方程的迎风二次型（而不是逐边独立
    松弛），精度更高但实现复杂得多。当前的 Dijkstra 近似对各向同性、
    分辨率较均匀的网格是合理的工程近似，但不应自称 FMM——已改正命名与
    文档，未改变算法本身（真正实现 FMM 是独立的、工作量更大的后续任务）。

    该方法考虑网格拓扑结构，比纯几何距离更符合 CFD 需求，特别是在复杂几何中。

    Args:
        mesh_nodes: 网格节点坐标，形状 (N, 3)
        wall_indices: 壁面节点索引
        connectivity: 网格连接关系，形状 (N, k)，k为每节点最大邻居数
        max_iter: 最大迭代次数；None（默认）时用 n_nodes——每个节点最多被
            accept 一次，这是算法本身的迭代次数上界，不是可调参数。此前
            硬编码默认值 500 在真实规模网格下会导致大量节点距离恒为 inf。

    Returns:
        distances: 满足 Eikonal 方程的距离场
    """
    import heapq

    n_nodes = len(mesh_nodes)
    if max_iter is None:
        max_iter = n_nodes
    distances = np.full(n_nodes, np.inf)
    distances[wall_indices] = 0.0

    accepted = np.zeros(n_nodes, dtype=bool)
    accepted[wall_indices] = True

    heap = []

    # 用壁面种子节点各自的邻居直接初始化堆——不把壁面节点自己压进堆里
    # 再靠主循环处理：壁面节点在上面已经被标记 accepted=True（它们的
    # 距离 0 是已知的最终解），如果按原实现把它们也压进堆、指望在主循环
    # 里 pop 出来时展开邻居，会在循环开头 `if accepted[min_idx]: continue`
    # 直接跳过——那正是它们本该展开邻居的唯一机会，结果是**所有壁面节点
    # 的邻居都永远不会被访问，扩散从未真正发生**。这不是 max_iter 太小
    # 的问题：用 3000 节点合成网格实测，无论 max_iter 多大，除壁面节点
    # 自身外其余节点全部保持 inf（inf 数量精确等于 n_nodes-n_wall），
    # 与真实 review 报告里"99.3% 为 inf"的数字吻合（当时的壁面节点占比
    # 恰好约 0.7%），说明 review 观测到的现象背后其实是这个更根本的
    # 逻辑 bug，max_iter=500 只是让它在小规模合成网格上更早暴露。
    for idx in wall_indices:
        neighbors = connectivity[idx]
        for neighbor in neighbors:
            if neighbor == -1 or accepted[neighbor]:
                continue
            delta_d = np.linalg.norm(mesh_nodes[neighbor] - mesh_nodes[idx])
            if delta_d < distances[neighbor]:
                distances[neighbor] = delta_d
                heapq.heappush(heap, (delta_d, neighbor))

    iter_count = 0
    while heap and iter_count < max_iter:
        dist, min_idx = heapq.heappop(heap)

        if accepted[min_idx]:
            continue

        accepted[min_idx] = True
        distances[min_idx] = dist
        iter_count += 1

        # 更新邻居
        neighbors = connectivity[min_idx]
        for neighbor in neighbors:
            if neighbor == -1 or accepted[neighbor]:
                continue

            # 计算欧氏距离增量
            delta_d = np.linalg.norm(mesh_nodes[neighbor] - mesh_nodes[min_idx])
            new_dist = dist + delta_d

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return distances


def compute_wall_distance(mesh_nodes: np.ndarray, 
                         wall_indices: np.ndarray,
                         connectivity: Optional[np.ndarray] = None,
                         use_eikonal: bool = False) -> np.ndarray:
    """
    统一的壁面距离计算接口。
    
    Args:
        mesh_nodes: 网格节点坐标
        wall_indices: 壁面节点索引
        connectivity: 网格连接关系（仅 Eikonal 模式需要）
        use_eikonal: 是否使用 Eikonal 方程求解
        
    Returns:
        distances: 壁面距离场
        
    Examples:
        >>> # 快速 KD-Tree 方法
        >>> d = compute_wall_distance(nodes, wall_idx)
        
        >>> # Eikonal 方法（更准确但较慢）
        >>> d = compute_wall_distance(nodes, wall_idx, conn, use_eikonal=True)
    """
    if use_eikonal:
        if connectivity is None:
            raise ValueError("connectivity required for Eikonal solver")
        return solve_eikonal_approximation(mesh_nodes, wall_indices, connectivity)
    else:
        return compute_wall_distance_kdtree(mesh_nodes, wall_indices)


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成测试网格
    n_nodes = 10000
    mesh_nodes = np.random.rand(n_nodes, 3) * 0.1
    
    # 假设前100个节点是壁面
    wall_indices = np.arange(100)
    
    # 测试 KD-Tree 方法
    print("Testing KD-Tree method...")
    distances = compute_wall_distance(mesh_nodes, wall_indices)
    print(f"Distance stats: min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}")
    
    print("Wall distance computation completed.")
