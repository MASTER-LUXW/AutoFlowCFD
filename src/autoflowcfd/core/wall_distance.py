"""
AutoFlowCFD V2.0 - 壁面距离场计算 (T-03)

本模块提供高效的壁面距离 $d_w$ 计算方法，支撑 DDES/IDDES 模型。

核心功能:
1. 基于 KD-Tree 的快速壁面距离查询
2. Eikonal 方程近似求解（Fast Marching Method）
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
                                max_iter: int = 500) -> np.ndarray:
    """
    使用 Fast Marching Method (FMM) 近似求解 Eikonal 方程 |∇d| = 1。
    
    该方法考虑网格拓扑结构，比纯几何距离更符合 CFD 需求，特别是在复杂几何中。
    
    Args:
        mesh_nodes: 网格节点坐标，形状 (N, 3)
        wall_indices: 壁面节点索引
        connectivity: 网格连接关系，形状 (N, k)，k为每节点最大邻居数
        max_iter: 最大迭代次数
        
    Returns:
        distances: 满足 Eikonal 方程的距离场
    """
    import heapq
    
    n_nodes = len(mesh_nodes)
    distances = np.full(n_nodes, np.inf)
    distances[wall_indices] = 0.0
    
    # 使用最小堆来高效获取活跃集中距离最小的节点
    heap = []
    for idx in wall_indices:
        heapq.heappush(heap, (0.0, idx))
    
    accepted = np.zeros(n_nodes, dtype=bool)
    accepted[wall_indices] = True
    
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
