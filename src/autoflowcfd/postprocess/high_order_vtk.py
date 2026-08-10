"""
AutoFlowCFD V2.0 - 高阶 VTK 导出与后处理 (P-01, P-02)

本模块支持将 FR 求解器的高阶多项式数据转换为 VTK Lagrange 单元格式，
并计算 Q-Criterion 和 Lambda2 用于涡结构识别。

核心功能:
1. 高阶数据到 VTK Lagrange 单元的转换
2. Q-Criterion（第二不变量）计算
3. Lambda2 准则计算
4. 支持 pyvista 和 vtk 库
"""

import numpy as np
from typing import Optional, Dict


def compute_q_criterion(grad_u: np.ndarray, grad_v: np.ndarray, 
                       grad_w: np.ndarray) -> np.ndarray:
    """
    计算 Q-Criterion（速度梯度张量的第二不变量）。
    
    Q = 0.5 * (||Ω||² - ||S||²)
    
    其中：
    - S = 0.5 * (∇u + ∇u^T) 是应变率张量
    - Ω = 0.5 * (∇u - ∇u^T) 是旋转率张量
    
    Q > 0 表示旋转主导区域（涡核）
    
    Args:
        grad_u: ∂u/∂x, ∂u/∂y, ∂u/∂z，形状 (n_points, 3)
        grad_v: ∂v/∂x, ∂v/∂y, ∂v/∂z，形状 (n_points, 3)
        grad_w: ∂w/∂x, ∂w/∂y, ∂w/∂z，形状 (n_points, 3)
        
    Returns:
        Q: Q-Criterion 标量场，形状 (n_points,)
    """
    n_points = grad_u.shape[0]
    
    # 组装完整的速度梯度张量 ∇u
    # grad_u[i, j] = ∂u_i/∂x_j
    grad_tensor = np.zeros((n_points, 3, 3))
    grad_tensor[:, 0, :] = grad_u  # u 的梯度
    grad_tensor[:, 1, :] = grad_v  # v 的梯度
    grad_tensor[:, 2, :] = grad_w  # w 的梯度
    
    # 计算应变率张量 S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    S = 0.5 * (grad_tensor + np.transpose(grad_tensor, (0, 2, 1)))
    
    # 计算旋转率张量 Ω_ij = 0.5 * (∂u_i/∂x_j - ∂u_j/∂x_i)
    Omega = 0.5 * (grad_tensor - np.transpose(grad_tensor, (0, 2, 1)))
    
    # 计算范数平方
    # ||S||² = S_ij * S_ij
    S_sq = np.einsum('nij,nij->n', S, S)
    
    # ||Ω||² = Ω_ij * Ω_ij
    Omega_sq = np.einsum('nij,nij->n', Omega, Omega)
    
    # Q = 0.5 * (||Ω||² - ||S||²)
    Q = 0.5 * (Omega_sq - S_sq)
    
    return Q


def compute_lambda2_criterion(grad_u: np.ndarray, grad_v: np.ndarray,
                             grad_w: np.ndarray) -> np.ndarray:
    """
    计算 Lambda2 准则（基于压力 Hessian 的特征值）。
    
    Lambda2 是矩阵 S² + Ω² 的第二大特征值。
    Lambda2 < 0 表示涡核区域。
    
    Args:
        grad_u, grad_v, grad_w: 速度梯度分量
        
    Returns:
        lambda2: Lambda2 标量场
    """
    n_points = grad_u.shape[0]
    
    # 组装速度梯度张量
    grad_tensor = np.zeros((n_points, 3, 3))
    grad_tensor[:, 0, :] = grad_u
    grad_tensor[:, 1, :] = grad_v
    grad_tensor[:, 2, :] = grad_w
    
    # 计算 S 和 Ω
    S = 0.5 * (grad_tensor + np.transpose(grad_tensor, (0, 2, 1)))
    Omega = 0.5 * (grad_tensor - np.transpose(grad_tensor, (0, 2, 1)))
    
    # 计算 S² + Ω²
    M = np.einsum('nik,nkj->nij', S, S) + np.einsum('nik,nkj->nij', Omega, Omega)
    
    # 计算特征值
    lambda2 = np.zeros(n_points)
    for i in range(n_points):
        eigenvalues = np.linalg.eigvals(M[i])
        # 排序并取第二大特征值
        eigenvalues_sorted = np.sort(np.real(eigenvalues))
        lambda2[i] = eigenvalues_sorted[1]  # 第二大的
    
    return lambda2


def export_to_vtk_lagrange(sps_coords: np.ndarray, U: np.ndarray, 
                          filename: str, order: int = 2,
                          field_names: Optional[Dict[int, str]] = None):
    """
    将 SPs 上的高阶数据导出为 VTK Lagrange 单元格式。
    
    Args:
        sps_coords: SPs 物理坐标，形状 (n_cells, n_sps_per_cell, 3)
        U: 守恒变量场，形状 (n_cells, n_sps_per_cell, n_vars)
        filename: 输出文件名 (.vtu 或 .vtk)
        order: 多项式阶数
        field_names: 变量名映射字典 {var_index: name}
        
    Example:
        >>> export_to_vtk_lagrange(coords, U, "result.vtu", order=2,
        ...                       field_names={0: 'density', 4: 'pressure'})
    """
    try:
        import pyvista as pv
    except ImportError:
        print("Warning: pyvista not available, using simple VTK writer")
        return _export_to_vtk_simple(sps_coords, U, filename, field_names)
    
    n_cells, n_sps, _ = sps_coords.shape
    
    # 展平坐标和变量
    points = sps_coords.reshape(-1, 3)
    
    # 创建 UnstructuredGrid
    grid = pv.UnstructuredGrid()
    grid.points = points
    
    # 添加单元连接关系（简化：假设六面体单元）
    # 实际应根据网格类型生成正确的连接
    cells = []
    for cell_id in range(n_cells):
        # 每个单元的 SPs 索引
        start_idx = cell_id * n_sps
        cell_indices = list(range(start_idx, start_idx + n_sps))
        cells.append(n_sps)  # 节点数
        cells.extend(cell_indices)
    
    cell_types = np.full(n_cells, pv.CellType.LAGRANGE_HEXAHEDRON)
    grid.set_cells(cell_types, np.array(cells))
    
    # 添加点数据
    for var_idx in range(U.shape[2]):
        var_data = U[:, :, var_idx].flatten()
        
        if field_names and var_idx in field_names:
            name = field_names[var_idx]
        else:
            name = f'var_{var_idx}'
        
        grid.point_data[name] = var_data
    
    # 保存
    grid.save(filename)
    print(f"Exported high-order data to {filename}")
    print(f"  Points: {len(points)}")
    print(f"  Cells: {n_cells}")
    print(f"  Order: P{order}")


def _export_to_vtk_simple(sps_coords: np.ndarray, U: np.ndarray,
                         filename: str, 
                         field_names: Optional[Dict[int, str]] = None):
    """
    简化的 VTK 导出（不依赖 pyvista）。
    
    Args:
        sps_coords: SPs 坐标
        U: 守恒变量场
        filename: 输出文件名
        field_names: 变量名映射
    """
    n_cells, n_sps, _ = sps_coords.shape
    n_total_points = n_cells * n_sps
    
    with open(filename, 'w') as f:
        # VTK 文件头
        f.write("# vtk DataFile Version 3.0\n")
        f.write("AutoFlowCFD V2.0 High-Order FR Results\n")
        f.write("ASCII\n\n")
        
        # 数据集
        f.write("DATASET UNSTRUCTURED_GRID\n\n")
        
        # 点坐标
        f.write(f"POINTS {n_total_points} float\n")
        for cell in range(n_cells):
            for sp in range(n_sps):
                x, y, z = sps_coords[cell, sp, :]
                f.write(f"{x:.6e} {y:.6e} {z:.6e}\n")
        f.write("\n")
        
        # 单元连接（简化为顶点集）
        n_cells_vtk = n_cells
        f.write(f"CELLS {n_cells_vtk} {n_cells_vtk * (n_sps + 1)}\n")
        for cell in range(n_cells):
            start_idx = cell * n_sps
            f.write(f"{n_sps} ")
            for sp in range(n_sps):
                f.write(f"{start_idx + sp} ")
            f.write("\n")
        f.write("\n")
        
        # 单元类型（VERTEX）
        f.write(f"CELL_TYPES {n_cells_vtk}\n")
        for _ in range(n_cells):
            f.write("1\n")  # VTK_VERTEX
        f.write("\n")
        
        # 点数据
        n_vars = U.shape[2]
        f.write(f"POINT_DATA {n_total_points}\n")
        
        for var_idx in range(n_vars):
            if field_names and var_idx in field_names:
                name = field_names[var_idx]
            else:
                name = f'var_{var_idx}'
            
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            
            for cell in range(n_cells):
                for sp in range(n_sps):
                    f.write(f"{U[cell, sp, var_idx]:.6e}\n")
            f.write("\n")
    
    print(f"Exported simple VTK to {filename}")


def compute_and_export_vorticity(sps_coords: np.ndarray, U: np.ndarray,
                                grad_operator: np.ndarray,
                                output_dir: str = "./results"):
    """
    计算涡量并导出可视化文件。
    
    Args:
        sps_coords: SPs 坐标
        U: 守恒变量场
        grad_operator: FR 微分算子
        output_dir: 输出目录
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n_cells, n_sps, _ = sps_coords.shape
    
    # 提取速度分量
    rho = U[:, :, 0]
    u = U[:, :, 1] / rho
    v = U[:, :, 2] / rho
    w = U[:, :, 3] / rho
    
    # 计算速度梯度
    grad_u = np.zeros((n_cells, n_sps, 3))
    grad_v = np.zeros((n_cells, n_sps, 3))
    grad_w = np.zeros((n_cells, n_sps, 3))
    
    for dim in range(3):
        grad_u[:, :, dim] = np.einsum('ijk,ik->ij', grad_operator[:, :, dim], u)
        grad_v[:, :, dim] = np.einsum('ijk,ik->ij', grad_operator[:, :, dim], v)
        grad_w[:, :, dim] = np.einsum('ijk,ik->ij', grad_operator[:, :, dim], w)
    
    # 计算 Q-Criterion
    Q = compute_q_criterion(grad_u.reshape(-1, 3),
                           grad_v.reshape(-1, 3),
                           grad_w.reshape(-1, 3))
    Q = Q.reshape(n_cells, n_sps)
    
    # 计算 Lambda2
    lambda2 = compute_lambda2_criterion(grad_u.reshape(-1, 3),
                                       grad_v.reshape(-1, 3),
                                       grad_w.reshape(-1, 3))
    lambda2 = lambda2.reshape(n_cells, n_sps)
    
    # 添加到 U 场
    U_extended = np.concatenate([U, Q[:, :, np.newaxis], lambda2[:, :, np.newaxis]], axis=2)
    
    field_names = {
        0: 'density',
        1: 'momentum_x',
        2: 'momentum_y',
        3: 'momentum_z',
        4: 'energy',
        5: 'Q_criterion',
        6: 'lambda2'
    }
    
    # 导出
    filename = f"{output_dir}/vorticity_analysis.vtu"
    export_to_vtk_lagrange(sps_coords, U_extended, filename, 
                          field_names=field_names)
    
    print(f"Vorticity analysis complete:")
    print(f"  Q-criterion range: [{Q.min():.6e}, {Q.max():.6e}]")
    print(f"  Lambda2 range: [{lambda2.min():.6e}, {lambda2.max():.6e}]")


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    n_cells = 10
    n_sps = 8
    
    # 创建测试数据
    sps_coords = np.random.rand(n_cells, n_sps, 3) * 0.1
    U = np.random.rand(n_cells, n_sps, 5)
    
    # 模拟速度梯度
    grad_u = np.random.rand(n_cells * n_sps, 3) * 100.0
    grad_v = np.random.rand(n_cells * n_sps, 3) * 100.0
    grad_w = np.random.rand(n_cells * n_sps, 3) * 100.0
    
    # 计算 Q-Criterion
    Q = compute_q_criterion(grad_u, grad_v, grad_w)
    print(f"Q-Criterion computed: min={Q.min():.6e}, max={Q.max():.6e}")
    
    # 计算 Lambda2
    lambda2 = compute_lambda2_criterion(grad_u, grad_v, grad_w)
    print(f"Lambda2 computed: min={lambda2.min():.6e}, max={lambda2.max():.6e}")
    
    # 测试 VTK 导出
    export_to_vtk_lagrange(sps_coords, U, "test_output.vtu", order=2)
    
    print("\nPostprocessing test completed.")
