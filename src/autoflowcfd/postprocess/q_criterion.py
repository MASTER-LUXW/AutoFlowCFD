"""Q-Criterion 涡识别准则计算模块 (P-02)。

Q-Criterion 是涡动力学中最常用的涡识别准则之一，定义为速度梯度张量
的反对称部分（旋转张量 Ω）与对称部分（应变率张量 S）的 Frobenius 范数
之差的一半：

    Q = 0.5 * (||Ω||² - ||S||²)

其中：
    L = ∇u（速度梯度张量）
    S = 0.5 * (L + L^T)（应变率张量）
    Ω = 0.5 * (L - L^T)（旋转张量）

Q > 0 的区域表示旋转占主导（涡核区域），Q < 0 的区域表示剪切/应变
占主导。

实现方式：
    使用 Green-Gauss 方法在非结构网格上重建速度梯度：

        ∇u ≈ (1/V_cell) * Σ_f (u_f ⊗ S_f)

    其中 u_f 为面上的速度（内部面取 owner/neighbor 均值，边界面取
    owner 单元值），S_f = n_f * A_f 为面面积向量。

    这是有限体积/通量重构方法中计算单元中心梯度的标准方法，与求解器
    自身计算梯度的方式一致。
"""

import numpy as np
from typing import Optional
from loguru import logger


def compute_velocity_gradient_green_gauss(
    cell_velocity: np.ndarray,
    face_connectivity: np.ndarray,
    face_area_vectors: np.ndarray,
    cell_volumes: np.ndarray,
) -> np.ndarray:
    """使用 Green-Gauss 方法计算单元中心速度梯度。

    Args:
        cell_velocity: (n_cells, 3) 单元中心速度
        face_connectivity: (n_faces, 2) 面连接关系，[:, 0]=owner，
            [:, 1]=neighbor（边界面 neighbor=-1）
        face_area_vectors: (n_faces, 3) 面面积向量 (normal * area)
        cell_volumes: (n_cells,) 单元体积

    Returns:
        (n_cells, 3, 3) 速度梯度张量，grad[i,j] = ∂u_i/∂x_j
    """
    n_cells = cell_velocity.shape[0]
    owner = face_connectivity[:, 0].astype(np.int64)
    neighbor = face_connectivity[:, 1].astype(np.int64)

    # 内部面：owner/neighbor 速度均值；边界面：仅用 owner 速度
    is_interior = neighbor >= 0
    is_boundary = ~is_interior

    # 面上速度 (n_faces, 3)
    u_face = np.empty_like(face_area_vectors)
    u_face[is_interior] = 0.5 * (
        cell_velocity[owner[is_interior]] + cell_velocity[neighbor[is_interior]]
    )
    u_face[is_boundary] = cell_velocity[owner[is_boundary]]

    # 通过散度定理计算梯度：∇u_i = (1/V) * Σ_f (u_i * S_f)
    # S_f = face_area_vectors (n_faces, 3)
    # 对每个速度分量 i 和空间方向 j：∂u_i/∂x_j = (1/V) * Σ_f (u_i * S_f_j)
    # 向量化：grad[cell] = (1/V[cell]) * Σ_f (u_face[f] ⊗ S_f)

    # 外积 u ⊗ S 的每个分量：(n_faces, 3, 3)
    # flux_outer[i,j] = u_face[:,i] * face_area_vectors[:,j]
    flux_outer = u_face[:, :, np.newaxis] * face_area_vectors[:, np.newaxis, :]

    # 对每个面贡献到 owner 单元（散度定理）
    grad = np.zeros((n_cells, 3, 3), dtype=np.float64)
    np.add.at(grad, owner, flux_outer)

    # 除以单元体积
    inv_vol = 1.0 / np.maximum(cell_volumes, 1e-30)
    grad *= inv_vol[:, np.newaxis, np.newaxis]

    return grad


def compute_q_criterion(
    cell_velocity: np.ndarray,
    face_connectivity: np.ndarray,
    face_area_vectors: np.ndarray,
    cell_volumes: np.ndarray,
) -> np.ndarray:
    """计算 Q-Criterion 涡识别准则。

    Q = 0.5 * (||Ω||² - ||S||²)

    其中 Ω 和 S 分别是速度梯度张量的反对称部分和对称部分。

    等价公式（不可压缩流简化形式，此处仅作参考，实际使用完整公式）：
        ||S||² = 0.5 * (||L||² + (tr L)²)
        ||Ω||² = 0.5 * (||L||² - (tr L)²)
        Q = 0.25 * ((tr L)² - 2*tr(L²))

    对可压缩流（本项目是可压缩 N-S），使用完整公式：
        Q = 0.5 * (||Ω||²_F - ||S||²_F)

    Args:
        cell_velocity: (n_cells, 3) 单元中心速度 (u, v, w)
        face_connectivity: (n_faces, 2) 面连接关系
        face_area_vectors: (n_faces, 3) 面面积向量
        cell_volumes: (n_cells,) 单元体积

    Returns:
        (n_cells,) Q-Criterion 标量场
    """
    # 计算速度梯度 L[i,j] = ∂u_i/∂x_j
    L = compute_velocity_gradient_green_gauss(
        cell_velocity, face_connectivity, face_area_vectors, cell_volumes
    )

    # 对称部分 S = 0.5 * (L + L^T)
    S = 0.5 * (L + L.transpose(0, 2, 1))

    # 反对称部分 Ω = 0.5 * (L - L^T)
    Omega = 0.5 * (L - L.transpose(0, 2, 1))

    # Frobenius 范数的平方：||A||² = Σ_{i,j} A_{ij}²
    S_norm_sq = np.sum(S * S, axis=(1, 2))
    Omega_norm_sq = np.sum(Omega * Omega, axis=(1, 2))

    # Q-Criterion
    Q = 0.5 * (Omega_norm_sq - S_norm_sq)

    return Q


def compute_q_criterion_from_grid_solution(
    grid_data,
    solution,
) -> Optional[np.ndarray]:
    """从网格数据和求解结果计算 Q-Criterion（供 VTK 导出使用）。

    自动从 grid_data 获取面连接关系（如不存在则尝试提取），从 solution
    获取单元中心速度，计算 Q-Criterion 标量场。

    Args:
        grid_data: GridData 或 VolumeMeshData 实例
        solution: SolutionVector 实例

    Returns:
        (n_cells,) Q-Criterion 标量场，无法计算时返回 None
    """
    if solution is None or solution.data is None or solution.n_cells == 0:
        logger.warning("Q-Criterion: solution data not available")
        return None

    # 获取单元中心速度
    try:
        u, v, w = solution.get_velocity()
        cell_velocity = np.column_stack([u, v, w])
    except Exception as e:
        logger.warning(f"Q-Criterion: failed to get velocity from solution: {e}")
        return None

    # 获取面数据
    faces = None
    if hasattr(grid_data, 'faces') and grid_data.faces is not None:
        faces = grid_data.faces
    elif hasattr(grid_data, 'ensure_faces_exist'):
        try:
            faces = grid_data.ensure_faces_exist()
        except Exception as e:
            logger.warning(f"Q-Criterion: failed to extract face data: {e}")
            return None

    if faces is None:
        logger.warning(
            "Q-Criterion: no face data available. This field requires a volume "
            "mesh with face connectivity. Regenerate the volume mesh or use a "
            "cached .pkl file."
        )
        return None

    # 获取单元体积
    if hasattr(grid_data, 'get_cell_volumes'):
        cell_volumes = grid_data.get_cell_volumes()
    elif hasattr(grid_data, 'cells') and hasattr(grid_data.cells, 'volumes'):
        cell_volumes = grid_data.cells.volumes
    else:
        logger.warning("Q-Criterion: cell volumes not available")
        return None

    # 计算 Q-Criterion
    try:
        Q = compute_q_criterion(
            cell_velocity=cell_velocity,
            face_connectivity=faces.connectivity,
            face_area_vectors=faces.area_vectors,
            cell_volumes=cell_volumes,
        )
        logger.info(
            f"Q-Criterion computed: min={Q.min():.4e}, max={Q.max():.4e}, "
            f"mean={Q.mean():.4e}"
        )
        return Q
    except Exception as e:
        logger.warning(f"Q-Criterion computation failed: {e}")
        return None
