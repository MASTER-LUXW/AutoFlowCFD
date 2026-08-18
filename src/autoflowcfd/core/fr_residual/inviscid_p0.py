"""
AutoFlowCFD V2.0 - P0 阶专用有限体积无粘残差 (S-02)

从 fr_residual_inviscid.py 拆出来（控制单文件行数，>400 行需拆分的
项目规范）。`compute_inviscid_residual_fr` 在 `mesh.n_points_1d==1`
时委托到这里，见该函数文档。

性能优化：将原纯 Python 逐面循环替换为 numba 并行 kernel
(inviscid_p0_kernel.py)，791K 单元 / 188 万面网格上从 ~25s 降至 ~1-2s。
"""

from typing import Callable, Optional

import numpy as np
import numba


def compute_inviscid_residual_fv_p0(
    U: np.ndarray,
    mesh,
    boundary_ghost_provider: Optional[Callable[[int, np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """P0（1 SP/cell，Order Continuation 最低阶）专用有限体积残差。

    算法与原 Python 版本完全一致（逐位等价，仅浮点重排顺序不同）：
    1. 提取 flat 面几何数组（单位法向、面积权重、面连接关系）
    2. 预计算边界幽灵态（Python 端，仅 ~40K 边界面）
    3. numba 并行 kernel 执行 AUSM+up 黎曼求解 + scatter-add
    4. per-thread buffer 归约得到最终残差

    关于棱柱四边形侧面拆分的处理：与原 Python 版本一致，按
    face_connectivity 的原始每条记录处理（不过滤 owner_is_primary），
    保证闭合单元面积/法向积分 Σ(n̂·A)=0 的几何恒等式不被破坏。

    Args:
        U: 守恒变量，形状 (n_cells, 1, n_vars)
        mesh: HighOrderMesh 实例（n_points_1d 必须为 1）
        boundary_ghost_provider: 同 compute_inviscid_residual_fr

    Returns:
        residual: 形状 (n_cells, 1, 5)
    """
    from .inviscid import conserved_to_primitive, DefaultGhostProvider
    from .inviscid_p0_kernel import _p0_inviscid_kernel

    n_cells = mesh.n_cells
    if mesh.cell_volumes is None:
        raise RuntimeError(
            "mesh.cell_volumes not available - required for the P0 finite-volume residual path "
            "(should have been computed once in load_from_volume_mesh at the mesh's target order)."
        )
    cell_volumes = mesh.cell_volumes

    Q_all = conserved_to_primitive(U[..., :5])[:, 0, :]  # (n_cells,5)

    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_faces = fc.n_faces

    # --- 提取 flat 面几何数组 ---
    unit_normals, area_weights = _extract_p0_face_geometry(ffp_list, n_faces)

    # --- 预计算边界幽灵态 ---
    ghost_provider = boundary_ghost_provider if boundary_ghost_provider is not None else DefaultGhostProvider()
    Q_ghost = _precompute_ghost_states(ffp_list, fc, ghost_provider, Q_all, n_faces)

    # --- numba 并行 kernel ---
    n_threads = numba.get_num_threads()
    owner_cell = fc.owner_cell.astype(np.int64)
    neighbor_cell = fc.neighbor_cell.astype(np.int64)
    is_boundary = fc.is_boundary.astype(np.bool_)

    residual_per_thread = _p0_inviscid_kernel(
        owner_cell, neighbor_cell, is_boundary,
        unit_normals, area_weights,
        Q_all, Q_ghost, cell_volumes,
        n_cells, n_threads,
    )

    # --- per-thread buffer 归约 ---
    residual5 = residual_per_thread.sum(axis=0)

    return residual5[:, None, :]


def _extract_p0_face_geometry(ffp_list, n_faces: int):
    """从 face_flux_points 提取 P0 需要的 flat 数组。

    支持两种数据源：
    - _KernelFaceData（快速路径）：直接读取 flat 数组
    - list of FaceFluxPointGeometry（慢速路径）：逐面提取

    Returns:
        unit_normals: (n_faces, 3) float64
        area_weights: (n_faces,) float64
    """
    from autoflowcfd.fr.face_flux_points_data import _KernelFaceData

    if isinstance(ffp_list, _KernelFaceData):
        # 快速路径：直接使用 flat 数组（P0: n_fp=1）
        unit_normals = np.ascontiguousarray(ffp_list.true_normal[:, 0, :])
        area_weights = np.ascontiguousarray(ffp_list.true_area_weight[:, 0])
    else:
        # 慢速路径：逐面提取（仅在非 _KernelFaceData 时）
        unit_normals = np.empty((n_faces, 3), dtype=np.float64)
        area_weights = np.empty(n_faces, dtype=np.float64)
        for f in range(n_faces):
            ffp = ffp_list[f]
            unit_normals[f] = ffp.true_normal[0]
            area_weights[f] = ffp.true_area_weight[0]

    return unit_normals, area_weights


def _precompute_ghost_states(ffp_list, fc, ghost_provider, Q_all, n_faces: int):
    """预计算边界面的幽灵态。

    在 Python 端遍历边界面（~40K 个），调用 ghost_provider 获取幽灵态，
    存储为 (n_cells, 5) 数组。numba kernel 内部通过 is_boundary 判断
    读取 Q_ghost[owner_cell] 而非 Q_all[neighbor_cell]。

    对于 DefaultGhostProvider（零梯度外插，ghost = owner），直接复制
    Q_all 即可（O(n_cells) 向量化操作，无需逐面循环）。
    """
    from .inviscid import DefaultGhostProvider

    if isinstance(ghost_provider, DefaultGhostProvider):
        # 快速路径：ghost = owner，直接复制
        return Q_all.copy()

    # 通用路径：逐面调用 ghost_provider
    Q_ghost = np.zeros_like(Q_all)
    boundary_mask = fc.is_boundary
    for f in range(n_faces):
        if boundary_mask[f]:
            oc = int(fc.owner_cell[f])
            ffp = ffp_list[f]
            Q_owner_fp = Q_all[oc:oc+1]  # (1,5)
            true_normal = ffp.true_normal  # (1,3)
            Q_ghost_fp = ghost_provider(f, Q_owner_fp, true_normal)  # (1,5)
            Q_ghost[oc] = Q_ghost_fp[0]

    return Q_ghost
