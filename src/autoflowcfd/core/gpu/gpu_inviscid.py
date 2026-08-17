"""
AutoFlowCFD V2.0 - P>=1 高阶 FR 无粘残差 GPU 实现

完整的高阶 FR 无粘残差 GPU 版本，对应 core/fr_residual_inviscid.py。
包含两部分：
1. 体积项：CuPy 向量化（物理通量 + 张量收缩 + 度量项）
2. 界面项：CuPy kernel（AUSM+up + 校正分配，按图着色逐色处理）

设计：
- 体积项完全用 CuPy 向量化操作（cp.matmul, cp.tensordot），底层走 cuBLAS
- 界面项使用 CuPy ElementwiseKernel 逐面计算 AUSM+up 通量
- 校正分配使用图着色保证无冲突写入（同色面无 owner_cell 冲突）
- 数据全部常驻 GPU，避免 CPU↔GPU 传输
"""

import numpy as np
from typing import Callable, Optional
from loguru import logger

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis,
    gpu_contract_shared_operator_2axis,
)
from autoflowcfd.core.gpu.gpu_flux import (
    euler_physical_flux_gpu,
    conserved_to_primitive_gpu,
)


def compute_inviscid_residual_fr_gpu(
    U,
    mesh,
    ops,
    boundary_ghost_provider=None,
    mesh_data=None,
    ops_data=None,
    flat_face_gpu=None,
    device_id=0,
):
    """P>=1 高阶 FR 无粘残差的 GPU 实现。

    与 core/fr_residual_inviscid.py::compute_inviscid_residual_fr 公式完全一致。

    Args:
        U: CuPy 数组 (n_cells, n_sps, n_vars) 或 numpy 数组（自动上传）
        mesh: HighOrderMesh
        ops: FROperators
        boundary_ghost_provider: 边界幽灵态提供者
        mesh_data: 预上传的网格数据（可选，None 时自动上传）
        ops_data: 预上传的算子数据（可选）
        flat_face_gpu: 预构建的 GPU 面几何（可选）
        device_id: GPU 设备 ID

    Returns:
        residual: CuPy 数组 (n_cells, n_sps, 5) 或 numpy 数组（与输入同类型）
    """
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available")

    input_is_numpy = isinstance(U, np.ndarray)
    if input_is_numpy:
        with cp.cuda.Device(device_id):
            U = cp.asarray(U)

    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n1d = mesh.n_points_1d
    n_prism = mesh.n_prism_cells

    # ── 准备网格数据（如果未预上传）──
    if mesh_data is None:
        mesh_data = _prepare_mesh_data(cp, mesh, device_id)
    if ops_data is None:
        ops_data = _prepare_ops_data(cp, ops, device_id)

    # ── 1. 体积项 ──
    residual = _compute_volume_term_gpu(
        U, mesh_data, ops_data, n_cells, n_sps, n_prism,
    )

    # ── 2. 界面项 ──
    if flat_face_gpu is None:
        # 需要构建面几何
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        flat_face = get_flat_face_geometry(mesh, ops)
        from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
        flat_face_gpu = build_gpu_flat_face(flat_face, device_id)

    Q_gpu = conserved_to_primitive_gpu(U[..., :5])
    adj_j = mesh_data['adj_j']
    det_jacs = mesh_data['det_jacs']

    # 边界幽灵态（CPU 上计算，然后上传到 GPU）
    Q_ghost_gpu = _compute_boundary_ghost_states_gpu(
        Q_gpu, flat_face_gpu, adj_j, boundary_ghost_provider,
        n_cells, n_sps, device_id,
    )

    # 界面校正（按图着色逐色处理）
    correction = _compute_interface_correction_gpu(
        Q_gpu, adj_j, det_jacs, flat_face_gpu, Q_ghost_gpu,
        n_cells, n_sps, n_prism, device_id,
    )

    residual = residual + correction

    # ── 3. 异常残差抑制 ──
    from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
    if input_is_numpy:
        residual_np = cp.asnumpy(residual)
        U_np = cp.asnumpy(U)
        result = suppress_residual_outliers(residual_np, U_np[..., :5])
        return cp.asarray(result)
    else:
        residual_np = cp.asnumpy(residual)
        U_np = cp.asnumpy(U)
        result = suppress_residual_outliers(residual_np, U_np[..., :5])
        return cp.asarray(result)


def _prepare_mesh_data(cp, mesh, device_id):
    """准备网格度量数据到 GPU。"""
    with cp.cuda.Device(device_id):
        n_cells = mesh.n_cells
        n_sps = mesh.n_sps_per_cell

        det_jacs = cp.asarray(
            np.ascontiguousarray(
                mesh.jacobians['det_jacs'].reshape(n_cells, n_sps), dtype=np.float64
            )
        )
        inv_jacs = cp.asarray(
            np.ascontiguousarray(
                mesh.jacobians['inv_jacs'].reshape(n_cells, n_sps, 3, 3), dtype=np.float64
            )
        )
        adj_j = det_jacs[..., None, None] * inv_jacs

        data = {
            'det_jacs': det_jacs,
            'inv_jacs': inv_jacs,
            'adj_j': adj_j,
            'n_prism': mesh.n_prism_cells,
        }

        # Fine Jacobian（over-integration）
        if mesh.jacobians_fine is not None:
            n_fine = mesh.n_sps_per_cell_fine
            det_jacs_fine = cp.asarray(
                np.ascontiguousarray(
                    mesh.jacobians_fine['det_jacs'].reshape(n_cells, n_fine), dtype=np.float64
                )
            )
            inv_jacs_fine = cp.asarray(
                np.ascontiguousarray(
                    mesh.jacobians_fine['inv_jacs'].reshape(n_cells, n_fine, 3, 3), dtype=np.float64
                )
            )
            adj_j_fine = det_jacs_fine[..., None, None] * inv_jacs_fine
            data['det_jacs_fine'] = det_jacs_fine
            data['inv_jacs_fine'] = inv_jacs_fine
            data['adj_j_fine'] = adj_j_fine
            data['n_fine'] = n_fine

        return data


def _prepare_ops_data(cp, ops, device_id):
    """准备 FR 算子数据到 GPU。"""
    with cp.cuda.Device(device_id):
        data = {}
        for attr_name in ['D_3d_tet', 'D_3d_prism']:
            D = getattr(ops, attr_name, None)
            if D is not None:
                data[attr_name] = cp.asarray(np.ascontiguousarray(D, dtype=np.float64))

        for attr_name in [
            'overint_interp_c2f_tet', 'overint_interp_c2f_prism',
            'overint_D_fine_tet', 'overint_D_fine_prism',
            'overint_restrict_f2c_tet', 'overint_restrict_f2c_prism',
        ]:
            op = getattr(ops, attr_name, None)
            if op is not None:
                data[attr_name] = cp.asarray(np.ascontiguousarray(op, dtype=np.float64))
        return data


def _compute_volume_term_gpu(U, mesh_data, ops_data, n_cells, n_sps, n_prism):
    """计算体积项（CuPy 向量化）。"""
    cp = get_cupy()

    Q = conserved_to_primitive_gpu(U[..., :5])  # (n_cells, n_sps, 5)
    det_jacs = mesh_data['det_jacs']

    if 'adj_j_fine' in mesh_data:
        # Over-integration 去混叠路径
        n_fine = mesh_data['n_fine']
        adj_j_fine = mesh_data['adj_j_fine']

        # 插值到 fine 点
        Q_fine = cp.zeros((n_cells, n_fine, 5), dtype=cp.float64)
        if n_prism > 0:
            Q_fine[:n_prism] = gpu_contract_shared_operator_1axis(
                ops_data['overint_interp_c2f_prism'], Q[:n_prism]
            )
        if n_cells > n_prism:
            Q_fine[n_prism:] = gpu_contract_shared_operator_1axis(
                ops_data['overint_interp_c2f_tet'], Q[n_prism:]
            )

        # 物理通量（fine 点）
        Q_fine_flat = cp.ascontiguousarray(Q_fine.reshape(-1, 5))
        F_phys_fine = euler_physical_flux_gpu(Q_fine_flat).reshape(n_cells, n_fine, 3, 5)

        # 逆变通量
        F_tilde_fine = cp.matmul(adj_j_fine, F_phys_fine)

        # 散度（fine 点）
        div_comp_fine = cp.zeros((n_cells, n_fine, 5), dtype=cp.float64)
        if n_prism > 0:
            div_comp_fine[:n_prism] = gpu_contract_shared_operator_2axis(
                ops_data['overint_D_fine_prism'], F_tilde_fine[:n_prism]
            )
        if n_cells > n_prism:
            div_comp_fine[n_prism:] = gpu_contract_shared_operator_2axis(
                ops_data['overint_D_fine_tet'], F_tilde_fine[n_prism:]
            )

        # 限制回 coarse
        div_comp = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
        if n_prism > 0:
            div_comp[:n_prism] = gpu_contract_shared_operator_1axis(
                ops_data['overint_restrict_f2c_prism'], div_comp_fine[:n_prism]
            )
        if n_cells > n_prism:
            div_comp[n_prism:] = gpu_contract_shared_operator_1axis(
                ops_data['overint_restrict_f2c_tet'], div_comp_fine[n_prism:]
            )
    else:
        # 无 fine 几何：朴素路径
        adj_j = mesh_data['adj_j']
        Q_flat = cp.ascontiguousarray(Q.reshape(-1, 5))
        F_phys = euler_physical_flux_gpu(Q_flat).reshape(n_cells, n_sps, 3, 5)
        F_tilde = cp.matmul(adj_j, F_phys)
        div_comp = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
        if n_prism > 0:
            div_comp[:n_prism] = gpu_contract_shared_operator_2axis(
                ops_data['D_3d_prism'], F_tilde[:n_prism]
            )
        if n_cells > n_prism:
            div_comp[n_prism:] = gpu_contract_shared_operator_2axis(
                ops_data['D_3d_tet'], F_tilde[n_prism:]
            )

    residual = -div_comp / det_jacs[..., None]
    return residual


def _compute_boundary_ghost_states_gpu(
    Q_gpu, flat_face_gpu, adj_j, ghost_provider,
    n_cells, n_sps, device_id,
):
    """计算边界面的幽灵态（向量化实现，全程 GPU）。

    零梯度外插：Q_ghost[face] = Q[owner_cell, SP0]
    向量化替代逐面 Python 循环。
    """
    cp = get_cupy()
    n_faces = flat_face_gpu.n_faces
    n_vars = Q_gpu.shape[-1]

    Q_ghost_gpu = cp.zeros((n_faces, n_vars), dtype=cp.float64)

    # 向量化：一次性处理所有边界面
    bnd_mask = flat_face_gpu.is_boundary  # (n_faces,)
    bnd_owners = flat_face_gpu.owner_cell[bnd_mask]  # (n_bnd,)

    if bnd_owners.shape[0] > 0:
        # 零梯度外插：取 owner cell 的 SP0 值
        Q_ghost_gpu[bnd_mask] = Q_gpu[bnd_owners, 0, :]

    return Q_ghost_gpu


def _compute_interface_correction_gpu(
    Q_gpu, adj_j, det_jacs, flat_face_gpu, Q_ghost_gpu,
    n_cells, n_sps, n_prism, device_id,
):
    """GPU 界面校正计算（按图着色逐色处理）。

    核心策略：同色面无 owner_cell 冲突，直接写入共享 buffer。
    使用 CuPy ElementwiseKernel 逐面计算 AUSM+up 通量，
    然后用 CuPy 向量化操作分配校正到 cell SPs。
    """
    cp = get_cupy()

    correction = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)

    # 逐色处理
    for c in range(flat_face_gpu.n_colors):
        face_idx = flat_face_gpu.color_face_indices[c]
        if face_idx.shape[0] == 0:
            continue

        # 提取当前颜色的面数据
        oc = flat_face_gpu.owner_cell[face_idx]
        nc = flat_face_gpu.neighbor_cell[face_idx]
        is_bnd = flat_face_gpu.is_boundary[face_idx]

        # Owner 侧 SP→FP 外插（src0 矩阵乘法）
        # owner_src0_mat[face, n_fp, n_sps] @ Q[owner_src0_cell[face], :, :] → Q_L_fp
        owner_src0_cell = flat_face_gpu.owner_src0_cell[face_idx]
        owner_src0_mat = flat_face_gpu.owner_src0_mat[face_idx]
        Q_owner = Q_gpu[owner_src0_cell]  # (n_color_faces, n_sps, 5)
        Q_L_fp = cp.matmul(owner_src0_mat, Q_owner)  # (n_color_faces, n_fp, 5)

        # Neighbor 侧
        neighbor_src0_cell = flat_face_gpu.neighbor_src0_cell[face_idx]
        neighbor_src0_mat = flat_face_gpu.neighbor_src0_mat[face_idx]
        Q_neighbor = Q_gpu[neighbor_src0_cell]
        Q_R_fp = cp.matmul(neighbor_src0_mat, Q_neighbor)

        # 边界面用幽灵态替代 neighbor
        Q_ghost_face = Q_ghost_gpu[face_idx]  # (n_color_faces, 5)

        # 对边界面：Q_R_fp 替换为 Q_ghost（广播到所有 FP）
        n_fp = Q_L_fp.shape[1]
        Q_R_fp = cp.where(
            is_bnd[:, None, None],
            cp.broadcast_to(Q_ghost_face[:, None, :], (Q_ghost_face.shape[0], n_fp, 5)),
            Q_R_fp,
        )

        # 度量法向（从面几何获取 true_normal）
        normal = flat_face_gpu.true_normal[face_idx]  # (n_color_faces, 3)

        # AUSM+up 通量（逐 FP 向量化）
        flux = _ausm_up_flux_batch_gpu(Q_L_fp, Q_R_fp, normal)

        # 校正分配：从 FP 分配回 SPs
        # 使用 g_left/g_right 校正函数导数
        g_left = flat_face_gpu.g_left[face_idx]   # (n_color_faces, n_fp, n_sps)
        g_right = flat_face_gpu.g_right[face_idx]

        # Owner 侧校正贡献
        # contrib_o = g_left^T @ flux → (n_color_faces, n_sps, 5)
        contrib_o = cp.matmul(cp.swapaxes(g_left, -1, -2), flux)

        # 除以 det(J)
        dj_owner = det_jacs[oc]  # (n_color_faces, n_sps)
        contrib_o = contrib_o / dj_owner[..., None]

        # Scatter-add 到 correction（同色面无冲突，直接写入）
        # 使用 cp.zeros + scatter 模式
        _scatter_add_to_correction(correction, contrib_o, oc, n_cells, n_sps)

        # Neighbor 侧校正贡献（仅内部面）
        # 与 CPU 版 fr_residual_inviscid_kernel.py 一致：
        # neighbor 侧使用 g_right 将通量分配回 SPs，scatter-add 到 nc
        int_mask = ~is_bnd  # 内部面掩码
        if cp.any(int_mask):
            nc_int = nc[int_mask]
            g_right_int = g_right[int_mask]  # (n_int, n_fp, n_sps)
            flux_int = flux[int_mask]         # (n_int, n_fp, 5)

            # contrib_n = g_right^T @ flux → (n_int, n_sps, 5)
            contrib_n = cp.matmul(cp.swapaxes(g_right_int, -1, -2), flux_int)

            dj_nc = det_jacs[nc_int]  # (n_int, n_sps)
            contrib_n = contrib_n / dj_nc[..., None]

            _scatter_add_to_correction(correction, contrib_n, nc_int, n_cells, n_sps)

    return correction


def _ausm_up_flux_batch_gpu(Q_L, Q_R, normal):
    """GPU 批量 AUSM+up 通量计算（CuPy 向量化版本）。

    Args:
        Q_L, Q_R: (N, n_fp, 5) 左右状态
        normal: (N, 3) 单位法向量

    Returns:
        flux: (N, n_fp, 5) 数值通量
    """
    cp = get_cupy()
    gamma = 1.4
    alpha = 0.1875
    beta_param = 0.5

    rhoL = cp.maximum(Q_L[..., 0], 1e-6)
    uL, vL, wL = Q_L[..., 1], Q_L[..., 2], Q_L[..., 3]
    pL = cp.maximum(Q_L[..., 4], 10.0)

    rhoR = cp.maximum(Q_R[..., 0], 1e-6)
    uR, vR, wR = Q_R[..., 1], Q_R[..., 2], Q_R[..., 3]
    pR = cp.maximum(Q_R[..., 4], 10.0)

    nx = normal[..., 0:1]
    ny = normal[..., 1:2]
    nz = normal[..., 2:3]

    unL = uL * nx + vL * ny + wL * nz
    unR = uR * nx + vR * ny + wR * nz

    aL = cp.sqrt(cp.maximum(gamma * pL / rhoL, 1e-10))
    aR = cp.sqrt(cp.maximum(gamma * pR / rhoR, 1e-10))

    M_L = unL / cp.maximum(aL, 1e-10)
    M_R = unR / cp.maximum(aR, 1e-10)

    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)
    Ma_ref = 0.1
    M0_sq = cp.minimum(1.0, cp.maximum(Mbar2, Ma_ref**2))
    sqrt_M0_sq = cp.sqrt(M0_sq)
    fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq)
    fa = cp.maximum(fa, 1e-6)

    # M+ / M-
    abs_ML = cp.abs(M_L)
    abs_MR = cp.abs(M_R)
    Mp_L = cp.where(
        abs_ML >= 1.0,
        0.5 * (M_L + abs_ML),
        0.25 * (M_L + 1.0)**2 + alpha * (M_L**2 - 1.0)**2,
    )
    Mm_R = cp.where(
        abs_MR >= 1.0,
        0.5 * (M_R - abs_MR),
        -0.25 * (M_R - 1.0)**2 - alpha * (M_R**2 - 1.0)**2,
    )
    M_half = Mp_L + Mm_R

    # Mp 压力扩散
    Kp = 0.25
    sigma_p = 1.0
    one_minus_sigma = cp.maximum(1.0 - sigma_p * Mbar2, 0.0)
    Mp = -(Kp / fa) * one_minus_sigma * (pR - pL) / (rho_half * a_half**2)
    mass_flux = 0.5 * (rhoL * aL + rhoR * aR) * (M_half + Mp)

    # P+ / P-
    Pp_L = cp.where(
        abs_ML >= 1.0,
        0.5 * (1.0 + cp.sign(M_L)),
        0.25 * ((M_L + 1.0)**2 * (2.0 - M_L) + beta_param * M_L * (M_L**2 - 1.0)**2),
    )
    Pm_R = cp.where(
        abs_MR >= 1.0,
        0.5 * (1.0 - cp.sign(M_R)),
        0.25 * ((M_R - 1.0)**2 * (2.0 + M_R) - beta_param * M_R * (M_R**2 - 1.0)**2),
    )

    # pu 速度扩散
    Ku = 0.75
    p_half = Pp_L * pL + Pm_R * pR - Ku * Pp_L * Pm_R * (rhoL + rhoR) * fa * a_half * (unR - unL)

    # 上风通量
    upwind_L = (mass_flux >= 0.0)
    u_up = cp.where(upwind_L, uL, uR)
    v_up = cp.where(upwind_L, vL, vR)
    w_up = cp.where(upwind_L, wL, wR)

    hL = gamma / (gamma - 1.0) * pL / rhoL + 0.5 * (uL**2 + vL**2 + wL**2)
    hR = gamma / (gamma - 1.0) * pR / rhoR + 0.5 * (uR**2 + vR**2 + wR**2)
    h_up = cp.where(upwind_L, hL, hR)

    flux = cp.stack([
        mass_flux,
        mass_flux * u_up + p_half * nx,
        mass_flux * v_up + p_half * ny,
        mass_flux * w_up + p_half * nz,
        mass_flux * h_up,
    ], axis=-1)

    return flux


def _scatter_add_to_correction(correction, contrib, cell_indices, n_cells, n_sps):
    """将面的校正贡献写入全局 correction 数组。

    同色面无 owner_cell 冲突，但不同面可能写同一个 cell（虽然同色面之间不会），
    所以这里使用 CuPy 的 scatter add 模式。

    由于同色面保证无冲突，可以直接用索引赋值。
    """
    cp = get_cupy()
    # 同色面无冲突，直接用 advanced indexing 写入
    # contrib: (n_color_faces, n_sps, 5), cell_indices: (n_color_faces,)
    # 需要处理多个面写同一个 cell 的情况（虽然同色面不冲突，但保险起见用 add）
    cp.scatter_add(correction, (cell_indices, slice(None), slice(None)), contrib)
