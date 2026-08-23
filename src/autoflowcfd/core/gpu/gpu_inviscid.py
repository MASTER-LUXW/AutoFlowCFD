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
    mach_ref=0.1,
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
        mach_ref: AUSM+up Weiss-Smith 预处理参考马赫数（见
            kernels.py::compute_ausm_up_flux 文档）。默认值 0.1 只是
            保留旧硬编码值，真正的求解器路径（gpu_solver.py）必须显式
            传入 `solver.freestream["mach_ref"]`，不能依赖这个默认值。

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
        n_cells, n_sps, n_prism, device_id, mach_ref,
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


def _extrap_q_to_fp(cp, mat, src_cell, Q_gpu):
    """(nF,n_fp,n_sps) @ Q_gpu[src_cell] (nF,n_sps,5) -> (nF,n_fp,5)。"""
    return cp.matmul(mat, Q_gpu[src_cell])


def _add_q_src1_to_fp(cp, out, src1_idx, src1_cell, src1_mat, Q_gpu):
    """叠加稀疏第二来源（分裂面场景），Q 专用版本，见
    gpu_viscous.py::_add_src1_to_fp 同名通用版本的文档（这里内联一份
    Q-only 版本，避免 gpu_inviscid.py<->gpu_viscous.py 产生循环 import：
    gpu_viscous.py 已经反过来 import 本文件的 _prepare_mesh_data 等）。"""
    has1 = src1_idx >= 0
    if not bool(cp.any(has1)):
        return out
    sel = cp.where(has1)[0]
    idx1 = src1_idx[sel]
    c1 = src1_cell[idx1]
    m1 = src1_mat[idx1]
    out[sel] = out[sel] + cp.matmul(m1, Q_gpu[c1])
    return out


def _ausm_direction_with_fallback(cp, adjrow, side, true_normal_ref):
    """按 CPU 版 inviscid_kernel.py 的"自洽方向 + true_normal 对齐安全阀"
    逻辑构造 AUSM+up 用的法向：adjrow 精确方向与 true_normal_ref 夹角
    过大（alignment<0.5）时回退到 true_normal_ref 本身。

    Args:
        adjrow: (n, n_fp, 3) 未归一化 adj(J) 行
        side: (n,) ±1，owner_side 或 neighbor_side
        true_normal_ref: (n, n_fp, 3) 对齐基准（owner 侧用 true_normal，
            neighbor 侧用 -true_normal，见 CPU kernel"neighbor 视角外
            法向恒为 -true_normal"）

    Returns:
        (direction, adj_mag)：direction (n,n_fp,3)，adj_mag (n,n_fp)
    """
    a0 = adjrow[..., 0]
    a1 = adjrow[..., 1]
    a2 = adjrow[..., 2]
    adj_mag = cp.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    adj_mag_safe = cp.maximum(adj_mag, 1e-300)
    s = side[:, None]
    dirx = a0 / adj_mag_safe * s
    diry = a1 / adj_mag_safe * s
    dirz = a2 / adj_mag_safe * s
    alignment = dirx * true_normal_ref[..., 0] + diry * true_normal_ref[..., 1] + dirz * true_normal_ref[..., 2]
    use_fallback = alignment < 0.5
    dirx = cp.where(use_fallback, true_normal_ref[..., 0], dirx)
    diry = cp.where(use_fallback, true_normal_ref[..., 1], diry)
    dirz = cp.where(use_fallback, true_normal_ref[..., 2], dirz)
    direction = cp.stack([dirx, diry, dirz], axis=-1)
    return direction, adj_mag


def _compute_interface_correction_gpu(
    Q_gpu, adj_j, det_jacs, flat_face_gpu, Q_ghost_gpu,
    n_cells, n_sps, n_prism, device_id, mach_ref,
):
    """GPU 界面校正计算（按图着色逐色处理）。

    真实 bug 修复（2026-08-23，本次移植 GPU 粘性界面项时顺带发现并
    修复，用户明确要求本轮一并处理）：此前的实现有两个复合缺陷：

    1. **无 owner_is_primary/neighbor_is_primary 过滤**：`owner_src0`/
       `neighbor_src0` 字段的真实语义是"跨单元交叉引用数据，只在对应
       角色 primary 时才被写入"（见 face_flux_points_merge.py
       "kernel 只在 owner_primary[f] 为真...时写入 _nb_interp[f]，其余
       情形保持全零"的说明）——此前这里把它们当"这条记录自己的原生值"
       无条件读取。对约 5% 因棱柱四边形侧面拆分成 2 条记录的面：非
       owner_primary 记录的 `neighbor_src0` 全零/cell=-1，AUSM+up 会用
       一个全零假邻居态算出真实错误的通量（不只是重复计数），owner 侧
       的真实贡献还会被两条记录各计入一次。
    2. **owner/neighbor 两侧共享同一个 AUSM+up 通量分别分配**：
       inviscid_kernel.py 模块文档"关键正确性约束"明确记录过这个
       "优化"在真实网格上被验证证伪——自由流场残差从 9e-5 恶化到
       3.1e7，owner/neighbor 两侧必须用各自的度量法向、各自原生 FP
       位置插值出的状态独立各调用一次 AUSM+up，不能合并复用。
    3. **符号约定**：CPU 版每一侧的贡献是 `correction[...] += -contrib
       /dj`（带负号，配合无粘残差 `-div(F)` 的整体约定），此前这里
       scatter 的是不带负号的 `contrib_o`/`contrib_n`。

    修复：逐字仿照 inviscid_kernel.py 的两段独立结构（owner-primary
    过滤块 + neighbor-primary 过滤块），owner/neighbor 各自的"自身
    原生值"改用 `boundary_extrap` 查表外插（与 primary 状态无关、对
    任何记录都有效，不依赖 src0），"跨单元值"才用 src0/src1（只在
    对应角色 primary 时读取有意义）；两侧各自独立调用一次批量 AUSM+up；
    accumulate 时统一带负号。

    本机没有 CUDA/cupy，这处修复无法在本地实际执行验证（全仓库此前
    也没有任何 `compute_inviscid_residual_fr_gpu` 的 crosscheck 测试，
    见 test_gpu_p1_inviscid_interface_crosscheck.py 模块文档——这是
    这次连带发现的又一个"从未被验证过"的 GPU 生产路径），只能靠对照
    已经过详细验证的 CPU kernel 逐字核对；请在有真实 GPU 的环境上跑
    该新增测试文件做最终确认。
    """
    cp = get_cupy()
    ff = flat_face_gpu
    correction = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)

    for c in range(ff.n_colors):
        face_idx = ff.color_face_indices[c]
        if face_idx.shape[0] == 0:
            continue

        is_bnd = ff.is_boundary[face_idx]
        owner_primary = ff.owner_is_primary[face_idx]
        neighbor_primary = ff.neighbor_is_primary[face_idx]

        # ── owner-primary 贡献块 ──
        mask_o = owner_primary
        if bool(cp.any(mask_o)):
            idx_o = face_idx[mask_o]
            oc = ff.owner_cell[idx_o]
            oax = ff.owner_axis[idx_o]
            oside = ff.owner_side[idx_o]
            is_bnd_o = ff.is_boundary[idx_o]

            celltype_o = cp.where(oc < n_prism, 0, 1)
            oside_idx = cp.where(oside < 0, 0, 1)
            E_o = ff.boundary_extrap[celltype_o, oax, oside_idx]  # (nO,n_fp,n_sps)
            Q_o = cp.matmul(E_o, Q_gpu[oc])  # (nO,n_fp,5)

            Q_n = _extrap_q_to_fp(cp, ff.neighbor_src0_mat[idx_o], ff.neighbor_src0_cell[idx_o], Q_gpu)
            Q_n = _add_q_src1_to_fp(
                cp, Q_n, ff.neighbor_src1_idx[idx_o], ff.neighbor_src1_cell, ff.neighbor_src1_mat, Q_gpu,
            )

            n_fp = Q_o.shape[1]
            Q_ghost_face = Q_ghost_gpu[idx_o]  # (nO,5)
            Q_n = cp.where(
                is_bnd_o[:, None, None],
                cp.broadcast_to(Q_ghost_face[:, None, :], (Q_ghost_face.shape[0], n_fp, 5)),
                Q_n,
            )

            adjrow_o = ff.owner_adj_row_exact[idx_o]
            direction_o, adj_mag_o = _ausm_direction_with_fallback(
                cp, adjrow_o, oside, ff.true_normal[idx_o],
            )

            nO = Q_o.shape[0]
            flux_o = _ausm_up_flux_batch_gpu(
                Q_o.reshape(nO, n_fp, 5), Q_n.reshape(nO, n_fp, 5), direction_o, mach_ref,
            )
            F_tilde_common_o = flux_o * adj_mag_o[..., None] * oside[:, None, None]

            a0 = adjrow_o[..., 0]
            a1 = adjrow_o[..., 1]
            a2 = adjrow_o[..., 2]
            F_phys_o = euler_physical_flux_gpu(Q_o.reshape(nO * n_fp, 5)).reshape(nO, n_fp, 3, 5)
            F_tilde_own_o = (
                a0[..., None] * F_phys_o[..., 0, :]
                + a1[..., None] * F_phys_o[..., 1, :]
                + a2[..., None] * F_phys_o[..., 2, :]
            )

            jump_owner = F_tilde_common_o - F_tilde_own_o

            g_left_o = ff.g_left[idx_o]
            g_right_o = ff.g_right[idx_o]
            g_prime_owner = cp.where(oside[:, None, None] < 0, g_left_o, g_right_o)
            contrib_o = cp.matmul(cp.swapaxes(g_prime_owner, -1, -2), jump_owner)
            contrib_o = contrib_o / det_jacs[oc][..., None]
            _scatter_add_to_correction(correction, -contrib_o, oc, n_cells, n_sps)

        # ── neighbor-primary 贡献块（仅内部面）──
        mask_n = neighbor_primary & (~is_bnd)
        if bool(cp.any(mask_n)):
            idx_n = face_idx[mask_n]
            nc = ff.neighbor_cell[idx_n]
            nax = ff.neighbor_axis[idx_n]
            nside = ff.neighbor_side[idx_n]

            celltype_n = cp.where(nc < n_prism, 0, 1)
            nside_idx = cp.where(nside < 0, 0, 1)
            E_n = ff.boundary_extrap[celltype_n, nax, nside_idx]
            Q_n_native = cp.matmul(E_n, Q_gpu[nc])  # (nN,n_fp,5)

            Q_o_at_n = _extrap_q_to_fp(cp, ff.owner_src0_mat[idx_n], ff.owner_src0_cell[idx_n], Q_gpu)
            Q_o_at_n = _add_q_src1_to_fp(
                cp, Q_o_at_n, ff.owner_src1_idx[idx_n], ff.owner_src1_cell, ff.owner_src1_mat, Q_gpu,
            )

            n_fp = Q_n_native.shape[1]
            nN = Q_n_native.shape[0]

            # neighbor 视角外法向恒为 -true_normal（见 CPU kernel 同名注释）
            adjrow_n = ff.neighbor_adj_row_exact[idx_n]
            tn_neg = -ff.true_normal[idx_n]
            direction_n, adj_mag_n = _ausm_direction_with_fallback(
                cp, adjrow_n, nside, tn_neg,
            )

            flux_n = _ausm_up_flux_batch_gpu(
                Q_n_native.reshape(nN, n_fp, 5), Q_o_at_n.reshape(nN, n_fp, 5), direction_n, mach_ref,
            )
            F_tilde_common_n = flux_n * adj_mag_n[..., None] * nside[:, None, None]

            a0n = adjrow_n[..., 0]
            a1n = adjrow_n[..., 1]
            a2n = adjrow_n[..., 2]
            F_phys_n = euler_physical_flux_gpu(Q_n_native.reshape(nN * n_fp, 5)).reshape(nN, n_fp, 3, 5)
            F_tilde_own_n = (
                a0n[..., None] * F_phys_n[..., 0, :]
                + a1n[..., None] * F_phys_n[..., 1, :]
                + a2n[..., None] * F_phys_n[..., 2, :]
            )

            jump_neighbor = F_tilde_common_n - F_tilde_own_n

            g_left_n = ff.g_left[idx_n]
            g_right_n = ff.g_right[idx_n]
            g_prime_neighbor = cp.where(nside[:, None, None] < 0, g_left_n, g_right_n)
            contrib_n = cp.matmul(cp.swapaxes(g_prime_neighbor, -1, -2), jump_neighbor)
            contrib_n = contrib_n / det_jacs[nc][..., None]
            _scatter_add_to_correction(correction, -contrib_n, nc, n_cells, n_sps)

    return correction


def _ausm_up_flux_batch_gpu(Q_L, Q_R, normal, mach_ref):
    """GPU 批量 AUSM+up 通量计算（CuPy 向量化版本，含 Weiss-Smith 低马赫
    数预处理）。与 kernels.py::compute_ausm_up_flux 逐字对应，理由/推导
    见该函数文档，这里不重复；两处必须同步修改（该文件模块文档要求
    "逐字对应"）。

    Args:
        Q_L, Q_R: (N, n_fp, 5) 左右状态
        normal: (N, 3) 单位法向量
        mach_ref: 参考（自由来流）马赫数，见 kernels.py::
            compute_ausm_up_flux 文档

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

    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)

    M0_sq = cp.minimum(1.0, cp.maximum(Mbar2, mach_ref**2))
    sqrt_M0_sq = cp.sqrt(M0_sq)
    fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq)
    fa = cp.maximum(fa, 1e-6)

    # Weiss-Smith 预处理声速（与 kernels.py::compute_ausm_up_flux 的
    # _WEISS_SMITH_K=1.1 同一个安全裕度常数、同一套 beta2 公式）。
    beta2 = cp.minimum(1.0, cp.maximum(cp.maximum(Mbar2, 1.1 * mach_ref**2), 1e-10))
    sqrt_beta2 = cp.sqrt(beta2)
    aL_p = sqrt_beta2 * aL
    aR_p = sqrt_beta2 * aR
    a_half_p = sqrt_beta2 * a_half

    M_L = unL / cp.maximum(aL_p, 1e-10)
    M_R = unR / cp.maximum(aR_p, 1e-10)

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
    Mp = -(Kp / fa) * one_minus_sigma * (pR - pL) / (rho_half * a_half_p**2)
    mass_flux = 0.5 * (rhoL * aL_p + rhoR * aR_p) * (M_half + Mp)

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
    p_half = Pp_L * pL + Pm_R * pR - Ku * Pp_L * Pm_R * (rhoL + rhoR) * fa * a_half_p * (unR - unL)

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
