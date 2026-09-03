"""
AutoFlowCFD V2.0 - P>=1 高阶 FR 无粘残差 GPU 体积项 (从 gpu_inviscid.py 拆出，
控制单文件行数)。

体积项与界面项在物理上是两个独立阶段（体积项是纯 CuPy 向量化的张量收缩，
不涉及面几何/AUSM+up），拆分之后 gpu_inviscid.py 只保留界面校正相关逻辑。
"""

import numpy as np

from autoflowcfd.core.gpu.residual.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis,
    gpu_contract_shared_operator_2axis,
)
from autoflowcfd.core.gpu.residual.gpu_flux import (
    euler_physical_flux_gpu,
    conserved_to_primitive_gpu,
)


def prepare_mesh_data(cp, mesh, device_id):
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


def prepare_ops_data(cp, ops, device_id):
    """准备 FR 算子数据到 GPU。"""
    with cp.cuda.Device(device_id):
        data = {}
        # `D_native_tet_padded`（native 四面体/路径C 体积项微分算子）：
        # 2026-09-03 起恒非 None（坍缩坐标四面体基已删除，见
        # fr/operators.py 模块文档），`ops.D_3d_tet` 现在就是它的别名，
        # 两个键上传的是同一份数据，下游直接读 `D_3d_tet` 即可。
        for attr_name in ['D_3d_tet', 'D_3d_prism', 'D_native_tet_padded']:
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


def compute_volume_term_gpu(cp, U, mesh_data, ops_data, n_cells, n_sps, n_prism):
    """计算体积项（CuPy 向量化）。

    四面体坍缩坐标基已删除（2026-09-03，见 fr/operators.py 模块文档），
    不再接受 `tet_basis_mode` 参数——`ops_data['D_3d_tet']` 现在恒别名到
    `D_native_tet_padded`（`prepare_ops_data`/`array_manager.py` 上传
    的键与 CPU 侧 `FROperators.D_3d_tet` 同一个别名约定），两条分支
    直接用它即可，不需要按模式分派。
    """
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


def distribute_face_correction_to_sps(
    cp, correction_fp, axis, side, dist_fp_of_sp, dist_axis_coord_of_sp, g_left, g_right,
):
    """把面 FP 上的校正量分配回 SPs——真实 bug 修复（V2.0 专家组盲审第四轮，
    2026-08-28）：`gpu_inviscid.py`/`gpu_viscous.py` 的界面校正此前用
    `ff.g_left[idx_o]`/`ff.g_right[idx_o]`（用面索引去索引 g_left/g_right）
    再 `cp.matmul` 分配——但 g_left/g_right 真实形状是 `(n1d,)`（已用真实
    网格实测确认，P1 阶数下是 (2,)），不是每面一份的 (n_fp,n_sps) 矩阵，
    用面索引（可达 n_faces-1，真实网格上远超 n1d）去索引这个长度仅 n1d
    的数组，会在真实 CUDA 硬件上第一次执行时直接 IndexError——本机没有
    CuPy，这处从未被实际执行验证过，只靠代码走查核对公式，此前误以为
    g_left/g_right 是逐面矩阵。

    正确机制（与 CPU numba kernel `fr_residual/inviscid_kernel.py::
    _distribute_point`/`turbulence/transport_kernel.py::
    _distribute_point_scalar` 完全一致，是 gather 不是矩阵乘法）：SP s 的
    贡献 = g_prime[axis_coord_of_sp[axis, s]] * correction_fp[fp_of_sp[axis, s]]，
    其中 g_prime 是 g_left（side<=0）或 g_right（side>0）——`dist_fp_of_sp`/
    `dist_axis_coord_of_sp`（GPUFlatFaceGeometry 已上传的同名字段，形状
    (3, n_sps)）正是为这个 gather 预计算好的查表数组。

    Args:
        correction_fp: (n, n_fp) 或 (n, n_fp, V) CuPy 数组，n 是调用方
            选定的一批面（owner-primary 或 neighbor-primary 子集）
        axis, side: (n,) CuPy 数组，这批面各自的 owner/neighbor axis/side
        dist_fp_of_sp, dist_axis_coord_of_sp: (3, n_sps) CuPy 数组
        g_left, g_right: (n1d,) CuPy 数组

    Returns:
        (n, n_sps) 或 (n, n_sps, V) CuPy 数组
    """
    n = correction_fp.shape[0]
    fp_id = dist_fp_of_sp[axis]           # (n, n_sps)
    coord = dist_axis_coord_of_sp[axis]   # (n, n_sps)
    g_left_val = g_left[coord]            # (n, n_sps)
    g_right_val = g_right[coord]          # (n, n_sps)
    g_prime_val = cp.where(side[:, None] > 0, g_right_val, g_left_val)  # (n, n_sps)

    batch_idx = cp.arange(n)[:, None]
    fp_val = correction_fp[batch_idx, fp_id]  # (n, n_sps) 或 (n, n_sps, V)

    if fp_val.ndim == 2:
        return g_prime_val * fp_val
    return g_prime_val[..., None] * fp_val
