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

    # 过积分改用共享 GPU helper 并**按段**处理（2026-09-17，与 CPU 端
    # 同一次改动）：native 四面体过积分的细网格轴不再填充到棱柱的
    # (oo+1)^3 宽度（P1 10 vs 27、P2 20 vs 64，填充槽位恒为零、对结果零
    # 贡献却要白算，其中 D_fine 的收缩是 O(n_fine^2)，P2 上 10.2 倍无效
    # FLOPs；CPU 侧实测 P1 加速 3.04x、P2 加速 4.63x，最大相对差
    # 1.4e-16 / 0.0）。两段 n_fine 不同，所以不能再共用一份
    # `(n_cells, n_fine, ...)` 的整场细点数组，必须逐段各自分配。
    from autoflowcfd.core.gpu.gpu_overintegration import (
        get_overintegration_segs_gpu,
    )

    _segs = get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
    if _segs is not None:
        div_comp = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
        for lo, hi, n_fine_seg, adj_seg, c2f, D_fine, f2c in _segs:
            if hi <= lo:
                continue
            Q_fine = gpu_contract_shared_operator_1axis(c2f, Q[lo:hi])
            F_phys_fine = euler_physical_flux_gpu(
                cp.ascontiguousarray(Q_fine.reshape(-1, 5))
            ).reshape(hi - lo, n_fine_seg, 3, 5)
            del Q_fine
            # `adj_seg` 已按段切好（整段处理，正好对应 [lo:hi]）
            F_tilde_fine = cp.matmul(adj_seg, F_phys_fine)
            del F_phys_fine
            div_fine = gpu_contract_shared_operator_2axis(D_fine, F_tilde_fine)
            del F_tilde_fine
            div_comp[lo:hi] = gpu_contract_shared_operator_1axis(f2c, div_fine)
            del div_fine
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
