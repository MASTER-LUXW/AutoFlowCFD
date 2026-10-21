"""
AutoFlowCFD V2.0 - P>=1 高阶 FR 无粘残差 GPU 体积项 (从 gpu_inviscid.py 拆出，
控制单文件行数)。

体积项与界面项在物理上是两个独立阶段（体积项是纯 CuPy 向量化的张量收缩，
不涉及面几何/AUSM+up），拆分之后 gpu_inviscid.py 只保留界面校正相关逻辑。
"""

import numpy as np

from autoflowcfd.core.gpu.residual.gpu_volume_contract import gpu_contract_shared_operator_1axis
from autoflowcfd.core.gpu.residual.gpu_flux import (
    euler_physical_flux_gpu,
    conserved_to_primitive_gpu,
)


def prepare_mesh_data(cp, mesh, device_id):
    """网格度量上传到 GPU（单 GPU 与多 GPU 同一个函数）：解点 det(J) 与 adj(J) = det(J) inv(J)（逆变通量、局部
    步长的度量尺度用它；物理梯度按 adj/det 现算，`gpu_gradients.py`），以及过积分细点度量。

    不另存 inv(J)：它与 adj(J) 只差逐点的 det 因子，多存一份是 9 个 double/解点（P2 30 万单元约 0.6 GB，
    2026-10-22 以前三份都存，I-113）。"""
    with cp.cuda.Device(device_id):
        n_cells = mesh.n_cells
        n_sps = mesh.n_sps_per_cell
        det_np = np.asarray(mesh.jacobians['det_jacs'], dtype=np.float64).reshape(n_cells, n_sps)
        inv_np = np.asarray(mesh.jacobians['inv_jacs'], dtype=np.float64).reshape(n_cells, n_sps, 3, 3)
        data = {
            'det_jacs': cp.asarray(np.ascontiguousarray(det_np)),
            # 与 CPU 体积项同一个乘积（`volume_contract.compute_adj_j`：det * inv）
            'adj_j': cp.asarray(np.ascontiguousarray(det_np[..., None, None] * inv_np)),
            'n_prism': mesh.n_prism_cells,
        }

        # 过积分细点度量（分段，见 `gpu_overintegration.upload_fine_metrics_gpu`）
        if mesh.jacobians_fine is not None:
            from autoflowcfd.core.gpu.gpu_overintegration import upload_fine_metrics_gpu

            data.update(upload_fine_metrics_gpu(cp, mesh.jacobians_fine))

        return data


def prepare_ops_data(cp, ops, device_id):
    """准备 FR 算子数据到 GPU。"""
    with cp.cuda.Device(device_id):
        data = {}
        # 体积微分算子（原生四面体 / 原生棱柱，零填充到解点宽度；两个都恒存在）
        for attr_name in ('D_native_tet_padded', 'D_native_prism_padded'):
            data[attr_name] = cp.asarray(np.ascontiguousarray(getattr(ops, attr_name), dtype=np.float64))

        from autoflowcfd.core.gpu.gpu_overintegration import upload_overintegration_ops_gpu
        upload_overintegration_ops_gpu(cp, ops, data)
        return data


def compute_volume_term_gpu(cp, U, mesh_data, ops_data, n_cells, n_sps, n_prism):
    """计算体积项（CuPy 向量化）。

    体积微分算子取 `ops_data['D_native_prism_padded']` / `ops_data['D_native_tet_padded']`（`prepare_ops_data` /
    `array_manager.py` 上传的键与 CPU 侧 `FROperators` 同名）。
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
        contract_lifted_divergence_gpu, get_overintegration_segs_gpu, lifted_divergence_gpu,
    )

    _segs = get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
    if _segs is None:
        # 与 CPU 端同一条约束（fr_residual/inviscid.py）：P>=1 必有过积分几何与算子
        raise RuntimeError(
            "P>=1 GPU 无粘残差需要过积分细点度量与 overint_* 算子，这里缺失：网格或算子是按"
            "不完整的阶数几何构造/上传的")
    div_comp = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)
    for (lo, hi, n_fine_seg, adj_seg, c2f, D_fine, f2c), (K_all, combo) in zip(
            _segs, lifted_divergence_gpu(mesh_data, ops_data)):
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
        # K = f2c·D_fine - Σ_面 lift·W·Tn：体积散度的 L2 投影减去修正项的本侧通量迹，
        # 界面只施加公共通量（与 CPU 同一个算子，fr/face_flux_trace.py）；K 按单元槽位组合取
        div_comp[lo:hi] = contract_lifted_divergence_gpu(cp, K_all, combo, F_tilde_fine)
        del F_tilde_fine

    residual = -div_comp / det_jacs[..., None]
    return residual
