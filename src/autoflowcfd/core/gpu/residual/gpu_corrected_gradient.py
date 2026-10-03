"""AutoFlowCFD V2.0 - 湍流模型用的速度梯度（GPU）。

CPU 版 `core/fr_operators/corrected_gradient.py::source_velocity_gradient` 的对应：P0 为
提升修正梯度（Green–Gauss），P>=1 为单元内多项式导数，阶数判断（`needs_lifting`）与
提升算法（`corrected_gradient`）都只有那一份。这里注入 GPU 的两个原语：按坐标系外插
两侧（`_extrapolate_scalar_pair_gpu`）与两侧跳变提升（`_lift_side_jumps_gpu`），两者与
CPU 的 numba 核逐项对应、已有逐位对照测试。边界幽灵态与 GPU 平均流残差同一个函数
（`_compute_boundary_ghost_states_gpu`）。
"""

from autoflowcfd.core.fr_operators.corrected_gradient import corrected_gradient, needs_lifting


def source_velocity_gradient_gpu(cp, Q, mesh_data, ops_data, flat_face_gpu, flat_face_cpu,
                                 ghost_provider, device_id, halo_refresh=None):
    """湍流模型用的速度梯度 `(n_cells, n_sps, 3, 3)`（CuPy），`[..., i, j] = du_i/dx_j`。

    Args:
        Q: `(n_cells, n_sps, >=5)` 原始变量（CuPy），前 5 列是 `(rho, u, v, w, p)`。
        flat_face_gpu / flat_face_cpu: 同一份面几何的设备端与主机端（幽灵态在主机上
            调用任意 Python 提供者，见 `_compute_boundary_ghost_states_gpu` 文档）。
        halo_refresh: 多 GPU 紧凑空间上刷新 halo 行（`mpi/compact_halo.py`，只在提升时需要）。
    """
    from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu

    vel = cp.ascontiguousarray(Q[..., 1:4])
    grad_broken = compute_physical_gradient_gpu(vel, mesh_data, ops_data)
    if not needs_lifting(vel.shape[1]):
        return grad_broken

    from autoflowcfd.core.gpu.residual.gpu_inviscid import _compute_boundary_ghost_states_gpu
    from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport.faces import (
        _extrapolate_scalar_pair_gpu, _lift_side_jumps_gpu,
    )

    n_cells, n_sps = Q.shape[:2]
    det_jacs = mesh_data["det_jacs"]
    Q_ghost = _compute_boundary_ghost_states_gpu(cp.ascontiguousarray(Q[..., :5]), flat_face_cpu,
                                                 ghost_provider, device_id)

    def extrapolate_pair(comp, frame):
        return _extrapolate_scalar_pair_gpu(cp, flat_face_gpu, comp, frame)

    def lift(jump_owner, jump_neighbor):
        return _lift_side_jumps_gpu(cp, flat_face_gpu, jump_owner, jump_neighbor, +1.0, det_jacs, n_cells, n_sps)

    grad = corrected_gradient(cp, vel, grad_broken, cp.ascontiguousarray(Q_ghost[..., 1:4]), flat_face_gpu,
                              extrapolate_pair, lift)
    return grad if halo_refresh is None else halo_refresh(grad)
