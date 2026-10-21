"""AutoFlowCFD V2.0 - SST 族 k 与 `w = ln(omega)` 的输运残差编排（GPU）

CPU 版 `core/turbulence/transport/residual.py::compute_turbulence_transport_residual` 的对应：离散经
`primitives.py::GpuScalarTransport`（梯度、对流、扩散与壁面 / 来流条件），本文件只决定 k 与 w 各自的
扩散系数与边界条件。2026-10-21（I-186）以前这里另拼一套调用参数（质量通量散度、开放边界掩码、扩散带宽、
内罚常数各写一遍），非有限值静默归零。
"""

from typing import Tuple

import numpy as np

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.turbulence.nonfinite import replace_nonfinite
from autoflowcfd.core.turbulence.sst.bounds import k_wall_zero_mask
from autoflowcfd.core.turbulence.sst.log_omega import log_omega, log_omega_gradient_source
from autoflowcfd.core.turbulence.transport.residual import turbulence_diffusivities

from .omega_wall import compute_omega_wall_target_gpu
from .primitives import GpuScalarTransport


def compute_turbulence_transport_residual_gpu(
    solver, grad_k=None, grad_log_omega=None,
) -> Tuple:
    """GPU 版 k 与 `w = ln(omega)` 的完整输运残差（对流 + 扩散，w 另加 `Gamma_w |grad w|^2`，
    见 `core/turbulence/sst/log_omega.py`），与 CPU 版逐项对应。

    Args:
        solver: `GPUFRSolver` 或多 GPU 的紧凑空间桩（`ctx.transport`）；`turb_model_gpu` 为 SST 族模型
            （SST/DDES/IDDES 都是 `GPUTurbulenceSST`），另需 `GpuScalarTransport` 要求的属性与
            `wall_distance_gpu`
        grad_k, grad_log_omega: 可选，调用方为源项算好的 k、ln(omega) 物理梯度，复用（物理梯度是热点）。
            有效扩散系数读模型上源项刷新的 `nu_t` 与 `F1`，调用方须先在同一组 `(k, omega)` 上求源项。

    Returns:
        (dk_dt_transport, dw_dt_transport)，各自 (n_cells, n_sps)
    """
    from autoflowcfd.core.turbulence.sst import SSTModelFR

    cp = get_cupy()
    turb = solver.turb_model_gpu
    if not isinstance(turb, SSTModelFR):
        raise TypeError(f"k/ln(omega) 输运残差只对 SST 族模型定义，收到 {type(turb).__name__}")
    mu = solver.mu_molecular
    transport = GpuScalarTransport(solver)
    rho = transport.rho

    w_log = log_omega(turb.omega_field, cp)
    if grad_k is None:
        grad_k = transport.gradient(turb.k_field)
    if grad_log_omega is None:
        grad_log_omega = transport.gradient(w_log)

    gamma_k, gamma_w = turbulence_diffusivities(turb, rho * turb.nu_t, mu)

    # k 的壁面：resolved 在物理无滑移壁面为 0，auto 为零通量（与 CPU 同一个函数 `sst/bounds.py::k_wall_zero_mask`）
    wall_mask_k = k_wall_zero_mask(transport.wall_zero, turb.wall_treatment, cp)
    conv_k = transport.convection(turb.k_field, turb.k_inf, wall_zero_face=wall_mask_k)
    diff_k = transport.diffusion(turb.k_field, gamma_k, wall_zero_face=wall_mask_k, grad=grad_k)
    dk_dt_transport = (conv_k + diff_k) / cp.maximum(rho, 1e-10)

    # omega 的壁面：解析壁面值作 Dirichlet 目标，w = ln(omega) 取它的对数（没有目标的面该值不被读取）
    omega_wall_value_face, has_omega_wall = compute_omega_wall_target_gpu(
        cp, solver.flat_face_gpu, transport.wall_zero, solver.wall_distance_gpu, solver.Q_gpu, mu, turb.beta1,
        omega_max=turb.omega_max, turb_k_field=turb.k_field, wall_treatment=turb.wall_treatment,
    )
    wall_w = (log_omega(omega_wall_value_face, cp), has_omega_wall)
    conv_w = transport.convection(w_log, float(np.log(turb.omega_inf)), wall_value=wall_w)
    diff_w = transport.diffusion(w_log, gamma_w, wall_value=wall_w, grad=grad_log_omega)
    dw_dt_transport = ((conv_w + diff_w + log_omega_gradient_source(gamma_w, grad_log_omega, cp))
                       / cp.maximum(rho, 1e-10))

    # 非有限值替换并计数告警（`turbulence/nonfinite.py`，与 CPU 同一份）
    dk_dt_transport = replace_nonfinite(turb, dk_dt_transport, "SST k 输运", cp)
    dw_dt_transport = replace_nonfinite(turb, dw_dt_transport, "SST ln(omega) 输运", cp)
    return dk_dt_transport, dw_dt_transport
