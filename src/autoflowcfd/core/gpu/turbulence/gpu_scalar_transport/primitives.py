"""AutoFlowCFD V2.0 - 单个被输运标量的离散原语（GPU；与湍流模型无关）。

CPU 版 `core/turbulence/transport/primitives.py::CpuScalarTransport` 的 GPU 对应，接口相同
（`gradient` / `convection` / `diffusion` / `xp`，两个离散可另给壁面条件：`wall_zero_face` 换掉默认的无滑移壁
掩码，`wall_value=(逐面值, 是否施加)` 改为非零 Dirichlet 值；SST 的 k/ln(omega) 与 SA-neg 都经本类调用）。`solver_like` 是单机 `GPUFRSolver` 或多 GPU 的
紧凑空间桩（`ctx.transport`），需要的属性与 `compute_turbulence_transport_residual_gpu` 相同：
`Q_gpu`、`mesh_data`、`ops_data`、`flat_face_gpu`、`mesh`、`_wall_mask_gpu`（无滑移壁面掩码）、
`_open_mask_gpu`（开放边界掩码，`compute_turbulence_face_masks_gpu`）。
"""

from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu
from autoflowcfd.core.turbulence.transport.primitives import wall_condition_kwargs

from .scalar_residuals import (
    compute_scalar_convection_residual_gpu,
    compute_scalar_diffusion_residual_gpu,
    scalar_convection_volume_divergence_gpu,
)


class GpuScalarTransport:
    """GPU 上一个求解器（或多 GPU 紧凑空间桩）的标量输运原语。质量通量体积散度只依赖平均流，
    构造时算一次（与 SST 输运同一个量）。"""

    __slots__ = ("solver", "xp", "rho", "vel", "n_cells", "n_prism", "n_sps", "mass_div", "c_ip", "wall_zero",
                 "open_face")

    def __init__(self, solver_like):
        cp = get_cupy()
        s = solver_like
        self.solver, self.xp = s, cp
        self.rho, self.vel = s.Q_gpu[:, :, 0], s.Q_gpu[:, :, 1:4]
        self.n_cells, self.n_sps = int(s.mesh.n_cells), int(s.mesh.n_sps_per_cell)
        self.n_prism = int(s.mesh_data.get('n_prism', s.mesh.n_prism_cells))
        self.mass_div = scalar_convection_volume_divergence_gpu(
            cp, cp.ones((self.n_cells, self.n_sps), dtype=cp.float64), self.rho, self.vel, s.mesh_data,
            s.ops_data, self.n_cells, self.n_prism, self.n_sps)
        self.c_ip = resolve_viscous_ip_constant(int(s.mesh.order))
        self.wall_zero = s._wall_mask_gpu          # 物理无滑移壁面
        self.open_face = s._open_mask_gpu          # 开放边界（来流条件）

    def gradient(self, phi):
        return compute_physical_scalar_gradient_gpu(self.xp.ascontiguousarray(phi), self.solver.mesh_data,
                                                    self.solver.ops_data)

    def convection(self, phi, freestream_value: float, wall_zero_face=None, wall_value=None):
        s = self.solver
        return compute_scalar_convection_residual_gpu(
            phi, self.rho, self.vel, s.mesh_data, s.ops_data, s.flat_face_gpu, self.n_cells, self.n_prism,
            self.n_sps, self.mass_div, **wall_condition_kwargs(self.wall_zero, wall_zero_face, wall_value),
            open_boundary_face=self.open_face, freestream_value=float(freestream_value))

    def diffusion(self, phi, gamma, wall_zero_face=None, wall_value=None, grad=None):
        """`grad`：`phi` 的物理梯度（`gradient(phi)` 的结果），调用方已有时传入、不重算（与 CPU 原语一致）。"""
        s = self.solver
        return compute_scalar_diffusion_residual_gpu(
            phi, gamma, s.mesh_data, s.ops_data, s.flat_face_gpu, self.n_cells, self.n_prism, self.n_sps,
            c_ip=self.c_ip, gamma_band=float(s.mu_molecular),
            **wall_condition_kwargs(self.wall_zero, wall_zero_face, wall_value),
            grad_phi=grad)
