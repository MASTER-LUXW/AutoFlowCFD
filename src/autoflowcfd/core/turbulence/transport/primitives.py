"""AutoFlowCFD V2.0 - 单个被输运标量的离散原语（CPU；与湍流模型无关）。

湍流模型的速率求值（例如 SA-neg 的 `turbulence/sa/rates.py`）只写一份算法，后端通过本接口注入
自己的梯度、对流与扩散离散：

    gradient(phi)                 单元内物理梯度 `(n_cells, n_sps, 3)`
    convection(phi, freestream)   对流残差（`rho dphi/dt` 的对流部分）：无滑移壁 Dirichlet 0、
                                  开放边界按来流值取上风（与 k 同一组边界规则）
    diffusion(phi, gamma)         扩散残差 `div(gamma grad phi)`：无滑移壁 Dirichlet 0
    xp                            数组模块

CPU 单机与 CPU 分布式（紧凑空间适配器）共用本类；GPU 版见
`core/gpu/turbulence/gpu_scalar_transport/primitives.py`。
"""

import numpy as np

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient

from .convection import compute_scalar_convection_residual
from .diffusion import compute_scalar_diffusion_residual
from .omega_wall import _compute_open_boundary_face_mask, _compute_wall_dirichlet_face_mask
from .residual import prepare_convection_geometry


class CpuScalarTransport:
    """CPU 上一个求解器（或分布式紧凑空间适配器）的标量输运原语。

    质量通量几何 `conv_geom` 只依赖平均流：隐式路径在 Newton 步起点算好传入，显式路径传 None
    时在这里现算一次。
    """

    __slots__ = ("solver", "flat", "conv_geom", "wall_zero", "open_face", "rho", "vel")
    xp = np

    def __init__(self, solver_like, conv_geom=None, flat_face_override=None):
        self.solver = solver_like
        self.flat = flat_face_override
        flat = flat_face_override if flat_face_override is not None else get_flat_face_geometry(
            solver_like.mesh, solver_like.ops)
        self.conv_geom = conv_geom if conv_geom is not None else prepare_convection_geometry(
            solver_like, flat_face_override)
        self.wall_zero = _compute_wall_dirichlet_face_mask(solver_like)
        self.open_face = _compute_open_boundary_face_mask(solver_like, flat)
        Q = solver_like.state.Q
        self.rho, self.vel = Q[:, :, 0], Q[:, :, 1:4]

    def gradient(self, phi):
        return compute_physical_scalar_gradient(np.ascontiguousarray(phi), self.solver.mesh, self.solver.ops)

    def convection(self, phi, freestream_value: float):
        s = self.solver
        return compute_scalar_convection_residual(
            phi, self.rho, self.vel, s.mesh, s.ops, wall_dirichlet_zero_face=self.wall_zero,
            flat_face_override=self.flat, conv_geom=self.conv_geom,
            open_boundary_face=self.open_face, freestream_value=float(freestream_value))

    def diffusion(self, phi, gamma):
        s = self.solver
        return compute_scalar_diffusion_residual(phi, gamma, s.mesh, s.ops, wall_dirichlet_zero_face=self.wall_zero,
                                                 flat_face_override=self.flat)
