"""AutoFlowCFD V2.0 - 单个被输运标量的离散原语（CPU；与湍流模型无关）。

湍流模型的速率求值（例如 SA-neg 的 `turbulence/sa/rates.py`）只写一份算法，后端通过本接口注入
自己的梯度、对流与扩散离散：

    gradient(phi)                 单元内物理梯度 `(n_cells, n_sps, 3)`
    convection(phi, freestream)   对流残差（`rho dphi/dt` 的对流部分）：无滑移壁 Dirichlet 0、
                                  开放边界按来流值取上风（与 k 同一组边界规则）
    diffusion(phi, gamma)         扩散残差 `div(gamma grad phi)`：无滑移壁 Dirichlet 0
    xp                            数组模块

两个离散都可另给壁面条件（CPU 版）：`wall_zero_face` 换掉默认的无滑移壁掩码（SST 的 k 按近壁处理方式取，
`sst/bounds.py::k_wall_zero_mask`），`wall_value=(逐面值, 是否施加)` 改为非零 Dirichlet 值（SST 的 ln(omega) 壁面
目标）。SST 的 k/ln(omega) 输运（`residual.py`）与 SA-neg 都经本类调用离散——2026-10-19（I-62）以前 SST 一侧
另拼一套调用参数（网格、面几何、质量通量几何、开放边界掩码、扩散带宽各写一遍）。

CPU 单机与 CPU 分布式（紧凑空间适配器）共用本类；GPU 版见
`core/gpu/turbulence/gpu_scalar_transport/primitives.py`。
"""

import numpy as np

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient

from .convection import compute_scalar_convection_residual
from .diffusion import compute_scalar_diffusion_residual
from .faces import precompute_scalar_convection_geometry
from .omega_wall import _compute_open_boundary_face_mask, _compute_wall_dirichlet_face_mask


def prepare_convection_geometry(solver, flat_face_override=None):
    """标量对流的共享几何（只依赖平均流），见 `compute_turbulence_transport_residual`
    的 `conv_geom` 参数。"""
    Q = solver.state.Q
    flat = (flat_face_override if flat_face_override is not None
            else get_flat_face_geometry(solver.mesh, solver.ops))
    return precompute_scalar_convection_geometry(Q[:, :, 0], Q[:, :, 1:4], solver.mesh, solver.ops, flat)


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
        self.open_face = _compute_open_boundary_face_mask(solver_like)
        Q = solver_like.state.Q
        self.rho, self.vel = Q[:, :, 0], Q[:, :, 1:4]

    def gradient(self, phi):
        return compute_physical_scalar_gradient(np.ascontiguousarray(phi), self.solver.mesh, self.solver.ops)

    def _wall(self, wall_zero_face, wall_value) -> dict:
        """壁面条件的关键字参数：`wall_value` 给出时为非零 Dirichlet 值，否则为零值掩码（缺省取无滑移壁）。"""
        if wall_value is not None:
            value_face, has_face = wall_value
            return {"wall_dirichlet_value_face": value_face, "has_wall_dirichlet_value": has_face}
        return {"wall_dirichlet_zero_face": self.wall_zero if wall_zero_face is None else wall_zero_face}

    def convection(self, phi, freestream_value: float, wall_zero_face=None, wall_value=None):
        s = self.solver
        return compute_scalar_convection_residual(
            phi, self.rho, self.vel, s.mesh, s.ops, **self._wall(wall_zero_face, wall_value),
            flat_face_override=self.flat, conv_geom=self.conv_geom,
            open_boundary_face=self.open_face, freestream_value=float(freestream_value))

    def diffusion(self, phi, gamma, wall_zero_face=None, wall_value=None, grad=None):
        """`grad`：`phi` 的物理梯度（`gradient(phi)` 的结果），调用方已有时传入、不重算。"""
        s = self.solver
        return compute_scalar_diffusion_residual(phi, gamma, s.mesh, s.ops, **self._wall(wall_zero_face, wall_value),
                                                 flat_face_override=self.flat, gamma_band=float(s.mu_molecular),
                                                 grad_phi=grad)
