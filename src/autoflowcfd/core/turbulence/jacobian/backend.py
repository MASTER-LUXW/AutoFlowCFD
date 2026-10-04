"""AutoFlowCFD V2.0 - k-omega 解析单元块装配器的后端适配。

湍流 Newton 步的块缓存（`fr_solver/turbulence/implicit.py::_newton_state`）每步向
它要一次块：`assembler(u0_flat, r0_flat)`，`u0` 是 Newton 行（本 rank 单元）上的
`(k, omega)`。与平均流装配器（`fr_residual/jacobian/backend.py`）同一套约定：
单机恒等映射；分布式先把 `(k, omega)` halo 交换到"棱柱在前"的紧凑空间
（与湍流残差同一个视图），装配后按 `row_compact` 取回本 rank 的行。
"""

from typing import Callable, Optional

import numpy as np

from autoflowcfd.core.fr_residual.jacobian.backend import select_rows

from .assemble import TurbulenceLinearization, assemble_turbulence_blocks


class TurbulenceBlockAssembler:
    """`(u0_flat, r0_flat) -> (blocks_prism, blocks_tet[, coupling[, cross_rank_coupling]])`（主机 float32，
    `n_var` 为模型的输运标量个数；跨 rank 耦合块只在分布式给出，见 `select_rows`）。"""

    __slots__ = ("ctx", "n_sps", "compact_state", "row_compact", "want_coupling")

    def __init__(self, ctx: TurbulenceLinearization, n_sps: int, *, compact_state: Optional[Callable] = None,
                 row_compact=None):
        self.ctx = ctx
        self.n_sps = int(n_sps)
        self.compact_state = compact_state
        self.row_compact = None if row_compact is None else np.asarray(row_compact, dtype=np.int64)
        self.want_coupling = False

    def __call__(self, u0_flat, r0_flat):
        u_dev = u0_flat.reshape(-1, self.n_sps, int(self.ctx.turb.n_transported))
        if self.row_compact is None:
            return assemble_turbulence_blocks(self.ctx, _host(u_dev), want_coupling=self.want_coupling)
        # halo 交换在状态自己的数组模块上做（多 GPU 的交换器收 cupy 数组），再搬回主机
        out = assemble_turbulence_blocks(self.ctx, _host(self.compact_state(u_dev)),
                                         want_coupling=self.want_coupling)
        return select_rows(out, self.row_compact, int(self.ctx.mesh.n_prism_cells), self.want_coupling)


def _host(a):
    return np.ascontiguousarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)


def sa_linearization_parts(xp, model, Q, grad_vel, d_wall, mu, grad_rho, wall_zero_face):
    """SA-neg 的线性化部件 `(dirichlet_faces, dirichlet_values, pointwise, strong_rows)`：全部后端共用
    （CPU 单机/分布式经 `turbulence_linearization`，GPU 经其块装配器）。输入是残差用的同一组冻结量，
    数组在模型的数组模块 `xp` 上；`wall_zero_face` 为主机布尔面掩码。"""
    from autoflowcfd.core.turbulence.sa.linearization import SAPointwise, sa_dirichlet_spec, sa_strong_rows

    faces, values = sa_dirichlet_spec(wall_zero_face)
    return faces, values, SAPointwise(xp, model, Q, grad_vel, d_wall, mu, grad_rho), sa_strong_rows(d_wall)


def turbulence_linearization(solver_like, turb, inputs, conv_geom, flat):
    """由残差用的同一组冻结输入构造 `TurbulenceLinearization`（单机与分布式共用）。

    `solver_like` 是湍流残差实际求值的那个对象（单机为求解器，分布式为紧凑空间适配器），
    壁面/开放边界掩码与 omega 壁面目标值取自与残差同一组函数。
    """
    from autoflowcfd.core.turbulence.sa import SAModel
    from autoflowcfd.core.turbulence.transport.omega_wall import (
        _compute_omega_wall_target, _compute_open_boundary_face_mask, _compute_wall_dirichlet_face_mask,
    )

    Q, grad_vel, d_wall, mu = inputs
    wall_zero = _compute_wall_dirichlet_face_mask(solver_like)
    pointwise = None
    strong_rows = None
    if isinstance(turb, SAModel):
        from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient

        grad_rho = compute_physical_scalar_gradient(np.ascontiguousarray(Q[..., 0]), solver_like.mesh,
                                                    solver_like.ops)
        faces, values, pointwise, strong_rows = sa_linearization_parts(np, turb, Q, grad_vel, d_wall, mu,
                                                                       grad_rho, wall_zero)
    else:
        from autoflowcfd.core.turbulence.sst.unknowns import sst_dirichlet_spec

        omega_wall, has_wall = _compute_omega_wall_target(solver_like, wall_zero, mu, Q[..., 0],
                                                          flat_face_override=flat)
        faces, values = sst_dirichlet_spec(wall_zero, omega_wall, has_wall)
    return TurbulenceLinearization(
        mesh=solver_like.mesh, ops=solver_like.ops, flat=flat, turb=turb, Q=Q, grad_vel=grad_vel,
        d_wall=d_wall, mu=float(mu), conv_geom=conv_geom, dirichlet_faces=faces, dirichlet_values=values,
        open_face=_compute_open_boundary_face_mask(solver_like, flat), pointwise=pointwise,
        strong_rows=strong_rows)
