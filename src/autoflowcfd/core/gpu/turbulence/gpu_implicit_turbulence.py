"""AutoFlowCFD V2.0 - 单机 GPU 的隐式稳态（NK）湍流适配器与紧耦合 Newton 适配器。

紧耦合 Newton 的算法只有一份：`core/time_integration/implicit/coupled_step.py`。本文件
只回答"GPU 上用哪一套求值件"——`GPUFRSolver` 上与 CPU `source.py` 三个求值件一一对应的
`_prepare_turbulence_inputs_gpu` / `_evaluate_turbulence_rates_gpu` /
`_finalize_turbulence_update_gpu`，以及平均流的 GPU 残差。湍流适配器接口见
`core/fr_solver/turbulence/implicit.py` 模块文档"后端"一节。
"""

import numpy as np

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions


class GpuTurbulenceBackend:
    """`GPUFRSolver`（单机）的隐式湍流适配器。"""

    __slots__ = ("solver", "model", "xp", "red", "shape", "cell_is_prism", "order", "_inputs")

    def __init__(self, solver):
        cp = get_cupy()
        self.solver = solver
        self.model = solver.turb_model_gpu
        self.xp = cp
        self.red = LocalReductions(cp)
        self.shape = tuple(self.model.transported_fields()[0].shape)
        self.cell_is_prism = np.arange(self.shape[0]) < int(solver.mesh.n_prism_cells)
        order = getattr(solver, "current_order", None)
        self.order = int(order if order is not None else solver.order)
        self._inputs = None

    def advance_ramp(self) -> None:
        self.solver._update_production_ramp_gpu()

    def prepare_inputs(self) -> None:
        self._inputs = self.solver._prepare_turbulence_inputs_gpu()

    def rates(self, apply_des: bool):
        return self.solver._evaluate_turbulence_rates_gpu(*self._inputs, apply_des=apply_des).total()

    def positivity(self) -> None:
        self.model.apply_positivity_limiter()

    def finalize(self, dtau) -> None:
        self.solver._finalize_turbulence_update_gpu()

    def cell_colors(self):
        return gpu_cell_colors(self.xp, self.solver.flat_face_gpu, self.shape[0])

    def coarse_context(self):
        return None             # 单机：本地多层预处理的最粗层就是全局的

    def block_assembler(self):
        """解析单元块装配器（输入为最近一次 `prepare_inputs` 与当前状态）：线性算子部分在主机上装配（与 CPU 同一份，
        `core/turbulence/jacobian`），逐点量 `(S, Gamma)` 用 GPU 模型自己的求值件
        （`gpu_turbulence_pointwise`），冻结的平均流输入与残差同一份。"""
        from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
        from autoflowcfd.core.turbulence.jacobian import TurbulenceBlockAssembler, TurbulenceLinearization
        from autoflowcfd.core.turbulence.transport import precompute_scalar_convection_geometry

        s, cp, m = self.solver, self.xp, self.model
        grad_vel, d_wall = self._inputs
        Q_h = _host(s.Q_gpu)
        flat = get_flat_face_geometry(s.mesh, s.ops)
        faces, values, pointwise, strong_rows = gpu_linearization_parts(
            cp, m, s.Q_gpu, grad_vel, d_wall, float(s.mu_molecular), s.flat_face_gpu, s._wall_mask_k_gpu,
            s.mesh_data, s.ops_data)
        ctx = TurbulenceLinearization(
            mesh=s.mesh, ops=s.ops, flat=flat, turb=m, Q=Q_h, grad_vel=_host(grad_vel), d_wall=_host(d_wall),
            mu=float(s.mu_molecular),
            conv_geom=precompute_scalar_convection_geometry(Q_h[..., 0], Q_h[..., 1:4], s.mesh, s.ops, flat),
            dirichlet_faces=faces, dirichlet_values=values, open_face=_host(s._open_mask_gpu),
            pointwise=pointwise, strong_rows=strong_rows)
        return TurbulenceBlockAssembler(ctx, self.shape[1])


class GpuCoupledBackend:
    """单机 GPU 的耦合 Newton 适配器（`time_integration/implicit/coupled_step.py` 模块文档
    "后端适配器"）。平均流残差以试探状态为参数；湍流求值读 `Q_gpu`，所以试探状态同样写进
    `U_gpu` 并更新原始变量。`nu_av`：本步冻结的人工扩散系数（未启用时 None）。"""

    __slots__ = ("solver", "turb", "red", "cell_is_prism", "order", "n_sps", "scales_mean", "positivity",
                 "coupling_graph", "_nu_av", "_n_rows")

    def __init__(self, solver, nu_av=None):
        from functools import partial

        from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        self.solver = solver
        self.turb = GpuTurbulenceBackend(solver)
        self.red = self.turb.red
        self.cell_is_prism = self.turb.cell_is_prism
        self.order = self.turb.order
        n_cells, self.n_sps = self.turb.shape
        self._n_rows = n_cells * self.n_sps
        self.scales_mean = _reference_scales(solver.freestream, 5)
        self.positivity = get_positivity_limiter(solver, xp=self.turb.xp)
        self.coupling_graph = partial(gpu_coupling_graph, self.turb.xp, solver.flat_face_gpu, n_cells)
        self._nu_av = nu_av

    def state(self):
        cp, m = self.turb.xp, self.turb.model
        x = cp.empty((self._n_rows, 5 + m.n_transported), dtype=cp.float64)
        x[:, :5] = self.solver.U_gpu.reshape(self._n_rows, -1)[:, :5]
        x[:, 5:] = m.newton_unknowns(cp)
        return x

    def snapshot(self):
        return self.solver.U_gpu, self.turb.model.field_snapshot()

    def restore(self, snap) -> None:
        self.solver.U_gpu = snap[0]
        self.turb.model.field_restore(snap[1])
        self.solver._update_primitives_gpu()

    def set_trial(self, x) -> None:
        cp, s = self.turb.xp, self.solver
        U = s.U_gpu.copy()
        U.reshape(self._n_rows, -1)[:, :5] = x[:, :5]
        s.U_gpu = U
        s._update_primitives_gpu()
        self.turb.model.set_newton_unknowns(x[:, 5:], cp)

    def trial_mu_t(self):
        return self.solver._turbulent_mu_t_gpu()

    def mean_residual(self, mu_t):
        """平均流 `Gamma R`（`gpu_solver/step.py::mean_flow_residual` 同一约定，`dU/dt = -R`）。"""
        s = self.solver
        U = s.U_gpu
        res = s.compute_viscous_residual_gpu(U, mu_t_field=mu_t, nu_av=self._nu_av)
        res += s.compute_inviscid_residual_gpu(U)
        res *= -1
        if s.low_mach_precond_enabled:
            from autoflowcfd.core.gpu.gpu_preconditioning import apply_low_mach_preconditioner_gpu

            res = apply_low_mach_preconditioner_gpu(res, U, s.freestream["mach_ref"], out=res)
        return res.reshape(self._n_rows, -1)[:, :5]

    def mean_assembler(self, mu_t):
        """平均流解析单元块（主机上装配，与 CPU 同一份实现，块由缓存上传）。"""
        from autoflowcfd.core.fr_residual.jacobian.backend import MeanFlowBlockAssembler, unsupported_reason

        s = self.solver
        if unsupported_reason(order=self.order, wmles=getattr(s, "wmles_model", None) is not None):
            return None
        return MeanFlowBlockAssembler(
            mesh=s.mesh, ops=s.ops, ghost_provider=s.boundary_ghost_provider, mu=s.mu_molecular,
            mach_ref=s.freestream["mach_ref"], low_mach=s.low_mach_precond_enabled, mu_t=mu_t,
            nu_av=self._nu_av, n_sps=self.n_sps)

    def cell_colors(self):
        return self.turb.cell_colors()

    def coarse_context(self):
        return None

    def install(self, x, dtau) -> None:
        """写入新状态，做湍流步后收尾，并在新状态上刷新涡粘与 DES 长度尺度。"""
        self.set_trial(x)
        turb = self.turb
        turb.positivity()
        turb.finalize(dtau.reshape(turb.shape))
        turb.prepare_inputs()
        turb.rates(apply_des=True)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def gpu_linearization_parts(cp, model, Q, grad_vel, d_wall, mu, flat_face_gpu, wall_mask_gpu, mesh_data, ops_data):
    """GPU 模型的线性化部件 `(dirichlet_faces, dirichlet_values, pointwise, strong_rows)`（单机与多 GPU
    compact 视图共用）：SA-neg 走全部后端共用的 `sa_linearization_parts`；SST 族的壁面 omega 目标值与
    逐点求值器取 GPU 残差同一份。"""
    from autoflowcfd.core.turbulence.sa import SAModel

    wall_zero = _host(wall_mask_gpu)
    if isinstance(model, SAModel):
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu
        from autoflowcfd.core.turbulence.jacobian.backend import sa_linearization_parts

        grad_rho = compute_physical_scalar_gradient_gpu(cp.ascontiguousarray(Q[..., 0]), mesh_data, ops_data)
        return sa_linearization_parts(cp, model, Q, grad_vel, d_wall, mu, grad_rho, wall_zero)
    from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_omega_wall_target_gpu
    from autoflowcfd.core.turbulence.sst.unknowns import sst_dirichlet_spec

    omega_wall, has_wall = compute_omega_wall_target_gpu(
        cp, flat_face_gpu, wall_mask_gpu, d_wall, Q, mu, model.beta1, omega_max=model.omega_max,
        turb_k_field=model.k_field)
    faces, values = sst_dirichlet_spec(wall_zero, _host(omega_wall), _host(has_wall))
    return faces, values, gpu_turbulence_pointwise(cp, model, Q, grad_vel, d_wall, mu), None


def gpu_turbulence_pointwise(cp, turb, Q, grad_vel, d_wall, mu):
    """GPU 模型上的逐点 `(S, Gamma)` 求值器（`core/turbulence/jacobian/pointwise.py::
    TurbulencePointwise`，注入 `compute_source_terms_gpu`，与 GPU 残差同一份）；单机与
    多 GPU（compact 视图）共用。"""
    from autoflowcfd.core.turbulence.jacobian.pointwise import TurbulencePointwise

    return TurbulencePointwise(cp, turb, Q, grad_vel, d_wall, mu, turb.compute_source_terms_gpu)


def gpu_coupling_graph(cp, flat_face_gpu, n_cells: int):
    """单机 GPU 的 P0 差分耦合图：设备端面相邻关系拷回主机（一次性）。"""
    from autoflowcfd.core.time_integration.implicit.coloring import coupling_graph_from_faces

    return coupling_graph_from_faces(np.asarray(cp.asnumpy(flat_face_gpu.owner_cell)),
                                     np.asarray(cp.asnumpy(flat_face_gpu.neighbor_cell)), int(n_cells))


def gpu_cell_colors(cp, flat_face_gpu, n_cells: int) -> np.ndarray:
    """单机 GPU 的块 Jacobi 着色：设备端面相邻关系拷回主机做贪心着色（一次性）。"""
    from autoflowcfd.core.time_integration.implicit.coloring import greedy_cell_coloring

    return greedy_cell_coloring(np.asarray(cp.asnumpy(flat_face_gpu.owner_cell)),
                                np.asarray(cp.asnumpy(flat_face_gpu.neighbor_cell)), int(n_cells))
