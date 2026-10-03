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
        self.shape = tuple(self.model.k_field.shape)
        self.cell_is_prism = np.arange(self.shape[0]) < int(solver.mesh.n_prism_cells)
        order = getattr(solver, "current_order", None)
        self.order = int(order if order is not None else solver.order)
        self._inputs = None

    def advance_ramp(self) -> None:
        self.solver._update_production_ramp_gpu()

    def prepare_inputs(self) -> None:
        self._inputs = self.solver._prepare_turbulence_inputs_gpu()

    def rates(self, apply_des: bool):
        dk, dw, tk, tw = self.solver._evaluate_turbulence_rates_gpu(*self._inputs, apply_des=apply_des)
        return (dk if tk is None else dk + tk), (dw if tw is None else dw + tw)

    def positivity(self) -> None:
        self.model.apply_positivity_limiter_gpu()

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
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_omega_wall_target_gpu
        from autoflowcfd.core.turbulence.jacobian import TurbulenceBlockAssembler, TurbulenceLinearization
        from autoflowcfd.core.turbulence.transport import precompute_scalar_convection_geometry

        s, cp, m = self.solver, self.xp, self.model
        grad_vel, d_wall = self._inputs
        Q_h = _host(s.Q_gpu)
        flat = get_flat_face_geometry(s.mesh, s.ops)
        omega_wall, has_wall = compute_omega_wall_target_gpu(
            cp, s.flat_face_gpu, s._wall_mask_k_gpu, d_wall, s.Q_gpu, s.mu_molecular,
            getattr(m, "beta1", 0.075), omega_max=getattr(m, "omega_max", 1e6),
            turb_k_field=getattr(m, "k_field", None))
        ctx = TurbulenceLinearization(
            mesh=s.mesh, ops=s.ops, flat=flat, turb=m, Q=Q_h, grad_vel=_host(grad_vel), d_wall=_host(d_wall),
            mu=float(s.mu_molecular),
            conv_geom=precompute_scalar_convection_geometry(Q_h[..., 0], Q_h[..., 1:4], s.mesh, s.ops, flat),
            wall_zero_face=_host(s._wall_mask_k_gpu), omega_wall_face=_host(omega_wall),
            has_omega_wall=_host(has_wall), open_face=_host(s._open_mask_gpu),
            pointwise=gpu_turbulence_pointwise(cp, m, s.Q_gpu, grad_vel, d_wall, float(s.mu_molecular)))
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
        from autoflowcfd.core.turbulence.sst.log_omega import log_omega

        cp, s, m = self.turb.xp, self.solver, self.turb.model
        x = cp.empty((self._n_rows, 7), dtype=cp.float64)
        x[:, :5] = s.U_gpu.reshape(self._n_rows, -1)[:, :5]
        x[:, 5] = m.k_field.ravel()
        x[:, 6] = log_omega(m.omega_field, cp).ravel()
        return x

    def snapshot(self):
        from autoflowcfd.core.turbulence.jacobian.pointwise import CACHED_MODEL_ATTRS

        m = self.turb.model
        return (self.solver.U_gpu, m.k_field, m.omega_field,
                {a: getattr(m, a) for a in CACHED_MODEL_ATTRS if hasattr(m, a)})

    def restore(self, snap) -> None:
        m = self.turb.model
        self.solver.U_gpu, m.k_field, m.omega_field = snap[0], snap[1], snap[2]
        for a, v in snap[3].items():
            setattr(m, a, v)
        self.solver._update_primitives_gpu()

    def set_trial(self, x) -> None:
        from autoflowcfd.core.turbulence.sst.log_omega import omega_from_log

        cp, s, m = self.turb.xp, self.solver, self.turb.model
        U = s.U_gpu.copy()
        U.reshape(self._n_rows, -1)[:, :5] = x[:, :5]
        s.U_gpu = U
        s._update_primitives_gpu()
        m.k_field = cp.ascontiguousarray(x[:, 5]).reshape(self.turb.shape)
        m.omega_field = omega_from_log(cp.ascontiguousarray(x[:, 6]), m.omega_max, cp).reshape(self.turb.shape)

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


def gpu_turbulence_pointwise(cp, turb, Q, grad_vel, d_wall, mu):
    """GPU 模型上的逐点 `(S, Gamma)` 求值器（`core/turbulence/jacobian/pointwise.py::
    TurbulencePointwise`，注入 `compute_source_terms_gpu` 与 `turbulence_diffusivities_gpu`，
    与 GPU 残差同一份）；单机与多 GPU（compact 视图）共用。"""
    from functools import partial

    from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import turbulence_diffusivities_gpu
    from autoflowcfd.core.turbulence.jacobian.pointwise import TurbulencePointwise

    return TurbulencePointwise(cp, turb, Q, grad_vel, d_wall, mu, turb.compute_strain_rate_magnitude_gpu(grad_vel),
                               turb.compute_source_terms_gpu, partial(turbulence_diffusivities_gpu, cp))


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
