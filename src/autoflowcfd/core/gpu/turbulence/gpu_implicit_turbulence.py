"""AutoFlowCFD V2.0 - 单机 GPU 的隐式 k-omega 更新适配器。

隐式 k-omega 的算法（分离式 PTC-Newton、残差内的 omega 壁面强约束、零填充
槽位与试探场保护）只有一份：`core/fr_solver/turbulence/implicit.py`。本文件
只回答"GPU 上用哪一套求值件"——`GPUFRSolver` 上与 CPU `source.py` 三个
求值件一一对应的 `_prepare_turbulence_inputs_gpu` /
`_evaluate_turbulence_rates_gpu` / `_finalize_turbulence_update_gpu`，壁面
目标值读 `gpu_scalar_transport.omega_wall_cell_targets_gpu`（与显式路径的
壁面松弛同一个来源）。接口见 `implicit.py` 模块文档"后端"一节。
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

    def prepare(self) -> None:
        self.solver._update_production_ramp_gpu()
        self._inputs = self.solver._prepare_turbulence_inputs_gpu()

    def rates(self, apply_des: bool):
        dk, dw, tk, tw = self.solver._evaluate_turbulence_rates_gpu(*self._inputs, apply_des=apply_des)
        return (dk if tk is None else dk + tk), (dw if tw is None else dw + tw)

    def wall_targets(self):
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import omega_wall_cell_targets_gpu

        return omega_wall_cell_targets_gpu(self.xp, self.solver)

    def positivity(self) -> None:
        self.model.apply_positivity_limiter_gpu()

    def norm_weights(self):
        """残差范数的逐行守恒权重（与平均流 Newton 步同一份，见 `jfnk.ResidualNorm`）。"""
        from autoflowcfd.core.time_integration.positivity import get_positivity_limiter

        return get_positivity_limiter(self.solver, xp=self.xp).W

    def finalize(self, dtau) -> None:
        self.solver._finalize_turbulence_update_gpu(omega_wall_relaxation=False)

    def cell_colors(self):
        return gpu_cell_colors(self.xp, self.solver.flat_face_gpu, self.shape[0])

    def block_assembler(self):
        """本步的解析单元块装配器：线性算子部分在主机上装配（与 CPU 同一份，
        `core/turbulence/jacobian`），逐点量 `(S, Gamma)` 用 GPU 模型自己的求值件
        （`GpuTurbulencePointwise`），冻结的平均流输入与残差同一份。"""
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
        hit, target = self.wall_targets()
        ctx = TurbulenceLinearization(
            mesh=s.mesh, ops=s.ops, flat=flat, turb=m, Q=Q_h, grad_vel=_host(grad_vel), d_wall=_host(d_wall),
            mu=float(s.mu_molecular),
            conv_geom=precompute_scalar_convection_geometry(Q_h[..., 0], Q_h[..., 1:4], s.mesh, s.ops, flat),
            wall_zero_face=_host(s._wall_mask_k_gpu), omega_wall_face=_host(omega_wall),
            has_omega_wall=_host(has_wall), open_face=_host(s._open_mask_gpu),
            wall_cells=_host(hit), wall_targets=_host(target),
            pointwise=GpuTurbulencePointwise(cp, m, s.Q_gpu, grad_vel, d_wall, float(s.mu_molecular)))
        return TurbulenceBlockAssembler(ctx, self.shape[1])


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class GpuTurbulencePointwise:
    """GPU 模型上的逐点 `(S, Gamma)` 求值器（`core/turbulence/jacobian/pointwise.py` 的
    `evaluate` 接口）：输入输出是主机数组，求值在设备上用 GPU 模型的源项与扩散系数
    （`compute_source_terms_gpu`、`turbulence_diffusivities_gpu`，与 GPU 残差同一份）。"""

    __slots__ = ("cp", "turb", "Q", "grad_vel", "d_wall", "mu", "S_mag")

    _CACHED = ("nu_t", "_last_beta_blend", "_omega_realizability_min")

    def __init__(self, cp, turb, Q, grad_vel, d_wall, mu):
        self.cp, self.turb, self.Q, self.grad_vel, self.d_wall, self.mu = cp, turb, Q, grad_vel, d_wall, mu
        self.S_mag = turb.compute_strain_rate_magnitude_gpu(grad_vel)

    def __call__(self, k, omega, grad_k, grad_omega):
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import turbulence_diffusivities_gpu
        from autoflowcfd.core.turbulence.sst.bounds import clip_gradient_magnitude

        cp, turb = self.cp, self.turb
        saved = (turb.k_field, turb.omega_field)
        saved_cache = {a: getattr(turb, a) for a in self._CACHED if hasattr(turb, a)}
        turb.k_field, turb.omega_field = cp.asarray(k), cp.asarray(omega)
        try:
            gk = clip_gradient_magnitude(cp.asarray(grad_k), cp)
            gw = clip_gradient_magnitude(cp.asarray(grad_omega), cp)
            Sk, Sw = turb.compute_source_terms_gpu(self.Q, self.grad_vel, self.d_wall, self.mu, gk, gw)
            rho = self.Q[:, :, 0]
            nu = self.mu / cp.maximum(rho, 1e-10)
            Gk, Gw = turbulence_diffusivities_gpu(cp, turb, turb.k_field, turb.omega_field, gk, gw, rho,
                                                  rho * turb.nu_t, nu, self.mu, self.S_mag, self.d_wall)
        finally:
            turb.k_field, turb.omega_field = saved
            for a, v in saved_cache.items():
                setattr(turb, a, v)
        return _host(cp.stack([Sk, Sw], axis=-1)), _host(cp.stack([Gk, Gw], axis=-1))


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
