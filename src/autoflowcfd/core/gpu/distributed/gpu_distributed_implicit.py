"""AutoFlowCFD V2.0 - 多 GPU 分布式的隐式稳态（NK）湍流适配器与紧耦合 Newton 适配器。

紧耦合 Newton 的算法只有一份（`time_integration/implicit/coupled_step.py`）；本文件只回答
"多 GPU 上用哪一套求值件"——`gpu_distributed_init/turb_source.py` 的
prepare / evaluate / finalize / write-back 与平均流的 `_compute_total_residual_gpu`，归约用跨
rank 的 `MPIReductions(cupy)`，块 Jacobi 着色与 CPU 分布式同一个全局一致着色
（`core/mpi/distributed_implicit.py`，那里的模块文档说明了为什么必须全局一致）。

湍流未知量是本 rank local 单元的 `(k, w = ln omega)`（原生排列，模型上存物理 omega）；
每次求值按当前 local 平均流重建 compact 视图，湍流场经 2 变量 halo 交换写进视图，结果按
`inv_perm` 换回原生排列、切 local 段。
"""

from functools import partial

import numpy as np

from autoflowcfd.core.fr_solver.turbulence.init import advance_production_ramp
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.mpi.distributed_implicit import (
    distributed_block_jacobi_colors, distributed_coupling_graph, distributed_mean_flow_assembler,
)
from autoflowcfd.core.mpi.reductions import MPIReductions


class MultiGpuTurbulenceBackend:
    """`MultiGPUDistributedSolver` 的隐式湍流适配器，接口见
    `fr_solver/turbulence/implicit.py` 模块文档"后端"一节。

    额外产出 `mu_t_compact`：最后一次（`apply_des=True`）求值之后 compact 空间
    的动力涡粘，供本步平均流的粘性残差使用（与显式路径的返回值同一个量）。
    """

    __slots__ = ("solver", "model", "xp", "red", "shape", "cell_is_prism", "order",
                 "_ctx", "mu_t_compact")

    def __init__(self, solver, cell_is_prism, order: int):
        cp = get_cupy()
        self.solver = solver
        self.model = solver.turb_model_gpu
        self.xp = cp
        self.red = MPIReductions(cp)
        self.shape = tuple(self.model.k_field.shape)
        self.cell_is_prism = cell_is_prism
        self.order = int(order)
        self._ctx = None
        self.mu_t_compact = None

    def _to_local(self, a_compact):
        return self.solver._unpermute_from_compact(a_compact)[:self.shape[0]]

    def advance_ramp(self) -> None:
        advance_production_ramp(self.solver, self.model)

    def prepare_inputs(self) -> None:
        """按**当前** local 平均流重建 compact 视图上下文。"""
        s = self.solver
        self._ctx = s._prepare_turbulence_view_distributed()
        # 视图先与当前场同步：壁面目标值（blended 档读 k）在构造残差时就要用
        s._sync_turbulence_view(self._ctx)

    def trial_mu_t_compact(self):
        """最近一次 `rates` 之后的紧凑空间动力涡粘（试探求值用，不写回模型）。"""
        return self._ctx.rho * self._ctx.view.nu_t

    def rates(self, apply_des: bool):
        s, ctx = self.solver, self._ctx
        s._sync_turbulence_view(ctx)
        dk, dw, tk, tw = s._evaluate_turbulence_rates_distributed(ctx, apply_des=apply_des)
        rate_k = dk if tk is None else dk + tk
        rate_w = dw if tw is None else dw + tw
        if apply_des:
            # 最终场上的求值：涡粘与 DES 长度尺度写回真正的模型（k/omega 已由
            # Newton 步更新在模型上）
            s._write_back_turbulence_distributed(ctx, fields=False)
            self.mu_t_compact = ctx.rho * ctx.view.nu_t
        return self._to_local(rate_k), self._to_local(rate_w)

    def positivity(self) -> None:
        self.model.apply_positivity_limiter_gpu()

    def finalize(self, dtau) -> None:
        # 模态滤波在 compact 视图上做（按"棱柱在前"分块），再写回 local
        s, ctx = self.solver, self._ctx
        s._sync_turbulence_view(ctx)
        s._finalize_turbulence_update_distributed(ctx)
        self.model.k_field = self._to_local(ctx.view.k_field).copy()
        self.model.omega_field = self._to_local(ctx.view.omega_field).copy()

    def cell_colors(self):
        return distributed_block_jacobi_colors(self.solver)

    def coarse_context(self):
        from autoflowcfd.core.mpi.distributed_coarse import CompactCellValues, coarse_comm_context

        s = self.solver
        return coarse_comm_context(s.partition, CompactCellValues(s.gpu_halo, s._perm_gpu, self.xp))

    def block_assembler(self):
        """解析单元块装配器（最近一次 `prepare_inputs` 的视图）：在与残差同一个紧凑视图上装配（线性算子部分在主机，
        逐点量用 GPU 模型的求值件），按 `inv_perm` 取回本 rank 的行。"""
        from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import _host, gpu_turbulence_pointwise
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_omega_wall_target_gpu
        from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter
        from autoflowcfd.core.turbulence.jacobian import TurbulenceBlockAssembler, TurbulenceLinearization
        from autoflowcfd.core.turbulence.transport import precompute_scalar_convection_geometry

        s, ctx, cp = self.solver, self._ctx, self.xp
        view, tr = ctx.view, ctx.transport
        dist_fc = s.dist_flat_face
        mesh = DistributedMeshAdapter(s.partition, dist_fc, s.mesh, s.ops)
        flat = dist_fc.base_flat
        Q_h = _host(ctx.Q)
        omega_wall, has_wall = compute_omega_wall_target_gpu(
            cp, s.flat_face_gpu, tr._wall_mask_k_gpu, ctx.d_wall, ctx.Q, s.mu_molecular,
            getattr(view, "beta1", 0.075), omega_max=getattr(view, "omega_max", 1e6),
            turb_k_field=getattr(view, "k_field", None))
        lin = TurbulenceLinearization(
            mesh=mesh, ops=s.ops, flat=flat, turb=view, Q=Q_h, grad_vel=_host(ctx.grad_vel),
            d_wall=_host(ctx.d_wall), mu=float(s.mu_molecular),
            conv_geom=precompute_scalar_convection_geometry(Q_h[..., 0], Q_h[..., 1:4], mesh, s.ops, flat),
            wall_zero_face=_host(tr._wall_mask_k_gpu), omega_wall_face=_host(omega_wall),
            has_omega_wall=_host(has_wall), open_face=_host(tr._open_mask_gpu),
            pointwise=gpu_turbulence_pointwise(cp, view, ctx.Q, ctx.grad_vel, ctx.d_wall, float(s.mu_molecular)))
        return TurbulenceBlockAssembler(lin, self.shape[1], compact_state=_MultiGpuTurbulenceCompactState(s),
                                        row_compact=np.asarray(dist_fc.inv_perm)[:self.shape[0]])


class _MultiGpuTurbulenceCompactState:
    """local `(k, w)`（设备数组，未知量；交换与换序是线性的，对 w 与对 omega 同样适用）
    -> 紧凑空间（与 `_sync_turbulence_view` 同一次交换与换序）。"""

    __slots__ = ("solver",)

    def __init__(self, solver):
        self.solver = solver

    def __call__(self, kw_local):
        s = self.solver
        return s._permute_to_compact(s.gpu_halo.exchange(kw_local))


class MultiGpuCoupledBackend:
    """`MultiGPUDistributedSolver` 的耦合 Newton 适配器（`time_integration/implicit/coupled_step.py`
    模块文档"后端适配器"）。平均流残差与湍流视图都读 `U_gpu`（local、5 个变量），试探状态写进去；
    halo 交换是集体调用，各 rank 求值次数一致（全局归约决定）。`nu_av_compact`：本步冻结的人工
    扩散系数（compact 排列，未启用时 None）；`coarse_ctx`：两套块共用的全局粗校正通信上下文。"""

    __slots__ = ("solver", "turb", "red", "cell_is_prism", "order", "n_sps", "scales_mean", "positivity",
                 "coupling_graph", "_nu_av", "_coarse", "_n_local")

    def __init__(self, solver, cell_is_prism, order: int, nu_av_compact, coarse_ctx):
        from autoflowcfd.core.fr_solver.residual_diagnostics import _reference_scales

        self.solver = solver
        self.turb = MultiGpuTurbulenceBackend(solver, cell_is_prism, order)
        self.red = self.turb.red
        self.cell_is_prism = cell_is_prism
        self.order = int(order)
        self._n_local, self.n_sps = self.turb.shape
        self.scales_mean = _reference_scales(solver.freestream, 5)
        self.positivity = solver._get_positivity_limiter_gpu()
        self.coupling_graph = partial(distributed_coupling_graph, solver, coarse_ctx)
        self._nu_av = nu_av_compact
        self._coarse = coarse_ctx

    def state(self):
        from autoflowcfd.core.turbulence.sst.log_omega import log_omega

        cp, m = self.turb.xp, self.turb.model
        x = cp.empty((self._n_local * self.n_sps, 7), dtype=cp.float64)
        x[:, :5] = self.solver.U_gpu.reshape(-1, 5)
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

    def set_trial(self, x) -> None:
        from autoflowcfd.core.turbulence.sst.log_omega import omega_from_log

        cp, m = self.turb.xp, self.turb.model
        self.solver.U_gpu = cp.ascontiguousarray(x[:, :5]).reshape(self._n_local, self.n_sps, 5)
        m.k_field = cp.ascontiguousarray(x[:, 5]).reshape(self.turb.shape)
        m.omega_field = omega_from_log(cp.ascontiguousarray(x[:, 6]), m.omega_max, cp).reshape(self.turb.shape)

    def trial_mu_t(self):
        return self.turb.trial_mu_t_compact()

    def mean_residual(self, mu_t):
        """平均流 `Gamma R`（`gpu_distributed/stepping.py::_residual` 同一约定，`dU/dt = -R`）。"""
        s = self.solver
        res = -s._compute_total_residual_gpu(mu_t_field=mu_t, inviscid=True, viscous=True,
                                             nu_av_compact=self._nu_av)
        if s.low_mach_precond_enabled:
            from autoflowcfd.core.gpu.gpu_preconditioning import apply_low_mach_preconditioner_gpu

            res = apply_low_mach_preconditioner_gpu(res, s.U_gpu, s.freestream["mach_ref"], out=res)
        return res.reshape(self._n_local * self.n_sps, 5)

    def mean_assembler(self, mu_t):
        s = self.solver
        return distributed_mean_flow_assembler(
            s, s, order=self.order, mu_t_compact=mu_t, exchange=s.gpu_halo.exchange, perm=s._perm_gpu,
            n_sps=self.n_sps, nu_av_compact=self._nu_av)

    def cell_colors(self):
        return distributed_block_jacobi_colors(self.solver)

    def coarse_context(self):
        return self._coarse

    def install(self, x, dtau) -> None:
        """写入新状态，做湍流步后收尾，并在新状态上刷新涡粘（写回模型并留下 compact 一份
        `turb.mu_t_compact`）与 DES 长度尺度。"""
        self.set_trial(x)
        turb = self.turb
        turb.positivity()
        turb.finalize(dtau.reshape(turb.shape))
        turb.prepare_inputs()
        turb.rates(apply_des=True)
