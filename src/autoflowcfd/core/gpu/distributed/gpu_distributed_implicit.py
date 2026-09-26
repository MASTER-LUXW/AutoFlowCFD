"""AutoFlowCFD V2.0 - 多 GPU 分布式的隐式 k-omega 适配器。

隐式 k-omega 的算法只有一份（`fr_solver/turbulence/implicit.py`）；本文件只回答
"多 GPU 上用哪一套求值件"——`gpu_distributed_init/turb_source.py` 的
prepare / evaluate / finalize / write-back，归约用跨 rank 的
`MPIReductions(cupy)`，块 Jacobi 着色与 CPU 分布式同一个全局一致着色
（`core/mpi/distributed_implicit.py`，那里的模块文档说明了为什么必须全局一致）。

未知量是本 rank local 单元的 `(k, omega)`（原生排列，即 `turb_model_gpu`）；
每次求值经 2 变量 halo 交换写进 compact 视图，结果按 `inv_perm` 换回原生
排列、切 local 段。
"""

from autoflowcfd.core.fr_solver.turbulence.init import advance_production_ramp
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.mpi.distributed_implicit import distributed_block_jacobi_colors
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

    def prepare(self) -> None:
        s = self.solver
        advance_production_ramp(s, self.model)
        self._ctx = s._prepare_turbulence_view_distributed()
        # 视图先与当前场同步：壁面目标值（blended 档读 k）在构造残差时就要用
        s._sync_turbulence_view(self._ctx)

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

    def wall_targets(self):
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import omega_wall_cell_targets_gpu

        xp = self.xp
        hit, target = omega_wall_cell_targets_gpu(xp, self._ctx.transport)
        native = xp.asarray(self.solver.dist_flat_face.perm)[hit]
        keep = native < self.shape[0]
        return native[keep], target[keep]

    def positivity(self) -> None:
        self.model.apply_positivity_limiter_gpu()

    def norm_weights(self):
        """残差范数的逐行守恒权重（与平均流 Newton 步同一份，见 `jfnk.ResidualNorm`）。"""
        return self.solver._get_positivity_limiter_gpu().W

    def finalize(self, dtau) -> None:
        # 模态滤波在 compact 视图上做（按"棱柱在前"分块），再写回 local
        s, ctx = self.solver, self._ctx
        s._sync_turbulence_view(ctx)
        s._finalize_turbulence_update_distributed(ctx, omega_wall_relaxation=False)
        self.model.k_field = self._to_local(ctx.view.k_field).copy()
        self.model.omega_field = self._to_local(ctx.view.omega_field).copy()

    def cell_colors(self):
        return distributed_block_jacobi_colors(self.solver)

    def block_assembler(self):
        """本步的解析单元块装配器：在与残差同一个紧凑视图上装配（线性算子部分在主机，
        逐点量用 GPU 模型的求值件），按 `inv_perm` 取回本 rank 的行。"""
        from autoflowcfd.core.gpu.turbulence.gpu_implicit_turbulence import GpuTurbulencePointwise, _host
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
            compute_omega_wall_target_gpu, omega_wall_cell_targets_gpu,
        )
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
        hit, target = omega_wall_cell_targets_gpu(cp, tr)
        lin = TurbulenceLinearization(
            mesh=mesh, ops=s.ops, flat=flat, turb=view, Q=Q_h, grad_vel=_host(ctx.grad_vel),
            d_wall=_host(ctx.d_wall), mu=float(s.mu_molecular),
            conv_geom=precompute_scalar_convection_geometry(Q_h[..., 0], Q_h[..., 1:4], mesh, s.ops, flat),
            wall_zero_face=_host(tr._wall_mask_k_gpu), omega_wall_face=_host(omega_wall),
            has_omega_wall=_host(has_wall), open_face=_host(tr._open_mask_gpu),
            wall_cells=_host(hit), wall_targets=_host(target),
            pointwise=GpuTurbulencePointwise(cp, view, ctx.Q, ctx.grad_vel, ctx.d_wall, float(s.mu_molecular)))
        return TurbulenceBlockAssembler(lin, self.shape[1], compact_state=_MultiGpuTurbulenceCompactState(s),
                                        row_compact=np.asarray(dist_fc.inv_perm)[:self.shape[0]])


class _MultiGpuTurbulenceCompactState:
    """local `(k, omega)`（设备数组）-> 紧凑空间（与 `_sync_turbulence_view` 同一次交换与换序）。"""

    __slots__ = ("solver",)

    def __init__(self, solver):
        self.solver = solver

    def __call__(self, kw_local):
        s = self.solver
        return s._permute_to_compact(s.turb_halo_gpu.exchange(kw_local))
