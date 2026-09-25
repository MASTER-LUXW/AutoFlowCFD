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

    def finalize(self, dtau) -> None:
        # 模态滤波在 compact 视图上做（按"棱柱在前"分块），再写回 local
        s, ctx = self.solver, self._ctx
        s._sync_turbulence_view(ctx)
        s._finalize_turbulence_update_distributed(ctx, omega_wall_relaxation=False)
        self.model.k_field = self._to_local(ctx.view.k_field).copy()
        self.model.omega_field = self._to_local(ctx.view.omega_field).copy()

    def cell_colors(self):
        return distributed_block_jacobi_colors(self.solver)
