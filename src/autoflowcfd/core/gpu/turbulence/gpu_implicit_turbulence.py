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

    __slots__ = ("solver", "model", "xp", "red", "shape", "n_prism", "order", "_inputs")

    def __init__(self, solver):
        cp = get_cupy()
        self.solver = solver
        self.model = solver.turb_model_gpu
        self.xp = cp
        self.red = LocalReductions(cp)
        self.shape = tuple(self.model.k_field.shape)
        self.n_prism = int(solver.mesh.n_prism_cells)
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

    def finalize(self, dtau) -> None:
        self.solver._finalize_turbulence_update_gpu(omega_wall_relaxation=False)

    def face_adjacency(self):
        ff = self.solver.flat_face_gpu
        cp = self.xp
        return (np.asarray(cp.asnumpy(ff.owner_cell)), np.asarray(cp.asnumpy(ff.neighbor_cell)))
