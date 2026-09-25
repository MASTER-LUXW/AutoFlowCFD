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

    def finalize(self, dtau) -> None:
        self.solver._finalize_turbulence_update_gpu(omega_wall_relaxation=False)

    def cell_colors(self):
        return gpu_cell_colors(self.xp, self.solver.flat_face_gpu, self.shape[0])


def gpu_cell_colors(cp, flat_face_gpu, n_cells: int) -> np.ndarray:
    """单机 GPU 的块 Jacobi 着色：设备端面相邻关系拷回主机做贪心着色（一次性）。"""
    from autoflowcfd.core.time_integration.implicit.block_jacobi import greedy_cell_coloring

    return greedy_cell_coloring(np.asarray(cp.asnumpy(flat_face_gpu.owner_cell)),
                                np.asarray(cp.asnumpy(flat_face_gpu.neighbor_cell)), int(n_cells))
