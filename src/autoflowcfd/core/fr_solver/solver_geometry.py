"""FRSolver 几何辅助方法混入类。

从 solver.py 拆出，控制单文件行数。提供度量通量面积标度、局部时间步长、
单元体积和网格尺度等几何辅助方法。
"""

import numpy as np


class _SolverGeometryMixin:
    """FRSolver 几何辅助方法混入。

    子类需要提供以下实例属性：
        mesh  - HighOrderMesh 对象
        state - 求解器状态（含 U 数组）
    """

    def _get_metric_flux_scale(self) -> np.ndarray:
        """逐 SP 的度量"通量面积"标度 sum_m ||adj(J)[:,m,:]||，只依赖网格
        几何（与流场状态无关），缓存后避免每个时间步重复计算——供
        _compute_local_time_step 的几何/度量 CFL 限制使用，见该方法文档。
        """
        cached = getattr(self, "_metric_flux_scale_cache", None)
        if cached is not None and cached.shape[0] == self.state.U.shape[0]:
            return cached
        det_jacs = self.mesh.jacobians["det_jacs"].reshape(self.mesh.n_cells, self.mesh.n_sps_per_cell)
        inv_jacs = self.mesh.jacobians["inv_jacs"].reshape(self.mesh.n_cells, self.mesh.n_sps_per_cell, 3, 3)
        adj_j = det_jacs[..., None, None] * inv_jacs  # (n_cells,n_sps,3,3), adj_j[...,m,i]
        adj_row_norms = np.linalg.norm(adj_j, axis=-1)  # (n_cells,n_sps,3): 每个参考方向 m 的 |adj(J)[:,m,:]|
        metric_flux_scale = np.sum(adj_row_norms, axis=-1)  # (n_cells,n_sps)
        self._metric_flux_scale_cache = metric_flux_scale
        return metric_flux_scale

    def _compute_local_time_step(self) -> np.ndarray:
        """计算局部时间步长（基于CFL条件）。实现见
        cfl.py::compute_local_time_step（从本文件拆出，控制
        单文件行数），文档字符串也在那里。"""
        from .cfl import compute_local_time_step

        return compute_local_time_step(self)

    def _get_cell_volumes(self) -> np.ndarray:
        """
        获取单元体积（精确求积，见 HighOrderMesh.get_all_cell_volumes）。

        Returns:
            volumes: 单元体积，形状 (n_cells,)
        """
        return self.mesh.get_all_cell_volumes()

    def _get_grid_scale(self) -> np.ndarray:
        """
        获取网格尺度（用于LES/SGS模型）。

        Returns:
            delta: 网格尺度，形状 (n_cells, n_sps)
        """
        n_cells, n_sps = self.state.U.shape[:2]

        volumes = self.mesh.get_all_cell_volumes()
        delta = np.power(np.abs(volumes), 1.0 / 3.0)
        return np.tile(delta[:, np.newaxis], (1, n_sps))
