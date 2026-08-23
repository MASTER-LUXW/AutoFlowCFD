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
        # 真实 bug（已修复，2026-08-23，用户追问 P1 发散根因促成排查，
        # CFL 假说被三次证伪后倒查出的真正问题）：这里此前只比较
        # `cached.shape[0]`（单元数 n_cells）——Order Continuation 跨
        # 阶数转换时 n_cells 恒定不变，只有 `shape[1]`（每单元 SPs 数
        # n_sps）会变，这个判据因此在任何一次阶数切换后都会误判缓存
        # 仍然有效，继续返回 P0 时代形状为 (n_cells,1) 的陈旧值。真正
        # 消费它的地方（cfl.py::compute_local_time_step 的 dt_geometric
        # 项）自己有一段"det_jacs.shape[1] != n_sps 时把两者一起 tile
        # 广播到当前 n_sps"的兼容逻辑——但那段逻辑判断的是*重新读取的*
        # det_jacs 是否需要广播（阶数切换后 mesh.jacobians 已经是新阶数
        # 形状，恒为 False），不会触发，metric_flux_scale 因此以
        # (n_cells,1) 的原始形状直接参与 `metric_flux_scale * wave_speed`
        # ——numpy 广播规则允许 size=1 维度对齐任意长度，不报错、不崩溃，
        # 静默用同一个 P0 阶段"整单元平均"的度量标度覆盖 P1 每个 SP
        # 本该独立的真实值。这正好是几何 CFL 项本来专门用来防的那类
        # 退化 SP 局部刚性失稳的探测机制被静默削弱——从阶数切换后的第一
        # 步起，直到进程结束（缓存永不失效）。改成同时比较 shape[1]（
        # 与 self.state.U.shape[1]，即当前真正的 n_sps，一致），跨阶数
        # 切换后缓存正确失效、重新计算。
        cached = getattr(self, "_metric_flux_scale_cache", None)
        if cached is not None and cached.shape == self.state.U.shape[:2]:
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
