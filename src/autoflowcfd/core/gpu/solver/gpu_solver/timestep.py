"""AutoFlowCFD V2.0 - 单机 GPU 的局部时间步长与当前 CFL

从 `src/autoflowcfd/core/gpu/solver/gpu_solver.py` 的 `GPUFRSolver` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `GPUFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np
from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.gpu_time_integration import (
    compute_local_cfl_step_gpu,
)


class _GPUSolverTimeStepMixin:
    """单机 GPU 的局部时间步长与当前 CFL"""

    def _compute_local_time_step_gpu(self, return_physical_too: bool = False):
        """GPU 计算局部 CFL 时间步长（使用所有 SP 的谱半径）。

        `return_physical_too=True` 时额外返回"用物理波速算出的"那一份：
        启用低马赫数预处理时平均流的 dt 按预处理波速放大，而湍流标量
        （k/omega）必须继续用物理波速那一份（与 CPU 侧 cfl.py 同一处理，
        理由见那里的文档）。未启用预处理时两者是同一个数组对象。
        """
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        fc = self.mesh.face_connectivity
        n_faces = fc.n_faces

        owner_cell = cp.asarray(fc.owner_cell)
        neighbor_cell = cp.asarray(
            np.where(fc.is_boundary, 0, fc.neighbor_cell)
        )
        is_boundary = cp.asarray(fc.is_boundary)

        # 面法向和面积：与 compute_inviscid_residual_gpu 同一类每步热路径
        # 性能修复，理由/验证方式相同（见该方法文档）——本方法是每步都要
        # 调用的局部 CFL 步长计算，逐面对象构造的开销在这里同样是每步
        # 复现，不是一次性成本。
        from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
        normal, area_w = _extract_p0_face_geometry(self.mesh.face_flux_points, fc, n_faces)
        normals_gpu = cp.asarray(normal)
        areas_gpu = cp.asarray(area_w)

        cell_volumes = self.mesh_data.get('cell_volumes')
        if cell_volumes is None:
            cell_volumes = cp.asarray(self.mesh.get_all_cell_volumes())

        # 几何/度量 CFL 限制所需数据（与 CPU 侧 cfl.py 的 dt_geometric
        # 同一机制，见 compute_local_cfl_step_gpu 参数文档）：det_jacs 已
        # 在 upload_mesh_data 里常驻显存，metric_flux_scale 只依赖网格
        # 几何（与流场状态无关），缓存后避免每步重复计算——与 CPU 侧
        # solver_geometry.py::_get_metric_flux_scale 同一缓存策略。
        det_jacs_gpu = self.mesh_data.get('det_jacs')
        adj_j_gpu = self.mesh_data.get('adj_j')
        metric_flux_scale_gpu = getattr(self, '_metric_flux_scale_gpu_cache', None)
        # 第四次评审第二轮复核发现：只用 `is None` 判断缓存是否有效，
        # 完全没有形状比较——CPU 侧 solver_geometry.py::_get_metric_flux_scale
        # 曾因"只比较 shape[0]、漏比 n_sps 维度"复现过跨阶数切换后返回
        # 陈旧形状缓存值的真实 bug（Order Continuation 切换阶数后 n_sps
        # 改变），这里连 shape[0] 都没比，是同一类问题的更宽松版本。
        # 当前 GPU 路径还没有接入 Order Continuation（CLI 直接以目标阶数
        # 一次性构造 GPUFRSolver），这个缺陷现在不会被触发，但保持与
        # CPU 侧同等的防御水位，不留一个"看起来复制了修复、实际没复制
        # 关键部分"的陷阱。
        if (metric_flux_scale_gpu is None or metric_flux_scale_gpu.shape != (n_cells, n_sps)) \
                and adj_j_gpu is not None:
            adj_row_norms = cp.linalg.norm(adj_j_gpu, axis=-1)  # (n_cells,n_sps,3)
            metric_flux_scale_gpu = cp.sum(adj_row_norms, axis=-1)  # (n_cells,n_sps)
            self._metric_flux_scale_gpu_cache = metric_flux_scale_gpu

        # 使用所有 SP 计算谱半径（取最大值），而非仅 SP0
        # 对每个 SP 独立计算 CFL 步长，然后取 cell 内最小值
        dt_all_sps = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        precond = getattr(self, "low_mach_precond_enabled", False)
        dt_phys_all_sps = (cp.zeros((n_cells, n_sps), dtype=cp.float64)
                           if precond else None)
        for sp in range(n_sps):
            U_sp = self.U_gpu[:, sp:sp+1, :]  # (n_cells, 1, n_vars)
            det_jacs_sp = det_jacs_gpu[:, sp] if det_jacs_gpu is not None else None
            metric_flux_scale_sp = (
                metric_flux_scale_gpu[:, sp] if metric_flux_scale_gpu is not None else None
            )
            out_sp = compute_local_cfl_step_gpu(
                U_sp, cell_volumes,
                owner_cell, neighbor_cell, is_boundary,
                normals_gpu, areas_gpu,
                None, None,
                cfl=self._current_cfl(),
                poly_order=getattr(self, "order", 0),
                det_jacs_sp=det_jacs_sp,
                metric_flux_scale_sp=metric_flux_scale_sp,
                mach_ref=(self.freestream["mach_ref"] if precond else None),
                return_physical_too=precond,
            )
            if precond:
                dt_all_sps[:, sp], dt_phys_all_sps[:, sp] = out_sp
            else:
                dt_all_sps[:, sp] = out_sp

        # 单元内取最小值时**只看真实自由度**（2026-09-15 系统性审计）：
        # native 四面体的 n_sps 槽位里只有前 n_native=(p+1)(p+2)(p+3)/6 个
        # 是真实解点，其余是零填充槽位——它们在初始化时复制真实 SP #0、
        # 之后残差行被填零，于是**永远冻结在初始条件上**。CPU 侧
        # `cfl.py::compute_local_time_step` 返回的是逐 SP 的 (n_cells,n_sps)
        # 数组、填充槽位的 dt 只会乘到一个恒为零的残差上，所以那边不受
        # 影响；这里做了 `min(axis=1)` 把它归约成逐单元一个标量，冻结
        # 槽位就真的参与了竞争。
        # 具体污染路径：det_jacs/adj_j 对直边 native 四面体是**每单元一个
        # 常数**（见 high_order_mesh_order.py::compute_native_tet_jacobians，
        # 真实行与填充行同值），所以 dt_geometric 不受影响；受影响的是
        # 逐 SP 的波速 (|u|+a) 与 rho/mu_eff——填充槽位给的是初始条件的值。
        # min 取的是"最大波速/最大 mu_eff"那一侧，因此典型来流初始化下
        # （壁面附近流动减速）冻结槽位会把 dt 压得偏小：方向上偏保守、
        # 不会失稳，但它是用初始条件去限制当前时间步，而且会让 CPU-GPU
        # 交叉校验在四面体上无声地对不上。
        from autoflowcfd.fr.native_padding import (
            order_from_n_sps, reduce_per_cell_over_real_sps,
        )
        _np_cells = int(self.mesh.n_prism_cells)
        # 阶数从数组的 SP 轴反解（见 order_from_n_sps 文档）：填充划分由
        # 被归约数组自身决定，不读 solver 上可能短暂不同步的阶数属性。
        _p = order_from_n_sps(dt_all_sps.shape[1])
        dt_mean = reduce_per_cell_over_real_sps(
            dt_all_sps, _np_cells, _p, 'min', xp=cp)  # (n_cells,)
        if not return_physical_too:
            return dt_mean
        dt_phys = (reduce_per_cell_over_real_sps(
            dt_phys_all_sps, _np_cells, _p, 'min', xp=cp) if precond else dt_mean)
        return dt_mean, dt_phys

    def _current_cfl(self) -> float:
        """当前 CFL 数：有自适应控制器时用它，否则退回固定值。

        与 CPU 侧 cfl.py 里同一段逻辑对应（那里是
        `_cfl_controller.cfl_number if ... else 0.1`）。
        """
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import current_cfl_number
        return current_cfl_number(self)
