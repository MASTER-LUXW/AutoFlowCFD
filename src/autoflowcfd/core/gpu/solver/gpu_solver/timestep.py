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

    def _compute_local_time_step_gpu(self, return_physical_too: bool = False, nu_av=None):
        """GPU 计算局部 CFL 时间步长（使用所有 SP 的谱半径）。

        `nu_av`：本步冻结的人工扩散系数（运动粘度，未启用时 None），按 `rho*nu`
        计入粘性限制的有效粘度（与 CPU `_get_cfl_viscosity_field` 同一口径）。

        `return_physical_too=True` 时额外返回"用物理波速算出的"那一份：
        启用低马赫数预处理时平均流的 dt 按预处理波速放大，而湍流标量
        （k/omega）必须继续用物理波速那一份（与 CPU 侧 cfl.py 同一处理，
        理由见那里的文档）。未启用预处理时两者是同一个数组对象。
        """
        cp = get_cupy()
        n_cells = self.mesh.n_cells
        n_sps = self.mesh.n_sps_per_cell

        owner_cell, neighbor_cell, is_boundary, normals_gpu, areas_gpu = self._face_geometry_gpu()

        cell_volumes = self._cell_volumes_gpu()

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

        # 粘性限制：有效粘度（分子 + 湍流涡粘 + 人工粘性）与各向异性长度尺度，
        # 公式与 CPU 同一个函数（`core/fr_solver/cfl_viscous.py`）。2026-10-01
        # 以前这里不传 mu_eff，粘性限制整段不生效。
        from autoflowcfd.core.fr_solver.cfl_viscous import viscous_length_scale_sq
        visc_Lc2 = viscous_length_scale_sq(cell_volumes, owner_cell, neighbor_cell, is_boundary, areas_gpu)
        mu_eff = self._effective_viscosity_gpu(nu_av)
        # 当前阶数（Order Continuation 期间与目标阶数不同；此前误读 `self.order`）
        _co = getattr(self, "current_order", None)
        poly_order = int(self.order if _co is None else _co)

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
                mu_eff=mu_eff[:, sp], visc_Lc2=visc_Lc2,
                poly_order=poly_order,
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

    def _face_geometry_gpu(self):
        """步长计算用的逐面数组 `(owner, neighbor, is_boundary, 单位法向, 物理面积)`，常驻显存（只依赖网格，
        换阶不变）。

        面积与法向取 `face_connectivity` 的物理面几何，与 CPU `cfl.py`、CPU 分布式 `distributed_cfl.py`、
        多 GPU 同一口径。2026-10-09 以前这里取 `inviscid_p0.py::_extract_p0_face_geometry` 的"面积权重"：
        那是每面**第一个通量点**的求积权重，只在 P0（每面一个通量点）等于整面面积；P1 下各单元的面积和
        只有真实值的 1/3.35、P2 只有 1/10.3（棱柱；四面体 1/2.54、1/7.30），对流步长限制随之放大同样的
        倍数——单 GPU 的实际 CFL 是名义值的 3~10 倍，且每步都把这两个数组重新上传一次。
        """
        fc = self.mesh.face_connectivity
        cached = getattr(self, "_face_geometry_gpu_cache", None)
        if cached is not None and cached[0] is fc:
            return cached[1]
        cp = get_cupy()
        normal = np.asarray(fc.normal, dtype=np.float64)
        unit = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        arrays = (cp.asarray(fc.owner_cell), cp.asarray(np.where(fc.is_boundary, 0, fc.neighbor_cell)),
                  cp.asarray(fc.is_boundary), cp.asarray(unit), cp.asarray(np.asarray(fc.area, dtype=np.float64)))
        self._face_geometry_gpu_cache = (fc, arrays)
        return arrays

    def _cell_volumes_gpu(self):
        """单元体积（显存常驻的那份；网格数据没上传它时现传一份）。"""
        vol = self.mesh_data.get('cell_volumes')
        return vol if vol is not None else get_cupy().asarray(self.mesh.get_all_cell_volumes())

    def _effective_viscosity_gpu(self, nu_av=None):
        """有效动力粘度 (n_cells, n_sps)：分子 + 当前湍流/SGS 涡粘 + 人工粘性 `rho*nu_av`。

        与 CPU `cfl.py` 同一时序：湍流取当前（上一步更新后）的场、密度取当前状态。
        """
        cp = get_cupy()
        mu = cp.full(self.Q_gpu.shape[:2], float(self.mu_molecular))
        if getattr(self, "turb_model_gpu", None) is not None:
            mu = mu + self._turbulent_mu_t_gpu()
        elif getattr(self, "sgs_model_gpu", None) is not None and self.sgs_model_gpu.nu_t is not None:
            mu = mu + self.Q_gpu[:, :, 0] * self.sgs_model_gpu.nu_t
        if nu_av is not None:
            mu = mu + self.U_gpu[:, :, 0] * nu_av
        return mu

    def _current_cfl(self) -> float:
        """当前 CFL 数：有自适应控制器时用它，否则退回固定值。

        与 CPU 侧 cfl.py 里同一段逻辑对应（那里是
        `_cfl_controller.cfl_number if ... else 0.1`）。
        """
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import current_cfl_number
        return current_cfl_number(self)
