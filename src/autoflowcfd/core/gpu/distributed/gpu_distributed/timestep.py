"""AutoFlowCFD V2.0 - `MultiGPUDistributedSolver` 的逐单元局部 CFL 步长（mixin，只含方法）。

从 `core/gpu/distributed/gpu_distributed.py` 拆出（2026-09-25）。
"""

import numpy as np

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.gpu_time_integration import compute_local_cfl_step_gpu


class _MultiGPUTimeStepMixin:
    """局部时间步长、当前 CFL 与自适应控制器的全局更新。"""

    def _compute_local_time_step_gpu(self, return_physical_too: bool = False):
        """逐单元局部 CFL 步长（2026-09-14 重写）。

        **这个方法此前是坏的，而且坏在两处**，只是从未被 `step()` 调用过
        （`step()` 当时用调用方传入的全局固定 dt，记作"已接受的简化"），
        所以两个 bug 一直潜伏：

        1. 它读 `self.mesh.face_connectivity`（**全局**面连接关系，
           owner/neighbor 是全局单元编号）去索引 `self.U_gpu`（现在只有
           local+halo 紧凑大小），是与 #1 修复同一类的索引空间不一致；
        2. 它把 `dist_fc` 当 `fc` 传给 `_extract_p0_face_geometry`，而
           `DistributedFlatFaceGeometry` 没有 `.normal`/`.area` 属性——
           一旦被调用必定 `AttributeError`（本轮在 CPU 侧实现分布式局部
           CFL 时实测触发过同一个错误）。

        另有一处被记作"与既有实现保持一致，不在此扩大范围"的简化：只用
        SP0 算谱半径。逐 SP 的几何/度量 CFL 限制（`dt_geometric`）正是
        为了防住坍缩坐标下同一单元内不同 SP 的 det(J) 相差几百倍导致的
        局部刚性失稳（项目记忆 `tet_collapsed_coord_anisotropy`），只取
        SP0 等于把这层保护削掉大部分。现在与单机 GPU 路径一致：按所有 SP
        逐个算、取单元内最小值。

        实现方式与 CPU 分布式一致：面积/法向取**与单机同一个几何量**
        （全局 `FRFaceConnectivity` 的逐面 area/normal，按
        `partition.local_faces` 切片），owner/neighbor 用紧凑索引空间，
        分区边界面与"halo"类面都按内部面处理（两侧累加谱半径）。完整
        论证见 `core/mpi/distributed_cfl.py` 模块文档。

        Returns:
            return_physical_too=False: (n_compact,) 紧凑排列的 dt；
            True: (dt_mean_flow, dt_physical)。**注意返回的是紧凑排列**，
            调用方按 `inv_perm` 换回原生排列后再切 local 段。
        """
        cp = get_cupy()
        dist_fc = self.dist_flat_face
        n_faces = dist_fc.n_faces

        owner_cell = cp.asarray(dist_fc.owner_cell_local)
        neighbor_raw = np.asarray(dist_fc.neighbor_cell_local)
        # 与 CPU 侧 _DistributedFaceConnectivityView 完全同一套语义：
        # 只有物理边界面（以及邻居索引无效的面）算边界；分区边界面与
        # "halo"类面（三个掩码全 False、本 rank 只持有 neighbor 侧）都
        # 有真实邻居，必须按内部面两侧累加。
        is_bnd_np = dist_fc.physical_boundary_mask | (neighbor_raw < 0)
        is_boundary = cp.asarray(is_bnd_np)
        neighbor_cell = cp.asarray(np.where(neighbor_raw < 0, 0, neighbor_raw))

        from autoflowcfd.core.mpi.distributed_cfl import (
            extract_local_face_area_normal,
        )
        area_np, normal_np = extract_local_face_area_normal(dist_fc, self.mesh)
        norms = np.linalg.norm(normal_np, axis=1, keepdims=True)
        unit_normal_np = normal_np / np.maximum(norms, 1e-30)
        normals_gpu = cp.asarray(np.ascontiguousarray(unit_normal_np))
        areas_gpu = cp.asarray(np.ascontiguousarray(area_np))
        assert areas_gpu.shape[0] == n_faces

        cell_volumes = self.mesh_data.get('cell_volumes')
        if cell_volumes is None:
            raise RuntimeError(
                "mesh_data 缺少 cell_volumes——紧凑网格视图构造有误，"
                "不能静默回退到全局 cell volumes（索引空间不同）")

        det_jacs_gpu = self.mesh_data.get('det_jacs')
        adj_j_gpu = self.mesh_data.get('adj_j')
        metric_flux_scale_gpu = None
        if adj_j_gpu is not None:
            cached = getattr(self, '_metric_flux_scale_gpu_cache', None)
            expected_shape = adj_j_gpu.shape[:2]
            if cached is None or cached.shape != expected_shape:
                adj_row_norms = cp.linalg.norm(adj_j_gpu, axis=-1)
                cached = cp.sum(adj_row_norms, axis=-1)
                self._metric_flux_scale_gpu_cache = cached
            metric_flux_scale_gpu = cached

        n_compact = self.U_gpu.shape[0]
        n_sps = self.U_gpu.shape[1]
        precond = getattr(self, "low_mach_precond_enabled", False)
        dt_all = cp.zeros((n_compact, n_sps), dtype=cp.float64)
        dt_phys_all = cp.zeros((n_compact, n_sps), dtype=cp.float64) if precond else None

        for sp in range(n_sps):
            U_sp = self.U_gpu[:, sp:sp + 1, :]
            det_sp = det_jacs_gpu[:, sp] if det_jacs_gpu is not None else None
            mfs_sp = (metric_flux_scale_gpu[:, sp]
                      if metric_flux_scale_gpu is not None else None)
            out_sp = compute_local_cfl_step_gpu(
                U_sp, cell_volumes,
                owner_cell, neighbor_cell, is_boundary,
                normals_gpu, areas_gpu,
                None, None,
                cfl=self._current_cfl(),
                poly_order=getattr(self, "order", 0),
                det_jacs_sp=det_sp,
                metric_flux_scale_sp=mfs_sp,
                mach_ref=(self.freestream["mach_ref"] if precond else None),
                return_physical_too=precond,
            )
            if precond:
                dt_all[:, sp], dt_phys_all[:, sp] = out_sp
            else:
                dt_all[:, sp] = out_sp

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
        # compact 索引空间同样是"棱柱在前"（见 base_flat.n_prism 文档）。
        _np_cells = int(self.dist_flat_face.base_flat.n_prism)
        # 阶数从数组的 SP 轴反解（见 order_from_n_sps 文档）：填充划分由
        # 被归约数组自身决定，不读 solver 上可能短暂不同步的阶数属性。
        _p = order_from_n_sps(dt_all.shape[1])
        dt_mean = reduce_per_cell_over_real_sps(
            dt_all, _np_cells, _p, 'min', xp=cp)
        if not return_physical_too:
            return dt_mean
        dt_phys = (reduce_per_cell_over_real_sps(
            dt_phys_all, _np_cells, _p, 'min', xp=cp) if precond else dt_mean)
        return dt_mean, dt_phys

    def _current_cfl(self) -> float:
        """当前 CFL 数：有自适应控制器时用它，否则退回固定值。

        与单机 GPU `GPUFRSolver._current_cfl` / CPU 侧 cfl.py 同一逻辑。
        控制器按**全局**残差范数更新（见 `step()` 末尾），所有 rank 因此
        得到同一个 CFL 数。
        """
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import current_cfl_number
        return current_cfl_number(self)

    def _update_cfl_controller(self, residual_norm: float) -> None:
        """用**全局**残差范数更新自适应 CFL 控制器。

        必须用全局值：所有 rank 因此得到同一个 CFL 数，进而得到一致的
        局部步长缩放。按各自的局部残差更新会让 rank 间 CFL 漂移，
        破坏分布式一致性。
        """
        c = getattr(self, "_cfl_controller", None)
        if c is not None:
            c.update(residual_norm)
