"""MultiGPUDistributedSolver 初始化和 I/O 混入类。

从 gpu_distributed.py 拆出，控制单文件行数。
"""

import numpy as np
from loguru import logger

from autoflowcfd.core.gpu import get_cupy


class _GPUDistributedInitMixin:
    """MultiGPUDistributedSolver 初始化/I/O 混入。"""

    def _rebuild_partition(self, partition_info):
        """从分区信息字典重建 DistributedPartition。"""
        return partition_info

    def _init_wall_distance_distributed(self):
        """分布式壁面距离计算（compact 索引空间，2026-09-02 修复）。

        真实 bug 修复：此前这里对**全局** `self.mesh.sps_coords`（"传统
        模式"下 `self.mesh` 是完整网格）整体 `reshape(-1, 3)` 后对全部
        SPs 做 KDTree 查询，产出长度为 `n_global_cells*n_sps` 的
        `dist_flat`，再 `.reshape(n_local, n_sps)`——这是一个真实的
        reshape bug：只有 `n_global_cells == n_local`（即单 rank）时
        总长度才恰好对得上，n_ranks>1 时这行 `reshape` 必然因元素总数
        不匹配而崩溃（若恰好凑巧不崩溃，也是读到与本 rank 完全无关的
        单元的距离值）。且即便修掉这个 reshape，"n_local 大小、
        partition.local_cells 自身顺序"这个目标形状本身也是错的——
        `_compute_turbulence_source_distributed` 消费 `wall_distance_
        gpu` 时是和 `self.mesh_data`（compact 索引空间，"棱柱在前"
        local+halo 排列）一起用的，两者索引空间必须一致，否则会读到
        错位的壁面距离。

        与 CPU 分布式版 `core/mpi/distributed_turbulence.py::compute_
        distributed_wall_distance` 同一个设计（KDTree 本身是 CPU 计算，
        算完再上传 GPU，不需要在 GPU 上重新实现）：按
        `dist_flat_face.compact_global_ids` 从全局 `sps_coords` 切出
        compact 索引空间的坐标子集，查询结果直接是 (n_compact, n_sps)
        形状，不需要 reshape 到任何 n_local 相关的尺寸。
        """
        cp = get_cupy()
        compact_global_ids = self.dist_flat_face.compact_global_ids
        n_compact = len(compact_global_ids)
        n_sps = self.mesh.n_sps_per_cell

        wall_indices = None
        boundary_groups = getattr(self.mesh, 'boundary_groups', None)
        # `hasattr` 本身对 `boundary_groups=None`（合成测试网格/无边界组
        # 元数据的网格）恒为 True——属性存在但值是 None，`.items()` 会
        # 真实 AttributeError（本次新增测试直接测出这个真实 bug，不是
        # 假设性的）；用 `getattr(...) is not None` 才是正确的存在性
        # 判据。
        if boundary_groups is not None:
            for bg_name, bg in boundary_groups.items():
                if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                    wall_indices = bg.get('node_indices')
                    break

        if wall_indices is not None and len(wall_indices) > 0:
            # 真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：坐标缺失
            # 时静默回退到硬编码常量 0.01m 是被禁止的简化——直接失败。
            if not (hasattr(self.mesh, 'sps_coords') and self.mesh.sps_coords is not None):
                raise RuntimeError(
                    f"Rank {self.rank}: wall distance field not computed for turbulence "
                    f"model '{getattr(self, 'turb_model_name', '?')}': mesh has no "
                    f"sps_coords. Industrial-grade calculation requires accurate wall "
                    f"distance, not simplified estimates."
                )
            sps_coords_compact = self.mesh.sps_coords[compact_global_ids]  # (n_compact,n_sps,3)
            from scipy.spatial import cKDTree
            wall_coords = self.mesh.nodes[wall_indices]
            tree = cKDTree(wall_coords)
            dist_flat, _ = tree.query(sps_coords_compact.reshape(-1, 3), k=1)
            self.wall_distance_gpu = cp.asarray(dist_flat.reshape(n_compact, n_sps))
        else:
            # 找不到 WALL 边界组时退回特征长度估计——物理上合理的近似，
            # 不属于本次修复目标（"假常量" 0.01m）范畴，保留原行为。
            # `cell_volumes` 已经是 compact 索引空间（见 __init__ 里
            # `_CompactMeshDataView` 构造处），直接用，不需要再切片。
            volumes = self.mesh_data.get('cell_volumes')
            if volumes is None:
                raise RuntimeError(
                    f"Rank {self.rank}: 既没有 WALL 边界节点也没有 cell_volumes，"
                    f"无法提供任何壁面距离估计。"
                )
            h_char = volumes ** (1.0 / 3.0)
            self.wall_distance_gpu = cp.broadcast_to(
                h_char[:, None], (n_compact, n_sps)
            ).copy()

    def _init_modal_filter_distributed(self):
        """分布式（多 GPU）模态滤波初始化。

        **真实缺陷修复（2026-09-18）**：此前这里传的是
        `n_prism = self.mesh.n_prism_cells`，而 `build_gpu_filter_func`
        按 `U[:n_prism]` 切片。两处叠加有两个错：

        1. `self.U_gpu` 是 GPU halo 交换的**原生**排列（local 在前、
           halo 在后，见 `gpu_halo_exchange.py::GPUHaloExchange.exchange`），
           棱柱与四面体在其中是**交错**的——"棱柱在前"只对
           `dist_flat_face` 的**紧凑**排列成立（残差计算靠 `_perm_gpu`
           换过去、`_inv_perm_gpu` 换回来）。
        2. "传统模式"下 `self.mesh` 是**完整全局网格**，
           `mesh.n_prism_cells` 是全局棱柱数，通常**大于** `n_local`。
           `U[:n_prism]` 于是被 CuPy 静默钳到 `n_local`，后果是**每个
           local 单元、包括四面体，都被施加了棱柱滤波矩阵**，不报任何
           错。四面体走 native PKD/Dubiner 基（带零填充槽位），与棱柱的
           张量积基完全不同，混用在数值上没有意义。

        改为传 `cell_is_prism`（原生 local 排列的单元类型掩码），与 CPU
        分布式 `build_filter_func_by_cell_type` 同一处理方式；
        `build_gpu_filter_func` 也加了"n_prism 越界就报错、不静默钳"。

        同时补齐 `AFCFD_FILTER_MODE=sensor`（2026-09-17 起的默认档）
        在本后端的接线，此前会退回 `project`（全局逐 stage 施加精确
        投影，功能上等于 legacy：P1 退化成 P0、壁面剪应力恒为零）。
        """
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell

        # 滤波档解析放在 try 之外，否则会被下面那个
        # `except Exception -> warning` 吞掉（显式请求得不到满足必须
        # 报错，见 fr_solver/filter.py::resolve_filter_mode）。
        from autoflowcfd.core.fr_solver.filter import resolve_filter_mode
        mode = resolve_filter_mode("gpu-mpi")

        try:
            filter_prism = self.ops.filter_prism
            filter_tet = self.ops.filter_tet
            if filter_prism is None and filter_tet is None:
                return

            dist_fc = self.dist_flat_face
            cct = np.asarray(dist_fc.compact_cell_type)
            inv_perm = np.asarray(dist_fc.inv_perm)
            # compact_cell_type 处在"棱柱在前"紧凑排列，用 inv_perm 换回
            # 原生排列再切 local 段——`self.U_gpu` 所在的空间。
            cell_is_prism = (cct[inv_perm][:n_local] == 0)

            if mode == "sensor":
                self.filter_func_gpu = (
                    self._build_sensor_gated_filter_distributed_gpu(
                        n_local, n_sps, cell_is_prism,
                        filter_prism, filter_tet))
            else:
                from autoflowcfd.core.gpu.gpu_modal_filter import (
                    build_gpu_filter_func,
                )
                self.filter_func_gpu = build_gpu_filter_func(
                    n_local, n_sps, 0,
                    filter_prism, filter_tet,
                    device_id=self.device_id,
                    cell_is_prism=cell_is_prism,
                )
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Modal filter init failed: {e}")

    def _build_sensor_gated_filter_distributed_gpu(
        self, n_local, n_sps, cell_is_prism, filter_prism, filter_tet
    ):
        """多 GPU 的**传感器门控**滤波回调（2026-09-18 接线）。

        **不新写一份门控实现**：复用 CPU 那一个
        `build_sensor_gated_filter_func_arrays`——它连同两个判据内核
        （Persson-Peraire、BJ 越界）都已改成数组模块无关，传 CuPy 矩阵
        进去整条回调就走 CuPy 的同名函数。一份实现服务四条后端。

        与 CPU MPI（`core/mpi/distributed_solver.py::_build_sensor_gated_
        filter_func_distributed`）完全同一套推导，三条约束逐字相同：

        1. 分区边界面**不能**当边界面排除——否则掩码随 rank 数变化，
           同一个算例换分区数得到不同的解；
        2. halo 扩展必须用**当前 stage** 的解（滤波在正定性投影之后
           施加），所以每 stage 多一次 halo 交换，不能复用残差求值时
           缓存的 `U_extended_gpu`；
        3. `owner_cell_local`/`neighbor_cell_local` 在"棱柱在前"紧凑
           排列，而场在原生排列，用 `owner_native = perm[oc]` 换算。
        """
        cp = get_cupy()
        from autoflowcfd.core.fr_solver.filter import (
            build_sensor_gated_filter_func_arrays,
        )
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor,
        )

        sensor = resolve_troubled_sensor()
        conn = {}
        if sensor in ("bounds", "both"):
            dist_fc = self.dist_flat_face
            perm = np.asarray(dist_fc.perm)
            n_total = int(self.partition.n_total_cells)
            if perm.size != n_total:
                raise RuntimeError(
                    f"dist_flat_face.perm 长度 {perm.size} 与 "
                    f"n_total_cells {n_total} 不符，索引换算不可靠")
            oc = np.asarray(dist_fc.owner_cell_local)
            nc = np.asarray(dist_fc.neighbor_cell_local)
            bnd = np.asarray(dist_fc.is_boundary, dtype=bool)
            # 与 CPU 侧同一条自洽性护栏：边界面的 neighbor_cell_local
            # 是 -1，两个集合必须严格重合，否则 BJ 包络会读到错误单元
            # 而在日志里完全看不出来。
            if not np.array_equal(nc < 0, bnd):
                n_mismatch = int(np.count_nonzero((nc < 0) != bnd))
                raise RuntimeError(
                    f"分布式面几何自洽性失败：{n_mismatch} 条面的 "
                    f"(neighbor_cell_local < 0) 与 is_boundary 不一致")
            owner_native = perm[oc]
            neigh_native = perm[np.where(nc >= 0, nc, 0)]
            gpu_halo = self.gpu_halo

            def halo_extend(U_local_3d):
                # 返回 (n_total, n_sps, n_vars) 原生排列的 CuPy 数组。
                return gpu_halo.exchange(cp.ascontiguousarray(U_local_3d))

            with cp.cuda.Device(self.device_id):
                conn = dict(owner_cell=cp.asarray(owner_native),
                            neighbor_cell=cp.asarray(neigh_native),
                            is_boundary=cp.asarray(bnd),
                            freestream=self.freestream,
                            halo_extend=halo_extend)

        order = int(getattr(self, "current_order", self.order))
        with cp.cuda.Device(self.device_id):
            fp = (filter_prism if hasattr(filter_prism, "device")
                  else cp.asarray(filter_prism))
            ft = (filter_tet if hasattr(filter_tet, "device")
                  else cp.asarray(filter_tet))
            return build_sensor_gated_filter_func_arrays(
                n_local, n_sps, order, fp, ft,
                cell_is_prism=cp.asarray(cell_is_prism),
                sensor=sensor, **conn)

    def _init_distributed_face_geometry(self):
        """初始化分布式面几何（GPU 版）。

        此前这里的调用方式与 build_distributed_flat_face 的真实签名
        `(mesh, ops, partition)` 不匹配——旧代码传的是
        `(flat_face, self.partition, self.rank, self.n_ranks)`（4 个
        位置参数，且第一个是已经算好的全局 FlatFaceGeometry 而不是
        mesh；该函数内部本就会自己调用 get_flat_face_geometry(mesh,
        ops)，不需要调用方先算一遍）——必然 TypeError，且被外层宽
        `except Exception` 吞掉，`self.flat_face_gpu` 静默设为 None。
        下游 gpu_inviscid.py 在 flat_face_gpu 为 None 时会用**未分区的
        全局网格**重新构建一份非分布式面几何，等于让多 GPU 求解器
        悄悄退化成"每个 rank 各算各的全局残差、不做 halo 耦合"——
        不是崩溃，是产出物理上错误的结果，比崩溃更隐蔽（V2.0 专家组
        评审逐行核实）。修复：改用真实签名调用；同时去掉吞掉一切异常
        的 try/except——分布式面几何构建失败时必须让求解器初始化
        真正失败，而不是静默退化成错误的单机行为。
        """
        from autoflowcfd.core.mpi.distributed_flat_face import (
            build_distributed_flat_face,
        )
        dist_flat_face = build_distributed_flat_face(
            self.mesh, self.ops, self.partition,
            cell_partition=getattr(self, 'cell_partition', None),
        )
        # #1（2026-08-28）：此前只保留 GPU 上传后的版本，丢弃了 CPU 侧
        # dist_flat_face 本身——但 __init__ 后续需要它的 perm/inv_perm/
        # compact_global_ids（构造压缩 mesh_data、在残差计算前后重排
        # U_extended_gpu，见 compute_inviscid_residual_gpu 文档）以及
        # base_flat（边界幽灵态计算需要的、按压缩索引重映射过的 CPU 侧
        # FlatFaceGeometry，见 gpu_inviscid.py::compute_inviscid_residual_
        # fr_gpu 的 flat_face_cpu 参数文档），因此这里保留一份引用。
        self.dist_flat_face = dist_flat_face
        from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
        # 真实 bug 修复（2026-09-02，排查"--multi-gpu 仍拒绝 native"时发现，
        # 与 native 无关，collapsed 模式下同样必现）：此前这里把
        # `dist_flat_face`（`DistributedFlatFaceGeometry` 包装对象）直接
        # 传给 `build_gpu_flat_face`，但该函数（`GPUFlatFaceGeometry.
        # __init__`）需要的是一个真正的 `FlatFaceGeometry`（`.owner_cell`/
        # `.neighbor_cell`/`.owner_adj_row_exact`/`.color_face_indices`等
        # 字段直接存在）——`DistributedFlatFaceGeometry` 只对其中一部分
        # 字段提供了 `@property` 转发到 `base_flat`（`owner_axis`/
        # `true_normal`/`boundary_extrap`等），但**没有**为 `owner_cell`/
        # `neighbor_cell`/`owner_adj_row_exact`/`neighbor_adj_row_exact`/
        # `owner_cube_face`/`neighbor_cube_face`/`true_area_weight`/
        # `boundary_extrap_native`/`lift_native`/`mixed_*`/
        # `color_face_indices`/`n_colors` 提供转发（该类真正持有的对应
        # 字段名是 `owner_cell_local`/`neighbor_cell_local`，与
        # `GPUFlatFaceGeometry` 期望的名字不同）——直接传 `dist_flat_face`
        # 会在 `GPUFlatFaceGeometry.__init__` 第一次访问 `flat_face.
        # owner_cell` 时 `AttributeError`，`MultiGPUDistributedSolver`
        # 从未真正构造成功过（本机无 CuPy/MPI，这条路径此前完全没有可
        # 执行的测试覆盖，用纯 CPU/numpy 的 `DistributedFlatFaceGeometry`
        # 构造+ `hasattr` 检查已经能独立于 CuPy 决定性验证这个
        # AttributeError，见 tests/unit/test_distributed_flat_face_gpu_
        # compat.py）。正确对象是 `dist_flat_face.base_flat`——已经是
        # 按 local+halo 压缩索引空间重新构造好的完整 `FlatFaceGeometry`
        # （`owner_cell=owner_cell_local`/`neighbor_cell=neighbor_cell_local`
        # 等，见 build_distributed_flat_face 构造 `sub_flat` 处），包含
        # native 四面体（路径C）字段在内的全部数据。
        self.flat_face_gpu = build_gpu_flat_face(dist_flat_face.base_flat, self.device_id)

    def _halo_exchange_gpu(self):
        """GPU 直接 halo 交换（优化版）。

        支持两种模式：
        1. CUDA-aware MPI：GPU buffer 直接通信（零拷贝）
        2. Staging buffer：GPU→CPU→MPI→CPU→GPU（只传输必要数据）
        """
        self.U_extended_gpu = self.gpu_halo.exchange(self.U_gpu)

    def _compute_turbulence_source_distributed(self, dt: float):
        """分布式 SST 源项+输运计算（真正实现，2026-09-02）。

        复用而不是重新实现单机数值逻辑（与 CPU MPI 路径 `core/mpi/
        distributed_turbulence.py::distributed_compute_turbulence_
        source_and_viscosity` 完全同一个设计）：k/omega 做与平均流
        `U_gpu` 同一套 halo 交换 + compact 索引空间重排，构造一个
        compact 大小的临时 `GPUTurbulenceSST`"视图"，调用完全相同的
        `compute_source_terms_gpu`/`compute_turbulence_transport_
        residual_gpu`/`update_fields_gpu`，只把 local cells 的结果
        写回真正的（n_local 大小的）`turb_model_gpu`。

        此前这里直接用 `self.U_gpu`（native 排列，只有 n_local，没有
        halo）算梯度、且引用了本类从未定义过的 `self.Q_gpu`/
        `self.ops_data`（后者已在 __init__ 补上）——两处都是真实 bug，
        不只是"wall_distance 索引空间不对齐"这一处，见本方法此前版本
        文档。

        Args:
            dt: 本步物理时间步长（标量）——与本类 mean-flow RK 推进
                本步的**逐单元局部物理步长的均值**（2026-09-14：分布式
                路径已补齐逐 cell 局部 CFL，此前"用全局固定步长"的既定
                简化已不存在，见 gpu_distributed.py::
                _compute_local_time_step_gpu 的重写说明），与单机 GPU
                路径的既定约定一致
                （`update_fields_gpu(dt: float, ...)` 本来就接受标量，
                单机版用 `float(cp.mean(dt_local))`，不是 CPU 路径那种
                per-cell dt_local 数组）。

        Returns:
            mu_t_field_compact: (n_compact, n_sps) 动力涡粘度场，供
            `compute_viscous_residual_gpu` 的 BR1 界面项消费；
            `turb_model_gpu is None` 时返回 None。
        """
        cp = get_cupy()

        if self.turb_model_gpu is None:
            if self.sgs_model_gpu is None:
                return None
            # 纯 LES（2026-09-02）：WALE 是纯代数模型，没有 SST 那种
            # k/omega 跨步 ODE 积分状态，不需要"halo 交换持久状态+构造
            # 临时视图"这套复杂度——直接用当前（halo 交换后的 compact）
            # 速度场现算 mu_t 即可，与单机版
            # `_apply_turbulence_corrections_gpu` 在操作分裂时序上的
            # 差异（单机版排在 step() 状态更新之后、供*下一步*用）在
            # 数学上等价：WALE 不依赖历史状态，"用当前状态现算"和"上一步
            # 结束时用同一个当前状态算好缓存"是同一个值，没有必要在
            # 分布式路径上复刻单机版那个只是为了性能缓存的调用时序。
            U_extended = self.gpu_halo.exchange(self.U_gpu)
            U_compact = self._permute_to_compact(U_extended)
            from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
            from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu
            Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
            rho_compact = Q_compact[:, :, 0]
            # 真实 bug 修复（2026-09-03，同下面 300 行附近 compute_source_
            # terms_gpu 调用点同一处修复）：不能对*守恒*变量 U_compact
            # 求梯度再切片动量分量冒充速度梯度，直接对 Q_compact 的速度
            # 分量求梯度。
            grad_vel = compute_physical_gradient_gpu(Q_compact[..., 1:4], self.mesh_data, self.ops_data)
            nu_t_compact = self.sgs_model_gpu.compute_eddy_viscosity_gpu(grad_vel, self._grid_scale_compact)
            return rho_compact * nu_t_compact

        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell
        n_compact = self.n_compact

        # 1. 平均流 halo 交换（与 mean-flow 残差同一个 self.gpu_halo）+
        # 重排到 compact 索引空间——只为了拿 Q（rho/velocity）用于
        # grad_vel，与 compute_inviscid_residual_gpu 同一处理。
        U_extended = self.gpu_halo.exchange(self.U_gpu)
        U_compact = self._permute_to_compact(U_extended)

        # 2. k/omega halo 交换（独立于平均流的 2-var 交换）。
        k_omega_local = cp.stack(
            [self.turb_model_gpu.k_field, self.turb_model_gpu.omega_field], axis=-1
        )  # (n_local,n_sps,2)
        k_omega_extended = self.turb_halo_gpu.exchange(k_omega_local)
        k_omega_compact = self._permute_to_compact(k_omega_extended)

        # 2b. des_length_scale halo 交换（DDES/IDDES 专用，2026-09-02，
        # 与 CPU MPI 路径 distributed_turbulence.py 同一处真实 bug 修复
        # 同一个根因——`des_length_scale` 是跨步持久状态，必须与
        # k_field/omega_field 同一套 halo 交换+compact 重排才能在第二
        # 次及以后的调用里正确使用，不能直接把 n_local 大小的数组
        # setattr 到 n_compact 大小的临时视图上）。只有 1 个分量，不能
        # 复用 2-var 的 turb_halo_gpu。
        des_length_scale_compact = None
        if self.ddes_model_gpu is not None and getattr(self.turb_model_gpu, "des_length_scale", None) is not None:
            des_len_extended = self.des_length_scale_halo_gpu.exchange(
                self.turb_model_gpu.des_length_scale[:, :, None]
            )[:, :, 0]
            des_length_scale_compact = self._permute_to_compact(des_len_extended)

        # 3. 构造 compact 索引空间的临时 GPUTurbulenceSST"视图"——不能
        # 直接复用真正的 turb_model_gpu（只有 n_local 大小），需要一个
        # 同形状为 n_compact 的临时对象供 compute_source_terms_gpu/
        # update_fields_gpu 读写，用完只把 local 部分写回真正的模型。
        from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
        turb_view = GPUTurbulenceSST(n_compact, n_sps, self.device_id)
        for attr in (
            "sigma_k1", "sigma_k2", "sigma_w1", "sigma_w2", "beta1", "beta2",
            "a1", "kappa", "beta_star", "k_max", "omega_max", "production_factor",
        ):
            if hasattr(self.turb_model_gpu, attr):
                setattr(turb_view, attr, getattr(self.turb_model_gpu, attr))
        # des_length_scale 已在上面 2b 步用正确的 halo 交换+compact 重排
        # 算好（None 表示还没有跨步长度尺度记忆，与首次调用/纯 SST 一致，
        # 不能像其余常量那样直接 setattr 原始 n_local 大小的值——见 2b
        # 步注释的真实 bug 说明）。
        turb_view.des_length_scale = des_length_scale_compact
        turb_view.k_field = k_omega_compact[..., 0].copy()
        turb_view.omega_field = k_omega_compact[..., 1].copy()

        from autoflowcfd.core.gpu.residual.gpu_flux import conserved_to_primitive_gpu
        from autoflowcfd.core.gpu.residual.gpu_gradients import (
            compute_physical_gradient_gpu,
            compute_physical_scalar_gradient_gpu,
        )

        Q_compact = conserved_to_primitive_gpu(U_compact[..., :5])
        rho_compact = Q_compact[:, :, 0]

        # 真实 bug 修复（2026-09-02 的 shape 修复 + 2026-09-03 的进一步
        # 修正，与单机 GPU 路径 gpu_solver_io.py 同一处独立发现的 bug
        # 同一个根因，见该文件对应修复说明）：`compute_source_terms_gpu`
        # 的 `grad_U` 形参实际期望速度梯度 (n_cells,n_sps,3,3)（内部
        # compute_strain_rate_magnitude_gpu 直接对末两维转置相加）。
        # 2026-09-02 的修复只解决了"切片成 3 变量"这一半（避免 shape
        # 崩溃），但对*守恒*变量 U_compact 求梯度再切片动量分量仍然是
        # 冒充速度梯度——grad(rho*u) != rho*grad(u)，除非密度梯度处处
        # 为零。改为直接对 Q_compact 的速度分量求梯度。
        grad_vel = compute_physical_gradient_gpu(Q_compact[..., 1:4], self.mesh_data, self.ops_data)

        d_wall = self.wall_distance_gpu
        if d_wall is None:
            raise RuntimeError(
                f"Rank {self.rank}: wall distance field not computed for turbulence "
                f"model '{getattr(self, 'turb_model_name', '?')}'. Please ensure "
                f"_init_wall_distance_distributed() ran during solver initialization. "
                f"Industrial-grade calculation requires accurate wall distance, not "
                f"simplified estimates."
            )

        grad_k = compute_physical_scalar_gradient_gpu(turb_view.k_field, self.mesh_data, self.ops_data)
        grad_omega = compute_physical_scalar_gradient_gpu(turb_view.omega_field, self.mesh_data, self.ops_data)

        max_grad_mag = 1e6
        grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
        if cp.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / cp.maximum(grad_k_mag, 1e-10)
            grad_k *= cp.clip(scale_k, 0, 1)[..., None]
        if cp.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / cp.maximum(grad_omega_mag, 1e-10)
            grad_omega *= cp.clip(scale_omega, 0, 1)[..., None]

        Sk, S_omega = turb_view.compute_source_terms_gpu(
            Q_compact, grad_vel, d_wall, self.mu_molecular, grad_k, grad_omega,
        )

        # DDES/IDDES 长度尺度更新（2026-09-02）：必须排在 compute_
        # source_terms_gpu **之后**——与单机版 gpu_solver_io.py 同一处
        # 真实顺序 bug 修复同一个根因（该文件"必须在 compute_source_
        # terms_gpu 之前"的旧注释是错的，CPU 参照的真实顺序是
        # compute_source_terms 用*上一步*的 des_length_scale（"慢一拍"
        # 耦合），算完之后才用*本步*的 nu_t 写入*下一步*要用的新
        # des_length_scale，两者互相依赖对方的输出、只能这样错开一步，
        # 见该文件对应修复文档的完整推导）。
        if self.ddes_model_gpu is not None:
            nu_field = self.mu_molecular / cp.maximum(rho_compact, 1e-10)
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUIDDESModel
            if isinstance(self.ddes_model_gpu, GPUIDDESModel):
                self.ddes_model_gpu.apply_to_sst_model_iddes_gpu(
                    turb_view, d_wall, self.iddes_h_max_compact, self.iddes_h_wn_compact,
                    nu_field, grad_vel,
                )
            else:
                cell_volumes = self.mesh_data.get('cell_volumes')
                self.ddes_model_gpu.apply_to_sst_model_gpu(
                    turb_view, d_wall, cell_volumes, nu_field, grad_vel,
                    h_max=getattr(self, 'iddes_h_max_compact', None),
                )

        dk_dt = Sk / cp.maximum(rho_compact, 1e-10)
        domega_dt = S_omega / cp.maximum(rho_compact, 1e-10)

        # 4. 完整输运项（对流+扩散）：与单机版同一个 compute_turbulence_
        # transport_residual_gpu，用一个只暴露它实际读取属性的最小鸭子
        # 类型 adapter（mesh 只需要 n_cells/n_sps_per_cell/n_prism_cells，
        # 不是完整网格对象——`n_prism_cells` 即便 `self.mesh_data` 里
        # 已经有 `'n_prism'` 键，`dict.get('n_prism', solver.mesh.
        # n_prism_cells)` 的默认值表达式仍会被 Python 无条件求值，
        # 缺这个属性会直接 AttributeError，不是"用不到就不用给"）。
        import types
        n_prism_compact = self.dist_flat_face.base_flat.n_prism
        transport_adapter = types.SimpleNamespace(
            turb_model_gpu=turb_view,
            mesh=types.SimpleNamespace(
                n_cells=n_compact, n_sps_per_cell=n_sps, n_prism_cells=n_prism_compact,
            ),
            Q_gpu=Q_compact,
            U_gpu=U_compact,
            mu_molecular=self.mu_molecular,
            mesh_data=self.mesh_data,
            ops_data=self.ops_data,
            flat_face_gpu=self.flat_face_gpu,
            wall_distance_gpu=d_wall,
            _wall_mask_k_gpu=self._wall_mask_k_gpu,
        )
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
            compute_turbulence_transport_residual_gpu,
        )
        transport_k, transport_omega = compute_turbulence_transport_residual_gpu(
            transport_adapter, grad_vel=grad_vel,
        )

        # 5. 场更新（点隐式阻尼+输运，见 update_fields_gpu 文档）。
        # `dt` 现在由调用方传入**逐单元局部物理步长的均值**（2026-09-14，
        # 与 CPU 分布式/单机同一时序：湍流标量必须用按**物理**波速算出的
        # 那一份 dt，不能跟着低马赫数预处理放大——k/omega 的显式更新刻意
        # 没有 point-implicit 阻尼）。此前这里用的是调用方传入的全局固定
        # dt，并注明"本类分布式路径目前统一用全局固定步长、
        # _compute_local_time_step_gpu 与 compact 索引空间不兼容"——那个
        # 前提已经不成立：该方法已重写为紧凑索引空间、逐 SP 取最小值。
        # `update_fields_gpu` 接标量，所以这里取均值（与该函数在其它路径
        # 上的既有用法一致，见 gpu_solver_io.py 的 `cp.mean(dt_physical)`）。
        turb_view.update_fields_gpu(
            float(dt), dk_dt, domega_dt,
            transport_k=transport_k, transport_omega=transport_omega,
        )

        # 5.4 真实 bug 修复（2026-09-12，与单机 GPU 路径 gpu_solver_io.py、
        # CPU 分布式路径（经 compute_turbulence_source 自动获得，见该处
        # 文档）同一处修复，完整推导见 gpu_modal_filter.py::
        # filter_scalar_field_gpu 文档）：k/omega 场同样需要模态滤波。
        # compact 索引空间下 n_prism 用 dist_fc.base_flat.n_prism（"棱柱
        # 在前"的既定 compact 排序约定，与 CPU 分布式路径的 mesh_adapter
        # 同一个量）。
        if n_sps > 1:
            from autoflowcfd.core.fr_solver.filter import resolve_turb_filter_gate
            from autoflowcfd.core.gpu.gpu_modal_filter import (
                filter_scalar_field_gated_gpu, filter_scalar_field_gpu,
            )
            n_prism_compact = self.flat_face_gpu.n_prism
        # 门控维度 `AFCFD_FILTER_TURB_GATE`（2026-09-15 系统性审计的 A 类
        # 发现）：这一维此前只在 CPU 路径接线过，GPU 这边无条件全场滤波
        # ——同一个环境变量在不同后端意味着不同的数值方案且无任何提示。
        # 现已用 GPU 版同一套传感器补齐（gpu_troubled_cell.py），默认
        # "all" 与此前行为逐位一致。
            if resolve_turb_filter_gate() == "sensor":
                from autoflowcfd.core.gpu.gpu_troubled_cell import (
                    compute_turb_troubled_mask_gpu,
                )
                _order = int(getattr(self, "current_order", None)
                             or getattr(self, "order", 0))
                troubled = compute_turb_troubled_mask_gpu(
                    turb_view.k_field, turb_view.omega_field,
                    n_prism_compact, _order)
                self._turb_filter_troubled_frac = float(cp.mean(troubled))
                turb_view.k_field = filter_scalar_field_gated_gpu(
                    turb_view.k_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet, troubled,
                )
                turb_view.omega_field = filter_scalar_field_gated_gpu(
                    turb_view.omega_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet, troubled,
                )
            else:
                turb_view.k_field = filter_scalar_field_gpu(
                    turb_view.k_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet,
                )
                turb_view.omega_field = filter_scalar_field_gpu(
                    turb_view.omega_field, n_prism_compact,
                    self.ops.filter_prism, self.ops.filter_tet,
                )
            turb_view.apply_positivity_limiter_gpu()

        # 5.5 真实缺口修复（2026-09-05，代码复审发现，与单机 GPU 路径
        # gpu_solver_io.py 同一处修复同一个根因）：omega 壁面 Wilcox
        # 解析值扩散侧未闭合，此前多GPU分布式路径完全没有移植这一步
        # ——直接复用上面已经构造好的 transport_adapter（它已经暴露了
        # enforce_omega_wall_relaxation_gpu 需要的全部属性：Q_gpu/
        # turb_model_gpu/flat_face_gpu/wall_distance_gpu/mu_molecular/
        # _wall_mask_k_gpu/mesh.n_cells），turb_view 是同一个对象引用，
        # 函数内部对 turb_model_gpu.omega_field 的原地修改直接反映到
        # turb_view 上。
        if getattr(self, "turb_model_name", "").upper() in ("SST", "DDES", "IDDES"):
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import (
                enforce_omega_wall_relaxation_gpu,
            )
            enforce_omega_wall_relaxation_gpu(cp, transport_adapter)

        # 6. 只把 local cells 的更新结果写回真正的 turb_model_gpu（native
        # 排列，见 dist_flat_face.inv_perm 文档——turb_view.k_field 是
        # compact 排列，需要先换回原生排列再切 [:n_local]，与平均流
        # 残差同一处理）。
        k_native = self._unpermute_from_compact(turb_view.k_field)
        omega_native = self._unpermute_from_compact(turb_view.omega_field)
        self.turb_model_gpu.k_field[:] = k_native[:n_local]
        self.turb_model_gpu.omega_field[:] = omega_native[:n_local]
        nu_t_native = self._unpermute_from_compact(turb_view.nu_t)
        self.turb_model_gpu.nu_t[:] = nu_t_native[:n_local]

        # DDES/IDDES：des_length_scale 是跨步持久状态（本步 apply_to_
        # sst_model[_iddes] 刚用本步 nu_t 算出、要写回供*下一步* 2b 步
        # 读取），与 k_field/omega_field 同一套写回处理。
        if self.ddes_model_gpu is not None and getattr(turb_view, "des_length_scale", None) is not None:
            des_len_native = self._unpermute_from_compact(turb_view.des_length_scale)
            self.turb_model_gpu.des_length_scale = des_len_native[:n_local].copy()

        # mu_t_field 保持 compact 排列（平均流粘性残差直接消费 compact
        # 索引空间的场，见 compute_viscous_residual_gpu 里 U_compact 的
        # 同一处理）。
        return rho_compact * turb_view.nu_t

    def get_state_cpu(self):
        """下载 GPU 状态到 CPU。"""
        cp = get_cupy()
        return {'U': cp.asnumpy(self.U_gpu)}

    def cleanup(self):
        """释放 GPU 资源。"""
        self.array_mgr.cleanup()

    def save_checkpoint_distributed(
        self, output_dir: str, iteration: int, input_file: str,
        order: int, turbulence_model: str, backend: str = "gpu",
        history: dict = None, target_order: int = None,
    ):
        """分布式 checkpoint 保存（#4，V2.0 专家组盲审第4轮，2026-08-28）。

        真实 bug 修复：此前这里调用 `distributed_save_checkpoint(
        U_local_cpu, local_cells, n_global_cell, path, self.rank,
        self.n_ranks)`（6 个位置参数），但该函数真实签名是 `(solver,
        output_dir, iteration, input_file, order, turbulence_model,
        backend, history=None)`（第一个是完整的 solver 对象，不是裸
        数组，且读取 `solver.state.U`/`solver.partition.n_local_cells/
        n_global_cells/local_cells`——CPU `DistributedFRSolver` 的接口
        形状）——参数个数和类型都对不上，且即便签名对齐，
        `MultiGPUDistributedSolver` 当时还有更深层的架构缺陷
        （`self.U_gpu` 按全局单元数分配、不是真正的"只存 local cells"），
        使得 checkpoint 即便"看起来跑通"也不会是真正的分布式局部状态。
        这两个前提现在都已经解决（见 __init__ 里 `self.U_gpu`/
        `self.mesh_data` 按 local+halo 压缩索引空间重构的说明）：这里
        先把 `self.U_gpu`（GPU，local cells）下载并同步进
        `self.state.U`（`DistributedFRState` 已在 __init__ 构造好但
        此前从未真正被写入过），再按真实签名调用
        `distributed_save_checkpoint`。

        Args:
            output_dir: 输出目录
            iteration: 当前迭代数
            input_file: 原始网格文件路径
            order: checkpoint 保存那一刻 `U_sps` 字段实际对应的阶数——
                Order Continuation 接入多GPU分布式路径后（2026-09-02
                续接，见 gpu_distributed.py::solve 文档），调用方必须传
                `self.current_order`（不是固定的目标阶数），否则爬坡
                阶段中途存的 checkpoint 会出现"metadata 记的阶数与
                `U_sps` 实际形状不符"的错配——与 CPU
                `distributed_save_checkpoint` 同名参数同一处修复同一个
                理由。
            turbulence_model: 湍流模型名（目前恒为 'none'，见 __init__）
            backend: 后端名，默认 'gpu'
            history: 收敛历史（可选）
            target_order: Order Continuation 的最终目标阶数
                （`self.order`）。None（默认，兼容旧调用方）时回退到
                `order` 本身。

        Returns:
            checkpoint 文件路径（仅 root rank 有值）
        """
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_save_checkpoint

        n_local = self.partition.n_local_cells
        self.state.U[:n_local] = cp.asnumpy(self.U_gpu)

        return distributed_save_checkpoint(
            self, output_dir, iteration, input_file, order, turbulence_model, backend,
            history=history, target_order=target_order,
        )

    def load_checkpoint_distributed(self, path: str):
        """分布式 checkpoint 加载（#4）。见 save_checkpoint_distributed
        文档说明同一处修复。加载结果写回 `self.state.U`/`self.U_gpu`
        （local cells 部分）。

        Returns:
            (metadata, iteration)：见 distributed_load_checkpoint 返回值
        """
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint

        U_local, metadata, iteration = distributed_load_checkpoint(path, self)
        n_local = self.partition.n_local_cells
        self.state.U[:n_local] = U_local
        with cp.cuda.Device(self.device_id):
            self.U_gpu = cp.asarray(U_local)
        self.iteration = iteration
        return metadata, iteration
