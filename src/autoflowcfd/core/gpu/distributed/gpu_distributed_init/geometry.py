"""AutoFlowCFD V2.0 - 多 GPU 分布式的几何/壁距/滤波初始化（其余职责见同目录 mixin）

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_init.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

from autoflowcfd.core.gpu import get_cupy
from .turb_source import _GPUDistributedTurbSourceMixin
from .checkpoint import _GPUDistributedCheckpointMixin


class _GPUDistributedInitMixin(_GPUDistributedTurbSourceMixin, _GPUDistributedCheckpointMixin):
    """MultiGPUDistributedSolver 初始化/I/O 混入。"""

    def _rebuild_partition(self, partition_info):
        """从分区信息字典重建 DistributedPartition。"""
        return partition_info

    def _init_wall_distance_distributed(self):
        """分布式壁面距离（compact 索引空间：local + halo，"棱柱在前"）。

        与 CPU 分布式**同一个函数**（`core/mpi/distributed_turbulence.py::
        compute_distributed_wall_distance`：同一个来源在本 rank compact 解点上
        查询），算完上传 GPU。换阶时 `gpu_distributed_order_continuation` 再调
        一次，用同一个来源重查。

        **2026-09-25 修复**：此前这里是一份独立拷贝——遍历 `mesh.boundary_groups`
        时把数组当字典读（真实网格上构造即崩溃），找不到壁面时退回"单元特征
        长度"。（2026-09-02 那次修掉的"全局 sps_coords 整体 reshape 成 n_local"
        缺陷，在共用函数里天然不存在：它按 compact_global_ids 切片。）
        """
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_turbulence import compute_distributed_wall_distance

        self.wall_distance_gpu = cp.asarray(compute_distributed_wall_distance(
            self.dist_flat_face, self.mesh, getattr(self, "_wall_distance_source", None)))

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

        # **2026-09-18：这里原本套着 `except Exception -> warning`，已删除。**
        # 理由与 `gpu_solver_init.py::_init_modal_filter_gpu` 同一处说明
        # 完全相同，也与本文件 `_init_distributed_face_geometry` 早已删掉
        # 宽 except 的理由相同：本轮新加的硬护栏（`perm` 长度不符、
        # `(neighbor_cell_local<0) != is_boundary` 自洽性、`n_prism` 越界、
        # cupyx 缺 scatter_max）全部落在这个 try 内，会被降级成一条
        # warning + **完全无滤波**继续跑。
        filter_prism = self.ops.filter_prism
        filter_tet = self.ops.filter_tet
        if filter_prism is None and filter_tet is None:
            return

        # 原生排列的 local 段——`self.U_gpu` 所在的空间。
        from autoflowcfd.core.mpi.distributed_flat_face import native_cell_is_prism

        cell_is_prism = native_cell_is_prism(self.dist_flat_face)[:n_local]

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

        **没有 CuPy 时整条门控退回 numpy**（本机与 CI 的 numpy 替身
        端到端测试就走这条）：`build_sensor_gated_filter_func_arrays` 从
        滤波矩阵推断数组模块，传 numpy 进去它就是 numpy 路径，数值与
        CPU 单机一致。**不静默跳过门控** —— 那会让"替身测试通过"与
        "真实 GPU 上门控真的生效"脱钩，而这正是本文件此前那个
        `except Exception -> warning` 兜底造成的问题（它把所有硬护栏
        一起吞了，2026-09-18 删除）。完整论证见
        `core/gpu/device_context.py` 模块文档。
        """
        from autoflowcfd.core.fr_solver.filter import (
            _warn_distributed_face_stencil,
            build_distributed_bounds_conn,
            build_sensor_gated_filter_func_arrays,
        )
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor,
        )
        from autoflowcfd.core.gpu.device_context import (
            ascontiguous_like, device_transfer,
        )

        _dev, _to_dev = device_transfer(self.device_id)

        sensor = resolve_troubled_sensor()
        conn = {}
        if sensor in ("bounds", "both"):
            # 索引换算、两条自洽性护栏、halo 扩展时机、两张边界表的惰性
            # 构造全部在共享实现里（见 `build_distributed_bounds_conn`），
            # CPU MPI 走的是同一条 —— 两条后端的掩码因此不可能悄悄分叉。
            #
            # 惰性取 provider 在这条后端上**是必需的**：本方法在
            # `self.boundary_ghost_provider` 建好之前就被调用
            # （gpu_distributed.py 里滤波初始化在 provider 构造之前）。
            with _dev:
                conn = build_distributed_bounds_conn(
                    self.dist_flat_face,
                    self.partition.n_total_cells,
                    lambda: self.gpu_halo,
                    lambda: getattr(self, "boundary_ghost_provider", None),
                    self.freestream,
                    to_device=_to_dev,
                    ascontiguous=ascontiguous_like)

        order = int(getattr(self, "current_order", self.order))
        with _dev:
            fp = (filter_prism if hasattr(filter_prism, "device")
                  else _to_dev(filter_prism))
            ft = (filter_tet if hasattr(filter_tet, "device")
                  else _to_dev(filter_tet))
            # 顶点邻域模板（BJ 判据用）在**分布式**路径上还不可用 ——
            # 如实记录为未完成项，不静默当成已修。
            #
            # 缺陷：面邻居在三维四面体上只有 4 个，其单元均值不能把本
            # 单元夹住，于是 O(h|grad u|) 的合法光滑变化被当成越界。
            # 实测标记比例在三档加密上**恒为 100%**（不收敛），换顶点
            # 邻域模板后 100% -> 25% -> 6.18%。完整数据见
            # `core/fr_operators/vertex_stencil.py` 模块文档。
            #
            # 为什么这里不接顶点模板：共享同一顶点的两个单元通过面邻接
            # 可能相隔 **2 个以上**面跳，而本项目的 halo 是 1 层面邻居。
            # 不完整的顶点邻域会让包络在分区边界上变窄，于是**同一算例
            # 换 rank 数得到不同的掩码** —— 结果依赖分区，这是求解器
            # 不可接受的（正是 `test_sensor_gate_distributed.py` 钉住的
            # 那条性质）。
            #
            # 为什么不硬失败：面模板在分布式上**是分区独立的**、现在就
            # 能用，只是带着上面那个过度标记的缺陷（与单机在本次修复前
            # 同一个缺陷）。把它改成崩溃等于删掉一个可用功能。
            #
            # 要在分布式上用顶点模板，需要的是一次**按顶点**的归约交换
            # （每个 rank 先算本地 node_max/node_min，再对共享顶点做
            # allreduce，然后散射回单元），那是一条与现有按单元的 halo
            # 交换不同的通信模式，属于独立一项。
            _warn_distributed_face_stencil(sensor)
            return build_sensor_gated_filter_func_arrays(
                n_local, n_sps, order, fp, ft,
                cell_is_prism=_to_dev(cell_is_prism),
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

    def cleanup(self):
        """释放 GPU 资源。"""
        self.array_mgr.cleanup()
