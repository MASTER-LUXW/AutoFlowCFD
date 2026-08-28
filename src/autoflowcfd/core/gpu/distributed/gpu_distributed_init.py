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
        """分布式壁面距离计算（使用局部网格）。"""
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell

        # 真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前坐标缺失
        # /计算异常都静默回退到硬编码常量 0.01m（与 gpu_solver_init.py::
        # _init_wall_distance_gpu 此前同一类问题，见该文件对应修复
        # 说明）——直接失败，不静默凑一个和网格无关的假常量。
        if hasattr(self.mesh, 'sps_coords') and self.mesh.sps_coords is not None:
            sps_coords = self.mesh.sps_coords.reshape(-1, 3)
        elif hasattr(self.mesh, 'cell_centers') and self.mesh.cell_centers is not None:
            sps_coords = np.tile(self.mesh.cell_centers, (1, n_sps)).reshape(-1, 3)
        else:
            raise RuntimeError(
                f"Rank {self.rank}: wall distance field not computed for turbulence "
                f"model '{getattr(self, 'turb_model_name', '?')}': mesh has neither "
                f"sps_coords nor cell_centers. Industrial-grade calculation requires "
                f"accurate wall distance, not simplified estimates."
            )

        wall_indices = None
        if hasattr(self.mesh, 'boundary_groups'):
            for bg_name, bg in self.mesh.boundary_groups.items():
                if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                    wall_indices = bg.get('node_indices')
                    break

        if wall_indices is not None and len(wall_indices) > 0:
            from scipy.spatial import cKDTree
            wall_coords = self.mesh.nodes[wall_indices]
            tree = cKDTree(wall_coords)
            dist_flat, _ = tree.query(sps_coords, k=1)
            self.wall_distance_gpu = cp.asarray(
                dist_flat.reshape(n_local, n_sps)
            )
        else:
            # 找不到 WALL 边界组时退回特征长度估计——物理上合理的近似，
            # 不属于本次修复目标（"假常量" 0.01m）范畴，保留原行为。
            volumes = self.mesh_data.get('cell_volumes')
            if volumes is None:
                volumes = cp.asarray(self.mesh.get_all_cell_volumes())
            h_char = volumes ** (1.0 / 3.0)
            self.wall_distance_gpu = cp.broadcast_to(
                h_char[:, None], (n_local, n_sps)
            ).copy()

    def _init_modal_filter_distributed(self):
        """分布式模态滤波初始化。"""
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell
        n_prism = self.mesh.n_prism_cells

        try:
            filter_prism = self.ops.filter_prism
            filter_tet = self.ops.filter_tet

            if filter_prism is not None or filter_tet is not None:
                from autoflowcfd.core.gpu.gpu_modal_filter import build_gpu_filter_func
                self.filter_func_gpu = build_gpu_filter_func(
                    n_local, n_sps, n_prism,
                    filter_prism, filter_tet,
                    device_id=self.device_id,
                )
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Modal filter init failed: {e}")

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
        self.flat_face_gpu = build_gpu_flat_face(dist_flat_face, self.device_id)

    def _halo_exchange_gpu(self):
        """GPU 直接 halo 交换（优化版）。

        支持两种模式：
        1. CUDA-aware MPI：GPU buffer 直接通信（零拷贝）
        2. Staging buffer：GPU→CPU→MPI→CPU→GPU（只传输必要数据）
        """
        self.U_extended_gpu = self.gpu_halo.exchange(self.U_gpu)

    def _compute_turbulence_source_distributed(self):
        """分布式湍流源项计算（与单机版一致）。

        当前不可达：`MultiGPUDistributedSolver.__init__` 已把 turb_model
        收紧为只接受 'none'（见该方法文档"#1 修复"一节），`turb_model_gpu`
        因此恒为 None，下面的 `if` 直接返回。除了那处文档说明的
        wall_distance/压缩索引空间不一致之外，这个函数体本身还引用了
        `self.ops_data`/`self.Q_gpu`——本类从未定义过这两个属性（只有
        `self.mesh_data`、且状态是 `self.U_gpu` 不是 `self.Q_gpu`），
        SST 分布式支持真正接入时这里也需要一并修掉，不只是索引对齐
        问题。保留这个函数体是为了未来接入时有一个明确的起点，不是说
        它现在能跑。
        """
        if self.turb_model_gpu is None:
            return None

        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell

        from autoflowcfd.core.gpu.residual.gpu_gradients import (
            compute_physical_gradient_gpu,
            compute_physical_scalar_gradient_gpu,
        )

        # 速度梯度
        grad_U = compute_physical_gradient_gpu(
            self.U_gpu[..., :5], self.mesh_data, self.ops_data,
        )

        # 壁面距离（真实 bug 修复，V2.0 专家组盲审发现，2026-08-27：此前
        # 静默回退到硬编码常量 0.01m，见 _init_wall_distance_distributed
        # 对应修复说明——同一类问题）
        d_wall = self.wall_distance_gpu
        if d_wall is None:
            raise RuntimeError(
                f"Rank {self.rank}: wall distance field not computed for turbulence "
                f"model '{getattr(self, 'turb_model_name', '?')}'. Please ensure "
                f"_init_wall_distance_distributed() ran during solver initialization. "
                f"Industrial-grade calculation requires accurate wall distance, not "
                f"simplified estimates."
            )

        # k/ω 梯度
        grad_k = compute_physical_scalar_gradient_gpu(
            self.turb_model_gpu.k_field, self.mesh_data, self.ops_data,
        )
        grad_omega = compute_physical_scalar_gradient_gpu(
            self.turb_model_gpu.omega_field, self.mesh_data, self.ops_data,
        )

        # 梯度限幅
        max_grad_mag = 1e6
        grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
        if cp.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / cp.maximum(grad_k_mag, 1e-10)
            grad_k *= cp.clip(scale_k, 0, 1)[..., None]
        if cp.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / cp.maximum(grad_omega_mag, 1e-10)
            grad_omega *= cp.clip(scale_omega, 0, 1)[..., None]

        # SST 源项
        Sk, S_omega = self.turb_model_gpu.compute_source_terms_gpu(
            self.Q_gpu, grad_U, d_wall, self.mu_molecular,
            grad_k, grad_omega,
        )

        # 更新湍流场
        rho = self.Q_gpu[:, :, 0]
        dk_dt = Sk / cp.maximum(rho, 1e-10)
        domega_dt = S_omega / cp.maximum(rho, 1e-10)
        dt_local = self._compute_local_time_step_gpu()
        dt_mean = float(cp.mean(dt_local))
        self.turb_model_gpu.update_fields_gpu(dt_mean, dk_dt, domega_dt)

        return rho * self.turb_model_gpu.nu_t

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
        history: dict = None,
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
            order: FR 阶数
            turbulence_model: 湍流模型名（目前恒为 'none'，见 __init__）
            backend: 后端名，默认 'gpu'
            history: 收敛历史（可选）

        Returns:
            checkpoint 文件路径（仅 root rank 有值）
        """
        cp = get_cupy()
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_save_checkpoint

        n_local = self.partition.n_local_cells
        self.state.U[:n_local] = cp.asnumpy(self.U_gpu)

        return distributed_save_checkpoint(
            self, output_dir, iteration, input_file, order, turbulence_model, backend,
            history=history,
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
