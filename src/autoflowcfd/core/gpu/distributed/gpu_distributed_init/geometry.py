"""AutoFlowCFD V2.0 - 多 GPU 分布式的几何/壁距/滤波初始化（其余职责见同目录 mixin）

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_init/` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

from autoflowcfd.core.gpu import get_cupy
from .turb_source import _GPUDistributedTurbSourceMixin
from .checkpoint import _GPUDistributedCheckpointMixin


class _GPUDistributedInitMixin(_GPUDistributedTurbSourceMixin, _GPUDistributedCheckpointMixin):
    """MultiGPUDistributedSolver 初始化/I/O 混入。"""

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
        from autoflowcfd.core.mpi.distributed_turbulence import (
            apply_distributed_wall_distance, compute_distributed_wall_distance,
        )

        self.wall_distance_gpu = cp.asarray(compute_distributed_wall_distance(
            self.dist_flat_face, self.mesh, getattr(self, "_wall_distance_source", None)))
        # omega 上界按全局最小壁距重定（与 CPU 分布式同一函数）
        apply_distributed_wall_distance(getattr(self, "turb_model_gpu", None), self.wall_distance_gpu, self)

    def _init_distributed_face_geometry(self):
        """初始化分布式面几何（GPU 版）。

        此前这里的调用方式与 build_distributed_flat_face 的真实签名
        `(mesh, ops, partition)` 不匹配——旧代码传的是
        `(flat_face, self.partition, self.rank, self.n_ranks)`（4 个
        位置参数，且第一个是已经算好的全局 FlatFaceGeometry 而不是
        mesh；该函数内部本就会自己调用 get_flat_face_geometry(mesh,
        ops)，不需要调用方先算一遍）——必然 TypeError，且被外层宽
        `except Exception` 吞掉，`self.flat_face_gpu` 静默设为 None。
        下游 gpu_inviscid/ 在 flat_face_gpu 为 None 时会用**未分区的
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
        # FlatFaceGeometry，见 gpu_inviscid/::compute_inviscid_residual_
        # fr_gpu 的 flat_face_cpu 参数文档），因此这里保留一份引用。
        self.dist_flat_face = dist_flat_face
        from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
        # 传 `dist_flat_face.base_flat`（按 local+halo 压缩索引空间构造好的完整 `FlatFaceGeometry`），不是
        # `DistributedFlatFaceGeometry` 包装对象：后者只转发部分字段，`GPUFlatFaceGeometry` 要的 `owner_cell` 等
        # 字段它没有（`tests/unit/test_distributed_flat_face_gpu_compat.py`）。
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
