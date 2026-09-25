"""AutoFlowCFD V2.0 - 全局网格几何到 local+halo 压缩索引空间的只读视图。

从 `core/gpu/distributed/gpu_distributed.py` 拆出（2026-09-25，项目「单文件不超
500 行」规范）。
"""

import numpy as np


class _CompactMeshDataView:
    """把完整全局 HighOrderMesh 的逐单元几何数据（jacobians/cell_volumes）
    限制+重映射到 local+halo 压缩索引空间（#1，V2.0 专家组盲审第4轮，
    2026-08-28）。

    `GPUArrayManager.upload_mesh_data(mesh, ops)` 只是鸭子类型地读
    `mesh.n_cells`/`n_prism_cells`/`n_sps_per_cell`/`n_points_1d`/
    `jacobians`/`jacobians_fine`/`cell_volumes` 这几个属性，不关心传入的
    是不是真正的 HighOrderMesh——分布式多 GPU 路径下这些逐单元几何数据
    必须按 `distributed_flat_face.py::DistributedFlatFaceGeometry.
    compact_global_ids` 给出的"棱柱在前、四面体在后"local+halo 压缩
    索引空间重新排列（与 `flat_face_gpu.owner_cell`/`neighbor_cell` 用
    的同一套索引空间一致），否则用压缩索引去查全局尺寸的 jacobians/
    cell_volumes 会读到完全不相关单元的几何数据。

    FR 算子（D_3d_tet/prism、over-integration 算子）与单元数量无关、
    只与阶数有关，`upload_mesh_data` 是从 `ops` 参数单独读取的，不受
    这层限制影响，本类不需要处理它们。
    """

    def __init__(self, mesh, compact_global_ids: np.ndarray, n_prism_compact: int):
        n_sps = mesh.n_sps_per_cell
        self.n_cells = len(compact_global_ids)
        self.n_sps_per_cell = n_sps
        self.n_points_1d = mesh.n_points_1d
        self.n_prism_cells = n_prism_compact

        self.jacobians = None
        if mesh.jacobians is not None:
            det_jacs = mesh.jacobians['det_jacs'].reshape(mesh.n_cells, n_sps)
            inv_jacs = mesh.jacobians['inv_jacs'].reshape(mesh.n_cells, n_sps, 3, 3)
            self.jacobians = {
                'det_jacs': det_jacs[compact_global_ids],
                'inv_jacs': inv_jacs[compact_global_ids],
            }

        # 真实 bug 修复（2026-09-02，实现 Order Continuation 时首次真正
        # 端到端构造 `MultiGPUDistributedSolver`——用 numpy-as-cupy 替身
        # 完整走一遍 __init__——才发现）：`GPUArrayManager.upload_mesh_
        # data` 在 `mesh.jacobians_fine is not None` 时无条件读
        # `mesh.n_sps_per_cell_fine`（`array_manager.py:204`），但本类
        # 此前从未把它设成 `self` 的属性（只在下面这个 if 块内部当局部
        # 变量 `n_fine` 用，构造完就丢失）——任何真正启用了过积分
        # （`jacobians_fine`，P>=1 阶数的默认反混叠策略，几乎所有真实
        # 生产网格都会触发）的 `MultiGPUDistributedSolver` 构造都会在
        # `upload_mesh_data` 里 `AttributeError` 崩溃。此前从未被任何
        # 测试捕捉到，是因为所有既有 GPU 分布式测试都只测试更底层的
        # 独立函数（`distributed_compute_les_viscosity` 等价 GPU 函数），
        # 从未真正走过 `MultiGPUDistributedSolver.__init__` 这条完整
        # 构造路径。
        self.n_sps_per_cell_fine = getattr(mesh, 'n_sps_per_cell_fine', None)
        self.jacobians_fine = None
        if getattr(mesh, 'jacobians_fine', None) is not None:
            n_fine = mesh.n_sps_per_cell_fine
            det_jacs_fine = mesh.jacobians_fine['det_jacs'].reshape(mesh.n_cells, n_fine)
            inv_jacs_fine = mesh.jacobians_fine['inv_jacs'].reshape(mesh.n_cells, n_fine, 3, 3)
            self.jacobians_fine = {
                'det_jacs': det_jacs_fine[compact_global_ids],
                'inv_jacs': inv_jacs_fine[compact_global_ids],
            }

        self.cell_volumes = None
        if getattr(mesh, 'cell_volumes', None) is not None:
            self.cell_volumes = mesh.cell_volumes[compact_global_ids]
