"""AutoFlowCFD V2.0 - 从完全分布式包构造多 GPU solver"""

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.mpi.distributed_solver.assembly import settings_from_package

from .upload import _upload_wall_geometry_compact


def build_multi_gpu_solver_from_fully_distributed_package(
    cls, package: dict, n_ranks: int,
    device_id=None, rank=None, root_context=None,
):
    """真正的"完全分布式加载"构造入口（见模块文档）。

    Args:
        cls: `MultiGPUDistributedSolver`（由该类的 `from_fully_
            distributed_package` classmethod 传入 `cls`，本函数只是把
            实现拆到独立模块控制单文件行数，与 CPU
            `DistributedFRSolver.from_fully_distributed_package` 同一个
            拆分动机）
        package: `build_fully_distributed_rank_package`（或经
            `distributed_mesh_load_v2` MPI 收发后本 rank 收到的那一份）
            的返回值——与 CPU 路径共用同一份数据结构。
        n_ranks: MPI rank 总数
        device_id: GPU 设备号（None 时按 rank 轮询分配，与主 `__init__`
            同一个默认策略）
        rank: 当前 rank（None 时从 MPI 获取）
        root_context: 仅 root rank 需要非 None——`distributed_mesh_
            load_v2` 返回的第二个值，供 Order Continuation 重新分发用，
            见模块文档。

    Returns:
        MultiGPUDistributedSolver 实例
    """
    from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
    from autoflowcfd.core.turbulence.registry import has_transport_equations
    from autoflowcfd.fr.operators import generate_fr_operators

    self = cls.__new__(cls)
    self._setup_identity_and_device(rank, n_ranks, device_id, package['order'], settings_from_package(package))
    self._is_fully_distributed = True
    self.cell_partition = None  # 本 rank 从未持有完整全局网格，见类文档
    self._root_context = root_context

    precompacted_mesh = package['precompacted_mesh']
    self.mesh = precompacted_mesh
    # 每个 rank 本地重新生成算子（纯函数，只依赖阶数），不随包 pickle 传输
    self.ops = generate_fr_operators(package['order'])
    self.partition = package['partition']
    # 隐式步的块 Jacobi 着色（全局一致，root 随包下发；非隐式时为 None）
    self._block_jacobi_colors_local = package['cell_colors']
    self._coupling_colors_local = package['cell_colors_d2']
    # 面几何与压缩索引空间的网格几何都已由 root 按本 rank 切好，只需上传设备
    self.dist_flat_face = package['dist_fc']
    self.flat_face_gpu = build_gpu_flat_face(self.dist_flat_face.base_flat, self.device_id)
    n_sps = precompacted_mesh.n_sps_per_cell
    self._setup_compact_geometry(precompacted_mesh, self.ops, n_sps)
    self._setup_time_integration(settings_from_package(package))
    self._setup_turbulence_models(n_sps)
    if has_transport_equations(self.turb_model_name):
        # 壁距与 DES 网格尺度由 root 在完整网格上算好、切成本 rank 的紧凑索引空间随包下发
        _upload_wall_geometry_compact(self, package, get_cupy(), self.device_id)
    # 边界幽灵态：root 已按全局网格构建并把 group_code 重映射到本 rank 的压缩面编号
    self._setup_state_and_boundary(package['boundary_ghost_provider'])
    self._finish_setup("fully-distributed mode: only root rank loaded the full mesh")
    return self
