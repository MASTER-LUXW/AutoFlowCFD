"""
AutoFlowCFD V2.0 - 多 GPU + MPI 分布式求解器

将 GPU 计算与 MPI 域分解结合：每个 MPI rank 使用一块 GPU 进行计算。

设计：
- 继承 DistributedFRSolver 的 MPI 基础设施（分区、halo 交换、全局归约）
- 残差计算使用 GPU 版本（gpu_inviscid.py / gpu_viscous.py）
- 数据常驻各 rank 的 GPU 显存
- GPU 直接 Halo 交换（gpu_halo_exchange.py）：
  - CUDA-aware MPI：零拷贝 GPU↔GPU
  - 非 CUDA-aware：staging buffer 优化（只传输 send/recv 列表数据）
- SSP-RK2/RK3 时间推进：每个 stage 执行 halo 交换 + 残差评估
- 全局残差归约：MPI Allreduce

使用:
    mpirun -np 4 autoflowcfd solve steady <grid> --backend gpu --multi-gpu
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger

from autoflowcfd.core.gpu import gpu_available, get_cupy
from autoflowcfd.core.gpu.array_manager import GPUArrayManager
from autoflowcfd.core.gpu.gpu_time_integration import (
    GPUTimeIntegrator,
    enforce_positivity_gpu,
    compute_local_cfl_step_gpu,
)
from autoflowcfd.core.mpi import get_rank, get_size, is_root, mpi_available
from autoflowcfd.core.mpi.partition import (
    partition_mesh, build_distributed_partition, DistributedPartition
)
from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.distributed_flat_face import (
    DistributedFlatFaceGeometry, build_distributed_flat_face
)
from autoflowcfd.core.mpi.comm import allreduce_sum, allreduce_min, barrier
from autoflowcfd.core.gpu.distributed.gpu_distributed_init import _GPUDistributedInitMixin


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


class MultiGPUDistributedSolver(_GPUDistributedInitMixin):
    """多 GPU + MPI 分布式求解器。

    每个 MPI rank 绑定一块 GPU，使用 GPU 进行所有计算，
    通过 MPI 进行 halo 交换和全局归约。

    Attributes:
        partition: 本 rank 的分区信息
        halo_exchange: halo 交换管理器
        array_mgr: GPU 数组管理器
        time_integrator: GPU 时间积分器
        U_gpu: 本 rank 的守恒变量（GPU 常驻）
        rank: 当前 MPI rank
        n_ranks: 总 rank 数
        device_id: 本 rank 使用的 GPU 设备 ID
    """

    def __init__(
        self,
        mesh,
        ops,
        n_ranks: int,
        face_connectivity_data=None,
        partition_info=None,
        rank: Optional[int] = None,
        device_id: Optional[int] = None,
        time_scheme: str = "ssp_rk3",
        cfl: float = 1.0,
        mu_molecular: float = 1.8e-5,
        rho_inf: float = 1.225,
        vel_inf: float = 33.33,
        p_inf: float = 101325.0,
        turb_model: str = "NONE",
    ):
        """初始化多 GPU 分布式求解器。

        Args:
            mesh: HighOrderMesh（局部网格或完整网格）
            ops: FROperators
            n_ranks: MPI rank 总数
            face_connectivity_data: 局部面连接关系数据（分布式加载模式）
            partition_info: 分区信息（分布式加载模式）
            rank: 当前 rank（默认从 MPI 获取）
            device_id: GPU 设备 ID（默认 rank % n_gpus）
            time_scheme: 时间积分方案
            cfl: CFL 数
            mu_molecular: 分子动力粘度
            rho_inf, vel_inf, p_inf: 自由来流条件
        """
        if not gpu_available:
            raise RuntimeError("CuPy required for multi-GPU solver")
        if not mpi_available:
            raise RuntimeError("MPI required for distributed solver")

        cp = get_cupy()
        # 真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前这里对
        # turb_model 不做任何校验，SST 之外的任何值（包括 DDES/WMLES/LES）
        # 会静默跳过下面的 GPUTurbulenceSST 初始化、`self.turb_model_gpu`
        # 保持 None，等价于悄悄退化成层流，且没有任何报错/警告——CLI
        # 侧同样从未真正传过 turb_model（另一处已修复的独立 bug），两者
        # 叠加此前 `--multi-gpu --turbulence-model ddes` 之类的请求会
        # 完全静默地跑出层流结果。
        #
        # 收紧为只接受 'none'（V2.0 专家组盲审第4轮，2026-08-28，#1 修复的
        # 一部分）：此前这里允许 'sst'，但 #1 修复过程中发现
        # `_compute_turbulence_source_distributed` 本身有更深的架构缺口——
        # `self.mesh_data`/`self.flat_face_gpu` 现在是按 local+halo 压缩
        # 索引空间构造的（见 __init__ 下方 mesh_data 构造处的说明），但
        # `self.wall_distance_gpu`（_init_wall_distance_distributed 构造）
        # 只有 n_local 大小、且是 partition.local_cells 自身顺序，与压缩
        # 索引空间是两套不同的排列——湍流源项计算需要把 grad_U 等场也按
        # 压缩索引空间对齐（与残差计算同一个道理，见 compute_inviscid_
        # residual_gpu 的 perm/inv_perm 重排文档），这部分尚未实现，
        # 勉强让它跑起来只会得到看似正常、实际物理错误的湍流场（读到
        # 错位的壁面距离/梯度）。与 DistributedFRSolver（CPU MPI 路径）
        # 对不支持湍流模型的处理方式保持一致：显式拒绝，而不是让一个
        # 尚未验证正确的路径悄悄跑完。
        if turb_model is not None and str(turb_model).upper() != "NONE":
            raise NotImplementedError(
                f"MultiGPUDistributedSolver（--multi-gpu）目前只支持 "
                f"turbulence_model='none'，收到的是 '{turb_model}'。SST 分布式 "
                f"湍流源项计算还需要把 wall_distance/梯度场对齐到 local+halo "
                f"压缩索引空间（#1 修复已解决平均流残差的这个问题，湍流输运"
                f"部分尚未做，见本方法文档），请显式传入 --turbulence-model "
                f"none（CLI `solve steady/transient` 的 --turbulence-model "
                f"默认值是 'sst'，不显式覆盖就会触发本错误）或改用单机 "
                f"GPU/CPU 后端。"
            )

        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        self.mesh = mesh
        self.ops = ops
        self.mu_molecular = mu_molecular
        # mach_ref：与 CPU 版 FRSolver.__init__（fr_solver/solver.py）
        # 同一套计算方式/同一个用途，见该文件对应注释。物理下限钳制同样与
        # CPU 版镜像同步（2026-08-26，P2 发散专项）：低于 0.1 的参考马赫数会让
        # AUSM+up Mp 压差扩散项的 1/mach_ref² 放大压倒显式推进稳定性，
        # 完整推导/实证标定记录见 fr_solver/solver.py::_MACH_REF_FLOOR。
        mach_ref = vel_inf / np.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10))
        mach_ref = max(mach_ref, 0.1)
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf, "mach_ref": mach_ref}
        self.turb_model_name = turb_model

        # GPU 设备选择：默认 round-robin 分配
        if device_id is None:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            device_id = self.rank % n_gpus
        self.device_id = device_id

        # 切换到指定 GPU
        with cp.cuda.Device(device_id):
            logger.info(f"Rank {self.rank} using GPU device {device_id}")

        # 初始化 GPU 数组管理器
        self.array_mgr = GPUArrayManager(device_id=device_id)

        # 分区
        if partition_info is not None:
            # 分布式加载模式：已有分区信息。这里 mesh 是局部网格而非全局
            # 网格，build_distributed_flat_face 的 halo 扩展步骤依赖的
            # get_flat_face_geometry(mesh, ops) 全局面几何假设在这条路径
            # 下不成立（更深层、超出本次修复范围的架构问题），因此不设
            # self.cell_partition，让 _init_distributed_face_geometry
            # 退回到"缺失依赖时报错"而不是尝试扩展。
            self.partition = self._rebuild_partition(partition_info)
            self._using_distributed_mesh = True
            self.cell_partition = None
        else:
            # 传统模式：所有 rank 有完整网格，执行分区
            fc = mesh.face_connectivity
            if self.rank == 0:
                cell_partition = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells)
            else:
                cell_partition = None
            from autoflowcfd.core.mpi.comm import bcast_from_root
            if n_ranks > 1:
                cell_partition = bcast_from_root(cell_partition)
            self.partition = build_distributed_partition(
                fc, cell_partition, self.rank, n_ranks
            )
            self._using_distributed_mesh = False
            # 供 _init_distributed_face_geometry 扩展 halo 层覆盖 FR Flux
            # Point 多源交叉插值依赖用（见该方法与 build_distributed_flat_face 文档）。
            self.cell_partition = cell_partition

        # n_sps 必须先于下面的 GPUHaloExchange 构造赋值——此前这里是反的：
        # GPUHaloExchange(..., n_sps=n_sps, ...) 在 `n_sps = mesh.
        # n_sps_per_cell` 赋值语句之前引用 n_sps，Python 函数作用域规则下
        # 是必然的 UnboundLocalError，MultiGPUDistributedSolver.__init__
        # 100% 崩溃且零测试覆盖（V2.0 专家组评审逐行核实）。
        n_sps = mesh.n_sps_per_cell
        n_local = self.partition.n_local_cells

        # 初始化局部面几何——必须排在 GPUHaloExchange/DistributedFRState/
        # mesh_data 上传**之前**：
        # 1. 当 self.cell_partition 非空时，这一步内部会调用
        #    extend_halo_for_flux_point_cross_references 原地扩展
        #    self.partition 的 halo_cells/send_lists/recv_lists（覆盖 FR
        #    Flux Point 多源交叉插值依赖，真实 cube_demo 网格上 16%~19%
        #    的单元需要这种扩展，不是边缘情况）。GPUHaloExchange/
        #    DistributedFRState 会按*构造时刻*的 send_lists/recv_lists
        #    大小预分配定长 buffer——如果它们先构造、面几何扩展后发生，
        #    这些 buffer 要么大小对不上（形状不匹配崩溃），要么根本没有
        #    新增邻居 rank 的 key（KeyError，或更隐蔽地静默跳过、留下
        #    未初始化的 halo 数据）。
        # 2. #1（V2.0 专家组盲审第4轮，2026-08-28）：mesh_data 现在必须
        #    按 self.dist_flat_face.compact_global_ids 给出的"棱柱在前"
        #    local+halo 压缩索引空间构造（不是直接用完整全局 mesh 上传），
        #    这个数组正是本步骤的产物，因此 mesh_data 上传必须排在
        #    本步骤之后——见下方 mesh_data 构造处的完整说明。
        self.flat_face_gpu = None
        self.dist_flat_face = None
        self._init_distributed_face_geometry()

        # #1（2026-08-28）：mesh_data 此前直接从完整全局 mesh 上传
        # （n_cells=全局单元数，jacobians/cell_volumes 按全局单元编号
        # 索引），但残差 kernel（gpu_inviscid.py/gpu_viscous.py/
        # gpu_gradients.py）读取的 Q/U 数组只有 local+halo 压缩索引空间
        # 大小、且按"棱柱在前"排列（见 dist_flat_face.compact_global_ids/
        # base_flat.n_prism 文档）——用全局尺寸的 mesh_data 配合压缩索引
        # 空间的单元编号去查 jacobians/cell_volumes，会读到完全不相关
        # 单元的几何数据。改为用 _CompactMeshDataView 把全局 mesh 的
        # 逐单元几何数据限制+重映射到这个压缩索引空间，再上传——FR 算子
        # （D_3d_tet/prism、over-integration 算子）与单元数量无关，从
        # `ops` 读取，不受这层限制影响，upload_mesh_data 本身不需要改。
        compact_mesh_view = _CompactMeshDataView(
            mesh, self.dist_flat_face.compact_global_ids, self.dist_flat_face.base_flat.n_prism,
        )
        self.mesh_data = self.array_mgr.upload_mesh_data(compact_mesh_view, ops)
        self.n_compact = compact_mesh_view.n_cells
        # perm/inv_perm 常驻 GPU，避免每步残差计算都重新上传（见
        # compute_inviscid_residual_gpu/compute_viscous_residual_gpu 对
        # 它们的消费方式）。
        self._perm_gpu = cp.asarray(self.dist_flat_face.perm)
        self._inv_perm_gpu = cp.asarray(self.dist_flat_face.inv_perm)

        # GPU 直接 Halo 交换（支持 CUDA-aware MPI 和 staging buffer 两种模式）
        self.gpu_halo = GPUHaloExchange(
            self.partition, n_sps=n_sps, n_vars=5, device_id=device_id
        )
        # 参数顺序修复：DistributedFRState.__init__ 真实签名是
        # (partition, n_sps, n_vars)，此前这里按 (n_cells, n_sps, 5,
        # partition) 四个位置参数传入，必然 TypeError——
        # MultiGPUDistributedSolver 此前从未真正构造成功过（第四次评审
        # 第二轮复核发现，与上面的 UnboundLocalError 修复是两个独立
        # 的、相邻的崩溃 bug）。
        self.state = DistributedFRState(self.partition, n_sps, 5)

        # 时间积分器
        self.time_integrator = GPUTimeIntegrator(scheme=time_scheme, cfl=cfl)

        # 初始化 GPU 湍流模型（与单机版一致）
        self.turb_model_gpu = None
        if turb_model == "SST":
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
            n_local_cells = self.partition.n_local_cells
            self.turb_model_gpu = GPUTurbulenceSST(n_local_cells, n_sps, device_id)
            logger.info(f"Rank {self.rank}: GPU SST model initialized")

        # 预计算壁面距离
        self.wall_distance_gpu = None
        if self.turb_model_gpu is not None:
            self._init_wall_distance_distributed()

        # 初始化 GPU 模态滤波
        self.filter_func_gpu = None
        self._init_modal_filter_distributed()

        # 初始化 GPU 上的求解状态。
        #
        # #1（2026-08-28）：此前这里恒按全局 n_cells 分配——每个 rank 都
        # 独立持有并演化一份全量状态，不是真正的"只存 local cells"（见
        # gpu_distributed_init.py 此前的文档说明）。现在按 partition.
        # local_cells 自身顺序（halo 交换协议的原生排列，与 self.gpu_halo/
        # self.state 一致）只分配 n_local 个单元——这是本 rank 真正拥有、
        # 会被时间推进更新的状态；参与残差计算所需的 halo 邻居数据通过
        # halo 交换实时获取（见 compute_inviscid_residual_gpu 文档），
        # 不常驻在 self.U_gpu 里。
        with cp.cuda.Device(device_id):
            self.U_gpu = cp.zeros((n_local, n_sps, 5), dtype=cp.float64)
            self.U_gpu[:, :, 0] = rho_inf
            self.U_gpu[:, :, 1] = rho_inf * vel_inf
            self.U_gpu[:, :, 4] = p_inf / (1.4 - 1.0) + 0.5 * rho_inf * vel_inf**2

        # boundary_ghost_provider（#3，2026-08-28）：此前完全没有这个
        # 机制——compute_inviscid_residual_gpu/compute_viscous_residual_gpu
        # 调用底层函数时不传 boundary_ghost_provider，边界面（WALL/INLET/
        # OUTLET/FARFIELD/SYMMETRY）全部退化成"镜像内部值"（等价于对
        # 边界不可见），与单机 GPUFRSolver（gpu_solver.py）已有的正确行为
        # 不一致。镜像 GPUFRSolver 的构造模式：真正构建一个反映边界条件
        # 的 ghost provider，而不是让调用方各自传 None 退化成
        # DefaultGhostProvider。
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        self.wmles_model = None  # build_boundary_ghost_provider 用 getattr 防御性读取
        self._turbulence_intensity = 0.01
        self._viscosity_ratio = 5.0
        self._sem_num_eddies = 200
        self.boundary_ghost_provider = build_boundary_ghost_provider(self, bc_overrides={})

        self.residual_history = []
        self.iteration = 0

        barrier()
        if is_root():
            logger.info(
                f"MultiGPUDistributedSolver initialized: {n_ranks} ranks, "
                f"{n_local} local cells/rank (+{self.partition.n_halo} halo)"
            )
            print(f"✅ MultiGPUDistributedSolver Ready:")
            print(f"   Ranks: {n_ranks}, Cells/rank: {n_local} (+{self.partition.n_halo} halo)")
            print(f"   GPU device: {device_id} per rank")

    def _halo_exchange_gpu(self):
        """GPU 直接 halo 交换（优化版）。

        支持两种模式：
        1. CUDA-aware MPI：GPU buffer 直接通信（零拷贝）
        2. Staging buffer：GPU→CPU→MPI→CPU→GPU（只传输必要数据）
        """
        self.U_extended_gpu = self.gpu_halo.exchange(self.U_gpu)

    def _permute_to_compact(self, U_native):
        """把 halo 交换协议原生排列（local在前、halo在后）的场数组重排到
        本类残差计算实际使用的"棱柱在前、四面体在后"压缩索引空间——见
        distributed_flat_face.py::DistributedFlatFaceGeometry.perm 文档。
        """
        return U_native[self._perm_gpu]

    def _unpermute_from_compact(self, field_compact):
        """`_permute_to_compact` 的逆操作：把"棱柱在前"压缩索引空间的
        结果换回 halo 交换协议原生排列，供 `[:n_local]` 切片取出本 rank
        真正拥有的 local cells 结果。"""
        return field_compact[self._inv_perm_gpu]

    def compute_inviscid_residual_gpu(self):
        """GPU 计算分布式无粘残差。

        #1（2026-08-28）：此前直接把 `self.U_gpu`（当时误按全局单元数
        分配）传给底层函数；现在 `self.U_gpu` 只有 n_local 个单元，
        真正参与残差计算（含分区边界处需要 halo 邻居数据的界面项）必须
        用 halo 交换后的扩展数组 `self.U_extended_gpu`（`_halo_exchange_
        gpu()` 产出，此前算出来后从未被消费——见该方法与
        `_compute_total_residual_gpu` 的调用关系）。

        halo 交换产出的 `U_extended_gpu` 是"local在前、halo在后"的原生
        排列，但 `self.mesh_data`/`self.flat_face_gpu` 是按"棱柱在前、
        四面体在后"的压缩索引空间构造的（体积项切片需要，见
        distributed_flat_face.py 模块文档）——两者不一致，必须先用
        `self._perm_gpu` 把 `U_extended_gpu` 重排到压缩索引空间，残差
        算完后再用 `self._inv_perm_gpu` 换回原生排列，才能在最后
        `[:n_local]` 切片时取到与 `self.U_gpu`（原生排列）对齐的本 rank
        local cells 结果。
        """
        U_compact = self._permute_to_compact(self.U_extended_gpu)
        from autoflowcfd.core.gpu.residual.gpu_inviscid import compute_inviscid_residual_fr_gpu
        residual_compact = compute_inviscid_residual_fr_gpu(
            U_compact, self.mesh, self.ops,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mesh_data=self.mesh_data,
            ops_data=self.mesh_data,
            flat_face_gpu=self.flat_face_gpu,
            flat_face_cpu=self.dist_flat_face.base_flat,
            device_id=self.device_id,
            mach_ref=self.freestream["mach_ref"],
        )
        residual_native = self._unpermute_from_compact(residual_compact)
        return residual_native[: self.partition.n_local_cells]

    def compute_viscous_residual_gpu(self, mu_t_field=None):
        """GPU 计算分布式粘性残差。

        mu_t_field: 湍流涡粘度场（可选）——此前本方法签名只有 self，
        但调用方 step() 以 mu_t_field=mu_t_field 关键字调用它，签名/
        调用不匹配，必然 TypeError（V2.0 专家组评审逐行核实）。目前
        __init__ 已把 turb_model 收紧为只接受 'none'，这个参数恒为 None，
        见 __init__ 里对应的 NotImplementedError 说明。

        halo 交换/压缩索引空间重排逻辑与 compute_inviscid_residual_gpu
        完全一致，见该方法文档。
        """
        U_compact = self._permute_to_compact(self.U_extended_gpu)
        from autoflowcfd.core.gpu.residual.gpu_viscous import compute_viscous_residual_fr_gpu
        residual_compact = compute_viscous_residual_fr_gpu(
            U_compact, self.mesh, self.ops,
            mu=self.mu_molecular,
            mu_t_field=mu_t_field,
            boundary_ghost_provider=self.boundary_ghost_provider,
            mesh_data=self.mesh_data,
            ops_data=self.mesh_data,
            flat_face_gpu=self.flat_face_gpu,
            flat_face_cpu=self.dist_flat_face.base_flat,
            device_id=self.device_id,
        )
        residual_native = self._unpermute_from_compact(residual_compact)
        return residual_native[: self.partition.n_local_cells]

    def _compute_local_time_step_gpu(self):
        """GPU 计算局部 CFL 步长。"""
        cp = get_cupy()
        fc = self.mesh.face_connectivity
        n_faces = fc.n_faces

        owner_cell = cp.asarray(fc.owner_cell)
        neighbor_cell = cp.asarray(
            np.where(fc.is_boundary, 0, fc.neighbor_cell)
        )
        is_boundary = cp.asarray(fc.is_boundary)

        # 面法向和面积：每步热路径性能修复，理由/验证方式同
        # gpu_solver.py::_compute_local_time_step_gpu（同一个真实复现、
        # 同一处遗漏，见该方法文档）。
        from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
        normal, area_w = _extract_p0_face_geometry(self.mesh.face_flux_points, fc, n_faces)
        normals_gpu = cp.asarray(normal)
        areas_gpu = cp.asarray(area_w)

        cell_volumes = self.mesh_data.get('cell_volumes')
        if cell_volumes is None:
            cell_volumes = cp.asarray(self.mesh.get_all_cell_volumes())

        # 几何/度量 CFL 限制（与 gpu_solver.py::_compute_local_time_step_gpu
        # 同一机制，见 compute_local_cfl_step_gpu 参数文档；此前 GPU 分布式
        # 路径完全没有这一限制，可能重新触发 CPU 侧已修复过的坍缩坐标
        # 度量刚性发散）。本路径只用 SP0（与上方 self.U_gpu 直接使用、不
        # 按 SP 循环的既有实现保持一致，不在此扩大范围引入 per-SP 循环）。
        det_jacs_gpu = self.mesh_data.get('det_jacs')
        adj_j_gpu = self.mesh_data.get('adj_j')
        det_jacs_sp0 = det_jacs_gpu[:, 0] if det_jacs_gpu is not None else None
        metric_flux_scale_sp0 = None
        if adj_j_gpu is not None:
            metric_flux_scale_gpu = getattr(self, '_metric_flux_scale_gpu_cache', None)
            # 形状校验（不能只判断 is None）：见 gpu_solver.py 同名缓存的
            # 第四次评审第二轮复核说明，CPU 侧 _get_metric_flux_scale
            # 曾因漏比对 n_sps 维度复现过跨阶数切换后返回陈旧缓存的真实
            # bug，这里保持同等防御水位。
            expected_shape = adj_j_gpu.shape[:2]
            if metric_flux_scale_gpu is None or metric_flux_scale_gpu.shape != expected_shape:
                adj_row_norms = cp.linalg.norm(adj_j_gpu, axis=-1)
                metric_flux_scale_gpu = cp.sum(adj_row_norms, axis=-1)
                self._metric_flux_scale_gpu_cache = metric_flux_scale_gpu
            metric_flux_scale_sp0 = metric_flux_scale_gpu[:, 0]

        return compute_local_cfl_step_gpu(
            self.U_gpu, cell_volumes,
            owner_cell, neighbor_cell, is_boundary,
            normals_gpu, areas_gpu,
            None, None,
            cfl=self.time_integrator.cfl,
            poly_order=getattr(self, "order", 0),
            det_jacs_sp=det_jacs_sp0,
            metric_flux_scale_sp=metric_flux_scale_sp0,
        )

    def _compute_total_residual_gpu(self, mu_t_field=None):
        """计算总残差（无粘 + 粘性），先执行 halo 交换。

        Args:
            mu_t_field: 动力涡粘度 (n_cells, n_sps) CuPy 数组（可选）
        """
        self._halo_exchange_gpu()
        inv_res = self.compute_inviscid_residual_gpu()
        visc_res = self.compute_viscous_residual_gpu(mu_t_field=mu_t_field)
        return inv_res + visc_res

    def step(self, dt: float = 0.0) -> float:
        """执行一个分布式时间步（SSP-RK 多 stage）。

        流程（SSP-RK3 为例，每个 stage 都重新计算残差+halo 交换）：
        Stage 0: 计算初始残差 R(U^n)
        Stage 1: U^(1) = U^n + dt*L(U^n); halo 交换; 计算 R(U^(1))
        Stage 2: U^(2) = 3/4*U^n + 1/4*(U^(1) + dt*L(U^(1))); halo; R(U^(2))
        Stage 3: U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt*L(U^(2)))

        对于 SSP-RK2 和 Forward Euler 类似处理。

        Args:
            dt: 时间步长

        Returns:
            residual_norm: 全局残差范数
        """
        cp = get_cupy()
        n_local = self.partition.n_local_cells
        n_sps = self.mesh.n_sps_per_cell

        # #1（2026-08-28）：此前这里用 self.mesh.n_cells（全局单元数）
        # 展平/reshape self.U_gpu——self.U_gpu 现在只有 n_local 个单元，
        # 用全局尺寸 reshape 会形状不匹配崩溃。改为统一使用 n_local。
        #
        # 局部 CFL 步长：与 CPU 版 DistributedFRSolver.step() 同一个
        # 已明确接受的简化（该文件模块文档："分布式路径目前用全局固定
        # 步长，不做单机路径那种逐 cell 局部 CFL 时间步"）——不调用
        # `_compute_local_time_step_gpu()`：那个方法读 `self.mesh.
        # face_connectivity`（完整全局面连接关系，owner/neighbor 是
        # 全局单元编号）去索引 `self.U_gpu`（现在是 n_local 大小、
        # partition.local_cells 自身顺序），这是与本次 #1 修复同一类的
        # 全局/local+halo 压缩索引空间不一致问题，尚未解决（见该方法
        # 文档新增的说明）；直接复用调用方传入的 `dt`（物理时间步长），
        # 对所有 rank/cell 一致，不做自适应步长，是与 CPU 分布式路径
        # 一致的、已被接受的简化，不是本次新引入的简化。
        dt_flat = cp.full((n_local * n_sps, 1), dt, dtype=cp.float64)

        # RK 系数表
        scheme = self.time_integrator.scheme
        table = self.time_integrator._table
        alpha = table["alpha"]
        beta = table["beta"]
        n_stages = table["stages"]

        # 湍流源项求值（算子分裂，每个 step 开始时计算一次）——turb_model
        # 已在 __init__ 收紧为只接受 'none'，这里恒返回 None。
        mu_t_field = self._compute_turbulence_source_distributed()

        U_flat = self.U_gpu.reshape(n_local * n_sps, 5)
        U0 = U_flat.copy()

        # === Stage 0: 初始残差 ===
        res0 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res0_flat = res0.reshape(n_local * n_sps, 5)
        L0 = -res0_flat

        # === Stage 1 ===
        U_stage1 = U0 + dt_flat * L0
        enforce_positivity_gpu(U_stage1)
        # 模态滤波
        if self.filter_func_gpu is not None:
            U_stage1 = self.filter_func_gpu(U_stage1)
        self.U_gpu = U_stage1.reshape(n_local, n_sps, 5)

        if n_stages == 1:
            residual_norm = self._global_residual_norm(res0_flat)
            self.residual_history.append(residual_norm)
            self.iteration += 1
            return residual_norm

        # === Stage 2 ===
        res1 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res1_flat = res1.reshape(n_local * n_sps, 5)
        L1 = -res1_flat
        U_stage2 = (alpha[1][0] * U0 + alpha[1][1] * U_stage1 + beta[1] * dt_flat * L1)
        enforce_positivity_gpu(U_stage2)
        if self.filter_func_gpu is not None:
            U_stage2 = self.filter_func_gpu(U_stage2)
        self.U_gpu = U_stage2.reshape(n_local, n_sps, 5)

        if n_stages == 2:
            residual_norm = self._global_residual_norm(res1_flat)
            self.residual_history.append(residual_norm)
            self.iteration += 1
            return residual_norm

        # === Stage 3 (RK3) ===
        res2 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res2_flat = res2.reshape(n_local * n_sps, 5)
        L2 = -res2_flat
        U_stage3 = (alpha[2][0] * U0 + alpha[2][1] * U_stage1 +
                    alpha[2][2] * U_stage2 + beta[2] * dt_flat * L2)
        enforce_positivity_gpu(U_stage3)
        if self.filter_func_gpu is not None:
            U_stage3 = self.filter_func_gpu(U_stage3)
        self.U_gpu = U_stage3.reshape(n_local, n_sps, 5)

        residual_norm = self._global_residual_norm(res2_flat)
        self.residual_history.append(residual_norm)
        self.iteration += 1
        return residual_norm

    def _global_residual_norm(self, res_flat) -> float:
        """MPI 全局残差归约。

        #1（2026-08-28）：分母此前是 `self.mesh.n_cells（全局）* n_sps *
        5 * self.n_ranks`——`self.mesh.n_cells` 在传统模式下本来就已经是
        全局单元数，再乘一次 `n_ranks` 会把分母错误放大 n_ranks 倍
        （与 self.U_gpu 此前错误按全局尺寸分配是两个独立的 bug，凑巧
        当时两边都用了同一个"全局 n_cells"所以没有立刻表现为形状错误，
        但归一化分母本身始终是错的）。改用 `partition.n_global_cells`
        （不确定 partition_mesh 是否严格均分负载，这是唯一真正的全局
        单元总数来源），不再额外乘 n_ranks。
        """
        cp = get_cupy()
        local_norm_sq = float(cp.sum(res_flat ** 2))
        global_norm_sq = allreduce_sum(local_norm_sq)
        n_sps = self.mesh.n_sps_per_cell
        n_global = self.partition.n_global_cells * n_sps * 5
        return np.sqrt(global_norm_sq / max(1, n_global))

    def solve(
        self,
        max_iter: int = 1000,
        dt: float = 1e-4,
        tol: float = 1e-6,
        output_interval: int = 10,
    ) -> Dict[str, Any]:
        """执行分布式稳态求解循环。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 收敛容差
            output_interval: 输出间隔

        Returns:
            结果字典
        """
        if is_root():
            print(f"Starting multi-GPU solve: {self.n_ranks} ranks, max_iter={max_iter}")

        converged = False
        final_residual = 1e10

        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res

            if is_root():
                if i == 0 or (i + 1) % output_interval == 0:
                    print(
                        f"Multi-GPU Iter {i+1}: Residual = {res:.6e} | "
                        f"Time/step: {t_end-t_start:.3f}s"
                    )

            if res < tol:
                converged = True
                if is_root():
                    print(f"✅ Multi-GPU Converged at iteration {i+1}")
                break

            if not np.isfinite(res):
                if is_root():
                    print(f"❌ Multi-GPU Diverged at iteration {i+1}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }

