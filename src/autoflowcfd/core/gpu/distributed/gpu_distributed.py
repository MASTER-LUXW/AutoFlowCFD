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

import os
import time
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger

from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
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
        # 攻角/侧滑角（度）。0/0 时来流严格沿 +x，与此前把方向硬编码
        # 成 +x 的行为逐位相同。约定见 core/utils/flow_direction.py。
        aoa_deg: float = 0.0,
        aos_deg: float = 0.0,
        turb_model: str = "NONE",
        turbulence_intensity: float = 0.01,
        viscosity_ratio: float = 5.0,
        cfl_start: Optional[float] = None,
        cfl_max: Optional[float] = None,
        cfl_min: Optional[float] = None,
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
            cfl_start, cfl_max: 自适应 CFL 的初始值/上限（与单机
                FRSolver/GPUFRSolver 同名参数同一语义）。None 时
                cfl_start 退回 `cfl`、cfl_max 退回 max(cfl, 0.5)，
                这样不传这两个参数的既有调用方行为不变。
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
        # 保持 None，等价于悄悄退化成层流，且没有任何报错/警告。
        #
        # SST 真正接入分布式状态与残差计算（2026-09-02，见
        # `_compute_turbulence_source_distributed` 文档——复用单机
        # `compute_source_terms_gpu`/`update_fields_gpu`/`compute_
        # turbulence_transport_residual_gpu` 的数值逻辑，只是把 halo
        # 交换+compact 索引空间重排接上，同一套模式已经在 CPU MPI 路径
        # （`core/mpi/distributed_turbulence.py`）验证过）：此前 2026-08-28
        # #1 修复只收紧到 'none' 是因为 wall_distance/compact 索引空间
        # 对齐这部分工作本身还没做，不是设计上不可行——现在补齐。
        # DDES/IDDES 已于 2026-09-02 真正移植（复用单机 GPUDDESModel/
        # GPUIDDESModel 数值逻辑+CPU MPI 分布式同一套 halo 交换+compact
        # 重排模式，见 `_compute_turbulence_source_distributed` 文档）。
        # LES 同日移植——WALE 是纯代数模型（不像 SST 的 k/omega 有跨步
        # ODE 积分状态），不需要"halo 交换持久状态"这层复杂度，只需要
        # 用当前状态现算 mu_t，实现和验证成本远低于 SST/DDES/IDDES。
        # WMLES（2026-09-02 续接）：此前认为"壁面剪应力修正需要分布式
        # 面级外插——WALL 面可能横跨分区边界"是独立更大的架构缺口，
        # 排查后发现这个理由不成立——WALL 面必然完全属于拥有该 owner
        # 单元的单个 rank（边界面没有"neighbor 侧"，不是内部面，不
        # 可能横跨分区），真正的障碍是 `compute_wmles_wall_stress_
        # correction` 此前直接用全局 `mesh.face_flux_points`对象列表
        # 逐面取值，不认 compact 索引空间——现已改用 `flat_face_
        # override`+`boundary_ghost_provider.group_code`识别 WALL 面
        # （与 CPU MPI 分布式同一处修复，见 core/utils/solver_helpers.py
        # 文档），不再需要完整全局网格的边界几何信息。
        if turb_model is not None and str(turb_model).upper() not in ("NONE", "SST", "DDES", "IDDES", "LES", "WMLES"):
            raise NotImplementedError(
                f"MultiGPUDistributedSolver（--multi-gpu）目前只支持 "
                f"turbulence_model='none'/'sst'/'ddes'/'iddes'/'les'/'wmles'，"
                f"收到的是 '{turb_model}'。请改用 --turbulence-model "
                f"none/sst/ddes/iddes/les/wmles，或改用单机 GPU/CPU 后端。"
            )

        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        self.mesh = mesh
        self.ops = ops

        # Order Continuation 支持（2026-09-02，见 core/gpu/distributed/
        # gpu_distributed_order_continuation.py 模块文档）——与 CPU
        # `DistributedFRSolver` 同一约定：`self.order`（目标阶数）/
        # `self.current_order`（当前实际所在阶数）。构造函数没有单独的
        # `order` 参数（`mesh`/`ops` 已经在调用方按目标阶数构造好），
        # 直接从 `mesh.order` 推断。
        self.order = int(getattr(mesh, 'order', 0))
        self.current_order = self.order
        self.order_continuation_enabled = True
        self.flux_type = 'radau'

        self.mu_molecular = mu_molecular
        # mach_ref：与 CPU 版 FRSolver.__init__（fr_solver/solver.py）
        # 同一套计算方式/同一个用途，见该文件对应注释。物理下限钳制同样与
        # CPU 版镜像同步（2026-08-26，P2 发散专项）：低于 0.1 的参考马赫数会让
        # AUSM+up Mp 压差扩散项的 1/mach_ref² 放大压倒显式推进稳定性，
        # 完整推导/实证标定记录见 fr_solver/solver.py::_MACH_REF_FLOOR。
        mach_ref = vel_inf / np.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10))
        mach_ref = max(mach_ref, 0.1)
        # 见 gpu_solver.py 同一处说明（2026-09-17）
        self.freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf,
                           "mach_ref": mach_ref,
                           "aoa_deg": float(aoa_deg), "aos_deg": float(aos_deg)}
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
        # WMLES 壁面剪应力修正（见下方 wmles_model 构造处）需要一个
        # numpy（非 GPU 上传）、compact 索引空间的 mesh-like 对象——
        # `compute_wmles_wall_stress_correction` 读 `mesh.n_prism_cells`/
        # `n_points_1d`/`jacobians['det_jacs']`，直接用 `self.mesh`（"传统
        # 模式"下是完整全局网格）会按全局索引空间取值，压缩索引传进去
        # 会读到不相关单元——`compact_mesh_view` 本身已经是这个 numpy
        # compact 视图（升 GPU 前的中间产物），保存下来复用，不需要再
        # 构造一份。
        self._compact_mesh_view = compact_mesh_view
        self.mesh_data = self.array_mgr.upload_mesh_data(compact_mesh_view, ops)
        self.n_compact = compact_mesh_view.n_cells
        # perm/inv_perm 常驻 GPU，避免每步残差计算都重新上传（见
        # compute_inviscid_residual_gpu/compute_viscous_residual_gpu 对
        # 它们的消费方式）。
        self._perm_gpu = cp.asarray(self.dist_flat_face.perm)
        self._inv_perm_gpu = cp.asarray(self.dist_flat_face.inv_perm)

        # ops_data：与单机 GPUFRSolver 同一个约定（gpu_solver.py::
        # `self.ops_data = {k: v for k, v in self.mesh_data.items()}`）——
        # `upload_mesh_data(mesh, ops)` 本来就把网格几何和 FR 算子
        # （D_3d_tet/D_3d_prism 等）打包进同一个字典，`mesh_data`/
        # `ops_data` 两个参数名只是历史上分别读取的接口约定，内容
        # 完全相同。`_compute_turbulence_source_distributed` 需要这个
        # 属性名（单机版 compute_turbulence_source_gpu 复用的同一套
        # `compute_physical_gradient_gpu(..., self.mesh_data,
        # self.ops_data)` 调用约定），此前分布式类从未设置过。
        self.ops_data = {k: v for k, v in self.mesh_data.items()}

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
        # 自适应 CFL + 低马赫数伪时间预处理（2026-09-14 补齐）：与 CPU
        # 分布式同一批改动、同一套语义。两者都以"存在由 CFL 数决定的
        # 逐单元局部步长"为前提，而这条路径此前用全局固定 dt（被记作
        # "已接受的简化"）。局部步长已补齐（见
        # `_compute_local_time_step_gpu` 的重写说明）。
        # 控制器按**全局**残差范数更新，所有 rank 得到同一个 CFL 数。
        self._cfl_controller = None
        if str(time_scheme) in ("ssp_rk2", "ssp_rk3") or getattr(
                time_scheme, "value", None) in ("ssp_rk2", "ssp_rk3"):
            from autoflowcfd.core.time_integration.adaptive_cfl import (
                AdaptiveCFLController,
            )
            self._cfl_controller = AdaptiveCFLController(
                cfl_start=cfl_start if cfl_start is not None else cfl,
                cfl_max=cfl_max if cfl_max is not None else max(cfl, 0.5),
                **({} if cfl_min is None else {'cfl_min': cfl_min}),
            )
        _env_pc = os.environ.get("AFCFD_LOW_MACH_PRECOND")
        _req_pc = True if _env_pc is None else (_env_pc == "1")
        self.low_mach_precond_enabled = _req_pc and (
            str(time_scheme) in ("ssp_rk2", "ssp_rk3")
            or getattr(time_scheme, "value", None) in ("ssp_rk2", "ssp_rk3"))

        # DUAL_TIME 模式下 BDF2 需要的上一物理时间层状态（2026-09-02，
        # 见 step() 里 DUAL_TIME 分支说明）——None 表示尚未跑过一个
        # 物理步，退化为 BDF1，与单机 GPU `gpu_solver.py` 同一个约定。
        self._dual_time_U_prev = None

        # 初始化 GPU 湍流模型（与单机版 gpu_solver.py 同一套 Tu/VR 推导
        # k_inf/omega_inf + k_max/omega_max 物理上界公式，2026-09-02
        # 补齐——此前这里只构造了默认初值的 GPUTurbulenceSST，没有做
        # 单机版早就有的这层初始化）。`self.turb_model_gpu` 只按
        # `n_local_cells` 分配（与 `self.state`/`self.U_gpu` 一致，
        # local+halo 的 k/omega 通过独立的 2-var halo 交换器实时获取，
        # 不常驻）——与 CPU 分布式 SST（`distributed_turbulence.py`）
        # 同一个设计。
        self.turb_model_gpu = None
        self.turb_halo_gpu = None
        self.ddes_model_gpu = None
        self.iddes_h_max_compact = None
        self.iddes_h_wn_compact = None
        self.des_length_scale_halo_gpu = None
        if turb_model in ("SST", "DDES", "IDDES"):
            from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
            n_local_cells = self.partition.n_local_cells
            nu = mu_molecular / max(rho_inf, 1e-10)
            k_inf = 1.5 * (vel_inf * turbulence_intensity) ** 2
            nu_t_inf = viscosity_ratio * nu
            omega_inf = k_inf / max(nu_t_inf, 1e-30)
            self.turb_model_gpu = GPUTurbulenceSST(
                n_local_cells, n_sps, device_id, k_inf=k_inf, omega_inf=omega_inf
            )
            self.turb_model_gpu.k_max = 0.5 * vel_inf ** 2
            self.turb_model_gpu.omega_max = 1e6
            self.turb_halo_gpu = GPUHaloExchange(self.partition, n_sps=n_sps, n_vars=2, device_id=device_id)
            logger.info(f"Rank {self.rank}: GPU SST model initialized "
                        f"(k_inf={k_inf:.4e}, omega_inf={omega_inf:.4e})")

            # DDES/IDDES（2026-09-02）：与单机 GPU 路径同一套构造
            # （gpu_solver.py），复用 CPU 版 des.py::compute_h_max_and_h_wn
            # （纯逐单元几何量，与相邻单元/分区无关，"传统模式"下
            # self.mesh 是完整全局网格，算完按 compact_global_ids 切一次
            # 片即可，不需要新的跨 rank 几何交换）。des_length_scale_
            # halo_gpu 是跨步持久状态专用的 1-var halo 交换器，见
            # `_compute_turbulence_source_distributed` 对应修复文档。
            if turb_model == "DDES":
                from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUDDESModel
                from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
                self.ddes_model_gpu = GPUDDESModel()
                # h_max（2026-09-02 补齐，与下面 IDDES 分支同一处几何量、
                # 同一个持久化+按 compact_global_ids 切片策略）：
                # `apply_to_sst_model_gpu` 现在优先用各向异性感知的
                # max_edge 网格尺度，见 CPU 版 des.py 对应方法文档。
                # 只需要 h_max，h_wn 是 IDDES 专属几何量。
                h_max_cpu, _ = compute_h_max_and_h_wn(mesh)
                self._iddes_h_max_global = h_max_cpu
                compact_global_ids = self.dist_flat_face.compact_global_ids
                with cp.cuda.Device(device_id):
                    self.iddes_h_max_compact = cp.asarray(h_max_cpu[compact_global_ids])
                logger.info(f"Rank {self.rank}: GPU DDES model initialized (based on SST)")
            elif turb_model == "IDDES":
                from autoflowcfd.core.gpu.turbulence.gpu_turbulence_des import GPUIDDESModel
                from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
                self.ddes_model_gpu = GPUIDDESModel()
                h_max_cpu, h_wn_cpu = compute_h_max_and_h_wn(mesh)
                # 持久化全局（阶数无关）h_max/h_wn（2026-09-02，Order
                # Continuation 支持需要——见 gpu_distributed_order_
                # continuation.py 文档）：阶数切换后 compact_global_ids
                # 会变，需要重新切片，但 compute_h_max_and_h_wn(mesh) 本身
                # 是纯逐单元几何量、与阶数无关（见该函数文档），不需要
                # 重新计算，只需要保留这份全局结果供重新切片。
                self._iddes_h_max_global = h_max_cpu
                self._iddes_h_wn_global = h_wn_cpu
                compact_global_ids = self.dist_flat_face.compact_global_ids
                with cp.cuda.Device(device_id):
                    self.iddes_h_max_compact = cp.asarray(h_max_cpu[compact_global_ids])
                    self.iddes_h_wn_compact = cp.asarray(h_wn_cpu[compact_global_ids])
                logger.info(f"Rank {self.rank}: GPU IDDES model initialized (based on SST)")

            if self.ddes_model_gpu is not None:
                self.des_length_scale_halo_gpu = GPUHaloExchange(
                    self.partition, n_sps=n_sps, n_vars=1, device_id=device_id
                )

        # LES（2026-09-02）：WALE 是纯代数模型（不像 SST 的 k/omega 有
        # 跨步 ODE 积分状态），`_compute_turbulence_source_distributed`
        # 用当前状态现算 mu_t，不需要任何跨步持久 halo 交换基础设施。
        self.sgs_model_gpu = None
        self._grid_scale_compact = None
        if turb_model == "LES":
            from autoflowcfd.core.gpu.turbulence.gpu_sgs import GPUWALEModel
            self.sgs_model_gpu = GPUWALEModel()
            logger.info(f"Rank {self.rank}: GPU LES with WALE SGS model initialized")

        self._turbulence_intensity = turbulence_intensity
        self._viscosity_ratio = viscosity_ratio

        # WMLES（2026-09-02）：没有 k/omega ODE 状态，不需要
        # turb_model_gpu/turb_halo_gpu——只需要真实的 CPU 版 WMLESModel
        # 实例（与单机 gpu_solver.py/CPU FRSolver.__init__ 构造
        # wmles_model 同一个模式：必须在下面 build_boundary_ghost_
        # provider 之前构造，该函数用 getattr(self,"wmles_model",None)
        # 判断 WALL 组是否要切换成 is_no_slip=False）+ wall_distance_gpu
        # （y+ 计算需要，见下方统一计算）。
        self.wmles_model = None
        if turb_model == "WMLES":
            from autoflowcfd.core.turbulence.wmles import WMLESModel
            self.wmles_model = WMLESModel(nu=mu_molecular / max(rho_inf, 1e-10))
            logger.info(f"Rank {self.rank}: GPU-distributed WMLES model initialized")

        # 预计算壁面距离（compact 索引空间，见 _init_wall_distance_
        # distributed 文档"真实 bug 修复"一节）
        self.wall_distance_gpu = None
        if self.turb_model_gpu is not None or self.wmles_model is not None:
            self._init_wall_distance_distributed()

        # 网格尺度 Delta = V^(1/3)（WALE 用，compact 索引空间——
        # `mesh_data['cell_volumes']` 已经是 compact 大小，见
        # `_CompactMeshDataView` 构造处，直接用不需要再切片）。
        if self.sgs_model_gpu is not None:
            cell_volumes_compact = self.mesh_data.get('cell_volumes')
            delta = cp.abs(cell_volumes_compact) ** (1.0 / 3.0)
            self._grid_scale_compact = cp.tile(delta[:, None], (1, n_sps))

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
        # self.wmles_model 已在上面湍流模型初始化处构造好（WMLES 时为
        # 真实 WMLESModel 实例，否则 None）——不在这里重置，
        # build_boundary_ghost_provider 用 getattr(self,"wmles_model",
        # None) 判断 WALL 组 is_no_slip 取值，必须能读到真实值。
        # self._turbulence_intensity/_viscosity_ratio 已在上面湍流模型
        # 初始化处设置（构造参数，不再在这里用硬编码默认值覆盖）。
        self._sem_num_eddies = 200
        self.boundary_ghost_provider = build_boundary_ghost_provider(self, bc_overrides={})

        # 真实 bug 修复（2026-09-02，实现分布式湍流模型时排查发现，与
        # 湍流本身无关——任何使用真实 WALL/INLET/OUTLET/FARFIELD/SYMMETRY
        # 区分的分布式算例都会中招）：`build_boundary_ghost_provider(self,
        # ...)` 用 `self.mesh`（完整全局网格）构造 `BoundaryGhostStateProvider.
        # group_code`，其长度/索引是**全局**面编号（0..n_faces_global-1）。
        # 但 `compute_boundary_ghost_states`（inviscid_kernel.py/gpu_
        # inviscid.py）调用 `ghost_provider(f, ...)` 时，`f` 是本 rank
        # 的 local+halo **压缩索引空间**面编号（0..n_faces_compact-1，
        # 见 distributed_flat_face.py 模块文档"棱柱在前"排列）——两套
        # 编号不是同一个索引空间的子区间（`partition.local_faces[i]`
        # 给出压缩空间第 i 个面对应的真实全局面编号，不是恒等映射，见
        # `build_distributed_flat_face` 用它切片 `global_flat.*
        # [local_face_indices]` 构造 `base_flat` 处）——用压缩索引直接
        # 查全局编号的 `group_code` 数组会读到不相关面的边界类型。真实
        # 合成网格验证（2-rank 分区，12 个压缩面）：8/12（67%）面被
        # 分配到错误的边界组编码。修复：把 `group_code` 重映射到本 rank
        # 的压缩索引空间——`group_code[i]`（压缩空间）= 原
        # `group_code[partition.local_faces[i]]`（全局空间）。只有真正
        # 构造出 `BoundaryGhostStateProvider`（有 `group_code` 属性）时
        # 才重映射，`None`/自定义 callable（没有这个属性）不受影响。
        if self.boundary_ghost_provider is not None and hasattr(self.boundary_ghost_provider, 'group_code'):
            self.boundary_ghost_provider.group_code = (
                self.boundary_ghost_provider.group_code[self.partition.local_faces]
            )

        # WALL 边界面拓扑掩码（compact 索引空间，SST k/omega 输运
        # Dirichlet BC 用，2026-09-02 补齐——与单机 GPUFRSolver 同一个
        # 一次性缓存策略，见 gpu_solver.py 对应构造处）：`compute_wall_
        # dirichlet_mask_gpu` 只需要 `mesh.face_connectivity.n_faces`
        # （纯计数）和 `boundary_ghost_provider.group_code`（上面已经
        # 重映射到 compact 空间）——用一个只提供这个计数的最小鸭子类型
        # `mesh` 代替真正的全局网格，`n_faces` 直接取 `dist_flat_face.
        # base_flat.n_faces`（compact 索引空间的面数，与 group_code 长度
        # 一致）。
        self._wall_mask_k_gpu = None
        if self.turb_model_gpu is not None:
            import types
            from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_wall_dirichlet_mask_gpu
            compact_mesh_stub = types.SimpleNamespace(
                face_connectivity=types.SimpleNamespace(n_faces=self.dist_flat_face.base_flat.n_faces)
            )
            wall_mask_np = compute_wall_dirichlet_mask_gpu(compact_mesh_stub, self.boundary_ghost_provider)
            with cp.cuda.Device(device_id):
                self._wall_mask_k_gpu = cp.asarray(wall_mask_np)

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

    @classmethod
    def from_fully_distributed_package(cls, package: dict, n_ranks: int,
                                        device_id=None, rank=None, root_context=None):
        """真正的"完全分布式加载"构造入口（#1，2026-09-02）——与主
        `__init__`（"传统模式"：每个 rank 独立加载完整全局网格）的关键
        区别：`package` 是 root rank 预先算好、已经按本 rank 的 compact
        索引空间切好的紧凑数据，本 rank 从未持有、也不需要持有完整
        全局网格。与 CPU `DistributedFRSolver.from_fully_distributed_
        package` 共用同一套 `build_fully_distributed_rank_package`/
        `distributed_mesh_load_v2`（package 构造逻辑与后端无关）。

        实现拆到独立模块 `gpu_distributed_fully_distributed.py`（控制
        单文件行数，与 `_interpolate_to_new_order`/
        `gpu_distributed_order_continuation.py` 同一个拆分动机），见该
        模块文档完整说明（范围边界：支持 turbulence_model='none'/'sst'/
        'ddes'/'iddes'/'wmles'/'les'，DUAL_TIME/checkpoint/Order
        Continuation 均已接入）。

        Args:
            package: `build_fully_distributed_rank_package` 的返回值
                （或 `distributed_mesh_load_v2` 经 MPI 收发后本 rank
                收到的那一份）
            n_ranks: MPI rank 总数
            device_id: GPU 设备号（None 时按 rank 轮询分配）
            rank: 当前 rank（None 时从 MPI 获取）
            root_context: 仅 root rank 需要非 None——`distributed_mesh_
                load_v2` 返回的第二个值，供 Order Continuation 使用，
                见 `gpu_distributed_fully_distributed.py` 模块文档。

        Returns:
            MultiGPUDistributedSolver 实例
        """
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            build_multi_gpu_solver_from_fully_distributed_package,
        )
        return build_multi_gpu_solver_from_fully_distributed_package(
            cls, package, n_ranks, device_id=device_id, rank=rank, root_context=root_context,
        )

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
        调用不匹配，必然 TypeError（V2.0 专家组评审逐行核实）。

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

        # WMLES 壁面剪应力修正（2026-09-02）：与单机 GPUFRSolver.
        # compute_viscous_residual_gpu 同一个施加时机（残差组装阶段，
        # 时间积分之前），直接调用 CPU 核心函数而不是
        # gpu_turbulence_wmles.py 的单机 facade（后者用 `solver.mesh`
        # 构造 extrap 所需的 n_prism_cells/jacobians，单机语境下就是
        # compact 空间本身；分布式场景下 `self.mesh` 在"传统模式"下是
        # **全局**网格，必须换成 `self._compact_mesh_view`，见该属性
        # 构造处的说明），自己搭建同样结构的 facade——与 CPU MPI 分布式
        # `distributed_compute_viscous_residual` 完全同一个模式。
        if self.wmles_model is not None:
            cp = get_cupy()
            import types
            from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
            from autoflowcfd.core.utils.solver_helpers import compute_wmles_wall_stress_correction

            U_compact_cpu = cp.asnumpy(U_compact)
            Q_compact_cpu = conserved_to_primitive(U_compact_cpu[..., :5])
            wall_distance_cpu = (
                cp.asnumpy(self.wall_distance_gpu) if self.wall_distance_gpu is not None else None
            )
            facade = types.SimpleNamespace(
                wmles_model=self.wmles_model, mesh=self._compact_mesh_view, ops=self.ops,
                wall_distance=wall_distance_cpu,
                state=types.SimpleNamespace(U=U_compact_cpu, Q=Q_compact_cpu),
                boundary_ghost_provider=self.boundary_ghost_provider,
            )
            correction_cpu = compute_wmles_wall_stress_correction(
                facade, flat_face_override=self.dist_flat_face.base_flat,
            )
            if correction_cpu is not None:
                correction_compact = cp.asarray(correction_cpu)
                residual_compact = residual_compact + correction_compact[..., :residual_compact.shape[-1]]

        residual_native = self._unpermute_from_compact(residual_compact)
        return residual_native[: self.partition.n_local_cells]

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
        from autoflowcfd.fr.native_tet_padding import (
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
        c = getattr(self, "_cfl_controller", None)
        return c.cfl_number if c is not None else self.time_integrator.cfl

    def _compute_total_residual_gpu(self, mu_t_field=None):
        """计算总残差（无粘 + 粘性），先执行 halo 交换。

        Args:
            mu_t_field: 动力涡粘度 (n_cells, n_sps) CuPy 数组（可选）
        """
        self._halo_exchange_gpu()
        inv_res = self.compute_inviscid_residual_gpu()
        visc_res = self.compute_viscous_residual_gpu(mu_t_field=mu_t_field)
        return inv_res + visc_res

    def _interpolate_to_new_order(self, target_p: int) -> None:
        """阶数切换（2026-09-02，见 core/gpu/distributed/
        gpu_distributed_order_continuation.py 模块文档）——与 CPU
        `DistributedFRSolver._interpolate_to_new_order` 同一个命名/
        调用约定，供 `run_distributed_order_continuation`（CPU/GPU
        共用同一份迭代循环，见 core/mpi/distributed_order_
        continuation.py）统一调用。"""
        from autoflowcfd.core.gpu.distributed.gpu_distributed_order_continuation import (
            gpu_interpolate_to_new_order,
        )
        gpu_interpolate_to_new_order(self, target_p)

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
        # 逐单元局部 CFL 步长（2026-09-14）：此前这里直接铺用调用方传入
        # 的全局固定 `dt`，并把它记作"与 CPU 分布式路径一致的、已被接受
        # 的简化"。用户明确指出本项目不接受简化，本轮与 CPU 分布式同批
        # 补齐——`_compute_local_time_step_gpu` 已重写（修掉两处潜伏 bug
        # 与"只用 SP0"的简化，见该方法文档），这里改用它。
        # 返回值是**紧凑排列**（棱柱在前），按 inv_perm 换回原生排列后
        # 切 local 段才能与 `self.U_gpu` 对齐。
        # 真实 bug 修复（2026-09-02，与 CPU 分布式 `DistributedFRSolver.
        # step` 同一处修复、同一个理由——用户明确要求"不允许出现完成度
        # 不是100%的功能点"后排查发现）：此前这里无论 `self.time_
        # integrator.scheme` 是什么都无条件走下面手动展开的 RK stage
        # 逻辑，`scheme` 只被用来查系数表——DUAL_TIME（真正时间精度的
        # 瞬态仿真，DES/LES 场景理应使用的模式）请求了也完全无路可走。
        # 与单机 GPU `gpu_solver.py::step` 同一个分派方式（`GPUTime
        # Integrator.step_dual_time` 本身早已实现，只是从未被这条
        # 分布式路径调用过）：构造一个把 trial U 临时写入 `self.U_gpu`
        # 再调用既有 `_compute_total_residual_gpu`（自带 halo 交换）的
        # 残差闭包，直接复用，不需要另起一套残差组装逻辑。
        if self.time_integrator.scheme == "dual_time":
            # DUAL_TIME 下预处理不启用，局部 dt 只有一份；湍流仍用它
            # （物理波速那一份）。
            _dtm = self._compute_local_time_step_gpu()
            _dt_turb = float(cp.mean(_dtm[self._inv_perm_gpu][:n_local]))
            mu_t_field = self._compute_turbulence_source_distributed(_dt_turb)
            U_flat = self.U_gpu.reshape(n_local * n_sps, 5)

            def _spatial_residual(U_flat_trial):
                U_trial = U_flat_trial.reshape(n_local, n_sps, 5)
                saved_U = self.U_gpu
                self.U_gpu = U_trial
                try:
                    res = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
                finally:
                    self.U_gpu = saved_U
                return (-res).reshape(n_local * n_sps, 5)

            # 内层伪时间迭代的局部加速步长：与单机一致用局部 CFL 步长
            # （`dt` 仍然是真正的物理时间步长，通过 dt_physical= 传入）。
            dt_mean_c = self._compute_local_time_step_gpu()
            dt_mean_local = dt_mean_c[self._inv_perm_gpu][:n_local]
            pseudo_dt = cp.broadcast_to(
                dt_mean_local[:, None], (n_local, n_sps)).reshape(n_local * n_sps)
            max_inner_iter = getattr(self.time_integrator, 'dual_time_steps', 5)
            U_new_flat = self.time_integrator.step_dual_time(
                U_flat, _spatial_residual, pseudo_dt, dt_physical=dt,
                solution_prev=self._dual_time_U_prev, max_inner_iter=max_inner_iter,
                filter_func=self.filter_func_gpu,
            )
            self._dual_time_U_prev = U_flat.copy()
            self.U_gpu = U_new_flat.reshape(n_local, n_sps, 5)
            final_res_flat = _spatial_residual(U_new_flat)
            residual_norm = self._global_residual_norm(final_res_flat)
            self.residual_history.append(residual_norm)
            self.iteration += 1
            self._update_cfl_controller(residual_norm)
            return residual_norm

        dt_mean_c, dt_phys_c = self._compute_local_time_step_gpu(
            return_physical_too=True)
        dt_mean_local = dt_mean_c[self._inv_perm_gpu][:n_local]
        dt_phys_local = dt_phys_c[self._inv_perm_gpu][:n_local]
        dt_flat = cp.broadcast_to(
            dt_mean_local[:, None], (n_local, n_sps)).reshape(n_local * n_sps, 1)

        # RK 系数表
        scheme = self.time_integrator.scheme
        table = self.time_integrator._table
        alpha = table["alpha"]
        beta = table["beta"]
        n_stages = table["stages"]

        # 湍流源项求值（算子分裂，每个 step 开始时计算一次，用当前——
        # 上一步末尾——的状态，与 CPU 分布式 SST 同一个时序，见
        # distributed_solver.py::step 文档）。turb_model_gpu 为 None
        # （turbulence_model='none'）时恒返回 None。
        # 湍流标量用**物理**波速算出的那一份 dt（见
        # gpu_distributed_init.py::_compute_turbulence_source_distributed
        # 第 5 步的说明与单机 step.py 的 `turb_dt = dt_physical`）。
        mu_t_field = self._compute_turbulence_source_distributed(
            float(cp.mean(dt_phys_local)))

        U_flat = self.U_gpu.reshape(n_local * n_sps, 5)
        U0 = U_flat.copy()

        def _precond(dudt_flat, U_state_flat):
            """施加低马赫数预处理：L = Gamma * (-R) = Gamma * dU/dt。

            Gamma 线性、逐点，作用在 dU/dt 上与作用在残差上等价（见
            core/utils/preconditioning.py 模块末尾）。它**必须**与上面
            按预处理波速取的 `dt_flat` 成对出现——只改步长不改方程就是
            2026-08-24 那次失稳。未启用时原样返回。
            残差范数用的仍是**未预处理**的 `resN_flat`（物理残差），与
            单机/CPU 分布式同一分工。
            """
            if not self.low_mach_precond_enabled:
                return dudt_flat
            from autoflowcfd.core.gpu.gpu_preconditioning import (
                apply_low_mach_preconditioner_gpu,
            )
            L3 = dudt_flat.reshape(n_local, n_sps, 5)
            out = apply_low_mach_preconditioner_gpu(
                L3, U_state_flat.reshape(n_local, n_sps, 5),
                self.freestream["mach_ref"], out=L3)
            return out.reshape(n_local * n_sps, 5)

        # === Stage 0: 初始残差 ===
        res0 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res0_flat = res0.reshape(n_local * n_sps, 5)
        L0 = _precond(-res0_flat, U0)

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
            self._update_cfl_controller(residual_norm)
            return residual_norm

        # === Stage 2 ===
        res1 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res1_flat = res1.reshape(n_local * n_sps, 5)
        L1 = _precond(-res1_flat, U_stage1)
        U_stage2 = (alpha[1][0] * U0 + alpha[1][1] * U_stage1 + beta[1] * dt_flat * L1)
        enforce_positivity_gpu(U_stage2)
        if self.filter_func_gpu is not None:
            U_stage2 = self.filter_func_gpu(U_stage2)
        self.U_gpu = U_stage2.reshape(n_local, n_sps, 5)

        if n_stages == 2:
            residual_norm = self._global_residual_norm(res1_flat)
            self.residual_history.append(residual_norm)
            self.iteration += 1
            self._update_cfl_controller(residual_norm)
            return residual_norm

        # === Stage 3 (RK3) ===
        res2 = self._compute_total_residual_gpu(mu_t_field=mu_t_field)
        res2_flat = res2.reshape(n_local * n_sps, 5)
        L2 = _precond(-res2_flat, U_stage2)
        U_stage3 = (alpha[2][0] * U0 + alpha[2][1] * U_stage1 +
                    alpha[2][2] * U_stage2 + beta[2] * dt_flat * L2)
        enforce_positivity_gpu(U_stage3)
        if self.filter_func_gpu is not None:
            U_stage3 = self.filter_func_gpu(U_stage3)
        self.U_gpu = U_stage3.reshape(n_local, n_sps, 5)

        residual_norm = self._global_residual_norm(res2_flat)
        self.residual_history.append(residual_norm)
        self.iteration += 1
        self._update_cfl_controller(residual_norm)
        return residual_norm

    def _update_cfl_controller(self, residual_norm: float) -> None:
        """用**全局**残差范数更新自适应 CFL 控制器。

        必须用全局值：所有 rank 因此得到同一个 CFL 数，进而得到一致的
        局部步长缩放。按各自的局部残差更新会让 rank 间 CFL 漂移，
        破坏分布式一致性。
        """
        c = getattr(self, "_cfl_controller", None)
        if c is not None:
            c.update(residual_norm)

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
        checkpoint_callback=None,
        phase_max_iter: Optional[int] = None,
        residual_drop_threshold: float = 1e2,
    ) -> Dict[str, Any]:
        """执行分布式稳态求解循环。

        真实 bug 修复（2026-09-02，与 CPU MPI 分布式 `DistributedFRSolver.
        solve` 同一处修复、同一个理由——用户明确要求"不允许出现完成度
        不是100%的功能点"后排查发现）：此前 `output_interval` 只控制
        `print` 进度打印频率，没有任何中间 checkpoint 保存机制。补上
        与单机 `FRSolver.solve`/CPU 分布式 `DistributedFRSolver.solve`
        同一个约定的 `checkpoint_callback(solver, iteration)` 回调。

        Order Continuation 自动分派（2026-09-02，见 core/gpu/distributed/
        gpu_distributed_order_continuation.py 模块文档）：与 CPU
        `DistributedFRSolver.solve()` 同一个判据——`self.order`（目标
        阶数）>= 2 时自动改用逐阶爬坡（`run_distributed_order_
        continuation`，CPU/GPU 共用同一份迭代循环），不需要调用方显式
        请求。该函数返回 `SolverResult`（dataclass），这里适配转换成
        本方法一贯的 dict 返回约定，不改变调用方（CLI）已有的
        `result['final_residual']`/`result['iterations']` 访问方式。

        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 收敛容差
            output_interval: 输出间隔
            checkpoint_callback: 可选，`callback(solver, iteration)`，
                每步结束后调用一次

        Returns:
            结果字典
        """
        if getattr(self, 'order_continuation_enabled', True) and self.order >= 2:
            from autoflowcfd.core.mpi.distributed_order_continuation import (
                run_distributed_order_continuation,
            )
            result = run_distributed_order_continuation(
                self, max_iter, dt, tol, checkpoint_callback=checkpoint_callback,
                phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
            )
            return {
                'converged': result.converged,
                'iterations': result.iterations,
                'final_residual': result.final_residual,
            }

        if is_root():
            print(f"Starting multi-GPU solve: {self.n_ranks} ranks, max_iter={max_iter}")

        converged = False
        final_residual = 1e10
        _last_finite = None

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

            # 发散检查必须在 checkpoint 回调**之前**（2026-09-16 修复）：
            # 此前顺序是先保存再检查，于是发散那一步的 NaN 状态会被如实
            # 写进 checkpoint。`res` 来自 `self.step()` 的全域 allreduce，
            # 各 rank 取值相同，所以全部 rank 会同时抛出、不会死锁。
            check_residual_finite(
                res, i + 1, last_finite=_last_finite,
                extra_hint=f"分布式路径：{self.n_ranks} 个 rank，"
                           f"残差是全域 allreduce 值（各 rank 一致）",
            )
            _last_finite = res

            if checkpoint_callback is not None:
                # 全部 rank 都要调用——保存需要每个 rank 各自贡献 local
                # cells 数据（见 CPU 分布式 solve 同一处注释）。
                checkpoint_callback(self, i + 1)

            if res < tol:
                converged = True
                if is_root():
                    print(f"✅ Multi-GPU Converged at iteration {i+1}")
                break

        return {
            'converged': converged,
            'iterations': self.iteration,
            'final_residual': final_residual,
            'residual_history': self.residual_history,
        }

