"""AutoFlowCFD V2.0 - `MultiGPUDistributedSolver.__init__` 的装配阶段（mixin，只含方法）。

从 `core/gpu/distributed/gpu_distributed.py` 拆出（2026-09-25）：原 `__init__` 478 行，
按"身份与设备 -> 分区与面几何 -> 时间积分 -> 湍流 -> 状态与边界"的依赖顺序切成
具名阶段，**顺序本身是契约**（各段注释记录的几处真实缺陷都出在顺序上，例如面几何
必须先于 halo 交换器与状态缓冲区构造、WMLES 模型必须先于边界幽灵态构造）。
"""

from loguru import logger

from autoflowcfd.core.gpu import get_cupy, gpu_available
from autoflowcfd.core.gpu.array_manager import GPUArrayManager
from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
from autoflowcfd.core.gpu.gpu_time_integration import GPUTimeIntegrator
from autoflowcfd.core.mpi import get_rank, mpi_available
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.partition import build_distributed_partition, partition_mesh
from autoflowcfd.core.time_integration.base import require_distributed_scheme

from .compact_view import _CompactMeshDataView


class _MultiGPUSetupMixin:
    """`MultiGPUDistributedSolver.__init__` 按依赖顺序调用的装配阶段。"""

    def _setup_identity_and_device(self, mesh, ops, n_ranks, rank, device_id, mu_molecular,
                                   rho_inf, vel_inf, p_inf, aoa_deg, aos_deg, turb_model):
        """环境与参数校验、rank/网格/阶数、来流字典、GPU 设备与数组管理器。"""
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

        self.mu_molecular = mu_molecular
        # mach_ref 的唯一来源（按 AUSM+up 预处理档钳下限）。此前这里手写一份、
        # 下限硬编码 0.1，而 CPU 自 2026-09-17 起在默认档用 0.05 —— 同一算例
        # 在 GPU 与 CPU 上拿到不同的 mach_ref（plate_demo：0.1 vs 0.0882）。
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref
        mach_ref = resolve_mach_ref(rho_inf, vel_inf, p_inf)
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

    def _setup_partition_and_geometry(self, mesh, ops, n_ranks, partition_info):
        """分区、局部面几何、压缩索引空间的网格数据、halo 交换器与分布式状态。

        Returns:
            (n_sps, n_local)
        """
        cp = get_cupy()
        device_id = self.device_id
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
        return n_sps, n_local

    def _setup_time_integration(self, time_scheme, cfl_start, cfl_max, cfl_min):
        """时间积分器、CFL 策略与低马赫预处理开关（均为全后端唯一来源）。"""
        # 时间积分器
        self.time_integrator = GPUTimeIntegrator(
            scheme=require_distributed_scheme(time_scheme))
        # 自适应 CFL + 低马赫数伪时间预处理（2026-09-14 补齐）：与 CPU
        # 分布式同一批改动、同一套语义。两者都以"存在由 CFL 数决定的
        # 逐单元局部步长"为前提，而这条路径此前用全局固定 dt（被记作
        # "已接受的简化"）。局部步长已补齐（见
        # `_compute_local_time_step_gpu` 的重写说明）。
        # 控制器按**全局**残差范数更新，所有 rank 得到同一个 CFL 数。
        # CFL 策略（控制器 or 固定 CFL）的唯一事实来源：
        # `time_integration/adaptive_cfl/policy.py::build_cfl_policy`（六个后端
        # 构造点此前各写一份且已分叉，见该模块文档）。
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
        self._cfl_controller, self.fixed_cfl_number = build_cfl_policy(
            time_scheme, cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min)
        from autoflowcfd.core.utils.preconditioning import resolve_low_mach_precond
        self.low_mach_precond_enabled = resolve_low_mach_precond(True, time_scheme)

        # DUAL_TIME 模式下 BDF2 需要的上一物理时间层状态（2026-09-02，
        # 见 step() 里 DUAL_TIME 分支说明）——None 表示尚未跑过一个
        # 物理步，退化为 BDF1，与单机 GPU `gpu_solver.py` 同一个约定。
        self._dual_time_U_prev = None

    def _setup_turbulence(self, mesh, turb_model, n_sps, mu_molecular, rho_inf, vel_inf,
                          turbulence_intensity, viscosity_ratio):
        """湍流模型（SST/DDES/IDDES/LES/WMLES）、壁面距离、网格尺度与模态滤波。"""
        cp = get_cupy()
        device_id = self.device_id
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

    def _setup_state_and_boundary(self, n_local, n_sps):
        """GPU 常驻求解状态、边界幽灵态（重映射到压缩面编号）与 WALL 掩码。"""
        cp = get_cupy()
        device_id = self.device_id
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
            from autoflowcfd.core.utils.flow_direction import (
                freestream_conservative_state,
            )

            # 速度方向必须取自 aoa/aos（2026-09-24 修复）：此前这里写死 (vel_inf, 0, 0)，
            # 而边界 Q_free 用的是正确方向，`--aoa` 非零时初场与边界不一致。
            # 8 处同类写法已统一到 `freestream_conservative_state`（见其文档）。
            self.U_gpu = cp.empty((n_local, n_sps, 5), dtype=cp.float64)
            self.U_gpu[:] = cp.asarray(
                freestream_conservative_state(self.freestream, 5))

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
