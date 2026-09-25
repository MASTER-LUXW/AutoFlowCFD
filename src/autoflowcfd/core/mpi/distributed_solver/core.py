"""AutoFlowCFD V2.0 - DistributedFRSolver 主类：传统模式构造

从 `src/autoflowcfd/core/mpi/distributed_solver.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

import numpy as np
from typing import Optional
from loguru import logger
from autoflowcfd.core.mpi import get_rank, is_root
from autoflowcfd.core.mpi.partition import (
    partition_mesh,
    build_distributed_partition,
)
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.distributed_flat_face import (
    build_distributed_flat_face,
)
from autoflowcfd.core.mpi.comm import barrier
from autoflowcfd.core.time_integration.base import (
    TimeIntegrator, TimeIntegrationScheme, require_distributed_scheme,
)
from .from_package import _DistributedFromPackageMixin
from .solve_loop import _DistributedSolveMixin
from .step import _DistributedStepMixin
from .support import _DistributedSupportMixin


class DistributedFRSolver(_DistributedFromPackageMixin, _DistributedStepMixin, _DistributedSolveMixin,
                          _DistributedSupportMixin):
    """MPI 域分解分布式 FR 求解器。

    组合 FRSolver（单机残差计算）+ MPI 基础设施（分区、halo 交换、
    全局归约）。每个 rank 独立持有一个 FRSolver 实例（只处理 local cells），
    通过 halo 交换获取邻居信息。

    Attributes:
        partition: 本 rank 的分区信息
        halo_exchange: halo 交换管理器
        state: 分布式状态
        rank: 当前 rank
        n_ranks: 总 rank 数
    """

    def __init__(
        self,
        mesh,
        ops,
        n_ranks: int,
        face_connectivity=None,
        face_connectivity_data=None,
        partition_info=None,
        rank: Optional[int] = None,
        wall_distance_source=None,
        **solver_kwargs,
    ):
        """初始化分布式求解器。

        Args:
            mesh: HighOrderMesh（分布式模式下为局部网格）
            ops: FROperators
            n_ranks: MPI rank 总数
            face_connectivity: FRFaceConnectivity（单机模式，已废弃）
            face_connectivity_data: dict（分布式模式，局部面连接关系数据）
            partition_info: dict（分布式模式，分区信息）
            rank: 当前 rank 编号（默认从 MPI 获取）
            **solver_kwargs: 传递给 FRSolver 的参数

        Raises:
            NotImplementedError: 请求了 'none'/'sst'/'ddes'/'iddes'/'wmles'/
                'les' 以外的湍流模型时。SST/DDES/IDDES 已真正接入分布式
                状态与残差计算（SST 2026-09-02，DDES/IDDES 同日续接，见
                core/mpi/distributed_turbulence.py）——三者共享同一套
                k/omega 输运基础设施，DDES/IDDES 只是多了一段 DES 长度
                尺度替换，且所需的额外几何量（cell_volumes、IDDES 的
                h_max/h_wn）都是纯逐单元局部量，不需要新的跨 rank 几何
                交换。WMLES（2026-09-02 续接）壁面剪应力修正本身也只是
                纯逐 owner 单元的局部操作（WALL 面必然只属于拥有该单元
                的 rank，不需要跨 rank 数据）——真正的障碍是
                `compute_wmles_wall_stress_correction`此前直接用全局
                `mesh.face_flux_points`对象列表逐面取值，不认 compact
                索引空间，现已改用`flat_face_override`（与本文件其余
                面残差函数同一个约定）+`boundary_ghost_provider.
                group_code`识别 WALL 面，不再需要完整全局网格的边界
                几何信息，见该函数文档。LES（同日续接）是纯代数 SGS
                模型（WALE，没有跨步 ODE 状态），"每步现算"即可，走独立
                的 `distributed_compute_les_viscosity`，不经过 SST/DDES/
                IDDES 共用的那套 k/omega 输运基础设施。单机模式已完整
                支持全部模型。
        """
        turb_model_name = solver_kwargs.get('turb_model_name', 'none')
        turb_model_upper = str(turb_model_name).upper() if turb_model_name is not None else 'NONE'
        if turb_model_upper not in ('NONE', 'SST', 'DDES', 'IDDES', 'WMLES', 'LES'):
            raise NotImplementedError(
                f"MPI 分布式求解器（--n-ranks/--np > 1，或 --multi-gpu）目前只支持 "
                f"turbulence_model='none'/'sst'/'ddes'/'iddes'/'wmles'/'les'，"
                f"收到的是 '{turb_model_name}'。"
                f"请改用 --turbulence-model none/sst/ddes/iddes/wmles/les，"
                f"或改用单机模式。"
            )
        self._turb_model_upper = turb_model_upper
        # 真实 bug 修复（2026-09-02，扩展 DDES/IDDES 分布式支持时发现，
        # 与本次新增功能无关，SST 分布式路径同样中招）：下面 SST/DDES/
        # IDDES 分支调用 `init_turbulence_models(self, n_local, n_sps)`
        # 内部按 `solver.turb_model_name` 分派，但此前这里从未真正
        # 设置过 `self.turb_model_name` 这个属性——`DistributedFRSolver`
        # 构造时只要请求 SST（或本次新增的 DDES/IDDES），必然在这里
        # `AttributeError` 崩溃。此前从未被测试捕捉到是因为
        # `test_distributed_turbulence.py` 只单独测试更底层的
        # `distributed_compute_turbulence_source_and_viscosity`，从未
        # 真正走过 `DistributedFRSolver.__init__` 这条构造路径。
        self.turb_model_name = turb_model_upper

        self.rank = rank if rank is not None else get_rank()
        self.n_ranks = n_ranks
        self.mesh = mesh
        self.ops = ops

        # Order Continuation 支持（2026-09-02，见 core/mpi/
        # distributed_order_continuation.py 模块文档）：`self.order`/
        # `self.current_order` 与单机 FRSolver 同一约定
        # （`order`=目标阶数，`current_order`=当前实际所在阶数，两者
        # 在阶数爬升过程中的中间阶段不相等）。默认从 `solver_kwargs`
        # 里读取目标 order（CLI 恒显式传入），退化用 `mesh` 当前活动
        # 阶数兜底。
        self.order = int(solver_kwargs.get('order', getattr(mesh, 'order', 0)))
        self.current_order = self.order
        self.order_continuation_enabled = solver_kwargs.get('order_continuation_enabled', True)
        self._is_fully_distributed = False

        # 分布式模式：使用传入的分区信息
        if partition_info is not None:
            cell_partition = partition_info['cell_partition']
            n_global_cells = partition_info['n_global_cells']
        elif face_connectivity is not None:
            # 兼容旧接口：所有 rank 独立执行分区
            if self.rank == 0:
                logger.info(f"Partitioning mesh into {n_ranks} parts (METIS on root)...")
                cell_partition = partition_mesh(face_connectivity, n_ranks, n_cells=mesh.n_cells)
            else:
                cell_partition = None
            if n_ranks > 1:
                from autoflowcfd.core.mpi.comm import bcast_from_root
                cell_partition = bcast_from_root(cell_partition)
            n_global_cells = face_connectivity.owner_cell.max() + 1
        else:
            raise ValueError("Either face_connectivity or partition_info must be provided")

        # Order Continuation 分布式支持（2026-09-02）：阶数切换需要用
        # 同一个 cell_partition 重建 partition/dist_flat_face（`mesh`
        # 在"传统模式"下是每个 rank 都持有的完整全局网格，`mesh.
        # face_connectivity` 是阶数无关的拓扑信息，见
        # `high_order_mesh_order.py::set_order` 文档——不随阶数变化，
        # 不需要额外持久化）。只对 `face_connectivity is not None`
        # 这条"兼容旧接口"/CLI 生产路径分支支持（`partition_info` 这条
        # 分支要求调用方自带一个可能不是完整全局网格的 `mesh`，CLI 从未
        # 真正使用过这条分支构造 `DistributedFRSolver`，见该分支上方
        # 文档——如实标注为不支持，而不是假装能用）。
        self._oc_cell_partition = cell_partition if face_connectivity is not None else None

        # 构建本 rank 的分区数据结构。
        #
        # build_distributed_partition 的 halo 探测（哪些面跨越分区边界、
        # 因此需要 halo 交换）必须在**全局、未裁剪**的面连接关系上做——
        # 它要用 cell_partition[owner]/cell_partition[neighbor] 判断一条
        # 面两侧是否属于不同 rank，cell_partition 本身是全局数组
        # （下标是全局 cell id）。此前这里传入的是
        # face_connectivity_data（distributed_mesh_load 提取的**局部**
        # 数据，owner_cell/neighbor_cell 已经重映射成本 rank 的局部索引，
        # 跨 rank 的 neighbor 被强制置为 -1，与真正的边界面用同一个
        # 哨兵值、在本 rank 视角下已经无法区分），拿它当 face_connectivity
        # 传给 build_distributed_partition 有两个独立问题：(1) 该
        # 函数第一行就要读 face_connectivity.n_faces，一个用普通
        # dataclass 拼出来的 LocalFaceConnectivity 对象没有这个属性，
        # 必然 AttributeError；(2) 即便补上这个属性，用局部索引去查
        # cell_partition（全局索引空间）也是错的，且跨 rank 邻居已经
        # 提前坍缩成 -1，halo 探测的 `if nc < 0: continue` 会直接跳过
        # 所有真正的分区边界面——halo_cells/send_lists/recv_lists 会
        # 算成空的，粘性/无粘残差在分区边界上完全得不到邻居数据
        # （V2.0 专家组评审逐行核实：这条路径此前从未被真正跑通过）。
        #
        # 修复：用 distributed_mesh_load 随 partition_info 一起广播的
        # **全局**面连接关系（global_owner_cell/global_neighbor_cell/
        # global_is_boundary）构建分区——这才是 build_distributed_
        # partition 设计时假设的输入形态，cell_partition 也是同一个
        # 全局索引空间，两者能正确对齐。
        if face_connectivity_data is not None:
            if partition_info is None or 'global_owner_cell' not in partition_info:
                raise ValueError(
                    "DistributedFRSolver(face_connectivity_data=...) 需要 "
                    "partition_info 里包含 global_owner_cell/global_neighbor_cell/"
                    "global_is_boundary（distributed_mesh_load 的输出）才能正确"
                    "构建分区——不能只用局部（已按 rank 裁剪）的面连接关系，"
                    "见本方法上方注释。"
                )
            from dataclasses import dataclass

            @dataclass
            class GlobalFaceConnectivityView:
                """构建分区专用的最小全局面连接关系视图（只读，不重新
                实例化完整 FRFaceConnectivity，避免要求调用方提供它
                不需要的其余几何字段）。"""
                owner_cell: np.ndarray
                neighbor_cell: np.ndarray
                is_boundary: np.ndarray

                @property
                def n_faces(self) -> int:
                    return len(self.owner_cell)

            global_fc = GlobalFaceConnectivityView(
                owner_cell=np.asarray(partition_info['global_owner_cell']),
                neighbor_cell=np.asarray(partition_info['global_neighbor_cell']),
                is_boundary=np.asarray(partition_info['global_is_boundary']),
            )
            self.partition = build_distributed_partition(
                global_fc, cell_partition, self.rank, n_ranks
            )
        else:
            # 兼容旧接口
            self.partition = build_distributed_partition(
                face_connectivity, cell_partition, self.rank, n_ranks
            )

        if is_root():
            logger.info(
                f"Rank {self.rank}: {self.partition.n_local_cells} local cells, "
                f"{self.partition.n_halo} halo cells, "
                f"{len(self.partition.neighbor_ranks)} neighbors"
            )

        # 3. 构建分布式面几何——必须排在下面 HaloExchange/DistributedFRState
        # 构造**之前**：当 cell_partition 非空时，这一步内部会调用
        # extend_halo_for_flux_point_cross_references 原地扩展
        # self.partition 的 halo_cells/send_lists/recv_lists（覆盖 FR
        # Flux Point 多源交叉插值依赖，真实 cube_demo 网格上 16%~19% 的
        # 单元需要这种扩展，不是边缘情况）。HaloExchange 会按*构造时刻*
        # 的 send_lists/recv_lists 大小预分配定长 buffer——如果它先构造、
        # 面几何扩展后发生，buffer 要么大小对不上（形状不匹配崩溃），
        # 要么根本没有新增邻居 rank 的 key（KeyError，或更隐蔽地静默
        # 跳过、留下未初始化的 halo 数据）。第四次评审第二轮复核发现
        # 此前的构造顺序正是反的，这里改为先做面几何（含 halo 扩展），
        # 再构造依赖最终 halo 布局的对象。
        #
        # 传入 cell_partition 以便扩展 halo 层；"完全分布式加载"模式下
        # mesh 是局部网格而非全局网格，这个扩展步骤依赖的
        # get_flat_face_geometry(mesh, ops) 全局面几何假设在那条路径下
        # 可能不成立——这是一个更深层、超出本次修复范围的架构问题，此处
        # 不展开，只保证"传统模式：所有 rank 有完整网格"这条路径
        # （cell_partition 在此处始终是全局数组）正确。
        self.dist_flat_face = build_distributed_flat_face(mesh, ops, self.partition, cell_partition=cell_partition)

        # 4. 初始化分布式状态（在面几何/halo 扩展之后构造）
        n_sps = mesh.n_sps_per_cell
        n_vars = solver_kwargs.get('n_vars', 5)
        self.state = DistributedFRState(self.partition, n_sps, n_vars)
        # 均匀自由流场初始化（真实 bug 修复，2026-09-02，见
        # DistributedFRState.initialize_uniform 文档）：此前这里从未
        # 对新构造的 state 赋初值，conserved state 恒为全零。
        # 初场速度方向必须与边界 Q_free 用同一个来流方向（2026-09-17）
        from autoflowcfd.core.utils.flow_direction import freestream_velocity
        _v0 = freestream_velocity(
            solver_kwargs.get('vel_inf', 33.33),
            solver_kwargs.get('aoa_deg', 0.0) or 0.0,
            solver_kwargs.get('aos_deg', 0.0) or 0.0)
        self.state.initialize_uniform(
            rho=solver_kwargs.get('rho_inf', 1.225),
            u=float(_v0[0]), v=float(_v0[1]), w=float(_v0[2]),
            p=solver_kwargs.get('p_inf', 101325.0),
        )

        # 5. 初始化 halo 交换器（同样必须在面几何/halo 扩展之后构造，
        # 理由见上）
        self.halo_exchange = HaloExchange(self.partition, n_sps, n_vars)

        # 6. 保存 solver kwargs 用于创建本地求解器
        self.solver_kwargs = solver_kwargs
        self._local_solver = None  # 延迟初始化
        self._p0_global_boundary_ghost_provider = None  # P0 分布式残差路径专用，见 local_solver 属性文档

        # 6b. SST 湍流模型初始化（2026-09-02，见 core/mpi/
        # distributed_turbulence.py 模块文档）。turb_model 只按
        # n_local_cells 分配（与 self.state 一致，local+halo 的 k/omega
        # 通过独立的 2-var halo 交换器实时获取，不常驻）。复用单机路径
        # 完全相同的 `init_turbulence_models`（鸭子类型：只需要
        # `.turb_model_name`/`.freestream`/`.mu_molecular`/
        # `._turbulence_intensity`/`._viscosity_ratio` 这几个属性，
        # `self` 已经或即将全部满足）。
        n_local = self.partition.n_local_cells
        self.turb_model = None
        self.turb_halo_exchange = None
        self.wall_distance_compact = None
        self.ddes_model = None
        self.iddes_h_max_compact = None
        self.iddes_h_wn_compact = None
        self.des_length_scale_halo_exchange = None
        self.wmles_model = None
        # LES（2026-09-02）：WALE 是纯代数 SGS 模型，没有 k/omega 那样的
        # 跨步 ODE 状态，也不需要 wall_distance（不像 WMLES 的 y+ 计算）
        # ——单独一个分支，不并入下面 SST/DDES/IDDES/WMLES 共用的构造块。
        self.sgs_model = None
        if turb_model_upper == 'LES':
            from autoflowcfd.core.turbulence.sgs import WALEModel
            self.sgs_model = WALEModel()
        # 自由来流条件**无条件**设置（2026-09-18）：单机 FRSolver 的
        # `self.freestream` 从来就是无条件的（solver.py::__init__），这边
        # 却挂在湍流分支里——于是 `turbulence_model=none/les` 的分布式
        # 运行没有 `.freestream`。任何"按来流量级归一化"的消费方都会因此
        # 在这条路径上崩溃或静默退回内置缺省，BJ 越界判据（门控滤波的
        # 默认判据，其绝对地板必须用来流参考量级）正是其中一个。
        # 同一个语义只允许有一个事实来源、且两条后端不能有不同的可用性。
        self.rho_inf = solver_kwargs.get('rho_inf', 1.225)
        self.vel_inf = solver_kwargs.get('vel_inf', 33.33)
        self.p_inf = solver_kwargs.get('p_inf', 101325.0)
        self.freestream = {
            "rho_inf": self.rho_inf, "vel_inf": self.vel_inf,
            "p_inf": self.p_inf,
            # 见 gpu_solver.py 同一处说明（2026-09-17）
            "aoa_deg": float(solver_kwargs.get("aoa_deg", 0.0) or 0.0),
            "aos_deg": float(solver_kwargs.get("aos_deg", 0.0) or 0.0),
        }
        # 决定物理解的参数同样**无条件**设置（2026-09-25）：此前只在
        # SST/DDES/IDDES/WMLES 分支里设，于是 none/les 运行的分布式 checkpoint
        # 按替身缺省值写粘度与 Tu/VR（见 core/utils/checkpoint_physics.py）。
        self.mu_molecular = solver_kwargs.get('mu_molecular', 1.8e-5)
        self._turbulence_intensity = solver_kwargs.get('turbulence_intensity', 0.01)
        self._viscosity_ratio = solver_kwargs.get('viscosity_ratio', 5.0)
        if turb_model_upper in ('SST', 'DDES', 'IDDES', 'WMLES'):
            if turb_model_upper in ('SST', 'DDES', 'IDDES'):
                from autoflowcfd.core.fr_solver.turbulence import init_turbulence_models
                init_turbulence_models(self, n_local, n_sps)  # 设置 self.turb_model,
                # self._turb_ramp_step/_turb_production_ramp_steps（产项渐变，
                # 见该函数与 _update_production_ramp 文档）。distributed_compute_
                # turbulence_source_and_viscosity 内部用的是一个每步都重新构造
                # 的临时 adapter 对象，_turb_ramp_step 的递增不会自动持久化，
                # step() 显式在每次调用后把结果写回 self._turb_ramp_step
                # （见该方法对应注释）。

                self.turb_halo_exchange = HaloExchange(self.partition, n_sps, 2)
            else:
                # WMLES（2026-09-02）：没有 k/omega ODE 状态，不需要
                # init_turbulence_models/turb_halo_exchange——只需要真实
                # 的 CPU 版 WMLESModel 实例（与单机 `gpu_solver.py`/CPU
                # `FRSolver.__init__` 构造 wmles_model 同一个模式）+
                # 下面统一计算的 wall_distance_compact（y+ 计算需要）。
                from autoflowcfd.core.turbulence.wmles import WMLESModel
                self.wmles_model = WMLESModel(nu=self.mu_molecular / max(self.rho_inf, 1e-10))

            # 壁面距离：与单机同一个来源（CLI 由体网格 WALL 边界面构造后以
            # `wall_distance_source=` 传入），在本 rank compact 解点上查询；
            # 换阶重建时用同一个来源重查（distributed_order_continuation）
            from autoflowcfd.core.mpi.distributed_turbulence import compute_distributed_wall_distance
            self._wall_distance_source = wall_distance_source
            self.wall_distance_compact = compute_distributed_wall_distance(
                self.dist_flat_face, mesh, wall_distance_source)

            # DDES/IDDES（2026-09-02）：`init_turbulence_models` 的 DDES/
            # IDDES 分支已经把 `self.ddes_model` 设成真实的 DDESModel/
            # IDDESModel 实例（复用单机同一份初始化逻辑，鸭子类型只需要
            # `.turb_model_name`）。IDDES 分支还顺带把
            # `self._iddes_h_max`/`self._iddes_h_wn` 设成了**全局**尺寸
            # （`compute_h_max_and_h_wn(solver.mesh)` 用 `self.mesh`——
            # "传统模式"下就是完整全局网格，与 h_max/h_wn 纯粹是"每个
            # 单元自身节点坐标的函数、和相邻单元/分区无关"这一事实一致，
            # 见 des.py::compute_h_max_and_h_wn 文档）——这里只需要按
            # compact_global_ids 切一次片，得到分布式 adapter 真正需要
            # 的 compact 索引空间版本，不需要任何新的跨 rank 几何交换。
            if turb_model_upper in ('DDES', 'IDDES'):
                assert self.ddes_model is not None, (
                    "init_turbulence_models 应该已经为 DDES/IDDES 设置了 "
                    "self.ddes_model，这里是 None 说明该函数的对应分支"
                    "被跳过或修改，需要检查 turbulence.py::init_turbulence_"
                    "models 的 DDES/IDDES 分支是否还在。"
                )
            if turb_model_upper == 'IDDES':
                compact_global_ids = self.dist_flat_face.compact_global_ids
                self.iddes_h_max_compact = self._iddes_h_max[compact_global_ids]
                self.iddes_h_wn_compact = self._iddes_h_wn[compact_global_ids]
            elif turb_model_upper == 'DDES':
                # DDES（2026-09-02 补齐，与 IDDES 同一处理）：
                # `init_turbulence_models` 现在也会为 DDES 分支设置
                # `self._iddes_h_max`（见 fr_solver/turbulence.py 对应
                # 分支——`apply_to_sst_model` 用它换成各向异性感知的
                # max_edge 网格尺度，不再是 cube_root(V)），这里同样切
                # 一次片得到 compact 索引空间版本。DDES 不需要 h_wn
                # （那是 IDDES 专属的近壁法向间距量），不切它。
                compact_global_ids = self.dist_flat_face.compact_global_ids
                self.iddes_h_max_compact = self._iddes_h_max[compact_global_ids]

            if turb_model_upper in ('DDES', 'IDDES'):
                # 真实 bug 修复（2026-09-02，两次连续调用才测出来）：
                # `des_length_scale` 是跨步持久状态，需要与 k_field/
                # omega_field 同一套 halo 交换+compact 重排才能在
                # 第二次及以后的调用里正确使用，见 distributed_
                # turbulence.py::distributed_compute_turbulence_source_
                # and_viscosity 对应修复文档。只有 1 个分量，不能复用
                # 2-var 的 turb_halo_exchange。
                self.des_length_scale_halo_exchange = HaloExchange(self.partition, n_sps, 1)

        # 7. 时间推进器：与单机 FRSolver 同一套 Shu-Osher SSP-RK3 stage
        # 实现（core/time_integration/base.py），保证分布式与单机路径
        # 时间精度一致（见 step() 文档：此前这里是自称"RK3"实际执行单步
        # 前向欧拉的简化实现）。dt 在这里只是占位——真正推进用的步长由
        # step() 每次显式构造的 dt_local 数组决定，不读 self.dt。
        #
        # 真实 bug 修复（2026-09-02，用户明确要求"不允许出现完成度不是
        # 100%的功能点"后排查发现）：此前这里无条件硬编码
        # `scheme=TimeIntegrationScheme.SSP_RK3`，完全忽略调用方通过
        # `solver_kwargs['time_scheme']` 传入的值——DUAL_TIME（真正
        # 时间精度的瞬态仿真，DES/LES 场景应该用的模式）请求了也会被
        # 静默换成稳态收敛加速模式，`step()` 内部原本也只会
        # `_ssp_rk_stage_step`，即便这里读对了 scheme 也无路可走。
        # 现在真正读取 `time_scheme`/`dual_time_inner_iter`，`step()`
        # 相应按 scheme 分派（与单机 `fr_solver/step.py` 同一个设计）。
        time_scheme = require_distributed_scheme(
            solver_kwargs.get('time_scheme', TimeIntegrationScheme.SSP_RK3))
        dual_time_steps = solver_kwargs.get('dual_time_inner_iter', 20)
        self._time_integrator = TimeIntegrator(
            scheme=time_scheme, dt=1.0, dual_time_steps=dual_time_steps,
        )
        # DUAL_TIME 模式下 BDF2 需要的上一物理时间层状态（None 表示
        # 尚未跑过一个物理步，退化为 BDF1——与单机
        # `solver._dual_time_U_prev` 同一个约定）。
        self._dual_time_U_prev = None
        # NEWTON_KRYLOV 跨步状态（构造时置初值，理由见 reset_newton_state 文档）
        from autoflowcfd.core.time_integration.implicit.mean_flow_step import reset_newton_state

        reset_newton_state(self)

        # 自适应 CFL + 低马赫数伪时间预处理（2026-09-14 补齐）。
        # 这两个机制此前在分布式路径上都不存在，因为它们都以"存在一个由
        # CFL 数决定的逐单元局部步长"为前提，而这条路径当时用全局固定
        # dt（被记作"已接受的简化"）。局部步长已补齐（见
        # `_compute_distributed_local_time_step`），这两个机制随之接入，
        # 语义与单机 `FRSolver` 完全一致：
        #   * 控制器按**全局**残差范数更新（allreduce 之后的那个值），
        #     所有 rank 因此得到同一个 CFL 数——这是分布式下唯一正确的
        #     做法，按各自的局部残差更新会让各 rank 的 CFL 漂移、
        #     破坏一致性；
        #   * 预处理只在 SSP-RK2/RK3 下启用（DUAL_TIME 的物理时间导数项
        #     与 IMEX 的残差拆分都需要单独推导 Gamma 的分配方式）；
        #   * 环境变量 AFCFD_LOW_MACH_PRECOND / AFCFD_CFL_LEGACY 同样生效。
        # CFL 策略（控制器 or 固定 CFL）的唯一事实来源：
        # `time_integration/adaptive_cfl/policy.py::build_cfl_policy`（六个后端
        # 构造点此前各写一份且已分叉，见该模块文档）。
        # 此前这里只对 SSP-RK2/RK3 建控制器，其余方案（IMEX、DUAL_TIME 的伪
        # 时间步）走 cfl.py 的替身回退 0.1，与单机的 cfl_start 不一致。
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
        self._cfl_controller, self.fixed_cfl_number = build_cfl_policy(
            time_scheme, cfl_start=solver_kwargs.get('cfl_start'),
            cfl_max=solver_kwargs.get('cfl_max'), cfl_min=solver_kwargs.get('cfl_min'))
        from autoflowcfd.core.utils.preconditioning import resolve_low_mach_precond
        self.low_mach_precond_enabled = resolve_low_mach_precond(
            solver_kwargs.get('low_mach_precond', True), time_scheme)
        # 上一步的涡粘场（local 排列），供下一步的粘性 CFL 限制使用——
        # 与单机 `_get_turbulent_viscosity_field` 读取湍流模型已存字段
        # （即上一步的结果）是同一个时序。
        self._prev_mu_t_local = None

        # 8. 同步
        barrier()
