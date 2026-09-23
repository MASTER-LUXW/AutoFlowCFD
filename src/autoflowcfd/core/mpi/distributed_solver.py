"""
AutoFlowCFD V2.0 - 分布式 FRSolver

将 FRSolver 扩展为 MPI 域分解版本。每个 rank 持有 local cells 的数据，
通过 halo 交换获取邻居信息，独立计算 local cells 的残差。

设计:
- 组合模式：DistributedFRSolver 持有 FRSolver 实例 + 分区/通信基础设施
- 覆盖残差计算方法：先 halo 交换，再调用单机残差（只处理 local cells）
- 全局操作（残差范数、时间步长）通过 MPI Allreduce

使用:
    # 所有 rank 加载同一网格文件
    mesh = load_mesh(grid_file)
    solver = DistributedFRSolver(mesh, ...)
    # 每个 rank 自动获取自己的分区，执行分布式求解
    for step in range(n_steps):
        solver.step(dt)
"""

import os

import numpy as np
from typing import Optional
from loguru import logger

from autoflowcfd.core.mpi import get_rank, get_size, is_root, mpi_available
from autoflowcfd.core.mpi.partition import (
    partition_mesh, build_distributed_partition, DistributedPartition
)
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.distributed_flat_face import (
    DistributedFlatFaceGeometry, build_distributed_flat_face
)
from autoflowcfd.core.mpi.comm import allreduce_sum, barrier
from autoflowcfd.core.time_integration.base import TimeIntegrator, TimeIntegrationScheme


class DistributedFRSolver:
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
        if turb_model_upper in ('SST', 'DDES', 'IDDES', 'WMLES'):
            self.mu_molecular = solver_kwargs.get('mu_molecular', 1.8e-5)
            self._turbulence_intensity = solver_kwargs.get('turbulence_intensity', 0.01)
            self._viscosity_ratio = solver_kwargs.get('viscosity_ratio', 5.0)

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

            wall_node_indices = None
            # 真实 bug 修复（2026-09-02，排查多GPU分布式SST时发现同一处
            # 拷贝粘贴的 bug，本文件同样中招）：`hasattr(mesh,
            # 'boundary_groups')` 对"属性存在但值是 None"恒为 True，
            # `.items()` 会真实 AttributeError——用 `getattr(...) is not
            # None` 才是正确的存在性判据。
            boundary_groups = getattr(mesh, 'boundary_groups', None)
            if boundary_groups is not None:
                for bg_name, bg in boundary_groups.items():
                    if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                        wall_node_indices = bg.get('node_indices')
                        break
            from autoflowcfd.core.mpi.distributed_turbulence import compute_distributed_wall_distance
            self.wall_distance_compact = compute_distributed_wall_distance(
                self.partition, self.dist_flat_face, mesh, wall_node_indices,
            )

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
        time_scheme = solver_kwargs.get('time_scheme', TimeIntegrationScheme.SSP_RK3)
        dual_time_steps = solver_kwargs.get('dual_time_inner_iter', 20)
        self._time_integrator = TimeIntegrator(
            scheme=time_scheme, dt=1.0, dual_time_steps=dual_time_steps,
        )
        # DUAL_TIME 模式下 BDF2 需要的上一物理时间层状态（None 表示
        # 尚未跑过一个物理步，退化为 BDF1——与单机
        # `solver._dual_time_U_prev` 同一个约定）。
        self._dual_time_U_prev = None

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
        self._cfl_controller = None
        if time_scheme in (TimeIntegrationScheme.SSP_RK2,
                           TimeIntegrationScheme.SSP_RK3):
            from autoflowcfd.core.time_integration.adaptive_cfl import (
                AdaptiveCFLController,
            )
            # **不再硬编码兜底默认值**（2026-09-17）：此前这里写死
            # `cfl_start=0.1, cfl_max=0.5`，于是控制器默认值一改（同日按
            # 直接谱测量与真实网格失效点重定为 0.03/0.06）分布式路径就与
            # 单机路径脱节——`test_distributed_solver_main_init.py::
            # TestDistributedStepMatchesSingleMachine` 当场测出 dt 相差
            # 2.33 倍。现在只传**非 None** 的键，默认值的单一事实来源是
            # `AdaptiveCFLController.__init__` 的签名。
            _cfl_kw = {k: solver_kwargs[k]
                       for k in ('cfl_start', 'cfl_max', 'cfl_min')
                       if solver_kwargs.get(k) is not None}
            self._cfl_controller = AdaptiveCFLController(**_cfl_kw)
        _env_pc = os.environ.get("AFCFD_LOW_MACH_PRECOND")
        _req_pc = (bool(solver_kwargs.get('low_mach_precond', True))
                   if _env_pc is None else (_env_pc == "1"))
        self.low_mach_precond_enabled = _req_pc and time_scheme in (
            TimeIntegrationScheme.SSP_RK2, TimeIntegrationScheme.SSP_RK3)
        # 上一步的涡粘场（local 排列），供下一步的粘性 CFL 限制使用——
        # 与单机 `_get_turbulent_viscosity_field` 读取湍流模型已存字段
        # （即上一步的结果）是同一个时序。
        self._prev_mu_t_local = None

        # 8. 同步
        barrier()

    @classmethod
    def from_fully_distributed_package(cls, package: dict, n_ranks: int, rank: Optional[int] = None,
                                        root_context: Optional[dict] = None):
        """真正的"完全分布式加载"构造入口（2026-09-02，见 core/mpi/
        distributed_mesh_loader.py::distributed_mesh_load_v2/
        build_fully_distributed_rank_package 模块文档）。

        与主构造函数（`__init__`，"传统模式"：每个 rank 独立加载完整
        全局网格）的关键区别：这里的 `package` 是 root rank 预先算好、
        已经按本 rank 的 compact 索引空间切好的紧凑数据（`partition`/
        `dist_fc`/`PrecompactedMeshData`/切好 `group_code` 的边界幽灵态
        提供者），本 rank 从未持有、也不需要持有完整全局网格——这是
        "完全分布式加载"名副其实的内存优化。

        用 `cls.__new__(cls)` 绕开主 `__init__`（那个构造函数的很多步骤
        ——如 `build_distributed_partition`/`build_distributed_flat_face`
        ——本身就要求一个完整全局网格，在这条路径下没有意义、也没有
        数据可用），直接把 package 里已经算好的内容赋到对应属性上。

        范围边界（与 `build_fully_distributed_rank_package` 文档一致，
        这里重复一遍避免只读一处文档漏掉）：支持 `turbulence_
        model='none'/'sst'/'ddes'/'iddes'/'wmles'/'les'`（2026-09-02
        补齐 SST/DDES/IDDES，同日续接 WMLES/LES——两者此前"需要额外
        基础设施"的排除理由排查后不成立：WMLES 壁面剪应力修正只是
        纯逐 owner 单元的局部操作，真正的障碍是 `compute_wmles_wall_
        stress_correction` 没有跟随 `flat_face_override` 约定，现已
        修复；LES（WALE）纯代数现算，不需要 root 预计算任何几何量）。
        DUAL_TIME、checkpoint 已接入。Order Continuation（2026-09-02
        续接）——见 `core/mpi/distributed_order_continuation.py` 模块
        文档：本 rank 只持有 compact 数据，阶数切换需要 root 用完整
        全局网格重新算一遍紧凑包再重新分发（"完全分布式加载"名副
        其实的代价），本方法本身不做这件事，由
        `DistributedFRSolver._interpolate_to_new_order` 调用
        `distributed_order_continuation.py::redistribute_fully_
        distributed_for_new_order`（需要 root 持有的 `_root_context`，
        见下面 `root_context` 参数）触发。

        Args:
            package: `build_fully_distributed_rank_package` 的返回值
                （或 `distributed_mesh_load_v2` 经 MPI 收发后本 rank
                收到的那一份）
            n_ranks: MPI rank 总数
            rank: 当前 rank（默认从 MPI 获取）
            root_context: 仅 root rank（rank 0）需要非 None——
                `distributed_mesh_load_v2` 返回的第二个值（见该函数
                文档"Order Continuation 支持"一节），持有完整全局
                `mesh`/`ops`/`cell_partition`/`face_connectivity`/
                `boundary_ghost_provider_global`/冻结的 root 端配置，
                供后续阶数切换时重新计算+重新分发紧凑包。非 root rank
                永远传 None（它们从未持有、也不需要这份数据）。不提供
                时（None）意味着这个 solver 实例无法执行 Order
                Continuation（`order>=2` 时 `solve()` 会 fail-fast
                拒绝而不是静默跳过升阶爬坡）。

        Returns:
            DistributedFRSolver 实例
        """
        import types
        from autoflowcfd.fr.operators import generate_fr_operators

        self = cls.__new__(cls)
        self.rank = rank if rank is not None else get_rank()
        self._is_fully_distributed = True
        self._root_context = root_context
        self.order = int(package['order'])
        self.current_order = self.order
        self.order_continuation_enabled = package.get('order_continuation_enabled', True)
        self.n_ranks = n_ranks

        precompacted_mesh = package['precompacted_mesh']
        self.mesh = precompacted_mesh
        # 每个 rank 本地重新生成算子（纯函数，只依赖 order/tet_basis_
        # mode/flux_point_type，见 fr/operators.py），而不是把 root 算好
        # 的 FROperators 对象整个 pickle 发过来——避免不必要的大数组
        # 序列化开销（微分算子矩阵与单元数无关，每个 rank 反正都要
        # 用同一份，本地重算比跨进程传输更便宜）。
        self.ops = generate_fr_operators(package['order'])

        self.partition = package['partition']
        self.dist_flat_face = package['dist_fc']

        n_sps = precompacted_mesh.n_sps_per_cell
        n_vars = 5
        self.state = DistributedFRState(self.partition, n_sps, n_vars)
        # 均匀自由流场初始化（同一处真实 bug 修复，见
        # DistributedFRState.initialize_uniform 文档）——"完全分布式
        # 加载"路径同样从未初始化过 state，同一个根因。
        # 初场方向同上（2026-09-17）；aoa/aos 由 root 打进 package 的
        # freestream 字典，见 distributed_mesh_loader.distributed_mesh_load_v2
        from autoflowcfd.core.utils.flow_direction import freestream_velocity
        _v0 = freestream_velocity(
            package['freestream'].get('vel_inf', 33.33),
            package['freestream'].get('aoa_deg', 0.0) or 0.0,
            package['freestream'].get('aos_deg', 0.0) or 0.0)
        self.state.initialize_uniform(
            rho=package['freestream'].get('rho_inf', 1.225),
            u=float(_v0[0]), v=float(_v0[1]), w=float(_v0[2]),
            p=package['freestream'].get('p_inf', 101325.0),
        )
        self.halo_exchange = HaloExchange(self.partition, n_sps, n_vars)

        self.solver_kwargs = {}
        # SST/DDES/IDDES/WMLES（2026-09-02 补齐）：root 已经把
        # wall_distance/h_max/h_wn 算好、按本 rank 的 compact 索引空间
        # 切好放进 package，这里只需要构造真正的 turb_model（compact
        # 索引空间外的部分，即 n_local 大小的持久状态）+ 对应的 halo
        # 交换器，不需要重新算任何几何量。
        turb_model_name = package.get('turb_model_name', 'NONE')
        # fail-fast 护栏（真实 bug 修复，2026-09-02）：拒绝任何拼写错误/
        # 未知的 turb_model_name，而不是静默把 `self.turb_model_name`
        # 设成该值但 `self.turb_model`/`.wmles_model`/`.sgs_model` 都
        # 留空——`step()` 会静默把它当成 'none' 跑（湍流物理完全缺失，
        # 不会有任何报错或警告）。
        if turb_model_name not in ('NONE', 'SST', 'DDES', 'IDDES', 'WMLES', 'LES'):
            raise NotImplementedError(
                f"DistributedFRSolver.from_fully_distributed_package: "
                f"'完全分布式加载' 模式目前只支持 turbulence_model="
                f"'none'/'sst'/'ddes'/'iddes'/'wmles'/'les'，收到的是 "
                f"'{turb_model_name}'。"
            )
        self.turb_model_name = turb_model_name
        self._turb_model_upper = turb_model_name
        # Order Continuation 支持（2026-09-02，见 core/mpi/distributed_
        # mesh_loader.py::redistribute_fully_distributed_for_new_order
        # 文档）：`self.freestream` 只在 SST/DDES/IDDES 分支才会被设置
        # （见下方），但阶数切换时需要对**任意** turb_model_name 都能
        # 拿到自由来流条件（重置 P0 均匀流场需要），这里存一份通用的、
        # 不依赖 turb_model_name 分支的副本。
        self._package_freestream = package['freestream']
        # 与传统模式同一条（见那边说明）：无条件设置 `self.freestream`。
        # 这里不带 mach_ref（下面 SST 分支会覆写成含 mach_ref 的版本），
        # 消费方只依赖 rho_inf/vel_inf/p_inf。
        self.freestream = dict(package['freestream'])
        self.turb_model = None
        self.turb_halo_exchange = None
        self.wall_distance_compact = package.get('wall_distance_compact')
        self.ddes_model = None
        self.iddes_h_max_compact = package.get('iddes_h_max_compact')
        self.iddes_h_wn_compact = package.get('iddes_h_wn_compact')
        self.des_length_scale_halo_exchange = None
        # `step()` 的 `residual_func` 无条件读取 `self.wmles_model`/
        # `self.sgs_model`（与主 `__init__` 同一个属性名约定），必须
        # 存在这两个属性，否则任何 turb_model_name 都会在 step() 里
        # AttributeError——两者默认 None，只在下面对应分支里真正构造。
        self.wmles_model = None
        self.sgs_model = None

        if turb_model_name == 'LES':
            # LES（2026-09-02）：WALE 纯代数 SGS 模型，不需要 root 预
            # 计算任何几何量（不像 SST 需要 wall_distance），package
            # 里也没有为它准备任何字段——直接构造即可。
            from autoflowcfd.core.turbulence.sgs import WALEModel
            self.sgs_model = WALEModel()
        elif turb_model_name == 'WMLES':
            # WMLES（2026-09-02）：没有 k/omega ODE 状态，不需要
            # SSTModelFR/turb_halo_exchange——只需要真实的 WMLESModel
            # 实例 + package 里 root 已经算好的 wall_distance_compact
            # （y+ 计算需要，上面 `self.wall_distance_compact = package.
            # get('wall_distance_compact')` 已经取到）。
            from autoflowcfd.core.turbulence.wmles import WMLESModel
            mu_molecular = package['mu_molecular']
            rho_inf = package['freestream'].get('rho_inf', 1.225)
            self.wmles_model = WMLESModel(nu=mu_molecular / max(rho_inf, 1e-10))

        if turb_model_name in ('SST', 'DDES', 'IDDES'):
            self.mu_molecular = package['mu_molecular']
            self.freestream = {**package['freestream'], "mach_ref": package['mach_ref']}
            self._turbulence_intensity = package.get('turbulence_intensity', 0.01)
            self._viscosity_ratio = package.get('viscosity_ratio', 5.0)

            n_local = self.partition.n_local_cells
            # 直接复用单机路径同一套 Tu/VR 推导 k_inf/omega_inf +
            # k_max/omega_max 物理上界公式（`_set_freestream_
            # turbulence`/`_set_turbulence_bounds` 都是鸭子类型函数，
            # 只需要 `.freestream`/`.mu_molecular`/`._turbulence_
            # intensity`/`._viscosity_ratio`/`.turb_model`，`self` 此时
            # 已经全部满足）——不直接调用 `init_turbulence_models`，
            # 因为它的 IDDES 分支会用 `solver.mesh` 重新计算 h_max/
            # h_wn（这里 `self.mesh` 是 `PrecompactedMeshData`，没有
            # 完整节点坐标/连接关系，算不出来），而 h_max/h_wn 本来就
            # 已经由 root 算好放进 package 了，不需要重算。
            from autoflowcfd.core.fr_solver.turbulence import (
                _set_freestream_turbulence, _set_turbulence_bounds,
            )
            from autoflowcfd.core.turbulence.sst import SSTModelFR
            k_inf, omega_inf = _set_freestream_turbulence(self)
            self.turb_model = SSTModelFR(n_local, n_sps, k_inf=k_inf, omega_inf=omega_inf)
            _set_turbulence_bounds(self)
            self._turb_ramp_step = 0
            self._turb_production_ramp_steps = 50
            self._turb_production_ramp_complete = False

            self.turb_halo_exchange = HaloExchange(self.partition, n_sps, 2)

            if turb_model_name == 'DDES':
                from autoflowcfd.core.turbulence.des import DDESModel
                self.ddes_model = DDESModel()
            elif turb_model_name == 'IDDES':
                from autoflowcfd.core.turbulence.des import IDDESModel
                self.ddes_model = IDDESModel()

            if self.ddes_model is not None:
                self.des_length_scale_halo_exchange = HaloExchange(self.partition, n_sps, 1)

        # 轻量级鸭子类型"local_solver"替身（不构造真正的 FRSolver——那
        # 需要完整全局网格重新生成 sps 几何/差分算子，在这条路径下既
        # 没有数据也没有必要）：`step()` 只读它的 `.config.physics.
        # enable_viscous`/`.mu_molecular`/`.boundary_ghost_provider`/
        # `.freestream["mach_ref"]` 这 4 个属性（已逐行核实
        # distributed_solver.py 全文只有这 4 处 `self.local_solver.`
        # 访问），不需要真正的 FRSolver 实例。
        self._local_solver = types.SimpleNamespace(
            config=types.SimpleNamespace(
                physics=types.SimpleNamespace(enable_viscous=package['enable_viscous'])
            ),
            mu_molecular=package['mu_molecular'],
            boundary_ghost_provider=package['boundary_ghost_provider'],
            freestream={**package['freestream'], "mach_ref": package['mach_ref']},
        )
        # `_p0_global_boundary_ghost_provider` 是"传统模式"专用的全局
        # provider（本 rank 持有完整全局网格时才有意义），"完全分布式
        # 加载"下恒为 None——但这**不代表** P0 阶段的分布式残差本身
        # 不支持：那条路径改用 `distributed_order_continuation.py::
        # compute_distributed_p0_inviscid_residual` 里 `_is_
        # fully_distributed` 分支的另一套机制（只有 root 通过
        # `self._root_context` 持有的完整网格计算，算完 broadcast 给
        # 全部 rank），2026-09-02 已实现，见该函数文档。
        self._p0_global_boundary_ghost_provider = None

        # 真实读取 package 里的 time_scheme/dual_time_inner_iter（不再
        # 硬编码 SSP_RK3）——`build_fully_distributed_rank_package`
        # 2026-09-02 已接入这两个字段（见 distributed_mesh_loader.py
        # 模块文档"DUAL_TIME 支持"一节），`.get(...)` 默认值只是兼容
        # 没有这两个字段的旧 checkpoint/package。
        time_scheme = package.get('time_scheme', TimeIntegrationScheme.SSP_RK3)
        dual_time_steps = package.get('dual_time_inner_iter', 20)
        self._time_integrator = TimeIntegrator(
            scheme=time_scheme, dt=1.0, dual_time_steps=dual_time_steps,
        )
        self._dual_time_U_prev = None

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
        self._cfl_controller = None
        if time_scheme in (TimeIntegrationScheme.SSP_RK2,
                           TimeIntegrationScheme.SSP_RK3):
            from autoflowcfd.core.time_integration.adaptive_cfl import (
                AdaptiveCFLController,
            )
            # None 感知（2026-09-15）：package 现在**总是**带 cfl_* 三个键
            # （CLI 未指定时值为 None），所以不能用 `.get(k, default)`
            # ——那会拿到显式的 None 而不是 default。
            # **不再硬编码兜底默认值**（2026-09-17）：此前这里写死
            # `cfl_start=0.1, cfl_max=0.5`，于是控制器默认值一改（同日按
            # 直接谱测量与真实网格失效点重定为 0.03/0.06）分布式路径就与
            # 单机路径脱节——`test_distributed_solver_main_init.py::
            # TestDistributedStepMatchesSingleMachine` 当场测出 dt 相差
            # 2.33 倍。现在只传**非 None** 的键，默认值的单一事实来源是
            # `AdaptiveCFLController.__init__` 的签名。
            _cfl_kw = {k: package[k]
                       for k in ('cfl_start', 'cfl_max', 'cfl_min')
                       if package.get(k) is not None}
            self._cfl_controller = AdaptiveCFLController(**_cfl_kw)
        _env_pc = os.environ.get("AFCFD_LOW_MACH_PRECOND")
        _req_pc = (bool(package.get('low_mach_precond', True))
                   if _env_pc is None else (_env_pc == "1"))
        self.low_mach_precond_enabled = _req_pc and time_scheme in (
            TimeIntegrationScheme.SSP_RK2, TimeIntegrationScheme.SSP_RK3)
        # 上一步的涡粘场（local 排列），供下一步的粘性 CFL 限制使用——
        # 与单机 `_get_turbulent_viscosity_field` 读取湍流模型已存字段
        # （即上一步的结果）是同一个时序。
        self._prev_mu_t_local = None

        barrier()
        return self

    @property
    def local_solver(self):
        """延迟初始化本地求解器（避免循环依赖）。

        FRSolver.__init__ 的第二个位置参数是 order: int（内部会自己调用
        generate_fr_operators(order) 重新构建算子，不接受外部预构建的
        FROperators 对象作为构造参数）——此前这里错误地把 self.ops
        （DistributedFRSolver 构造时外部传入的 FROperators 实例）当成
        order 位置传参，导致 `order + 1` 处 TypeError（真实复现，
        DistributedFRSolver(..., n_ranks=1, turb_model_name='none') 后
        调用 step() 必现）；生产 CLI 路径（solve_steady_command.py）还
        会同时把 order=order 塞进 solver_kwargs，两者相加变成
        "got multiple values for argument 'order'"，同一个根因。
        用 mesh.order（构建 local_mesh 时用的同一个阶数）作为默认值，
        solver_kwargs 里若显式提供了 order（生产路径就是如此，且与
        mesh.order 恒一致，见 solve_steady_command.py）则优先使用它，
        避免与下面的显式关键字参数重复传参报错。
        """
        if self._local_solver is None:
            from autoflowcfd.core.fr_solver.solver import FRSolver
            kwargs = dict(self.solver_kwargs)
            kwargs.setdefault('order', self.mesh.order)
            self._local_solver = FRSolver(self.mesh, **kwargs)

            # 真实 bug 修复（2026-09-02，实现分布式湍流模型时排查发现，
            # 与湍流本身无关——任何使用真实 WALL/INLET/OUTLET/FARFIELD/
            # SYMMETRY 区分的分布式算例都会中招，与 gpu_distributed.py
            # 同一处修复同一个根因）：`FRSolver(self.mesh, ...)` 用
            # `self.mesh`（完整全局网格，"传统模式"下如此）构造的
            # `boundary_ghost_provider.group_code` 是**全局**面编号索引，
            # 但 `distributed_compute_inviscid_residual`/`_viscous_residual`
            # 最终调用 `ghost_provider(f, ...)` 时 `f` 是本 rank 的
            # local+halo**压缩索引空间**面编号——两套编号不是同一个索引
            # 空间的子区间（见 `distributed_flat_face.py::partition.
            # local_faces` 文档）。真实合成网格验证：2-rank 分区，8/12
            # （67%）压缩面会被分配到错误的边界组编码。修复：把
            # `group_code` 重映射到压缩索引空间。
            provider = getattr(self._local_solver, 'boundary_ghost_provider', None)
            if provider is not None and hasattr(provider, 'group_code'):
                # P0 分布式残差路径（2026-09-02，见 core/mpi/
                # distributed_order_continuation.py 模块文档"P0 有限
                # 体积残差"一节）需要一份**未经压缩索引空间重映射**的
                # provider——P0 kernel 直接对 `self.mesh`（完整全局网格）
                # 逐全局面 id 调用，不经过下面这行的压缩重映射。在原地
                # 覆写 `group_code` 之前，先浅拷贝一份 provider 对象、
                # 保留原始全局 `group_code`，供 P0 路径使用；下面这行
                # 仍然原地重映射 `self._local_solver.boundary_ghost_
                # provider` 本身（P1+ 路径继续使用压缩索引空间版本，
                # 行为不变）。
                import copy as _copy
                self._p0_global_boundary_ghost_provider = _copy.copy(provider)
                provider.group_code = provider.group_code[self.partition.local_faces]
        return self._local_solver

    def _interpolate_to_new_order(self, target_p: int) -> None:
        """将解从当前阶数插值到新的阶数（分布式 Order Continuation
        核心逻辑，2026-09-02，见 core/mpi/distributed_order_
        continuation.py 模块文档）——按本实例的构造方式分派到对应的
        重建函数：'传统模式'（`self._is_fully_distributed is False`，
        每个 rank 持有完整全局网格）复用同一进程内的 `cell_partition`
        重建 partition/dist_flat_face；'完全分布式加载'
        （`self._is_fully_distributed is True`）需要 root 重新计算+
        重新分发紧凑包，见 `redistribute_fully_distributed_for_new_
        order` 文档。
        """
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            cpu_traditional_interpolate_to_new_order,
        )

        if self._is_fully_distributed:
            from autoflowcfd.core.mpi.distributed_mesh_loader import (
                redistribute_fully_distributed_for_new_order,
            )
            redistribute_fully_distributed_for_new_order(self, target_p)
        else:
            cpu_traditional_interpolate_to_new_order(self, target_p)

    def exchange_halo(self, U_local: np.ndarray) -> np.ndarray:
        """执行 halo 交换。

        Args:
            U_local: (n_local_cells, n_sps, n_vars) local cell 数据

        Returns:
            U_extended: (n_total_cells, n_sps, n_vars) 含 halo 的扩展数据
        """
        return self.halo_exchange.exchange(U_local)

    def _build_sensor_gated_filter_func_distributed(
        self, n_local: int, n_sps: int, cell_is_prism: np.ndarray
    ):
        """CPU MPI 路径的**传感器门控**模态滤波回调（2026-09-18 接线）。

        `AFCFD_FILTER_MODE=sensor` 是 2026-09-17 定下的默认档，但当时只有
        单机 CPU 接线，本后端会退回 `project`（全局逐 stage 施加精确投影
        = 功能上等于 legacy，P1 退化成 P0、壁面剪应力恒为零）。这里补齐。

        缺的那一块是 BJ 越界判据要**面邻居的单元均值**，而分区边界上的
        邻居是 halo 单元。三点必须说明：

        1. **不能把分区边界面当边界面排除。** 分区边界不是物理边界，
           排除它等于在任意位置人为切断邻域包络，掩码会随 rank 数变化
           —— 同一个算例换分区数得到不同结果，这对求解器不可接受。
        2. **必须用当前 stage 的解做扩展，不能复用残差求值时缓存的
           `U_extended`。** 滤波在每个 stage 的正定性投影之后施加，那时
           解已经更新过；用缓存值会比单机路径滞后，两条后端就不再逐位
           可比。代价是每个 stage 多一次 halo 交换——与
           `_compute_distributed_local_time_step` 同一个既有权衡（局部 dt
           的谱半径同样需要额外一次交换），而滤波只在 `troubled` 非空时
           才真正动数据，交换本身是 `n_halo * n_sps * n_vars`。
        3. **索引空间**：`state.U` / 滤波回调的 `U_flat` 都是 halo 交换的
           **原生**排列（local 在前、halo 在后），而 `dist_flat_face` 的
           `owner_cell_local`/`neighbor_cell_local` 是"棱柱在前"的**紧凑**
           排列。两者用 `perm` 换算：紧凑下标 k 对应原生下标 `perm[k]`
           （因为 `array_native[perm] == array_permuted`），所以
           `owner_native = perm[owner_cell_local]`。掩码在原生扩展空间上
           算，取前 `n_local` 项；halo 行的邻域不完整、算出的掩码丢弃。

        判据内核与后端无关（`bounds_sensor.compute_bounds_violation_mask`
        是纯数组接口），这里不复制一份实现——只提供索引换算与 halo 扩展。
        """
        from autoflowcfd.core.fr_solver.filter import (
            _warn_distributed_face_stencil,
            build_distributed_bounds_conn,
            build_sensor_gated_filter_func_arrays,
        )
        from autoflowcfd.core.fr_operators.bounds_sensor import (
            resolve_troubled_sensor,
        )

        sensor = resolve_troubled_sensor()
        conn = {}
        if sensor in ("bounds", "both"):
            # 索引换算、两条自洽性护栏、halo 扩展时机、两张边界表的惰性
            # 构造全部在共享实现里（见 `build_distributed_bounds_conn`），
            # 多 GPU 走的是同一条。
            #
            # provider 取自 `self.local_solver`（惰性属性），它的
            # `group_code` 已经被重切到 `partition.local_faces`——与
            # `dist_flat_face` 同一索引空间，见 `local_solver` 文档里那处
            # "group_code 重映射"的说明。惰性（传 lambda）是因为构造
            # local_solver 本身有代价、且此刻未必已经建好。
            conn = build_distributed_bounds_conn(
                self.dist_flat_face,
                self.partition.n_total_cells,
                lambda: self.halo_exchange,
                lambda: getattr(self.local_solver,
                                "boundary_ghost_provider", None),
                self.freestream)

        order = int(getattr(self, "current_order", self.order))
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
            n_local, n_sps, order,
            self.ops.filter_prism, self.ops.filter_tet,
            cell_is_prism=cell_is_prism, sensor=sensor, **conn)

    def compute_global_residual_norm(self) -> float:
        """全局残差 L2 范数。"""
        return self.state.global_residual_norm()

    def get_load_balance_report(self) -> str:
        """生成分区负载平衡报告。"""
        n_local = self.partition.n_local_cells
        n_halo = self.partition.n_halo
        n_total_global = self.partition.n_global_cells
        ideal = n_total_global / self.n_ranks
        imbalance = abs(n_local - ideal) / ideal * 100

        from autoflowcfd.core.mpi.comm import allreduce_sum
        total_local = allreduce_sum(n_local)
        total_halo = allreduce_sum(n_halo)

        return (
            f"Load balance: rank {self.rank} has {n_local} cells "
            f"(ideal {ideal:.0f}, imbalance {imbalance:.1f}%), "
            f"{n_halo} halo cells. "
            f"Global total: {total_local} cells, {total_halo} halo."
        )

    def _compute_distributed_local_time_step(self, U_local, mu_t_local=None,
                                             return_physical_too: bool = False):
        """逐单元局部 CFL 步长（2026-09-14 补齐，取代此前的全局固定 dt）。

        此前这条路径用调用方传入的全局固定 dt 铺满所有单元，源码里记作
        "已接受的简化"。那条"简化"有三重真实代价（步长被全场最苛刻单元
        卡死、逐 SP 的几何/度量 CFL 保护完全失效、自适应 CFL 与低马赫数
        预处理都无从生效），完整论证与实现见
        `core/mpi/distributed_cfl.py` 模块文档。

        需要一次额外的 halo 交换：局部 dt 的谱半径求和必须读到邻居单元
        （含 halo）的速度与声速，这是这项功能的内在需求，不是可以省掉的
        开销（相比之下 RK3 每个 stage 各有一次交换）。
        """
        from autoflowcfd.core.mpi.distributed_cfl import (
            compute_distributed_local_time_step,
        )
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        dist_fc = self.dist_flat_face
        U_extended = self.halo_exchange.exchange(U_local)
        U_compact = U_extended[dist_fc.perm]
        Q_compact = conserved_to_primitive(U_compact[..., :5])

        # 几何量（jacobians/cell_volumes）复用 DistributedMeshAdapter 的
        # 抽取逻辑——它已经处理好"传统模式 vs 完全分布式加载"两种索引
        # 语义的区别（见该类文档），这里不重复那段判断。
        from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter
        adapter = DistributedMeshAdapter(self.partition, dist_fc, self.mesh, self.ops)

        return compute_distributed_local_time_step(
            U_compact, Q_compact, dist_fc, self.mesh,
            jacobians=adapter.jacobians,
            cell_volumes=adapter.cell_volumes,
            n_local_cells=self.partition.n_local_cells,
            mu_molecular=self.local_solver.mu_molecular,
            freestream=self.local_solver.freestream,
            cfl_controller=self._cfl_controller,
            current_order=getattr(self, "current_order", self.order),
            low_mach_precond_enabled=getattr(
                self, "low_mach_precond_enabled", False),
            mu_t_local=mu_t_local,
            return_physical_too=return_physical_too,
        )

    def step(self, dt: float) -> float:
        """执行一步时间推进（分布式版本）。

        真正的 3-stage Shu-Osher SSP-RK3（与单机 FRSolver 同一套实现，
        `TimeIntegrator._ssp_rk_stage_step`，见 core/fr_solver/step.py::
        step 的 mean_flow_residual，这里的 `residual_func` 是它的分布式
        版本，两套约定必须严格一致）：每个 stage 都要用该 stage 的中间解
        重新做一次完整的 halo 交换 + 残差求值——不能像旧版本那样只算
        一次残差就套用 RK3 的名字（旧实现自己的注释承认"简化为单步
        Euler"，与 SSP-RK3 的时间精度/稳定域完全不是一回事）。

        两个容易踩错、已用独立数值脚本验证过的约定，都严格照抄
        `fr_solver/step.py::step`：
        1. `_ssp_rk_stage_step` 期望的 `solution`/`dt_local` 是展平成
           `(n_local_cells*n_sps, n_vars)` / `(n_local_cells*n_sps,)` 的
           2D/1D 数组（`dt = dt_local[:, None]` 只能对 2D solution 广播），
           不是 `(n_local_cells, n_sps, n_vars)` 的 3D 数组——直接传 3D
           会在第一个 stage 就因广播形状不匹配抛 ValueError。
        2. `compute_*_residual_fr` 返回的是 dU/dt 本身，`residual_func`
           必须返回其**负值**（`TimeIntegrator` 的约定是 dU/dt=-R(U)）；
           旧的 Euler 实现 `U += dt*total_residual` 直接用未取负的和，
           这一点在那个实现里恰好是自洽的（因为它没有经过 R→L=-R 这层
           转换），但复用 `_ssp_rk_stage_step` 就必须显式取负，否则解会
           往错误的时间方向积分。

        SST 湍流模型（2026-09-02）：与单机 `fr_solver/step.py::step` 同一个
        算子分裂设计——湍流源项+输运在物理步开始时求值一次（用当前状态，
        不在每个 RK 子迭代里重算），产出的 `mu_t_field_compact` 在本步
        全部 RK 子阶段内保持不变，供粘性残差 BR1 界面项消费；k/omega 场
        本身用独立于平均流 RK 的显式-半隐式更新（`SSTModelFR.update_
        fields`），不在这里更新。`__init__` 只接受 'none'/'SST'。

        Args:
            dt: 物理时间步长（分布式路径目前用全局固定步长，不做单机
                路径那种逐 cell 局部 CFL 时间步——旧实现本来就是全局
                dt，这里不新增自适应步长这个单独的功能点）

        Returns:
            residual_norm: 全局残差 L2 范数（RK3 第 0 阶段的 dU/dt 范数，
                与旧实现的报告口径一致，用于跨迭代收敛监控）
        """
        from autoflowcfd.core.mpi.distributed_compute import (
            distributed_compute_inviscid_residual,
            distributed_compute_viscous_residual,
        )
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        # 真实 bug 修复（2026-09-02，"传统模式"主 __init__ + 真正 step()
        # 端到端测试此前从未存在，用户明确要求补齐测试覆盖后才发现）：
        # `self.local_solver.config.physics.enable_viscous` 假设
        # `local_solver` 是"完全分布式加载"专用的 `types.SimpleNamespace`
        # 鸭子类型替身（见 `from_fully_distributed_package` 里
        # `config=types.SimpleNamespace(physics=types.SimpleNamespace(
        # enable_viscous=...))` 的构造），但"传统模式"下 `local_solver`
        # 是 `local_solver` 这个 @property 真正构造出的、货真价实的
        # `FRSolver` 实例——`FRSolver` 类本身完全没有 `.config` 属性
        # （单机路径的粘性残差本来就无条件计算，从未有过开关），这里
        # 无条件访问 `.config` 必然 `AttributeError`——"传统模式"这条
        # CLI 生产路径（`solve steady --n-ranks>1`，不加 `--fully-
        # distributed`）因此 100% 必现崩溃，此前从未被任何测试捕捉到。
        # 修复：`local_solver` 没有 `.config` 时（真实 FRSolver 场景）
        # 退回 `True`，与单机 `FRSolver` 的真实行为（粘性残差恒计算）
        # 一致；"完全分布式加载"的替身仍按其显式提供的值。
        _local_config = getattr(self.local_solver, 'config', None)
        enable_viscous = (
            _local_config.physics.enable_viscous if _local_config is not None else True
        )
        mu = self.local_solver.mu_molecular
        boundary_ghost_provider = self.local_solver.boundary_ghost_provider
        mach_ref = self.local_solver.freestream["mach_ref"]
        n_local = self.partition.n_local_cells
        n_sps = self.state.n_sps
        n_vars = self.state.n_vars

        # SST 湍流源项+输运（算子分裂，物理步开始时求值一次，见本方法
        # 文档）——用当前（上一步末尾的）状态，产出的 mu_t_field_compact
        # 供本步全部 RK 子阶段的粘性残差使用。
        # 逐单元局部 CFL 步长（2026-09-14 取代全局固定 dt，见
        # `_compute_distributed_local_time_step` 与
        # core/mpi/distributed_cfl.py 模块文档）。
        # `dt_mean_local` 是平均流用的（启用低马赫数预处理时按预处理
        # 波速放大），`dt_phys_local` 是按物理波速那一份——湍流标量必须
        # 用后者，与单机 `fr_solver/step.py` 里 `turb_dt = dt_physical`
        # 完全一致（k/omega 的显式更新刻意没有 point-implicit 阻尼）。
        U_local_now = self.state.get_local_U()
        dt_mean_local, dt_phys_local = self._compute_distributed_local_time_step(
            U_local_now, mu_t_local=self._prev_mu_t_local, return_physical_too=True,
        )

        mu_t_field_compact = None
        if self.turb_model is not None:
            from autoflowcfd.core.mpi.distributed_turbulence import (
                distributed_compute_turbulence_source_and_viscosity,
            )
            dt_local = dt_phys_local
            mu_t_field_compact, self._turb_ramp_step = distributed_compute_turbulence_source_and_viscosity(
                self.state.get_local_U()[..., :5], self.partition, self.halo_exchange,
                self.turb_halo_exchange, self.dist_flat_face, self.mesh, self.ops,
                self.turb_model, mu, self.wall_distance_compact, dt_local,
                turb_ramp_step=self._turb_ramp_step,
                turb_ramp_steps=self._turb_production_ramp_steps,
                turb_model_name=self.turb_model_name, ddes_model=self.ddes_model,
                iddes_h_max_compact=self.iddes_h_max_compact,
                iddes_h_wn_compact=self.iddes_h_wn_compact,
                des_length_scale_halo_exchange=self.des_length_scale_halo_exchange,
            )
        elif self.sgs_model is not None:
            # LES（2026-09-02）：WALE 纯代数模型，用当前状态现算，见
            # distributed_compute_les_viscosity 文档。
            from autoflowcfd.core.mpi.distributed_turbulence import (
                distributed_compute_les_viscosity,
            )
            mu_t_field_compact = distributed_compute_les_viscosity(
                self.state.get_local_U()[..., :5], self.partition, self.halo_exchange,
                self.dist_flat_face, self.mesh, self.ops, self.sgs_model,
            )

        def residual_func_raw(U_flat_trial: np.ndarray) -> np.ndarray:
            """未经预处理的物理残差（TimeIntegrator 约定 dU/dt = -R）。RK3 每个
            stage 都会调用一次：对该 stage 的中间解重新做 halo 交换 +
            残差求值（halo 数据在每个 stage 之间会变化，不能复用上一个
            stage 交换到的邻居数据）。"""
            U_stage_local = U_flat_trial.reshape(n_local, n_sps, n_vars)
            if n_sps == 1:
                # P0（Order Continuation 最低阶）：单机无粘残差在这个
                # 阶数完全绕开 flat-face 压缩抽象（见 core/fr_residual/
                # inviscid.py::compute_inviscid_residual_fr 的
                # `mesh.n_points_1d==1` 分支文档），P1+ 路径共用的
                # compact/halo 机制在这里不适用，见
                # distributed_order_continuation.py::compute_
                # distributed_p0_inviscid_residual 文档。
                from autoflowcfd.core.mpi.distributed_order_continuation import (
                    compute_distributed_p0_inviscid_residual,
                )
                inviscid_residual = compute_distributed_p0_inviscid_residual(
                    self, U_stage_local,
                )
            else:
                inviscid_residual = distributed_compute_inviscid_residual(
                    U_stage_local, self.partition, self.halo_exchange,
                    self.dist_flat_face, self.mesh, self.ops,
                    boundary_ghost_provider, mach_ref=mach_ref,
                )
            if enable_viscous:
                viscous_residual = distributed_compute_viscous_residual(
                    U_stage_local, self.partition, self.halo_exchange,
                    self.dist_flat_face, self.mesh, self.ops,
                    mu, boundary_ghost_provider,
                    mu_t_field_compact=mu_t_field_compact,
                    wmles_model=self.wmles_model,
                    wall_distance_compact=self.wall_distance_compact,
                )
                total_dudt = inviscid_residual + viscous_residual
            else:
                total_dudt = inviscid_residual
            return -total_dudt

        def residual_func(U_flat_trial: np.ndarray) -> np.ndarray:
            """供时间积分器推进用：启用低马赫数预处理时返回 `Gamma R`。

            Gamma 线性、逐点，作用在 R 上与作用在 dU/dtau 上等价；它
            **必须**与上面按预处理波速取的 `dt_mean_local` 成对出现
            （完整推导见 core/utils/preconditioning.py 模块末尾；只改
            步长不改方程就是 2026-08-24 那次失稳）。
            Gamma 需要的原始变量由 `apply_low_mach_preconditioner` 从
            试探态算出的 Q 提供——这里显式转换，不依赖任何调用顺序上的
            隐式副作用（与单机 CPU 路径复用 `state.Q` 的做法不同，
            理由同 GPU 侧，见 core/gpu/gpu_preconditioning.py 文档）。
            """
            res = residual_func_raw(U_flat_trial)
            if self.low_mach_precond_enabled:
                from autoflowcfd.core.utils.preconditioning import (
                    apply_low_mach_preconditioner,
                )
                U_trial = U_flat_trial.reshape(n_local, n_sps, n_vars)
                Q_trial = conserved_to_primitive(U_trial[..., :5])
                res = apply_low_mach_preconditioner(
                    res, Q_trial, mach_ref, out=res)
            return res.reshape(n_local * n_sps, n_vars)

        U_flat = self.state.get_local_U().reshape(n_local * n_sps, n_vars)
        dt_local_flat = dt_mean_local.reshape(n_local * n_sps)

        # 模态滤波（2026-09-14 补齐）：单机 `fr_solver/step.py` 每个 RK
        # stage 后都施加（`build_filter_func`），多 GPU 分布式也有
        # `filter_func_gpu`——CPU 分布式此前是唯一没有施加的路径。它是
        # P>=1 的稳定性机制，不是可选项（坍缩坐标/配置点法对高阶模态
        # 混叠天然敏感，真实复现记录见 fr_solver/filter.py）。
        # local 排列里棱柱/四面体交错，所以用按单元类型掩码分派的变体；
        # 单元类型取自 `dist_fc.compact_cell_type`（0=棱柱/1=四面体，
        # 紧凑排列），换回原生排列后切 local 段。
        filter_func = None
        if n_sps > 1:
            from autoflowcfd.core.fr_solver.filter import (
                build_filter_func_by_cell_type, resolve_filter_mode,
            )
            mode = resolve_filter_mode("cpu-mpi")
            cct = self.dist_flat_face.compact_cell_type
            cell_is_prism = (cct[self.dist_flat_face.inv_perm][:n_local] == 0)
            if mode == "sensor":
                filter_func = self._build_sensor_gated_filter_func_distributed(
                    n_local, n_sps, cell_is_prism)
            else:
                filter_func = build_filter_func_by_cell_type(
                    self.ops, n_local, n_sps, cell_is_prism)

        # Stage 0 残差单独算一次：既用于收敛监控（与旧实现报告口径一致），
        # 也通过 residual0= 传给 _ssp_rk_stage_step 复用，避免它内部再重复
        # 算一次同样的 R(U^n)。
        # `residual0_raw` 是**物理**残差：收敛监控（state.dU_dt ->
        # compute_global_residual_norm）与自适应 CFL 都必须用它，不能用
        # 预处理值（Gamma 可逆、两者同时趋零，但量级不同，用预处理值会
        # 让打印的残差与历史算例失去可比性）。与单机 step.py 同一分工。
        residual0_raw = residual_func_raw(U_flat)
        self.state.dU_dt[:n_local] = -residual0_raw
        if self.low_mach_precond_enabled:
            from autoflowcfd.core.utils.preconditioning import (
                apply_low_mach_preconditioner,
            )
            residual0 = apply_low_mach_preconditioner(
                residual0_raw, self.state.Q[:n_local], mach_ref,
                out=residual0_raw,
            ).reshape(n_local * n_sps, n_vars)
        else:
            residual0 = residual0_raw.reshape(n_local * n_sps, n_vars)

        # 真实 bug 修复（2026-09-02，见 __init__ 里 self._time_integrator
        # 构造处同一处说明）：此前这里无条件调用 `_ssp_rk_stage_step`，
        # DUAL_TIME（真正时间精度的瞬态仿真）请求了也无路可走——现在
        # 与单机 `fr_solver/step.py::step` 同一个分派方式：`residual_
        # func` 本身就是 `spatial_residual(U) -> R(U)`（`dU/dt=-R(U)`
        # 约定，与 `step_dual_time` 需要的语义完全一致，不需要额外
        # 包装），直接复用。
        if self._time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME:
            U_new_flat = self._time_integrator.step_dual_time(
                U_flat, residual_func, dt_local_flat, dt_physical=dt,
                solution_prev=self._dual_time_U_prev,
                max_inner_iter=self._time_integrator.dual_time_steps,
                filter_func=filter_func,
            )
            self._dual_time_U_prev = U_flat.copy()
        else:
            U_new_flat = self._time_integrator._ssp_rk_stage_step(
                U_flat, residual_func, dt_local_flat, p_floor=1.0, residual0=residual0,
                filter_func=filter_func,
            )

        U_new_local = U_new_flat.reshape(n_local, n_sps, n_vars)
        self.state.U[:n_local] = U_new_local
        self.state.Q[:n_local] = conserved_to_primitive(U_new_local[..., :5])

        # 下一步的粘性 CFL 限制要用本步算出的涡粘（与单机读取湍流模型
        # 已存字段是同一时序）。转成 local 排列缓存：mu_t_field_compact
        # 在"棱柱在前"紧凑空间，必须先 inv_perm 换回原生排列再切 local
        # （紧凑空间的前 n_local 段**不是** local 单元，见
        # distributed_cfl.py 模块文档）。
        if mu_t_field_compact is not None:
            self._prev_mu_t_local = (
                mu_t_field_compact[self.dist_flat_face.inv_perm][:n_local])

        residual_norm = self.compute_global_residual_norm()
        # 自适应 CFL 按**全局**残差范数更新：所有 rank 喂同一个值，因此
        # 得到同一个 CFL 数（按各自局部残差更新会让 rank 间 CFL 漂移）。
        if self._cfl_controller is not None:
            self._cfl_controller.update(residual_norm)
        return residual_norm

    def solve(self, n_steps: int, dt: float, output_interval: int = 100, checkpoint_callback=None,
              tol: float = 1e-6, phase_max_iter: Optional[int] = None,
              residual_drop_threshold: float = 1e2):
        """运行分布式求解循环。

        真实 bug 修复（2026-09-02，用户明确要求"不允许出现完成度不是
        100%的功能点"后排查发现）：此前 `output_interval` 只控制
        `logger.info` 进度打印的频率，从未触发任何中间 checkpoint
        保存——分布式路径此前只在 CLI 里 `solve()` 返回*之后*保存一次
        最终 checkpoint（见 `solve_steady_command.py`），跑到一半被
        杀掉/崩溃会丢失全部进度，且没有任何"从分布式 checkpoint 继续
        跑"的机制（`solve resume` 命令对 `--n-ranks`/`--multi-gpu`
        完全没有感知）。与单机路径 `FRSolver.solve(...,
        checkpoint_callback=...)` 同一个设计补上回调机制：调用方
        （CLI）传入的 `checkpoint_callback(solver, iteration)` 在每步
        结束后被调用，由回调自己决定何时/如何保存（通常内部判断
        `iteration % checkpoint_interval`），不在这里耦合具体的保存
        格式——与单机路径的分工完全一致。

        Order Continuation 自动分派（2026-09-02，见 core/mpi/
        distributed_order_continuation.py 模块文档）：与单机
        `FRSolver.solve()`（`self.order_continuation_enabled and
        self.order >= 2` 时自动改用逐阶爬坡）同一个判据——`self.order`
        （目标阶数）>= 2 时自动委托给 `run_distributed_order_
        continuation`，不需要 CLI/调用方显式请求。P0/P1 直接求解
        （真实数值复核见 order_continuation.py 文档"曾经在这里跳过
        P=1"一节，两条阶数下均匀自由流场残差都很好，不需要爬坡）。

        Args:
            n_steps: 最大时间步数
            dt: 时间步长
            output_interval: 输出间隔
            checkpoint_callback: 可选，`callback(solver, iteration)`，
                每步结束后调用一次（与单机 `FRSolver.solve` 同名参数
                同一个约定）
            tol, phase_max_iter, residual_drop_threshold: 仅在触发
                Order Continuation（`self.order >= 2`）时生效，与单机
                `run_order_continuation` 同名参数同一含义。
        """
        from autoflowcfd.core.mpi.comm import barrier

        if getattr(self, 'order_continuation_enabled', True) and self.order >= 2:
            from autoflowcfd.core.mpi.distributed_order_continuation import (
                run_distributed_order_continuation,
            )
            return run_distributed_order_continuation(
                self, n_steps, dt, tol,
                checkpoint_callback=checkpoint_callback,
                phase_max_iter=phase_max_iter,
                residual_drop_threshold=residual_drop_threshold,
            )

        if is_root():
            logger.info(f"Starting distributed solve: {n_steps} steps, dt={dt}")

        for step_idx in range(n_steps):
            # 执行一步
            residual_norm = self.step(dt)

            # 输出进度
            if step_idx % output_interval == 0 and is_root():
                logger.info(
                    f"Step {step_idx}/{n_steps}, "
                    f"residual_norm={residual_norm:.6e}"
                )

            if checkpoint_callback is not None:
                # 全部 rank 都要调用（checkpoint_callback 内部的
                # distributed_save_checkpoint 本身就是集体操作——需要
                # 每个 rank 各自贡献 local cells 数据才能在 root 组装
                # 出正确的全局状态，只在 root 调用会在非 root rank 的
                # gather 那一侧永久阻塞）。
                checkpoint_callback(self, step_idx + 1)

            # 同步（可选，用于调试）
            # barrier()

        if is_root():
            logger.info("Distributed solve completed.")
