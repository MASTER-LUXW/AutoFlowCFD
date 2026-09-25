"""AutoFlowCFD V2.0 - 从完全分布式 package 构造，以及本 rank 的局部求解器视图

从 `src/autoflowcfd/core/mpi/distributed_solver.py` 的 `DistributedFRSolver` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `DistributedFRSolver` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import os
from typing import Optional
from autoflowcfd.core.mpi import get_rank
from autoflowcfd.core.mpi.halo import HaloExchange
from autoflowcfd.core.mpi.distributed_state import DistributedFRState
from autoflowcfd.core.mpi.comm import barrier
from autoflowcfd.core.time_integration.base import (
    TimeIntegrator, TimeIntegrationScheme, require_distributed_scheme,
)


class _DistributedFromPackageMixin:
    """从完全分布式 package 构造，以及本 rank 的局部求解器视图"""

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
        time_scheme = require_distributed_scheme(
            package.get('time_scheme', TimeIntegrationScheme.SSP_RK3))
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
        # CFL 策略（控制器 or 固定 CFL）的唯一事实来源：
        # `time_integration/adaptive_cfl/policy.py::build_cfl_policy`（六个后端
        # 构造点此前各写一份且已分叉，见该模块文档）。
        # 此前这里只对 SSP-RK2/RK3 建控制器，其余方案（IMEX、DUAL_TIME 的伪
        # 时间步）走 cfl.py 的替身回退 0.1，与单机的 cfl_start 不一致。
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import build_cfl_policy
        self._cfl_controller, self.fixed_cfl_number = build_cfl_policy(
            time_scheme, cfl_start=package.get('cfl_start'),
            cfl_max=package.get('cfl_max'), cfl_min=package.get('cfl_min'))
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
